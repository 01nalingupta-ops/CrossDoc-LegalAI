"""Model adapters for CrossDoc-LegalAI Part 4 auditor bake-off.

The orchestration layer depends only on AuditorModelAdapter.judge(...). Add new
providers by subclassing AuditorModelAdapter and returning the normalized raw
Auditor output schema documented in README.md.
"""

from __future__ import annotations

import json
import os
import urllib.request
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Callable

import numeric_reasoning


AUDITOR_PROMPT_TEMPLATE = """You are CrossDoc-LegalAI Auditor Agent (LangGraph Node 2).

Task: Decide whether the Service/SOW clause contradicts either matched Master/MSA clause.
Consider these contradiction categories: Payment Terms, Liability Cap, IP Ownership,
Termination Notice, Governing Jurisdiction, Confidentiality Scope, Indemnification,
Insurance Requirements.

Rules:
- Return ONLY one JSON object, no markdown.
- Use exactly these keys: has_contradiction, confidence, severity, conflict_explanation,
  msa_exact_quote, sow_exact_quote, suggested_redline.
- severity must be one of: High, Medium, Low, None.
- If no contradiction, set has_contradiction=false, severity="None", quotes and redline empty.
- If contradiction, msa_exact_quote must be a verbatim substring from one matched Master
  chunk and sow_exact_quote must be a verbatim substring from the Service chunk.
- Do not quote text that is not present in the provided chunks.

Service chunk:
{service_chunk_text}

Matched Master chunks:
{master_chunks_json}

Additional numeric evidence (advisory only; do not quote from this section):
{numeric_evidence_json}
"""


class AuditorModelAdapter(ABC):
    """Single interface implemented by every bake-off auditor model adapter."""

    model_id: str

    @abstractmethod
    def judge(self, service_chunk_text: str, master_chunks: list[dict[str, Any]]) -> dict[str, Any]:
        """Return the raw structured Auditor prediction for one Service chunk."""


class LocalNLIAdapter(AuditorModelAdapter):
    """Local HuggingFace NLI adapter for the candidate_2 bake-off slot.

    The model is loaded lazily on the first ``judge`` call so importing this module and
    constructing the adapter do not download model weights or require accelerator setup.
    """

    def __init__(
        self,
        config_path: str | Path | None = None,
        nli_pipeline: Callable[[dict[str, str]], Any] | None = None,
    ) -> None:
        self.config_path = Path(config_path or os.environ.get("CROSSDOC_AUDITOR_CONFIG", "auditor_config.json"))
        self.config = _load_local_nli_config(self.config_path)
        self.model_name = str(self.config["model_name"])
        self.model_revision = str(self.config.get("model_revision", "main"))
        self.model_id = f"local-nli:{self.model_name}@{self.model_revision}"
        self._pipeline = nli_pipeline

    def judge(self, service_chunk_text: str, master_chunks: list[dict[str, Any]]) -> dict[str, Any]:
        numeric_evidence = numeric_evidence_for_clause(service_chunk_text, master_chunks)
        if not master_chunks:
            return _with_numeric_evidence(
                _prediction(False, 0.0, "None", "No matched master clause was available for local NLI comparison.", "", "", ""),
                numeric_evidence,
            )

        scored = [self._score_chunk(service_chunk_text, chunk) for chunk in master_chunks]
        best = max(scored, key=lambda row: (row["contradiction_probability"], row["master_chunk_id"]))
        confidence = float(best["contradiction_probability"])
        has_contradiction = confidence > float(self.config["contradiction_threshold"])
        if not has_contradiction:
            return _with_numeric_evidence(
                _prediction(
                    False,
                    confidence,
                    "None",
                    f"Local NLI classified the strongest matched clause as non-contradictory with contradiction confidence {confidence:.3f}.",
                    "",
                    "",
                    "",
                ),
                numeric_evidence,
            )

        severity = _severity_for_confidence(confidence, self.config["severity_bands"])
        master_text = best["master_chunk_text"]
        return _with_numeric_evidence(
            _prediction(
                True,
                confidence,
                severity,
                f"Local NLI classified the matched master/service clauses as contradiction with confidence {confidence:.3f}.",
                master_text,
                service_chunk_text,
                f"Revise the service clause to align with the matched master clause: {master_text}",
            ),
            numeric_evidence,
        )

    def _score_chunk(self, service_chunk_text: str, master_chunk: dict[str, Any]) -> dict[str, Any]:
        master_text = str(master_chunk.get("master_chunk_text", ""))
        scores = _coerce_nli_scores(self._nli_pipeline()({"text": master_text, "text_pair": service_chunk_text}))
        contradiction_probability = _score_for_label(scores, self.config["label_aliases"]["contradiction"])
        return {
            "master_chunk_id": str(master_chunk.get("master_chunk_id", "")),
            "master_chunk_text": master_text,
            "contradiction_probability": contradiction_probability,
        }

    def _nli_pipeline(self) -> Callable[[dict[str, str]], Any]:
        if self._pipeline is None:
            try:
                from transformers import pipeline
            except ImportError as exc:
                raise RuntimeError(
                    "LocalNLIAdapter requires transformers and torch. Install project dependencies from requirements.txt."
                ) from exc
            self._pipeline = pipeline(
                "text-classification",
                model=self.model_name,
                revision=self.model_revision,
                top_k=None,
                truncation=True,
            )
        return self._pipeline


class OpenAICompatibleAdapter(AuditorModelAdapter):
    """Generic OpenAI-compatible chat-completions adapter.

    Configure with constructor arguments or environment variables:
    OPENAI_COMPATIBLE_API_KEY, OPENAI_COMPATIBLE_BASE_URL, OPENAI_COMPATIBLE_MODEL.
    Every call uses temperature=0.0 as required by the bake-off.
    """

    def __init__(
        self,
        model: str | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
        model_id: str | None = None,
    ) -> None:
        self.model = model or os.environ.get("OPENAI_COMPATIBLE_MODEL", "gpt-4o-mini")
        self.api_key = api_key or os.environ.get("OPENAI_COMPATIBLE_API_KEY", "")
        self.base_url = (base_url or os.environ.get("OPENAI_COMPATIBLE_BASE_URL", "https://api.openai.com/v1")).rstrip("/")
        self.model_id = model_id or self.model

    def judge(self, service_chunk_text: str, master_chunks: list[dict[str, Any]]) -> dict[str, Any]:
        prompt = AUDITOR_PROMPT_TEMPLATE.format(
            service_chunk_text=service_chunk_text,
            master_chunks_json=json.dumps(master_chunks, indent=2, sort_keys=True),
            numeric_evidence_json=json.dumps(
                numeric_evidence_for_clause(service_chunk_text, master_chunks), indent=2, sort_keys=True
            ),
        )
        payload = {
            "model": self.model,
            "temperature": 0.0,
            "response_format": {"type": "json_object"},
            "messages": [{"role": "user", "content": prompt}],
        }
        request = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=60) as response:
            body = json.loads(response.read().decode("utf-8"))
        return json.loads(body["choices"][0]["message"]["content"])


class MockAdapter(AuditorModelAdapter):
    """Deterministic no-network adapter for tests and local CLI runs."""

    model_id = "mock-auditor-v1"

    def judge(self, service_chunk_text: str, master_chunks: list[dict[str, Any]]) -> dict[str, Any]:
        lower = service_chunk_text.lower()
        master_text = "\n".join(chunk.get("master_chunk_text", "") for chunk in master_chunks)
        if "hallucinate" in lower:
            return _prediction(True, 0.93, "High", "The model reports a payment conflict but fabricates the MSA quote.", "Payment shall be made within 999 days.", service_chunk_text, "Revise to match the MSA payment period.")
        if "net 60" in lower and "net 30" in master_text.lower():
            return _prediction(True, 0.91, "Medium", "The SOW extends payment from Net 30 to Net 60.", "Customer shall pay all undisputed invoices within Net 30 days.", "Customer shall pay all undisputed invoices within Net 60 days.", "Customer shall pay all undisputed invoices within Net 30 days.")
        if "liability" in lower and "uncapped" in lower:
            return _prediction(True, 0.89, "High", "The SOW makes liability uncapped despite the MSA cap.", "Supplier's aggregate liability is capped at fees paid in the prior twelve months.", "Supplier's liability is uncapped for all claims under this SOW.", "Supplier's aggregate liability is capped at fees paid in the prior twelve months.")
        return _prediction(False, 0.86, "None", "No contradiction was identified in the matched clauses.", "", "", "")



def numeric_evidence_for_clause(service_chunk_text: str, master_chunks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return advisory numeric evidence for an adapter clause comparison."""
    retrieval_result = {"service_chunk_text": service_chunk_text, "matched_master_chunks": master_chunks}
    return numeric_reasoning.evidence_for_retrieval_result(retrieval_result)


def _with_numeric_evidence(prediction: dict[str, Any], evidence: list[dict[str, Any]]) -> dict[str, Any]:
    return {**prediction, "numeric_evidence": evidence}


def _load_local_nli_config(config_path: Path) -> dict[str, Any]:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    local_nli = config.get("local_nli")
    if not isinstance(local_nli, dict):
        raise ValueError("auditor_config.json must contain a local_nli object")
    required = {"model_name", "contradiction_threshold", "severity_bands", "label_aliases"}
    missing = sorted(required - set(local_nli))
    if missing:
        raise ValueError(f"local_nli config missing required keys: {', '.join(missing)}")
    return local_nli


def _coerce_nli_scores(raw_scores: Any) -> list[dict[str, float | str]]:
    if isinstance(raw_scores, list) and raw_scores and isinstance(raw_scores[0], list):
        raw_scores = raw_scores[0]
    if not isinstance(raw_scores, list):
        raise ValueError("NLI pipeline must return a list of label/score dictionaries")
    scores = []
    for row in raw_scores:
        if not isinstance(row, dict) or "label" not in row or "score" not in row:
            raise ValueError("NLI pipeline scores must include label and score")
        scores.append({"label": str(row["label"]).lower(), "score": float(row["score"])})
    return scores


def _score_for_label(scores: list[dict[str, float | str]], aliases: list[str]) -> float:
    normalized_aliases = {alias.lower() for alias in aliases}
    for row in scores:
        if str(row["label"]).lower() in normalized_aliases:
            return float(row["score"])
    raise ValueError(f"NLI scores did not include any contradiction label alias: {sorted(normalized_aliases)}")


def _severity_for_confidence(confidence: float, severity_bands: list[dict[str, Any]]) -> str:
    for band in sorted(severity_bands, key=lambda row: float(row["min_confidence"]), reverse=True):
        if confidence >= float(band["min_confidence"]):
            return str(band["severity"])
    return "None"


def _prediction(has: bool, confidence: float, severity: str, explanation: str, msa: str, sow: str, redline: str) -> dict[str, Any]:
    return {
        "has_contradiction": has,
        "confidence": confidence,
        "severity": severity,
        "conflict_explanation": explanation,
        "msa_exact_quote": msa,
        "sow_exact_quote": sow,
        "suggested_redline": redline,
    }
