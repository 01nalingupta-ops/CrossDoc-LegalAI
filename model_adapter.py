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
from typing import Any


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
"""


class AuditorModelAdapter(ABC):
    """Single interface implemented by every bake-off auditor model adapter."""

    model_id: str

    @abstractmethod
    def judge(self, service_chunk_text: str, master_chunks: list[dict[str, Any]]) -> dict[str, Any]:
        """Return the raw structured Auditor prediction for one Service chunk."""


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
