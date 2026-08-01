"""Auditor + deterministic quote guardrail pipeline for CrossDoc-LegalAI Part 4."""

from __future__ import annotations

import argparse
import json
import re
from typing import Any

from model_adapter import AuditorModelAdapter, MockAdapter

_ALLOWED_SEVERITIES = {"High", "Medium", "Low", "None"}


def normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def quote_in_document(quote: str, document_text: str) -> bool:
    if quote == "":
        return True
    return normalize_whitespace(quote) in normalize_whitespace(document_text)


def apply_guardrail(raw_prediction: dict[str, Any], master_full_text: str, service_full_text: str) -> dict[str, Any]:
    """Verify exact quotes and withhold contradiction claims when evidence fails.

    Downgrade behavior: if either non-empty quote is not verifiable after whitespace
    normalization, the user-facing claim is withheld by forcing has_contradiction=False,
    confidence=0.0, severity="None", clearing suggested_redline, and setting
    guardrail_action="claim_withheld". The original explanation/quotes remain visible
    for debugging and audit trails, but should be shown as unverified by any UI.
    """
    output = _coerce_prediction(raw_prediction)
    msa_ok = quote_in_document(output["msa_exact_quote"], master_full_text)
    sow_ok = quote_in_document(output["sow_exact_quote"], service_full_text)
    if msa_ok and sow_ok:
        output["guardrail_verified"] = True
        output["guardrail_action"] = "passed"
        return output

    output["has_contradiction"] = False
    output["confidence"] = 0.0
    output["severity"] = "None"
    output["suggested_redline"] = ""
    output["guardrail_verified"] = False
    output["guardrail_action"] = "claim_withheld"
    return output


def run_auditor_and_guardrail(
    retrieval_result: dict[str, Any],
    master_full_text: str,
    service_full_text: str,
    adapter: AuditorModelAdapter,
    pair_id: str,
) -> dict[str, Any]:
    raw = adapter.judge(retrieval_result["service_chunk_text"], retrieval_result.get("matched_master_chunks", []))
    verified = apply_guardrail(raw, master_full_text, service_full_text)
    return {
        "pair_id": pair_id,
        "model_id": adapter.model_id,
        "service_chunk_id": retrieval_result["service_chunk_id"],
        **verified,
    }


def _coerce_prediction(raw: dict[str, Any]) -> dict[str, Any]:
    severity = raw.get("severity", "None")
    if severity not in _ALLOWED_SEVERITIES:
        severity = "None"
    return {
        "has_contradiction": bool(raw.get("has_contradiction", False)),
        "confidence": max(0.0, min(1.0, float(raw.get("confidence", 0.0)))),
        "severity": severity,
        "conflict_explanation": str(raw.get("conflict_explanation", "")),
        "msa_exact_quote": str(raw.get("msa_exact_quote", "")),
        "sow_exact_quote": str(raw.get("sow_exact_quote", "")),
        "suggested_redline": str(raw.get("suggested_redline", "")),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Part 4 Auditor + Guardrail over RetrievalResult JSON.")
    parser.add_argument("input_json", help="JSON file with retrieval_results, master_full_text, service_full_text, and optional pair_id.")
    args = parser.parse_args()
    with open(args.input_json, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    adapter = MockAdapter()
    results = [
        run_auditor_and_guardrail(item, payload["master_full_text"], payload["service_full_text"], adapter, payload.get("pair_id", "pair-unknown"))
        for item in payload["retrieval_results"]
    ]
    print(json.dumps(results, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
