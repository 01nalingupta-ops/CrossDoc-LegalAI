"""Standalone stub integration layer for CrossDoc-LegalAI Part 7.

These functions intentionally match the contracts of the real ingestion, retrieval, and
auditor+guardrail modules. Replace this module with real implementations by changing one
import line in app.py; the UI consumes only these signatures and schemas.
"""

from __future__ import annotations

import hashlib
import math
import re
from pathlib import Path
from typing import Any, BinaryIO

CHUNK_SIZE = 800
OVERLAP = 150
SEVERITY_RANK = {"High": 3, "Medium": 2, "Low": 1, "None": 0}


def parse_document(file: str | Path | bytes | BinaryIO, doc_type: str) -> dict[str, Any]:
    """STUB for real ingestion: parse uploaded PDF-ish bytes into ParsedDocument."""
    if doc_type not in {"master", "service"}:
        raise ValueError("doc_type must be 'master' or 'service'")
    data, source_path = _read_filelike(file)
    text = _extract_pdf_literal_text(data) or data.decode("utf-8", errors="ignore")
    text = re.sub(r"\s+", " ", text).strip() or "No extractable text found in uploaded document."
    doc_hash = hashlib.sha256(data + doc_type.encode("utf-8")).hexdigest()[:10]
    doc_id = f"{doc_type}-{doc_hash}"
    return {
        "doc_id": doc_id,
        "doc_type": doc_type,
        "source_path": source_path,
        "full_text": text,
        "page_count": max(1, data.count(b"/Type /Page")),
        "extraction_method": "digital",
        "chunks": _chunk_text(doc_id, text),
    }


def retrieve_matches(master_doc: dict[str, Any], service_doc: dict[str, Any]) -> list[dict[str, Any]]:
    """STUB for real retrieval: return top-2 lexical-overlap matches per service chunk."""
    results = []
    for service_chunk in service_doc["chunks"]:
        scored = []
        for master_chunk in master_doc["chunks"]:
            score = _cosine_token_similarity(service_chunk["text"], master_chunk["text"])
            scored.append(
                {
                    "master_chunk_id": master_chunk["chunk_id"],
                    "master_chunk_text": master_chunk["text"],
                    "similarity_score": round(score, 4),
                }
            )
        results.append(
            {
                "service_chunk_id": service_chunk["chunk_id"],
                "service_chunk_text": service_chunk["text"],
                "matched_master_chunks": sorted(scored, key=lambda row: row["similarity_score"], reverse=True)[:2],
            }
        )
    return results


def run_auditor_and_guardrail(
    retrieval_results: list[dict[str, Any]],
    master_full_text: str,
    service_full_text: str,
    pair_id: str = "demo-pair",
    model_id: str = "stub-rule-auditor-v1",
) -> list[dict[str, Any]]:
    """STUB for real auditor+guardrail: deterministic keyword mismatch detector."""
    predictions = []
    for result in retrieval_results:
        service_text = result["service_chunk_text"]
        master_text = " ".join(chunk["master_chunk_text"] for chunk in result.get("matched_master_chunks", []))
        raw = _rule_based_prediction(service_text, master_text)
        predictions.append(_apply_stub_guardrail(raw, pair_id, model_id, result["service_chunk_id"], master_full_text, service_full_text))
    return predictions


def sort_predictions_for_display(predictions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(predictions, key=lambda row: (SEVERITY_RANK.get(row.get("severity", "None"), 0), row.get("confidence", 0)), reverse=True)


def infer_category(prediction: dict[str, Any]) -> str:
    text = prediction.get("conflict_explanation", "").lower()
    if "payment" in text or "net 30" in text or "net 60" in text:
        return "Payment Terms"
    if "liability" in text or "uncapped" in text or "cap" in text:
        return "Liability Cap"
    if "unverified" in text or "withheld" in text:
        return "Unverified Claim"
    return "Contract Mismatch"


def _rule_based_prediction(service_text: str, master_text: str) -> dict[str, Any]:
    service_lower = service_text.lower()
    master_lower = master_text.lower()
    if "net 60" in service_lower and "net 30" in master_lower:
        return _prediction(True, 0.88, "Medium", "Payment Terms mismatch: the service document uses Net 60 while the master requires Net 30.", _quote_containing(master_text, "Net 30"), _quote_containing(service_text, "Net 60"), "Revise the service payment clause to require payment within Net 30 days.")
    if "uncapped" in service_lower and ("capped" in master_lower or "cap" in master_lower):
        return _prediction(True, 0.91, "High", "Liability Cap mismatch: the service document makes liability uncapped despite the master cap.", _quote_containing(master_text, "cap"), _quote_containing(service_text, "uncapped"), "Replace the uncapped liability language with the master agreement liability cap.")
    if "unverified" in service_lower or "hallucination" in service_lower:
        return _prediction(True, 0.76, "Medium", "Placeholder auditor deliberately emits an unverified claim to exercise claim_withheld UI behavior.", "This master quote does not exist in the source.", _quote_containing(service_text, "Unverified"), "Withheld until source evidence can be verified.")
    return _prediction(False, 0.72, "None", "No mismatch found by the offline stub auditor.", "", "", "")


def _apply_stub_guardrail(raw: dict[str, Any], pair_id: str, model_id: str, service_chunk_id: str, master_full_text: str, service_full_text: str) -> dict[str, Any]:
    msa_ok = _quote_ok(raw["msa_exact_quote"], master_full_text)
    sow_ok = _quote_ok(raw["sow_exact_quote"], service_full_text)
    output = {"pair_id": pair_id, "model_id": model_id, "service_chunk_id": service_chunk_id, **raw}
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


def _read_filelike(file: str | Path | bytes | BinaryIO) -> tuple[bytes, str]:
    if isinstance(file, bytes):
        return file, "uploaded-bytes.pdf"
    if isinstance(file, (str, Path)):
        path = Path(file)
        return path.read_bytes(), str(path)
    name = getattr(file, "name", "uploaded.pdf")
    data = file.getvalue() if hasattr(file, "getvalue") else file.read()
    return data, name


def _extract_pdf_literal_text(data: bytes) -> str:
    raw = data.decode("latin-1", errors="ignore")
    snippets = re.findall(r"\(([^()]*)\)\s*Tj", raw)
    array_snippets = re.findall(r"\[(.*?)\]\s*TJ", raw, flags=re.DOTALL)
    for array in array_snippets:
        snippets.extend(re.findall(r"\(([^()]*)\)", array))
    return " ".join(_unescape_pdf_text(s) for s in snippets)


def _unescape_pdf_text(text: str) -> str:
    return text.replace(r"\(", "(").replace(r"\)", ")").replace(r"\\", "\\")


def _chunk_text(doc_id: str, text: str) -> list[dict[str, Any]]:
    chunks = []
    start = 0
    index = 1
    while start < len(text):
        end = min(len(text), start + CHUNK_SIZE)
        chunks.append({"chunk_id": f"{doc_id}-chunk-{index:04d}", "text": text[start:end], "char_start": start, "char_end": end, "page_num": 1, "overlap_chars": OVERLAP})
        if end == len(text):
            break
        start = max(0, end - OVERLAP)
        index += 1
    return chunks or [{"chunk_id": f"{doc_id}-chunk-0001", "text": "", "char_start": 0, "char_end": 0, "page_num": 1, "overlap_chars": OVERLAP}]


def _cosine_token_similarity(a: str, b: str) -> float:
    counts_a = _token_counts(a)
    counts_b = _token_counts(b)
    shared = set(counts_a) & set(counts_b)
    numerator = sum(counts_a[token] * counts_b[token] for token in shared)
    denom_a = math.sqrt(sum(value * value for value in counts_a.values()))
    denom_b = math.sqrt(sum(value * value for value in counts_b.values()))
    return 0.0 if denom_a == 0 or denom_b == 0 else numerator / (denom_a * denom_b)


def _token_counts(text: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for token in re.findall(r"[a-zA-Z0-9]+", text.lower()):
        counts[token] = counts.get(token, 0) + 1
    return counts


def _quote_ok(quote: str, full_text: str) -> bool:
    if not quote:
        return True
    return re.sub(r"\s+", " ", quote).strip() in re.sub(r"\s+", " ", full_text).strip()


def _quote_containing(text: str, needle: str) -> str:
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    for sentence in sentences:
        if needle.lower() in sentence.lower():
            return sentence.strip()
    return _first_sentence(text)


def _first_sentence(text: str) -> str:
    return re.split(r"(?<=[.!?])\s+", text.strip())[0][:400]
