from auditor_guardrail import apply_guardrail, quote_in_document, run_auditor_and_guardrail
from model_adapter import MockAdapter


def _raw(msa="Master clause text.", sow="Service clause text."):
    return {
        "has_contradiction": True,
        "confidence": 0.8,
        "severity": "Medium",
        "conflict_explanation": "Conflict.",
        "msa_exact_quote": msa,
        "sow_exact_quote": sow,
        "suggested_redline": "Use master language.",
    }


def test_guardrail_passes_when_both_quotes_are_substrings():
    result = apply_guardrail(_raw(), "Intro. Master clause text. Outro.", "Intro. Service clause text. Outro.")
    assert result["guardrail_verified"] is True
    assert result["guardrail_action"] == "passed"
    assert result["has_contradiction"] is True
    assert result["severity"] == "Medium"


def test_guardrail_rejects_and_withholds_when_quote_is_fabricated():
    result = apply_guardrail(_raw(msa="Fabricated master text."), "Intro. Master clause text. Outro.", "Intro. Service clause text. Outro.")
    assert result["guardrail_verified"] is False
    assert result["guardrail_action"] == "claim_withheld"
    assert result["has_contradiction"] is False
    assert result["confidence"] == 0.0
    assert result["severity"] == "None"
    assert result["suggested_redline"] == ""


def test_guardrail_passes_whitespace_only_differences():
    master = "Payment:\nCustomer shall pay all undisputed invoices within\tNet 30 days."
    service = "Service says Customer shall pay all undisputed invoices within Net 60 days."
    result = apply_guardrail(
        _raw(
            msa="Customer shall pay all undisputed invoices within Net 30 days.",
            sow="Customer shall pay all undisputed invoices within\nNet 60 days.",
        ),
        master,
        service,
    )
    assert result["guardrail_verified"] is True
    assert quote_in_document(result["msa_exact_quote"], master)
    assert quote_in_document(result["sow_exact_quote"], service)


def test_pipeline_uses_mock_adapter_and_passes_context_fields():
    retrieval = {
        "service_chunk_id": "sow-1",
        "service_chunk_text": "Customer shall pay all undisputed invoices within Net 60 days.",
        "matched_master_chunks": [{"master_chunk_id": "msa-1", "master_chunk_text": "Customer shall pay all undisputed invoices within Net 30 days.", "similarity_score": 0.9}],
    }
    result = run_auditor_and_guardrail(
        retrieval,
        "Customer shall pay all undisputed invoices within Net 30 days.",
        "Customer shall pay all undisputed invoices within Net 60 days.",
        MockAdapter(),
        "pair-1",
    )
    assert result["pair_id"] == "pair-1"
    assert result["model_id"] == "mock-auditor-v1"
    assert result["service_chunk_id"] == "sow-1"
    assert result["guardrail_action"] == "passed"
