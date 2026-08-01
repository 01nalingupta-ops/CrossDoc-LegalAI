import json
from pathlib import Path

from pdf_utils import build_report_pdf
from pipeline_stubs import parse_document, retrieve_matches, run_auditor_and_guardrail, sort_predictions_for_display

ROOT = Path(__file__).resolve().parents[1]
MASTER = ROOT / "sample_docs" / "sample_master_msa.pdf"
SERVICE = ROOT / "sample_docs" / "sample_service_sow.pdf"


def test_stub_pipeline_sample_pdfs_render_report_data():
    master_doc = parse_document(MASTER, "master")
    service_doc = parse_document(SERVICE, "service")
    retrieval = retrieve_matches(master_doc, service_doc)
    predictions = run_auditor_and_guardrail(retrieval, master_doc["full_text"], service_doc["full_text"], pair_id="smoke")

    assert master_doc["doc_type"] == "master"
    assert service_doc["doc_type"] == "service"
    assert retrieval
    assert all("matched_master_chunks" in row for row in retrieval)
    assert any(row["has_contradiction"] and row["guardrail_action"] == "passed" for row in predictions)
    assert any(row["guardrail_action"] == "claim_withheld" for row in predictions)
    assert sort_predictions_for_display(predictions)[0]["severity"] in {"High", "Medium"}

    report_json = json.dumps({"predictions": predictions})
    report_pdf = build_report_pdf(predictions)
    assert "claim_withheld" in report_json
    assert report_pdf.startswith(b"%PDF")


def test_contract_shapes_have_required_keys():
    master_doc = parse_document(MASTER, "master")
    service_doc = parse_document(SERVICE, "service")
    retrieval = retrieve_matches(master_doc, service_doc)
    predictions = run_auditor_and_guardrail(retrieval, master_doc["full_text"], service_doc["full_text"])

    parsed_keys = {"doc_id", "doc_type", "source_path", "full_text", "page_count", "extraction_method", "chunks"}
    retrieval_keys = {"service_chunk_id", "service_chunk_text", "matched_master_chunks"}
    prediction_keys = {"pair_id", "model_id", "service_chunk_id", "has_contradiction", "confidence", "severity", "conflict_explanation", "msa_exact_quote", "sow_exact_quote", "suggested_redline", "guardrail_verified", "guardrail_action"}
    assert parsed_keys <= master_doc.keys()
    assert retrieval_keys <= retrieval[0].keys()
    assert prediction_keys <= predictions[0].keys()
