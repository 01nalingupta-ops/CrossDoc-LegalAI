import json

from auditor_guardrail import apply_guardrail
from model_adapter import LocalNLIAdapter, numeric_evidence_for_clause
from numeric_reasoning import compare_numeric_values, evidence_for_retrieval_result, extract_numeric_values


class FakeNLIPipeline:
    def __call__(self, payload):
        return [
            {"label": "entailment", "score": 0.1},
            {"label": "neutral", "score": 0.0},
            {"label": "contradiction", "score": 0.9},
        ]


def _config(tmp_path):
    path = tmp_path / "auditor_config.json"
    path.write_text(
        json.dumps(
            {
                "local_nli": {
                    "model_name": "unit-test-nli",
                    "model_revision": "test-revision",
                    "contradiction_threshold": 0.55,
                    "severity_bands": [{"severity": "High", "min_confidence": 0.85}],
                    "label_aliases": {
                        "contradiction": ["contradiction"],
                        "neutral": ["neutral"],
                        "entailment": ["entailment"],
                    },
                }
            }
        ),
        encoding="utf-8",
    )
    return path


def test_extracts_dates_amounts_percentages_and_durations():
    values = extract_numeric_values(
        "Fees are $1,250.50, retainage is 10%, completion is due January 15, 2027, "
        "and notice is thirty days."
    )

    assert {value["value_type"] for value in values} == {"amount", "percentage", "date", "duration"}
    assert any(value["value"] == 1250.50 and value["unit"] == "USD" for value in values)
    assert any(value["value"] == 10.0 and value["unit"] == "percent" for value in values)
    assert any(value["value"] == "2027-01-15" and value["unit"] == "date" for value in values)
    assert any(value["value"] == 30.0 and value["unit"] == "day" for value in values)


def test_compares_each_numeric_value_type_as_structured_mismatch_evidence():
    evidence = compare_numeric_values(
        "Payment is $1,000, the fee is 5%, delivery is due 01/15/2027, and notice is 30 days.",
        "Payment is $1,500, the fee is 7%, delivery is due January 20, 2027, and notice is 60 days.",
    )

    by_type = {item["value_type"]: item for item in evidence}
    assert by_type["amount"]["status"] == "mismatch"
    assert by_type["percentage"]["status"] == "mismatch"
    assert by_type["date"]["status"] == "mismatch"
    assert by_type["duration"]["status"] == "mismatch"


def test_mismatched_duration_units_are_ambiguous_not_auto_resolved():
    evidence = compare_numeric_values("The cure period is 30 days.", "The cure period is one month.")

    assert evidence == [
        {
            "value_type": "duration",
            "status": "ambiguous",
            "master_value": {
                "value_type": "duration",
                "value": 30.0,
                "unit": "day",
                "text": "30 days",
                "char_start": 19,
                "char_end": 26,
            },
            "service_value": {
                "value_type": "duration",
                "value": 1.0,
                "unit": "month",
                "text": "one month",
                "char_start": 19,
                "char_end": 28,
            },
            "explanation": "Found duration values with non-comparable units: day vs month.",
        }
    ]


def test_retrieval_numeric_evidence_carries_master_chunk_id():
    evidence = evidence_for_retrieval_result(
        {
            "service_chunk_text": "Invoices are payable within 60 days.",
            "matched_master_chunks": [
                {
                    "master_chunk_id": "m-payment",
                    "master_chunk_text": "Invoices are payable within 30 days.",
                }
            ],
        }
    )

    assert evidence[0]["master_chunk_id"] == "m-payment"
    assert evidence[0]["value_type"] == "duration"
    assert evidence[0]["status"] == "mismatch"


def test_local_nli_adapter_includes_numeric_evidence_without_replacing_judgment(tmp_path):
    adapter = LocalNLIAdapter(config_path=_config(tmp_path), nli_pipeline=FakeNLIPipeline())
    prediction = adapter.judge(
        "Invoices are payable within 60 days.",
        [{"master_chunk_id": "m1", "master_chunk_text": "Invoices are payable within 30 days."}],
    )

    assert prediction["has_contradiction"] is True
    assert prediction["numeric_evidence"][0]["status"] == "mismatch"
    assert numeric_evidence_for_clause(
        "Invoices are payable within 60 days.",
        [{"master_chunk_id": "m1", "master_chunk_text": "Invoices are payable within 30 days."}],
    ) == prediction["numeric_evidence"]


def test_numeric_evidence_does_not_bypass_guardrail_verbatim_quote_check():
    raw_prediction = {
        "has_contradiction": True,
        "confidence": 0.95,
        "severity": "High",
        "conflict_explanation": "Numeric mismatch detected.",
        "msa_exact_quote": "Payment is due within 999 days.",
        "sow_exact_quote": "Payment is due within 60 days.",
        "suggested_redline": "Use 30 days.",
        "numeric_evidence": compare_numeric_values(
            "Payment is due within 30 days.", "Payment is due within 60 days."
        ),
    }

    guarded = apply_guardrail(
        raw_prediction,
        "Payment is due within 30 days.",
        "Payment is due within 60 days.",
    )

    assert guarded["has_contradiction"] is False
    assert guarded["guardrail_action"] == "claim_withheld"
