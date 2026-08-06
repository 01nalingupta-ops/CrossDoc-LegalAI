import json
from pathlib import Path

import pytest

from auditor_guardrail import run_auditor_and_guardrail
from model_adapter import LocalNLIAdapter


def _config(tmp_path, threshold=0.55):
    path = tmp_path / "auditor_config.json"
    path.write_text(
        json.dumps(
            {
                "local_nli": {
                    "model_name": "unit-test-nli",
                    "model_revision": "test-revision",
                    "contradiction_threshold": threshold,
                    "severity_bands": [
                        {"severity": "High", "min_confidence": 0.85},
                        {"severity": "Medium", "min_confidence": 0.70},
                        {"severity": "Low", "min_confidence": 0.55},
                    ],
                    "label_aliases": {
                        "contradiction": ["contradiction", "label_2"],
                        "neutral": ["neutral", "label_1"],
                        "entailment": ["entailment", "label_0"],
                    },
                }
            }
        ),
        encoding="utf-8",
    )
    return path


class FakeNLIPipeline:
    def __init__(self, by_premise):
        self.by_premise = by_premise
        self.calls = []

    def __call__(self, payload):
        self.calls.append(payload)
        score = self.by_premise[payload["text"]]
        return [
            {"label": "entailment", "score": 1.0 - score},
            {"label": "neutral", "score": 0.0},
            {"label": "contradiction", "score": score},
        ]


def test_local_nli_adapter_selects_highest_contradiction_chunk(tmp_path):
    master_chunks = [
        {"master_chunk_id": "m-low", "master_chunk_text": "Payment is due within Net 45 days."},
        {"master_chunk_id": "m-high", "master_chunk_text": "Payment is due within Net 30 days."},
    ]
    service_text = "Payment is due within Net 60 days."
    fake_pipeline = FakeNLIPipeline(
        {
            "Payment is due within Net 45 days.": 0.62,
            "Payment is due within Net 30 days.": 0.91,
        }
    )

    adapter = LocalNLIAdapter(config_path=_config(tmp_path), nli_pipeline=fake_pipeline)
    prediction = adapter.judge(service_text, master_chunks)

    assert adapter.model_id == "local-nli:unit-test-nli@test-revision"
    assert len(fake_pipeline.calls) == 2
    assert prediction["has_contradiction"] is True
    assert prediction["confidence"] == pytest.approx(0.91)
    assert prediction["severity"] == "High"
    assert prediction["msa_exact_quote"] == "Payment is due within Net 30 days."
    assert prediction["sow_exact_quote"] == service_text
    assert "Local NLI classified" in prediction["conflict_explanation"]
    assert "Payment is due within Net 30 days." in prediction["suggested_redline"]


def test_local_nli_adapter_below_threshold_returns_no_contradiction(tmp_path):
    adapter = LocalNLIAdapter(
        config_path=_config(tmp_path, threshold=0.80),
        nli_pipeline=FakeNLIPipeline({"Master clause text.": 0.79}),
    )

    prediction = adapter.judge("Service clause text.", [{"master_chunk_id": "m1", "master_chunk_text": "Master clause text."}])

    assert prediction["has_contradiction"] is False
    assert prediction["confidence"] == pytest.approx(0.79)
    assert prediction["severity"] == "None"
    assert prediction["msa_exact_quote"] == ""
    assert prediction["sow_exact_quote"] == ""


def test_local_nli_adapter_quotes_pass_guardrail_by_construction(tmp_path):
    master_text = "Supplier must maintain insurance coverage."
    service_text = "Supplier is not required to maintain insurance coverage."
    adapter = LocalNLIAdapter(config_path=_config(tmp_path), nli_pipeline=FakeNLIPipeline({master_text: 0.88}))
    retrieval = {
        "service_chunk_id": "s1",
        "service_chunk_text": service_text,
        "matched_master_chunks": [{"master_chunk_id": "m1", "master_chunk_text": master_text, "similarity_score": 1.0}],
    }

    prediction = run_auditor_and_guardrail(retrieval, master_text, service_text, adapter, "pair-1")

    assert prediction["has_contradiction"] is True
    assert prediction["guardrail_verified"] is True
    assert prediction["guardrail_action"] == "passed"


def test_local_nli_adapter_lazy_loads_model(tmp_path):
    adapter = LocalNLIAdapter(config_path=_config(tmp_path), nli_pipeline=FakeNLIPipeline({"Master": 0.2}))

    assert adapter._pipeline is not None
    prediction = adapter.judge("Service", [{"master_chunk_id": "m1", "master_chunk_text": "Master"}])

    assert prediction["has_contradiction"] is False


def test_pipeline_live_default_auditor_adapter_is_local_nli(monkeypatch, tmp_path):
    import pipeline_live

    monkeypatch.setenv("CROSSDOC_AUDITOR_ADAPTER", "local-nli")
    monkeypatch.setenv("CROSSDOC_AUDITOR_CONFIG", str(_config(tmp_path)))

    adapter = pipeline_live.get_auditor_adapter()

    assert isinstance(adapter, LocalNLIAdapter)
    assert adapter.model_id == "local-nli:unit-test-nli@test-revision"
