import argparse
import json
from pathlib import Path

import run_benchmark


def _args(tmp_path, dry_run=False):
    return argparse.Namespace(
        corpus_dir="base_corpus",
        out_dir=str(tmp_path / "benchmark"),
        seed_manifest=None,
        master_seed=42,
        num_pairs=1,
        dry_run=dry_run,
    )


def test_dry_run_has_no_side_effects(tmp_path, capsys):
    result = run_benchmark.run_benchmark(_args(tmp_path, dry_run=True))

    assert result == 0
    assert not (tmp_path / "benchmark").exists()
    output = capsys.readouterr().out
    assert "[DRY-RUN]" in output
    assert "candidate_1" in output
    assert "candidate_4" in output


def test_non_dry_run_wires_real_call_seams(monkeypatch, tmp_path):
    calls = []

    def fake_ingestion(corpus_dir, parsed_dir):
        calls.append(("ingestion", Path(corpus_dir), Path(parsed_dir)))
        parsed_dir.mkdir(parents=True, exist_ok=True)
        return []

    def fake_dataset(corpus_dir, num_pairs, manifest, out_path):
        calls.append(("dataset", Path(corpus_dir), num_pairs, Path(out_path)))
        dataset = [
            {
                "pair_id": "pair-0001",
                "has_contradiction": False,
                "contradiction_category": None,
                "master_doc_path": "master.txt",
                "service_doc_path": "service.txt",
            }
        ]
        out_path.write_text(json.dumps(dataset) + "\n", encoding="utf-8")
        return dataset

    def fake_retrieval(model_slot, dataset, parsed_documents, manifest, out_path):
        calls.append(("retrieval", model_slot, Path(out_path)))
        rows = [{"pair_id": "pair-0001", "model_slot": model_slot, "retrieval_results": []}]
        out_path.write_text(json.dumps(rows[0]) + "\n", encoding="utf-8")
        return rows

    def fake_auditor(model_slot, dataset, retrieval_results, manifest):
        calls.append(("auditor", model_slot))
        return [
            {
                "pair_id": "pair-0001",
                "model_id": model_slot,
                "model_slot": model_slot,
                "has_contradiction": False,
                "confidence": 0.0,
                "severity": "None",
                "conflict_explanation": "test",
                "msa_exact_quote": "",
                "sow_exact_quote": "",
                "suggested_redline": "",
                "guardrail_verified": True,
                "guardrail_action": "passed",
            }
        ]

    def fake_evaluation(dataset_path, predictions_path, out_path, manifest):
        calls.append(("evaluation", Path(dataset_path), Path(predictions_path), Path(out_path)))
        result = {"per_model_metrics": {}, "pairwise_mcnemar": {}, "pairwise_delong": {}}
        out_path.write_text(json.dumps(result) + "\n", encoding="utf-8")
        return result

    def fake_report(eval_path, seed_manifest_path, dataset_hash, out_dir):
        calls.append(("report", Path(eval_path), Path(seed_manifest_path), dataset_hash, Path(out_dir)))
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "report.txt").write_text("ok\n", encoding="utf-8")

    monkeypatch.setattr(run_benchmark, "call_ingestion_module", fake_ingestion)
    monkeypatch.setattr(run_benchmark, "call_dataset_generator", fake_dataset)
    monkeypatch.setattr(run_benchmark, "call_retrieval_module", fake_retrieval)
    monkeypatch.setattr(run_benchmark, "call_auditor_guardrail_module", fake_auditor)
    monkeypatch.setattr(run_benchmark, "call_evaluation_module", fake_evaluation)
    monkeypatch.setattr(run_benchmark, "call_report_export_module", fake_report)

    result = run_benchmark.run_benchmark(_args(tmp_path))

    assert result == 0
    assert [call[0] for call in calls].count("retrieval") == len(run_benchmark.MODEL_SLOTS)
    assert [call[0] for call in calls].count("auditor") == len(run_benchmark.MODEL_SLOTS)
    predictions_path = tmp_path / "benchmark" / "verified_predictions.jsonl"
    assert predictions_path.is_file()
    assert len(predictions_path.read_text(encoding="utf-8").splitlines()) == len(run_benchmark.MODEL_SLOTS)
    assert (tmp_path / "benchmark" / "report_export" / "report.txt").is_file()


def test_benchmark_embedding_backend_can_use_fixture(monkeypatch):
    import matchmaker

    monkeypatch.setenv("CROSSDOC_BENCHMARK_EMBEDDING_BACKEND", "keyword-fixture")
    assert isinstance(run_benchmark._configured_benchmark_embedder(matchmaker), matchmaker.KeywordFixtureEmbedder)

    monkeypatch.setenv("CROSSDOC_BENCHMARK_EMBEDDING_BACKEND", "sentence-transformers")
    assert run_benchmark._configured_benchmark_embedder(matchmaker) is None


def test_unconfigured_candidate_placeholders_keep_verified_prediction_schema():
    item = {"pair_id": "pair-0001"}
    retrieval = {"pair_id": "pair-0001", "retrieval_results": []}

    prediction = run_benchmark._placeholder_prediction(item, "candidate_3", retrieval)

    assert prediction["model_id"] == "candidate_3"
    assert prediction["has_contradiction"] is False
    assert prediction["guardrail_verified"] is True
    assert prediction["guardrail_action"] == "passed"
