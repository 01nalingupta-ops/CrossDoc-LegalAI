"""Top-level orchestrator for CrossDoc-LegalAI benchmark reproduction.

Design note for future integration
==================================
Each ``call_*`` function below is a deliberate placeholder seam. Replace only the body of
that function with the real module's subprocess or import call, preserving its inputs and
outputs so the top-level JSON contracts remain stable:

* ``call_ingestion_module`` -> ``python ingestion.py <file> <master|service>`` returning
  ParsedDocument JSON files.
* ``call_dataset_generator`` -> ``python generate_data.py --corpus-dir ... --num-pairs ...
  --master-seed ... --out benchmark_dataset.json`` returning benchmark dataset entries.
* ``call_retrieval_module`` -> retrieval CLI returning RetrievalResult JSON.
* ``call_auditor_guardrail_module`` -> auditor/guardrail CLI returning VerifiedPrediction JSON.
* ``call_evaluation_module`` -> ``python evaluation.py --ground-truth ... --predictions ... --out ...``.
* ``call_report_export_module`` -> ``python report_export.py --eval-results ... --seed ...
  --dataset-hash ... --out-dir ...``.

Dry-run mode logs the planned sequence without side effects; normal runs call the real modules.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Sequence

from seed_manifest import generate_seed_manifest, validate_seed_manifest, write_manifest

MODEL_SLOTS = ["candidate_1", "candidate_2", "candidate_3", "candidate_4"]


def run_benchmark(args: argparse.Namespace) -> int:
    out_dir = Path(args.out_dir)
    planned = build_plan(args)
    if args.dry_run:
        print("[DRY-RUN] Planned CrossDoc-LegalAI benchmark call sequence:")
        for step in planned:
            print(f"[DRY-RUN] {step}")
        return 0

    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = load_or_generate_manifest(args)
    manifest_path = out_dir / "seed_manifest.json"
    write_manifest(manifest, manifest_path)
    print(f"[ORCH] Seed manifest ready: {manifest_path}")

    parsed_dir = out_dir / "parsed_documents"
    parsed_documents = call_ingestion_module(Path(args.corpus_dir), parsed_dir)

    dataset_path = out_dir / "benchmark_dataset.json"
    dataset = call_dataset_generator(Path(args.corpus_dir), args.num_pairs, manifest, dataset_path)
    manifest["dataset_content_hash"] = sha256_file(dataset_path)
    write_manifest(manifest, manifest_path)
    print(f"[ORCH] Updated dataset_content_hash: {manifest['dataset_content_hash']}")

    predictions_path = out_dir / "verified_predictions.jsonl"
    with predictions_path.open("w", encoding="utf-8") as handle:
        for model_slot in MODEL_SLOTS:
            retrieval_path = out_dir / f"retrieval_{model_slot}.jsonl"
            retrieval_results = call_retrieval_module(model_slot, dataset, parsed_documents, manifest, retrieval_path)
            model_predictions = call_auditor_guardrail_module(model_slot, dataset, retrieval_results, manifest)
            for prediction in model_predictions:
                handle.write(json.dumps(prediction, sort_keys=True) + "\n")
    print(f"[ORCH] Combined VerifiedPrediction log: {predictions_path}")

    eval_path = out_dir / "evaluation_results.json"
    call_evaluation_module(dataset_path, predictions_path, eval_path, manifest)

    report_dir = out_dir / "report_export"
    call_report_export_module(eval_path, manifest_path, manifest["dataset_content_hash"], report_dir)
    print("[ORCH] Benchmark pipeline complete.")
    return 0


def build_plan(args: argparse.Namespace) -> list[str]:
    manifest_source = args.seed_manifest or f"generate with master_seed={args.master_seed}, num_pairs={args.num_pairs}"
    steps = [
        f"Generate/load seed manifest ({manifest_source})",
        f"Ingest documents from {args.corpus_dir} via: python ingestion.py <file> <master|service>",
        "Generate benchmark_dataset.json via: python generate_data.py --corpus-dir ... --num-pairs ... --master-seed ...",
    ]
    for model_slot in MODEL_SLOTS:
        steps.append(f"{model_slot}: call retrieval module to produce RetrievalResult records")
        steps.append(f"{model_slot}: call auditor+guardrail module to produce VerifiedPrediction records")
    steps.extend(
        [
            "Evaluate predictions via: python evaluation.py --ground-truth ... --predictions ... --out ...",
            "Export report via: python report_export.py --eval-results ... --seed ... --dataset-hash ... --out-dir ...",
        ]
    )
    return steps


def load_or_generate_manifest(args: argparse.Namespace) -> dict[str, Any]:
    if args.seed_manifest:
        manifest = json.loads(Path(args.seed_manifest).read_text(encoding="utf-8"))
        print(f"[ORCH] Loaded seed manifest from {args.seed_manifest}")
    else:
        manifest = generate_seed_manifest(args.master_seed, args.num_pairs)
        print(f"[ORCH] Generated seed manifest from master_seed={args.master_seed}")
    errors = validate_seed_manifest(manifest)
    if errors:
        raise ValueError("Invalid seed manifest:\n" + "\n".join(errors))
    return manifest


def call_ingestion_module(corpus_dir: Path, parsed_dir: Path) -> list[dict[str, Any]]:
    """Parse corpus documents with the real Part 1 ingestion module."""
    import ingestion

    print("[ORCH] Ingestion module: parsing source corpus documents.")
    parsed_dir.mkdir(parents=True, exist_ok=True)
    documents: list[dict[str, Any]] = []
    files = sorted(path for path in corpus_dir.glob("**/*") if path.is_file() and path.suffix.lower() in {".pdf", ".docx", ".txt"})
    for path in files:
        doc_type = _infer_doc_type(path)
        parsed = ingestion.parse_document(str(path), doc_type)
        (parsed_dir / f"{parsed['doc_id']}.json").write_text(json.dumps(parsed, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        documents.append(parsed)
    return documents


def call_dataset_generator(corpus_dir: Path, num_pairs: int, manifest: dict[str, Any], out_path: Path) -> list[dict[str, Any]]:
    """Generate benchmark_dataset.json with the real Part 3 generator."""
    from generate_data import generate_dataset

    print("[ORCH] Dataset generator: creating benchmark_dataset.json from corpus templates.")
    dataset, _fragment = generate_dataset(corpus_dir, num_pairs, manifest["master_seed"], out_path)
    return dataset


def call_retrieval_module(
    model_slot: str,
    dataset: list[dict[str, Any]],
    parsed_documents: list[dict[str, Any]],
    manifest: dict[str, Any],
    out_path: Path,
) -> list[dict[str, Any]]:
    """Produce RetrievalResult records with the real Part 2 matchmaker."""
    import ingestion
    import matchmaker

    print(f"[ORCH] Retrieval module for {model_slot}: producing RetrievalResult records.")
    parsed_by_path = {Path(doc["source_path"]).resolve(): doc for doc in parsed_documents}
    results: list[dict[str, Any]] = []
    embedder = _configured_benchmark_embedder(matchmaker)
    for item in dataset:
        master_doc = _parsed_for_path(Path(item["master_doc_path"]), "master", parsed_by_path, ingestion)
        service_doc = _parsed_for_path(Path(item["service_doc_path"]), "service", parsed_by_path, ingestion)
        retrieval = matchmaker.retrieve_pair(master_doc, service_doc, embedder=embedder)
        record = {"pair_id": item["pair_id"], "model_slot": model_slot, "retrieval_results": retrieval, "retrieval_tiebreak_seed": manifest["retrieval_tiebreak_seed"]}
        results.append(record)
    with out_path.open("w", encoding="utf-8") as handle:
        for result in results:
            handle.write(json.dumps(result, sort_keys=True) + "\n")
    return results


def call_auditor_guardrail_module(
    model_slot: str,
    dataset: list[dict[str, Any]],
    retrieval_results: list[dict[str, Any]],
    manifest: dict[str, Any],
) -> list[dict[str, Any]]:
    """Produce one pair-level VerifiedPrediction per dataset pair through Part 4 guardrail."""
    import ingestion
    import auditor_guardrail
    import numeric_reasoning
    import clause_classifier
    import risk_scoring
    from model_adapter import MockAdapter
    from pipeline_live import get_auditor_adapter

    if model_slot in {"candidate_3", "candidate_4"}:
        print(f"[ORCH] Auditor+guardrail for {model_slot}: adapter not configured; logging placeholder negatives.")
        return [_placeholder_prediction(item, model_slot, retrieval_results[index]) for index, item in enumerate(dataset)]

    adapter = MockAdapter() if model_slot == "candidate_1" else get_auditor_adapter()
    print(f"[ORCH] Auditor+guardrail for {model_slot}: using {adapter.model_id}.")
    predictions: list[dict[str, Any]] = []
    for item, retrieval_record in zip(dataset, retrieval_results):
        master_doc = ingestion.parse_document(item["master_doc_path"], "master")
        service_doc = ingestion.parse_document(item["service_doc_path"], "service")
        chunk_predictions = []
        for result in retrieval_record["retrieval_results"]:
            chunk_prediction = auditor_guardrail.run_auditor_and_guardrail(
                result, master_doc["full_text"], service_doc["full_text"], adapter, item["pair_id"]
            )
            chunk_prediction["numeric_evidence"] = numeric_reasoning.evidence_for_retrieval_result(result)
            category = clause_classifier.infer_category(result["service_chunk_text"])
            chunk_prediction["category"] = category
            chunk_prediction["risk_score"] = risk_scoring.score_prediction(chunk_prediction, category)["risk_score"]
            chunk_predictions.append(chunk_prediction)
        best = _select_pair_prediction(chunk_predictions)
        best["model_id"] = model_slot
        best["adapter_model_id"] = adapter.model_id
        best["model_slot"] = model_slot
        predictions.append(best)
    return predictions


def call_evaluation_module(dataset_path: Path, predictions_path: Path, out_path: Path, manifest: dict[str, Any]) -> dict[str, Any]:
    """Compute metrics with the real Part 5 evaluation module."""
    from evaluation import evaluate

    print("[ORCH] Evaluation module: computing metrics and statistical tests.")
    ground_truth = json.loads(dataset_path.read_text(encoding="utf-8"))
    predictions = [json.loads(line) for line in predictions_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    results = evaluate(ground_truth, predictions, bootstrap_resamples=100, seed=manifest["bootstrap_resampling_seed"])
    out_path.write_text(json.dumps(results, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return results


def call_report_export_module(eval_path: Path, seed_manifest_path: Path, dataset_hash: str, out_dir: Path) -> None:
    """Export report artifacts with the real Part 6 report exporter."""
    from report_export import export_report

    print("[ORCH] Report export module: creating figures/tables with seed and dataset hash in filenames.")
    manifest = json.loads(seed_manifest_path.read_text(encoding="utf-8"))
    eval_results = json.loads(eval_path.read_text(encoding="utf-8"))
    export_report(eval_results, manifest["master_seed"], dataset_hash, out_dir)


def _configured_benchmark_embedder(matchmaker_module: Any) -> Any:
    backend = os.environ.get("CROSSDOC_BENCHMARK_EMBEDDING_BACKEND", "sentence-transformers")
    if backend == "sentence-transformers":
        return None
    if backend == "keyword-fixture":
        return matchmaker_module.KeywordFixtureEmbedder()
    raise ValueError("CROSSDOC_BENCHMARK_EMBEDDING_BACKEND must be 'sentence-transformers' or 'keyword-fixture'")


def _infer_doc_type(path: Path) -> str:
    name = path.name.lower()
    return "master" if any(token in name for token in ("msa", "master")) else "service"


def _parsed_for_path(path: Path, doc_type: str, parsed_by_path: dict[Path, dict[str, Any]], ingestion_module: Any) -> dict[str, Any]:
    resolved = path.resolve()
    if resolved not in parsed_by_path:
        parsed_by_path[resolved] = ingestion_module.parse_document(str(path), doc_type)
    return parsed_by_path[resolved]


def _select_pair_prediction(predictions: list[dict[str, Any]]) -> dict[str, Any]:
    severity_rank = {"High": 3, "Medium": 2, "Low": 1, "None": 0}
    return dict(max(predictions, key=lambda row: (bool(row.get("has_contradiction")), severity_rank.get(row.get("severity", "None"), 0), float(row.get("confidence", 0.0)))))


def _placeholder_prediction(item: dict[str, Any], model_slot: str, retrieval_result: dict[str, Any]) -> dict[str, Any]:
    return {
        "pair_id": item["pair_id"],
        "model_id": model_slot,
        "model_slot": model_slot,
        "has_contradiction": False,
        "confidence": 0.0,
        "severity": "None",
        "conflict_explanation": "No adapter configured for this candidate slot; retained in the audit log but excluded from evaluation metrics.",
        "msa_exact_quote": "",
        "sow_exact_quote": "",
        "suggested_redline": "",
        "guardrail_verified": True,
        "guardrail_action": "passed",
        "evaluation_excluded": True,
        "retrieval_result": retrieval_result,
    }

def sha256_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the placeholder CrossDoc-LegalAI benchmark pipeline.")
    parser.add_argument("--corpus-dir", default="corpus", help="Directory containing raw document pairs")
    parser.add_argument("--out-dir", default="benchmark_run", help="Directory for all generated artifacts")
    parser.add_argument("--seed-manifest", help="Existing seed_manifest.json to load")
    parser.add_argument("--master-seed", type=int, default=42, help="Master seed used when generating a manifest")
    parser.add_argument("--num-pairs", type=int, default=60, help="Number of benchmark pairs to generate")
    parser.add_argument("--dry-run", action="store_true", help="Print the planned call sequence without writing artifacts")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    return run_benchmark(parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
