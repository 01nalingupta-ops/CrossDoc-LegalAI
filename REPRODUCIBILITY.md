# CrossDoc-LegalAI Reproducibility Guide

This guide explains how a peer reviewer can rerun the CrossDoc-LegalAI benchmark from a published release package. It is written around JSON contracts and command-line interfaces so each module can be independently implemented and audited.

## What is a seed manifest?

A `seed_manifest.json` is the published record of all random choices used in the benchmark. Instead of only publishing one master seed, CrossDoc-LegalAI derives named sub-seeds for dataset generation, template selection, defect placement, severity assignment, retrieval tie-breaking, and evaluation bootstrap resampling.

Publishing these seeds matters because it lets reviewers regenerate the same benchmark choices, verify that no hidden randomness changed the results, and independently compare the frozen benchmark dataset against the dataset hash recorded in the manifest.

The derivation scheme is:

1. Build the exact UTF-8 message `CrossDoc-LegalAI-seed-v1:<master_seed>:<label>`.
2. Compute SHA-256 of that message.
3. Interpret the first 8 digest bytes as an unsigned big-endian integer.
4. Reduce it modulo `2_147_483_647`.

For per-pair dataset seeds, labels are `dataset_generation_seed:0000`, `dataset_generation_seed:0001`, and so on through the requested number of pairs.

## Commands to regenerate a benchmark run

A reviewer can run the integrated pipeline today. The top-level orchestrator preserves a side-effect-free dry-run mode, while normal runs call the real ingestion, dataset generation, retrieval, auditor/guardrail, evaluation, and report-export modules behind the documented JSON contracts.

### 1. Create or validate the seed manifest

```bash
python seed_manifest.py --master-seed 42 --num-pairs 60 --out seed_manifest.json
```

If the publication already bundles `seed_manifest.json`, use that file rather than generating a new one.

### 2. Inspect the planned pipeline without writing outputs

```bash
python run_benchmark.py --seed-manifest seed_manifest.json --corpus-dir raw_doc_pairs --out-dir reproduced_run --dry-run
```

This prints the intended ingestion, dataset generation, retrieval, auditor/guardrail, evaluation, and report-export call sequence.

### 3. Run the full orchestration

```bash
python run_benchmark.py --seed-manifest seed_manifest.json --corpus-dir raw_doc_pairs --out-dir reproduced_run
```

This produces a real `benchmark_dataset.json`, `verified_predictions.jsonl`, `evaluation_results.json`, and `report_export/` artifacts. Retrieval uses the real matchmaker with the default `sentence-transformers` backend; constrained smoke-test environments may set `CROSSDOC_BENCHMARK_EMBEDDING_BACKEND=keyword-fixture` to exercise the same orchestration contracts without downloading the embedding model. Candidate slots are still explicit: `candidate_1` is the rule-based `MockAdapter` baseline, `candidate_2` uses the live pipeline adapter hook (also `MockAdapter` during Part 9), and `candidate_3`/`candidate_4` are logged as unconfigured placeholder candidate slots until future adapters are added.

### 4. Module commands represented by the orchestrator

The orchestrator is written against these documented CLI shapes:

```bash
python ingestion.py <file> <master|service>
python generate_data.py --corpus-dir raw_doc_pairs --num-pairs 60 --master-seed 42 --out benchmark_dataset.json
python evaluation.py --ground-truth benchmark_dataset.json --predictions verified_predictions.json --out evaluation_results.json
python report_export.py --eval-results evaluation_results.json --seed <master-seed> --dataset-hash <sha256> --out-dir report_export
```

Retrieval and auditor/guardrail are invoked through stable Python seams in `run_benchmark.py` and exchange `RetrievalResult` and `VerifiedPrediction` JSON-compatible records. The persisted combined prediction log is JSONL for auditability; evaluation loads those records directly in-process.

## What a publication release package should bundle

A complete reproducibility package should include:

- Frozen `benchmark_dataset.json`.
- The raw master/service document pairs used to generate the benchmark.
- The exact `seed_manifest.json` used for the published run.
- Raw per-model, per-pair prediction logs, ideally as `verified_predictions.jsonl`.
- `evaluation_results.json` containing per-model metrics plus pairwise McNemar's and DeLong's test results.
- Exported figures and tables with filenames embedding the master seed and/or `dataset_content_hash`.
- This `REPRODUCIBILITY.md` file.

The `dataset_content_hash` field in `seed_manifest.json` should be the SHA-256 hash of the frozen `benchmark_dataset.json`. Reviewers should recompute the hash and compare it to the manifest before trusting downstream evaluation results.

## Wiring maintenance note

Future engineers should preserve the signatures of the `call_*` functions in `run_benchmark.py` when swapping implementation details. This keeps the top-level pipeline reproducible while individual modules and candidate adapters evolve independently.
