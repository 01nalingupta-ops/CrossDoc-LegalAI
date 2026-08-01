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

A reviewer can run the placeholder pipeline today. In the final integrated system, the same top-level command sequence will call the real modules behind the documented JSON contracts.

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

### 3. Run the full placeholder orchestration

```bash
python run_benchmark.py --seed-manifest seed_manifest.json --corpus-dir raw_doc_pairs --out-dir reproduced_run
```

Today this produces mock JSON artifacts with `[PLACEHOLDER]` log lines. In the final system, each placeholder seam in `run_benchmark.py` is replaced by the real module CLI/function call while preserving the same JSON contracts.

### 4. Final integrated module commands represented by the orchestrator

The orchestrator is written against these documented CLI shapes:

```bash
python ingestion.py <file> <master|service>
python generate_data.py --corpus-dir raw_doc_pairs --num-pairs 60 --master-seed 42 --out benchmark_dataset.json
python evaluation.py --ground-truth benchmark_dataset.json --predictions verified_predictions.jsonl --out evaluation_results.json
python report_export.py --eval-results evaluation_results.json --seed seed_manifest.json --dataset-hash <sha256> --out-dir report_export
```

Retrieval and auditor/guardrail modules are also invoked through placeholder seams and are expected to exchange `RetrievalResult` and `VerifiedPrediction` JSON records.

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

## Future wiring design note

Future engineers should replace the body, not the signature, of each placeholder function in `run_benchmark.py`:

- `call_ingestion_module`: call Part 1's `python ingestion.py <file> <doc_type>` and store ParsedDocument JSON.
- `call_dataset_generator`: call the dataset generator CLI and write `benchmark_dataset.json`.
- `call_retrieval_module`: call the retrieval module and consume/produce `RetrievalResult` JSON.
- `call_auditor_guardrail_module`: call the auditor+guardrail module and return `VerifiedPrediction` records.
- `call_evaluation_module`: call the evaluator and write `evaluation_results.json`.
- `call_report_export_module`: call report export with the evaluation path, seed manifest, dataset hash, and output directory.

Keeping those seams stable allows the top-level pipeline to remain reproducible while individual modules evolve independently.
