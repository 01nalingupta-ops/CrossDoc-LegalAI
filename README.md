# CrossDoc-LegalAI Part 1: Document Ingestion

This repository contains a standalone ingestion module that converts a contract document into the `ParsedDocument` JSON structure used by downstream CrossDoc-LegalAI modules.

## Install

```bash
python -m pip install -r requirements.txt
```

PDF digital text extraction uses PyMuPDF. Scanned/image-only PDF OCR fallback uses PyMuPDF rendering plus `pytesseract` and Pillow. `pytesseract` also requires the system Tesseract binary to be installed and available on `PATH`.

## Run from Python

```python
from ingestion import parse_document

parsed = parse_document("/path/to/contract.pdf", "master")
```

`doc_type` must be either `"master"` or `"service"`.

## Run from the CLI

```bash
python ingestion.py /path/to/contract.pdf master
```

The CLI prints the parsed JSON document to stdout.

## Output schema

```json
{
  "doc_id": "string, unique per document, derived from filename and content hash",
  "doc_type": "master | service",
  "source_path": "string, original file path",
  "full_text": "string, complete extracted text",
  "page_count": "integer",
  "extraction_method": "digital | ocr",
  "chunks": [
    {
      "chunk_id": "string, unique within doc, e.g. <doc_id>-chunk-0001",
      "text": "string, up to 800 chars",
      "char_start": "integer, offset into full_text",
      "char_end": "integer, offset into full_text",
      "page_num": "integer, page containing char_start",
      "overlap_chars": 150
    }
  ]
}
```

Chunking uses fixed 800-character windows with 150 characters of overlap. Documents shorter than 800 characters still produce exactly one chunk. Multi-page PDFs track page starts while constructing `full_text`, so each chunk's `page_num` maps to the page containing the chunk's `char_start` offset.

## Tests

```bash
pytest -q
```

The PDF tests synthesize test PDFs with PyMuPDF and are skipped automatically when PyMuPDF is unavailable. The OCR fallback test monkeypatches OCR text extraction so it does not require the external Tesseract binary during unit testing.

# CrossDoc-LegalAI Part 2: Matchmaker Agent

`matchmaker.py` performs semantic retrieval only. It builds an index over Master document chunks and queries that index with Service document chunks. It does not call LLMs, judge contradictions, or apply guardrails.

## Matchmaker input JSON

The CLI expects two ParsedDocument JSON files, one with `doc_type: "master"` and one with `doc_type: "service"`:

```json
{
  "doc_id": "string",
  "doc_type": "master | service",
  "source_path": "string",
  "full_text": "string",
  "page_count": "integer",
  "extraction_method": "digital | ocr",
  "chunks": [
    {
      "chunk_id": "string",
      "text": "string, ~800 chars",
      "char_start": "int",
      "char_end": "int",
      "page_num": "int",
      "overlap_chars": 150
    }
  ]
}
```

## Matchmaker output JSON

The output is a JSON array with one RetrievalResult object per Service chunk. Each object contains the top 2 Master chunk matches by cosine similarity:

```json
[
  {
    "service_chunk_id": "string, from the Service ParsedDocument",
    "service_chunk_text": "string",
    "matched_master_chunks": [
      {
        "master_chunk_id": "string",
        "master_chunk_text": "string",
        "similarity_score": "float 0.0-1.0"
      }
    ]
  }
]
```

## Matchmaker usage

```bash
python matchmaker.py tests/fixtures/master_parsed.json tests/fixtures/service_payment_confidentiality.json
```

Programmatic use:

```python
from matchmaker import build_master_index, retrieve_for_service_doc

index = build_master_index(master_doc)
retrieval_results = retrieve_for_service_doc(index, service_doc)
```

The default embedder is fixed to the sentence-transformers model `all-MiniLM-L6-v2`. Master chunk embeddings are computed once and cached inside the index; Service chunks are embedded in a single batch per retrieval call.

# CrossDoc-LegalAI Part 3: Synthetic Dataset Generator

`generate_data.py` creates a synthetic adversarial benchmark dataset from fictional MSA/SOW template pairs in `base_corpus/`. It produces clean pairs and single-contradiction pairs only; it does not perform retrieval, auditing, guardrails, or evaluation.

## Dataset generator CLI

```bash
python generate_data.py --corpus-dir base_corpus/ --num-pairs 60 --master-seed 42 --out benchmark_dataset.json
```

The command also writes `dataset_generation_seeds.json` beside the dataset output.

## Dataset output schema

`benchmark_dataset.json` is a JSON array. Every entry has exactly this shape:

```json
{
  "pair_id": "string",
  "seed": "integer",
  "master_doc_path": "string",
  "service_doc_path": "string",
  "has_contradiction": "boolean",
  "contradiction_category": "Payment Terms | Liability Cap | IP Ownership | Termination Notice | Governing Jurisdiction | Confidentiality Scope | Indemnification | Insurance Requirements | null",
  "severity": "High | Medium | Low | None",
  "master_clause_location": {"page": "int", "char_offset": "int", "clause_id": "string"},
  "service_clause_location": {"page": "int", "char_offset": "int", "clause_id": "string"},
  "master_exact_text": "string, verbatim substring of the master document",
  "service_exact_text": "string, verbatim substring of the service document",
  "generation_model": "string",
  "generation_timestamp": "ISO-8601 string"
}
```

The generator validates each quote before output: `master_exact_text` and `service_exact_text` must be real substrings at or near the recorded `char_offset` in the generated documents.

## Defect taxonomy

The fixed supported contradiction categories are:

1. Payment Terms
2. Liability Cap
3. IP Ownership
4. Termination Notice
5. Governing Jurisdiction
6. Confidentiality Scope
7. Indemnification
8. Insurance Requirements

Clean pairs have `has_contradiction: false`, `contradiction_category: null`, and `severity: "None"`.

## Dataset seed derivation

All choices are deterministic from one `master_seed`. For a per-pair decision, the generator builds the exact UTF-8 message:

```text
CrossDoc-LegalAI-dataset-v1:<master_seed>:<purpose>:<pair_index>
```

For global manifest seeds, it omits `<pair_index>`:

```text
CrossDoc-LegalAI-dataset-v1:<master_seed>:<purpose>
```

It then computes SHA-256, interprets the first 8 digest bytes as an unsigned big-endian integer, and reduces the value modulo `2_147_483_647`. Re-running with the same corpus, pair count, and `master_seed` produces byte-identical selection decisions and generated JSON.

# CrossDoc-LegalAI Part 4: Auditor Agent and Zero-Hallucination Guardrail

This standalone module consumes Part 2 `RetrievalResult` objects and produces one verified contradiction prediction per `(model, service_chunk)` pair. It intentionally does **not** implement retrieval, embeddings, evaluation metrics, statistics, or a UI.

## Component A: Auditor Agent (LangGraph Node 2)

`model_adapter.py` defines the adapter boundary used for the four-model bake-off:

```python
class AuditorModelAdapter(ABC):
    model_id: str

    def judge(self, service_chunk_text: str, master_chunks: list[dict]) -> dict:
        ...
```

Adapters receive the Service chunk text and the list of matched Master chunks from the retrieval result. They return the raw Auditor schema:

```json
{
  "has_contradiction": "boolean",
  "confidence": "float 0.0-1.0",
  "severity": "High | Medium | Low | None",
  "conflict_explanation": "string",
  "msa_exact_quote": "string, verbatim quote from a matched Master chunk or empty",
  "sow_exact_quote": "string, verbatim quote from the Service chunk or empty",
  "suggested_redline": "string, corrected Service clause or empty"
}
```

Three adapters are included:

- `MockAdapter`: deterministic, no-network test adapter used as the `candidate_1` rule baseline.
- `LocalNLIAdapter`: local HuggingFace NLI adapter used by the live pipeline's `candidate_2` hook. It runs the configured NLI model on each matched Master chunk as `premise=master_chunk_text` and `hypothesis=service_chunk_text`, selects the highest contradiction probability, derives severity from configured bands, and emits exact chunk texts as quotes so the deterministic guardrail can verify evidence by construction.
- `OpenAICompatibleAdapter`: generic `/chat/completions` adapter configurable with `OPENAI_COMPATIBLE_API_KEY`, `OPENAI_COMPATIBLE_BASE_URL`, and `OPENAI_COMPATIBLE_MODEL`. It is not required by the default pipeline.

`LocalNLIAdapter` is configured in `auditor_config.json`. The default config pins `cross-encoder/nli-deberta-v3-small` at revision `main`, sets `contradiction_threshold`, defines confidence-to-severity bands, and lists label aliases for contradiction/neutral/entailment outputs. Override the config path with `CROSSDOC_AUDITOR_CONFIG=/path/to/auditor_config.json`. The adapter lazy-loads `transformers.pipeline(...)` on first use so importing `model_adapter.py` does not download or initialize model weights.

The live pipeline uses `CROSSDOC_AUDITOR_ADAPTER=local-nli` by default. Set `CROSSDOC_AUDITOR_ADAPTER=mock` only for dependency-light smoke tests or deterministic UI demos.

Every real adapter call sends `temperature=0.0` where the backend exposes a sampling temperature. Adding another provider should only require a small new subclass of `AuditorModelAdapter`; orchestration code should not import provider SDKs or provider-specific settings.

### Fixed prompt template

The prompt in `AUDITOR_PROMPT_TEMPLATE` asks the model to judge contradictions between a Service/SOW clause and matched Master/MSA clauses, considering these categories: Payment Terms, Liability Cap, IP Ownership, Termination Notice, Governing Jurisdiction, Confidentiality Scope, Indemnification, and Insurance Requirements. It requires exactly the seven raw-output keys above, requires JSON-only output, requires `severity` to be `High`, `Medium`, `Low`, or `None`, and instructs the model to use only verbatim quotes present in the provided chunks.

## Component B: Deterministic Guardrail (LangGraph Node 3)

`apply_guardrail(raw_prediction, master_full_text, service_full_text)` verifies quotes without an LLM:

1. `msa_exact_quote` must be a substring of the full Master document after whitespace normalization.
2. `sow_exact_quote` must be a substring of the full Service document after whitespace normalization.
3. Empty quotes are valid for no-contradiction predictions.
4. If both quote checks pass, the guardrail sets `guardrail_verified=true` and `guardrail_action="passed"`.
5. If either quote check fails, the module withholds the claim by forcing `has_contradiction=false`, `confidence=0.0`, `severity="None"`, and `suggested_redline=""`, while preserving the explanation and quotes for audit/debug visibility. The guardrail sets `guardrail_verified=false` and `guardrail_action="claim_withheld"`.

The final output schema is exactly:

```json
{
  "pair_id": "string",
  "model_id": "string",
  "service_chunk_id": "string",
  "has_contradiction": "boolean",
  "confidence": "float 0.0-1.0",
  "severity": "High | Medium | Low | None",
  "conflict_explanation": "string",
  "msa_exact_quote": "string",
  "sow_exact_quote": "string",
  "suggested_redline": "string",
  "guardrail_verified": "boolean",
  "guardrail_action": "passed | quote_rejected | claim_withheld"
}
```

This implementation uses `claim_withheld` for failed verification because the claim is not safe for end users. `quote_rejected` is reserved by the shared schema for consumers that want to represent quote-only rejection separately.

## Pipeline API

```python
from auditor_guardrail import run_auditor_and_guardrail
from model_adapter import MockAdapter

verified = run_auditor_and_guardrail(
    retrieval_result=retrieval_result,
    master_full_text=master_full_text,
    service_full_text=service_full_text,
    adapter=MockAdapter(),
    pair_id="caller-context-id",
)
```

## CLI

The CLI expects a JSON object with `pair_id`, `master_full_text`, `service_full_text`, and `retrieval_results` (an array of Part 2 `RetrievalResult` objects). It uses `MockAdapter` by default and prints the verified prediction array.

```bash
python auditor_guardrail.py tests/fixtures/auditor_guardrail_fixture.json
```

## Part 4 tests

```bash
pytest -q tests/test_auditor_guardrail.py
```

The fixture file `tests/fixtures/auditor_guardrail_fixture.json` contains five hand-written retrieval results covering payment contradiction, liability contradiction, clean confidentiality, clean termination, and a hallucinated quote path that the guardrail withholds.

# CrossDoc-LegalAI Part 5: Offline Evaluation Metrics

`evaluation.py` is a standalone offline statistics module. It consumes reduced ground-truth records and per-model prediction records, computes model metrics and paired statistical comparisons, and writes one `evaluation_results.json` object. It does not call LLMs and does not import any other CrossDoc-LegalAI module.

## Evaluation inputs

Ground truth is a JSON array with one record per benchmark pair:

```json
{
  "pair_id": "string",
  "has_contradiction": "boolean",
  "contradiction_category": "string or null",
  "severity": "High | Medium | Low | None"
}
```

Predictions are a JSON array with one record per `(model, pair)`; multiple `model_id` values may be interleaved:

```json
{
  "pair_id": "string",
  "model_id": "string",
  "has_contradiction": "boolean",
  "confidence": "float 0.0-1.0 or null",
  "severity": "High | Medium | Low | None",
  "guardrail_verified": "boolean"
}
```

## Metrics and statistical tests

For each model, the module computes accuracy, precision, recall, F1, faithfulness, ROC-AUC, an overall confusion matrix, confusion matrices by contradiction category, and bootstrap 95% confidence intervals for accuracy, F1, and AUC. Faithfulness is the fraction of positive model claims where `guardrail_verified=true`; models with no positive claims receive faithfulness `1.0` because there were no unsupported positive claims.

ROC-AUC uses `confidence` as the ranking score. If `confidence` is missing or `null`, the evaluator falls back to an ordinal severity score: `None=0`, `Low=1`, `Medium=2`, and `High=3`. This fallback is implemented in `_prediction_score(...)` so model outputs remain evaluable even when confidence is unavailable.

For every pair of model IDs present in the prediction input, the module computes:

- McNemar's paired comparison from per-pair correctness. For fewer than 25 discordant pairs it uses an exact two-sided binomial test; otherwise it uses the continuity-corrected chi-squared statistic `(abs(n01 - n10) - 1)^2 / (n01 + n10)` with 1 degree of freedom.
- DeLong's correlated ROC-AUC comparison using Mann-Whitney structural components (`V10`/`V01`) to estimate variance and covariance on the same benchmark pairs, followed by a two-sided standard-normal p-value.
- Holm-Bonferroni correction separately across the McNemar p-value family and the DeLong p-value family at `alpha=0.05`.

The generic `paired_mcnemar_comparison(predictions_a, predictions_b, ground_truth)` helper can be reused for ablations such as guardrail-on vs guardrail-off or agentic pipeline vs baseline; it is not hardcoded to bake-off `model_id` groupings.

## Evaluation output schema

The CLI writes exactly one JSON object:

```json
{
  "per_model_metrics": {
    "<model_id>": {
      "accuracy": "float",
      "precision": "float",
      "recall": "float",
      "f1": "float",
      "faithfulness": "float",
      "auc": "float",
      "confusion_matrix": {"tp": "int", "tn": "int", "fp": "int", "fn": "int"},
      "confusion_matrix_by_category": {
        "<category>": {"tp": "int", "tn": "int", "fp": "int", "fn": "int"}
      },
      "bootstrap_ci": {
        "accuracy": ["low", "high"],
        "f1": ["low", "high"],
        "auc": ["low", "high"]
      }
    }
  },
  "pairwise_mcnemar": {
    "<model_a>__vs__<model_b>": {
      "chi2_or_exact": "float",
      "p_value": "float",
      "p_value_holm_adjusted": "float",
      "significant": "boolean"
    }
  },
  "pairwise_delong": {
    "<model_a>__vs__<model_b>": {
      "z": "float",
      "p_value": "float",
      "p_value_holm_adjusted": "float",
      "significant": "boolean"
    }
  }
}
```

## Evaluation CLI

```bash
python evaluation.py --ground-truth tests/fixtures/evaluation_ground_truth.json \
  --predictions tests/fixtures/evaluation_predictions.json \
  --out evaluation_results.json
```

Optional flags:

- `--bootstrap-resamples`: defaults to `1000`.
- `--seed`: deterministic bootstrap seed, defaults to `12345`.

## Part 5 tests

```bash
pytest -q tests/test_evaluation.py
```

The tests include hand-checked confusion-matrix metrics, a McNemar exact-binomial p-value, a severity-fallback AUC case, and a bootstrap CI sanity check that verifies larger samples produce narrower intervals than smaller samples with the same error rate.

# CrossDoc-LegalAI Part 6: Report Exporter

`report_export.py` is a presentation-only module. It consumes a completed `evaluation_results.json` object plus two caller-supplied run identifiers (`seed` and `dataset_hash`) and exports paper-ready figures, LaTeX table snippets, CSV inspection tables, and draft captions. It does **not** compute metrics or statistical tests; all numbers are trusted from the input JSON.

## Report input schema

The required input is one JSON object with this shape:

```json
{
  "per_model_metrics": {
    "<model_id>": {
      "accuracy": "float",
      "precision": "float",
      "recall": "float",
      "f1": "float",
      "faithfulness": "float",
      "auc": "float",
      "confusion_matrix": {"tp": "int", "tn": "int", "fp": "int", "fn": "int"},
      "confusion_matrix_by_category": {
        "<category>": {"tp": "int", "tn": "int", "fp": "int", "fn": "int"}
      },
      "bootstrap_ci": {
        "accuracy": ["low", "high"],
        "f1": ["low", "high"],
        "auc": ["low", "high"]
      }
    }
  },
  "pairwise_mcnemar": {
    "<model_a>__vs__<model_b>": {
      "p_value": "float",
      "p_value_holm_adjusted": "float",
      "significant": "boolean"
    }
  },
  "pairwise_delong": {
    "<model_a>__vs__<model_b>": {
      "p_value": "float",
      "p_value_holm_adjusted": "float",
      "significant": "boolean"
    }
  }
}
```

Optional ROC points can be supplied as a separate JSON file with:

```json
{"<model_id>": [{"fpr": 0.0, "tpr": 0.0}, {"fpr": 1.0, "tpr": 1.0}]}
```

When ROC points are not provided, the ROC figure gracefully falls back to an AUC bar chart and notes that the figure is based only on per-model AUC values.

## Exported report artifacts

For every figure, the exporter writes vector-style `pdf` and `eps` files plus a 300-DPI `png` preview. It also writes a `captions.md` file with one draft caption per figure. The required figures are:

1. Grouped bar chart for accuracy, precision, recall, and F1 by model.
2. Overlaid ROC curves, or AUC-bar fallback when raw ROC points are absent.
3. Confusion-matrix small multiples by model.
4. Bootstrap confidence-interval plot for accuracy, F1, and AUC.
5. McNemar Holm-adjusted p-value heatmap with significant cells outlined.
6. DeLong Holm-adjusted p-value heatmap with significant cells outlined.
7. Generic two-dictionary ablation bar chart.
8. Generic two-dictionary baseline-vs-best bar chart.

The required tables are exported as both `.tex` tabular snippets and `.csv` files:

- `per_model_metrics`: Accuracy, Precision, Recall, F1, AUC, and Faithfulness.
- `mcnemar_pvalues`: pairwise Holm-adjusted McNemar p-values, with `*` on significant cells.
- `delong_pvalues`: pairwise Holm-adjusted DeLong p-values, with `*` on significant cells.

## Style and reproducibility

The preferred plotting backend is Matplotlib with `DejaVu Sans` at a consistent base font size of 10, which reads cleanly in LNCS-style paper drafts. Model colors are assigned once from sorted `model_id` values and reused across all figures so each model remains visually consistent. In minimal environments where Matplotlib is unavailable, the module falls back to a small Pillow renderer so the CLI and tests still produce all required files.

Every exported filename is built through `build_artifact_path(...)`, which embeds the run seed and dataset hash, for example:

```text
model_metric_bars_seed42_hashabc123.pdf
mcnemar_pvalues_seed42_hashabc123.tex
captions_seed42_hashabc123.md
```

## Report CLI

```bash
python report_export.py --eval-results tests/fixtures/report_evaluation_results.json \
  --seed 42 \
  --dataset-hash abc123 \
  --out-dir exports/
```

Optional ROC-points input:

```bash
python report_export.py --eval-results evaluation_results.json \
  --seed 42 \
  --dataset-hash abc123 \
  --out-dir exports/ \
  --roc-points roc_points.json
```

## Part 6 tests

```bash
pytest -q tests/test_report_export.py
```

The tests run a full export against a 4-model mock `evaluation_results.json`, assert that every required figure/table/caption file exists, verify all filenames contain the seed/hash linkage, and confirm the optional ROC-points path does not raise.

# CrossDoc-LegalAI Part 7: Streamlit Comparison App

`app.py` implements the standalone "Master vs. Service Document" comparison UI. It is runnable with offline stubs by default and requires no API keys. The UI is intentionally separated from `pipeline_stubs.py`, which owns the three contract-compatible integration functions that can be swapped for production modules later.

## Run the app

```bash
streamlit run app.py
```

Use the included fictional sample PDFs for an immediate demo:

- `sample_docs/sample_master_msa.pdf`
- `sample_docs/sample_service_sow.pdf`

Upload the Master PDF on the left, upload the Service/SOW PDF on the right, and click **Compare documents**.

## Stubbed integration contracts

The app imports these three functions from `pipeline_stubs.py`:

```python
parse_document(file, doc_type: str) -> dict
retrieve_matches(master_doc: dict, service_doc: dict) -> list[dict]
run_auditor_and_guardrail(
    retrieval_results: list[dict],
    master_full_text: str,
    service_full_text: str,
    pair_id: str = "demo-pair",
    model_id: str = "stub-rule-auditor-v1",
) -> list[dict]
```

`INTEGRATION.md` documents the exact schemas and the one-line import swap needed to replace these stubs with real ingestion, retrieval, and auditor+guardrail modules.

## UI behavior

Results render as mismatch cards sorted by severity. Each card shows a category-style header, severity badge, verification badge, side-by-side Master and Service quotes, explanation, and suggested redline. If a prediction has `guardrail_action="claim_withheld"`, the UI visibly warns: "This claim could not be verified against the source text and has been withheld." The sample SOW intentionally exercises this trust-and-safety state.

Users can export the full report as JSON and a simple PDF summary from the results area.

## Part 7 QA and tests

Manual QA steps are in `MANUAL_QA.md`.

```bash
pytest -q tests/test_part7_pipeline.py
```

The automated smoke tests parse both sample PDFs, run the stub retrieval and auditor/guardrail pipeline, assert at least one passed contradiction and one `claim_withheld` result, and verify JSON/PDF report export data can be produced.
