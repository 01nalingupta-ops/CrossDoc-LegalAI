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

Two adapters are included:

- `MockAdapter`: deterministic, no-network test adapter used by default by the CLI.
- `OpenAICompatibleAdapter`: generic `/chat/completions` adapter configurable with `OPENAI_COMPATIBLE_API_KEY`, `OPENAI_COMPATIBLE_BASE_URL`, and `OPENAI_COMPATIBLE_MODEL`.

Every real adapter call sends `temperature=0.0`. Adding another provider should only require a small new subclass of `AuditorModelAdapter`; orchestration code should not import provider SDKs or provider-specific settings.

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
