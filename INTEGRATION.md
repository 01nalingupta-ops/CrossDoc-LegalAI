# CrossDoc-LegalAI Integration Seams

The Streamlit UI in `app.py` now selects a pipeline module with `CROSSDOC_PIPELINE`:

- `CROSSDOC_PIPELINE=live` (default): uses `pipeline_live.py`, which calls the real Part 1 ingestion, Part 2 retrieval, and Part 4 auditor+guardrail modules. The live auditor adapter defaults to `LocalNLIAdapter` for the `candidate_2` research path; set `CROSSDOC_AUDITOR_ADAPTER=mock` only for dependency-light smoke tests.
- `CROSSDOC_PIPELINE=stub`: uses `pipeline_stubs.py`, preserving the standalone/offline Part 7 demo and grading path.

For low-dependency smoke tests of the live UI path, set `CROSSDOC_EMBEDDING_BACKEND=keyword-fixture` and, when model weights are unavailable, `CROSSDOC_AUDITOR_ADAPTER=mock`. For low-dependency benchmark smoke tests, set `CROSSDOC_BENCHMARK_EMBEDDING_BACKEND=keyword-fixture`. Both retrieval defaults remain `sentence-transformers` with `all-MiniLM-L6-v2`; the auditor default remains local NLI and requires the configured HuggingFace model weights.

Both pipeline modules export the same UI contract: `parse_document`, `retrieve_matches`, `run_auditor_and_guardrail`, `sort_predictions_for_display`, and `infer_category`.

## 1. Ingestion contract

```python
parse_document(file, doc_type: str) -> dict
```

- `file`: Streamlit `UploadedFile`, path, bytes, or file-like object.
- `doc_type`: exactly `"master"` or `"service"`.
- Live mode delegates to `ingestion.parse_document`; uploaded bytes are written to a temporary file only long enough for the ingestion call.
- Stub mode remains dependency-free and extracts literal text from simple demo PDFs.

Returns a `ParsedDocument` dict:

```json
{
  "doc_id": "string",
  "doc_type": "master | service",
  "source_path": "string",
  "full_text": "string",
  "page_count": "integer",
  "extraction_method": "digital | ocr",
  "chunks": [
    {"chunk_id": "string", "text": "string", "char_start": "int", "char_end": "int", "page_num": "int", "overlap_chars": 150}
  ]
}
```

## 2. Retrieval contract

```python
retrieve_matches(master_doc: dict, service_doc: dict) -> list[dict]
```

Returns one `RetrievalResult` per service chunk:

```json
{
  "service_chunk_id": "string",
  "service_chunk_text": "string",
  "matched_master_chunks": [
    {"master_chunk_id": "string", "master_chunk_text": "string", "similarity_score": "float"}
  ]
}
```

Live mode delegates to `matchmaker.retrieve_pair` and returns top-2 semantic matches from a master-only in-memory index. Stub mode uses lexical cosine similarity and also returns the top two master chunks.

## 3. Auditor + guardrail contract

```python
run_auditor_and_guardrail(
    retrieval_results: list[dict],
    master_full_text: str,
    service_full_text: str,
    pair_id: str = "demo-pair",
    model_id: str | None = None,
) -> list[dict]
```

Returns `VerifiedPrediction` objects:

```json
{
  "pair_id": "string",
  "model_id": "string",
  "service_chunk_id": "string",
  "has_contradiction": "boolean",
  "confidence": "float",
  "severity": "High|Medium|Low|None",
  "conflict_explanation": "string",
  "msa_exact_quote": "string",
  "sow_exact_quote": "string",
  "suggested_redline": "string",
  "guardrail_verified": "boolean",
  "guardrail_action": "passed | quote_rejected | claim_withheld"
}
```

Live mode calls `auditor_guardrail.run_auditor_and_guardrail` for each retrieval result and every adapter output flows through `apply_guardrail(...)` unchanged. `LocalNLIAdapter` is configured in `auditor_config.json` with model name/revision, threshold, severity bands, and label aliases. Stub mode keeps its standalone deterministic auditor and substring guardrail so the offline UI still demonstrates both verified findings and `claim_withheld` safety behavior.
