# CrossDoc-LegalAI Part 7 Integration Seams

The Streamlit UI in `app.py` depends on exactly three pipeline functions. Today they are imported from `pipeline_stubs.py`:

```python
from pipeline_stubs import parse_document, retrieve_matches, run_auditor_and_guardrail
```

To swap in production modules later, create a module that exports the same three names with the same signatures and schemas, then change that one import line in `app.py`.

## 1. Ingestion stub

```python
parse_document(file, doc_type: str) -> dict
```

- `file`: Streamlit `UploadedFile`, path, bytes, or file-like object.
- `doc_type`: exactly `"master"` or `"service"`.
- Returns a `ParsedDocument` dict:

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

The current stub extracts literal PDF text from simple demo PDFs and chunks at approximately 800 characters with 150-character overlap.

## 2. Retrieval stub

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

The current stub uses lexical cosine similarity and returns the top two master chunks.

## 3. Auditor + guardrail stub

```python
run_auditor_and_guardrail(
    retrieval_results: list[dict],
    master_full_text: str,
    service_full_text: str,
    pair_id: str = "demo-pair",
    model_id: str = "stub-rule-auditor-v1",
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

The current stub is deterministic and offline. It flags simple Net 60 vs Net 30 and uncapped-liability mismatches, then performs a real substring guardrail check. The sample SOW includes an intentionally unverified trigger so the UI visibly renders the required `claim_withheld` safety state.
