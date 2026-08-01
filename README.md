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
