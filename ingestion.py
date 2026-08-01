"""Standalone document ingestion for CrossDoc-LegalAI.

Run manually:
    python ingestion.py <file> <master|service>

The command prints one JSON object with this schema:
    {
      "doc_id": "unique stable id derived from path and content hash",
      "doc_type": "master | service",
      "source_path": "original file path",
      "full_text": "complete extracted text",
      "page_count": integer,
      "extraction_method": "digital | ocr",
      "chunks": [
        {
          "chunk_id": "<doc_id>-chunk-0001",
          "text": "up to 800 characters",
          "char_start": integer,
          "char_end": integer,
          "page_num": integer,
          "overlap_chars": 150
        }
      ]
    }

PDFs are read with PyMuPDF when available. Digital text is used when any page has an
extractable text layer; otherwise each page is rendered and passed to pytesseract OCR.
TXT and DOCX inputs are supported as convenience formats.
"""
from __future__ import annotations

import argparse
import bisect
import hashlib
import json
from pathlib import Path
from typing import Callable, Iterable, Literal, Sequence

DocType = Literal["master", "service"]
ExtractionMethod = Literal["digital", "ocr"]

CHUNK_SIZE = 800
OVERLAP_CHARS = 150


class IngestionError(RuntimeError):
    """Raised when a document cannot be parsed."""


def parse_document(file_path: str, doc_type: str) -> dict:
    """Parse a PDF, DOCX, or TXT file into the ParsedDocument JSON-compatible dict."""
    if doc_type not in {"master", "service"}:
        raise ValueError("doc_type must be 'master' or 'service'")

    path = Path(file_path)
    if not path.is_file():
        raise FileNotFoundError(file_path)

    full_text, page_starts, page_count, method = _extract_text(path)
    doc_id = _make_doc_id(path)
    chunks = chunk_text(full_text, page_starts, doc_id)

    return {
        "doc_id": doc_id,
        "doc_type": doc_type,
        "source_path": str(file_path),
        "full_text": full_text,
        "page_count": page_count,
        "extraction_method": method,
        "chunks": chunks,
    }


def chunk_text(
    full_text: str,
    page_starts: Sequence[int],
    doc_id: str,
    chunk_size: int = CHUNK_SIZE,
    overlap_chars: int = OVERLAP_CHARS,
) -> list[dict]:
    """Split text into fixed windows and map each chunk start offset to a page number."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if overlap_chars < 0 or overlap_chars >= chunk_size:
        raise ValueError("overlap_chars must be non-negative and smaller than chunk_size")

    starts = list(page_starts) or [0]
    step = chunk_size - overlap_chars
    chunks: list[dict] = []

    if len(full_text) <= chunk_size:
        ranges = [(0, len(full_text))]
    else:
        ranges = []
        start = 0
        while start < len(full_text):
            end = min(start + chunk_size, len(full_text))
            ranges.append((start, end))
            if end == len(full_text):
                break
            start += step

    for index, (start, end) in enumerate(ranges, start=1):
        chunks.append(
            {
                "chunk_id": f"{doc_id}-chunk-{index:04d}",
                "text": full_text[start:end],
                "char_start": start,
                "char_end": end,
                "page_num": _page_for_offset(start, starts),
                "overlap_chars": overlap_chars,
            }
        )
    return chunks


def _extract_text(path: Path) -> tuple[str, list[int], int, ExtractionMethod]:
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return _extract_pdf(path)
    if suffix == ".txt":
        text = path.read_text(encoding="utf-8")
        return text, [0], 1, "digital"
    if suffix == ".docx":
        return _extract_docx(path)
    raise ValueError("unsupported file type; expected .pdf, .docx, or .txt")


def _extract_pdf(path: Path) -> tuple[str, list[int], int, ExtractionMethod]:
    try:
        import fitz  # PyMuPDF
    except ImportError as exc:
        raise IngestionError("PDF parsing requires PyMuPDF. Install with: pip install pymupdf") from exc

    with fitz.open(path) as doc:
        page_texts = [page.get_text("text") or "" for page in doc]
        if any(text.strip() for text in page_texts):
            full_text, starts = _join_pages(page_texts)
            return full_text, starts, doc.page_count, "digital"

        ocr_texts = [_ocr_pdf_page(page) for page in doc]
        full_text, starts = _join_pages(ocr_texts)
        return full_text, starts, doc.page_count, "ocr"


def _ocr_pdf_page(page) -> str:
    try:
        import pytesseract
        from PIL import Image
    except ImportError as exc:
        raise IngestionError(
            "OCR fallback requires pytesseract and Pillow. Install with: pip install pytesseract pillow"
        ) from exc

    pix = page.get_pixmap(dpi=300, alpha=False)
    image = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
    return pytesseract.image_to_string(image) or ""


def _extract_docx(path: Path) -> tuple[str, list[int], int, ExtractionMethod]:
    try:
        from docx import Document
    except ImportError as exc:
        raise IngestionError("DOCX parsing requires python-docx. Install with: pip install python-docx") from exc
    document = Document(path)
    text = "\n".join(p.text for p in document.paragraphs)
    return text, [0], 1, "digital"


def _join_pages(page_texts: Iterable[str]) -> tuple[str, list[int]]:
    parts: list[str] = []
    starts: list[int] = []
    offset = 0
    for page_text in page_texts:
        if parts:
            parts.append("\n")
            offset += 1
        starts.append(offset)
        parts.append(page_text)
        offset += len(page_text)
    return "".join(parts), starts or [0]


def _page_for_offset(offset: int, page_starts: Sequence[int]) -> int:
    return bisect.bisect_right(page_starts, offset)


def _make_doc_id(path: Path) -> str:
    digest = hashlib.sha256(path.read_bytes()).hexdigest()[:12]
    safe_stem = "".join(ch if ch.isalnum() or ch in "-_" else "-" for ch in path.stem).strip("-") or "document"
    return f"{safe_stem}-{digest}"


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Parse a legal document into ParsedDocument JSON.")
    parser.add_argument("file", help="Path to a PDF, DOCX, or TXT file")
    parser.add_argument("doc_type", choices=["master", "service"], help="Document type")
    args = parser.parse_args(argv)
    print(json.dumps(parse_document(args.file, args.doc_type), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
