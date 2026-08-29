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
          "overlap_chars": 150,
          "clause_heading": "optional nullable heading text when clause boundaries are detected"
        }
      ]
    }

PDFs are read with PyMuPDF when available. Digital text is used when any page has an
extractable text layer; otherwise each page is rendered and passed to pytesseract OCR.
TXT and DOCX inputs are supported as convenience formats.

Changelog:
    Part 11 adds clause/heading-aware chunking with unchanged fixed-window fallback.
"""
from __future__ import annotations

import argparse
import bisect
import hashlib
import json
import re
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
    """Split text into clause-aware chunks, falling back to fixed windows when needed."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if overlap_chars < 0 or overlap_chars >= chunk_size:
        raise ValueError("overlap_chars must be non-negative and smaller than chunk_size")

    starts = list(page_starts) or [0]
    clause_ranges = _clause_ranges(full_text)
    if not clause_ranges:
        return _fixed_window_chunks(full_text, starts, doc_id, chunk_size, overlap_chars)

    chunks: list[dict] = []
    for clause_start, clause_end, heading in clause_ranges:
        for start, end in _fixed_window_ranges(full_text, clause_start, clause_end, chunk_size, overlap_chars):
            chunks.append(
                {
                    "chunk_id": f"{doc_id}-chunk-{len(chunks) + 1:04d}",
                    "text": full_text[start:end],
                    "char_start": start,
                    "char_end": end,
                    "page_num": _page_for_offset(start, starts),
                    "overlap_chars": overlap_chars,
                    "clause_heading": heading,
                }
            )
    return chunks


def _fixed_window_chunks(
    full_text: str,
    page_starts: Sequence[int],
    doc_id: str,
    chunk_size: int,
    overlap_chars: int,
) -> list[dict]:
    """Split text into fixed windows and map each chunk start offset to a page number."""
    chunks: list[dict] = []

    ranges = _fixed_window_ranges(full_text, 0, len(full_text), chunk_size, overlap_chars)
    for index, (start, end) in enumerate(ranges, start=1):
        chunks.append(
            {
                "chunk_id": f"{doc_id}-chunk-{index:04d}",
                "text": full_text[start:end],
                "char_start": start,
                "char_end": end,
                "page_num": _page_for_offset(start, page_starts),
                "overlap_chars": overlap_chars,
            }
        )
    return chunks


def _fixed_window_ranges(
    full_text: str,
    range_start: int,
    range_end: int,
    chunk_size: int,
    overlap_chars: int,
) -> list[tuple[int, int]]:
    """Return fixed-window ranges within ``range_start`` and ``range_end``."""
    text_length = range_end - range_start
    if text_length <= chunk_size:
        return [(range_start, range_end)]

    step = chunk_size - overlap_chars
    ranges: list[tuple[int, int]] = []
    start = range_start
    while start < range_end:
        end = min(start + chunk_size, range_end)
        ranges.append((start, end))
        if end == range_end:
            break
        start += step
    return ranges


_NUMBERED_HEADING_RE = re.compile(r"^\s*(?:\d+(?:\.\d+)*[.)]?|[A-Z][.)]|\([a-zA-Z0-9]+\))\s+\S+")
_ALL_CAPS_HEADING_RE = re.compile(r"^\s*[A-Z][A-Z0-9 &,;:'\"()/-]{2,}\s*$")
_BRACKET_TAG_HEADING_RE = re.compile(r"^\[[\w-]+\]\s+\S")


_REPEATED_CHAR_RE = re.compile(r"^(.)\1*$")


def _clause_ranges(full_text: str) -> list[tuple[int, int, str | None]]:
    """Detect clause/section ranges from common legal-document heading patterns."""
    headings: list[tuple[int, str]] = []
    offset = 0
    for line in full_text.splitlines(keepends=True):
        stripped = line.strip()
        if stripped and (
            _NUMBERED_HEADING_RE.match(line)
            or (_ALL_CAPS_HEADING_RE.match(line) and not _REPEATED_CHAR_RE.match(stripped))
            or _BRACKET_TAG_HEADING_RE.match(line)
        ):
            headings.append((offset, stripped))
        offset += len(line)

    if not headings:
        return []

    ranges: list[tuple[int, int, str | None]] = []
    first_heading_start = headings[0][0]
    if full_text[:first_heading_start].strip():
        ranges.append((0, first_heading_start, None))

    for index, (start, heading) in enumerate(headings):
        end = headings[index + 1][0] if index + 1 < len(headings) else len(full_text)
        ranges.append((start, end, heading))
    return ranges


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
