from pathlib import Path

import pytest

from ingestion import CHUNK_SIZE, OVERLAP_CHARS, chunk_text, parse_document


def _fitz():
    return pytest.importorskip("fitz")


def make_pdf(path: Path, pages: list[str]) -> None:
    fitz = _fitz()
    doc = fitz.open()
    for text in pages:
        page = doc.new_page()
        rect = fitz.Rect(72, 72, page.rect.width - 72, page.rect.height - 72)
        page.insert_textbox(rect, text, fontsize=11)
    doc.save(path)
    doc.close()


def make_image_only_pdf(path: Path) -> None:
    fitz = _fitz()
    doc = fitz.open()
    page = doc.new_page()
    # Draw a rectangle so the PDF has page content but no extractable text layer.
    page.draw_rect(fitz.Rect(72, 72, 250, 150), color=(0, 0, 0), fill=(0.9, 0.9, 0.9))
    doc.save(path)
    doc.close()


def test_short_digital_text_pdf(tmp_path):
    pdf = tmp_path / "short.pdf"
    make_pdf(pdf, ["Short master agreement text."])

    parsed = parse_document(str(pdf), "master")

    assert parsed["doc_type"] == "master"
    assert parsed["page_count"] == 1
    assert parsed["extraction_method"] == "digital"
    assert "Short master agreement text." in parsed["full_text"]
    assert len(parsed["chunks"]) == 1
    assert parsed["chunks"][0]["char_start"] == 0
    assert parsed["chunks"][0]["char_end"] == len(parsed["full_text"])
    assert parsed["chunks"][0]["page_num"] == 1


def test_multi_page_digital_pdf_chunk_page_numbers(tmp_path):
    pdf = tmp_path / "multi.pdf"
    page1 = "A" * 1000
    page2 = "B" * 1000
    make_pdf(pdf, [page1, page2])

    parsed = parse_document(str(pdf), "service")

    assert parsed["page_count"] == 2
    assert parsed["extraction_method"] == "digital"
    assert len(parsed["chunks"]) >= 2
    assert parsed["chunks"][0]["char_start"] == 0
    assert parsed["chunks"][0]["char_end"] == CHUNK_SIZE
    assert parsed["chunks"][1]["char_start"] == CHUNK_SIZE - OVERLAP_CHARS
    assert parsed["chunks"][1]["page_num"] == 1
    assert any(chunk["page_num"] == 2 for chunk in parsed["chunks"])


def test_ocr_fallback_pdf(monkeypatch, tmp_path):
    pdf = tmp_path / "scanned.pdf"
    make_image_only_pdf(pdf)

    monkeypatch.setattr("ingestion._ocr_pdf_page", lambda page: "OCR fallback text on scanned page.")
    parsed = parse_document(str(pdf), "master")

    assert parsed["extraction_method"] == "ocr"
    assert parsed["full_text"] == "OCR fallback text on scanned page."
    assert parsed["chunks"][0]["page_num"] == 1


def test_text_shorter_than_one_chunk_produces_one_chunk():
    chunks = chunk_text("tiny", [0], "doc")

    assert chunks == [
        {
            "chunk_id": "doc-chunk-0001",
            "text": "tiny",
            "char_start": 0,
            "char_end": 4,
            "page_num": 1,
            "overlap_chars": 150,
        }
    ]


def test_numbered_clauses_align_to_clause_boundaries():
    text = "1. Payment Terms\nFees are due within 30 days.\n2. Termination\nEither party may terminate for breach."

    chunks = chunk_text(text, [0], "doc")

    assert [chunk["clause_heading"] for chunk in chunks] == ["1. Payment Terms", "2. Termination"]
    assert chunks[0]["text"] == "1. Payment Terms\nFees are due within 30 days.\n"
    assert chunks[0]["char_start"] == 0
    assert chunks[0]["char_end"] == text.index("2. Termination")
    assert chunks[1]["text"] == "2. Termination\nEither party may terminate for breach."
    assert chunks[1]["char_start"] == text.index("2. Termination")
    assert chunks[1]["page_num"] == 1


def test_all_caps_headings_create_clause_chunks():
    text = "RECITALS\nThe parties agree to the background facts.\nCONFIDENTIALITY\nRecipient protects information."

    chunks = chunk_text(text, [0], "doc")

    assert len(chunks) == 2
    assert chunks[0]["clause_heading"] == "RECITALS"
    assert chunks[0]["text"].startswith("RECITALS\n")
    assert chunks[1]["clause_heading"] == "CONFIDENTIALITY"
    assert chunks[1]["char_start"] == text.index("CONFIDENTIALITY")


def test_bracket_tag_clause_headings_prevent_mid_word_truncation():
    text = (
        "Fictional Alpha Master Services Agreement\n"
        "[payment_terms] Payment Terms: Client will pay undisputed invoices within thirty (30) days after receipt.\n"
        "[confidentiality_scope] Confidentiality Scope: Confidentiality obligations are mutual and survive for three (3) years after disclosure.\n"
        "[indemnification] Indemnification: Each party will indemnify the other for third-party claims caused by its negligence, misconduct, or IP infringement.\n"
    )

    chunks = chunk_text(text, [0], "doc")

    confidentiality_chunk = next(
        c for c in chunks if c["clause_heading"] and c["clause_heading"].startswith("[confidentiality_scope]")
    )
    assert confidentiality_chunk["text"].startswith("[confidentiality_scope]")
    assert not confidentiality_chunk["text"].startswith("entiality_scope]")


def test_unstructured_text_fallback_output_is_unchanged():
    text = "plain unstructured text " * 80

    chunks = chunk_text(text, [0], "doc")

    assert "clause_heading" not in chunks[0]
    assert chunks[0] == {
        "chunk_id": "doc-chunk-0001",
        "text": text[:CHUNK_SIZE],
        "char_start": 0,
        "char_end": CHUNK_SIZE,
        "page_num": 1,
        "overlap_chars": OVERLAP_CHARS,
    }
    assert chunks[1]["char_start"] == CHUNK_SIZE - OVERLAP_CHARS
    assert "clause_heading" not in chunks[1]
