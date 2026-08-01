"""Tiny dependency-free PDF helpers for Part 7 demo files and report export."""

from __future__ import annotations

from pathlib import Path


def write_simple_text_pdf(path: str | Path, title: str, lines: list[str]) -> None:
    path = Path(path)
    content_lines = [f"BT /F1 12 Tf 72 740 Td ({_escape(title)}) Tj ET"]
    y = 710
    for line in lines:
        content_lines.append(f"BT /F1 10 Tf 72 {y} Td ({_escape(line[:110])}) Tj ET")
        y -= 18
    stream = "\n".join(content_lines).encode("latin-1", errors="replace")
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Resources << /Font << /F1 4 0 R >> >> /Contents 5 0 R >>",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
        b"<< /Length " + str(len(stream)).encode("ascii") + b" >>\nstream\n" + stream + b"\nendstream",
    ]
    payload = bytearray(b"%PDF-1.4\n")
    offsets = [0]
    for idx, obj in enumerate(objects, start=1):
        offsets.append(len(payload))
        payload.extend(f"{idx} 0 obj\n".encode("ascii") + obj + b"\nendobj\n")
    xref_offset = len(payload)
    payload.extend(f"xref\n0 {len(objects)+1}\n0000000000 65535 f \n".encode("ascii"))
    for offset in offsets[1:]:
        payload.extend(f"{offset:010d} 00000 n \n".encode("ascii"))
    payload.extend(f"trailer << /Size {len(objects)+1} /Root 1 0 R >>\nstartxref\n{xref_offset}\n%%EOF\n".encode("ascii"))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(bytes(payload))


def build_report_pdf(predictions: list[dict], title: str = "CrossDoc-LegalAI Comparison Report") -> bytes:
    lines = []
    for pred in predictions[:24]:
        status = pred.get("guardrail_action", "unknown")
        severity = pred.get("severity", "None")
        summary = pred.get("conflict_explanation", "")[:95]
        lines.append(f"[{severity}] {status}: {summary}")
    temp = Path("/tmp/crossdoc_legalai_report.pdf")
    write_simple_text_pdf(temp, title, lines or ["No report results available."])
    return temp.read_bytes()


def _escape(text: str) -> str:
    return text.replace("\\", r"\\").replace("(", r"\(").replace(")", r"\)")
