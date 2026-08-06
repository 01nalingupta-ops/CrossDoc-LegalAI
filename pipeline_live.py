"""Live integration layer for CrossDoc-LegalAI.

This module intentionally exports the same public functions consumed by ``app.py`` as
``pipeline_stubs.py`` while delegating to the real Part 1 ingestion, Part 2 retrieval,
and Part 4 auditor+guardrail modules. The default live auditor adapter is the
Part 10 local NLI adapter; tests and offline smoke runs may explicitly select the mock adapter.
"""
from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Any, BinaryIO

import auditor_guardrail
import ingestion
import matchmaker
from model_adapter import AuditorModelAdapter, LocalNLIAdapter, MockAdapter
from pipeline_stubs import infer_category, sort_predictions_for_display


def parse_document(file: str | Path | bytes | BinaryIO, doc_type: str) -> dict[str, Any]:
    """Parse an uploaded/path document using the real ingestion module."""
    if isinstance(file, (str, Path)):
        return ingestion.parse_document(str(file), doc_type)

    data, suffix = _read_upload(file)
    with tempfile.NamedTemporaryFile(prefix=f"crossdoc-{doc_type}-", suffix=suffix, delete=False) as handle:
        handle.write(data)
        temp_path = Path(handle.name)
    try:
        return ingestion.parse_document(str(temp_path), doc_type)
    finally:
        temp_path.unlink(missing_ok=True)


def retrieve_matches(master_doc: dict[str, Any], service_doc: dict[str, Any]) -> list[dict[str, Any]]:
    """Retrieve top-2 matched master chunks using the real matchmaker module."""
    embedder = _configured_embedder()
    return matchmaker.retrieve_pair(master_doc, service_doc, embedder=embedder)


def run_auditor_and_guardrail(
    retrieval_results: list[dict[str, Any]],
    master_full_text: str,
    service_full_text: str,
    pair_id: str = "demo-pair",
    model_id: str | None = None,
) -> list[dict[str, Any]]:
    """Run Part 4 auditor+guardrail over every retrieval result.

    ``model_id`` is accepted for signature compatibility with ``pipeline_stubs`` but the
    returned model identifier is owned by the active adapter.
    """
    adapter = get_auditor_adapter()
    return [
        auditor_guardrail.run_auditor_and_guardrail(result, master_full_text, service_full_text, adapter, pair_id)
        for result in retrieval_results
    ]


def get_auditor_adapter() -> AuditorModelAdapter:
    """Return the live pipeline's current auditor adapter for orchestrator bake-offs."""
    adapter_name = os.environ.get("CROSSDOC_AUDITOR_ADAPTER")
    if adapter_name is None and os.environ.get("CROSSDOC_EMBEDDING_BACKEND") == "keyword-fixture":
        adapter_name = "mock"
    adapter_name = (adapter_name or "local-nli").strip().lower()
    if adapter_name == "local-nli":
        return LocalNLIAdapter()
    if adapter_name == "mock":
        return MockAdapter()
    raise ValueError("CROSSDOC_AUDITOR_ADAPTER must be 'local-nli' or 'mock'")


def _configured_embedder() -> matchmaker.EmbeddingModel | None:
    backend = os.environ.get("CROSSDOC_EMBEDDING_BACKEND", "sentence-transformers")
    if backend == "sentence-transformers":
        return None
    if backend == "keyword-fixture":
        return matchmaker.KeywordFixtureEmbedder()
    raise ValueError("CROSSDOC_EMBEDDING_BACKEND must be 'sentence-transformers' or 'keyword-fixture'")


def _read_upload(file: bytes | BinaryIO) -> tuple[bytes, str]:
    if isinstance(file, bytes):
        return file, ".pdf"
    name = getattr(file, "name", "uploaded.pdf")
    suffix = Path(name).suffix or ".pdf"
    if hasattr(file, "getvalue"):
        data = file.getvalue()
    else:
        data = file.read()
    return data, suffix
