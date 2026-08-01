"""CrossDoc-LegalAI Part 2: Matchmaker Agent semantic retrieval.

Input: two ParsedDocument JSON objects, one ``master`` and one ``service``::

    {
      "doc_id": "string",
      "doc_type": "master | service",
      "source_path": "string",
      "full_text": "string",
      "page_count": 1,
      "extraction_method": "digital | ocr",
      "chunks": [
        {"chunk_id": "string", "text": "string", "char_start": 0,
         "char_end": 800, "page_num": 1, "overlap_chars": 150}
      ]
    }

Output: a RetrievalResult JSON array, one object per service chunk::

    [
      {
        "service_chunk_id": "string",
        "service_chunk_text": "string",
        "matched_master_chunks": [
          {"master_chunk_id": "string", "master_chunk_text": "string",
           "similarity_score": 0.0}
        ]
      }
    ]

The default embedder is sentence-transformers model ``all-MiniLM-L6-v2``. The service
document is never indexed; its chunks are embedded as queries against a master-only index.
"""
from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, Sequence

EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
DEFAULT_TOP_K = 2


class EmbeddingModel(Protocol):
    """Minimal swappable embedding interface used by the in-memory vector index."""

    def encode(self, texts: Sequence[str]) -> Sequence[Sequence[float]]:
        """Return one embedding vector per input text."""


class SentenceTransformerEmbedder:
    """Lazy wrapper around the fixed sentence-transformers model required by this module."""

    def __init__(self, model_name: str = EMBEDDING_MODEL_NAME) -> None:
        if model_name != EMBEDDING_MODEL_NAME:
            raise ValueError(f"model_name must be {EMBEDDING_MODEL_NAME!r}")
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise RuntimeError(
                "The default Matchmaker embedder requires sentence-transformers. "
                "Install project dependencies, then rerun the CLI."
            ) from exc
        self._model = SentenceTransformer(model_name)

    def encode(self, texts: Sequence[str]) -> Sequence[Sequence[float]]:
        embeddings = self._model.encode(list(texts), normalize_embeddings=True)
        return embeddings.tolist() if hasattr(embeddings, "tolist") else embeddings


class KeywordFixtureEmbedder:
    """Small deterministic offline embedder for fixture/manual tests only.

    Production usage should rely on ``SentenceTransformerEmbedder`` with
    ``all-MiniLM-L6-v2``.
    """

    _buckets = [
        {"payment", "terms", "invoices", "invoice", "paid", "payable", "fees", "due", "thirty", "30"},
        {"confidential", "confidentiality", "trade", "secrets", "nonpublic", "data", "disclosure", "protected"},
        {"termination", "terminate", "end", "breach", "material", "uncured", "cure"},
    ]

    def encode(self, texts: Sequence[str]) -> Sequence[Sequence[float]]:
        vectors = []
        for text in texts:
            tokens = {token.strip(".,;:()[]").lower() for token in text.split()}
            vectors.append([float(len(tokens & bucket)) for bucket in self._buckets])
        return vectors


@dataclass(frozen=True)
class Match:
    chunk: dict
    similarity: float


class InMemoryCosineIndex:
    """Swappable master-chunk-only cosine index with cached master embeddings."""

    def __init__(self, chunks: Sequence[dict], embeddings: Sequence[Sequence[float]], embedder: EmbeddingModel) -> None:
        if len(chunks) != len(embeddings):
            raise ValueError("chunks and embeddings must have identical lengths")
        self.chunks = list(chunks)
        self.embeddings = [_normalize(vector) for vector in embeddings]
        self.embedder = embedder

    def search(self, query_texts: Sequence[str], top_k: int = DEFAULT_TOP_K) -> list[list[Match]]:
        if top_k <= 0:
            raise ValueError("top_k must be positive")
        query_embeddings = [_normalize(vector) for vector in self.embedder.encode(list(query_texts))]
        results: list[list[Match]] = []
        for query in query_embeddings:
            scored = [Match(chunk, _cosine(query, master_vector)) for chunk, master_vector in zip(self.chunks, self.embeddings)]
            scored.sort(key=lambda item: (-item.similarity, item.chunk.get("chunk_id", "")))
            results.append(scored[: min(top_k, len(scored))])
        return results


def build_master_index(master_doc: dict, embedder: EmbeddingModel | None = None) -> InMemoryCosineIndex:
    """Embed and index only the Master ParsedDocument chunks once."""
    _validate_parsed_document(master_doc, expected_doc_type="master")
    model = embedder or SentenceTransformerEmbedder()
    chunks = list(master_doc["chunks"])
    texts = [chunk["text"] for chunk in chunks]
    embeddings = model.encode(texts)
    return InMemoryCosineIndex(chunks=chunks, embeddings=embeddings, embedder=model)


def retrieve_for_service_doc(index: InMemoryCosineIndex, service_doc: dict, top_k: int = DEFAULT_TOP_K) -> list[dict]:
    """Return top-k matched master chunks for every Service ParsedDocument chunk."""
    _validate_parsed_document(service_doc, expected_doc_type="service")
    if top_k != DEFAULT_TOP_K:
        # The project contract fixes k=2. Tests may pass this explicitly, but the public
        # default and orchestrated behavior should remain top-2.
        if top_k <= 0:
            raise ValueError("top_k must be positive")
    service_chunks = list(service_doc["chunks"])
    service_texts = [chunk["text"] for chunk in service_chunks]
    matches_by_query = index.search(service_texts, top_k=top_k)
    output: list[dict] = []
    for service_chunk, matches in zip(service_chunks, matches_by_query):
        output.append(
            {
                "service_chunk_id": service_chunk["chunk_id"],
                "service_chunk_text": service_chunk["text"],
                "matched_master_chunks": [
                    {
                        "master_chunk_id": match.chunk["chunk_id"],
                        "master_chunk_text": match.chunk["text"],
                        "similarity_score": _clamp_similarity(match.similarity),
                    }
                    for match in matches
                ],
            }
        )
    return output


def retrieve_pair(master_doc: dict, service_doc: dict, embedder: EmbeddingModel | None = None) -> list[dict]:
    """Convenience helper for one Master/Service ParsedDocument pair."""
    return retrieve_for_service_doc(build_master_index(master_doc, embedder=embedder), service_doc, top_k=DEFAULT_TOP_K)


def _validate_parsed_document(document: dict, expected_doc_type: str) -> None:
    if not isinstance(document, dict):
        raise ValueError("ParsedDocument must be a JSON object")
    if document.get("doc_type") != expected_doc_type:
        raise ValueError(f"expected doc_type {expected_doc_type!r}")
    chunks = document.get("chunks")
    if not isinstance(chunks, list) or not chunks:
        raise ValueError("ParsedDocument.chunks must be a non-empty list")
    for index, chunk in enumerate(chunks):
        if not isinstance(chunk, dict):
            raise ValueError(f"chunk {index} must be an object")
        if not isinstance(chunk.get("chunk_id"), str) or not chunk["chunk_id"]:
            raise ValueError(f"chunk {index}.chunk_id must be a non-empty string")
        if not isinstance(chunk.get("text"), str):
            raise ValueError(f"chunk {index}.text must be a string")


def _normalize(vector: Sequence[float]) -> list[float]:
    values = [float(value) for value in vector]
    norm = math.sqrt(sum(value * value for value in values))
    if norm == 0.0:
        return [0.0 for _ in values]
    return [value / norm for value in values]


def _cosine(left: Sequence[float], right: Sequence[float]) -> float:
    return sum(a * b for a, b in zip(left, right))


def _clamp_similarity(score: float) -> float:
    return max(0.0, min(1.0, float(score)))


def _load_json(path: str | Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Retrieve top-2 Master chunks for each Service chunk.")
    parser.add_argument("master_json", help="Path to Master ParsedDocument JSON")
    parser.add_argument("service_json", help="Path to Service ParsedDocument JSON")
    parser.add_argument(
        "--embedding-backend",
        choices=["sentence-transformers", "keyword-fixture"],
        default="sentence-transformers",
        help="Use the required all-MiniLM-L6-v2 backend, or a deterministic fixture backend for offline smoke tests.",
    )
    args = parser.parse_args(argv)

    embedder = KeywordFixtureEmbedder() if args.embedding_backend == "keyword-fixture" else None
    result = retrieve_pair(_load_json(args.master_json), _load_json(args.service_json), embedder=embedder)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
