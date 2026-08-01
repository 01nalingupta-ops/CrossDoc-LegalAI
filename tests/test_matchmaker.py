import json
from pathlib import Path

from matchmaker import DEFAULT_TOP_K, build_master_index, retrieve_for_service_doc

FIXTURE_DIR = Path(__file__).parent / "fixtures"
VOCABULARY = [
    {"payment", "terms", "invoices", "invoice", "paid", "payable", "fees", "due", "thirty", "30"},
    {"confidential", "confidentiality", "trade", "secrets", "nonpublic", "data", "disclosure", "protected"},
    {"termination", "terminate", "end", "breach", "material", "uncured", "cure"},
]


class KeywordEmbeddingModel:
    def __init__(self):
        self.calls = []

    def encode(self, texts):
        self.calls.append(list(texts))
        vectors = []
        for text in texts:
            tokens = {token.strip(".,;:()[]").lower() for token in text.split()}
            vectors.append([len(tokens & bucket) for bucket in VOCABULARY])
        return vectors


def load_fixture(name):
    return json.loads((FIXTURE_DIR / name).read_text(encoding="utf-8"))


def test_payment_and_confidentiality_service_chunks_return_sensible_top_matches():
    embedder = KeywordEmbeddingModel()
    index = build_master_index(load_fixture("master_parsed.json"), embedder=embedder)

    results = retrieve_for_service_doc(index, load_fixture("service_payment_confidentiality.json"))

    assert len(results) == 2
    assert results[0]["service_chunk_id"] == "service-a-payment"
    assert results[0]["matched_master_chunks"][0]["master_chunk_id"] == "master-chunk-payment"
    assert len(results[0]["matched_master_chunks"]) == DEFAULT_TOP_K
    assert results[1]["service_chunk_id"] == "service-a-confidentiality"
    assert results[1]["matched_master_chunks"][0]["master_chunk_id"] == "master-chunk-confidentiality"


def test_termination_service_chunk_returns_termination_master_chunk_first():
    embedder = KeywordEmbeddingModel()
    index = build_master_index(load_fixture("master_parsed.json"), embedder=embedder)

    results = retrieve_for_service_doc(index, load_fixture("service_termination.json"))

    assert results[0]["service_chunk_id"] == "service-b-termination"
    assert results[0]["matched_master_chunks"][0]["master_chunk_id"] == "master-chunk-termination"
    assert results[1]["matched_master_chunks"][0]["master_chunk_id"] == "master-chunk-payment"


def test_master_embeddings_are_cached_and_service_chunks_are_batched():
    embedder = KeywordEmbeddingModel()
    master_doc = load_fixture("master_parsed.json")
    service_doc = load_fixture("service_payment_confidentiality.json")

    index = build_master_index(master_doc, embedder=embedder)
    retrieve_for_service_doc(index, service_doc)

    assert len(embedder.calls) == 2
    assert embedder.calls[0] == [chunk["text"] for chunk in master_doc["chunks"]]
    assert embedder.calls[1] == [chunk["text"] for chunk in service_doc["chunks"]]
