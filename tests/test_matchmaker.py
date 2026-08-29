import json
from pathlib import Path

from matchmaker import (
    DEFAULT_TOP_K,
    RetrievalScoringConfig,
    build_master_index,
    retrieve_for_service_doc,
)

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


class AmbiguousEmbeddingModel:
    def __init__(self, vectors):
        self.vectors = vectors
        self.calls = []

    def encode(self, texts):
        self.calls.append(list(texts))
        return [self.vectors[text] for text in texts]


def parsed_doc(doc_type, chunks):
    return {
        "doc_id": f"{doc_type}-fixture",
        "doc_type": doc_type,
        "source_path": "fixture.txt",
        "full_text": "\n".join(chunk["text"] for chunk in chunks),
        "page_count": 1,
        "extraction_method": "digital",
        "chunks": chunks,
    }


def test_hybrid_retrieval_uses_lexical_signal_when_semantic_scores_are_tied():
    master_chunks = [
        {"chunk_id": "master-a-alpha", "text": "Generic supplier obligations apply to routine services."},
        {"chunk_id": "master-z-target", "text": "Escrow source code deposit required after release milestone."},
    ]
    service_chunks = [
        {"chunk_id": "service-escrow", "text": "The service schedule requires source code escrow deposit."},
    ]
    vectors = {chunk["text"]: [1.0, 0.0] for chunk in master_chunks + service_chunks}
    index = build_master_index(parsed_doc("master", master_chunks), embedder=AmbiguousEmbeddingModel(vectors))

    semantic_only = retrieve_for_service_doc(index, parsed_doc("service", service_chunks), top_k=1)
    hybrid = retrieve_for_service_doc(index, parsed_doc("service", service_chunks), top_k=1, use_hybrid=True)

    assert semantic_only[0]["matched_master_chunks"][0]["master_chunk_id"] == "master-a-alpha"
    assert hybrid[0]["matched_master_chunks"][0]["master_chunk_id"] == "master-z-target"


def test_hybrid_retrieval_keeps_semantic_signal_stronger_than_single_keyword_overlap():
    master_chunks = [
        {"chunk_id": "master-a-lexical-distractor", "text": "Warranty."},
        {"chunk_id": "master-z-semantic-target", "text": "Support obligations continue for ninety days after go-live."},
    ]
    service_chunks = [
        {"chunk_id": "service-support", "text": "Ongoing helpdesk assistance continues after launch warranty period."},
    ]
    vectors = {
        master_chunks[0]["text"]: [0.0, 1.0],
        master_chunks[1]["text"]: [1.0, 0.0],
        service_chunks[0]["text"]: [1.0, 0.0],
    }
    doc = parsed_doc("master", master_chunks)
    lexical_only_index = build_master_index(
        doc,
        embedder=AmbiguousEmbeddingModel(vectors),
        scoring_config=RetrievalScoringConfig(lexical_weight=1.0),
    )
    hybrid_index = build_master_index(doc, embedder=AmbiguousEmbeddingModel(vectors))

    lexical_only = retrieve_for_service_doc(
        lexical_only_index, parsed_doc("service", service_chunks), top_k=1, use_hybrid=True
    )
    hybrid = retrieve_for_service_doc(hybrid_index, parsed_doc("service", service_chunks), top_k=1, use_hybrid=True)

    assert lexical_only[0]["matched_master_chunks"][0]["master_chunk_id"] == "master-a-lexical-distractor"
    assert hybrid[0]["matched_master_chunks"][0]["master_chunk_id"] == "master-z-semantic-target"
