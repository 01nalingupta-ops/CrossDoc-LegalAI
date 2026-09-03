"""Deterministic clause classification for the eight benchmark categories plus Other."""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from generate_data import DEFECT_CATEGORIES

OTHER_CATEGORY = "Other"
CATEGORY_LABELS = [*DEFECT_CATEGORIES, OTHER_CATEGORY]
MIN_CLASSIFICATION_SCORE = 1.0

CATEGORY_PROFILES: dict[str, tuple[str, ...]] = {
    "Payment Terms": ("payment", "pay", "paid", "payable", "invoice", "invoices", "fees", "receipt", "net", "undisputed"),
    "Liability Cap": ("liability", "liable", "cap", "capped", "uncapped", "unlimited", "aggregate", "damages"),
    "IP Ownership": ("ip", "intellectual", "ownership", "own", "owns", "deliverables", "license", "work", "product", "sublicense"),
    "Termination Notice": ("termination", "terminate", "notice", "convenience", "cure", "breach", "written"),
    "Governing Jurisdiction": ("governing", "jurisdiction", "venue", "law", "courts", "county", "california", "york", "texas"),
    "Confidentiality Scope": ("confidential", "confidentiality", "disclosure", "nonpublic", "marked", "secrets", "survive", "obligations"),
    "Indemnification": ("indemnify", "indemnification", "indemnity", "claims", "third", "party", "defend", "defense"),
    "Insurance Requirements": ("insurance", "coverage", "commercial", "general", "liability", "certificates", "self", "maintain"),
}


@dataclass(frozen=True)
class ClauseClassification:
    category: str
    confidence: float
    scores: dict[str, float]

    def to_dict(self) -> dict[str, Any]:
        return {"category": self.category, "confidence": self.confidence, "scores": dict(self.scores)}


def classify_clause(text: str) -> dict[str, Any]:
    """Classify clause text into the fixed benchmark categories or ``Other``."""
    tokens = _tokenize(text)
    scores = {category: _score_category(tokens, terms) for category, terms in CATEGORY_PROFILES.items()}
    best_category, best_score = max(scores.items(), key=lambda item: (item[1], item[0]))
    total_score = sum(scores.values())
    if best_score < MIN_CLASSIFICATION_SCORE or total_score == 0.0:
        return ClauseClassification(OTHER_CATEGORY, 0.0, scores).to_dict()
    return ClauseClassification(best_category, best_score / total_score, scores).to_dict()


def infer_category(text: str) -> str:
    """Compatibility helper returning only the best category label."""
    return str(classify_clause(text)["category"])


def _score_category(tokens: list[str], profile_terms: tuple[str, ...]) -> float:
    token_counts = {token: tokens.count(token) for token in set(tokens)}
    return float(sum(token_counts.get(term, 0) for term in profile_terms))


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())
