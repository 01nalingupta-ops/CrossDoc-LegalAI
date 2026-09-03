"""Configured deterministic risk scoring for verified predictions."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

DEFAULT_RISK_CONFIG: dict[str, Any] = {
    "severity_weights": {"None": 0.0, "Low": 0.35, "Medium": 0.65, "High": 1.0},
    "category_weights": {
        "Payment Terms": 0.75,
        "Liability Cap": 1.0,
        "IP Ownership": 0.9,
        "Termination Notice": 0.7,
        "Governing Jurisdiction": 0.8,
        "Confidentiality Scope": 0.85,
        "Indemnification": 0.95,
        "Insurance Requirements": 0.8,
        "Other": 0.5,
    },
    "factors": {"severity": 0.5, "confidence": 0.35, "category": 0.15},
    "scale": 100.0,
}


@dataclass(frozen=True)
class RiskScoreBreakdown:
    risk_score: float
    severity_weight: float
    confidence_weight: float
    category_weight: float

    def to_dict(self) -> dict[str, float]:
        return {
            "risk_score": self.risk_score,
            "severity_weight": self.severity_weight,
            "confidence_weight": self.confidence_weight,
            "category_weight": self.category_weight,
        }


def score_prediction(prediction: dict[str, Any], category: str, config: dict[str, Any] | None = None) -> dict[str, float]:
    """Score a prediction using configured severity, confidence, and category weights."""
    active = config or DEFAULT_RISK_CONFIG
    severity = str(prediction.get("severity", "None"))
    confidence = _clamp(float(prediction.get("confidence", 0.0) or 0.0))
    severity_weight = float(active["severity_weights"].get(severity, active["severity_weights"]["None"]))
    category_weight = float(active["category_weights"].get(category, active["category_weights"].get("Other", 0.0)))
    factors = active["factors"]
    raw_score = (
        float(factors["severity"]) * severity_weight
        + float(factors["confidence"]) * confidence
        + float(factors["category"]) * category_weight
    )
    risk_score = round(_clamp(raw_score) * float(active["scale"]), 2)
    return RiskScoreBreakdown(risk_score, severity_weight, confidence, category_weight).to_dict()


def _clamp(value: float) -> float:
    return max(0.0, min(1.0, value))
