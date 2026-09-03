from risk_scoring import DEFAULT_RISK_CONFIG, score_prediction


def test_risk_score_is_monotonic_for_confidence_with_same_severity_and_category():
    low = score_prediction({"severity": "Medium", "confidence": 0.2}, "Payment Terms")
    high = score_prediction({"severity": "Medium", "confidence": 0.8}, "Payment Terms")

    assert high["risk_score"] > low["risk_score"]


def test_risk_score_is_monotonic_for_configured_severity_weights():
    low = score_prediction({"severity": "Low", "confidence": 0.7}, "Payment Terms")
    high = score_prediction({"severity": "High", "confidence": 0.7}, "Payment Terms")

    assert high["risk_score"] > low["risk_score"]


def test_risk_score_uses_configured_category_weights_without_inline_conditionals():
    custom_config = {
        **DEFAULT_RISK_CONFIG,
        "category_weights": {**DEFAULT_RISK_CONFIG["category_weights"], "Other": 0.1, "Liability Cap": 1.0},
    }

    other = score_prediction({"severity": "Medium", "confidence": 0.7}, "Other", config=custom_config)
    liability = score_prediction({"severity": "Medium", "confidence": 0.7}, "Liability Cap", config=custom_config)

    assert liability["risk_score"] > other["risk_score"]
    assert liability["category_weight"] == 1.0
