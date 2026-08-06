import json
from pathlib import Path

import pytest

from evaluation import (
    bootstrap_ci,
    confusion_matrix,
    evaluate,
    paired_mcnemar_comparison,
    roc_auc,
)

FIXTURES = Path(__file__).parent / "fixtures"


def test_confusion_matrix_and_metrics_known_values():
    gt = json.loads((FIXTURES / "evaluation_ground_truth.json").read_text())
    preds = json.loads((FIXTURES / "evaluation_predictions.json").read_text())
    result = evaluate(gt, preds, bootstrap_resamples=50, seed=7)

    alpha = result["per_model_metrics"]["alpha"]
    assert alpha["confusion_matrix"] == {"tp": 2, "tn": 2, "fp": 1, "fn": 1}
    assert alpha["accuracy"] == pytest.approx(4 / 6)
    assert alpha["precision"] == pytest.approx(2 / 3)
    assert alpha["recall"] == pytest.approx(2 / 3)
    assert alpha["f1"] == pytest.approx(2 / 3)
    assert alpha["faithfulness"] == pytest.approx(2 / 3)
    assert alpha["auc"] == pytest.approx(8 / 9)
    assert alpha["confusion_matrix_by_category"]["Payment Terms"] == {"tp": 1, "tn": 0, "fp": 0, "fn": 0}


def test_mcnemar_exact_p_value_hand_verified():
    gt = [{"pair_id": f"p{i}", "has_contradiction": True} for i in range(1, 13)]
    preds_a = []
    preds_b = []
    for i in range(1, 13):
        # n01 = 2 (A wrong/B right), n10 = 8 (A right/B wrong), discordant=10.
        preds_a.append({"pair_id": f"p{i}", "has_contradiction": i > 2})
        preds_b.append({"pair_id": f"p{i}", "has_contradiction": i <= 2 or i > 10})
    result = paired_mcnemar_comparison(preds_a, preds_b, gt)
    assert result["chi2_or_exact"] == 2.0
    # Exact two-sided binomial: 2 * sum_{k=0}^2 C(10,k) / 2^10 = 112 / 1024.
    assert result["p_value"] == pytest.approx(112 / 1024)


def test_bootstrap_ci_sanity_check_more_data_narrows_accuracy_ci():
    small_gt = {f"p{i}": {"pair_id": f"p{i}", "has_contradiction": i % 2 == 0} for i in range(8)}
    small_preds = [
        {"pair_id": pid, "has_contradiction": row["has_contradiction"], "confidence": 0.9 if row["has_contradiction"] else 0.1}
        for pid, row in small_gt.items()
    ]
    small_preds[0]["has_contradiction"] = not small_preds[0]["has_contradiction"]

    large_gt = {f"p{i}": {"pair_id": f"p{i}", "has_contradiction": i % 2 == 0} for i in range(80)}
    large_preds = [
        {"pair_id": pid, "has_contradiction": row["has_contradiction"], "confidence": 0.9 if row["has_contradiction"] else 0.1}
        for pid, row in large_gt.items()
    ]
    for i in range(10):
        large_preds[i]["has_contradiction"] = not large_preds[i]["has_contradiction"]

    small_ci = bootstrap_ci(small_preds, small_gt, resamples=400, seed=1)["accuracy"]
    large_ci = bootstrap_ci(large_preds, large_gt, resamples=400, seed=1)["accuracy"]
    assert small_ci[1] - small_ci[0] > large_ci[1] - large_ci[0]


def test_auc_uses_severity_fallback_when_confidence_missing():
    labels = [True, False, True, False]
    assert roc_auc(labels, [3, 0, 1, 0]) == pytest.approx(1.0)
    gt = [
        {"pair_id": "p1", "has_contradiction": True, "contradiction_category": "Payment Terms", "severity": "High"},
        {"pair_id": "p2", "has_contradiction": False, "contradiction_category": None, "severity": "None"},
    ]
    preds = [
        {"pair_id": "p1", "model_id": "fallback", "has_contradiction": True, "confidence": None, "severity": "High", "guardrail_verified": True},
        {"pair_id": "p2", "model_id": "fallback", "has_contradiction": False, "confidence": None, "severity": "None", "guardrail_verified": True},
    ]
    result = evaluate(gt, preds, bootstrap_resamples=20)
    assert result["per_model_metrics"]["fallback"]["auc"] == pytest.approx(1.0)


def test_evaluation_excludes_unconfigured_candidate_placeholders_from_pairwise_tests():
    gt = [
        {"pair_id": "p1", "has_contradiction": True, "contradiction_category": "Payment Terms", "severity": "High"},
        {"pair_id": "p2", "has_contradiction": False, "contradiction_category": None, "severity": "None"},
    ]
    preds = [
        {"pair_id": "p1", "model_id": "candidate_1", "has_contradiction": True, "confidence": 0.9, "severity": "High", "guardrail_verified": True},
        {"pair_id": "p2", "model_id": "candidate_1", "has_contradiction": False, "confidence": 0.1, "severity": "None", "guardrail_verified": True},
        {"pair_id": "p1", "model_id": "candidate_2", "has_contradiction": True, "confidence": 0.8, "severity": "Medium", "guardrail_verified": True},
        {"pair_id": "p2", "model_id": "candidate_2", "has_contradiction": False, "confidence": 0.2, "severity": "None", "guardrail_verified": True},
        {"pair_id": "p1", "model_id": "candidate_3", "has_contradiction": False, "confidence": 0.0, "severity": "None", "guardrail_verified": True, "evaluation_excluded": True},
        {"pair_id": "p2", "model_id": "candidate_3", "has_contradiction": False, "confidence": 0.0, "severity": "None", "guardrail_verified": True, "evaluation_excluded": True},
    ]

    result = evaluate(gt, preds, bootstrap_resamples=10, seed=3)

    assert set(result["per_model_metrics"]) == {"candidate_1", "candidate_2"}
    assert set(result["pairwise_mcnemar"]) == {"candidate_1__vs__candidate_2"}
    assert set(result["pairwise_delong"]) == {"candidate_1__vs__candidate_2"}
