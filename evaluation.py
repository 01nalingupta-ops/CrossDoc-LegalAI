"""Offline evaluation metrics for CrossDoc-LegalAI Part 5.

This module is intentionally decoupled from ingestion, retrieval, LLM auditing, and UI
code. It consumes reduced ground-truth JSON and per-model prediction JSON, then emits the
exact evaluation_results.json schema documented in README.md.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import random
from collections import defaultdict
from typing import Any, Iterable

import numpy as np
from scipy import stats

SEVERITY_SCORES = {"None": 0.0, "Low": 1.0, "Medium": 2.0, "High": 3.0}


def evaluate(
    ground_truth: list[dict[str, Any]],
    predictions: list[dict[str, Any]],
    bootstrap_resamples: int = 1000,
    seed: int = 12345,
) -> dict[str, Any]:
    gt_by_pair = {row["pair_id"]: row for row in ground_truth}
    preds_by_model = _group_predictions_by_model(predictions)

    per_model = {}
    correctness_by_model = {}
    scores_by_model = {}
    labels_by_model = {}
    for model_id in sorted(preds_by_model):
        model_preds = _ordered_predictions_for_model(preds_by_model[model_id], gt_by_pair)
        labels = [bool(gt_by_pair[p["pair_id"]]["has_contradiction"]) for p in model_preds]
        pred_labels = [bool(p["has_contradiction"]) for p in model_preds]
        scores = [_prediction_score(p) for p in model_preds]
        correctness = [pred == label for pred, label in zip(pred_labels, labels)]
        correctness_by_model[model_id] = dict(zip([p["pair_id"] for p in model_preds], correctness))
        scores_by_model[model_id] = dict(zip([p["pair_id"] for p in model_preds], scores))
        labels_by_model[model_id] = dict(zip([p["pair_id"] for p in model_preds], labels))

        cm = confusion_matrix(labels, pred_labels)
        metrics = metrics_from_confusion(cm)
        metrics["faithfulness"] = faithfulness(model_preds)
        metrics["auc"] = roc_auc(labels, scores)
        metrics["confusion_matrix"] = cm
        metrics["confusion_matrix_by_category"] = confusion_by_category(model_preds, gt_by_pair)
        metrics["bootstrap_ci"] = bootstrap_ci(model_preds, gt_by_pair, bootstrap_resamples, seed)
        per_model[model_id] = metrics

    pairwise_mcnemar = {}
    pairwise_delong = {}
    for a, b in itertools.combinations(sorted(preds_by_model), 2):
        common_pair_ids = sorted(set(correctness_by_model[a]) & set(correctness_by_model[b]))
        mcnemar = paired_mcnemar_comparison(
            [preds_by_model[a][pid] for pid in common_pair_ids],
            [preds_by_model[b][pid] for pid in common_pair_ids],
            [gt_by_pair[pid] for pid in common_pair_ids],
        )
        pairwise_mcnemar[f"{a}__vs__{b}"] = mcnemar
        pairwise_delong[f"{a}__vs__{b}"] = delong_comparison(
            [labels_by_model[a][pid] for pid in common_pair_ids],
            [scores_by_model[a][pid] for pid in common_pair_ids],
            [scores_by_model[b][pid] for pid in common_pair_ids],
        )

    _apply_holm_bonferroni(pairwise_mcnemar)
    _apply_holm_bonferroni(pairwise_delong)
    return {
        "per_model_metrics": per_model,
        "pairwise_mcnemar": pairwise_mcnemar,
        "pairwise_delong": pairwise_delong,
    }


def confusion_matrix(labels: Iterable[bool], predictions: Iterable[bool]) -> dict[str, int]:
    cm = {"tp": 0, "tn": 0, "fp": 0, "fn": 0}
    for label, pred in zip(labels, predictions):
        if label and pred:
            cm["tp"] += 1
        elif not label and not pred:
            cm["tn"] += 1
        elif not label and pred:
            cm["fp"] += 1
        else:
            cm["fn"] += 1
    return cm


def metrics_from_confusion(cm: dict[str, int]) -> dict[str, float]:
    tp, tn, fp, fn = cm["tp"], cm["tn"], cm["fp"], cm["fn"]
    total = tp + tn + fp + fn
    accuracy = _safe_div(tp + tn, total)
    precision = _safe_div(tp, tp + fp)
    recall = _safe_div(tp, tp + fn)
    f1 = _safe_div(2 * precision * recall, precision + recall)
    return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1}


def faithfulness(predictions: list[dict[str, Any]]) -> float:
    positive_claims = [p for p in predictions if bool(p.get("has_contradiction", False))]
    if not positive_claims:
        return 1.0
    verified = sum(1 for p in positive_claims if bool(p.get("guardrail_verified", False)))
    return verified / len(positive_claims)


def roc_auc(labels: list[bool], scores: list[float]) -> float:
    positives = [s for y, s in zip(labels, scores) if y]
    negatives = [s for y, s in zip(labels, scores) if not y]
    if not positives or not negatives:
        return math.nan
    wins = 0.0
    for pos in positives:
        for neg in negatives:
            if pos > neg:
                wins += 1.0
            elif pos == neg:
                wins += 0.5
    return wins / (len(positives) * len(negatives))


def confusion_by_category(predictions: list[dict[str, Any]], gt_by_pair: dict[str, dict[str, Any]]) -> dict[str, dict[str, int]]:
    rows = defaultdict(lambda: {"labels": [], "predictions": []})
    for pred in predictions:
        gt = gt_by_pair[pred["pair_id"]]
        category = gt.get("contradiction_category") or "None"
        rows[category]["labels"].append(bool(gt["has_contradiction"]))
        rows[category]["predictions"].append(bool(pred["has_contradiction"]))
    return {category: confusion_matrix(data["labels"], data["predictions"]) for category, data in sorted(rows.items())}


def paired_mcnemar_comparison(
    predictions_a: list[dict[str, Any]],
    predictions_b: list[dict[str, Any]],
    ground_truth: list[dict[str, Any]],
) -> dict[str, float | bool]:
    gt_by_pair = {row["pair_id"]: bool(row["has_contradiction"]) for row in ground_truth}
    a_by_pair = {row["pair_id"]: bool(row["has_contradiction"]) for row in predictions_a}
    b_by_pair = {row["pair_id"]: bool(row["has_contradiction"]) for row in predictions_b}
    n01 = n10 = 0
    for pair_id in sorted(set(gt_by_pair) & set(a_by_pair) & set(b_by_pair)):
        a_right = a_by_pair[pair_id] == gt_by_pair[pair_id]
        b_right = b_by_pair[pair_id] == gt_by_pair[pair_id]
        if not a_right and b_right:
            n01 += 1
        elif a_right and not b_right:
            n10 += 1
    discordant = n01 + n10
    if discordant == 0:
        statistic = 0.0
        p_value = 1.0
    elif discordant < 25:
        statistic = float(min(n01, n10))
        p_value = float(stats.binomtest(min(n01, n10), discordant, 0.5, alternative="two-sided").pvalue)
    else:
        statistic = (abs(n01 - n10) - 1) ** 2 / discordant
        p_value = float(stats.chi2.sf(statistic, df=1))
    return {"chi2_or_exact": float(statistic), "p_value": p_value, "p_value_holm_adjusted": p_value, "significant": False}


def delong_comparison(labels: list[bool], scores_a: list[float], scores_b: list[float]) -> dict[str, float | bool]:
    labels_arr = np.asarray(labels, dtype=bool)
    a = np.asarray(scores_a, dtype=float)
    b = np.asarray(scores_b, dtype=float)
    auc_a, v10_a, v01_a = _delong_structural_components(labels_arr, a)
    auc_b, v10_b, v01_b = _delong_structural_components(labels_arr, b)
    m = len(v10_a)
    n = len(v01_a)
    if m == 0 or n == 0:
        z = math.nan
        p_value = math.nan
    else:
        var_a = _sample_variance(v10_a) / m + _sample_variance(v01_a) / n
        var_b = _sample_variance(v10_b) / m + _sample_variance(v01_b) / n
        cov_ab = _sample_covariance(v10_a, v10_b) / m + _sample_covariance(v01_a, v01_b) / n
        denom = var_a + var_b - 2 * cov_ab
        if denom <= 0:
            z = 0.0 if auc_a == auc_b else math.copysign(math.inf, auc_a - auc_b)
            p_value = 1.0 if auc_a == auc_b else 0.0
        else:
            z = (auc_a - auc_b) / math.sqrt(denom)
            p_value = float(2 * stats.norm.sf(abs(z)))
    return {"z": float(z), "p_value": float(p_value), "p_value_holm_adjusted": float(p_value), "significant": False}


def bootstrap_ci(
    predictions: list[dict[str, Any]],
    gt_by_pair: dict[str, dict[str, Any]],
    resamples: int = 1000,
    seed: int = 12345,
) -> dict[str, list[float]]:
    rng = random.Random(seed)
    n = len(predictions)
    values = {"accuracy": [], "f1": [], "auc": []}
    for _ in range(resamples):
        sample = [predictions[rng.randrange(n)] for _ in range(n)]
        labels = [bool(gt_by_pair[p["pair_id"]]["has_contradiction"]) for p in sample]
        pred_labels = [bool(p["has_contradiction"]) for p in sample]
        scores = [_prediction_score(p) for p in sample]
        metric_row = metrics_from_confusion(confusion_matrix(labels, pred_labels))
        values["accuracy"].append(metric_row["accuracy"])
        values["f1"].append(metric_row["f1"])
        auc = roc_auc(labels, scores)
        if not math.isnan(auc):
            values["auc"].append(auc)
    return {name: _percentile_ci(vals) for name, vals in values.items()}


def _delong_structural_components(labels: np.ndarray, scores: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    positives = scores[labels]
    negatives = scores[~labels]
    if len(positives) == 0 or len(negatives) == 0:
        return math.nan, np.array([]), np.array([])
    comparisons = _heaviside(positives[:, None] - negatives[None, :])
    v10 = comparisons.mean(axis=1)
    v01 = comparisons.mean(axis=0)
    return float(comparisons.mean()), v10, v01


def _heaviside(diff: np.ndarray) -> np.ndarray:
    return (diff > 0).astype(float) + 0.5 * (diff == 0).astype(float)


def _sample_variance(values: np.ndarray) -> float:
    if len(values) <= 1:
        return 0.0
    return float(np.var(values, ddof=1))


def _sample_covariance(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) <= 1:
        return 0.0
    return float(np.cov(a, b, ddof=1)[0, 1])


def _prediction_score(prediction: dict[str, Any]) -> float:
    confidence = prediction.get("confidence")
    if confidence is None:
        return SEVERITY_SCORES.get(str(prediction.get("severity", "None")), 0.0)
    return float(confidence)


def _percentile_ci(values: list[float]) -> list[float]:
    if not values:
        return [math.nan, math.nan]
    return [float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))]


def _safe_div(numerator: float, denominator: float) -> float:
    return 0.0 if denominator == 0 else numerator / denominator


def _group_predictions_by_model(predictions: list[dict[str, Any]]) -> dict[str, dict[str, dict[str, Any]]]:
    grouped: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for pred in predictions:
        grouped[pred["model_id"]][pred["pair_id"]] = pred
    return dict(grouped)


def _ordered_predictions_for_model(preds_by_pair: dict[str, dict[str, Any]], gt_by_pair: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    return [preds_by_pair[pair_id] for pair_id in sorted(gt_by_pair) if pair_id in preds_by_pair]


def _apply_holm_bonferroni(results: dict[str, dict[str, Any]], alpha: float = 0.05) -> None:
    finite = [(key, row["p_value"]) for key, row in results.items() if not math.isnan(float(row["p_value"]))]
    m = len(finite)
    previous = 0.0
    for rank, (key, p_value) in enumerate(sorted(finite, key=lambda item: item[1]), start=1):
        adjusted = min(1.0, max(previous, p_value * (m - rank + 1)))
        previous = adjusted
        results[key]["p_value_holm_adjusted"] = adjusted
        results[key]["significant"] = adjusted <= alpha


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute CrossDoc-LegalAI Part 5 evaluation metrics.")
    parser.add_argument("--ground-truth", required=True, help="Path to reduced ground-truth JSON array.")
    parser.add_argument("--predictions", required=True, help="Path to per-model prediction JSON array.")
    parser.add_argument("--out", required=True, help="Path for evaluation_results.json.")
    parser.add_argument("--bootstrap-resamples", type=int, default=1000, help="Bootstrap resamples for 95%% CIs.")
    parser.add_argument("--seed", type=int, default=12345, help="Deterministic bootstrap seed.")
    args = parser.parse_args()
    with open(args.ground_truth, "r", encoding="utf-8") as handle:
        ground_truth = json.load(handle)
    with open(args.predictions, "r", encoding="utf-8") as handle:
        predictions = json.load(handle)
    result = evaluate(ground_truth, predictions, args.bootstrap_resamples, args.seed)
    with open(args.out, "w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, sort_keys=True)
        handle.write("\n")


if __name__ == "__main__":
    main()
