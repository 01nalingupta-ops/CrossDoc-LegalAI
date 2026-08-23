"""Figure and table exports for CrossDoc-LegalAI Part 6.

The module presents an existing evaluation_results.json object; it does not compute
metrics or statistical tests. It uses matplotlib when installed and falls back to a small
Pillow renderer in minimal environments. Every artifact filename embeds the caller's
seed and dataset_hash for reproducibility.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

if importlib.util.find_spec("matplotlib") is not None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
else:
    plt = None
    Rectangle = None

if importlib.util.find_spec("PIL") is not None:
    from PIL import Image, ImageDraw, ImageFont
else:
    Image = None
    ImageDraw = None
    ImageFont = None

FONT_FAMILY = "DejaVu Sans"
FONT_SIZE = 10
FIGURE_FORMATS = ("pdf", "eps", "png")
METRIC_COLUMNS = ["accuracy", "precision", "recall", "f1", "auc", "faithfulness"]
BAR_METRICS = ["accuracy", "precision", "recall", "f1"]

if plt is not None:
    plt.rcParams.update({"font.family": FONT_FAMILY, "font.size": FONT_SIZE, "axes.titlesize": FONT_SIZE + 1, "axes.labelsize": FONT_SIZE})


def export_report(eval_results: dict[str, Any], seed: int, dataset_hash: str, out_dir: str | Path, roc_points: dict[str, list[dict[str, float]]] | None = None) -> list[Path]:
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    model_ids = sorted(eval_results["per_model_metrics"])
    colors = build_model_colors(model_ids)
    captions: list[str] = []
    artifacts: list[Path] = []
    figure_specs = [
        ("model_metric_bars", lambda: plot_metric_bars(eval_results, colors), "Figure 1. Accuracy, precision, recall, and F1 are compared across all evaluated models; see the per-model metrics table for exact values."),
        ("roc_curves", lambda: plot_roc_or_auc(eval_results, colors, roc_points), "Figure 2. ROC performance is shown using supplied curve points when available, otherwise as an AUC fallback bar chart from evaluation_results.json."),
        ("confusion_matrices", lambda: plot_confusion_matrices(eval_results, colors), "Figure 3. Overall confusion matrices summarize true/false positives and negatives for each model."),
        ("bootstrap_ci", lambda: plot_bootstrap_ci(eval_results, colors), "Figure 4. Bootstrap confidence intervals show uncertainty bounds for accuracy, F1, and AUC by model."),
        ("mcnemar_heatmap", lambda: plot_pvalue_heatmap(eval_results["pairwise_mcnemar"], model_ids, "McNemar Holm-adjusted p-values"), "Figure 5. McNemar Holm-adjusted p-values compare paired correctness between models, with significant cells outlined."),
        ("delong_heatmap", lambda: plot_pvalue_heatmap(eval_results["pairwise_delong"], model_ids, "DeLong Holm-adjusted p-values"), "Figure 6. DeLong Holm-adjusted p-values compare correlated ROC-AUC estimates, with significant cells outlined."),
    ]
    for stem, factory, caption in figure_specs:
        artifacts.extend(save_figure(factory(), stem, seed, dataset_hash, out_path))
        captions.append(caption)
    model_a, model_b = _first_two_metric_dicts(eval_results)
    artifacts.extend(save_figure(plot_two_metric_dicts(model_a[1], model_b[1], model_a[0], model_b[0], title="Generic ablation comparison"), "ablation_bar", seed, dataset_hash, out_path))
    captions.append("Figure 7. Generic ablation comparison displays any two labeled metric dictionaries supplied by the caller.")
    artifacts.extend(save_figure(plot_two_metric_dicts(model_a[1], model_b[1], model_a[0], model_b[0], title="Generic baseline-vs-best comparison"), "baseline_vs_best_bar", seed, dataset_hash, out_path))
    captions.append("Figure 8. Generic baseline-vs-best comparison reuses the two-dictionary bar chart for pipeline-versus-baseline reporting.")
    artifacts.extend(export_metrics_table(eval_results, seed, dataset_hash, out_path))
    artifacts.extend(export_pvalue_table(eval_results["pairwise_mcnemar"], model_ids, "mcnemar_pvalues", seed, dataset_hash, out_path))
    artifacts.extend(export_pvalue_table(eval_results["pairwise_delong"], model_ids, "delong_pvalues", seed, dataset_hash, out_path))
    captions_path = build_artifact_path("captions", seed, dataset_hash, out_path, "md")
    captions_path.write_text("\n".join(f"- {caption}" for caption in captions) + "\n", encoding="utf-8")
    artifacts.append(captions_path)
    return artifacts


def build_artifact_path(stem: str, seed: int, dataset_hash: str, out_dir: Path, extension: str) -> Path:
    safe_hash = "".join(ch for ch in dataset_hash if ch.isalnum() or ch in ("-", "_"))
    return out_dir / f"{stem}_seed{seed}_hash{safe_hash}.{extension}"


def build_model_colors(model_ids: list[str]) -> dict[str, Any]:
    palette = ["#4C78A8", "#F58518", "#54A24B", "#E45756", "#72B7B2", "#B279A2", "#FF9DA6", "#9D755D"]
    if plt is not None:
        cmap = plt.get_cmap("tab10")
        return {model_id: cmap(index % 10) for index, model_id in enumerate(sorted(model_ids))}
    return {model_id: palette[index % len(palette)] for index, model_id in enumerate(sorted(model_ids))}


def save_figure(fig: Any, stem: str, seed: int, dataset_hash: str, out_dir: Path) -> list[Path]:
    paths = []
    for extension in FIGURE_FORMATS:
        path = build_artifact_path(stem, seed, dataset_hash, out_dir, extension)
        if plt is not None and hasattr(fig, "savefig"):
            fig.savefig(path, dpi=300 if extension == "png" else None, bbox_inches="tight")
        else:
            _save_pillow_chart(fig, path)
        paths.append(path)
    if plt is not None and hasattr(fig, "savefig"):
        plt.close(fig)
    return paths


def plot_metric_bars(eval_results: dict[str, Any], colors: dict[str, Any]) -> Any:
    if plt is None:
        return _spec("Core classification metrics by model", eval_results, colors)
    model_ids = sorted(eval_results["per_model_metrics"])
    x = np.arange(len(model_ids)); width = 0.18
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for offset, metric in enumerate(BAR_METRICS):
        ax.bar(x + (offset - 1.5) * width, [eval_results["per_model_metrics"][m][metric] for m in model_ids], width, label=metric.title())
    ax.set_xticks(x); ax.set_xticklabels(model_ids, rotation=20, ha="right"); ax.set_ylim(0, 1.05); ax.set_ylabel("Score"); ax.set_title("Core classification metrics by model"); ax.legend(ncol=2); ax.grid(axis="y", alpha=0.25)
    return fig


def plot_roc_or_auc(eval_results: dict[str, Any], colors: dict[str, Any], roc_points: dict[str, list[dict[str, float]]] | None) -> Any:
    if plt is None:
        title = "ROC curves by model" if roc_points else "ROC fallback: AUC by model"
        return _spec(title, eval_results, colors)
    model_ids = sorted(eval_results["per_model_metrics"]); fig, ax = plt.subplots(figsize=(6, 5))
    if roc_points:
        for model in model_ids:
            points = sorted(roc_points.get(model, []), key=lambda row: row["fpr"])
            if points:
                ax.plot([p["fpr"] for p in points], [p["tpr"] for p in points], label=f"{model} (AUC={eval_results['per_model_metrics'][model]['auc']:.3f})", color=colors[model])
        ax.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1); ax.set_xlabel("False positive rate"); ax.set_ylabel("True positive rate"); ax.set_title("ROC curves by model"); ax.legend()
    else:
        ax.bar(model_ids, [eval_results["per_model_metrics"][m]["auc"] for m in model_ids], color=[colors[m] for m in model_ids]); ax.set_ylim(0, 1.05); ax.set_ylabel("AUC"); ax.set_title("ROC fallback: AUC by model"); ax.text(0.5, -0.18, "Raw ROC points were not supplied; bars show per-model AUC values.", transform=ax.transAxes, ha="center", va="top")
    ax.grid(alpha=0.25); return fig


def plot_confusion_matrices(eval_results: dict[str, Any], colors: dict[str, Any]) -> Any:
    if plt is None:
        return _spec("Confusion matrices by model", eval_results, colors)
    model_ids = sorted(eval_results["per_model_metrics"]); cols = min(4, len(model_ids)); rows = math.ceil(len(model_ids) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(3 * cols, 3 * rows), squeeze=False)
    for ax in axes.ravel(): ax.axis("off")
    im = None
    for ax, model in zip(axes.ravel(), model_ids):
        cm = eval_results["per_model_metrics"][model]["confusion_matrix"]; matrix = np.array([[cm["tn"], cm["fp"]], [cm["fn"], cm["tp"]]])
        ax.axis("on"); im = ax.imshow(matrix, cmap="Blues", vmin=0); ax.set_title(model, color=colors[model]); ax.set_xticks([0, 1], labels=["Pred 0", "Pred 1"]); ax.set_yticks([0, 1], labels=["True 0", "True 1"])
        for i in range(2):
            for j in range(2): ax.text(j, i, str(matrix[i, j]), ha="center", va="center")
    fig.suptitle("Confusion matrices by model")
    if im is not None: fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.7)
    return fig


def plot_bootstrap_ci(eval_results: dict[str, Any], colors: dict[str, Any]) -> Any:
    if plt is None:
        return _spec("Bootstrap confidence intervals", eval_results, colors)
    model_ids = sorted(eval_results["per_model_metrics"]); metrics = ["accuracy", "f1", "auc"]; fig, ax = plt.subplots(figsize=(8, 4.5)); x_base = np.arange(len(metrics)); width = 0.8 / max(1, len(model_ids))
    for idx, model in enumerate(model_ids):
        centers = x_base - 0.4 + width / 2 + idx * width; lows=[]; mids=[]; highs=[]
        for metric in metrics:
            low, high = eval_results["per_model_metrics"][model]["bootstrap_ci"][metric]; mid = eval_results["per_model_metrics"][model][metric]
            lows.append(mid - low); highs.append(high - mid); mids.append(mid)
        ax.errorbar(centers, mids, yerr=[lows, highs], fmt="o", capsize=4, label=model, color=colors[model])
    ax.set_xticks(x_base, labels=[m.title() for m in metrics]); ax.set_ylim(0, 1.05); ax.set_ylabel("Score with 95% CI"); ax.set_title("Bootstrap confidence intervals"); ax.legend(); ax.grid(axis="y", alpha=0.25); return fig


def plot_pvalue_heatmap(pairwise: dict[str, dict[str, Any]], model_ids: list[str], title: str) -> Any:
    if plt is None:
        return {"title": title, "models": model_ids, "pairwise": pairwise}
    matrix = np.full((len(model_ids), len(model_ids)), np.nan); significant = np.zeros_like(matrix, dtype=bool); index = {m: i for i, m in enumerate(model_ids)}
    for key, row in pairwise.items():
        a, b = key.split("__vs__"); i, j = index[a], index[b]; value = row.get("p_value_holm_adjusted", row.get("p_value", math.nan)); matrix[i, j] = matrix[j, i] = value; significant[i, j] = significant[j, i] = bool(row.get("significant", False))
    np.fill_diagonal(matrix, 1.0); fig, ax = plt.subplots(figsize=(5.5, 4.8)); im = ax.imshow(matrix, cmap="viridis_r", vmin=0, vmax=1); ax.set_xticks(range(len(model_ids)), labels=model_ids, rotation=30, ha="right"); ax.set_yticks(range(len(model_ids)), labels=model_ids); ax.set_title(title)
    for i in range(len(model_ids)):
        for j in range(len(model_ids)):
            ax.text(j, i, "—" if i == j else f"{matrix[i, j]:.3f}", ha="center", va="center", color="white" if matrix[i, j] < 0.45 else "black")
            if significant[i, j] and i != j and Rectangle is not None: ax.add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False, edgecolor="red", linewidth=2.5))
    fig.colorbar(im, ax=ax, label="Holm-adjusted p-value"); return fig


def plot_two_metric_dicts(metrics_a: dict[str, float], metrics_b: dict[str, float], label_a: str, label_b: str, colors: dict[str, Any] | None = None, title: str = "Metric comparison") -> Any:
    if plt is None:
        return {"title": title, "rows": [(label_a, metrics_a), (label_b, metrics_b)]}
    metrics = [m for m in BAR_METRICS if m in metrics_a and m in metrics_b]; x = np.arange(len(metrics)); fig, ax = plt.subplots(figsize=(6.5, 4)); color_a = colors.get(label_a) if colors and label_a in colors else "#4C78A8"; color_b = colors.get(label_b) if colors and label_b in colors else "#F58518"
    ax.bar(x - 0.18, [metrics_a[m] for m in metrics], 0.36, label=label_a, color=color_a); ax.bar(x + 0.18, [metrics_b[m] for m in metrics], 0.36, label=label_b, color=color_b); ax.set_xticks(x, labels=[m.title() for m in metrics]); ax.set_ylim(0, 1.05); ax.set_ylabel("Score"); ax.set_title(title); ax.legend(); ax.grid(axis="y", alpha=0.25); return fig


def export_metrics_table(eval_results: dict[str, Any], seed: int, dataset_hash: str, out_dir: Path) -> list[Path]:
    rows = [[model] + [metrics[metric] for metric in METRIC_COLUMNS] for model, metrics in sorted(eval_results["per_model_metrics"].items())]
    return _write_table(["model_id"] + METRIC_COLUMNS, rows, "per_model_metrics", seed, dataset_hash, out_dir)


def export_pvalue_table(pairwise: dict[str, dict[str, Any]], model_ids: list[str], stem: str, seed: int, dataset_hash: str, out_dir: Path) -> list[Path]:
    table = {(a, b): "—" for a in model_ids for b in model_ids}
    for key, row in pairwise.items():
        a, b = key.split("__vs__"); value = row.get("p_value_holm_adjusted", row.get("p_value", math.nan)); display = f"{value:.4f}" + ("*" if row.get("significant", False) else ""); table[(a, b)] = table[(b, a)] = display
    rows = [[a] + [table[(a, b)] for b in model_ids] for a in model_ids]
    return _write_table(["model_id"] + model_ids, rows, stem, seed, dataset_hash, out_dir)


def _write_table(headers: list[str], rows: list[list[Any]], stem: str, seed: int, dataset_hash: str, out_dir: Path) -> list[Path]:
    csv_path = build_artifact_path(stem, seed, dataset_hash, out_dir, "csv")
    tex_path = build_artifact_path(stem, seed, dataset_hash, out_dir, "tex")
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(headers)
        writer.writerows(rows)
    tex_lines = [
        "\\begin{tabular}{" + "l" * len(headers) + "}",
        "\\toprule",
        " & ".join(_fmt(h) for h in headers) + r" \\",
        "\\midrule",
    ]
    for row in rows:
        tex_lines.append(" & ".join(_fmt(value) for value in row) + r" \\")
    tex_lines.extend(["\\bottomrule", "\\end{tabular}", ""])
    tex_path.write_text("\n".join(tex_lines), encoding="utf-8")
    return [csv_path, tex_path]


def _fmt(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value).replace("_", "\\_")


def _first_two_metric_dicts(eval_results: dict[str, Any]) -> tuple[tuple[str, dict[str, float]], tuple[str, dict[str, float]]]:
    models = sorted(eval_results["per_model_metrics"])
    if len(models) == 1:
        return (models[0], eval_results["per_model_metrics"][models[0]]), (models[0], eval_results["per_model_metrics"][models[0]])
    return (models[0], eval_results["per_model_metrics"][models[0]]), (models[1], eval_results["per_model_metrics"][models[1]])


def _spec(title: str, eval_results: dict[str, Any], colors: dict[str, Any]) -> dict[str, Any]:
    return {"title": title, "eval_results": eval_results, "colors": colors}


def _save_pillow_chart(spec: dict[str, Any], path: Path) -> None:
    if Image is None or ImageDraw is None:
        path.write_text(f"{spec.get('title', 'CrossDoc-LegalAI chart')}\n", encoding="utf-8")
        return
    image = Image.new("RGB", (1800, 1200), "white"); draw = ImageDraw.Draw(image); font = ImageFont.load_default() if ImageFont is not None else None
    title = spec.get("title", "CrossDoc-LegalAI chart"); draw.text((60, 40), title, fill="black", font=font)
    metrics = spec.get("eval_results", {}).get("per_model_metrics", {})
    x = 80; y = 130; bar_w = 55; scale = 700
    for model, row in sorted(metrics.items()):
        draw.text((x, y + scale + 20), model, fill="black", font=font)
        for i, metric in enumerate(BAR_METRICS):
            value = row.get(metric, row.get("auc", 0)); h = int(value * scale); color = spec.get("colors", {}).get(model, "#4C78A8")
            draw.rectangle((x + i * (bar_w + 8), y + scale - h, x + i * (bar_w + 8) + bar_w, y + scale), fill=color)
        x += 260
    image.save(path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Export CrossDoc-LegalAI Part 6 report figures and tables.")
    parser.add_argument("--eval-results", required=True, help="Path to evaluation_results.json.")
    parser.add_argument("--seed", required=True, type=int, help="Run seed to embed in exported filenames.")
    parser.add_argument("--dataset-hash", required=True, help="Dataset hash to embed in exported filenames.")
    parser.add_argument("--out-dir", required=True, help="Directory for exported figures, tables, and captions.")
    parser.add_argument("--roc-points", help="Optional JSON file: {model_id: [{fpr: float, tpr: float}, ...]}.")
    args = parser.parse_args()
    with open(args.eval_results, "r", encoding="utf-8") as handle:
        eval_results = json.load(handle)
    roc_points = None
    if args.roc_points:
        with open(args.roc_points, "r", encoding="utf-8") as handle:
            roc_points = json.load(handle)
    export_report(eval_results, args.seed, args.dataset_hash, args.out_dir, roc_points=roc_points)


if __name__ == "__main__":
    main()
