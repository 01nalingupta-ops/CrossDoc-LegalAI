import json
from pathlib import Path

from report_export import build_artifact_path, export_report

FIXTURE = Path(__file__).parent / "fixtures" / "report_evaluation_results.json"

FIGURE_STEMS = [
    "model_metric_bars",
    "roc_curves",
    "confusion_matrices",
    "bootstrap_ci",
    "mcnemar_heatmap",
    "delong_heatmap",
    "ablation_bar",
    "baseline_vs_best_bar",
]
TABLE_STEMS = ["per_model_metrics", "mcnemar_pvalues", "delong_pvalues"]


def test_full_report_export_creates_expected_seed_hash_files(tmp_path):
    eval_results = json.loads(FIXTURE.read_text())
    artifacts = export_report(eval_results, seed=42, dataset_hash="abc123", out_dir=tmp_path)
    artifact_names = {path.name for path in artifacts}

    for stem in FIGURE_STEMS:
        for extension in ("pdf", "eps", "png"):
            expected = f"{stem}_seed42_hashabc123.{extension}"
            assert expected in artifact_names
            assert (tmp_path / expected).exists()
            assert (tmp_path / expected).stat().st_size > 0

    for stem in TABLE_STEMS:
        for extension in ("csv", "tex"):
            expected = f"{stem}_seed42_hashabc123.{extension}"
            assert expected in artifact_names
            assert (tmp_path / expected).exists()
            assert (tmp_path / expected).stat().st_size > 0

    captions = tmp_path / "captions_seed42_hashabc123.md"
    assert captions.name in artifact_names
    assert captions.read_text().count("Figure") == 8
    assert all("seed42_hashabc123" in path.name for path in artifacts)


def test_artifact_filename_helper_sanitizes_hash_and_embeds_metadata(tmp_path):
    path = build_artifact_path("demo", 7, "abc/123!?", tmp_path, "png")
    assert path.name == "demo_seed7_hashabc123.png"


def test_full_report_export_accepts_optional_roc_points(tmp_path):
    eval_results = json.loads(FIXTURE.read_text())
    roc_points = {
        model: [{"fpr": 0.0, "tpr": 0.0}, {"fpr": 0.2, "tpr": metrics["recall"]}, {"fpr": 1.0, "tpr": 1.0}]
        for model, metrics in eval_results["per_model_metrics"].items()
    }
    export_report(eval_results, seed=99, dataset_hash="roc999", out_dir=tmp_path, roc_points=roc_points)
    assert (tmp_path / "roc_curves_seed99_hashroc999.pdf").exists()
