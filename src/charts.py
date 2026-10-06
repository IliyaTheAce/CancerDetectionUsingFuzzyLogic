"""Build the paper figures from the saved metric tables.

Images go under `charts/<membership>/`, using the membership named in
`membership_params.csv`. A quartile run writes `charts/quartile/`, an MRI
run writes `charts/mri/`, and the same for `eau`, `classic`, and `gray_zone`.
This step only reads CSVs written by the earlier pipeline stages; it does
not refit models.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay, roc_curve

from src.compare import MODEL_COLORS, MODEL_LABELS, MODEL_ORDER
from src import fuzzify
from src.fuzzify import SELECTED_FEATURES, TERMS, membership_degrees
from src.preprocess import OUTPUT_DIR, ROOT, TARGET_CODE_COLUMN

METRIC_ORDER = (
    ("accuracy", "Accuracy"),
    ("precision", "Precision"),
    ("sensitivity", "Sensitivity"),
    ("specificity", "Specificity"),
    ("f1", "F1"),
)
KEY_METRICS = (("accuracy", "Accuracy"), ("f1", "F1"), ("roc_auc", "ROC-AUC"))
FEATURE_TITLES = {"PSAD": "PSAD", "f_t_psa": "f/t PSA", "PV": "Prostate volume"}
TERM_COLORS = {"low": "#4C78A8", "medium": "#F58518", "high": "#E45756"}


def _membership_name() -> str:
    params_path = OUTPUT_DIR / "membership_params.csv"
    if params_path.exists():
        params = pd.read_csv(params_path)
        if "membership" in params.columns and not params.empty:
            return str(params["membership"].iloc[0])
    return fuzzify.ACTIVE_MEMBERSHIP


def charts_dir() -> Path:
    folder = ROOT / "charts" / _membership_name()
    folder.mkdir(parents=True, exist_ok=True)
    return folder


def _save(figure: plt.Figure, name: str) -> Path:
    path = charts_dir() / name
    figure.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(figure)
    return path


def _knot_triple(row: object) -> tuple[float, float, float]:
    if hasattr(row, "low_shoulder"):
        return float(row.low_shoulder), float(row.medium_peak), float(row.high_shoulder)
    return float(row.q25_low_shoulder), float(row.q50_medium_peak), float(row.q75_high_shoulder)


def _membership() -> Path:
    params = pd.read_csv(OUTPUT_DIR / "membership_params.csv")
    train = pd.read_csv(OUTPUT_DIR / "train_model.csv")
    knots = {row.feature: _knot_triple(row) for row in params.itertuples(index=False)}
    membership_name = _membership_name()

    figure, axes = plt.subplots(1, 3, figsize=(11.2, 3.6))
    for axis, feature in zip(axes, SELECTED_FEATURES):
        low_knot, medium_knot, high_knot = knots[feature]
        left = min(float(train[feature].quantile(0.01)), low_knot)
        right = high_knot + (high_knot - medium_knot)
        grid = np.linspace(left, right, 400)
        grid = np.sort(np.unique(np.concatenate([grid, [low_knot, medium_knot, high_knot]])))
        degrees = membership_degrees(grid, low_knot, medium_knot, high_knot)
        for term in TERMS:
            axis.plot(grid, degrees[term], color=TERM_COLORS[term], linewidth=2, label=term)
        axis.set_title(FEATURE_TITLES[feature])
        axis.set_ylim(-0.02, 1.05)
        axis.set_xlabel(feature)
        axis.set_ylabel("Membership")
        for knot in (low_knot, medium_knot, high_knot):
            axis.axvline(knot, color="#b0b0b0", linewidth=0.6, linestyle=":")

    axes[0].legend(frameon=False)
    figure.suptitle(f"Membership: {membership_name}", y=1.02)
    figure.tight_layout()
    return _save(figure, "membership_functions.png")


def _holdout_metrics() -> pd.DataFrame:
    metrics = pd.read_csv(OUTPUT_DIR / "comparison_metrics.csv")
    holdout = metrics.loc[metrics["view"] == "held_out"].copy()
    holdout["model"] = pd.Categorical(holdout["model"], categories=MODEL_ORDER, ordered=True)
    return holdout.sort_values("model")


def _metrics_comparison() -> Path:
    holdout = _holdout_metrics()
    labels = [MODEL_LABELS[name] for name in holdout["model"]]
    x = np.arange(len(labels))
    width = 0.15
    offsets = np.linspace(-(len(METRIC_ORDER) - 1) / 2, (len(METRIC_ORDER) - 1) / 2, len(METRIC_ORDER))

    figure, axis = plt.subplots(figsize=(12.4, 4.8))
    for offset, (column, title) in zip(offsets, METRIC_ORDER):
        values = holdout[column].to_numpy()
        axis.bar(x + offset * width, values, width=width, label=title)

    axis.set_xticks(x)
    axis.set_xticklabels(labels, rotation=15, ha="right")
    axis.set_ylim(0, 1.05)
    axis.set_ylabel("Score")
    axis.set_title("Holdout comparison")
    axis.legend(frameon=False, ncol=len(METRIC_ORDER), loc="upper center", bbox_to_anchor=(0.5, 1.14))
    axis.grid(axis="y", color="#e0e0e0", linewidth=0.6)
    figure.tight_layout()
    return _save(figure, "metrics_comparison.png")


def _key_metrics() -> Path:
    holdout = _holdout_metrics()
    labels = [MODEL_LABELS[name] for name in holdout["model"]]
    x = np.arange(len(labels))
    width = 0.25
    offsets = (-width, 0.0, width)

    figure, axis = plt.subplots(figsize=(11.2, 4.6))
    for offset, (column, title) in zip(offsets, KEY_METRICS):
        axis.bar(x + offset, holdout[column].to_numpy(), width=width, label=title)

    axis.set_xticks(x)
    axis.set_xticklabels(labels, rotation=15, ha="right")
    axis.set_ylim(0, 1.05)
    axis.set_ylabel("Score")
    axis.set_title("Holdout Accuracy, F1, and ROC-AUC")
    axis.legend(frameon=False)
    axis.grid(axis="y", color="#e0e0e0", linewidth=0.6)
    figure.tight_layout()
    return _save(figure, "accuracy_f1_auc.png")


def _roc_curves() -> Path:
    scores = pd.read_csv(OUTPUT_DIR / "comparison_scores.csv")
    metrics = pd.read_csv(OUTPUT_DIR / "comparison_metrics.csv")
    test = scores.loc[scores["cohort"] == "test"]
    y_true = test[TARGET_CODE_COLUMN].to_numpy()
    auc_by_model = (
        metrics.loc[metrics["view"] == "held_out"].set_index("model")["roc_auc"].to_dict()
    )

    figure, axis = plt.subplots(figsize=(7.2, 5.6))
    for name in MODEL_ORDER:
        false_positive, true_positive, _ = roc_curve(y_true, test[name].to_numpy())
        axis.plot(
            false_positive,
            true_positive,
            color=MODEL_COLORS[name],
            linewidth=2,
            label=f"{MODEL_LABELS[name]} ({auc_by_model[name]:.3f})",
        )

    axis.plot([0, 1], [0, 1], color="#b0b0b0", linewidth=1, linestyle="--")
    axis.set_xlabel("False positive rate")
    axis.set_ylabel("True positive rate")
    axis.set_title("Holdout ROC curves")
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1)
    axis.legend(frameon=False, loc="lower right")
    figure.tight_layout()
    return _save(figure, "roc_curves.png")


def _confusion_matrix(model: str, file_name: str) -> Path:
    metrics = pd.read_csv(OUTPUT_DIR / "comparison_metrics.csv")
    row = metrics.loc[(metrics["model"] == model) & (metrics["view"] == "held_out")].iloc[0]
    matrix = np.array([[int(row.tn), int(row.fp)], [int(row.fn), int(row.tp)]])
    figure, axis = plt.subplots(figsize=(4.8, 4.2))
    display = ConfusionMatrixDisplay(
        confusion_matrix=matrix,
        display_labels=["Negative", "Positive"],
    )
    display.plot(ax=axis, cmap="Blues", colorbar=False, values_format="d")
    axis.set_title(f"{MODEL_LABELS[model]}, holdout")
    figure.tight_layout()
    return _save(figure, file_name)


def _repeated_boxplot() -> Path:
    results = pd.read_csv(OUTPUT_DIR / "repeated_metrics.csv")
    figure, axes = plt.subplots(1, 3, figsize=(13.2, 4.4))
    for axis, metric, title in zip(
        axes,
        ("accuracy", "f1", "roc_auc"),
        ("Accuracy", "F1", "ROC-AUC"),
    ):
        data = [results.loc[results["model"] == model, metric].to_numpy() for model in MODEL_ORDER]
        boxes = axis.boxplot(data, patch_artist=True, widths=0.6, medianprops={"color": "black"})
        for patch, model in zip(boxes["boxes"], MODEL_ORDER):
            patch.set_facecolor(MODEL_COLORS[model])
            patch.set_alpha(0.75)
        axis.set_xticks(range(1, len(MODEL_ORDER) + 1))
        axis.set_xticklabels([MODEL_LABELS[model] for model in MODEL_ORDER], rotation=20, ha="right")
        axis.set_title(title)
        axis.grid(axis="y", color="#e0e0e0", linewidth=0.6)
    figure.suptitle("100 stratified 80/20 splits", y=1.02)
    figure.tight_layout()
    return _save(figure, "repeated_metrics.png")


def _repeated_error_bars() -> Path:
    summary = pd.read_csv(OUTPUT_DIR / "repeated_summary.csv")
    figure, axes = plt.subplots(1, 3, figsize=(13.2, 4.4))
    x = np.arange(len(MODEL_ORDER))
    for axis, metric, title in zip(
        axes,
        ("accuracy", "f1", "roc_auc"),
        ("Accuracy", "F1", "ROC-AUC"),
    ):
        part = summary.loc[summary["metric"] == metric].set_index("model").loc[list(MODEL_ORDER)]
        colors = [MODEL_COLORS[name] for name in MODEL_ORDER]
        axis.bar(x, part["mean"].to_numpy(), yerr=part["sd"].to_numpy(), color=colors, capsize=4, alpha=0.85)
        axis.set_xticks(x)
        axis.set_xticklabels([MODEL_LABELS[name] for name in MODEL_ORDER], rotation=20, ha="right")
        axis.set_ylim(0, 1.05)
        axis.set_title(title)
        axis.grid(axis="y", color="#e0e0e0", linewidth=0.6)
    figure.suptitle("Mean ± SD over 100 splits", y=1.02)
    figure.tight_layout()
    return _save(figure, "repeated_mean_sd.png")


def print_chart_data() -> None:
    """Print the numeric tables behind each chart figure."""
    params = pd.read_csv(OUTPUT_DIR / "membership_params.csv")
    holdout = _holdout_metrics()
    metrics = pd.read_csv(OUTPUT_DIR / "comparison_metrics.csv")
    fuzzy = metrics.loc[(metrics["model"] == "fuzzy") & (metrics["view"] == "held_out")].iloc[0]
    boosted = metrics.loc[(metrics["model"] == "fuzzy_boost") & (metrics["view"] == "held_out")].iloc[0]
    summary = pd.read_csv(OUTPUT_DIR / "repeated_summary.csv")

    membership_name = _membership_name()
    print(f"=== charts/{membership_name}/membership_functions.png ===")
    print(params.to_string(index=False))
    print()

    metric_columns = [column for column, _ in METRIC_ORDER] + ["roc_auc"]
    holdout_table = holdout.loc[:, ["model", *metric_columns]].copy()
    holdout_table["model"] = holdout_table["model"].map(MODEL_LABELS)
    print(f"=== charts/{membership_name}/metrics_comparison.png / accuracy_f1_auc.png (holdout) ===")
    print(holdout_table.round(3).to_string(index=False))
    print()

    print(f"=== charts/{membership_name}/roc_curves.png (holdout ROC-AUC) ===")
    roc_table = holdout.loc[:, ["model", "roc_auc"]].copy()
    roc_table["model"] = roc_table["model"].map(MODEL_LABELS)
    print(roc_table.round(3).to_string(index=False))
    print()

    print("=== confusion matrices (holdout counts) ===")
    for label, row in (("Fuzzy system", fuzzy), ("Fuzzy boost", boosted)):
        print(
            f"{label}: TN={int(row.tn)}  FP={int(row.fp)}  FN={int(row.fn)}  TP={int(row.tp)}  "
            f"accuracy={row.accuracy:.3f}  f1={row.f1:.3f}"
        )
    print()

    repeated_metrics = ("accuracy", "f1", "roc_auc")
    repeated = summary.loc[summary["metric"].isin(repeated_metrics)].copy()
    repeated["model"] = repeated["model"].map(MODEL_LABELS)
    repeated = repeated.loc[:, ["model", "metric", "mean", "sd", "median", "p2_5", "p97_5"]]
    print(f"=== charts/{membership_name}/repeated_metrics.png / repeated_mean_sd.png (100 splits) ===")
    print(repeated.round(3).to_string(index=False))
    print()


def run() -> list[Path]:
    paths = [
        _membership(),
        _metrics_comparison(),
        _key_metrics(),
        _roc_curves(),
        _confusion_matrix("fuzzy", "confusion_matrix_fuzzy.png"),
        _confusion_matrix("fuzzy_boost", "confusion_matrix_fuzzy_boost.png"),
        _repeated_boxplot(),
        _repeated_error_bars(),
    ]
    return paths


if __name__ == "__main__":
    print_chart_data()
    print("Charts:")
    for path in run():
        print(f"  {path}")
