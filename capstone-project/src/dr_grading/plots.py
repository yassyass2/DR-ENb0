"""
Publication-style plots for the DR-grading evaluation (matplotlib + seaborn).

Covers the classification-metrics visualisation from the research proposal
(Table 6.3a): confusion matrix, per-class sensitivity / specificity / ROC-AUC,
and the training curves (loss, accuracy, validation QWK) of a finished run.

The module is both a library and a runnable script:

    python capstone-project/src/plots.py

It reads ``artifacts/test_report.json`` (written by src/model.py) for the
test-set metrics and pulls the per-epoch training history of the most recent
MLflow run (dr_grading/model.py logs every epoch there), then writes the figures to
``artifacts/figures/``:

    confusion_matrix.png   counts + row-normalised heatmaps
    training_curves.png    loss / accuracy / val QWK per epoch
    per_class_metrics.png  sensitivity, specificity and AUC per DR grade
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")  # headless: only file output, no display needed.

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from dr_grading.paths import ARTIFACTS_DIR, PROJECT_DIR

DEFAULT_REPORT = ARTIFACTS_DIR / "test_report.json"
DEFAULT_OUT_DIR = ARTIFACTS_DIR / "figures"
DEFAULT_EXPERIMENT = "aptos-dr-classification"

# Curves plotted from the per-epoch history, with their display labels.
HISTORY_PANELS = (
    (("loss", "val_loss"), "Loss", ("train", "validation")),
    (("accuracy", "val_accuracy"), "Accuracy", ("train", "validation")),
    (("val_qwk",), "Quadratic Weighted Kappa", ("validation",)),
)


def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: list[str],
    out_path: Path,
) -> Path:
    """
    Confusion-matrix heatmaps: absolute counts and row-normalised side by side.

    The row-normalised panel divides each row by its class support, so the
    diagonal reads directly as per-class sensitivity -- the per-grade view the
    proposal calls for on an imbalanced test set.
    """
    cm = np.asarray(cm)
    with np.errstate(invalid="ignore"):
        cm_norm = cm / cm.sum(axis=1, keepdims=True)

    labels = [f"{i} {name}" for i, name in enumerate(class_names)]
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels,
        cbar_kws={"label": "count"},
        ax=axes[0],
    )
    axes[0].set_title("Confusion matrix (counts)")

    sns.heatmap(
        cm_norm,
        annot=True,
        fmt=".2f",
        cmap="Blues",
        vmin=0.0,
        vmax=1.0,
        xticklabels=labels,
        yticklabels=labels,
        cbar_kws={"label": "fraction of true grade"},
        ax=axes[1],
    )
    axes[1].set_title("Confusion matrix (row-normalised)")

    for ax in axes:
        ax.set_xlabel("Predicted grade")
        ax.set_ylabel("True grade")
        ax.tick_params(axis="x", rotation=30)
        ax.tick_params(axis="y", rotation=0)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def plot_per_class_metrics(report: dict[str, Any], out_path: Path) -> Path:
    """Grouped bars of sensitivity / specificity / ROC-AUC per DR grade."""
    per_class = report["per_class"]
    grades = sorted(per_class, key=int)
    labels = [f"{c} {per_class[c]['name']}" for c in grades]
    metrics = ("sensitivity", "specificity", "auc")

    x = np.arange(len(grades))
    width = 0.26
    fig, ax = plt.subplots(figsize=(10, 5.5))
    palette = sns.color_palette("deep", len(metrics))

    for i, metric in enumerate(metrics):
        values = [per_class[c][metric] for c in grades]
        bars = ax.bar(x + (i - 1) * width, values, width, label=metric, color=palette[i])
        ax.bar_label(bars, fmt="%.2f", fontsize=8, padding=2)

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0.0, 1.12)
    ax.set_ylabel("Score")
    ax.set_xlabel("DR grade")
    ax.set_title(
        f"Per-class test metrics "
        f"(QWK={report['qwk']:.3f}, accuracy={report['accuracy']:.3f}, "
        f"n={report['num_samples']})"
    )
    ax.legend(loc="lower right")
    sns.despine(fig)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def plot_training_curves(history: dict[str, list[float]], out_path: Path) -> Path:
    """
    Loss, accuracy and validation QWK per epoch.

    The epoch with the best validation QWK (the checkpoint that becomes
    ``best_model.keras``) is marked in every panel.
    """
    best_epoch = (
        int(np.argmax(history["val_qwk"])) if history.get("val_qwk") else None
    )

    fig, axes = plt.subplots(1, len(HISTORY_PANELS), figsize=(16, 4.8))
    palette = sns.color_palette("deep")

    for ax, (keys, title, labels) in zip(axes, HISTORY_PANELS):
        for i, (key, label) in enumerate(zip(keys, labels)):
            values = history.get(key)
            if not values:
                continue
            epochs = np.arange(1, len(values) + 1)
            ax.plot(epochs, values, label=label, color=palette[i], linewidth=1.8)
        if best_epoch is not None:
            ax.axvline(
                best_epoch + 1,
                color="grey",
                linestyle="--",
                linewidth=1.0,
                label=f"best val QWK (epoch {best_epoch + 1})",
            )
        ax.set_title(title)
        ax.set_xlabel("Epoch")
        ax.legend(fontsize=8)
    sns.despine(fig)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def load_history_from_mlflow(
    experiment_name: str = DEFAULT_EXPERIMENT,
    run_name: str | None = None,
) -> dict[str, list[float]]:
    """
    Reconstruct the per-epoch training history from the most recent MLflow run.

    ``dr_grading/model.py`` logs every epoch of every history key as a step metric, so
    the metric history in the tracking database is exactly the merged Keras
    history. Returns ``{metric: [value per epoch]}`` for the latest finished
    run (optionally filtered by run name).
    """
    import mlflow
    from dr_grading.tracking import configure_mlflow

    configure_mlflow(experiment_name, PROJECT_DIR)
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is None:
        raise RuntimeError(f"MLflow experiment not found: {experiment_name}")

    filter_string = (
        f"attributes.run_name = '{run_name}'" if run_name is not None else ""
    )
    runs = mlflow.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string=filter_string,
        order_by=["attributes.start_time DESC"],
        max_results=1,
        output_format="list",
    )
    if not runs:
        raise RuntimeError(
            f"No MLflow runs found in experiment '{experiment_name}'"
            + (f" with run name '{run_name}'" if run_name else "")
        )
    run = runs[0]

    client = mlflow.tracking.MlflowClient()
    history: dict[str, list[float]] = {}
    for key in run.data.metrics:
        points = client.get_metric_history(run.info.run_id, key)
        points = sorted(points, key=lambda m: m.step)
        if len(points) > 1:  # per-epoch curves only; skip one-off test metrics.
            history[key] = [float(p.value) for p in points]
    return history


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot evaluation metrics and training curves."
    )
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--experiment", default=DEFAULT_EXPERIMENT)
    parser.add_argument(
        "--run-name",
        default=None,
        help="MLflow run to take the training history from (default: latest).",
    )
    parser.add_argument(
        "--skip-history",
        action="store_true",
        help="Skip the training-curves figure (no MLflow lookup).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    sns.set_theme(style="white", context="paper")

    with args.report.open("r", encoding="utf-8") as handle:
        report = json.load(handle)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    written = [
        plot_confusion_matrix(
            np.asarray(report["confusion_matrix"]),
            report["class_names"],
            args.out_dir / "confusion_matrix.png",
        ),
        plot_per_class_metrics(report, args.out_dir / "per_class_metrics.png"),
    ]

    if not args.skip_history:
        history = load_history_from_mlflow(args.experiment, args.run_name)
        written.append(
            plot_training_curves(history, args.out_dir / "training_curves.png")
        )

    for path in written:
        print(f"Wrote {path}")


if __name__ == "__main__":
    main()
