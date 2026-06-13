"""
Evaluation metrics for the APTOS-2019 diabetic-retinopathy classifier.

The metrics here are the ones that matter for graded DR severity (ordinal,
heavily imbalanced):

  - Quadratic Weighted Kappa (QWK) -- the official APTOS competition metric;
    rewards predictions that are *close* to the true grade and penalises
    far-off mistakes quadratically.
  - Per-class sensitivity (recall / true-positive rate) and specificity
    (true-negative rate) -- read straight off the confusion matrix in a
    one-vs-rest fashion, so we can see how well each individual grade is
    detected despite the class imbalance.
  - Per-class and macro ROC-AUC (one-vs-rest) -- threshold-independent ranking
    quality from the predicted probabilities.

All functions take plain numpy arrays / sequences so they can be reused from
``src/model.py`` (after training) or from a notebook. ``main()`` lets the module
be run standalone against a saved ``.keras`` model.
"""

from __future__ import annotations

import itertools
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from sklearn.metrics import (
    cohen_kappa_score,
    confusion_matrix,
    roc_auc_score,
)

# APTOS-2019 diagnosis grades (0-4).
DR_CLASS_NAMES = (
    "No DR",
    "Mild",
    "Moderate",
    "Severe",
    "Proliferative",
)


def quadratic_weighted_kappa(
    y_true: Sequence[int] | np.ndarray,
    y_pred: Sequence[int] | np.ndarray,
) -> float:
    """
    Quadratic Weighted Kappa between true and predicted grades.

    Returns a value in [-1, 1]; 1 is perfect agreement, 0 is chance-level.
    """
    return float(cohen_kappa_score(y_true, y_pred, weights="quadratic"))


def scores_to_grades(
    scores: Sequence[float] | np.ndarray,
    thresholds: Sequence[float],
    num_classes: int = 5,
) -> np.ndarray:
    """Convert ordinal regression scores to integer grades using thresholds."""
    thresholds = np.asarray(thresholds, dtype=np.float32)
    if len(thresholds) != num_classes - 1:
        raise ValueError(
            f"Expected {num_classes - 1} thresholds for {num_classes} classes, "
            f"got {len(thresholds)}"
        )
    if np.any(np.diff(thresholds) <= 0):
        raise ValueError("Regression thresholds must be strictly increasing")

    grades = np.digitize(np.asarray(scores, dtype=np.float32).reshape(-1), thresholds)
    return np.clip(grades, 0, num_classes - 1).astype(int)


def score_to_class_affinity(
    scores: Sequence[float] | np.ndarray,
    num_classes: int = 5,
    sigma: float = 0.75,
) -> np.ndarray:
    """
    Convert scalar severity scores into class-affinity values for AUC reporting.

    These are not softmax probabilities; they are normalized ordinal affinities
    based on distance to class centers. The hard grades remain threshold-based.
    """
    scores = np.asarray(scores, dtype=np.float32).reshape(-1, 1)
    centers = np.arange(num_classes, dtype=np.float32).reshape(1, -1)
    affinity = np.exp(-0.5 * ((scores - centers) / sigma) ** 2)
    return affinity / affinity.sum(axis=1, keepdims=True)


def optimize_regression_thresholds(
    y_true: Sequence[int] | np.ndarray,
    scores: Sequence[float] | np.ndarray,
    num_classes: int = 5,
    grid_size: int = 41,
) -> tuple[list[float], float]:
    """
    Tune ordered score thresholds on validation data by maximizing QWK.

    The search is a deterministic grid over the score range. It is cheap at
    validation-set scale and keeps the final test evaluation independent.
    """
    y_true = np.asarray(y_true).astype(int)
    scores = np.asarray(scores, dtype=np.float32).reshape(-1)
    lower = float(min(scores.min(), 0.0))
    upper = float(max(scores.max(), num_classes - 1.0))
    grid = np.linspace(lower, upper, grid_size)

    best_thresholds = [float(i + 0.5) for i in range(num_classes - 1)]
    best_qwk = quadratic_weighted_kappa(
        y_true, scores_to_grades(scores, best_thresholds, num_classes)
    )
    for thresholds in itertools.combinations(grid[1:-1], num_classes - 1):
        y_pred = scores_to_grades(scores, thresholds, num_classes)
        qwk = quadratic_weighted_kappa(y_true, y_pred)
        if qwk > best_qwk:
            best_qwk = qwk
            best_thresholds = [float(t) for t in thresholds]

    return best_thresholds, float(best_qwk)


def per_class_sensitivity_specificity(
    y_true: Sequence[int] | np.ndarray,
    y_pred: Sequence[int] | np.ndarray,
    num_classes: int,
) -> dict[int, dict[str, float]]:
    """
    Per-class sensitivity and specificity in a one-vs-rest scheme.

    For each class ``c`` the confusion matrix gives:

        sensitivity = TP / (TP + FN)   (how many true c's we found)
        specificity = TN / (TN + FP)   (how many non-c's we correctly rejected)

    Returns ``{class_index: {"sensitivity", "specificity", "support"}}``.
    Sensitivity/specificity are ``nan`` when the denominator is zero.
    """
    cm = confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))
    total = int(cm.sum())

    result: dict[int, dict[str, float]] = {}
    for c in range(num_classes):
        tp = int(cm[c, c])
        fn = int(cm[c, :].sum()) - tp
        fp = int(cm[:, c].sum()) - tp
        tn = total - tp - fn - fp

        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
        specificity = tn / (tn + fp) if (tn + fp) > 0 else float("nan")

        result[c] = {
            "sensitivity": float(sensitivity),
            "specificity": float(specificity),
            "support": tp + fn,
        }
    return result


def per_class_auc(
    y_true: Sequence[int] | np.ndarray,
    y_proba: np.ndarray,
    num_classes: int,
) -> dict[str, Any]:
    """
    One-vs-rest ROC-AUC per class, plus the macro average.

    ``y_proba`` is the softmax output of shape ``(N, num_classes)``. A class
    whose AUC is undefined (only one label present in the one-vs-rest split)
    is reported as ``nan`` and excluded from the macro average.
    """
    y_true = np.asarray(y_true).astype(int)
    y_proba = np.asarray(y_proba)
    y_onehot = np.eye(num_classes)[y_true]

    per_class: dict[int, float] = {}
    for c in range(num_classes):
        # roc_auc_score needs both positives and negatives present.
        if len(np.unique(y_onehot[:, c])) < 2:
            per_class[c] = float("nan")
        else:
            per_class[c] = float(roc_auc_score(y_onehot[:, c], y_proba[:, c]))

    valid = [v for v in per_class.values() if not np.isnan(v)]
    macro = float(np.mean(valid)) if valid else float("nan")
    return {"per_class": per_class, "macro": macro}


def evaluate_classification(
    y_true: Sequence[int] | np.ndarray,
    y_pred: Sequence[int] | np.ndarray,
    y_proba: np.ndarray,
    num_classes: int = 5,
    class_names: Sequence[str] | None = None,
) -> dict[str, Any]:
    """
    Compute the full evaluation report: QWK, accuracy, per-class
    sensitivity/specificity/AUC, their macro averages, and the confusion
    matrix.

    Returns a JSON-serialisable dict (nested numpy values cast to Python types).
    """
    class_names = list(class_names) if class_names is not None else list(DR_CLASS_NAMES)
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    y_proba = np.asarray(y_proba)

    qwk = quadratic_weighted_kappa(y_true, y_pred)
    accuracy = float((y_true == y_pred).mean())
    sens_spec = per_class_sensitivity_specificity(y_true, y_pred, num_classes)
    auc = per_class_auc(y_true, y_proba, num_classes)
    cm = confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))

    per_class = {
        c: {
            "name": class_names[c] if c < len(class_names) else str(c),
            "sensitivity": sens_spec[c]["sensitivity"],
            "specificity": sens_spec[c]["specificity"],
            "auc": auc["per_class"][c],
            "support": sens_spec[c]["support"],
        }
        for c in range(num_classes)
    }

    def _macro(key: str) -> float:
        vals = [v[key] for v in per_class.values() if not np.isnan(v[key])]
        return float(np.mean(vals)) if vals else float("nan")

    return {
        "qwk": qwk,
        "accuracy": accuracy,
        "macro_sensitivity": _macro("sensitivity"),
        "macro_specificity": _macro("specificity"),
        "macro_auc": auc["macro"],
        "per_class": per_class,
        "confusion_matrix": cm.tolist(),
        "class_names": class_names,
        "num_samples": int(len(y_true)),
    }


def metrics_for_logging(report: dict[str, Any], prefix: str = "test") -> dict[str, float]:
    """
    Flatten an ``evaluate_classification`` report into scalar metrics suitable
    for ``mlflow.log_metrics`` (drops ``nan`` values, which MLflow rejects).
    """
    flat: dict[str, float] = {
        f"{prefix}_qwk": report["qwk"],
        f"{prefix}_accuracy": report["accuracy"],
        f"{prefix}_macro_sensitivity": report["macro_sensitivity"],
        f"{prefix}_macro_specificity": report["macro_specificity"],
        f"{prefix}_macro_auc": report["macro_auc"],
    }
    for c, vals in report["per_class"].items():
        flat[f"{prefix}_class{c}_sensitivity"] = vals["sensitivity"]
        flat[f"{prefix}_class{c}_specificity"] = vals["specificity"]
        flat[f"{prefix}_class{c}_auc"] = vals["auc"]
    return {k: float(v) for k, v in flat.items() if v == v}  # v == v drops nan


def format_report(report: dict[str, Any]) -> str:
    """Render an ``evaluate_classification`` report as a readable text block."""
    names = report["class_names"]
    lines: list[str] = []
    lines.append("=" * 64)
    lines.append(f"Evaluation on {report['num_samples']} samples")
    lines.append("=" * 64)
    lines.append(f"  Quadratic Weighted Kappa : {report['qwk']:.4f}")
    lines.append(f"  Accuracy                 : {report['accuracy']:.4f}")
    lines.append(f"  Macro sensitivity        : {report['macro_sensitivity']:.4f}")
    lines.append(f"  Macro specificity        : {report['macro_specificity']:.4f}")
    lines.append(f"  Macro ROC-AUC (OvR)      : {report['macro_auc']:.4f}")
    lines.append("")
    lines.append("  Per-class metrics:")
    header = f"    {'grade':<16}{'sens':>8}{'spec':>8}{'auc':>8}{'n':>8}"
    lines.append(header)
    lines.append("    " + "-" * (len(header) - 4))
    for c, vals in report["per_class"].items():
        name = f"{c} {vals['name']}"
        lines.append(
            f"    {name:<16}"
            f"{vals['sensitivity']:>8.3f}"
            f"{vals['specificity']:>8.3f}"
            f"{vals['auc']:>8.3f}"
            f"{vals['support']:>8d}"
        )
    lines.append("")
    lines.append("  Confusion matrix (rows = true grade, cols = predicted):")
    col_header = "      " + "".join(f"{j:>6}" for j in range(len(report["confusion_matrix"])))
    lines.append(col_header)
    for i, row in enumerate(report["confusion_matrix"]):
        lines.append(f"    {i} " + "".join(f"{v:>6d}" for v in row))
    lines.append("=" * 64)
    return "\n".join(lines)


def evaluate_predictions(
    y_true: Sequence[int] | np.ndarray,
    y_proba: np.ndarray,
    num_classes: int = 5,
    class_names: Sequence[str] | None = None,
) -> dict[str, Any]:
    """
    Convenience wrapper: turn predicted probabilities into hard predictions
    (argmax) and run the full evaluation.
    """
    y_pred = np.asarray(y_proba).argmax(axis=1)
    return evaluate_classification(
        y_true, y_pred, y_proba, num_classes=num_classes, class_names=class_names
    )


def main() -> None:
    """
    Standalone evaluation: load a saved ``.keras`` model and the preprocessed
    test split, then print the metric report.

        python src/evaluate.py --model artifacts/best_model.keras
    """
    import argparse
    import sys

    project_dir = Path(__file__).resolve().parents[1]
    if str(project_dir) not in sys.path:
        sys.path.insert(0, str(project_dir))

    from src.data import load_preprocessed_data

    parser = argparse.ArgumentParser(description="Evaluate a trained DR model.")
    parser.add_argument(
        "--model",
        type=Path,
        default=project_dir / "artifacts" / "best_model.keras",
        help="Path to the saved .keras model.",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=project_dir / "data" / "preprocessed",
        help="Directory with the preprocessed .npy arrays.",
    )
    parser.add_argument("--num-classes", type=int, default=5)
    parser.add_argument(
        "--report-out",
        type=Path,
        default=None,
        help="Optional path to write the report as JSON.",
    )
    args = parser.parse_args()

    if not args.model.exists():
        raise FileNotFoundError(
            f"Model not found: {args.model}. Train one first with src/model.py."
        )

    import keras

    dataset = load_preprocessed_data(args.data_dir)
    model = keras.models.load_model(args.model)

    proba = model.predict(dataset.X_test, verbose=0)
    report = evaluate_predictions(
        dataset.y_test, proba, num_classes=args.num_classes
    )
    print(format_report(report))

    if args.report_out is not None:
        args.report_out.parent.mkdir(parents=True, exist_ok=True)
        with args.report_out.open("w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2)
        print(f"Wrote report to {args.report_out}")


if __name__ == "__main__":
    main()
