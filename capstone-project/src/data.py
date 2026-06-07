from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class DatasetSummary:
    train_samples: int
    val_samples: int
    test_samples: int
    image_shape: tuple[int, ...]
    class_distribution: dict[str, dict[int, int]]


@dataclass(frozen=True)
class PreprocessedDataset:
    X_train: np.ndarray
    y_train: np.ndarray
    X_val: np.ndarray
    y_val: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    summary: DatasetSummary


REQUIRED_ARRAYS = (
    "X_train",
    "y_train",
    "X_val",
    "y_val",
    "X_test",
    "y_test",
)


def load_preprocessed_data(data_dir: str | Path) -> PreprocessedDataset:
    data_dir = Path(data_dir)
    arrays = {name: _load_array(data_dir, name) for name in REQUIRED_ARRAYS}

    _validate_alignment(arrays["X_train"], arrays["y_train"], "train")
    _validate_alignment(arrays["X_val"], arrays["y_val"], "val")
    _validate_alignment(arrays["X_test"], arrays["y_test"], "test")

    summary = DatasetSummary(
        train_samples=int(arrays["X_train"].shape[0]),
        val_samples=int(arrays["X_val"].shape[0]),
        test_samples=int(arrays["X_test"].shape[0]),
        image_shape=tuple(arrays["X_train"].shape[1:]),
        class_distribution={
            "train": _class_distribution(arrays["y_train"]),
            "val": _class_distribution(arrays["y_val"]),
            "test": _class_distribution(arrays["y_test"]),
        },
    )

    return PreprocessedDataset(summary=summary, **arrays)


def _load_array(data_dir: Path, name: str) -> np.ndarray:
    path = data_dir / f"{name}.npy"
    if not path.exists():
        raise FileNotFoundError(f"Missing preprocessed array: {path}")
    return np.load(path)


def _validate_alignment(X: np.ndarray, y: np.ndarray, split: str) -> None:
    if X.shape[0] != y.shape[0]:
        raise ValueError(
            f"{split} split is misaligned: X has {X.shape[0]} samples, "
            f"y has {y.shape[0]} labels"
        )


def _class_distribution(y: np.ndarray) -> dict[int, int]:
    labels, counts = np.unique(y, return_counts=True)
    return {
        int(label): int(count)
        for label, count in zip(labels.tolist(), counts.tolist(), strict=True)
    }
