"""Apply the full preprocessing pipeline (dr_grading/pre_proc_pipeline.py) to the
APTOS-2019 data and save the model-ready arrays to ``data/preprocessed``.

Data source: the dataset is pulled with ``kagglehub`` rather than the local
``data/raw`` folder. ``data/raw`` is only a partial copy (1463 of the 2930
train images), whereas kagglehub provides the complete set (verified
byte-identical images and CSVs). Using kagglehub recovers the ~1467 missing
train images, which include most of the minority-class samples.

The data is already split into train / validation / test, each with its own
CSV and image folder:

    <dataset>/
        train_1.csv   ->  train_images/train_images/<id_code>.png
        valid.csv     ->  val_images/val_images/<id_code>.png
        test.csv      ->  test_images/test_images/<id_code>.png

Because the split already exists, we do NOT re-split here. Instead each split
gets the paper's per-image pipeline (resize 224x224 -> CLAHE -> median blur
(kernel 3) -> normalise to [0, 1]), and SMOTE class balancing is applied to the
TRAINING split only. Validation and test keep their natural class distribution
for an honest evaluation.

Outputs (float32 arrays in [0, 1] of shape (N, 224, 224, 3) and integer label
arrays):

    data/preprocessed/
        X_train.npy, y_train.npy   (SMOTE-balanced)
        X_val.npy,   y_val.npy
        X_test.npy,  y_test.npy
"""

from __future__ import annotations

import argparse
from datetime import UTC, datetime
import json
from pathlib import Path
import subprocess
from typing import Any

import numpy as np
import pandas as pd

from dr_grading import __version__
from dr_grading.paths import DEFAULT_PREPROCESSED_DIR

# (split name, csv file, image folder) -- folder is nested one level deep.
SPLITS = {
    "train": ("train_1.csv", "train_images/train_images"),
    "val": ("valid.csv", "val_images/val_images"),
    "test": ("test.csv", "test_images/test_images"),
}

ID_COLUMN = "id_code"
LABEL_COLUMN = "diagnosis"
IMAGE_EXTENSION = ".png"


def load_split_df(raw_dir: Path, csv_name: str, images_dir: Path) -> pd.DataFrame:
    """Load a split CSV and keep only the rows whose image is present on disk."""
    df = pd.read_csv(raw_dir / csv_name)

    exists = df[ID_COLUMN].apply(
        lambda code: (images_dir / f"{code}{IMAGE_EXTENSION}").exists()
    )
    n_missing = int((~exists).sum())
    if n_missing:
        print(f"  {n_missing} rows have no image on disk and are skipped.")

    return df[exists].reset_index(drop=True)


def run_preprocessing(
    output_dir: str | Path | None = None,
    dataset_dir: str | Path | None = None,
) -> Path:
    """Build preprocessed train/validation/test arrays and return the output dir."""
    from dr_grading.pre_proc_pipeline import apply_oversampling, build_dataset

    if dataset_dir is None:
        import kagglehub

        raw_dir = Path(kagglehub.dataset_download("mariaherrerot/aptos2019"))
    else:
        raw_dir = Path(dataset_dir)

    out_dir = Path(output_dir) if output_dir is not None else DEFAULT_PREPROCESSED_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    split_summaries: dict[str, dict[str, Any]] = {}

    for split, (csv_name, folder) in SPLITS.items():
        images_dir = raw_dir / folder
        print(f"\n=== {split.upper()} ===")

        df = load_split_df(raw_dir, csv_name, images_dir)

        # Per-image pipeline (paper procedure): resize -> CLAHE -> median blur
        # (kernel 3) -> normalise to [0, 1]. skip_cut_off=True -> no cut-off
        # filtering, matching the paper (which crops/filters nothing).
        X, y = build_dataset(
            df,
            images_dir,
            id_column=ID_COLUMN,
            label_column=LABEL_COLUMN,
            image_extension=IMAGE_EXTENSION,
            skip_cut_off=True,
        )

        # SMOTE balancing on the training split only.
        if split == "train":
            counts = dict(zip(*np.unique(y, return_counts=True)))
            print(f"  Class distribution before SMOTE: {counts}")
            split_summaries[split] = {
                "before_smote": _class_distribution(y),
            }
            X, y = apply_oversampling(X, y)
            # apply_oversampling returns float32; interpolating two values in
            # [0, 1] stays in [0, 1], so the data is already model-ready.
            X = X.astype(np.float32)
            counts = dict(zip(*np.unique(y, return_counts=True)))
            print(f"  Class distribution after  SMOTE: {counts}")
            split_summaries[split]["after_smote"] = _class_distribution(y)
            split_summaries[split]["samples"] = int(len(y))
        else:
            split_summaries[split] = {
                "distribution": _class_distribution(y),
                "samples": int(len(y)),
            }

        np.save(out_dir / f"X_{split}.npy", X)
        np.save(out_dir / f"y_{split}.npy", y)
        print(f"  Saved X_{split} {X.shape} ({X.dtype}) and y_{split} {y.shape}")

    manifest_path = write_preprocessing_manifest(
        raw_dir,
        out_dir,
        split_summaries,
        git_commit=_current_git_commit(),
        package_version=__version__,
    )
    print(f"\nDone. Preprocessed arrays written to {out_dir.resolve()}")
    print(f"Wrote preprocessing manifest to {manifest_path}")
    return out_dir


def write_preprocessing_manifest(
    dataset_dir: str | Path,
    output_dir: str | Path,
    split_summaries: dict[str, dict[str, Any]],
    git_commit: str | None = None,
    package_version: str = __version__,
) -> Path:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest = {
        "created_at": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "dataset": "mariaherrerot/aptos2019",
        "dataset_dir": str(Path(dataset_dir).resolve()),
        "output_dir": str(output_dir.resolve()),
        "code": {
            "git_commit": git_commit or "unknown",
            "package_version": package_version,
        },
        "preprocessing": {
            "image_size": 224,
            "clahe": {
                "color_space": "LAB",
                "channel": "lightness",
                "clip_limit": 2.0,
                "tile_grid_size": [8, 8],
            },
            "denoise": {
                "method": "median",
                "kernel_size": 3,
            },
            "normalization": "[0, 1] float32",
            "cut_off_filtering": False,
            "smote_applied_to": "train",
        },
        "splits": split_summaries,
    }

    path = output_dir / "preprocessing_manifest.json"
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return path


def _class_distribution(y: np.ndarray) -> dict[int, int]:
    labels, counts = np.unique(y, return_counts=True)
    return {
        int(label): int(count)
        for label, count in zip(labels.tolist(), counts.tolist(), strict=True)
    }


def _current_git_commit() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return "unknown"
    return result.stdout.strip() or "unknown"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Preprocess the APTOS 2019 dataset into model-ready arrays."
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_PREPROCESSED_DIR)
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=None,
        help="Existing APTOS dataset directory; defaults to kagglehub download.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    run_preprocessing(output_dir=args.output_dir, dataset_dir=args.dataset_dir)


if __name__ == "__main__":
    main()
