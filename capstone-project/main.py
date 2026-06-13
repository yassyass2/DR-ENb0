"""
Apply the full preprocessing pipeline (src/pre_proc_pipeline.py) to the
APTOS-2019 data and save the model-ready arrays.

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
gets the canonical per-image pipeline (black-border crop -> padded resize ->
green-channel CLAHE -> median blur -> normalise to [0, 1]). SMOTE class
balancing is optional and is applied to the TRAINING split only when
``--smote-train`` is passed. Validation and test keep their natural class
distribution for an honest evaluation.

Outputs (float32 arrays in [0, 1] of shape (N, 224, 224, 1) and integer label
arrays):

    <out-dir>/
        X_train.npy, y_train.npy
        X_val.npy,   y_val.npy
        X_test.npy,  y_test.npy
        splits/
            train.csv, val.csv, test.csv
            metadata.json
"""

import argparse
import json
from pathlib import Path

import kagglehub
import numpy as np
import pandas as pd

from src.pre_proc_pipeline import apply_oversampling, build_dataset
from src.split_metadata import make_dataset_provided_split_metadata

DATASET_SLUG = "mariaherrerot/aptos2019"
DEFAULT_OUT_DIR = Path("data/preprocessed")

# (split name, csv file, image folder) -- folder is nested one level deep.
SPLITS = {
    "train": ("train_1.csv", "train_images/train_images"),
    "val": ("valid.csv", "val_images/val_images"),
    "test": ("test.csv", "test_images/test_images"),
}

ID_COLUMN = "id_code"
LABEL_COLUMN = "diagnosis"
IMAGE_EXTENSION = ".png"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Preprocess APTOS 2019 data.")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--smote-train", action="store_true")
    return parser.parse_args()


def download_raw_dir() -> Path:
    return Path(kagglehub.dataset_download(DATASET_SLUG))


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


def main() -> None:
    args = parse_args()
    raw_dir = download_raw_dir()
    out_dir = args.out_dir
    apply_smote = bool(args.smote_train)

    out_dir.mkdir(parents=True, exist_ok=True)
    splits_dir = out_dir / "splits"
    splits_dir.mkdir(parents=True, exist_ok=True)

    split_metadata = {
        "source": DATASET_SLUG,
        "strategy": "dataset_provided_train_val_test",
        "id_column": ID_COLUMN,
        "label_column": LABEL_COLUMN,
        "image_extension": IMAGE_EXTENSION,
        "smote_train": apply_smote,
        "splits": {},
    }

    for split, (csv_name, folder) in SPLITS.items():
        images_dir = raw_dir / folder
        print(f"\n=== {split.upper()} ===")

        df = load_split_df(raw_dir, csv_name, images_dir)
        df.to_csv(splits_dir / f"{split}.csv", index=False)

        X, y = build_dataset(
            df,
            images_dir,
            id_column=ID_COLUMN,
            label_column=LABEL_COLUMN,
            image_extension=IMAGE_EXTENSION,
            skip_cut_off=True,
        )

        if split == "train" and apply_smote:
            counts = dict(zip(*np.unique(y, return_counts=True)))
            print(f"  Class distribution before SMOTE: {counts}")
            X, y = apply_oversampling(X, y)
            # apply_oversampling returns float32; interpolating two values in
            # [0, 1] stays in [0, 1], so the data is already model-ready.
            X = X.astype(np.float32)
            counts = dict(zip(*np.unique(y, return_counts=True)))
            print(f"  Class distribution after  SMOTE: {counts}")

        np.save(out_dir / f"X_{split}.npy", X)
        np.save(out_dir / f"y_{split}.npy", y)
        split_metadata["splits"][split] = make_dataset_provided_split_metadata(
            split=split,
            csv_name=csv_name,
            image_folder=folder,
            df=df,
            y_saved=y,
            id_column=ID_COLUMN,
            label_column=LABEL_COLUMN,
            smote_applied_to_saved_arrays=(split == "train" and apply_smote),
        )
        print(f"  Saved X_{split} {X.shape} ({X.dtype}) and y_{split} {y.shape}")

    with (splits_dir / "metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(split_metadata, handle, indent=2)

    print(f"\nDone. Preprocessed arrays written to {out_dir.resolve()}")


if __name__ == "__main__":
    main()
