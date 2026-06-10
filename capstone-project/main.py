"""
Apply the full preprocessing pipeline (src/pre_proc_pipeline.py) to the
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
gets the per-image pipeline (cut-off filtering -> crop -> resize 224x224 ->
noise reduction -> green channel + CLAHE -> normalise to [0, 1]), and SMOTE
class balancing is applied to the TRAINING split only. Validation and test keep
their natural class distribution for an honest evaluation.

Outputs (float32 arrays in [0, 1] of shape (N, 224, 224, 3) and integer label
arrays):

    data/preprocessed/
        X_train.npy, y_train.npy   (SMOTE-balanced)
        X_val.npy,   y_val.npy
        X_test.npy,  y_test.npy
"""

from pathlib import Path

import kagglehub
import numpy as np
import pandas as pd

from src.pre_proc_pipeline import build_dataset, apply_oversampling

RAW_DIR = Path(kagglehub.dataset_download("mariaherrerot/aptos2019"))
OUT_DIR = Path("data/preprocessed")

# (split name, csv file, image folder) -- folder is nested one level deep.
SPLITS = {
    "train": ("train_1.csv", "train_images/train_images"),
    "val": ("valid.csv", "val_images/val_images"),
    "test": ("test.csv", "test_images/test_images"),
}

ID_COLUMN = "id_code"
LABEL_COLUMN = "diagnosis"
IMAGE_EXTENSION = ".png"


def load_split_df(csv_name: str, images_dir: Path) -> pd.DataFrame:
    """Load a split CSV and keep only the rows whose image is present on disk."""
    df = pd.read_csv(RAW_DIR / csv_name)

    exists = df[ID_COLUMN].apply(
        lambda code: (images_dir / f"{code}{IMAGE_EXTENSION}").exists()
    )
    n_missing = int((~exists).sum())
    if n_missing:
        print(f"  {n_missing} rows have no image on disk and are skipped.")

    return df[exists].reset_index(drop=True)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    for split, (csv_name, folder) in SPLITS.items():
        images_dir = RAW_DIR / folder
        print(f"\n=== {split.upper()} ===")

        df = load_split_df(csv_name, images_dir)

        # Per-image pipeline: cut-off filter -> crop -> resize -> noise
        # reduction -> green + CLAHE -> normalise to [0, 1].
        # skip_cut_off=False -> cut-off filtering IS applied (see
        # pre_proc_pipeline.preprocess_image_path for the flag semantics).
        X, y = build_dataset(
            df,
            images_dir,
            id_column=ID_COLUMN,
            label_column=LABEL_COLUMN,
            image_extension=IMAGE_EXTENSION,
            skip_cut_off=False,
        )

        # SMOTE balancing on the training split only.
        if split == "train":
            counts = dict(zip(*np.unique(y, return_counts=True)))
            print(f"  Class distribution before SMOTE: {counts}")
            X, y = apply_oversampling(X, y)
            # apply_oversampling returns float32; interpolating two values in
            # [0, 1] stays in [0, 1], so the data is already model-ready.
            X = X.astype(np.float32)
            counts = dict(zip(*np.unique(y, return_counts=True)))
            print(f"  Class distribution after  SMOTE: {counts}")

        np.save(OUT_DIR / f"X_{split}.npy", X)
        np.save(OUT_DIR / f"y_{split}.npy", y)
        print(f"  Saved X_{split} {X.shape} ({X.dtype}) and y_{split} {y.shape}")

    print(f"\nDone. Preprocessed arrays written to {OUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
