"""
Complete preprocessing pipeline for the APTOS 2019 diabetic-retinopathy
dataset, targeting an EfficientNet-B0 classifier with 224 x 224 x 3 input.

Pipeline:

  Optional quality filtering
  -> crop black border
  -> resize to 224 x 224
  -> mild noise reduction
  -> green channel extraction
  -> CLAHE
  -> stack to 3 channels
  -> SMOTE on training data only

Images are kept as uint8 in the [0, 255] range. This matches the usual
EfficientNet input flow where model preprocessing/normalisation happens later.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

from src.preprocessing import (
    crop_black_border,
    reduce_fundus_noise,
    apply_oversampling,
)

DEFAULT_IMAGE_SIZE = 224  # EfficientNet-B0 input resolution


# ---------------------------------------------------------------------------
# Step 1: cut-off / quality filtering
# ---------------------------------------------------------------------------
def create_retina_mask(image: np.ndarray, threshold: int = 15) -> np.ndarray:
    """
    Create a binary mask of the bright retina against the black background.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    mask = (gray > threshold).astype(np.uint8)

    kernel = np.ones((7, 7), np.uint8)

    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    return mask


def _border_retina_ratios(
    mask: np.ndarray,
    border_width_ratio: float = 0.02,
) -> dict[str, float]:
    """
    Calculate the fraction of retina pixels sitting on each border band.
    """
    height, width = mask.shape

    border_w = max(1, int(width * border_width_ratio))
    border_h = max(1, int(height * border_width_ratio))

    return {
        "left": float(mask[:, :border_w].mean()),
        "right": float(mask[:, width - border_w:].mean()),
        "top": float(mask[:border_h, :].mean()),
        "bottom": float(mask[height - border_h:, :].mean()),
    }


def is_cut_off(
    image: np.ndarray,
    threshold: int = 15,
    border_width_ratio: float = 0.01,
    max_side_ratio: float = 0.30,
    min_cut_sides: int = 3,
) -> bool:
    """
    Decide whether a retina image is genuinely cut off / over-cropped.

    An image is flagged when the fundus touches at least min_cut_sides of the
    four image-border bands by more than max_side_ratio.
    """
    mask = create_retina_mask(image, threshold=threshold)
    ratios = _border_retina_ratios(mask, border_width_ratio=border_width_ratio)

    n_cut_sides = sum(value > max_side_ratio for value in ratios.values())

    return n_cut_sides >= min_cut_sides


# ---------------------------------------------------------------------------
# Steps 2-6: per-image transform
# ---------------------------------------------------------------------------
def extract_green_channel_clahe(
    image: np.ndarray,
    clip_limit: float = 2.0,
    tile_grid_size: tuple[int, int] = (8, 8),
    to_three_channels: bool = True,
) -> np.ndarray:
    """
    Extract the green channel and apply CLAHE.

    OpenCV loads images as BGR, so the green channel is image[:, :, 1].

    By default, the enhanced green channel is copied into 3 channels so
    EfficientNet-B0 still receives a 224 x 224 x 3 input.
    """
    green = image[:, :, 1]

    clahe = cv2.createCLAHE(
        clipLimit=clip_limit,
        tileGridSize=tile_grid_size,
    )

    green = clahe.apply(green)

    if to_three_channels:
        return cv2.merge([green, green, green]).astype(np.uint8)

    return green.astype(np.uint8)


def preprocess_image(
    image: np.ndarray,
    image_size: int = DEFAULT_IMAGE_SIZE,
    denoise_method: str = "bilateral",
    clip_limit: float = 2.0,
    tile_grid_size: tuple[int, int] = (8, 8),
) -> np.ndarray:
    """
    Apply the per-image pipeline to one already-loaded BGR image.

    Order:
        crop black border
        -> resize
        -> noise reduction
        -> green channel
        -> CLAHE
        -> 3-channel uint8 output
    """
    image = crop_black_border(image)

    image = cv2.resize(
        image,
        (image_size, image_size),
        interpolation=cv2.INTER_AREA,
    )

    image = reduce_fundus_noise(
        image,
        method=denoise_method,
    )

    image = extract_green_channel_clahe(
        image,
        clip_limit=clip_limit,
        tile_grid_size=tile_grid_size,
        to_three_channels=True,
    )

    return image.astype(np.uint8)


def preprocess_image_path(
    image_path: str | Path,
    image_size: int = DEFAULT_IMAGE_SIZE,
    skip_cut_off: bool = True,
    denoise_method: str = "bilateral",
    **clahe_kwargs,
) -> np.ndarray | None:
    """
    Read an image from disk and run the per-image preprocessing pipeline.

    Returns:
        uint8 image of shape (image_size, image_size, 3), or None if the image
        could not be read or was filtered out as cut off.

    Important:
        skip_cut_off=True means cut-off filtering is skipped.
        skip_cut_off=False means cut-off filtering is applied.
    """
    image = cv2.imread(str(image_path))

    if image is None:
        print(f"Could not read image: {image_path}")
        return None

    if not skip_cut_off and is_cut_off(image):
        return None

    return preprocess_image(
        image,
        image_size=image_size,
        denoise_method=denoise_method,
        **clahe_kwargs,
    )


# ---------------------------------------------------------------------------
# Dataset assembly
# ---------------------------------------------------------------------------
def build_dataset(
    df,
    images_dir: str | Path,
    id_column: str = "id_code",
    label_column: str = "diagnosis",
    image_extension: str = ".png",
    image_size: int = DEFAULT_IMAGE_SIZE,
    skip_cut_off: bool = True,
    denoise_method: str = "bilateral",
    **clahe_kwargs,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Build the preprocessed image array and label array from a dataframe.

    Cut-off images are dropped together with their labels only when
    skip_cut_off=False.

    Returns:
        X: uint8 array, shape (N, image_size, image_size, 3)
        y: int64 array, shape (N,)
    """
    images_dir = Path(images_dir)

    images: list[np.ndarray] = []
    labels: list[int] = []
    n_skipped = 0

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Preprocessing"):
        image_path = images_dir / f"{row[id_column]}{image_extension}"

        processed = preprocess_image_path(
            image_path,
            image_size=image_size,
            skip_cut_off=skip_cut_off,
            denoise_method=denoise_method,
            **clahe_kwargs,
        )

        if processed is None:
            n_skipped += 1
            continue

        images.append(processed)
        labels.append(int(row[label_column]))

    print(f"Kept {len(images)} images, skipped {n_skipped} unreadable/cut-off images.")

    X = np.asarray(images, dtype=np.uint8)
    y = np.asarray(labels, dtype=np.int64)

    return X, y


def run_pipeline(
    df,
    images_dir: str | Path,
    id_column: str = "id_code",
    label_column: str = "diagnosis",
    image_extension: str = ".png",
    image_size: int = DEFAULT_IMAGE_SIZE,
    test_size: float = 0.2,
    random_state: int = 42,
    balance_train: bool = True,
    skip_cut_off: bool = True,
    denoise_method: str = "bilateral",
    **clahe_kwargs,
) -> dict[str, np.ndarray]:
    """
    End-to-end preprocessing pipeline.

    This version creates a train/validation split internally.

    Use this when your dataset is not already split.

    If your dataset already has train/val/test CSVs, use build_dataset()
    separately for each split and apply SMOTE only to the training split.
    """
    from sklearn.model_selection import train_test_split

    X, y = build_dataset(
        df,
        images_dir,
        id_column=id_column,
        label_column=label_column,
        image_extension=image_extension,
        image_size=image_size,
        skip_cut_off=skip_cut_off,
        denoise_method=denoise_method,
        **clahe_kwargs,
    )

    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    if balance_train:
        X_train, y_train = apply_oversampling(
            X_train,
            y_train,
            random_state=random_state,
        )

        X_train = np.clip(np.round(X_train), 0, 255).astype(np.uint8)

        print(f"After SMOTE: training set has {len(X_train)} images.")

    return {
        "X_train": X_train,
        "y_train": y_train,
        "X_val": X_val,
        "y_val": y_val,
    }


# ---------------------------------------------------------------------------
# Example usage
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import kagglehub
    import pandas as pd

    dataset_path = Path(kagglehub.dataset_download("mariaherrerot/aptos2019"))

    df = pd.read_csv(dataset_path / "train_1.csv")
    images_dir = dataset_path / "train_images" / "train_images"

    data = run_pipeline(
        df,
        images_dir,
        skip_cut_off=True,
        denoise_method="bilateral",
    )

    print("X_train:", data["X_train"].shape, data["X_train"].dtype)
    print("y_train:", data["y_train"].shape, data["y_train"].dtype)
    print("X_val:  ", data["X_val"].shape, data["X_val"].dtype)
    print("y_val:  ", data["y_val"].shape, data["y_val"].dtype)
