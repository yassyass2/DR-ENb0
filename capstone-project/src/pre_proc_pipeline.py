"""
Complete preprocessing pipeline for the APTOS 2019 diabetic-retinopathy
dataset, targeting an EfficientNet-B0 classifier (224 x 224 x 3 input).

This module unifies the three preprocessing concerns of the project into a
single pipeline:

  1. Cut-off detection      (logic from ``src/cut_off.py``)
  2. Green-channel + CLAHE   (new contrast-enhancement step)
  3. SMOTE class balancing   (logic from ``src/preprocessing.py``)

Order of operations
-------------------
The ordering below follows common practice in published DR / fundus-image
work (e.g. Kaggle APTOS solutions, retinal-vessel and lesion-detection
papers). The guiding principles are: filter bad data first, standardise
geometry before enhancing intensity, and only balance classes *after* the
train/validation split so that synthetic samples never leak into validation.

  Per-image (steps 1-5):
    1. Quality filtering   -> discard cut-off / partially-imaged retinas
    2. Retina cropping     -> remove the uninformative black border
    3. Resize              -> 224 x 224 (EfficientNet-B0 input size)
    4. Green channel       -> the green channel has the best vessel/lesion
                              contrast and least noise in RGB fundus images
    5. CLAHE               -> Contrast Limited Adaptive Histogram
                              Equalisation on the green channel, then stack
                              to 3 channels so the input shape matches the
                              EfficientNet-B0 stem

  Dataset-level (steps 6-7):
    6. Train/validation split
    7. SMOTE               -> balance the TRAINING partition only

Note on intensity scaling: images are kept as ``uint8`` in the [0, 255] range
throughout. The Keras ``EfficientNetB0`` model includes its own rescaling /
normalisation layer (``efficientnet.preprocess_input``), so the network should
be fed raw [0, 255] values rather than pre-normalised ones.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

# Black-border crop
from src.preprocessing import crop_black_border, apply_oversampling

DEFAULT_IMAGE_SIZE = 224  # EfficientNet-B0 input resolution


# Step 1: cut-off / quality filtering
def create_retina_mask(image: np.ndarray, threshold: int = 15) -> np.ndarray:
    """Binary mask of the (bright) retina against the black background."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    mask = (gray > threshold).astype(np.uint8)

    kernel = np.ones((7, 7), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    return mask


def _border_retina_ratios(
    mask: np.ndarray, border_width_ratio: float = 0.02
) -> dict[str, float]:
    """Fraction of retina pixels sitting on each image border band."""
    height, width = mask.shape

    border_w = max(1, int(width * border_width_ratio))
    border_h = max(1, int(height * border_width_ratio))

    return {
        "left": mask[:, :border_w].mean(),
        "right": mask[:, width - border_w:].mean(),
        "top": mask[:border_h, :].mean(),
        "bottom": mask[height - border_h:, :].mean(),
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

    An image is flagged when the fundus touches at least ``min_cut_sides`` of
    the four image-border bands by more than ``max_side_ratio``.

    Requiring multiple sides (rather than any single one) is important for this
    dataset: normally-framed fundus photos are wider than they are tall, so the
    retinal disk fills the full sensor height and legitimately touches the top
    and bottom edges while the left/right edges stay black. Flagging on a single
    border would discard ~60% of these normal images (and most of the minority
    classes).

    Operates on an already-loaded BGR image
    """
    mask = create_retina_mask(image, threshold=threshold)
    ratios = _border_retina_ratios(mask, border_width_ratio=border_width_ratio)
    n_cut_sides = sum(value > max_side_ratio for value in ratios.values())
    return n_cut_sides >= min_cut_sides


# ---------------------------------------------------------------------------
# Steps 2-5: per-image transform
# ---------------------------------------------------------------------------
def extract_green_channel_clahe(
    image: np.ndarray,
    clip_limit: float = 2.0,
    tile_grid_size: tuple[int, int] = (8, 8),
    to_three_channels: bool = True,
) -> np.ndarray:
    """
    Extract the green channel and apply CLAHE for contrast enhancement.

    The green channel is preferred for fundus images because vessels and
    lesions show the highest contrast there. CLAHE then locally equalises the
    contrast without amplifying noise globally.

    By default the single enhanced channel is stacked into 3 identical
    channels so the output is compatible with EfficientNet-B0's 3-channel stem.
    """
    # OpenCV loads images as BGR -> green is index 1.
    green = image[:, :, 1]

    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    green = clahe.apply(green)

    if to_three_channels:
        return cv2.merge([green, green, green])
    return green


def preprocess_image(
    image: np.ndarray,
    image_size: int = DEFAULT_IMAGE_SIZE,
    clip_limit: float = 2.0,
    tile_grid_size: tuple[int, int] = (8, 8),
) -> np.ndarray:
    """
    Apply the per-image part of the pipeline (steps 2-5) to a single
    already-loaded BGR image.

      crop black border -> resize to 224x224 -> green channel + CLAHE
    """
    image = crop_black_border(image)
    image = cv2.resize(image, (image_size, image_size))
    image = extract_green_channel_clahe(
        image, clip_limit=clip_limit, tile_grid_size=tile_grid_size
    )
    return image


def preprocess_image_path(
    image_path: str | Path,
    image_size: int = DEFAULT_IMAGE_SIZE,
    skip_cut_off: bool = True,
    **clahe_kwargs,
) -> np.ndarray | None:
    """
    Read an image from disk and run steps 1-5.

    Returns the preprocessed ``uint8`` image of shape
    ``(image_size, image_size, 3)``, or ``None`` if the image could not be read
    or was filtered out as cut off.
    """
    image = cv2.imread(str(image_path))
    if image is None:
        print(f"Could not read image: {image_path}")
        return None

    if skip_cut_off and is_cut_off(image):
        return None

    return preprocess_image(image, image_size=image_size, **clahe_kwargs)


# ---------------------------------------------------------------------------
# Dataset assembly + steps 6-7
# ---------------------------------------------------------------------------
def build_dataset(
    df,
    images_dir: str | Path,
    id_column: str = "id_code",
    label_column: str = "diagnosis",
    image_extension: str = ".png",
    image_size: int = DEFAULT_IMAGE_SIZE,
    skip_cut_off: bool = True,
    **clahe_kwargs,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Build the preprocessed image array and label array from a dataframe with an
    id column and a label column. Cut-off images are dropped together with
    their labels.

    Returns
    -------
    X : np.ndarray, shape (N, image_size, image_size, 3), dtype uint8
    y : np.ndarray, shape (N,)
    """
    images_dir = Path(images_dir)

    images: list[np.ndarray] = []
    labels: list = []
    n_skipped = 0

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Preprocessing"):
        image_path = images_dir / f"{row[id_column]}{image_extension}"
        processed = preprocess_image_path(
            image_path,
            image_size=image_size,
            skip_cut_off=skip_cut_off,
            **clahe_kwargs,
        )

        if processed is None:
            n_skipped += 1
            continue

        images.append(processed)
        labels.append(row[label_column])

    print(f"Kept {len(images)} images, skipped {n_skipped} (unreadable/cut-off).")

    X = np.asarray(images, dtype=np.uint8)
    y = np.asarray(labels)
    return X, y


def run_pipeline(
    df,
    images_dir: str | Path,
    id_column: str = "id_code",
    label_column: str = "diagnosis",
    image_size: int = DEFAULT_IMAGE_SIZE,
    test_size: float = 0.2,
    random_state: int = 42,
    balance_train: bool = True,
    skip_cut_off: bool = True,
    **clahe_kwargs,
) -> dict[str, np.ndarray]:
    """
    End-to-end preprocessing pipeline.

      1-5. Per-image preprocessing (build_dataset)
      6.   Stratified train/validation split
      7.   SMOTE on the TRAINING partition only

    SMOTE is applied after the split so synthetic samples never leak into the
    validation set. It is intentionally *not* applied to the validation set,
    which is kept at its natural class distribution for honest evaluation.

    Returns a dict with ``X_train``, ``y_train``, ``X_val`` and ``y_val``.
    """
    # Imported here so the heavy sklearn import is only paid when running the
    # full dataset pipeline.
    from sklearn.model_selection import train_test_split

    # Steps 1-5
    X, y = build_dataset(
        df,
        images_dir,
        id_column=id_column,
        label_column=label_column,
        image_size=image_size,
        skip_cut_off=skip_cut_off,
        **clahe_kwargs,
    )

    # Step 6: stratified split keeps the class ratios in both partitions
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # Step 7: balance the training set only
    if balance_train:
        X_train, y_train = apply_oversampling(
            X_train, y_train, random_state=random_state
        )
        X_train = X_train.astype(np.uint8)
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

    data = run_pipeline(df, images_dir)

    print("X_train:", data["X_train"].shape, data["X_train"].dtype)
    print("y_train:", data["y_train"].shape)
    print("X_val:  ", data["X_val"].shape)
    print("y_val:  ", data["y_val"].shape)
