"""
Per-image preprocessing pipeline for the APTOS 2019 diabetic-retinopathy
dataset, reproducing the procedure of the reference paper for an
EfficientNet-B0 classifier (224 x 224 x 3 input).

Per-image steps (paper order):
  1. Resize        -> 224 x 224
  2. CLAHE         -> contrast enhancement on the LAB lightness channel
  3. Noise         -> median blur, kernel size 3 (applied after CLAHE)
  4. Normalisation -> divide by 255 so each channel lies in [0, 1] (float32)

Dataset level:
  5. Train/validation split
  6. SMOTE         -> balance the TRAINING partition only

The paper applies no cropping or quality filtering. Optional cut-off filtering
(``is_cut_off``) remains available through the ``skip_cut_off`` flag but is
disabled by default to match the paper.

Note on intensity scaling: pixel values are normalised to [0, 1] (the paper's
final step), so the saved arrays are ``float32``. The Keras ``EfficientNetB0``
model ships its own ``preprocess_input`` rescaling layer; feed it these
already-[0, 1] values *without* calling ``preprocess_input`` again, otherwise
the scaling is applied twice.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

from dr_grading.preprocessing import (
    DEFAULT_IMAGE_SIZE,
    apply_oversampling,
    reduce_fundus_noise,
)


# ---------------------------------------------------------------------------
# Optional cut-off / quality filtering (off by default; not part of the paper)
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
# Per-image transform (paper steps 1-4)
# ---------------------------------------------------------------------------
def apply_clahe(
    image: np.ndarray,
    clip_limit: float = 2.0,
    tile_grid_size: tuple[int, int] = (8, 8),
) -> np.ndarray:
    """
    Apply CLAHE to a colour (BGR) image via its LAB lightness channel.

    CLAHE operates on a single channel, so we convert BGR -> LAB, equalise the
    L (lightness) channel, and convert back. This enhances local contrast
    without distorting colours and preserves the 3-channel shape that
    EfficientNet-B0 expects.
    """
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    lightness, a_channel, b_channel = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    lightness = clahe.apply(lightness)

    lab = cv2.merge([lightness, a_channel, b_channel])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


# Kept for reference / reuse: green-channel CLAHE. Not used by the
# paper-matching pipeline, which enhances the full colour image via apply_clahe.
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
    denoise_method: str = "median",
    clip_limit: float = 2.0,
    tile_grid_size: tuple[int, int] = (8, 8),
) -> np.ndarray:
    """
    Apply the paper's per-image preprocessing to one already-loaded BGR image,
    in the paper's order:

      resize to 224x224 -> CLAHE -> noise reduction (median, kernel 3)
      -> normalise to [0, 1]

    ``denoise_method`` selects the noise-reduction filter (see
    ``reduce_fundus_noise``); the paper uses ``"median"`` (a 3x3 median blur).

    Returns a ``float32`` image with values in [0, 1].
    """
    # Step 2 (paper): resize to the EfficientNet-B0 input resolution.
    image = cv2.resize(
        image,
        (image_size, image_size),
        interpolation=cv2.INTER_AREA,
    )

    # Step 3: contrast enhancement with CLAHE (LAB lightness channel).
    image = apply_clahe(image, clip_limit=clip_limit, tile_grid_size=tile_grid_size)

    # Step 4: noise reduction, applied after CLAHE. The paper uses a 3x3 median
    # blur; reduce_fundus_noise(method="median") is exactly cv2.medianBlur(., 3).
    image = reduce_fundus_noise(image, method=denoise_method)

    # Step 5: normalise pixel intensities to [0, 1] (paper eq. 1). Done last, as
    # the OpenCV operations above require uint8 input.
    image = image.astype(np.float32) / 255.0
    return image


def preprocess_image_path(
    image_path: str | Path,
    image_size: int = DEFAULT_IMAGE_SIZE,
    skip_cut_off: bool = True,
    denoise_method: str = "median",
    **clahe_kwargs,
) -> np.ndarray | None:
    """
    Read an image from disk and run the per-image preprocessing pipeline.

    Returns the preprocessed ``float32`` image of shape
    ``(image_size, image_size, 3)`` with values in [0, 1], or ``None`` if the
    image could not be read or (when ``skip_cut_off=False``) was filtered out as
    cut off.

    Note:
        skip_cut_off=True  -> cut-off filtering is skipped (paper default).
        skip_cut_off=False -> cut-off filtering is applied.
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
# Dataset assembly + steps 5-6
# ---------------------------------------------------------------------------
def build_dataset(
    df,
    images_dir: str | Path,
    id_column: str = "id_code",
    label_column: str = "diagnosis",
    image_extension: str = ".png",
    image_size: int = DEFAULT_IMAGE_SIZE,
    skip_cut_off: bool = True,
    denoise_method: str = "median",
    **clahe_kwargs,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Build the preprocessed image array and label array from a dataframe.

    Cut-off images are dropped together with their labels only when
    skip_cut_off=False (off by default, matching the paper).

    Returns
    -------
    X : np.ndarray, shape (N, image_size, image_size, 3), dtype float32, [0, 1]
    y : np.ndarray, shape (N,), dtype int64
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

    X = np.asarray(images, dtype=np.float32)
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
    denoise_method: str = "median",
    **clahe_kwargs,
) -> dict[str, np.ndarray]:
    """
    End-to-end preprocessing pipeline.

      1-4. Per-image preprocessing (build_dataset)
      5.   Stratified train/validation split
      6.   SMOTE on the TRAINING partition only

    This version creates a train/validation split internally. Use it when your
    dataset is not already split. If your dataset already has train/val/test
    CSVs, use build_dataset() separately for each split and apply SMOTE only to
    the training split.
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
        # apply_oversampling preserves the float32 [0, 1] range; interpolating
        # two values in [0, 1] stays in [0, 1], so no clipping is needed.
        X_train = X_train.astype(np.float32)

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
    print("y_train:", data["y_train"].shape, data["y_train"].dtype)
    print("X_val:  ", data["X_val"].shape, data["X_val"].dtype)
    print("y_val:  ", data["y_val"].shape, data["y_val"].dtype)
