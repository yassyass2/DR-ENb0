import cv2
import numpy as np
from imblearn.over_sampling import SMOTE

DEFAULT_IMAGE_SIZE = 224  # For EfficientNet-B0


def crop_black_border(image: np.ndarray, threshold: int = 20) -> np.ndarray:
    """
    Crop the black background around a fundus image.

    Input:
        image: BGR uint8 image

    Output:
        cropped BGR uint8 image
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    mask = gray > threshold

    if not mask.any():
        return image

    y_indices, x_indices = np.where(mask)

    y1, y2 = y_indices.min(), y_indices.max()
    x1, x2 = x_indices.min(), x_indices.max()

    return image[y1:y2 + 1, x1:x2 + 1]


def reduce_fundus_noise(
    image: np.ndarray,
    method: str = "bilateral",
) -> np.ndarray:
    """
    Mild noise reduction for retinal fundus images.

    Keep this conservative. Diabetic retinopathy features like microaneurysms,
    hemorrhages, vessels, and small exudates can be tiny, so aggressive blur
    can remove useful information.

    Recommended default:
        bilateral

    Input:
        image: BGR image, usually uint8

    Output:
        denoised BGR uint8 image
    """
    if image.dtype != np.uint8:
        image = np.clip(image, 0, 255).astype(np.uint8)

    if method == "none":
        return image

    if method == "bilateral":
        return cv2.bilateralFilter(
            image,
            d=5,
            sigmaColor=35,
            sigmaSpace=35,
        )

    if method == "nlmeans":
        return cv2.fastNlMeansDenoisingColored(
            image,
            None,
            h=3,
            hColor=3,
            templateWindowSize=7,
            searchWindowSize=21,
        )

    if method == "median":
        return cv2.medianBlur(image, 3)

    if method == "gaussian":
        return cv2.GaussianBlur(image, (3, 3), 0)

    raise ValueError(f"Unknown denoising method: {method}")


def apply_oversampling(
    X_train: np.ndarray,
    y_train: np.ndarray,
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Balance the TRAINING set with SMOTE.

    SMOTE operates on flattened pixels. We cast to float32 first because SMOTE
    interpolates between samples. Doing that directly on uint8 can corrupt
    values through integer wraparound.

    Returns:
        X_resampled: uint8 image array, shape (N, H, W, C)
        y_resampled: label array
    """
    n, h, w, c = X_train.shape

    X_flat = X_train.reshape(n, h * w * c).astype(np.float32)

    classes, counts = np.unique(y_train, return_counts=True)
    min_count = int(counts.min())

    if len(classes) < 2:
        raise ValueError("SMOTE needs at least 2 classes in y_train.")

    if min_count < 2:
        raise ValueError(
            "SMOTE needs at least 2 samples in every class. "
            f"Class distribution: {dict(zip(classes.tolist(), counts.tolist()))}"
        )

    # SMOTE default k_neighbors=5 fails if the smallest class has <=5 samples.
    k_neighbors = min(5, min_count - 1)

    smote = SMOTE(
        random_state=random_state,
        k_neighbors=k_neighbors,
    )

    X_resampled_flat, y_resampled = smote.fit_resample(X_flat, y_train)

    X_resampled = X_resampled_flat.reshape(-1, h, w, c)
    X_resampled = np.clip(np.round(X_resampled), 0, 255).astype(np.uint8)

    y_resampled = y_resampled.astype(y_train.dtype)

    return X_resampled, y_resampled
