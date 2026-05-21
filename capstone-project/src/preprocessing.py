import cv2
import numpy as np
from imblearn.over_sampling import SMOTE

DEFAULT_IMAGE_SIZE = 224  # For EfficientNet-B0


def crop_black_border(image, threshold=20):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    mask = gray > threshold

    if not mask.any():
        return image

    y_indices, x_indices = np.where(mask)
    y1, y2 = y_indices.min(), y_indices.max()
    x1, x2 = x_indices.min(), x_indices.max()

    return image[y1: y2 + 1, x1: x2 + 1]


def preprocess_dr_image(
    image_path: str, image_size: int = DEFAULT_IMAGE_SIZE
) -> cv2.typing.MatLike:
    image = cv2.imread(image_path)

    if image is None:
        raise ValueError(f"Could not read image: {image_path}")

    # Resize
    # image = cv2.resize(image, (image_size, image_size))

    image = crop_black_border(image)

    return image


def apply_oversampling(X_train, y_train, random_state=42):
    """
    Balance the TRAINING set with SMOTE (Synthetic Minority Over-sampling
    Technique), operating on flattened pixels.
    """

    # Image dimensions
    n, h, w, c = X_train.shape

    # Flatten image to an array of integers
    X_flat = X_train.reshape(n, h * w * c)

    smote = SMOTE(random_state=random_state)
    X_resampled_flat, y_resampled = smote.fit_resample(X_flat, y_train)

    # Reshape back to normal original image dimensions
    X_resampled = X_resampled_flat.reshape(-1, h, w, c)

    return X_resampled, y_resampled
