import cv2
import numpy as np

DEFAULT_IMAGE_SIZE = 224  # For EfficientNet-B0


def crop_black_border(image, threshold=20):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    mask = gray > threshold

    if not mask.any():
        return image

    y_indices, x_indices = np.where(mask)
    y1, y2 = y_indices.min(), y_indices.max()
    x1, x2 = x_indices.min(), x_indices.max()

    return image[y1 : y2 + 1, x1 : x2 + 1]


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
