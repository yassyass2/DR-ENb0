"""
This file is for flagging cut off retina images.
"""

import cv2
import shutil
from pathlib import Path
import kagglehub
import numpy as np
from tqdm import tqdm


def create_retina_mask(image, threshold=15):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    mask = (gray > threshold).astype(np.uint8)

    kernel = np.ones((7, 7), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    return mask


def border_retina_ratios(mask, border_width_ratio=0.02):
    height, width = mask.shape

    border_w = max(1, int(width * border_width_ratio))
    border_h = max(1, int(height * border_width_ratio))

    left = mask[:, :border_w].mean()
    right = mask[:, width - border_w :].mean()
    top = mask[:border_h, :].mean()
    bottom = mask[height - border_h :, :].mean()

    return {
        "left": left,
        "right": right,
        "top": top,
        "bottom": bottom,
    }


def is_cut_off_by_border_band(
    image_path,
    threshold=15,
    border_width_ratio=0.02,
    max_side_ratio=0.35,
):
    image = cv2.imread(str(image_path))

    if image is None:
        print(f"Could not read image: {image_path}")
        return False, None

    mask = create_retina_mask(image, threshold=threshold)

    ratios = border_retina_ratios(mask, border_width_ratio=border_width_ratio)

    is_cut = any(value > max_side_ratio for value in ratios.values())

    return is_cut, ratios


dataset_path_str = kagglehub.dataset_download("mariaherrerot/aptos2019")
dataset_path = Path(dataset_path_str)

source_dir = dataset_path / "train_images" / "train_images"

cutoff_dir = Path("cutoff_images")
normal_dir = Path("normal_images")

cutoff_dir.mkdir(exist_ok=True)
normal_dir.mkdir(exist_ok=True)

image_paths = list(source_dir.glob("*.png"))

print(f"Dataset path: {dataset_path}")
print(f"Source dir: {source_dir}")
print(f"Found {len(image_paths)} images")

for image_path in tqdm(image_paths):
    is_cut, ratios = is_cut_off_by_border_band(
        image_path,
        threshold=15,
        border_width_ratio=0.01,
        max_side_ratio=0.30,
    )

    if is_cut:
        shutil.copy2(image_path, cutoff_dir / image_path.name)
        print(image_path.name, ratios)
    else:
        shutil.copy2(image_path, normal_dir / image_path.name)
