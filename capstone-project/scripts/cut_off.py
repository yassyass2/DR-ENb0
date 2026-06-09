"""
QA utility: sort the APTOS-2019 training images into "cut-off" and "normal"
folders so the cut-off filter can be eyeballed.

This is a one-off diagnostic script, NOT part of the training pipeline. It
reuses the canonical cut-off detector from ``src/pre_proc_pipeline.py`` (the
3-of-4-border criterion that the pipeline actually applies) rather than
re-implementing it, so what you see here matches what ``build_dataset`` drops.

Run it directly from the ``capstone-project`` directory:

    python scripts/cut_off.py

It downloads the dataset via kagglehub and writes ``cutoff_images/`` and
``normal_images/`` in the current working directory.
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

import cv2
import kagglehub
from tqdm import tqdm

PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

# Reuse the pipeline's cut-off logic instead of duplicating it. _border_retina_ratios
# is imported only for the diagnostic print below.
from src.pre_proc_pipeline import (
    create_retina_mask,
    is_cut_off,
    _border_retina_ratios,
)


def main() -> None:
    dataset_path = Path(kagglehub.dataset_download("mariaherrerot/aptos2019"))
    source_dir = dataset_path / "train_images" / "train_images"

    cutoff_dir = Path("cutoff_images")
    normal_dir = Path("normal_images")
    cutoff_dir.mkdir(exist_ok=True)
    normal_dir.mkdir(exist_ok=True)

    image_paths = list(source_dir.glob("*.png"))

    print(f"Dataset path: {dataset_path}")
    print(f"Source dir:   {source_dir}")
    print(f"Found {len(image_paths)} images")

    n_cut = 0
    for image_path in tqdm(image_paths):
        image = cv2.imread(str(image_path))
        if image is None:
            print(f"Could not read image: {image_path}")
            continue

        if is_cut_off(image):
            n_cut += 1
            shutil.copy2(image_path, cutoff_dir / image_path.name)
            ratios = _border_retina_ratios(create_retina_mask(image))
            print(image_path.name, ratios)
        else:
            shutil.copy2(image_path, normal_dir / image_path.name)

    print(f"\nFlagged {n_cut} cut-off / {len(image_paths) - n_cut} normal images.")


if __name__ == "__main__":
    main()
