from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from src.preprocessing import (
    DEFAULT_IMAGE_SIZE,
    crop_black_border_rgb,
    preprocess_fundus_rgb,
    resize_with_padding,
)
from src.pre_proc_pipeline import build_dataset


class PreprocessingContractTests(unittest.TestCase):
    def test_crop_black_border_rgb_removes_black_margin(self) -> None:
        image = np.zeros((10, 12, 3), dtype=np.uint8)
        image[2:8, 3:9] = [20, 80, 30]

        cropped = crop_black_border_rgb(image, threshold=10)

        self.assertEqual(cropped.shape, (6, 6, 3))
        self.assertTrue(np.all(cropped == [20, 80, 30]))

    def test_resize_with_padding_preserves_aspect_ratio(self) -> None:
        image = np.full((20, 10, 3), 255, dtype=np.uint8)

        resized = resize_with_padding(image, target_size=DEFAULT_IMAGE_SIZE)

        self.assertEqual(resized.shape, (DEFAULT_IMAGE_SIZE, DEFAULT_IMAGE_SIZE, 3))
        content_mask = resized[:, :, 0] > 0
        ys, xs = np.where(content_mask)
        self.assertEqual(ys.min(), 0)
        self.assertEqual(ys.max(), DEFAULT_IMAGE_SIZE - 1)
        self.assertGreater(xs.min(), 0)
        self.assertLess(xs.max(), DEFAULT_IMAGE_SIZE - 1)

    def test_preprocess_fundus_rgb_outputs_green_float_contract(self) -> None:
        image = np.zeros((80, 120, 3), dtype=np.uint8)
        image[10:70, 20:100, 0] = 20
        image[10:70, 20:100, 1] = np.linspace(30, 220, 80, dtype=np.uint8)[None, :]
        image[10:70, 20:100, 2] = 10

        processed = preprocess_fundus_rgb(image)

        self.assertEqual(processed.shape, (DEFAULT_IMAGE_SIZE, DEFAULT_IMAGE_SIZE, 1))
        self.assertEqual(processed.dtype, np.float32)
        self.assertGreaterEqual(float(processed.min()), 0.0)
        self.assertLessEqual(float(processed.max()), 1.0)
        self.assertGreater(float(processed.max()), float(processed.min()))

    def test_preprocess_fundus_rgb_rejects_non_rgb_input(self) -> None:
        image = np.zeros((80, 120), dtype=np.uint8)

        with self.assertRaises(ValueError):
            preprocess_fundus_rgb(image)

    def test_build_dataset_uses_canonical_one_channel_contract(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            images_dir = Path(tmp)
            image = np.zeros((80, 120, 3), dtype=np.uint8)
            image[10:70, 20:100, 0] = 20
            image[10:70, 20:100, 1] = np.linspace(30, 220, 80, dtype=np.uint8)[
                None, :
            ]
            image[10:70, 20:100, 2] = 10
            cv2.imwrite(
                str(images_dir / "sample.png"),
                cv2.cvtColor(image, cv2.COLOR_RGB2BGR),
            )
            df = pd.DataFrame({"id_code": ["sample"], "diagnosis": [2]})

            X, y = build_dataset(df, images_dir)

        self.assertEqual(X.shape, (1, DEFAULT_IMAGE_SIZE, DEFAULT_IMAGE_SIZE, 1))
        self.assertEqual(X.dtype, np.float32)
        self.assertGreaterEqual(float(X.min()), 0.0)
        self.assertLessEqual(float(X.max()), 1.0)
        self.assertEqual(y.tolist(), [2])


if __name__ == "__main__":
    unittest.main()
