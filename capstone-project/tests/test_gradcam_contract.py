from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from src.gradcam import (
    build_gradcam_model,
    compute_gradcam,
    compute_gradcam_plus_plus,
    compute_scorecam,
    overlay_heatmap,
)
from src.model import build_model


class GradCAMContractTests(unittest.TestCase):
    def test_gradcam_predictions_match_original_model(self) -> None:
        model, _ = build_model(
            num_classes=5,
            image_size=224,
            dropout=0.3,
            weights=None,
            augment=True,
            seed=42,
        )
        gradcam_model = build_gradcam_model(model)
        images = np.random.default_rng(1).random((2, 224, 224, 1), dtype=np.float32)

        original_predictions = model.predict(images, verbose=0)
        _, gradcam_predictions = gradcam_model.predict(images, verbose=0)

        np.testing.assert_allclose(
            gradcam_predictions,
            original_predictions,
            atol=1e-6,
        )

    def test_compute_gradcam_returns_resized_heatmaps(self) -> None:
        model, _ = build_model(
            num_classes=5,
            image_size=224,
            dropout=0.3,
            weights=None,
            augment=False,
            seed=42,
        )
        gradcam_model = build_gradcam_model(model)
        images = np.random.default_rng(2).random((1, 224, 224, 1), dtype=np.float32)

        heatmaps, probabilities = compute_gradcam(gradcam_model, images)

        self.assertEqual(heatmaps.shape, (1, 224, 224))
        self.assertEqual(probabilities.shape, (1, 5))
        self.assertGreaterEqual(float(heatmaps.min()), 0.0)
        self.assertLessEqual(float(heatmaps.max()), 1.0)

    def test_compute_gradcam_plus_plus_returns_resized_heatmaps(self) -> None:
        model, _ = build_model(
            num_classes=5,
            image_size=224,
            dropout=0.3,
            weights=None,
            augment=False,
            seed=42,
        )
        gradcam_model = build_gradcam_model(model)
        images = np.random.default_rng(3).random((1, 224, 224, 1), dtype=np.float32)

        heatmaps, probabilities = compute_gradcam_plus_plus(gradcam_model, images)

        self.assertEqual(heatmaps.shape, (1, 224, 224))
        self.assertEqual(probabilities.shape, (1, 5))
        self.assertGreaterEqual(float(heatmaps.min()), 0.0)
        self.assertLessEqual(float(heatmaps.max()), 1.0)

    def test_compute_scorecam_returns_resized_heatmaps_with_limited_channels(self) -> None:
        model, _ = build_model(
            num_classes=5,
            image_size=224,
            dropout=0.3,
            weights=None,
            augment=False,
            seed=42,
        )
        gradcam_model = build_gradcam_model(model)
        images = np.random.default_rng(4).random((1, 224, 224, 1), dtype=np.float32)

        heatmaps, probabilities = compute_scorecam(
            gradcam_model,
            images,
            max_channels=8,
            batch_size=4,
        )

        self.assertEqual(heatmaps.shape, (1, 224, 224))
        self.assertEqual(probabilities.shape, (1, 5))
        self.assertGreaterEqual(float(heatmaps.min()), 0.0)
        self.assertLessEqual(float(heatmaps.max()), 1.0)

    def test_overlay_heatmap_accepts_one_channel_input(self) -> None:
        image = np.full((224, 224, 1), 0.5, dtype=np.float32)
        heatmap = np.ones((224, 224), dtype=np.float32)

        overlay = overlay_heatmap(image, heatmap)

        self.assertEqual(overlay.shape, (224, 224, 3))
        self.assertGreaterEqual(float(overlay.min()), 0.0)
        self.assertLessEqual(float(overlay.max()), 1.0)


if __name__ == "__main__":
    unittest.main()
