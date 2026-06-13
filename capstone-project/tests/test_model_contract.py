from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from src.model import build_model


class ModelContractTests(unittest.TestCase):
    def test_model_accepts_one_channel_normalized_input(self) -> None:
        model, base_model = build_model(
            num_classes=5,
            image_size=224,
            dropout=0.3,
            weights=None,
            augment=False,
            seed=42,
        )

        self.assertEqual(model.input_shape, (None, 224, 224, 1))
        self.assertEqual(model.output_shape, (None, 5))
        self.assertFalse(base_model.trainable)

        batch = np.random.default_rng(0).random((2, 224, 224, 1), dtype=np.float32)
        predictions = model.predict(batch, verbose=0)

        self.assertEqual(predictions.shape, (2, 5))
        np.testing.assert_allclose(predictions.sum(axis=1), np.ones(2), atol=1e-5)


if __name__ == "__main__":
    unittest.main()
