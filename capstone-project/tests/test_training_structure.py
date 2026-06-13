from __future__ import annotations

import os
import sys
import tempfile
import tomllib
import unittest
from pathlib import Path

import numpy as np

PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from src.data import DatasetSummary, load_preprocessed_data
from src.model import balanced_class_weights
from src.tracking import configure_mlflow, load_training_config


class TrainingStructureTests(unittest.TestCase):
    def test_load_training_config_reads_defaults(self) -> None:
        config = load_training_config(PROJECT_DIR / "configs" / "efficientnet_b0.json")

        self.assertEqual(config["model"]["name"], "efficientnet_b0")
        self.assertEqual(config["data"]["preprocessed_dir"], "data/preprocessed")
        self.assertIn("run_name", config["experiment"])
        self.assertEqual(config["training"]["checkpoint_monitor"], "val_qwk")
        self.assertEqual(config["training"]["checkpoint_mode"], "max")

    def test_load_preprocessed_data_loads_all_splits_and_summary(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            data_dir = Path(tmp)
            arrays = {
                "X_train": np.zeros((2, 224, 224, 1), dtype=np.float32),
                "y_train": np.array([0, 1], dtype=np.int64),
                "X_val": np.ones((1, 224, 224, 1), dtype=np.float32),
                "y_val": np.array([1], dtype=np.int64),
                "X_test": np.full((1, 224, 224, 1), 0.5, dtype=np.float32),
                "y_test": np.array([0], dtype=np.int64),
            }
            for name, value in arrays.items():
                np.save(data_dir / f"{name}.npy", value)

            dataset = load_preprocessed_data(data_dir)

        self.assertEqual(dataset.X_train.shape, (2, 224, 224, 1))
        self.assertEqual(dataset.X_train.dtype, np.float32)
        self.assertEqual(dataset.y_val.tolist(), [1])
        self.assertIsInstance(dataset.summary, DatasetSummary)
        self.assertEqual(dataset.summary.train_samples, 2)
        self.assertEqual(dataset.summary.image_shape, (224, 224, 1))
        self.assertEqual(dataset.summary.class_distribution["train"], {0: 1, 1: 1})

    def test_load_preprocessed_data_rejects_noncanonical_image_arrays(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            data_dir = Path(tmp)
            arrays = {
                "X_train": np.zeros((2, 224, 224, 3), dtype=np.uint8),
                "y_train": np.array([0, 1], dtype=np.int64),
                "X_val": np.ones((1, 224, 224, 1), dtype=np.float32),
                "y_val": np.array([1], dtype=np.int64),
                "X_test": np.full((1, 224, 224, 1), 0.5, dtype=np.float32),
                "y_test": np.array([0], dtype=np.int64),
            }
            for name, value in arrays.items():
                np.save(data_dir / f"{name}.npy", value)

            with self.assertRaisesRegex(ValueError, "train images must have shape"):
                load_preprocessed_data(data_dir)

    def test_configure_mlflow_uses_local_default_tracking_uri(self) -> None:
        previous = os.environ.pop("MLFLOW_TRACKING_URI", None)
        try:
            with tempfile.TemporaryDirectory() as tmp:
                tracking_uri = configure_mlflow(
                    experiment_name="unit-test-experiment",
                    project_dir=Path(tmp),
                )
        finally:
            if previous is not None:
                os.environ["MLFLOW_TRACKING_URI"] = previous

        self.assertEqual(tracking_uri, f"sqlite:///{Path(tmp) / 'mlflow.db'}")

    def test_balanced_class_weights_match_sklearn_formula(self) -> None:
        weights = balanced_class_weights(np.array([0, 0, 0, 1], dtype=np.int64))

        self.assertEqual(weights[0], 4 / (2 * 3))
        self.assertEqual(weights[1], 4 / (2 * 1))

    def test_project_runtime_targets_tensorflow_compatible_python(self) -> None:
        with (PROJECT_DIR.parent / "pyproject.toml").open("rb") as handle:
            project = tomllib.load(handle)["project"]

        self.assertEqual(project["requires-python"], ">=3.13,<3.14")
        self.assertIn("tensorflow>=2.21,<2.22", project["dependencies"])


if __name__ == "__main__":
    unittest.main()
