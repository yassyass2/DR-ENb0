from __future__ import annotations

import os
import sys
import tempfile
import tomllib
import unittest
from pathlib import Path

import numpy as np

PROJECT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from dr_grading.data import DatasetSummary, load_preprocessed_data
from dr_grading.download import ensure_preprocessed_archive, extract_file_id
from dr_grading.paths import DEFAULT_MODEL_PREPROCESSED_DOWNLOAD_SOURCE
from dr_grading.tracking import configure_mlflow, load_training_config


class TrainingStructureTests(unittest.TestCase):
    def test_load_training_config_reads_defaults(self) -> None:
        config = load_training_config(PROJECT_DIR / "configs" / "efficientnet_b0.json")

        self.assertEqual(config["model"]["name"], "efficientnet_b0")
        self.assertEqual(config["data"]["preprocessed_dir"], "data/preprocessed")
        self.assertIn("run_name", config["experiment"])

    def test_load_preprocessed_data_loads_all_splits_and_summary(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            data_dir = Path(tmp)
            arrays = {
                "X_train": np.zeros((2, 224, 224, 3), dtype=np.uint8),
                "y_train": np.array([0, 1], dtype=np.int64),
                "X_val": np.ones((1, 224, 224, 3), dtype=np.uint8),
                "y_val": np.array([1], dtype=np.int64),
                "X_test": np.full((1, 224, 224, 3), 2, dtype=np.uint8),
                "y_test": np.array([0], dtype=np.int64),
            }
            for name, value in arrays.items():
                np.save(data_dir / f"{name}.npy", value)

            dataset = load_preprocessed_data(data_dir)

        self.assertEqual(dataset.X_train.shape, (2, 224, 224, 3))
        self.assertEqual(dataset.y_val.tolist(), [1])
        self.assertIsInstance(dataset.summary, DatasetSummary)
        self.assertEqual(dataset.summary.train_samples, 2)
        self.assertEqual(dataset.summary.class_distribution["train"], {0: 1, 1: 1})

    def test_load_preprocessed_data_reads_npz_archives(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            archive_path = Path(tmp) / "dr_preprocessed_data.npz"
            arrays = {
                "X_train": np.zeros((2, 224, 224, 3), dtype=np.float32),
                "y_train": np.array([0, 1], dtype=np.int64),
                "X_val": np.ones((1, 224, 224, 3), dtype=np.float32),
                "y_val": np.array([1], dtype=np.int64),
                "X_test": np.full((1, 224, 224, 3), 2.0, dtype=np.float32),
                "y_test": np.array([0], dtype=np.int64),
            }
            np.savez(archive_path, **arrays)

            dataset = load_preprocessed_data(archive_path)

        self.assertEqual(dataset.X_test.shape, (1, 224, 224, 3))
        self.assertEqual(dataset.y_train.tolist(), [0, 1])
        self.assertIsInstance(dataset.summary, DatasetSummary)
        self.assertEqual(dataset.summary.test_samples, 1)
        self.assertEqual(dataset.summary.class_distribution["test"], {0: 1})

    def test_google_drive_link_extracts_expected_file_id(self) -> None:
        file_id = extract_file_id(DEFAULT_MODEL_PREPROCESSED_DOWNLOAD_SOURCE)

        self.assertEqual(file_id, "1vVVKvVr0b3AwXn3IEZkWktAEV-HIBkRS")

    def test_ensure_preprocessed_archive_keeps_existing_local_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            archive_path = Path(tmp) / "dr_preprocessed_data.npz"
            archive_path.write_bytes(b"already here")

            resolved = ensure_preprocessed_archive(archive_path)

        self.assertEqual(resolved, archive_path.resolve())

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

    def test_project_runtime_targets_tensorflow_compatible_python(self) -> None:
        with (PROJECT_DIR.parent / "pyproject.toml").open("rb") as handle:
            project = tomllib.load(handle)["project"]

        self.assertEqual(project["requires-python"], ">=3.13,<3.14")
        self.assertIn("tensorflow>=2.21,<2.22", project["dependencies"])


if __name__ == "__main__":
    unittest.main()
