from __future__ import annotations

import importlib
import json
import sys
import types
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


def test_dr_grading_package_exposes_stable_project_paths() -> None:
    from dr_grading.paths import DEFAULT_PREPROCESSED_DIR, PROJECT_DIR

    assert PROJECT_DIR.name == "capstone-project"
    assert DEFAULT_PREPROCESSED_DIR == PROJECT_DIR / "data" / "preprocessed"


def test_preprocess_module_does_not_download_dataset_on_import(monkeypatch) -> None:
    def fail_on_download(_: str) -> str:
        raise AssertionError("dataset_download must run only inside preprocessing")

    fake_kagglehub = types.SimpleNamespace(dataset_download=fail_on_download)
    monkeypatch.setitem(sys.modules, "kagglehub", fake_kagglehub)
    sys.modules.pop("dr_grading.preprocess", None)

    module = importlib.import_module("dr_grading.preprocess")

    assert hasattr(module, "run_preprocessing")


def test_gradcam_replays_the_full_training_head() -> None:
    from dr_grading.gradcam import HEAD_LAYERS

    assert HEAD_LAYERS == (
        "avg_pool",
        "head_bn",
        "head_dropout",
        "head_dense",
        "head_dropout_2",
        "predictions",
    )


def test_training_artifacts_are_written_under_a_run_directory() -> None:
    from dr_grading.artifacts import resolve_artifact_paths
    from dr_grading.paths import ARTIFACTS_DIR

    paths = resolve_artifact_paths(
        {
            "experiment": {
                "run_name": "EfficientNet B0 baseline / paper run",
            }
        },
        started_at="20260615T120000Z",
    )

    assert paths.run_dir == ARTIFACTS_DIR / "efficientnet-b0-baseline-paper-run-20260615T120000Z"
    assert paths.checkpoint.name == "best_model.keras"
    assert paths.final_model.parent == paths.run_dir
    assert paths.report.parent == paths.run_dir


def test_preprocessing_manifest_records_dataset_output_and_split_counts(tmp_path) -> None:
    from dr_grading.preprocess import write_preprocessing_manifest

    manifest_path = write_preprocessing_manifest(
        dataset_dir=tmp_path / "aptos",
        output_dir=tmp_path / "preprocessed",
        split_summaries={
            "train": {
                "before_smote": {0: 2, 1: 1},
                "after_smote": {0: 2, 1: 2},
                "samples": 4,
            },
            "val": {
                "distribution": {0: 1, 1: 1},
                "samples": 2,
            },
        },
        git_commit="abc123",
        package_version="0.1.0",
    )

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["dataset_dir"] == str((tmp_path / "aptos").resolve())
    assert manifest["output_dir"] == str((tmp_path / "preprocessed").resolve())
    assert manifest["code"]["git_commit"] == "abc123"
    assert manifest["code"]["package_version"] == "0.1.0"
    assert manifest["splits"]["train"]["after_smote"] == {"0": 2, "1": 2}
    assert manifest["preprocessing"]["smote_applied_to"] == "train"


def test_training_logs_artifact_run_dir_to_mlflow(tmp_path) -> None:
    from dr_grading.artifacts import log_run_artifact_location

    calls: list[tuple[str, str]] = []
    fake_mlflow = types.SimpleNamespace(
        set_tag=lambda key, value: calls.append((key, value))
    )

    log_run_artifact_location(fake_mlflow, tmp_path / "artifacts" / "run")

    assert calls == [("artifact_run_dir", str((tmp_path / "artifacts" / "run").resolve()))]
