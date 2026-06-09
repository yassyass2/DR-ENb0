from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any


def load_training_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def configure_mlflow(
    experiment_name: str,
    project_dir: str | Path = ".",
) -> str:
    project_dir = Path(project_dir)
    default_tracking_uri = f"sqlite:///{project_dir / 'mlflow.db'}"
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", default_tracking_uri)

    try:
        import mlflow
    except ModuleNotFoundError:
        return tracking_uri

    mlflow.set_tracking_uri(tracking_uri)
    if tracking_uri == default_tracking_uri and mlflow.get_experiment_by_name(experiment_name) is None:
        mlflow.create_experiment(
            experiment_name,
            artifact_location=f"file:{project_dir / 'mlartifacts'}",
        )
    mlflow.set_experiment(experiment_name)
    return tracking_uri


def flatten_params(values: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    flattened: dict[str, Any] = {}
    for key, value in values.items():
        name = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict):
            flattened.update(flatten_params(value, name))
        else:
            flattened[name] = value
    return flattened


def dataset_summary_params(summary: Any) -> dict[str, Any]:
    return {
        "data.train_samples": summary.train_samples,
        "data.val_samples": summary.val_samples,
        "data.test_samples": summary.test_samples,
        "data.image_shape": str(summary.image_shape),
        "data.class_distribution.train": str(summary.class_distribution["train"]),
        "data.class_distribution.val": str(summary.class_distribution["val"]),
        "data.class_distribution.test": str(summary.class_distribution["test"]),
    }
