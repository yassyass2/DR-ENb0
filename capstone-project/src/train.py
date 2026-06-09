from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from src.data import load_preprocessed_data
from src.tracking import (
    configure_mlflow,
    dataset_summary_params,
    flatten_params,
    load_training_config,
)

DEFAULT_CONFIG = PROJECT_DIR / "configs" / "efficientnet_b0.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Training template with MLflow.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_training_config(args.config)
    dataset = load_preprocessed_data(PROJECT_DIR / config["data"]["preprocessed_dir"])

    tracking_uri = configure_mlflow(config["experiment"]["name"], PROJECT_DIR)

    import mlflow

    with mlflow.start_run(run_name=config["experiment"]["run_name"]):
        mlflow.set_tag("tracking_uri", tracking_uri)
        mlflow.log_params(flatten_params(config))
        mlflow.log_params(dataset_summary_params(dataset.summary))

        print("MLflow run started and dataset metadata logged.")
        print("Add the actual model training code here when experiments start.")


if __name__ == "__main__":
    main()
