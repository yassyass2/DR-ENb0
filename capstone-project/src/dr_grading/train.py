"""
Training entry point for the APTOS-2019 diabetic-retinopathy classifier.

This is a thin CLI wrapper around ``dr_grading/model.run_training`` -- the actual
EfficientNet-B0 model and two-phase training loop live in ``dr_grading/model.py``.
Both of these are equivalent:

    python capstone-project/src/train.py
    python capstone-project/src/model.py

Override the default config (``configs/efficientnet_b0.json``) with ``--config``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from dr_grading.model import run_training
from dr_grading.paths import DEFAULT_TRAINING_CONFIG
from dr_grading.tracking import load_training_config

DEFAULT_CONFIG = DEFAULT_TRAINING_CONFIG


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the EfficientNet-B0 DR classifier."
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    config = load_training_config(args.config)
    run_training(config)


if __name__ == "__main__":
    main()
