"""
Training entry point for the APTOS-2019 diabetic-retinopathy classifier.

This is a thin CLI wrapper around ``src/model.run_training`` -- the actual
EfficientNet-B0 model and two-phase training loop live in ``src/model.py``.
Both of these are equivalent:

    python capstone-project/src/train.py
    python capstone-project/src/model.py

Override the default config (``configs/efficientnet_b0.json``) with ``--config``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from src.model import run_training
from src.tracking import load_training_config

DEFAULT_CONFIG = PROJECT_DIR / "configs" / "efficientnet_b0.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the EfficientNet-B0 DR classifier."
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_training_config(args.config)
    run_training(config)


if __name__ == "__main__":
    main()
