from __future__ import annotations

import argparse
from pathlib import Path

from dr_grading.paths import DEFAULT_TRAINING_CONFIG


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="dr-grading",
        description="APTOS 2019 diabetic retinopathy grading research pipeline.",
    )
    subcommands = parser.add_subparsers(dest="command", required=True)

    preprocess = subcommands.add_parser(
        "preprocess",
        help="Build model-ready arrays from the APTOS 2019 split files.",
    )
    preprocess.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for preprocessed .npy arrays.",
    )
    preprocess.add_argument(
        "--dataset-dir",
        type=Path,
        default=None,
        help="Existing APTOS dataset directory; defaults to kagglehub download.",
    )
    preprocess.set_defaults(handler=_run_preprocess)

    train = subcommands.add_parser(
        "train",
        help="Train and evaluate the EfficientNet-B0 DR classifier.",
    )
    train.add_argument("--config", type=Path, default=DEFAULT_TRAINING_CONFIG)
    train.set_defaults(handler=_run_train)

    return parser


def _run_preprocess(args: argparse.Namespace) -> None:
    from dr_grading.preprocess import run_preprocessing

    run_preprocessing(output_dir=args.output_dir, dataset_dir=args.dataset_dir)


def _run_train(args: argparse.Namespace) -> None:
    from dr_grading.train import main as train_main

    train_main(["--config", str(args.config)])


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    args.handler(args)


if __name__ == "__main__":
    main()
