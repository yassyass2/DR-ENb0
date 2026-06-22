from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

from dr_grading.data import PreprocessedDataset, load_preprocessed_data
from dr_grading.download import ensure_preprocessed_archive
from dr_grading.paths import (
    DEFAULT_MODEL_PREPROCESSED_ARCHIVE,
    DEFAULT_MODEL_PREPROCESSED_DOWNLOAD_SOURCE,
    DEFAULT_SAVED_MODELS_DIR,
)


DR_CLASS_NAMES = (
    "No DR",
    "Mild",
    "Moderate",
    "Severe",
    "Proliferative",
)


SPLIT_FIELDS = {
    "train": ("X_train", "y_train"),
    "val": ("X_val", "y_val"),
    "test": ("X_test", "y_test"),
}


def build_parser() -> argparse.ArgumentParser:
    default_models_dir = DEFAULT_SAVED_MODELS_DIR
    discovered_models_text = format_discovered_models_for_help(default_models_dir)
    parser = argparse.ArgumentParser(
        description="Run a saved DR grading model on one preprocessed image sample.",
        epilog=(
            "Discovered models in the default models folder:\n"
            f"{discovered_models_text}\n\n"
            "Use `--list-models` to print the current model list at runtime."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help=(
            "Model filename, full path, or 1-based number from the discovered model list. "
            "Run without --model for an interactive prompt."
        ),
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=DEFAULT_MODEL_PREPROCESSED_ARCHIVE,
        help=(
            "Preprocessed dataset source: a .npz archive or directory of .npy arrays. "
            "If the archive is missing, the default .npz can be downloaded automatically."
        ),
    )
    parser.add_argument(
        "--download-source",
        type=str,
        default=DEFAULT_MODEL_PREPROCESSED_DOWNLOAD_SOURCE,
        help=(
            "Google Drive share URL or file ID used when the .npz archive is missing. "
            "Defaults to the repository's configured dataset link."
        ),
    )
    parser.add_argument(
        "--models-dir",
        type=Path,
        default=DEFAULT_SAVED_MODELS_DIR,
        help="Directory containing saved .keras models.",
    )
    parser.add_argument(
        "--split",
        choices=tuple(SPLIT_FIELDS),
        default="test",
        help="Dataset split to sample from.",
    )
    parser.add_argument(
        "--index",
        type=int,
        default=None,
        help="0-based sample index within the selected split.",
    )
    parser.add_argument(
        "--random",
        action="store_true",
        help="Choose a random sample from the selected split.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed used for random sample selection.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="How many class probabilities to print.",
    )
    parser.add_argument(
        "--list-models",
        action="store_true",
        help="List discovered .keras models and exit.",
    )
    return parser


def discover_saved_models(models_dir: Path) -> list[Path]:
    models_dir = Path(models_dir)
    if not models_dir.exists():
        raise FileNotFoundError(f"Models directory does not exist: {models_dir}")
    if not models_dir.is_dir():
        raise NotADirectoryError(f"Expected a models directory, got: {models_dir}")
    return sorted(path.resolve() for path in models_dir.glob("*.keras"))


def format_discovered_models_for_help(models_dir: Path) -> str:
    try:
        models = discover_saved_models(models_dir)
    except (FileNotFoundError, NotADirectoryError):
        return f"  none found in {models_dir}"

    if not models:
        return f"  none found in {models_dir}"

    lines = [f"  {index}. {path.name}" for index, path in enumerate(models, start=1)]
    return "\n".join(lines)


def resolve_model_path(selection: str | None, models_dir: Path) -> Path:
    models = discover_saved_models(models_dir)
    if not models:
        raise FileNotFoundError(f"No .keras models found in {models_dir}")

    if selection is None:
        _print_models(models)
        selection = _prompt(
            "Choose a model number",
            allow_empty=False,
        )

    stripped = selection.strip()
    candidate = Path(stripped).expanduser()
    if candidate.exists():
        return candidate.resolve()

    if stripped.isdigit():
        model_number = int(stripped)
        if model_number < 1 or model_number > len(models):
            raise IndexError(
                f"Model number {model_number} is out of range. Choose 1-{len(models)}."
            )
        return models[model_number - 1]

    matches = [
        path
        for path in models
        if path.name == stripped or path.stem == stripped
    ]
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        raise ValueError(
            f"Model selection '{stripped}' matches multiple files. Use a number or full path."
        )

    available = ", ".join(path.name for path in models)
    raise FileNotFoundError(
        f"Could not resolve model '{selection}'. Available models: {available}"
    )


def select_sample_index(
    requested_index: int | None,
    split_size: int,
    split_name: str,
    random_sample: bool,
    seed: int,
) -> int:
    if split_size <= 0:
        raise ValueError(f"The {split_name} split is empty.")

    if requested_index is not None and random_sample:
        raise ValueError("Use either --index or --random, not both.")

    if requested_index is not None:
        if requested_index < 0 or requested_index >= split_size:
            raise IndexError(
                f"Sample index {requested_index} is out of range for {split_name}. "
                f"Valid indices: 0-{split_size - 1}."
            )
        return requested_index

    if random_sample:
        rng = np.random.default_rng(seed)
        return int(rng.integers(0, split_size))

    choice = _prompt(
        f"Choose a {split_name} sample index (0-{split_size - 1}) or press Enter for a random sample",
        allow_empty=True,
    )
    if choice == "":
        rng = np.random.default_rng(seed)
        return int(rng.integers(0, split_size))

    if not choice.isdigit():
        raise ValueError(f"Expected an integer sample index, got: {choice!r}")

    index = int(choice)
    if index < 0 or index >= split_size:
        raise IndexError(
            f"Sample index {index} is out of range for {split_name}. "
            f"Valid indices: 0-{split_size - 1}."
        )
    return index


def split_arrays(dataset: PreprocessedDataset, split_name: str) -> tuple[np.ndarray, np.ndarray]:
    x_field, y_field = SPLIT_FIELDS[split_name]
    return getattr(dataset, x_field), getattr(dataset, y_field)


def predict_one_sample(
    model_path: Path,
    dataset: PreprocessedDataset,
    split_name: str,
    index: int,
    top_k: int,
) -> str:
    try:
        from keras.models import load_model
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Keras/TensorFlow is not available in this Python environment. "
            "Run the script with the project environment, for example via `uv run`."
        ) from exc

    X, y = split_arrays(dataset, split_name)
    image = X[index]
    true_label = int(y[index])

    model = load_model(model_path, compile=False)
    probabilities = np.asarray(model.predict(image[np.newaxis, ...], verbose=0)[0], dtype=float)
    predicted_label = int(probabilities.argmax())
    confidence = float(probabilities[predicted_label])
    sorted_indices = np.argsort(probabilities)[::-1]
    top_k = max(1, min(int(top_k), len(sorted_indices)))
    top_indices = sorted_indices[:top_k]

    lines = [
        "=" * 72,
        f"Model      : {model_path.name}",
        f"Model path : {model_path}",
    ]
    lines.extend(
        [
            f"Split      : {split_name}",
            f"Sample     : {index} / {len(X) - 1}",
            f"Image shape: {tuple(image.shape)}",
            f"Image dtype: {image.dtype}",
            f"Image range: [{float(image.min()):.4f}, {float(image.max()):.4f}]",
            f"True label : {true_label} ({class_name(true_label)})",
            f"Prediction : {predicted_label} ({class_name(predicted_label)})",
            f"Confidence : {confidence:.4f}",
            f"Correct    : {'yes' if predicted_label == true_label else 'no'}",
            f"Margin     : {prediction_margin(probabilities):.4f}",
            f"Entropy    : {prediction_entropy(probabilities):.4f}",
            "",
            "Class probabilities:",
        ]
    )
    for rank, class_index in enumerate(top_indices, start=1):
        lines.append(
            f"  {rank}. class {class_index} ({class_name(int(class_index))})"
            f" -> {float(probabilities[class_index]):.4f}"
        )
    lines.append("=" * 72)
    return "\n".join(lines)


def class_name(label: int) -> str:
    if 0 <= label < len(DR_CLASS_NAMES):
        return DR_CLASS_NAMES[label]
    return f"Class {label}"


def prediction_margin(probabilities: np.ndarray) -> float:
    if probabilities.size < 2:
        return float(probabilities[0]) if probabilities.size == 1 else 0.0
    top_two = np.partition(probabilities, -2)[-2:]
    return float(np.max(top_two) - np.min(top_two))


def prediction_entropy(probabilities: np.ndarray) -> float:
    safe = np.clip(np.asarray(probabilities, dtype=float), 1e-12, 1.0)
    return float(-np.sum(safe * np.log(safe)))


def _print_models(models: list[Path]) -> None:
    print("Available saved models:")
    for index, path in enumerate(models, start=1):
        size_mb = path.stat().st_size / (1024 * 1024)
        print(f"  {index}. {path.name} ({size_mb:.1f} MB)")


def _prompt(message: str, allow_empty: bool) -> str:
    if not sys.stdin.isatty():
        raise ValueError(
            f"{message}. Interactive input is not available, so pass the needed CLI flag."
        )
    while True:
        response = input(f"{message}: ").strip()
        if response or allow_empty:
            return response


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    models_dir = Path(args.models_dir)

    if args.list_models:
        models = discover_saved_models(models_dir)
        _print_models(models)
        return

    model_path = resolve_model_path(args.model, models_dir)
    data_path = ensure_preprocessed_archive(
        data_path=Path(args.data),
        download_source=args.download_source,
    )
    dataset = load_preprocessed_data(data_path)
    X, _ = split_arrays(dataset, args.split)
    sample_index = select_sample_index(
        requested_index=args.index,
        split_size=len(X),
        split_name=args.split,
        random_sample=args.random,
        seed=args.seed,
    )

    output = predict_one_sample(
        model_path=model_path,
        dataset=dataset,
        split_name=args.split,
        index=sample_index,
        top_k=args.top_k,
    )

    print(f"Data path  : {data_path.resolve()}")
    print(
        "Dataset    : "
        f"train={dataset.summary.train_samples}, "
        f"val={dataset.summary.val_samples}, "
        f"test={dataset.summary.test_samples}"
    )
    print(output)


if __name__ == "__main__":
    main()
