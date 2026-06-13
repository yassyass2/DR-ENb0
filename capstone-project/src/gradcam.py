"""
Class-activation explainability for the EfficientNet-B0 DR classifier (proposal
phase 4).

Grad-CAM (Selvaraju et al., 2020), Grad-CAM++, and Score-CAM produce coarse
heatmaps of the retinal regions that drove the prediction.
For EfficientNet-B0 with ``include_top=False`` the backbone output *is* the
last convolutional activation (``top_activation``, 7x7x1280), so the saved
model needs no surgery beyond exposing that tensor alongside the softmax.

The module is both a library (``build_gradcam_model`` / ``compute_gradcam`` /
``compute_gradcam_plus_plus`` / ``compute_scorecam`` / ``overlay_heatmap``)
and a runnable script:

    python capstone-project/src/gradcam.py
    python capstone-project/src/gradcam.py --samples-per-grade 6

The script loads ``artifacts/best_model.keras`` and the preprocessed test
split, picks examples of every DR grade (correctly-classified first, then
misclassified ones, which are flagged in red), and writes per-grade overlay
grids plus a one-example-per-grade summary to ``artifacts/figures/``:

    gradcam_grade_0.png ... gradcam_grade_4.png
    gradcam_summary.png

Use ``--method gradcampp`` or ``--method scorecam`` for the additional XAI
methods. Score-CAM is computed over the top-k activated feature maps by default
because full EfficientNetB0 Score-CAM requires 1280 masked forward passes per
image.

Backend note: the gradient computation dispatches on the active Keras 3
backend, so it works both on the torch backend (GPU training setup) and on
plain TensorFlow CPU installs.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")  # headless: only file output, no display needed.

import matplotlib.pyplot as plt
import numpy as np
import keras

PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from src.data import load_preprocessed_data
from src.evaluate import DR_CLASS_NAMES

DEFAULT_MODEL = PROJECT_DIR / "artifacts" / "best_model.keras"
DEFAULT_DATA_DIR = PROJECT_DIR / "data" / "preprocessed"
DEFAULT_OUT_DIR = PROJECT_DIR / "artifacts" / "figures"
METHODS = ("gradcam", "gradcampp", "scorecam")

def _find_backbone(model: keras.Model) -> keras.Model:
    for layer in model.layers:
        if isinstance(layer, keras.Model) and "efficientnet" in layer.name.lower():
            return layer
    raise ValueError("Could not find EfficientNet backbone in model")


def build_gradcam_model(model: keras.Model) -> keras.Model:
    """
    Rebuild the trained model so it returns ``(conv_features, predictions)``.

    The augmentation block is dropped (inference no-op) and the remaining
    layers of the original model are reused, so all weights are shared with
    ``model``. ``conv_features`` is the backbone output (last convolutional
    activation), the standard Grad-CAM target for EfficientNet.
    """
    backbone = _find_backbone(model)
    inputs = keras.Input(shape=model.input_shape[1:], name="gradcam_input")

    x = inputs
    conv_features = None
    seen_backbone = False

    for layer in model.layers[1:]:
        if layer is backbone:
            conv_features = layer(x, training=False)
            x = conv_features
            seen_backbone = True
            continue

        if isinstance(layer, keras.layers.Concatenate):
            x = layer([x, x, x])
        elif isinstance(layer, keras.layers.Dropout):
            x = layer(x, training=False)
        elif isinstance(layer, keras.Sequential):
            x = layer(x, training=False)
        elif layer.__class__.__name__.startswith("Random"):
            x = layer(x, training=False)
        else:
            x = layer(x)

    if not seen_backbone or conv_features is None:
        raise ValueError("Could not route through EfficientNet backbone")

    return keras.Model(inputs, [conv_features, x], name="gradcam_model")


def compute_gradcam(
    gradcam_model: keras.Model,
    images: np.ndarray,
    class_idx: np.ndarray | int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Grad-CAM heatmaps for a batch of images.

    ``class_idx`` selects the class whose score is explained: ``None`` uses
    each image's predicted class, an int explains the same class for the whole
    batch, an array gives one class per image.

    Returns ``(heatmaps, probabilities)`` where ``heatmaps`` has shape
    ``(N, H, W)`` in [0, 1] at the input resolution and ``probabilities`` is
    the softmax output ``(N, num_classes)``.
    """
    images = np.asarray(images, dtype=np.float32)

    backend = keras.backend.backend()
    if backend == "torch":
        conv, proba, grads = _conv_grads_torch(gradcam_model, images, class_idx)
    elif backend == "tensorflow":
        conv, proba, grads = _conv_grads_tf(gradcam_model, images, class_idx)
    else:
        raise NotImplementedError(f"Unsupported Keras backend: {backend}")

    weights = grads.mean(axis=(1, 2), keepdims=True)
    cam = np.maximum((conv * weights).sum(axis=-1), 0.0)

    return _resize_and_normalize_cams(cam, images.shape[1:3]), proba


def compute_gradcam_plus_plus(
    gradcam_model: keras.Model,
    images: np.ndarray,
    class_idx: np.ndarray | int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Grad-CAM++ heatmaps for a batch of images.

    This follows the common Grad-CAM++ weighting approximation based on the
    first-order gradients available from the model graph. Compared with
    Grad-CAM's global-average gradient weights, it gives stronger weight to
    spatial locations with large positive gradients before channel aggregation.
    """
    images = np.asarray(images, dtype=np.float32)

    backend = keras.backend.backend()
    if backend == "torch":
        conv, proba, grads = _conv_grads_torch(gradcam_model, images, class_idx)
    elif backend == "tensorflow":
        conv, proba, grads = _conv_grads_tf(gradcam_model, images, class_idx)
    else:
        raise NotImplementedError(f"Unsupported Keras backend: {backend}")

    grads_squared = grads * grads
    grads_cubed = grads_squared * grads
    denominator = 2.0 * grads_squared + (conv * grads_cubed).sum(
        axis=(1, 2), keepdims=True
    )
    alphas = grads_squared / np.maximum(denominator, 1e-8)
    weights = (alphas * np.maximum(grads, 0.0)).sum(axis=(1, 2), keepdims=True)
    cam = np.maximum((conv * weights).sum(axis=-1), 0.0)

    return _resize_and_normalize_cams(cam, images.shape[1:3]), proba


def compute_scorecam(
    gradcam_model: keras.Model,
    images: np.ndarray,
    class_idx: np.ndarray | int | None = None,
    max_channels: int | None = 64,
    batch_size: int = 32,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Score-CAM heatmaps for a batch of images.

    Score-CAM avoids gradients: it upsamples activation maps, uses them as
    masks over the input image, and weights each mask by the model's class
    score for the masked image. For EfficientNetB0 the full feature map has
    1280 channels, so ``max_channels`` defaults to the top 64 channels by mean
    activation to keep report artifact generation practical.
    """
    images = np.asarray(images, dtype=np.float32)
    conv, proba = gradcam_model.predict(images, batch_size=batch_size, verbose=0)
    targets = _resolve_targets(proba, class_idx, len(images))

    heatmaps = np.empty((len(images), *images.shape[1:3]), dtype=np.float32)
    for i, image in enumerate(images):
        feature_maps = conv[i]
        channel_scores = feature_maps.mean(axis=(0, 1))
        if max_channels is not None and max_channels < feature_maps.shape[-1]:
            channels = np.argsort(channel_scores)[-max_channels:]
        else:
            channels = np.arange(feature_maps.shape[-1])

        masks = np.stack(
            [
                _normalize_map(
                    cv2.resize(
                        feature_maps[:, :, channel],
                        (images.shape[2], images.shape[1]),
                    )
                )
                for channel in channels
            ],
            axis=0,
        ).astype(np.float32)

        masked_images = image[None, ...] * masks[..., None]
        _, masked_proba = gradcam_model.predict(
            masked_images,
            batch_size=batch_size,
            verbose=0,
        )
        weights = np.maximum(masked_proba[:, targets[i]], 0.0)
        cam = np.maximum((masks * weights[:, None, None]).sum(axis=0), 0.0)
        heatmaps[i] = _normalize_map(cam)

    return heatmaps, proba


def _normalize_map(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    values = values - float(values.min())
    peak = float(values.max())
    if peak > 0:
        values = values / peak
    return values


def _resize_and_normalize_cams(
    cam: np.ndarray,
    size: tuple[int, int],
) -> np.ndarray:
    heatmaps = np.empty((len(cam), *size), dtype=np.float32)
    for i, single in enumerate(cam):
        heatmaps[i] = _normalize_map(cv2.resize(single, (size[1], size[0])))
    return heatmaps


def _resolve_targets(proba, class_idx, n: int) -> np.ndarray:
    if class_idx is None:
        return np.asarray(proba).argmax(axis=1)
    return np.broadcast_to(np.asarray(class_idx, dtype=int), (n,)).copy()


def _conv_grads_torch(gradcam_model, images, class_idx):
    import torch

    with torch.enable_grad():
        conv, proba = gradcam_model(images, training=False)
        conv.retain_grad()
        targets = _resolve_targets(proba.detach().cpu().numpy(), class_idx, len(images))
        index = torch.as_tensor(targets, device=proba.device).unsqueeze(1)
        proba.gather(1, index).sum().backward()
        grads = conv.grad

    return (
        conv.detach().cpu().numpy(),
        proba.detach().cpu().numpy(),
        grads.detach().cpu().numpy(),
    )


def _conv_grads_tf(gradcam_model, images, class_idx):
    import tensorflow as tf

    with tf.GradientTape() as tape:
        conv, proba = gradcam_model(images, training=False)
        tape.watch(conv)
        targets = _resolve_targets(proba.numpy(), class_idx, len(images))
        score = tf.gather(proba, targets, batch_dims=1)
    grads = tape.gradient(score, conv)

    return conv.numpy(), proba.numpy(), grads.numpy()


def overlay_heatmap(
    image: np.ndarray,
    heatmap: np.ndarray,
    alpha: float = 0.40,
) -> np.ndarray:
    """Blend a heatmap onto a one-channel or RGB image in [0, 1]."""
    image = np.asarray(image, dtype=np.float32)
    if image.ndim == 3 and image.shape[-1] == 1:
        image = np.repeat(image, 3, axis=-1)
    colored = plt.get_cmap("jet")(heatmap)[..., :3]
    return np.clip((1.0 - alpha) * image + alpha * colored, 0.0, 1.0)


# ---------------------------------------------------------------------------
# Figure generation
# ---------------------------------------------------------------------------
def select_grade_samples(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    grade: int,
    limit: int,
) -> np.ndarray:
    """
    Indices of test images of ``grade``, correctly-classified ones first.

    Misclassified samples are included after the correct ones: seeing where
    the model looked when it got a grade wrong is exactly the failure-mode
    insight the explainability phase is after.
    """
    of_grade = np.flatnonzero(y_true == grade)
    correct = of_grade[y_pred[of_grade] == grade]
    wrong = of_grade[y_pred[of_grade] != grade]
    return np.concatenate([correct, wrong])[:limit]


def plot_grade_grid(
    images: np.ndarray,
    heatmaps: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    grade: int,
    out_path: Path,
    method_label: str = "Grad-CAM",
) -> Path:
    """Two-row grid per grade: preprocessed images on top, overlays below."""
    n = len(images)
    fig, axes = plt.subplots(2, n, figsize=(2.6 * n, 5.6), squeeze=False)

    for i in range(n):
        axes[0][i].imshow(images[i])
        axes[1][i].imshow(overlay_heatmap(images[i], heatmaps[i]))
        correct = y_true[i] == y_pred[i]
        axes[0][i].set_title(
            f"true {y_true[i]} / pred {y_pred[i]}",
            fontsize=9,
            color="black" if correct else "crimson",
        )
        for row in axes:
            row[i].axis("off")

    axes[0][0].set_ylabel("input", fontsize=10)
    axes[1][0].set_ylabel(method_label, fontsize=10)
    fig.suptitle(
        f"{method_label} -- grade {grade} ({DR_CLASS_NAMES[grade]})", fontsize=12
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def plot_summary(
    examples: list[tuple[int, np.ndarray, np.ndarray, int]],
    out_path: Path,
    method_label: str = "Grad-CAM",
) -> Path:
    """One correctly-graded example per grade: overlay row for the report."""
    fig, axes = plt.subplots(1, len(examples), figsize=(2.8 * len(examples), 3.2))
    for ax, (grade, image, heatmap, pred) in zip(np.atleast_1d(axes), examples):
        ax.imshow(overlay_heatmap(image, heatmap))
        ax.set_title(f"{grade} {DR_CLASS_NAMES[grade]}\n(pred {pred})", fontsize=9)
        ax.axis("off")
    fig.suptitle(f"{method_label} per DR severity grade", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate Grad-CAM heatmaps for the trained DR model."
    )
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--samples-per-grade", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--method", choices=METHODS, default="gradcam")
    parser.add_argument(
        "--scorecam-max-channels",
        type=int,
        default=64,
        help="Top activation channels per image for Score-CAM.",
    )
    return parser.parse_args()


def compute_method_heatmaps(
    method: str,
    gradcam_model: keras.Model,
    images: np.ndarray,
    class_idx: np.ndarray | int | None,
    batch_size: int,
    scorecam_max_channels: int,
) -> tuple[np.ndarray, np.ndarray]:
    if method == "gradcam":
        return compute_gradcam(gradcam_model, images, class_idx=class_idx)
    if method == "gradcampp":
        return compute_gradcam_plus_plus(gradcam_model, images, class_idx=class_idx)
    if method == "scorecam":
        return compute_scorecam(
            gradcam_model,
            images,
            class_idx=class_idx,
            max_channels=scorecam_max_channels,
            batch_size=batch_size,
        )
    raise ValueError(f"Unsupported method: {method}")


def method_label(method: str) -> str:
    return {
        "gradcam": "Grad-CAM",
        "gradcampp": "Grad-CAM++",
        "scorecam": "Score-CAM",
    }[method]


def main() -> None:
    args = parse_args()
    if not args.model.exists():
        raise FileNotFoundError(
            f"Model not found: {args.model}. Train one first with src/model.py."
        )

    dataset = load_preprocessed_data(args.data_dir)
    model = keras.models.load_model(args.model)
    gradcam_model = build_gradcam_model(model)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    label = method_label(args.method)

    y_test = np.asarray(dataset.y_test).astype(int)
    proba = model.predict(dataset.X_test, batch_size=args.batch_size, verbose=0)
    y_pred = proba.argmax(axis=1)

    summary_examples: list[tuple[int, np.ndarray, np.ndarray, int]] = []
    for grade in range(len(DR_CLASS_NAMES)):
        indices = select_grade_samples(
            y_test, y_pred, grade, args.samples_per_grade
        )
        if len(indices) == 0:
            print(f"Grade {grade}: no test samples, skipped.")
            continue

        images = dataset.X_test[indices]
        # Explain the TRUE grade so the heatmap shows evidence for the actual
        # pathology, also on images the model got wrong.
        heatmaps, _ = compute_method_heatmaps(
            args.method,
            gradcam_model,
            images,
            class_idx=grade,
            batch_size=args.batch_size,
            scorecam_max_channels=args.scorecam_max_channels,
        )

        path = plot_grade_grid(
            images,
            heatmaps,
            y_test[indices],
            y_pred[indices],
            grade,
            args.out_dir / f"{args.method}_grade_{grade}.png",
            method_label=label,
        )
        print(f"Wrote {path}")
        summary_examples.append((grade, images[0], heatmaps[0], int(y_pred[indices[0]])))

    if summary_examples:
        path = plot_summary(
            summary_examples,
            args.out_dir / f"{args.method}_summary.png",
            method_label=label,
        )
        print(f"Wrote {path}")


if __name__ == "__main__":
    main()
