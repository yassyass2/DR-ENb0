"""
Grad-CAM explainability for the EfficientNet-B0 DR classifier (proposal
phase 4).

Grad-CAM (Selvaraju et al., 2020) weights the channels of the last
convolutional feature map by the gradient of the predicted class score,
producing a coarse heatmap of the retinal regions that drove the prediction.
For EfficientNet-B0 with ``include_top=False`` the backbone output *is* the
last convolutional activation (``top_activation``, 7x7x1280), so the saved
model needs no surgery beyond exposing that tensor alongside the softmax.

The module is both a library (``build_gradcam_model`` / ``compute_gradcam`` /
``overlay_heatmap``) and a runnable script:

    python capstone-project/src/gradcam.py
    python capstone-project/src/gradcam.py --samples-per-grade 6

The script loads ``artifacts/best_model.keras`` and the preprocessed test
split, picks examples of every DR grade (correctly-classified first, then
misclassified ones, which are flagged in red), and writes per-grade overlay
grids plus a one-example-per-grade summary to ``artifacts/figures/``:

    gradcam_grade_0.png ... gradcam_grade_4.png
    gradcam_summary.png

Backend note: the gradient computation dispatches on the active Keras 3
backend, so it works both on the torch backend (GPU training setup) and on
plain TensorFlow CPU installs.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import keras

from dr_grading.data import load_preprocessed_data
from dr_grading.evaluate import DR_CLASS_NAMES
from dr_grading.paths import ARTIFACTS_DIR, DEFAULT_PREPROCESSED_DIR

DEFAULT_MODEL = ARTIFACTS_DIR / "best_model.keras"
DEFAULT_DATA_DIR = DEFAULT_PREPROCESSED_DIR
DEFAULT_OUT_DIR = ARTIFACTS_DIR / "figures"

# Head layers re-applied (in order) on top of the backbone features when
# rebuilding the model with the conv feature map exposed.
HEAD_LAYERS = (
    "avg_pool",
    "head_bn",
    "head_dropout",
    "head_dense",
    "head_dropout_2",
    "predictions",
)


def build_gradcam_model(model: "keras.Model") -> "keras.Model":
    """
    Rebuild the trained model so it returns ``(conv_features, predictions)``.

    The augmentation block is dropped (inference no-op) and the remaining
    layers of the original model are reused, so all weights are shared with
    ``model``. ``conv_features`` is the backbone output (last convolutional
    activation), the standard Grad-CAM target for EfficientNet.
    """
    import keras

    # The augmentation block is a Sequential (also a Model); the backbone is
    # the only nested *functional* model.
    backbone = next(
        layer
        for layer in model.layers
        if isinstance(layer, keras.Model) and not isinstance(layer, keras.Sequential)
    )

    inputs = keras.Input(shape=model.input_shape[1:], name="image")
    x = model.get_layer("to_efficientnet_range")(inputs)
    conv_features = backbone(x)

    x = conv_features
    for name in HEAD_LAYERS:
        x = model.get_layer(name)(x)

    return keras.Model(inputs, [conv_features, x], name="gradcam_model")


def compute_gradcam(
    gradcam_model: "keras.Model",
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

    import keras

    backend = keras.backend.backend()
    if backend == "torch":
        conv, proba, grads = _conv_grads_torch(gradcam_model, images, class_idx)
    elif backend == "tensorflow":
        conv, proba, grads = _conv_grads_tf(gradcam_model, images, class_idx)
    else:
        raise NotImplementedError(f"Unsupported Keras backend: {backend}")

    # Grad-CAM: channel weights = spatially-pooled gradients; heatmap = ReLU of
    # the weighted channel sum.
    weights = grads.mean(axis=(1, 2), keepdims=True)
    cam = np.maximum((conv * weights).sum(axis=-1), 0.0)

    size = images.shape[1:3]
    heatmaps = np.empty((len(images), *size), dtype=np.float32)
    import cv2

    for i, single in enumerate(cam):
        peak = single.max()
        if peak > 0:
            single = single / peak
        heatmaps[i] = cv2.resize(single, (size[1], size[0]))
    return heatmaps, proba


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
    """
    Blend a [0, 1] heatmap onto a [0, 1] RGB image with the jet colormap.

    Returns a float RGB image in [0, 1].
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

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
) -> Path:
    """Two-row grid per grade: preprocessed images on top, overlays below."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

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
    axes[1][0].set_ylabel("Grad-CAM", fontsize=10)
    fig.suptitle(
        f"Grad-CAM -- grade {grade} ({DR_CLASS_NAMES[grade]})", fontsize=12
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def plot_summary(
    examples: list[tuple[int, np.ndarray, np.ndarray, int]],
    out_path: Path,
) -> Path:
    """One correctly-graded example per grade: overlay row for the report."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, len(examples), figsize=(2.8 * len(examples), 3.2))
    for ax, (grade, image, heatmap, pred) in zip(np.atleast_1d(axes), examples):
        ax.imshow(overlay_heatmap(image, heatmap))
        ax.set_title(f"{grade} {DR_CLASS_NAMES[grade]}\n(pred {pred})", fontsize=9)
        ax.axis("off")
    fig.suptitle("Grad-CAM per DR severity grade", fontsize=12)
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
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.model.exists():
        raise FileNotFoundError(
            f"Model not found: {args.model}. Train one first with src/model.py."
        )

    dataset = load_preprocessed_data(args.data_dir)
    import keras

    model = keras.models.load_model(args.model)
    gradcam_model = build_gradcam_model(model)
    args.out_dir.mkdir(parents=True, exist_ok=True)

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
        heatmaps, _ = compute_gradcam(gradcam_model, images, class_idx=grade)

        path = plot_grade_grid(
            images,
            heatmaps,
            y_test[indices],
            y_pred[indices],
            grade,
            args.out_dir / f"gradcam_grade_{grade}.png",
        )
        print(f"Wrote {path}")
        summary_examples.append((grade, images[0], heatmaps[0], int(y_pred[indices[0]])))

    if summary_examples:
        path = plot_summary(summary_examples, args.out_dir / "gradcam_summary.png")
        print(f"Wrote {path}")


if __name__ == "__main__":
    main()
