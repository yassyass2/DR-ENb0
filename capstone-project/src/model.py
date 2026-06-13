"""
EfficientNet-B0 diabetic-retinopathy classifier: build + train, end to end.

This module is both a library (``build_model`` / ``train_model`` /
``fine_tune``) and a runnable training script:

    python capstone-project/src/model.py
    python capstone-project/src/model.py --config configs/efficientnet_b0.json

What it does, start to finish:

  1. Load the preprocessed arrays from ``data/preprocessed`` (src/data.py).
  2. Build an EfficientNet-B0 transfer-learning model with a small custom head.
  3. Phase 1 -- warm up the head with the frozen ImageNet backbone.
  4. Phase 2 -- unfreeze the backbone (BatchNorm kept frozen) and fine-tune at a
     lower learning rate, checkpointing on validation QWK.
  5. Evaluate on the held-out test split with QWK, per-class sensitivity /
     specificity and ROC-AUC (src/evaluate.py).
  6. Log params / metrics / artifacts to MLflow when it is installed (optional).

Input contract (IMPORTANT)
--------------------------
The preprocessing pipeline saves one-channel green images as ``float32`` already
normalised to ``[0, 1]``. Keras 3's ``EfficientNetB0`` ships a *baked-in*
``Rescaling(1/255)`` layer (it cannot be disabled), so the backbone expects
three-channel pixels in ``[0, 255]`` and divides by 255 internally. To avoid
dividing by 255 twice, the model starts with a ``Rescaling(255.0)`` layer and
then repeats the green channel into RGB: its external input contract stays
``[0, 1]`` (matching the saved arrays), while the backbone still receives the
``[0, 255]`` three-channel tensor it expects. Feed the saved arrays in directly
-- do NOT call ``efficientnet.preprocess_input`` (a no-op in Keras 3 anyway).

Class imbalance is handled upstream: the training split is SMOTE-balanced by the
preprocessing step, so no class weights are applied here. Validation and test
keep their natural distribution, which is why QWK and per-class sensitivity /
specificity (not plain accuracy) are used for model selection and reporting.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import keras
from keras import layers

PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from src.data import PreprocessedDataset, load_preprocessed_data
from src.evaluate import (
    evaluate_predictions,
    format_report,
    metrics_for_logging,
    quadratic_weighted_kappa,
)
from src.tracking import (
    configure_mlflow,
    dataset_summary_params,
    flatten_params,
    load_training_config,
)

DEFAULT_CONFIG = PROJECT_DIR / "configs" / "efficientnet_b0.json"


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
def set_seeds(seed: int) -> None:
    """Seed Python, NumPy and TensorFlow RNGs for repeatable runs."""
    keras.utils.set_random_seed(int(seed))


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
def build_augmentation(seed: int) -> keras.Sequential:
    """
    Light, label-preserving augmentation for fundus images.

    Fundus photographs have no canonical orientation, so flips and small
    rotations / zooms are safe and cheap regularisation. These layers are
    no-ops at inference time. ``fill_mode="constant"`` pads with black (0),
    matching the cropped fundus border.
    """
    return keras.Sequential(
        [
            layers.RandomFlip("horizontal_and_vertical", seed=seed),
            layers.RandomRotation(0.10, fill_mode="constant", seed=seed),
            layers.RandomZoom(0.10, fill_mode="constant", seed=seed),
        ],
        name="data_augmentation",
    )


def build_model(
    num_classes: int = 5,
    image_size: int = 224,
    dropout: float = 0.5,
    weights: str | None = "imagenet",
    augment: bool = True,
    seed: int = 42,
) -> tuple[keras.Model, keras.Model]:
    """
    Build the EfficientNet-B0 transfer-learning model.

    Architecture::

        Input([0, 1], 1 channel) # saved arrays are green float32 in [0, 1]
        -> Rescaling(255.0)      # undo [0,1] so the backbone's baked-in
                                 #   Rescaling(1/255) lands back on [0, 1]
        -> repeat green to RGB   # EfficientNetB0 keeps 3-channel ImageNet input
        -> data augmentation     # train-only
        -> EfficientNetB0(top-less, ImageNet)
        -> GlobalAveragePooling2D
        -> BatchNormalization
        -> Dropout
        -> Dense(1024, relu)
        -> Dropout
        -> Dense(num_classes, softmax)

    The backbone is returned alongside the full model so the caller can
    unfreeze it for fine-tuning (see ``fine_tune``). It starts frozen.

    Returns ``(model, base_model)``.
    """
    inputs = keras.Input(shape=(image_size, image_size, 1), name="green_image")

    x = inputs

    # See the module docstring: our data is [0, 1] but the Keras 3 backbone
    # has a non-removable Rescaling(1/255), so scale back up to [0, 255] here.
    x = layers.Rescaling(255.0, name="to_efficientnet_range")(x)
    x = layers.Concatenate(axis=-1, name="repeat_green_to_rgb")([x, x, x])

    if augment:
        x = build_augmentation(seed)(x)

    base_model = keras.applications.EfficientNetB0(
        include_top=False,
        weights=weights,
        input_shape=(image_size, image_size, 3),
        pooling=None,
    )
    base_model.trainable = False  # Phase 1: frozen backbone.

    # training=False keeps the backbone's BatchNorm in inference mode, the
    # recommended setup for EfficientNet fine-tuning (preserves ImageNet stats).
    x = base_model(x, training=False)

    x = layers.GlobalAveragePooling2D(name="avg_pool")(x)
    x = layers.BatchNormalization(name="head_bn")(x)
    x = layers.Dropout(dropout, name="head_dropout")(x)
    x = layers.Dense(1024, activation="relu", name="head_dense")(x)
    x = layers.Dropout(dropout, name="head_dropout_2")(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="predictions")(x)

    model = keras.Model(inputs, outputs, name="efficientnet_b0_dr")
    return model, base_model


def compile_model(model: keras.Model, learning_rate: float) -> keras.Model:
    """Compile with Adam + sparse categorical cross-entropy (integer labels)."""
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def fine_tune(base_model: keras.Model, fine_tune_at: int | None = None) -> None:
    """
    Unfreeze the backbone for phase-2 fine-tuning, in place.

    ``fine_tune_at`` is the index into ``base_model.layers`` from which layers
    become trainable (everything before it stays frozen). ``None`` unfreezes the
    whole backbone. BatchNormalization layers are always kept frozen so their
    pretrained running statistics are not destroyed by the small DR dataset.

    The model must be re-compiled (typically at a lower learning rate) after
    calling this for the change to take effect.
    """
    base_model.trainable = True

    if fine_tune_at is not None:
        for layer in base_model.layers[:fine_tune_at]:
            layer.trainable = False

    for layer in base_model.layers:
        if isinstance(layer, layers.BatchNormalization):
            layer.trainable = False


# ---------------------------------------------------------------------------
# Validation-QWK callback (used for checkpointing / early stopping)
# ---------------------------------------------------------------------------
class QWKCallback(keras.callbacks.Callback):
    """
    Compute quadratic weighted kappa on the validation set each epoch and
    expose it as ``val_qwk`` in the logs, so ``ModelCheckpoint``,
    ``EarlyStopping`` and ``ReduceLROnPlateau`` can monitor it.

    Place this FIRST in the callback list so ``val_qwk`` is populated before the
    monitoring callbacks read it.
    """

    def __init__(self, x_val: np.ndarray, y_val: np.ndarray, batch_size: int = 32) -> None:
        super().__init__()
        self.x_val = x_val
        self.y_val = np.asarray(y_val)
        self.batch_size = batch_size

    def on_epoch_end(self, epoch: int, logs: dict[str, Any] | None = None) -> None:
        logs = logs if logs is not None else {}
        proba = self.model.predict(self.x_val, batch_size=self.batch_size, verbose=0)
        y_pred = proba.argmax(axis=1)
        qwk = quadratic_weighted_kappa(self.y_val, y_pred)
        logs["val_qwk"] = qwk
        print(f"  - val_qwk: {qwk:.4f}")


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
def train_model(
    model: keras.Model,
    base_model: keras.Model,
    dataset: PreprocessedDataset,
    config: dict[str, Any],
    checkpoint_path: Path,
) -> dict[str, Any]:
    """
    Two-phase transfer-learning training.

    Phase 1 (warm-up): frozen backbone, train only the new head.
    Phase 2 (fine-tune): unfreeze the backbone (BN frozen), train at a lower LR
    with QWK-monitored checkpointing, early stopping and LR reduction.

    The best weights (by validation QWK) are restored into ``model`` before
    returning. Returns the merged training history.
    """
    training = config.get("training", {})
    model_cfg = config.get("model", {})

    batch_size = int(training.get("batch_size", 16))
    warmup_lr = float(training.get("learning_rate", 1e-4))
    warmup_epochs = int(training.get("warmup_epochs", 3))
    fine_tune_epochs = int(training.get("fine_tune_epochs", training.get("epochs", 15)))
    fine_tune_lr = float(training.get("fine_tune_learning_rate", warmup_lr / 10.0))
    patience = int(training.get("early_stopping_patience", 5))
    fine_tune_at = model_cfg.get("fine_tune_at")

    x_train, y_train = dataset.X_train, dataset.y_train
    val_data = (dataset.X_val, dataset.y_val)
    qwk_callback = QWKCallback(dataset.X_val, dataset.y_val, batch_size=batch_size)

    # -- Phase 1: warm up the head -----------------------------------------
    print(f"\n=== Phase 1: warm-up head ({warmup_epochs} epochs, lr={warmup_lr}) ===")
    compile_model(model, warmup_lr)
    history_warmup = model.fit(
        x_train,
        y_train,
        validation_data=val_data,
        epochs=warmup_epochs,
        batch_size=batch_size,
        callbacks=[qwk_callback],
        verbose=2,
    )

    # -- Phase 2: fine-tune the backbone -----------------------------------
    print(
        f"\n=== Phase 2: fine-tune backbone "
        f"({fine_tune_epochs} epochs, lr={fine_tune_lr}, "
        f"fine_tune_at={fine_tune_at}) ==="
    )
    fine_tune(base_model, fine_tune_at)
    compile_model(model, fine_tune_lr)  # re-compile so trainable changes apply.

    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    callbacks = [
        qwk_callback,  # must be first: populates val_qwk for the others.
        keras.callbacks.ModelCheckpoint(
            str(checkpoint_path),
            monitor="val_qwk",
            mode="max",
            save_best_only=True,
            verbose=1,
        ),
        keras.callbacks.EarlyStopping(
            monitor="val_qwk",
            mode="max",
            patience=patience,
            restore_best_weights=True,
            verbose=1,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_qwk",
            mode="max",
            factor=0.5,
            patience=max(1, patience // 2),
            min_lr=1e-7,
            verbose=1,
        ),
    ]

    total_epochs = warmup_epochs + fine_tune_epochs
    history_ft = model.fit(
        x_train,
        y_train,
        validation_data=val_data,
        epochs=total_epochs,
        initial_epoch=warmup_epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=2,
    )

    return _merge_histories(history_warmup, history_ft)


def _merge_histories(*histories: keras.callbacks.History) -> dict[str, list[float]]:
    """Concatenate per-epoch metrics across the warm-up and fine-tune fits."""
    merged: dict[str, list[float]] = {}
    for history in histories:
        for key, values in history.history.items():
            merged.setdefault(key, []).extend(float(v) for v in values)
    return merged


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
def evaluate_on_test(
    model: keras.Model,
    dataset: PreprocessedDataset,
    num_classes: int,
    batch_size: int = 32,
) -> dict[str, Any]:
    """Predict on the test split and build the full metric report."""
    proba = model.predict(dataset.X_test, batch_size=batch_size, verbose=0)
    return evaluate_predictions(dataset.y_test, proba, num_classes=num_classes)


# ---------------------------------------------------------------------------
# MLflow helpers (optional dependency)
# ---------------------------------------------------------------------------
def _maybe_mlflow():
    try:
        import mlflow

        return mlflow
    except ModuleNotFoundError:
        return None


def _log_history(mlflow, history: dict[str, list[float]]) -> None:
    for key, values in history.items():
        for step, value in enumerate(values):
            mlflow.log_metric(key, float(value), step=step)


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------
def run_training(config: dict[str, Any]) -> dict[str, Any]:
    """
    Full pipeline for one config: load data, build, train, evaluate, log.

    Returns the test-set evaluation report.
    """
    data_cfg = config.get("data", {})
    model_cfg = config.get("model", {})
    training_cfg = config.get("training", {})
    experiment_cfg = config.get("experiment", {})

    seed = int(training_cfg.get("seed", 42))
    num_classes = int(data_cfg.get("num_classes", 5))
    image_size = int(data_cfg.get("image_size", 224))
    batch_size = int(training_cfg.get("batch_size", 16))

    set_seeds(seed)

    # 1. Data ---------------------------------------------------------------
    data_dir = PROJECT_DIR / data_cfg.get("preprocessed_dir", "data/preprocessed")
    dataset = load_preprocessed_data(data_dir)
    summary = dataset.summary
    print(
        f"Loaded data: train={summary.train_samples}, "
        f"val={summary.val_samples}, test={summary.test_samples}, "
        f"image_shape={summary.image_shape}"
    )

    artifacts_dir = PROJECT_DIR / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = artifacts_dir / "best_model.keras"
    final_model_path = artifacts_dir / "final_model.keras"
    report_path = artifacts_dir / "test_report.json"

    # 2. MLflow (optional) --------------------------------------------------
    mlflow = _maybe_mlflow()
    if mlflow is not None:
        tracking_uri = configure_mlflow(
            experiment_cfg.get("name", "aptos-dr-classification"), PROJECT_DIR
        )
        run_ctx = mlflow.start_run(run_name=experiment_cfg.get("run_name"))
    else:
        print("MLflow not installed -- continuing without experiment tracking.")
        tracking_uri = None
        run_ctx = contextlib.nullcontext()

    with run_ctx:
        if mlflow is not None:
            mlflow.set_tag("tracking_uri", tracking_uri)
            mlflow.log_params(flatten_params(config))
            mlflow.log_params(dataset_summary_params(summary))

        # 3. Build ----------------------------------------------------------
        model, base_model = build_model(
            num_classes=num_classes,
            image_size=image_size,
            dropout=float(model_cfg.get("dropout", 0.5)),
            weights=model_cfg.get("weights", "imagenet"),
            augment=bool(training_cfg.get("augment", True)),
            seed=seed,
        )
        model.summary(print_fn=lambda line: print(line))

        # 4. Train ----------------------------------------------------------
        history = train_model(model, base_model, dataset, config, checkpoint_path)
        if mlflow is not None:
            _log_history(mlflow, history)

        if checkpoint_path.exists():
            model = keras.models.load_model(checkpoint_path)

        # 5. Evaluate -------------------------------------------------------
        report = evaluate_on_test(model, dataset, num_classes, batch_size=batch_size)
        print("\n" + format_report(report))

        # 6. Persist + log --------------------------------------------------
        model.save(final_model_path)
        with report_path.open("w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2)
        print(f"\nSaved model to {final_model_path}")
        print(f"Best checkpoint at {checkpoint_path}")
        print(f"Wrote test report to {report_path}")

        if mlflow is not None:
            mlflow.log_metrics(metrics_for_logging(report, prefix="test"))
            mlflow.log_artifact(str(report_path))
            if checkpoint_path.exists():
                mlflow.log_artifact(str(checkpoint_path))

    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build and train the EfficientNet-B0 DR classifier."
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_training_config(args.config)
    run_training(config)


if __name__ == "__main__":
    main()
