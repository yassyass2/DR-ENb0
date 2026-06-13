# APTOS 2019 Blindness Detection - Capstone Project

## Team

**Course:** AI for Health - Group 5  
**Members:** Yassine Abderrazik, Adel Atzouza, Ozeir Moradi, Musab Sivrikaya, Mark Salloum

**Project:** Diabetic Retinopathy Classification using CNNs

## Overview

This project develops a CNN-based classifier for detecting diabetic retinopathy severity using the APTOS 2019 Blindness Detection dataset. The model classifies retinal images into 5 categories (0-4) representing different stages of diabetic retinopathy.

## Project flow

The current project is split into three clear steps:

1. Preprocess the APTOS dataset into model-ready numpy arrays.
2. Validate those arrays before training.
3. Train and track experiments with MLflow.

## Preprocessing

From the `capstone-project` directory:

```bash
uv run python main.py --out-dir data/preprocessed
uv run python scripts/validate_preprocessed.py
```

This writes the following files under `data/preprocessed`:

```text
X_train.npy, y_train.npy
X_val.npy, y_val.npy
X_test.npy, y_test.npy
splits/train.csv, splits/val.csv, splits/test.csv, splits/metadata.json
```

SMOTE is a separate experiment, not the default baseline preprocessing:

```bash
uv run python main.py --out-dir data/preprocessed_smote --smote-train
uv run python src/train.py --config configs/efficientnet_b0_smote.json
```

SMOTE training writes to `artifacts/smote/` so it does not overwrite the
canonical baseline artifacts in `artifacts/`.

## Training structure

Training code lives under `src`:

```text
src/data.py       Load and summarize preprocessed arrays.
src/model.py      EfficientNet-B0 model + two-phase training loop (core logic).
src/tracking.py   Configure MLflow and load training config.
src/train.py      Training entry point (CLI wrapper around src/model.py).
src/evaluate.py   QWK, per-class sensitivity/specificity and ROC-AUC metrics.
```

The default training config is:

```text
configs/efficientnet_b0.json
```

Run training from the repository root:

```bash
uv run python capstone-project/src/train.py
```

This runs the full pipeline: load the preprocessed arrays, build EfficientNet-B0
(frozen ImageNet backbone + a small custom head), warm up the head, then unfreeze
and fine-tune the backbone with QWK-monitored checkpointing, early stopping and
LR reduction, and finally evaluate on the held-out test split. Outputs are written
to `artifacts/` (`best_model.keras`, `final_model.keras`, `test_report.json`) and,
if MLflow is installed, logged to the tracking store.

`src/train.py` and `src/model.py` are equivalent entry points; both accept
`--config` to point at a different config file.

By default MLflow uses a local SQLite tracking store at `capstone-project/mlflow.db` and stores artifacts in `capstone-project/mlartifacts`. To use a shared tracking server, set `MLFLOW_TRACKING_URI` before running training.

```bash
MLFLOW_TRACKING_URI=http://localhost:5000 python capstone-project/src/train.py
```

Start the local MLflow UI with:

```bash
mlflow server --backend-store-uri sqlite:///capstone-project/mlflow.db --default-artifact-root capstone-project/mlartifacts --port 5000
```

## Training runtime

The training runtime is pinned to Python 3.13 with TensorFlow 2.21. TensorFlow 2.21 supports Python 3.10-3.13; Python 3.14 is not a supported TensorFlow runtime for this project.

## Current baseline result

The canonical EfficientNetB0 baseline uses APTOS 2019 only, one-channel
green-channel inputs, class-weighted training, and checkpoint selection on
validation QWK. The current held-out test result is:

```text
QWK                 0.8655
Accuracy            0.7842
Macro sensitivity   0.6431
Macro specificity   0.9492
Macro ROC-AUC       0.9204
```

The weakest recalls are grade 1 mild (`0.5333`) and grade 4 proliferative
(`0.4545`). A separate SMOTE ablation reached similar QWK (`0.8629`) but lower
macro sensitivity (`0.5997`) because grade 1 recall dropped to `0.2667`.
Therefore the class-weighted baseline remains the selected result.

## Evaluation and Grad-CAM artifacts

After training, regenerate evaluation plots from the latest report and MLflow
history:

```bash
uv run python capstone-project/src/plots.py \
  --report capstone-project/artifacts/test_report.json \
  --out-dir capstone-project/artifacts/figures \
  --run-name efficientnet-b0-baseline
```

Generate Grad-CAM figures from the selected checkpoint:

```bash
uv run python capstone-project/src/gradcam.py \
  --model capstone-project/artifacts/best_model.keras \
  --data-dir capstone-project/data/preprocessed \
  --out-dir capstone-project/artifacts/figures \
  --samples-per-grade 6
```

Generate Grad-CAM++ and limited top-k Score-CAM figures:

```bash
uv run python capstone-project/src/gradcam.py \
  --model capstone-project/artifacts/best_model.keras \
  --data-dir capstone-project/data/preprocessed \
  --out-dir capstone-project/artifacts/figures \
  --method gradcampp \
  --samples-per-grade 6

uv run python capstone-project/src/gradcam.py \
  --model capstone-project/artifacts/best_model.keras \
  --data-dir capstone-project/data/preprocessed \
  --out-dir capstone-project/artifacts/figures \
  --method scorecam \
  --samples-per-grade 2 \
  --scorecam-max-channels 32
```

XAI is qualitative in the current scope. The implementation validates that the
Grad-CAM model predictions match the original model predictions before heatmaps
are generated. Score-CAM is limited to the top activated channels to keep the
runtime practical. The current project does not claim lesion-mask overlap
metrics.
