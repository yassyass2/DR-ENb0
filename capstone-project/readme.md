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
python main.py
python scripts/validate_preprocessed.py
```

This writes the following files under `data/preprocessed`:

```text
X_train.npy, y_train.npy
X_val.npy, y_val.npy
X_test.npy, y_test.npy
```

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
python capstone-project/src/train.py
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
