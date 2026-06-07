# APTOS 2019 Blindness Detection - Capstone Project

## Team

**Course:** AI for Health - Group 5  
**Members:** Yassine Abderrazik, Adel Atzouza, Ozeir Moradi, Musab Sivrikaya

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
src/model.py      Placeholder for the model architecture.
src/tracking.py   Configure MLflow and load training config.
src/train.py      Small MLflow template for logging dataset/config metadata.
src/evaluate.py   Placeholder for the final evaluation code.
```

The default training config is:

```text
configs/efficientnet_b0.json
```

Run training from the repository root:

```bash
python capstone-project/src/train.py
```

For now this only starts an MLflow run and logs config + dataset metadata. The
actual model, training loop, callbacks, and metrics should be added when the
training experiments start.

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
