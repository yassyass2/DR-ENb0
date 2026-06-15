# APTOS 2019 Diabetic Retinopathy Grading

This repository contains the Group 5 AI for Health capstone pipeline for five-class diabetic retinopathy grading on APTOS 2019 fundus images. The project is organised as a reproducible research codebase: preprocessing, model training, evaluation, experiment tracking and explainability are available from one Python package, `dr_grading`.

The model pipeline uses an EfficientNet-B0 backbone with a custom classification head. Images are resized to 224 x 224, enhanced with CLAHE on the LAB lightness channel, denoised with a 3 x 3 median filter and normalised to `[0, 1]`. The training split is balanced with SMOTE after preprocessing. Validation and test splits keep their natural class distribution.

## Project Layout

```text
capstone-project/
  configs/                 Training configuration
  docs/                    Methodology notes and paper assets
  scripts/                 One-off QA utilities
  src/dr_grading/          Research pipeline package
  tests/                   Contract tests for structure and training utilities
  artifacts/               Generated reports, figures and model outputs
  data/                    Local generated data; not a source artifact
```

The old script paths under `capstone-project/src/*.py` are kept as compatibility wrappers. New work should import from `dr_grading` or use the `dr-grading` CLI.

## Reproducible Commands

Run from the repository root.

```bash
uv run dr-grading preprocess
uv run python capstone-project/scripts/validate_preprocessed.py
uv run dr-grading train --config capstone-project/configs/efficientnet_b0.json
```

Preprocessing writes `preprocessing_manifest.json` next to the generated arrays. The manifest records the dataset location, package version, Git commit, preprocessing parameters, split distributions and the fact that SMOTE was applied only to the training split.

Training writes each run under a timestamped artifact directory:

```text
capstone-project/artifacts/<run-name>-<YYYYMMDDTHHMMSSZ>/
  best_model.keras
  final_model.keras
  test_report.json
```

The artifact directory is also logged to MLflow as `artifact_run_dir`, so the tracking run and filesystem outputs can be matched later.

Legacy commands still work:

```bash
uv run python capstone-project/main.py
uv run python capstone-project/src/train.py
```

## Experiment Tracking

By default, MLflow uses a local SQLite tracking store:

```text
capstone-project/mlflow.db
capstone-project/mlartifacts/
```

For a shared tracking server:

```bash
MLFLOW_TRACKING_URI=http://localhost:5000 \
  uv run dr-grading train --config capstone-project/configs/efficientnet_b0.json
```

Start a local UI:

```bash
uv run mlflow server \
  --backend-store-uri sqlite:///capstone-project/mlflow.db \
  --default-artifact-root capstone-project/mlartifacts \
  --port 5000
```

## Evaluation

The main selection metric is Quadratic Weighted Kappa, matching the ordinal nature of DR severity. The test report also includes accuracy, macro sensitivity, macro specificity, macro ROC-AUC, per-class sensitivity/specificity/AUC and the confusion matrix.

Current tracked figures and historical reports are generated artifacts, not the paper-final source of truth. Before using numbers in the paper, regenerate preprocessing, training, evaluation, plots and Grad-CAM from the same commit and MLflow run.

## Runtime

The runtime is pinned to Python `>=3.13,<3.14` with TensorFlow `>=2.21,<2.22`. Python 3.14 is intentionally excluded because it is outside the supported TensorFlow range for this project.
