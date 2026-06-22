# DR-ENb0

Diabetic retinopathy severity grading on APTOS 2019 fundus images using EfficientNet-B0, SMOTE-based class balancing, focal-loss variants, and Grad-CAM-based qualitative explainability.

## Requirements

- Python `3.13`
- `uv`

## Setup

```bash
git clone https://github.com/yassyass2/DR-ENb0.git
cd DR-ENb0
uv sync
```

## Project Layout

```text
capstone-project/
├── configs/                 # Training configuration
├── models/                  # Saved .keras models and model notebooks
│   └── preprocessed/        # Saved .npz dataset archive for inference
├── scripts/                 # Standalone utility scripts
├── src/dr_grading/          # Package code
└── tests/                   # Lightweight structural tests
pyproject.toml
uv.lock
```

## Main Commands

Preprocess the dataset:

```bash
uv run dr-grading preprocess
```

Train with the default config:

```bash
uv run dr-grading train
```

List saved models:

```bash
uv run python capstone-project/scripts/predict_saved_model.py --list-models
```

Run one saved model on one preprocessed sample:

```bash
uv run python capstone-project/scripts/predict_saved_model.py --model 1 --split test --index 0
```

Interactive mode:

```bash
uv run python capstone-project/scripts/predict_saved_model.py
```

Show script help with the current discovered model names:

```bash
uv run python capstone-project/scripts/predict_saved_model.py --help
```

Download the preprocessed `.npz` archive from Google Drive:

```bash
uv run python capstone-project/scripts/download_preprocessed_npz.py
```

## Saved Models

The `capstone-project/models/` folder currently contains these final saved `.keras` models:

1. `SMOTE.keras`
2. `SMOTE_+_Focal_Loss_IoU.keras`
3. `SMOTE_+_Focal_loss.keras`
4. `SMOTE_IoU.keras`

The prediction script accepts:

- a model number from the discovered list
- a model filename such as `SMOTE_IoU.keras`
- a full filesystem path to a `.keras` file

## Preprocessed Inference Data

The saved-model prediction workflow uses:

`capstone-project/models/preprocessed/dr_preprocessed_data.npz`

It loads these arrays:

- `X_train`, `y_train`
- `X_val`, `y_val`
- `X_test`, `y_test`

By default the prediction script uses the `.npz` archive above, but it can also load a directory of split `.npy` arrays.

If the archive is missing locally, `predict_saved_model.py` now downloads it automatically from the configured Google Drive link before running inference.

You can also download it manually with:

```bash
uv run python capstone-project/scripts/download_preprocessed_npz.py
```

## Prediction Output

For a selected image, the script prints:

- true label and class name
- predicted label and class name
- confidence of the top prediction
- whether the prediction was correct
- prediction margin between the top two classes
- entropy of the probability distribution
- per-class probabilities

## Dataset

[APTOS 2019 Blindness Detection](https://www.kaggle.com/competitions/aptos2019-blindness-detection)

Labels are the diabetic retinopathy severity grades `0` through `4`.

## Contributors

- Adel
- Mark
- Musab
- Ozeir
- Yassine
