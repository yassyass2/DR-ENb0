# Capstone Project

This folder contains the diabetic retinopathy grading pipeline, saved models, experiment notebooks, and lightweight scripts used for preprocessing, training, and single-sample inference.

## Package CLI

The package exposes the `dr-grading` command:

```bash
uv run dr-grading preprocess
uv run dr-grading train
uv run dr-grading predict --list-models
```

The `predict` subcommand supports:

- `--model` to choose a discovered model number, filename, or full path
- `--split` to choose `train`, `val`, or `test`
- `--index` to choose a specific sample
- `--random` to sample randomly from the selected split
- `--top-k` to control how many class probabilities are printed

Show help:

```bash
uv run dr-grading predict --help
```

## Standalone Prediction Script

There is also a separate script entrypoint:

```bash
uv run python capstone-project/scripts/predict_saved_model.py
```

Useful examples:

```bash
uv run python capstone-project/scripts/predict_saved_model.py --list-models
uv run python capstone-project/scripts/predict_saved_model.py --model 2 --split val --index 10
uv run python capstone-project/scripts/predict_saved_model.py --model SMOTE_IoU.keras --random
```

The script help dynamically shows the currently discovered `.keras` model names in `capstone-project/models/`.

## Dataset Downloader Script

If `dr_preprocessed_data.npz` is stored on Google Drive instead of GitHub, use:

```bash
uv run python capstone-project/scripts/download_preprocessed_npz.py
```

The downloader already knows the repository's default Google Drive link, but it also accepts a raw Google Drive file ID or another share URL:

```bash
uv run python capstone-project/scripts/download_preprocessed_npz.py "<file-id>"
```

By default it saves to:

`capstone-project/models/preprocessed/dr_preprocessed_data.npz`

Use `--force` to overwrite an existing file.

When you run:

```bash
uv run python capstone-project/scripts/predict_saved_model.py
```

the script automatically downloads `dr_preprocessed_data.npz` first if it is not already present locally.

## Model Files

Current saved `.keras` models:

1. `SMOTE.keras`
2. `SMOTE_+_Focal_Loss_IoU.keras`
3. `SMOTE_+_Focal_loss.keras`
4. `SMOTE_IoU.keras`

Additional training notebooks for these variants are stored in the same folder.

## Preprocessed Dataset Archive

The saved-model inference workflow uses:

`capstone-project/models/preprocessed/dr_preprocessed_data.npz`

Expected arrays:

- `X_train`
- `y_train`
- `X_val`
- `y_val`
- `X_test`
- `y_test`

The dataset loader also supports the older directory layout with individual `.npy` files for each split.

## Output

When you run a prediction, the script prints:

- selected model name and path
- chosen split and sample index
- image shape, dtype, and value range
- true label and predicted label
- prediction confidence
- correctness
- margin and entropy
- ranked class probabilities
