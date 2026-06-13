# DR Pipeline Fix Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build one canonical, tested diabetic-retinopathy pipeline that aligns preprocessing, data splits, model input contracts, Grad-CAM, metrics, artifacts, and the paper narrative.

**Architecture:** The pipeline will use a single source of truth for data contracts: APTOS 2019 only, deterministic split metadata, green-channel preprocessing into `float32 [0, 1]`, and one canonical EfficientNetB0 model builder. Grad-CAM must be derived from the exact trained model graph and must be tested by prediction-equivalence before any heatmaps are trusted.

**Tech Stack:** Python 3.13, uv, NumPy, OpenCV, pandas, scikit-learn, imbalanced-learn, TensorFlow/Keras, MLflow, pytest, matplotlib/seaborn.

---

## Scope Decisions

The current objective is pipeline correctness, not final paper results.

The canonical v1 scope is:

- Dataset: APTOS 2019 only.
- Input labels: five ICDR grades, `0` through `4`.
- Preprocessing: black-border crop, aspect-ratio-preserving resize with zero padding, green channel, CLAHE, median blur, normalize to `[0, 1]`.
- Saved image array contract: `float32`, shape `(N, 224, 224, 1)`, values in `[0, 1]`.
- Model input contract: accepts `(224, 224, 1)` green-channel tensors in `[0, 1]`; model itself repeats/scales as needed for EfficientNetB0.
- First training baseline: EfficientNetB0 frozen backbone, class weights, standard augmentation, validation QWK checkpointing.
- SMOTE: separate experiment after baseline; never mixed into the baseline by default.
- MSAG: separate optional experiment after baseline and SMOTE are stable.
- Explainability: Grad-CAM only for v1. No Grad-CAM++, Score-CAM, IDRiD masks, pointing-game accuracy, IoU, EyePACS, or Messidor-2 until implemented.
- Paper alignment: paper must describe exactly this pipeline until additional experiments exist.

## File Structure Map

- `capstone-project/src/preprocessing.py`
  Canonical per-image preprocessing functions and oversampling helper.

- `capstone-project/src/pre_proc_pipeline.py`
  Dataset assembly using canonical preprocessing, deterministic split metadata, and array persistence.

- `capstone-project/main.py`
  CLI entry point for preprocessing and split generation.

- `capstone-project/src/data.py`
  Load and validate preprocessed arrays and split metadata.

- `capstone-project/src/model.py`
  Canonical EfficientNetB0 model builder, training loop, QWK callback, experiment run orchestration.

- `capstone-project/src/gradcam.py`
  Grad-CAM implementation that rebuilds from the exact model graph and verifies prediction equivalence.

- `capstone-project/src/evaluate.py`
  Metrics: QWK, accuracy, macro/per-class sensitivity, specificity, AUC, confusion matrix.

- `capstone-project/src/plots.py`
  Figures from reports and MLflow/history artifacts.

- `capstone-project/configs/efficientnet_b0.json`
  Baseline config with preprocessing and model contract fields.

- `capstone-project/scripts/validate_preprocessed.py`
  Static and visual validation of generated arrays.

- `capstone-project/tests/test_preprocessing_contract.py`
  Unit tests for preprocessing shape, dtype, range, padding, green channel.

- `capstone-project/tests/test_split_contract.py`
  Unit tests for deterministic split metadata and no overlap.

- `capstone-project/tests/test_model_contract.py`
  Unit tests for model input/output shape and preprocessing scale contract.

- `capstone-project/tests/test_gradcam_contract.py`
  Unit tests proving Grad-CAM prediction equivalence and non-empty heatmaps.

- `capstone-project/docs/pipeline_scope.md`
  Human-readable canonical scope and paper-alignment decisions.

---

### Task 1: Add Canonical Pipeline Scope Document

**Files:**
- Create: `capstone-project/docs/pipeline_scope.md`
- Test: no automated test; review with `sed -n '1,220p' capstone-project/docs/pipeline_scope.md`

- [ ] **Step 1: Create the scope document**

Add `capstone-project/docs/pipeline_scope.md` with this exact structure:

```markdown
# Canonical DR Pipeline Scope

## Current Goal

The current goal is to fix and stabilize the technical pipeline before final paper results are reported.

## Included In Pipeline v1

- APTOS 2019 dataset only.
- Five-class diabetic retinopathy severity classification.
- Green-channel preprocessing:
  - black-border crop
  - aspect-ratio-preserving resize with zero padding
  - green-channel extraction
  - CLAHE with `clipLimit=2.0` and `tileGridSize=(8, 8)`
  - median blur with kernel size `3`
  - normalization to `float32 [0, 1]`
- EfficientNetB0 baseline.
- Class-weighted baseline training.
- Optional SMOTE experiment after baseline.
- Grad-CAM only for explainability.
- QWK, accuracy, per-class sensitivity/specificity, macro AUC, confusion matrix.

## Excluded Until Implemented

- EyePACS.
- Messidor-2.
- IDRiD lesion masks.
- Grad-CAM++.
- Score-CAM.
- Pointing-game accuracy.
- IoU against lesion masks.

## Data Contract

Saved preprocessed arrays use shape `(N, 224, 224, 1)`, dtype `float32`, and values in `[0, 1]`.

## Model Contract

The canonical model accepts one-channel green images in `[0, 1]`. The model graph is responsible for converting those tensors to the range and channel count expected by EfficientNetB0.

## Paper Rule

The paper may only claim methods and datasets implemented in this repository. Planned extensions must be described as future work, not completed work.
```

- [ ] **Step 2: Verify the document renders as plain markdown**

Run:

```bash
sed -n '1,220p' capstone-project/docs/pipeline_scope.md
```

Expected: the document prints without malformed tables, raw LaTeX fragments, or missing sections.

- [ ] **Step 3: Commit**

```bash
git add capstone-project/docs/pipeline_scope.md
git commit -m "docs: define canonical DR pipeline scope"
```

---

### Task 2: Test Canonical Preprocessing Contract

**Files:**
- Create: `capstone-project/tests/test_preprocessing_contract.py`
- Modify: `capstone-project/src/preprocessing.py`

- [ ] **Step 1: Write failing preprocessing contract tests**

Create `capstone-project/tests/test_preprocessing_contract.py`:

```python
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import cv2
import numpy as np

PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from src.preprocessing import (
    DEFAULT_IMAGE_SIZE,
    crop_black_border_rgb,
    preprocess_fundus_rgb,
    resize_with_padding,
)


class PreprocessingContractTests(unittest.TestCase):
    def test_crop_black_border_rgb_removes_black_margin(self) -> None:
        image = np.zeros((10, 12, 3), dtype=np.uint8)
        image[2:8, 3:9] = [20, 80, 30]

        cropped = crop_black_border_rgb(image, threshold=10)

        self.assertEqual(cropped.shape, (6, 6, 3))
        self.assertTrue(np.all(cropped == [20, 80, 30]))

    def test_resize_with_padding_preserves_aspect_ratio(self) -> None:
        image = np.full((20, 10, 3), 255, dtype=np.uint8)

        resized = resize_with_padding(image, target_size=DEFAULT_IMAGE_SIZE)

        self.assertEqual(resized.shape, (DEFAULT_IMAGE_SIZE, DEFAULT_IMAGE_SIZE, 3))
        content_mask = resized[:, :, 0] > 0
        ys, xs = np.where(content_mask)
        self.assertEqual(ys.min(), 0)
        self.assertEqual(ys.max(), DEFAULT_IMAGE_SIZE - 1)
        self.assertGreater(xs.min(), 0)
        self.assertLess(xs.max(), DEFAULT_IMAGE_SIZE - 1)

    def test_preprocess_fundus_rgb_outputs_green_float_contract(self) -> None:
        image = np.zeros((80, 120, 3), dtype=np.uint8)
        image[10:70, 20:100, 0] = 20
        image[10:70, 20:100, 1] = np.linspace(30, 220, 80, dtype=np.uint8)[None, :]
        image[10:70, 20:100, 2] = 10

        processed = preprocess_fundus_rgb(image)

        self.assertEqual(processed.shape, (DEFAULT_IMAGE_SIZE, DEFAULT_IMAGE_SIZE, 1))
        self.assertEqual(processed.dtype, np.float32)
        self.assertGreaterEqual(float(processed.min()), 0.0)
        self.assertLessEqual(float(processed.max()), 1.0)
        self.assertGreater(float(processed.max()), float(processed.min()))

    def test_preprocess_fundus_rgb_rejects_non_rgb_input(self) -> None:
        image = np.zeros((80, 120), dtype=np.uint8)

        with self.assertRaises(ValueError):
            preprocess_fundus_rgb(image)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run tests to verify failure**

Run:

```bash
uv run python -m pytest capstone-project/tests/test_preprocessing_contract.py -q
```

Expected: failure because `crop_black_border_rgb`, `resize_with_padding`, and `preprocess_fundus_rgb` do not exist yet.

- [ ] **Step 3: Implement canonical preprocessing functions**

Modify `capstone-project/src/preprocessing.py` by adding these functions while preserving existing `apply_oversampling`:

```python
def crop_black_border_rgb(image: np.ndarray, threshold: int = 10) -> np.ndarray:
    """Crop near-black camera background from an RGB image."""
    if image.ndim != 3 or image.shape[-1] != 3:
        raise ValueError(f"Expected RGB image with shape (H, W, 3), got {image.shape}")

    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    mask = gray > threshold
    if not mask.any():
        return image

    rows = np.where(mask.any(axis=1))[0]
    cols = np.where(mask.any(axis=0))[0]
    y1, y2 = int(rows[0]), int(rows[-1])
    x1, x2 = int(cols[0]), int(cols[-1])
    return image[y1 : y2 + 1, x1 : x2 + 1]


def resize_with_padding(
    image: np.ndarray,
    target_size: int = DEFAULT_IMAGE_SIZE,
    pad_value: int = 0,
) -> np.ndarray:
    """Resize an RGB image to a square canvas without distorting aspect ratio."""
    if image.ndim != 3 or image.shape[-1] != 3:
        raise ValueError(f"Expected RGB image with shape (H, W, 3), got {image.shape}")
    if target_size <= 0:
        raise ValueError("target_size must be positive")

    height, width = image.shape[:2]
    scale = target_size / max(height, width)
    new_width = max(1, int(round(width * scale)))
    new_height = max(1, int(round(height * scale)))

    resized = cv2.resize(
        image,
        (new_width, new_height),
        interpolation=cv2.INTER_AREA,
    )
    canvas = np.full(
        (target_size, target_size, 3),
        pad_value,
        dtype=resized.dtype,
    )
    y_offset = (target_size - new_height) // 2
    x_offset = (target_size - new_width) // 2
    canvas[y_offset : y_offset + new_height, x_offset : x_offset + new_width] = resized
    return canvas


def apply_green_clahe(
    image: np.ndarray,
    clip_limit: float = 2.0,
    tile_grid_size: tuple[int, int] = (8, 8),
) -> np.ndarray:
    """Extract the green channel and apply CLAHE."""
    if image.ndim != 3 or image.shape[-1] != 3:
        raise ValueError(f"Expected RGB image with shape (H, W, 3), got {image.shape}")

    green = image[:, :, 1]
    clahe = cv2.createCLAHE(
        clipLimit=clip_limit,
        tileGridSize=tile_grid_size,
    )
    return clahe.apply(green)


def preprocess_fundus_rgb(
    image: np.ndarray,
    image_size: int = DEFAULT_IMAGE_SIZE,
    crop_threshold: int = 10,
    clip_limit: float = 2.0,
    tile_grid_size: tuple[int, int] = (8, 8),
    median_kernel_size: int = 3,
) -> np.ndarray:
    """
    Canonical per-image preprocessing.

    Input: RGB uint8 image.
    Output: float32 tensor, shape (image_size, image_size, 1), values in [0, 1].
    """
    if image.ndim != 3 or image.shape[-1] != 3:
        raise ValueError(f"Expected RGB image with shape (H, W, 3), got {image.shape}")

    if image.dtype != np.uint8:
        image = np.clip(image, 0, 255).astype(np.uint8)

    image = crop_black_border_rgb(image, threshold=crop_threshold)
    image = resize_with_padding(image, target_size=image_size)
    green = apply_green_clahe(
        image,
        clip_limit=clip_limit,
        tile_grid_size=tile_grid_size,
    )
    denoised = cv2.medianBlur(green, median_kernel_size)
    normalized = denoised.astype(np.float32) / 255.0
    return normalized[..., np.newaxis]


def preprocess_image_path(
    image_path: str | Path,
    image_size: int = DEFAULT_IMAGE_SIZE,
    crop_threshold: int = 10,
    clip_limit: float = 2.0,
    tile_grid_size: tuple[int, int] = (8, 8),
    median_kernel_size: int = 3,
) -> np.ndarray:
    """Read an image from disk and apply canonical preprocessing."""
    image_bgr = cv2.imread(str(image_path))
    if image_bgr is None:
        raise ValueError(f"Could not read image: {image_path}")
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    return preprocess_fundus_rgb(
        image_rgb,
        image_size=image_size,
        crop_threshold=crop_threshold,
        clip_limit=clip_limit,
        tile_grid_size=tile_grid_size,
        median_kernel_size=median_kernel_size,
    )
```

- [ ] **Step 4: Run preprocessing tests**

Run:

```bash
uv run python -m pytest capstone-project/tests/test_preprocessing_contract.py -q
```

Expected: all tests pass.

- [ ] **Step 5: Run existing tests**

Run:

```bash
uv run python -m pytest capstone-project/tests -q
```

Expected: all tests pass.

- [ ] **Step 6: Commit**

```bash
git add capstone-project/src/preprocessing.py capstone-project/tests/test_preprocessing_contract.py
git commit -m "test: define canonical preprocessing contract"
```

---

### Task 3: Make Dataset Splits Reproducible

**Files:**
- Modify: `capstone-project/src/pre_proc_pipeline.py`
- Modify: `capstone-project/main.py`
- Create: `capstone-project/tests/test_split_contract.py`
- Create: `capstone-project/src/split_metadata.py`
- Create: `capstone-project/tests/test_split_metadata.py`

Implementation note: `main.py` currently consumes KaggleHub's existing
train/validation/test CSVs. We therefore keep `make_stratified_splits` as the
tested helper for one-CSV datasets, but the entrypoint records the dataset
provider's actual split CSVs and metadata instead of re-splitting them.

- [x] **Step 1: Write split contract tests**

Create `capstone-project/tests/test_split_contract.py`:

```python
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from src.pre_proc_pipeline import make_stratified_splits


class SplitContractTests(unittest.TestCase):
    def test_make_stratified_splits_is_deterministic_and_disjoint(self) -> None:
        df = pd.DataFrame(
            {
                "id_code": [f"img_{i:03d}" for i in range(100)],
                "diagnosis": [i % 5 for i in range(100)],
            }
        )

        first = make_stratified_splits(
            df,
            id_column="id_code",
            label_column="diagnosis",
            random_state=42,
            val_size=0.1,
            test_size=0.1,
        )
        second = make_stratified_splits(
            df,
            id_column="id_code",
            label_column="diagnosis",
            random_state=42,
            val_size=0.1,
            test_size=0.1,
        )

        self.assertEqual(first["train"]["id_code"].tolist(), second["train"]["id_code"].tolist())
        self.assertEqual(first["val"]["id_code"].tolist(), second["val"]["id_code"].tolist())
        self.assertEqual(first["test"]["id_code"].tolist(), second["test"]["id_code"].tolist())

        train_ids = set(first["train"]["id_code"])
        val_ids = set(first["val"]["id_code"])
        test_ids = set(first["test"]["id_code"])

        self.assertFalse(train_ids & val_ids)
        self.assertFalse(train_ids & test_ids)
        self.assertFalse(val_ids & test_ids)
        self.assertEqual(len(train_ids | val_ids | test_ids), len(df))

    def test_split_sizes_are_80_10_10(self) -> None:
        df = pd.DataFrame(
            {
                "id_code": [f"img_{i:03d}" for i in range(100)],
                "diagnosis": [i % 5 for i in range(100)],
            }
        )

        splits = make_stratified_splits(df)

        self.assertEqual(len(splits["train"]), 80)
        self.assertEqual(len(splits["val"]), 10)
        self.assertEqual(len(splits["test"]), 10)


if __name__ == "__main__":
    unittest.main()
```

- [x] **Step 2: Run split tests to verify failure**

Run:

```bash
uv run python -m pytest capstone-project/tests/test_split_contract.py -q
```

Expected: failure because `make_stratified_splits` is not implemented.

- [x] **Step 3: Implement deterministic split helper**

Add this to `capstone-project/src/pre_proc_pipeline.py`:

```python
def make_stratified_splits(
    df,
    id_column: str = "id_code",
    label_column: str = "diagnosis",
    random_state: int = 42,
    val_size: float = 0.1,
    test_size: float = 0.1,
) -> dict[str, object]:
    """Create deterministic train/val/test splits from one labelled dataframe."""
    if val_size <= 0 or test_size <= 0 or val_size + test_size >= 1:
        raise ValueError("val_size and test_size must be positive and sum to less than 1")

    from sklearn.model_selection import train_test_split

    df = df.sort_values(id_column).reset_index(drop=True)
    train_size = 1.0 - val_size - test_size
    holdout_size = val_size + test_size

    train_df, holdout_df = train_test_split(
        df,
        test_size=holdout_size,
        random_state=random_state,
        stratify=df[label_column],
    )
    relative_test_size = test_size / holdout_size
    val_df, test_df = train_test_split(
        holdout_df,
        test_size=relative_test_size,
        random_state=random_state,
        stratify=holdout_df[label_column],
    )

    return {
        "train": train_df.sort_values(id_column).reset_index(drop=True),
        "val": val_df.sort_values(id_column).reset_index(drop=True),
        "test": test_df.sort_values(id_column).reset_index(drop=True),
        "metadata": {
            "random_state": random_state,
            "train_size": train_size,
            "val_size": val_size,
            "test_size": test_size,
            "id_column": id_column,
            "label_column": label_column,
        },
    }
```

- [x] **Step 4: Update `main.py` to persist split CSVs**

Modify `capstone-project/main.py` so preprocessing writes:

```text
data/preprocessed/splits/train.csv
data/preprocessed/splits/val.csv
data/preprocessed/splits/test.csv
data/preprocessed/splits/metadata.json
```

For one-CSV datasets, use the helper:

```python
splits = make_stratified_splits(
    df,
    id_column=ID_COLUMN,
    label_column=LABEL_COLUMN,
    random_state=42,
    val_size=0.1,
    test_size=0.1,
)
```

Persist metadata:

```python
import json

splits_dir = OUT_DIR / "splits"
splits_dir.mkdir(parents=True, exist_ok=True)
for split in ("train", "val", "test"):
    splits[split].to_csv(splits_dir / f"{split}.csv", index=False)
with (splits_dir / "metadata.json").open("w", encoding="utf-8") as handle:
    json.dump(splits["metadata"], handle, indent=2)
```

For the current KaggleHub APTOS source, `main.py` persists the source-provided
CSV for each split and writes metadata with raw counts, saved-array counts,
class distributions, and whether SMOTE was applied to the saved arrays.

- [x] **Step 5: Run split tests**

Run:

```bash
uv run python -m pytest capstone-project/tests/test_split_contract.py -q
```

Expected: all split tests pass.

- [x] **Step 6: Run all tests**

Run:

```bash
uv run python -m pytest capstone-project/tests -q
```

Expected: all tests pass.

- [x] **Step 7: Commit**

```bash
git add capstone-project/src/pre_proc_pipeline.py capstone-project/main.py capstone-project/tests/test_split_contract.py
git commit -m "feat: add deterministic DR dataset splits"
```

---

### Task 4: Validate Preprocessed Arrays Against Canonical Contract

**Files:**
- Modify: `capstone-project/src/data.py`
- Modify: `capstone-project/scripts/validate_preprocessed.py`
- Modify: `capstone-project/tests/test_training_structure.py`

- [x] **Step 1: Extend data loader tests**

Modify `test_load_preprocessed_data_loads_all_splits_and_summary` in `capstone-project/tests/test_training_structure.py` so its fake arrays use one channel:

```python
"X_train": np.zeros((2, 224, 224, 1), dtype=np.float32),
"X_val": np.ones((1, 224, 224, 1), dtype=np.float32),
"X_test": np.full((1, 224, 224, 1), 0.5, dtype=np.float32),
```

Add assertions:

```python
self.assertEqual(dataset.X_train.dtype, np.float32)
self.assertEqual(dataset.summary.image_shape, (224, 224, 1))
```

- [x] **Step 2: Add array contract validation in `src/data.py`**

Add this function:

```python
def _validate_image_contract(X: np.ndarray, split: str) -> None:
    if X.ndim != 4:
        raise ValueError(f"{split} images must be rank-4, got shape {X.shape}")
    if X.shape[1:] != (224, 224, 1):
        raise ValueError(f"{split} images must have shape (N, 224, 224, 1), got {X.shape}")
    if X.dtype != np.float32:
        raise ValueError(f"{split} images must be float32, got {X.dtype}")
    if not np.isfinite(X).all():
        raise ValueError(f"{split} images contain NaN or Inf")
    x_min = float(X.min())
    x_max = float(X.max())
    if x_min < 0.0 or x_max > 1.0:
        raise ValueError(f"{split} images must be in [0, 1], got [{x_min}, {x_max}]")
```

Call it inside `load_preprocessed_data` for train, val, and test after alignment checks:

```python
_validate_image_contract(arrays["X_train"], "train")
_validate_image_contract(arrays["X_val"], "val")
_validate_image_contract(arrays["X_test"], "test")
```

- [x] **Step 3: Update validator expected shape**

Modify `capstone-project/scripts/validate_preprocessed.py`:

```python
EXPECTED_SHAPE = (224, 224, 1)
EXPECTED_DTYPE = np.float32
```

Update sample export so grayscale arrays render correctly:

```python
grid = np.concatenate([X[i].squeeze(axis=-1) for i in idx], axis=1)
grid = (np.clip(grid, 0.0, 1.0) * 255).round().astype(np.uint8)
out = SAMPLES_DIR / f"{split}_samples.png"
cv2.imwrite(str(out), grid)
```

- [x] **Step 4: Run tests**

Run:

```bash
uv run python -m pytest capstone-project/tests -q
```

Expected: all tests pass.

- [x] **Step 5: Commit**

```bash
git add capstone-project/src/data.py capstone-project/scripts/validate_preprocessed.py capstone-project/tests/test_training_structure.py
git commit -m "test: enforce preprocessed array contract"
```

---

### Task 5: Canonical EfficientNetB0 Model Input Contract

**Files:**
- Modify: `capstone-project/src/model.py`
- Create: `capstone-project/tests/test_model_contract.py`
- Modify: `capstone-project/configs/efficientnet_b0.json`

- [x] **Step 1: Write model contract tests**

Create `capstone-project/tests/test_model_contract.py`:

```python
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from src.model import build_model


class ModelContractTests(unittest.TestCase):
    def test_model_accepts_one_channel_normalized_input(self) -> None:
        model, base_model = build_model(
            num_classes=5,
            image_size=224,
            dropout=0.3,
            weights=None,
            augment=False,
            seed=42,
        )

        self.assertEqual(model.input_shape, (None, 224, 224, 1))
        self.assertEqual(model.output_shape, (None, 5))
        self.assertFalse(base_model.trainable)

        batch = np.random.default_rng(0).random((2, 224, 224, 1), dtype=np.float32)
        predictions = model.predict(batch, verbose=0)

        self.assertEqual(predictions.shape, (2, 5))
        np.testing.assert_allclose(predictions.sum(axis=1), np.ones(2), atol=1e-5)


if __name__ == "__main__":
    unittest.main()
```

- [x] **Step 2: Run model contract test to verify failure**

Run:

```bash
uv run python -m pytest capstone-project/tests/test_model_contract.py -q
```

Expected: failure because current model expects three channels.

- [x] **Step 3: Modify `build_model` to accept one-channel input**

In `capstone-project/src/model.py`, change the input shape:

```python
inputs = keras.Input(shape=(image_size, image_size, 1), name="green_image")
```

Replace the early preprocessing block with:

```python
x = inputs

# Saved arrays are one-channel float32 in [0, 1]. EfficientNetB0 expects
# ImageNet-style three-channel input and contains its own internal rescaling.
x = layers.Rescaling(255.0, name="to_efficientnet_range")(x)
x = layers.Concatenate(axis=-1, name="repeat_green_to_rgb")([x, x, x])

if augment:
    x = build_augmentation(seed)(x)
```

Keep EfficientNetB0 as:

```python
input_shape=(image_size, image_size, 3)
```

- [x] **Step 4: Update config**

Modify `capstone-project/configs/efficientnet_b0.json`:

```json
"data": {
  "preprocessed_dir": "data/preprocessed",
  "image_size": 224,
  "num_classes": 5,
  "channels": 1,
  "value_range": "[0,1]"
}
```

Set baseline dropout to match team experiment baseline unless intentionally changed:

```json
"dropout": 0.3
```

- [x] **Step 5: Run model test**

Run:

```bash
uv run python -m pytest capstone-project/tests/test_model_contract.py -q
```

Expected: pass.

- [x] **Step 6: Run all tests**

Run:

```bash
uv run python -m pytest capstone-project/tests -q
```

Expected: all tests pass.

- [x] **Step 7: Commit**

```bash
git add capstone-project/src/model.py capstone-project/configs/efficientnet_b0.json capstone-project/tests/test_model_contract.py
git commit -m "feat: make EfficientNetB0 accept canonical green-channel input"
```

---

### Task 6: Fix Grad-CAM Against the Exact Model Graph

**Files:**
- Modify: `capstone-project/src/gradcam.py`
- Create: `capstone-project/tests/test_gradcam_contract.py`

- [ ] **Step 1: Write Grad-CAM contract tests**

Create `capstone-project/tests/test_gradcam_contract.py`:

```python
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from src.gradcam import build_gradcam_model, compute_gradcam
from src.model import build_model


class GradCAMContractTests(unittest.TestCase):
    def test_gradcam_predictions_match_original_model(self) -> None:
        model, _ = build_model(
            num_classes=5,
            image_size=224,
            dropout=0.3,
            weights=None,
            augment=False,
            seed=42,
        )
        gradcam_model = build_gradcam_model(model)
        images = np.random.default_rng(1).random((2, 224, 224, 1), dtype=np.float32)

        original_predictions = model.predict(images, verbose=0)
        _, gradcam_predictions = gradcam_model.predict(images, verbose=0)

        np.testing.assert_allclose(
            gradcam_predictions,
            original_predictions,
            atol=1e-6,
        )

    def test_compute_gradcam_returns_resized_heatmaps(self) -> None:
        model, _ = build_model(
            num_classes=5,
            image_size=224,
            dropout=0.3,
            weights=None,
            augment=False,
            seed=42,
        )
        gradcam_model = build_gradcam_model(model)
        images = np.random.default_rng(2).random((1, 224, 224, 1), dtype=np.float32)

        heatmaps, probabilities = compute_gradcam(gradcam_model, images)

        self.assertEqual(heatmaps.shape, (1, 224, 224))
        self.assertEqual(probabilities.shape, (1, 5))
        self.assertGreaterEqual(float(heatmaps.min()), 0.0)
        self.assertLessEqual(float(heatmaps.max()), 1.0)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run Grad-CAM tests to verify failure**

Run:

```bash
uv run python -m pytest capstone-project/tests/test_gradcam_contract.py -q
```

Expected: failure if current `build_gradcam_model` skips model layers or assumes three-channel input.

- [ ] **Step 3: Replace layer replay with graph traversal**

Modify `capstone-project/src/gradcam.py` so `build_gradcam_model` replays all layers from the input until the EfficientNet backbone, captures the backbone output, then replays every downstream layer in order.

Implementation:

```python
def _find_backbone(model: keras.Model) -> keras.Model:
    for layer in model.layers:
        if isinstance(layer, keras.Model) and "efficientnet" in layer.name.lower():
            return layer
    raise ValueError("Could not find EfficientNet backbone in model")


def build_gradcam_model(model: keras.Model) -> keras.Model:
    """
    Build a model returning `(conv_features, predictions)` while preserving
    exact prediction equivalence with the original model.
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

        if isinstance(layer, keras.layers.Dropout):
            x = layer(x, training=False)
        elif isinstance(layer, keras.layers.RandomFlip | keras.layers.RandomRotation | keras.layers.RandomZoom):
            x = layer(x, training=False)
        else:
            x = layer(x)

    if not seen_backbone or conv_features is None:
        raise ValueError("Could not route through EfficientNet backbone")

    gradcam_model = keras.Model(
        inputs,
        [conv_features, x],
        name="gradcam_model",
    )
    return gradcam_model
```

If Python rejects union pattern matching in `isinstance`, replace that line with:

```python
elif layer.__class__.__name__.startswith("Random"):
    x = layer(x, training=False)
```

- [ ] **Step 4: Update overlay for one-channel images**

Modify `overlay_heatmap`:

```python
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
```

- [ ] **Step 5: Run Grad-CAM tests**

Run:

```bash
uv run python -m pytest capstone-project/tests/test_gradcam_contract.py -q
```

Expected: pass.

- [ ] **Step 6: Run all tests**

Run:

```bash
uv run python -m pytest capstone-project/tests -q
```

Expected: all tests pass.

- [ ] **Step 7: Commit**

```bash
git add capstone-project/src/gradcam.py capstone-project/tests/test_gradcam_contract.py
git commit -m "fix: make Grad-CAM match canonical model predictions"
```

---

### Task 7: Align Training Selection With QWK

**Files:**
- Modify: `capstone-project/src/model.py`
- Modify: `capstone-project/configs/efficientnet_b0.json`
- Modify: `capstone-project/tests/test_training_structure.py`

- [ ] **Step 1: Extend config test**

In `capstone-project/tests/test_training_structure.py`, add:

```python
self.assertEqual(config["training"]["checkpoint_monitor"], "val_qwk")
self.assertEqual(config["training"]["checkpoint_mode"], "max")
```

- [ ] **Step 2: Update config**

Modify `capstone-project/configs/efficientnet_b0.json`:

```json
"training": {
  "batch_size": 16,
  "epochs": 30,
  "warmup_epochs": 0,
  "fine_tune_epochs": 30,
  "learning_rate": 0.001,
  "fine_tune_learning_rate": 0.00001,
  "early_stopping_patience": 8,
  "augment": true,
  "seed": 42,
  "class_weight": "balanced",
  "checkpoint_monitor": "val_qwk",
  "checkpoint_mode": "max"
}
```

- [ ] **Step 3: Use config monitor fields in callbacks**

In `train_model`, add:

```python
checkpoint_monitor = training.get("checkpoint_monitor", "val_qwk")
checkpoint_mode = training.get("checkpoint_mode", "max")
```

Use those in `ModelCheckpoint`, `EarlyStopping`, and `ReduceLROnPlateau`:

```python
monitor=checkpoint_monitor,
mode=checkpoint_mode,
```

- [ ] **Step 4: Add class-weight helper**

Add:

```python
def balanced_class_weights(y_train: np.ndarray) -> dict[int, float]:
    """Compute sklearn-style balanced class weights."""
    labels, counts = np.unique(y_train, return_counts=True)
    total = int(counts.sum())
    n_classes = int(len(labels))
    return {
        int(label): float(total / (n_classes * count))
        for label, count in zip(labels, counts, strict=True)
    }
```

Use it in both `model.fit` calls when `training.get("class_weight") == "balanced"`:

```python
class_weight = (
    balanced_class_weights(y_train)
    if training.get("class_weight") == "balanced"
    else None
)
```

Pass:

```python
class_weight=class_weight,
```

- [ ] **Step 5: Run tests**

Run:

```bash
uv run python -m pytest capstone-project/tests -q
```

Expected: all tests pass.

- [ ] **Step 6: Commit**

```bash
git add capstone-project/src/model.py capstone-project/configs/efficientnet_b0.json capstone-project/tests/test_training_structure.py
git commit -m "feat: select baseline checkpoints by validation QWK"
```

---

### Task 8: Make SMOTE a Separate Experiment

**Files:**
- Create: `capstone-project/configs/efficientnet_b0_smote.json`
- Modify: `capstone-project/src/pre_proc_pipeline.py`
- Modify: `capstone-project/main.py`
- Modify: `capstone-project/readme.md`

- [ ] **Step 1: Create SMOTE config**

Create `capstone-project/configs/efficientnet_b0_smote.json`:

```json
{
  "experiment": {
    "name": "aptos-dr-classification",
    "run_name": "efficientnet-b0-smote-green"
  },
  "data": {
    "preprocessed_dir": "data/preprocessed_smote",
    "image_size": 224,
    "num_classes": 5,
    "channels": 1,
    "value_range": "[0,1]"
  },
  "model": {
    "name": "efficientnet_b0",
    "weights": "imagenet",
    "dropout": 0.3,
    "fine_tune_at": null
  },
  "training": {
    "batch_size": 16,
    "epochs": 30,
    "warmup_epochs": 0,
    "fine_tune_epochs": 30,
    "learning_rate": 0.001,
    "fine_tune_learning_rate": 0.00001,
    "early_stopping_patience": 8,
    "augment": true,
    "seed": 42,
    "class_weight": null,
    "checkpoint_monitor": "val_qwk",
    "checkpoint_mode": "max"
  },
  "preprocessing": {
    "smote_train": true,
    "smote_random_state": 42
  }
}
```

- [ ] **Step 2: Ensure baseline preprocessing does not apply SMOTE**

In `main.py`, keep baseline output at:

```python
OUT_DIR = Path("data/preprocessed")
APPLY_SMOTE = False
```

Add a CLI flag:

```python
import argparse

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Preprocess APTOS 2019 data.")
    parser.add_argument("--out-dir", type=Path, default=Path("data/preprocessed"))
    parser.add_argument("--smote-train", action="store_true")
    return parser.parse_args()
```

Use:

```python
args = parse_args()
out_dir = args.out_dir
apply_smote = args.smote_train
```

- [ ] **Step 3: Apply SMOTE only when explicitly requested**

In the train split branch:

```python
if split == "train" and apply_smote:
    X, y = apply_oversampling(X, y)
```

Validation and test must never be oversampled.

- [ ] **Step 4: Update README commands**

Add:

```markdown
Baseline preprocessing:

```bash
uv run python capstone-project/main.py --out-dir data/preprocessed
```

SMOTE experiment preprocessing:

```bash
cd capstone-project
uv run python main.py --out-dir data/preprocessed_smote --smote-train
```
```

- [ ] **Step 5: Run tests**

Run:

```bash
uv run python -m pytest capstone-project/tests -q
```

Expected: all tests pass.

- [ ] **Step 6: Commit**

```bash
git add capstone-project/configs/efficientnet_b0_smote.json capstone-project/main.py capstone-project/src/pre_proc_pipeline.py capstone-project/readme.md
git commit -m "feat: separate SMOTE preprocessing experiment"
```

---

### Task 9: Regenerate Data and Validate Locally

**Files:**
- Generated but not committed: `capstone-project/data/preprocessed/*.npy`
- Generated but not committed: `capstone-project/data/preprocessed/splits/*.csv`
- Generated sample images: `capstone-project/data/preprocessed/_samples/*.png`

- [ ] **Step 1: Generate baseline preprocessed arrays**

Run from `capstone-project`:

```bash
uv run python main.py --out-dir data/preprocessed
```

Expected:

```text
Saved X_train (..., 224, 224, 1) (float32)
Saved y_train (...)
Saved X_val (..., 224, 224, 1) (float32)
Saved y_val (...)
Saved X_test (..., 224, 224, 1) (float32)
Saved y_test (...)
```

- [ ] **Step 2: Validate baseline arrays**

Run:

```bash
uv run python scripts/validate_preprocessed.py
```

Expected: no failures. Warnings are acceptable only if they are manually understood and documented in the final implementation notes.

- [ ] **Step 3: Generate SMOTE arrays**

Run:

```bash
uv run python main.py --out-dir data/preprocessed_smote --smote-train
```

Expected: train split balanced; val/test natural.

- [ ] **Step 4: Validate SMOTE arrays**

Run:

```bash
uv run python scripts/validate_preprocessed.py --data-dir data/preprocessed_smote
```

If `--data-dir` does not exist yet, first modify the validator to accept:

```python
parser.add_argument("--data-dir", type=Path, default=Path("data/preprocessed"))
```

- [ ] **Step 5: Commit only validator changes and sample images if intentionally tracked**

Generated `.npy` files should stay ignored unless the team explicitly decides otherwise.

```bash
git status --short
git add capstone-project/scripts/validate_preprocessed.py
git commit -m "chore: allow validating alternate preprocessed data dirs"
```

---

### Task 10: Run Baseline Training and Produce Artifacts

**Files:**
- Generated: `capstone-project/artifacts/test_report.json`
- Generated: `capstone-project/artifacts/figures/*.png`
- Generated but ignored: `capstone-project/artifacts/*.keras`
- Generated but ignored: `capstone-project/mlflow.db`, `capstone-project/mlartifacts`

- [ ] **Step 1: Train canonical baseline**

Run from repository root:

```bash
uv run python capstone-project/src/train.py --config capstone-project/configs/efficientnet_b0.json
```

Expected:

```text
Loaded data: train=..., val=..., test=..., image_shape=(224, 224, 1)
val_qwk: ...
Wrote test report to .../artifacts/test_report.json
```

- [ ] **Step 2: Inspect test report**

Run:

```bash
python -m json.tool capstone-project/artifacts/test_report.json
```

Expected keys:

```text
qwk
accuracy
macro_sensitivity
macro_specificity
macro_auc
per_class
confusion_matrix
num_samples
```

- [ ] **Step 3: Generate plots**

Run:

```bash
uv run python capstone-project/src/plots.py
```

Expected files:

```text
capstone-project/artifacts/figures/confusion_matrix.png
capstone-project/artifacts/figures/per_class_metrics.png
capstone-project/artifacts/figures/training_curves.png
```

- [ ] **Step 4: Commit report and figures only after reviewing consistency**

```bash
git add capstone-project/artifacts/test_report.json capstone-project/artifacts/figures
git commit -m "artifacts: add canonical baseline evaluation outputs"
```

---

### Task 11: Generate and Verify Grad-CAM Artifacts

**Files:**
- Generated: `capstone-project/artifacts/figures/gradcam_*.png`
- Modify if needed: `capstone-project/src/gradcam.py`

- [ ] **Step 1: Run Grad-CAM tests before generating artifacts**

Run:

```bash
uv run python -m pytest capstone-project/tests/test_gradcam_contract.py -q
```

Expected: pass.

- [ ] **Step 2: Generate Grad-CAM figures**

Run:

```bash
uv run python capstone-project/src/gradcam.py --samples-per-grade 6
```

Expected files:

```text
capstone-project/artifacts/figures/gradcam_grade_0.png
capstone-project/artifacts/figures/gradcam_grade_1.png
capstone-project/artifacts/figures/gradcam_grade_2.png
capstone-project/artifacts/figures/gradcam_grade_3.png
capstone-project/artifacts/figures/gradcam_grade_4.png
capstone-project/artifacts/figures/gradcam_summary.png
```

- [ ] **Step 3: Manually inspect Grad-CAM outputs**

Open the generated PNGs and check:

- Heatmaps are not blank.
- Heatmaps are not always concentrated on padding or corners.
- Correct and incorrect predictions are both distinguishable.
- Captions show true/predicted grades.

- [ ] **Step 4: Commit Grad-CAM artifacts**

```bash
git add capstone-project/src/gradcam.py capstone-project/artifacts/figures/gradcam_*.png
git commit -m "artifacts: add verified Grad-CAM visualizations"
```

---

### Task 12: Paper Alignment Pass

**Files:**
- Modify source paper files if present in repo.
- Modify: `capstone-project/docs/preprocessing_methodology.tex`
- Modify: `capstone-project/readme.md`

- [ ] **Step 1: Remove unsupported claims from paper text**

Search current paper source for unsupported terms:

```bash
rg -n "EyePACS|Messidor|IDRiD|Grad-CAM\\+\\+|Score-CAM|pointing|Intersection-over-Union|IoU" .
```

Any occurrence must be either removed or moved to a future-work sentence.

- [ ] **Step 2: Align preprocessing section**

The paper preprocessing section must say:

```text
Images are converted to RGB, black borders are cropped using a threshold of 10,
the fundus is resized to 224 x 224 with aspect-ratio-preserving zero padding,
the green channel is extracted, CLAHE is applied with clipLimit 2.0 and an
8 x 8 tile grid, median filtering is applied with a 3 x 3 kernel, and pixel
values are normalized to float32 [0, 1].
```

- [ ] **Step 3: Align model section**

The paper model section must say:

```text
The model accepts one-channel preprocessed green images in [0, 1]. Inside the
model, the channel is scaled to the EfficientNetB0 input range and repeated
to three channels for compatibility with ImageNet-pretrained EfficientNetB0.
```

- [ ] **Step 4: Align explainability section**

The paper explainability section must say:

```text
Grad-CAM is generated from the trained EfficientNetB0 model by exposing the
final convolutional activation and replaying the exact classifier head. The
implementation is validated by checking that Grad-CAM model predictions match
the original model predictions before heatmaps are generated.
```

- [ ] **Step 5: Fix malformed LaTeX table**

The broken table fragment:

```text
@p1.4cm p3.4cm Y@
```

must be replaced with a valid LaTeX table, for example:

```latex
\begin{tabular}{p{1.4cm}p{3.4cm}p{8.0cm}}
\toprule
RQ & Evaluation dimension & Primary instruments \\
\midrule
RQ1 & Classification performance & Quadratic Weighted Kappa, per-class sensitivity and specificity, macro AUC, and accuracy. \\
RQ2 & Explainability quality & Grad-CAM visual inspection against expected retinal regions and failure-mode review. \\
RQ3 & Imbalance sensitivity & Difference in QWK and per-class sensitivity between baseline and SMOTE experiment. \\
\bottomrule
\end{tabular}
```

- [ ] **Step 6: Rebuild PDF and visually inspect pages**

Run the project’s LaTeX build command. If no command exists, add one to the README.

Expected visual checks:

- No raw LaTeX appears in the PDF.
- No empty headings remain.
- Results are either clearly marked pending or populated from generated artifacts.
- Tables fit page width.

- [ ] **Step 7: Commit paper alignment**

```bash
git add capstone-project/docs/preprocessing_methodology.tex capstone-project/readme.md
git commit -m "docs: align paper text with canonical pipeline"
```

---

### Task 13: Final Verification Checklist

**Files:**
- No new files required unless verification output is documented.

- [ ] **Step 1: Check branch**

Run:

```bash
git branch --show-current
```

Expected:

```text
musab/pipeline-fix-plan
```

- [ ] **Step 2: Check tests**

Run:

```bash
uv run python -m pytest capstone-project/tests -q
```

Expected: all tests pass.

- [ ] **Step 3: Check preprocessing validator**

Run from `capstone-project`:

```bash
uv run python scripts/validate_preprocessed.py
```

Expected: no failures.

- [ ] **Step 4: Check artifacts**

Run:

```bash
python -m json.tool capstone-project/artifacts/test_report.json
ls capstone-project/artifacts/figures
```

Expected: report is valid JSON and figures are present.

- [ ] **Step 5: Check unsupported paper claims**

Run:

```bash
rg -n "EyePACS|Messidor|IDRiD|Grad-CAM\\+\\+|Score-CAM|pointing|Intersection-over-Union|IoU" capstone-project
```

Expected: no unsupported completed-work claims. Future-work mentions are acceptable only if explicitly phrased as future work.

- [ ] **Step 6: Check git state**

Run:

```bash
git status --short --branch
```

Expected: no unintended generated files staged or unstaged.

---

## Execution Recommendation

Execute Tasks 1 through 8 before running any expensive training. This prevents wasting training time on a pipeline whose contracts may still change.

After Task 8:

1. Regenerate data.
2. Validate arrays.
3. Train baseline.
4. Generate metrics and plots.
5. Generate Grad-CAM.
6. Align paper.

Do not merge notebook experiments directly into the repo. Extract only validated ideas:

- black-border crop
- aspect-ratio padding
- green-channel preprocessing
- class-weight baseline
- optional SMOTE experiment
- optional MSAG experiment after baseline stability

The Drive notebook remains useful as experimental evidence, but the repository must become the source of truth.
