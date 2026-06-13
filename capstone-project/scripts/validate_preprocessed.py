"""
Sanity-check the preprocessed .npy arrays in ``data/preprocessed`` before
committing them or training a model.

Runs two cheap checks that together catch ~90% of preprocessing bugs:

  1. Static checks   -- shapes, dtypes, value ranges, label ranges, label
                        alignment, SMOTE balance on train, natural imbalance
                        on val/test.
  2. Visual check    -- save a small image grid per split so a human can
                        eyeball that crop/resize/green+CLAHE produced sensible
                        retina images and that labels match.

Usage:
    python scripts/validate_preprocessed.py

Exits with status 0 if all checks pass, 1 otherwise.
"""

from __future__ import annotations

import sys
import json
from pathlib import Path

import cv2
import numpy as np

DATA_DIR = Path("data/preprocessed")
SAMPLES_DIR = DATA_DIR / "_samples"
SPLITS = ["train", "val", "test"]
EXPECTED_SHAPE = (224, 224, 1)
EXPECTED_DTYPE = np.float32  # pipeline normalises pixels to [0, 1]
EXPECTED_CLASSES = {0, 1, 2, 3, 4}  # APTOS-2019 diagnosis labels


class Report:
    def __init__(self) -> None:
        self.failures: list[str] = []
        self.warnings: list[str] = []

    def fail(self, msg: str) -> None:
        self.failures.append(msg)
        print(f"  [FAIL] {msg}")

    def warn(self, msg: str) -> None:
        self.warnings.append(msg)
        print(f"  [WARN] {msg}")

    def ok(self, msg: str) -> None:
        print(f"  [ OK ] {msg}")


def load_split(split: str, report: Report) -> tuple[np.ndarray, np.ndarray] | None:
    x_path = DATA_DIR / f"X_{split}.npy"
    y_path = DATA_DIR / f"y_{split}.npy"
    if not x_path.exists() or not y_path.exists():
        report.fail(f"missing {x_path.name} or {y_path.name}")
        return None
    return np.load(x_path), np.load(y_path)


def load_smote_flag(report: Report) -> bool:
    metadata_path = DATA_DIR / "splits" / "metadata.json"
    if not metadata_path.exists():
        report.warn(f"missing {metadata_path}; assuming baseline without SMOTE")
        return False
    with metadata_path.open("r", encoding="utf-8") as handle:
        metadata = json.load(handle)
    return bool(metadata.get("smote_train", False))


def check_split(
    split: str,
    X: np.ndarray,
    y: np.ndarray,
    report: Report,
    smote_train: bool,
) -> None:
    # Shape / dtype
    if X.shape[1:] != EXPECTED_SHAPE:
        report.fail(f"X_{split} shape {X.shape[1:]} != expected {EXPECTED_SHAPE}")
    else:
        report.ok(f"X_{split} per-image shape {X.shape[1:]}")

    if X.dtype != EXPECTED_DTYPE:
        report.fail(f"X_{split} dtype {X.dtype} != {EXPECTED_DTYPE} "
                    f"(pipeline normalises to float32 [0,1])")
    else:
        report.ok(f"X_{split} dtype {X.dtype}")

    # Length alignment
    if X.shape[0] != y.shape[0]:
        report.fail(f"{split}: X has {X.shape[0]} images but y has {y.shape[0]} labels")
        return
    if X.shape[0] == 0:
        report.fail(f"{split}: empty split (0 samples)")
        return
    report.ok(f"{split}: {X.shape[0]} samples, X/y aligned")

    # Value range -- the pipeline normalises pixels to [0, 1]
    x_min, x_max = float(X.min()), float(X.max())
    if x_min < 0.0 or x_max > 1.0:
        report.fail(f"X_{split} values out of [0,1]: [{x_min:.3f},{x_max:.3f}] "
                    f"-- the pipeline should normalise pixels to [0,1]")
    elif x_max == 0.0:
        report.warn(f"X_{split} is entirely zero -- images look blank")
    else:
        report.ok(f"X_{split} value range [{x_min:.3f},{x_max:.3f}]")

    # No NaN/Inf -- the normalised float arrays can carry them if a bug slips in
    if X.dtype.kind == "f" and not np.isfinite(X).all():
        report.fail(f"X_{split} contains NaN or Inf")

    # Labels
    classes, counts = np.unique(y, return_counts=True)
    class_dist = dict(zip(classes.tolist(), counts.tolist()))
    unknown = set(classes.tolist()) - EXPECTED_CLASSES
    if unknown:
        report.fail(f"y_{split} has unexpected classes: {unknown}")
    missing = EXPECTED_CLASSES - set(classes.tolist())
    if missing:
        report.warn(f"y_{split} missing classes: {missing}")
    report.ok(f"y_{split} class distribution: {class_dist}")

    # SMOTE expectation on train only when this output directory opted into it.
    if split == "train":
        if smote_train and len(set(counts.tolist())) == 1:
            report.ok("train is class-balanced (SMOTE applied)")
        elif smote_train:
            report.fail(f"train is NOT class-balanced -- SMOTE may not have run. "
                        f"Counts: {class_dist}")
        elif len(set(counts.tolist())) == 1 and len(counts) > 1:
            report.warn("train is perfectly balanced although SMOTE metadata is false")
        else:
            report.ok("train keeps natural class distribution (baseline)")
    else:
        if len(set(counts.tolist())) == 1 and len(counts) > 1:
            report.warn(f"{split} is perfectly balanced -- unexpected for a natural "
                        f"DR split. SMOTE may have leaked into {split}.")


def save_samples(split: str, X: np.ndarray, y: np.ndarray, n: int = 6) -> Path:
    SAMPLES_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    idx = rng.choice(len(X), size=min(n, len(X)), replace=False)
    grid = np.concatenate([X[i].squeeze(axis=-1) for i in idx], axis=1)
    # X is float32 in [0, 1]; scale back to 8-bit for PNG export.
    grid = (np.clip(grid, 0.0, 1.0) * 255).round().astype(np.uint8)
    out = SAMPLES_DIR / f"{split}_samples.png"
    cv2.imwrite(str(out), grid)
    labels = [int(y[i]) for i in idx]
    print(f"  wrote {out}  (labels left->right: {labels})")
    return out


def main() -> int:
    if not DATA_DIR.exists():
        print(f"[FAIL] {DATA_DIR} does not exist. Run main.py first.")
        return 1

    report = Report()
    smote_train = load_smote_flag(report)
    loaded: dict[str, tuple[np.ndarray, np.ndarray]] = {}

    for split in SPLITS:
        print(f"\n=== {split.upper()} ===")
        data = load_split(split, report)
        if data is None:
            continue
        loaded[split] = data
        check_split(split, data[0], data[1], report, smote_train=smote_train)

    print("\n=== SAMPLES ===")
    for split, (X, y) in loaded.items():
        save_samples(split, X, y)
    print(f"  open {SAMPLES_DIR} and check the retinas look reasonable "
          f"(resized, contrast-enhanced, denoised).")

    print("\n=== SUMMARY ===")
    if report.failures:
        print(f"  {len(report.failures)} failure(s):")
        for msg in report.failures:
            print(f"    - {msg}")
    if report.warnings:
        print(f"  {len(report.warnings)} warning(s):")
        for msg in report.warnings:
            print(f"    - {msg}")
    if not report.failures and not report.warnings:
        print("  all checks passed.")
    elif not report.failures:
        print("  passed with warnings.")
    else:
        print("  FAILED -- do not train on this data until fixed.")

    return 1 if report.failures else 0


if __name__ == "__main__":
    sys.exit(main())
