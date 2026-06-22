from __future__ import annotations

from pathlib import Path


PACKAGE_DIR = Path(__file__).resolve().parent
SRC_DIR = PACKAGE_DIR.parent
PROJECT_DIR = SRC_DIR.parent
REPOSITORY_DIR = PROJECT_DIR.parent

CONFIG_DIR = PROJECT_DIR / "configs"
DATA_DIR = PROJECT_DIR / "data"
ARTIFACTS_DIR = PROJECT_DIR / "artifacts"
REPORTS_DIR = PROJECT_DIR / "reports"
MODELS_DIR = PROJECT_DIR / "models"

DEFAULT_PREPROCESSED_DIR = DATA_DIR / "preprocessed"
DEFAULT_TRAINING_CONFIG = CONFIG_DIR / "efficientnet_b0.json"
DEFAULT_TEST_REPORT = ARTIFACTS_DIR / "test_report.json"
DEFAULT_SAVED_MODELS_DIR = MODELS_DIR
DEFAULT_MODEL_PREPROCESSED_ARCHIVE = MODELS_DIR / "preprocessed" / "dr_preprocessed_data.npz"
DEFAULT_MODEL_PREPROCESSED_DOWNLOAD_SOURCE = (
    "https://drive.google.com/file/d/1vVVKvVr0b3AwXn3IEZkWktAEV-HIBkRS/view?usp=sharing"
)
