from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
import re
from typing import Any

from dr_grading.paths import ARTIFACTS_DIR


@dataclass(frozen=True)
class ArtifactPaths:
    run_dir: Path
    checkpoint: Path
    final_model: Path
    report: Path


def resolve_artifact_paths(
    config: dict[str, Any],
    started_at: str | None = None,
    artifacts_dir: Path = ARTIFACTS_DIR,
) -> ArtifactPaths:
    experiment = config.get("experiment", {})
    run_name = str(experiment.get("run_name") or experiment.get("name") or "run")
    stamp = started_at or datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    run_dir = artifacts_dir / f"{_slugify(run_name)}-{_safe_stamp(stamp)}"

    return ArtifactPaths(
        run_dir=run_dir,
        checkpoint=run_dir / "best_model.keras",
        final_model=run_dir / "final_model.keras",
        report=run_dir / "test_report.json",
    )


def log_run_artifact_location(mlflow: Any, run_dir: Path) -> None:
    mlflow.set_tag("artifact_run_dir", str(Path(run_dir).resolve()))


def _slugify(value: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", value.strip()).strip("-").lower()
    return slug or "run"


def _safe_stamp(value: str) -> str:
    stamp = re.sub(r"[^a-zA-Z0-9]+", "", value.strip())
    return stamp or datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
