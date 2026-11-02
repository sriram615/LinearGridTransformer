"""Centralized, environment-configurable API settings."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RUN_NAME = "tetouan_t4_20261101_v1"
LATEST_EXPERIMENT_ID = "extended_training_v2"


@dataclass(frozen=True)
class ApiConfig:
    run_name: str = DEFAULT_RUN_NAME
    artifact_dir: Path = REPO_ROOT / "data" / "processed" / "tetouan"
    experiment_dir: Path = REPO_ROOT / "results" / "tetouan" / DEFAULT_RUN_NAME
    checkpoint_dir: Path = REPO_ROOT / "checkpoints" / "tetouan"
    raw_dir: Path = REPO_ROOT / "data" / "raw" / "tetouan"
    latest_experiment_id: str = LATEST_EXPERIMENT_ID
    latest_experiment_dir: Path = REPO_ROOT / "results" / "tetouan" / LATEST_EXPERIMENT_ID
    latest_checkpoint_dir: Path = REPO_ROOT / "checkpoints" / "tetouan" / LATEST_EXPERIMENT_ID
    latest_evaluation_dir: Path = REPO_ROOT / "results" / "tetouan" / LATEST_EXPERIMENT_ID / "evaluation_test_v1"
    latest_inference_run_name: str | None = None
    device: str | None = None
    cors_origins: tuple[str, ...] = ("http://localhost:8443", "http://127.0.0.1:8443")
    title: str = "LinearGridTransformer Multi-Zone Grid API"
    version: str = "0.2.0"
    active_dataset: str = "tetouan"
    grid_experiment_id: str = "grid_colab_v1"
    grid_artifact_dir: Path = REPO_ROOT / "data" / "processed" / "grid"
    grid_experiment_dir: Path = REPO_ROOT / "results" / "grid"
    grid_checkpoint_dir: Path = REPO_ROOT / "checkpoints" / "grid"
    grid_run_name: str = "grid_colab_s42"
    grid_db_url: str = "postgresql://postgres:postgres@localhost:5433/grid_forecasting"

    @classmethod
    def from_env(cls) -> "ApiConfig":
        """Load paths/device overrides without embedding machine-specific paths."""
        defaults = cls()
        return cls(
            run_name=os.getenv("TETOUAN_RUN_NAME", defaults.run_name),
            artifact_dir=Path(os.getenv("TETOUAN_ARTIFACT_DIR", defaults.artifact_dir)),
            experiment_dir=Path(os.getenv("TETOUAN_EXPERIMENT_DIR", defaults.experiment_dir)),
            checkpoint_dir=Path(os.getenv("TETOUAN_CHECKPOINT_DIR", defaults.checkpoint_dir)),
            raw_dir=Path(os.getenv("TETOUAN_RAW_DIR", defaults.raw_dir)),
            latest_experiment_id=os.getenv("TETOUAN_LATEST_EXPERIMENT_ID", defaults.latest_experiment_id),
            latest_experiment_dir=Path(os.getenv("TETOUAN_LATEST_EXPERIMENT_DIR", defaults.latest_experiment_dir)),
            latest_checkpoint_dir=Path(os.getenv("TETOUAN_LATEST_CHECKPOINT_DIR", defaults.latest_checkpoint_dir)),
            latest_evaluation_dir=Path(os.getenv("TETOUAN_LATEST_EVALUATION_DIR", defaults.latest_evaluation_dir)),
            latest_inference_run_name=os.getenv("TETOUAN_LATEST_INFERENCE_RUN_NAME") or None,
            device=os.getenv("DEVICE", os.getenv("TETOUAN_DEVICE")) or None,
            cors_origins=tuple(
                origin.strip()
                for origin in os.getenv(
                    "API_CORS_ORIGINS", ",".join(defaults.cors_origins)
                ).split(",")
                if origin.strip()
            ),
            title=os.getenv("API_TITLE", defaults.title),
            version=os.getenv("API_VERSION", defaults.version),
            active_dataset=os.getenv("ACTIVE_DATASET", defaults.active_dataset),
            grid_experiment_id=os.getenv("GRID_EXPERIMENT_ID", defaults.grid_experiment_id),
            grid_artifact_dir=Path(os.getenv("GRID_ARTIFACT_DIR", defaults.grid_artifact_dir)),
            grid_experiment_dir=Path(os.getenv("GRID_EXPERIMENT_DIR", defaults.grid_experiment_dir)),
            grid_checkpoint_dir=Path(os.getenv("GRID_CHECKPOINT_DIR", defaults.grid_checkpoint_dir)),
            grid_run_name=os.getenv("GRID_RUN_NAME", defaults.grid_run_name),
            grid_db_url=os.getenv("GRID_DB_URL", defaults.grid_db_url),
        )

