"""Central configuration for the local MLflow tracking store."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT_NAME = "Tetouan Energy Forecasting"
RUN_NAME = "tetouan_t4_20261101_v1"

# Separate historical import identity and artifact locations for the extended run.
EXTENDED_SOURCE_EXPERIMENT_ID = "extended_training_v2"
EXTENDED_PARENT_RUN_NAME = "tetouan_extended_v2_historical_import"
EXTENDED_SEEDS = (42, 123, 2026)
EXTENDED_EXPERIMENT_DIR = REPO_ROOT / "results" / "tetouan" / EXTENDED_SOURCE_EXPERIMENT_ID
EXTENDED_CHECKPOINT_DIR = REPO_ROOT / "checkpoints" / "tetouan" / EXTENDED_SOURCE_EXPERIMENT_ID
EXTENDED_EVALUATION_DIR = EXTENDED_EXPERIMENT_DIR / "evaluation_test_v1"
PROCESSED_DATA_DIR = REPO_ROOT / "data" / "processed" / "tetouan"


@dataclass(frozen=True)
class MlflowConfig:
    tracking_uri: str
    artifact_root: Path


def get_config() -> MlflowConfig:
    """Resolve local defaults, with environment overrides for portable use."""
    default_db = (REPO_ROOT / ".mlflow" / "tracking.db").as_posix()
    default_artifacts = REPO_ROOT / ".mlflow" / "artifacts"
    return MlflowConfig(
        tracking_uri=os.environ.get("MLFLOW_TRACKING_URI", f"sqlite:///{default_db}"),
        artifact_root=Path(os.environ.get("MLFLOW_ARTIFACT_ROOT", str(default_artifacts))).expanduser().resolve(),
    )


def ensure_local_storage(config: MlflowConfig) -> None:
    """Create only the ignored local storage directories needed by MLflow."""
    config.artifact_root.mkdir(parents=True, exist_ok=True)
    if config.tracking_uri.startswith("sqlite:///"):
        db_path = Path(config.tracking_uri.removeprefix("sqlite:///"))
        if not db_path.is_absolute():
            db_path = REPO_ROOT / db_path
        db_path.parent.mkdir(parents=True, exist_ok=True)


def get_or_create_experiment(client, name: str, config: MlflowConfig) -> str:
    """Find an experiment or create it with this config's explicit artifact root."""
    experiment = client.get_experiment_by_name(name)
    if experiment is not None:
        return experiment.experiment_id
    try:
        return client.create_experiment(name, artifact_location=config.artifact_root.as_uri())
    except Exception:
        # Handle a concurrent creator without hiding unrelated backend errors.
        experiment = client.get_experiment_by_name(name)
        if experiment is None:
            raise
        return experiment.experiment_id
