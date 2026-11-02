"""Import the saved Tetouan experiment into MLflow as a historical run.

This reads the saved reports, processed metadata and checkpoints. It never
loads a model, iterates a dataset, trains, evaluates or changes source artifacts.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.mlops.mlflow_config import (  # noqa: E402
    EXPERIMENT_NAME,
    REPO_ROOT as CONFIG_REPO_ROOT,
    RUN_NAME,
    ensure_local_storage,
    get_config,
)


def _safe_json(value: Any, key: str = "") -> Any:
    """Return a JSON-safe copy with absolute local paths reduced to basenames."""
    if isinstance(value, dict):
        result = {}
        for child_key, child_value in value.items():
            if re.search(r"(^|_)(path|paths|directory|root)$", child_key, re.I):
                if isinstance(child_value, str) and Path(child_value).is_absolute():
                    result[child_key] = Path(child_value).name
                    continue
                if isinstance(child_value, dict):
                    result[child_key] = _safe_json(child_value, child_key)
                    continue
            result[child_key] = _safe_json(child_value, child_key)
        return result
    if isinstance(value, list):
        return [_safe_json(item, key) for item in value]
    if isinstance(value, str) and Path(value).is_absolute():
        return Path(value).name
    return value


def _flatten_scalars(prefix: str, value: Any):
    if isinstance(value, dict):
        for key, child in value.items():
            yield from _flatten_scalars(f"{prefix}.{key}" if prefix else str(key), child)
    elif isinstance(value, (str, int, float, bool)) or value is None:
        if value is not None:
            yield prefix, str(value) if not isinstance(value, (int, float, bool)) else value


def _metric_name(*parts: str) -> str:
    name = ".".join(re.sub(r"[^A-Za-z0-9_.-]+", "_", str(part)).strip("._") for part in parts)
    if len(name) > 250:
        raise ValueError(f"Generated MLflow metric name is too long: {name}")
    return name


def _log_numeric_tree(mlflow, prefix: str, value: Any, step: int = 0) -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            _log_numeric_tree(mlflow, f"{prefix}.{key}" if prefix else str(key), child, step)
    elif isinstance(value, (int, float)) and not isinstance(value, bool):
        number = float(value)
        if math.isfinite(number):
            mlflow.log_metric(prefix, number, step=step)


def _read(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path.name}")
    return value


def _paths(experiment_dir: Path, processed_dir: Path, checkpoint_dir: Path) -> dict[str, Path]:
    return {
        "manifest": experiment_dir / f"{RUN_NAME}_manifest.json",
        "history": experiment_dir / f"{RUN_NAME}_history.json",
        "evaluation": experiment_dir / f"{RUN_NAME}_test_evaluation.json",
        "horizon": experiment_dir / f"{RUN_NAME}_horizon_evaluation.json",
        "dataset_metadata": processed_dir / "metadata.json",
        "transformer_checkpoint": checkpoint_dir / f"{RUN_NAME}_lineargridtransformer.pt",
        "cdlinear_checkpoint": checkpoint_dir / f"{RUN_NAME}_cdlinear.pt",
    }


def _validate_sources(paths: dict[str, Path]) -> None:
    absent = [str(path) for path in paths.values() if not path.is_file()]
    if absent:
        raise FileNotFoundError("Required saved experiment artifacts are missing: " + ", ".join(absent))


def import_historical_run(
    *,
    tracking_uri: str | None = None,
    artifact_root: Path | None = None,
    experiment_dir: Path | None = None,
    processed_dir: Path | None = None,
    checkpoint_dir: Path | None = None,
    experiment_name: str = EXPERIMENT_NAME,
    run_name: str = RUN_NAME,
):
    """Create the imported run once, then return its ID unchanged on repeats."""
    import mlflow
    from mlflow.tracking import MlflowClient

    defaults = get_config()
    tracking_uri = tracking_uri or defaults.tracking_uri
    artifact_root = (artifact_root or defaults.artifact_root).expanduser().resolve()
    experiment_dir = (experiment_dir or CONFIG_REPO_ROOT / "results/tetouan" / RUN_NAME).resolve()
    processed_dir = (processed_dir or CONFIG_REPO_ROOT / "data/processed/tetouan").resolve()
    checkpoint_dir = (checkpoint_dir or CONFIG_REPO_ROOT / "checkpoints/tetouan").resolve()
    paths = _paths(experiment_dir, processed_dir, checkpoint_dir)
    _validate_sources(paths)

    manifest = _read(paths["manifest"])
    history = _read(paths["history"])
    evaluation = _read(paths["evaluation"])
    horizon = _read(paths["horizon"])
    dataset_metadata = _read(paths["dataset_metadata"])
    original_id = manifest.get("run_metadata", {}).get("run_name")
    if not original_id or original_id != RUN_NAME:
        raise ValueError(f"Manifest run identifier is missing or unexpected: {original_id!r}")

    artifact_root.mkdir(parents=True, exist_ok=True)
    if tracking_uri.startswith("sqlite:///"):
        db_path = Path(tracking_uri.removeprefix("sqlite:///"))
        if not db_path.is_absolute():
            db_path = CONFIG_REPO_ROOT / db_path
        db_path.parent.mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient(tracking_uri=tracking_uri)
    existing = client.get_experiment_by_name(experiment_name)
    if existing is None:
        experiment_id = client.create_experiment(
            experiment_name,
            artifact_location=artifact_root.as_uri(),
        )
    else:
        experiment_id = existing.experiment_id

    filter_string = (
        f"tags.original_experiment_id = '{original_id}' "
        "and tags.imported_historical = 'true'"
    )
    imported = client.search_runs([experiment_id], filter_string=filter_string, max_results=10)
    if imported:
        prior = imported[0]
        return {"run_id": prior.info.run_id, "status": "reused", "experiment_id": experiment_id}

    configs = manifest.get("models")
    if not isinstance(configs, dict) or not configs:
        raise ValueError("Manifest does not contain model configurations")
    training = manifest.get("run_metadata", {}).get("training", {})
    source_metadata = evaluation.get("dataset", {})
    with mlflow.start_run(experiment_id=experiment_id, run_name=run_name) as run:
        mlflow.set_tags({
            "imported_historical": "true",
            "source_experiment_id": original_id,
            "original_experiment_id": original_id,
            "source_provenance": source_metadata.get("source_url", "UCI Tetouan dataset"),
            "import_timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "run_classification": "historical_import; not an MLflow training run",
            "test_set_used_for_training_or_selection": "false",
        })

        params = {
            "source_experiment_id": original_id,
            "dataset.name": source_metadata.get("name"),
            "dataset.source_sha256": source_metadata.get("source_sha256"),
            "dataset.row_count": dataset_metadata.get("row_count"),
            "dataset.features": dataset_metadata.get("features", {}).get("count"),
            "dataset.targets": dataset_metadata.get("targets", {}).get("count"),
            "dataset.lookback": dataset_metadata.get("windows", {}).get("lookback"),
            "dataset.horizon": dataset_metadata.get("windows", {}).get("horizon"),
            "dataset.cadence_minutes": dataset_metadata.get("timestamps", {}).get("cadence_minutes"),
            "dataset.normalization": dataset_metadata.get("normalization", {}).get("method"),
            "dataset.normalization_fit_partition": dataset_metadata.get("normalization", {}).get("fit_partition"),
            "training.optimizer": manifest.get("run_metadata", {}).get("optimizer"),
            "training.loss": manifest.get("run_metadata", {}).get("loss"),
            "training.epochs": training.get("epochs"),
            "training.batch_size": training.get("batch_size"),
            "training.num_workers": training.get("num_workers"),
            "training.learning_rate": training.get("learning_rate"),
            "training.weight_decay": training.get("weight_decay"),
            "training.seed": training.get("seed"),
            "training.gradient_clip_norm": manifest.get("run_metadata", {}).get("gradient_clipping_max_norm"),
            "training.test_partition_used": manifest.get("run_metadata", {}).get("test_partition_used"),
        }
        for model_name, configuration in configs.items():
            for key, value in configuration.get("configuration", {}).items():
                params[f"model.{model_name}.{key}"] = value
            params[f"model.{model_name}.best_validation_epoch"] = configuration.get("best_validation_epoch")
            params[f"model.{model_name}.best_validation_loss"] = configuration.get("best_validation_loss")
        mlflow.log_params({key: str(value) if isinstance(value, (dict, list)) else value
                           for key, value in params.items() if value is not None})

        for model_name, model_history in history.get("models", {}).items():
            for epoch_record in model_history.get("history", []):
                epoch = int(epoch_record["epoch"])
                for field in ("train_loss", "validation_loss"):
                    if field in epoch_record:
                        mlflow.log_metric(_metric_name("training", model_name, field),
                                          float(epoch_record[field]), step=epoch)
            if model_history.get("best_epoch") is not None:
                epoch = int(model_history["best_epoch"])
                loss = float(model_history["best_validation_loss"])
                mlflow.log_metric(_metric_name("validation", model_name, "best_loss"), loss, step=epoch)

        metrics = evaluation.get("metrics", {})
        for model_name, model_metrics in metrics.get("models", {}).items():
            _log_numeric_tree(mlflow, _metric_name("test", model_name), model_metrics)
        for baseline_name, baseline_metrics in metrics.get("baselines", {}).items():
            _log_numeric_tree(mlflow, _metric_name("test", "baseline", baseline_name), baseline_metrics)
        for model_name, model_data in horizon.get("models", {}).items():
            for record in model_data.get("per_horizon", []):
                step = int(record["horizon_step"])
                _log_numeric_tree(
                    mlflow,
                    _metric_name("test", "horizon", model_name),
                    {key: value for key, value in record.items() if key not in ("horizon_step", "horizon_minutes")},
                    step=step,
                )
        for model_name, model_data in horizon.get("models", {}).items():
            consistency = model_data.get("consistency_with_existing_overall_metrics", {}).get("all_match")
            if consistency is not None:
                mlflow.set_tag(f"horizon_metrics_consistent.{model_name}", str(bool(consistency)).lower())

        # Keep a privacy-safe copy of the source reports and manifest in MLflow.
        artifacts = {
            "metadata/manifest.json": manifest,
            "metadata/training_history.json": history,
            "metadata/test_evaluation.json": evaluation,
            "metadata/horizon_evaluation.json": horizon,
            "metadata/dataset_metadata.json": dataset_metadata,
            "metadata/model_configurations.json": {name: item.get("configuration") for name, item in configs.items()},
        }
        for artifact_path, payload in artifacts.items():
            mlflow.log_dict(_safe_json(payload), artifact_path)

        # Both checkpoint files total under 1 MiB; retain convenient immutable
        # copies in the run so the imported historical record is self-contained.
        mlflow.log_artifact(str(paths["transformer_checkpoint"]), artifact_path="checkpoints")
        mlflow.log_artifact(str(paths["cdlinear_checkpoint"]), artifact_path="checkpoints")
        return {"run_id": run.info.run_id, "status": "created", "experiment_id": experiment_id}


def main(argv: list[str] | None = None) -> int:
    config = get_config()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tracking-uri", default=config.tracking_uri)
    parser.add_argument("--artifact-root", type=Path, default=config.artifact_root)
    parser.add_argument("--experiment-dir", type=Path,
                        default=CONFIG_REPO_ROOT / "results/tetouan" / RUN_NAME)
    parser.add_argument("--processed-dir", type=Path, default=CONFIG_REPO_ROOT / "data/processed/tetouan")
    parser.add_argument("--checkpoint-dir", type=Path, default=CONFIG_REPO_ROOT / "checkpoints/tetouan")
    args = parser.parse_args(argv)
    ensure_local_storage(type(config)(args.tracking_uri, args.artifact_root.expanduser().resolve()))
    result = import_historical_run(
        tracking_uri=args.tracking_uri,
        artifact_root=args.artifact_root,
        experiment_dir=args.experiment_dir,
        processed_dir=args.processed_dir,
        checkpoint_dir=args.checkpoint_dir,
    )
    print(f"MLflow experiment ID: {result['experiment_id']}")
    print(f"MLflow run ID: {result['run_id']}")
    print(f"Import status: {result['status']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
