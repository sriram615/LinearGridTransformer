"""Import saved extended_training_v2 results into MLflow without retraining.

All logged values are copied from manifests, histories and evaluation reports.
The importer does not load model weights, datasets, predictions or recalculate metrics.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
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
    EXTENDED_CHECKPOINT_DIR,
    EXTENDED_EVALUATION_DIR,
    EXTENDED_EXPERIMENT_DIR,
    EXTENDED_PARENT_RUN_NAME,
    EXTENDED_SEEDS,
    EXTENDED_SOURCE_EXPERIMENT_ID,
    PROCESSED_DATA_DIR,
    get_config,
)


SOURCE_ID = EXTENDED_SOURCE_EXPERIMENT_ID
PARENT_RUN_NAME = EXTENDED_PARENT_RUN_NAME
SEEDS = EXTENDED_SEEDS
MODELS = ("LinearGridTransformer", "CDLinear")
SCHEMA_VERSION = "1"


class ImportConflictError(RuntimeError):
    """An existing import does not match the current source or is incomplete."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _safe_json(value: Any) -> Any:
    """Remove machine-specific absolute paths from uploaded JSON copies."""
    if isinstance(value, dict):
        return {key: _safe_json(child) for key, child in value.items()}
    if isinstance(value, list):
        return [_safe_json(child) for child in value]
    if isinstance(value, str) and Path(value).is_absolute():
        return Path(value).name
    return value


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path.name}.")
    return value


def _paths(
    experiment_dir: Path, checkpoint_dir: Path, processed_dir: Path, evaluation_dir: Path
) -> dict[str, Path]:
    paths = {
        "dataset_metadata": processed_dir / "metadata.json",
        "evaluation_report": evaluation_dir / "evaluation_report.json",
        "overall_metrics_csv": evaluation_dir / "overall_metrics_by_seed.csv",
        "per_zone_metrics_csv": evaluation_dir / "per_zone_metrics_by_seed.csv",
        "per_horizon_metrics_csv": evaluation_dir / "per_horizon_metrics_by_seed.csv",
        "three_seed_summary_csv": evaluation_dir / "three_seed_summary.csv",
    }
    for seed in SEEDS:
        stem = f"tetouan_extended_v2_s{seed}"
        paths[f"manifest_seed_{seed}"] = experiment_dir / f"{stem}_manifest.json"
        paths[f"history_seed_{seed}"] = experiment_dir / f"{stem}_history.json"
        for model in MODELS:
            suffix = "lineargridtransformer" if model == "LinearGridTransformer" else "cdlinear"
            paths[f"checkpoint_{model}_seed_{seed}"] = checkpoint_dir / f"{stem}_{suffix}.pt"
    absent = [path.name for path in paths.values() if not path.is_file()]
    if absent:
        raise FileNotFoundError("Missing required extended experiment artifacts: " + ", ".join(absent))
    return paths


def _fingerprint(paths: dict[str, Path]) -> tuple[str, dict[str, dict[str, Any]]]:
    records = {
        key: {"name": path.name, "size_bytes": path.stat().st_size, "sha256": _sha256(path)}
        for key, path in sorted(paths.items())
    }
    canonical = json.dumps(records, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(canonical).hexdigest(), records


def _metric_name(*parts: Any) -> str:
    return ".".join(
        re.sub(r"[^A-Za-z0-9_.-]+", "_", str(part)).strip("._")
        for part in parts
    )[:250]


def _number(value: Any, field: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"Non-finite metric value in {field}.")
    return result


def _log_metric(mlflow, name: str, value: Any, *, step: int = 0) -> None:
    mlflow.log_metric(name, _number(value, name), step=step)


def _child_name(seed: int, model: str) -> str:
    slug = "lineargridtransformer" if model == "LinearGridTransformer" else "cdlinear"
    return f"tetouan_extended_v2_s{seed}_{slug}"


def import_extended_experiment(
    *,
    tracking_uri: str | None = None,
    experiment_name: str = EXPERIMENT_NAME,
    experiment_dir: str | Path | None = None,
    checkpoint_dir: str | Path | None = None,
    processed_dir: str | Path | None = None,
    evaluation_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Import once; reuse an exact prior import and reject changed/partial imports."""
    import mlflow
    from mlflow.tracking import MlflowClient

    config = get_config()
    tracking_uri = tracking_uri or config.tracking_uri
    experiment_dir = Path(experiment_dir or EXTENDED_EXPERIMENT_DIR)
    checkpoint_dir = Path(checkpoint_dir or EXTENDED_CHECKPOINT_DIR)
    processed_dir = Path(processed_dir or PROCESSED_DATA_DIR)
    evaluation_dir = Path(evaluation_dir or EXTENDED_EVALUATION_DIR)
    paths = _paths(experiment_dir, checkpoint_dir, processed_dir, evaluation_dir)
    fingerprint, source_fingerprints = _fingerprint(paths)
    manifests = {seed: _read_json(paths[f"manifest_seed_{seed}"]) for seed in SEEDS}
    histories = {seed: _read_json(paths[f"history_seed_{seed}"]) for seed in SEEDS}
    report = _read_json(paths["evaluation_report"])
    dataset = _read_json(paths["dataset_metadata"])

    for seed in SEEDS:
        manifest = manifests[seed]
        history = histories[seed]
        expected_name = f"tetouan_extended_v2_s{seed}"
        run_meta = manifest.get("run_metadata", {})
        if run_meta.get("run_name") != expected_name or run_meta.get("training", {}).get("seed") != seed:
            raise ValueError(f"Seed {seed} manifest identity/configuration is inconsistent.")
        if run_meta.get("test_partition_used") is not False:
            raise ValueError(f"Seed {seed} manifest does not establish test isolation.")
        if history.get("run_metadata", {}).get("run_name") != expected_name:
            raise ValueError(f"Seed {seed} history belongs to a different run.")
        for model in MODELS:
            if len(history.get("models", {}).get(model, {}).get("history", [])) != 50:
                raise ValueError(f"Seed {seed} {model} history must contain exactly 50 epochs.")
            record = manifest.get("models", {}).get(model, {})
            if not record.get("configuration"):
                raise ValueError(f"Seed {seed} {model} configuration is missing.")

    client = MlflowClient(tracking_uri=tracking_uri)
    mlflow.set_tracking_uri(tracking_uri)
    experiment = client.get_experiment_by_name(experiment_name)
    if experiment is None:
        raise ValueError(f"Existing MLflow experiment {experiment_name!r} was not found.")

    matches = client.search_runs(
        [experiment.experiment_id],
        filter_string=f"tags.source_experiment_id = '{SOURCE_ID}' and tags.import_type = 'historical'",
        max_results=100,
    )
    parents = [
        run for run in matches
        if run.data.tags.get("run_role") == "parent"
        and run.data.tags.get("importer_schema_version") == SCHEMA_VERSION
    ]
    if parents:
        if len(parents) != 1:
            raise ImportConflictError("Multiple matching historical parent runs exist; manual resolution is required.")
        parent = parents[0]
        if parent.data.tags.get("source_fingerprint") != fingerprint:
            raise ImportConflictError(
                "Source artifacts changed since the existing import. No runs were changed; "
                "review the differences and choose an explicit new import/version."
            )
        children = client.search_runs(
            [experiment.experiment_id],
            filter_string=f"tags.mlflow.parentRunId = '{parent.info.run_id}'",
            max_results=100,
        )
        expected = {_child_name(seed, model) for seed in SEEDS for model in MODELS}
        actual = {run.data.tags.get("mlflow.runName") for run in children}
        if actual != expected or len(children) != 6:
            raise ImportConflictError(
                "Matching parent exists but its six child runs are incomplete or duplicated; "
                "manual resolution is required. No existing run was changed."
            )
        return {
            "status": "reused",
            "experiment_id": experiment.experiment_id,
            "parent_run_id": parent.info.run_id,
            "child_run_ids": {
                run.data.tags["mlflow.runName"]: run.info.run_id for run in children
            },
            "source_fingerprint": fingerprint,
        }

    # Summary values and test-selection provenance are copied from the saved report.
    inference_seed = min(
        SEEDS,
        key=lambda seed: float(manifests[seed]["models"]["CDLinear"]["best_validation_loss"]),
    )
    local_mlflow_version = mlflow.__version__
    import torch
    local_torch_version = torch.__version__
    parent_params = {
        "source_experiment_id": SOURCE_ID,
        "source_fingerprint": fingerprint,
        "seed_count": len(SEEDS),
        "model_count": len(MODELS),
        "dataset.row_count": dataset.get("row_count"),
        "dataset.source_sha256": dataset.get("source", {}).get("sha256"),
        "dataset.lookback": dataset.get("windows", {}).get("lookback"),
        "dataset.horizon": dataset.get("windows", {}).get("horizon"),
        "dataset.cadence_minutes": dataset.get("timestamps", {}).get("cadence_minutes"),
        "dataset.feature_count": dataset.get("features", {}).get("count"),
        "dataset.target_count": dataset.get("targets", {}).get("count"),
        "evaluation.test_windows": report.get("test_windows"),
        "local.mlflow_version": local_mlflow_version,
    }
    parent_tags = {
        "source_experiment_id": SOURCE_ID,
        "import_type": "historical",
        "run_role": "parent",
        "historical_import": "true",
        "training_executed_under_mlflow": "false",
        "test_metrics_scope": "held_out_test; used in historical model-family selection",
        "test_selection_bias": "true",
        "selected_model_by_saved_test_report": "CDLinear",
        "inference_checkpoint_seed": str(inference_seed),
        "inference_seed_selection_metric": "CDLinear best validation loss",
        "training_git_commit": "null (unavailable in source manifests)",
        "source_fingerprint": fingerprint,
        "importer_schema_version": SCHEMA_VERSION,
        "local_mlflow_version": local_mlflow_version,
        "local_python_version": platform.python_version(),
        "local_pytorch_version": str(local_torch_version),
        "local_mlflow_environment_differs_from_training": "true",
        "recorded_training_python": str(manifests[SEEDS[0]]["run_metadata"].get("python_version")),
        "recorded_training_torch": str(manifests[SEEDS[0]]["run_metadata"].get("pytorch_version")),
        "recorded_training_cuda": str(manifests[SEEDS[0]]["run_metadata"].get("cuda_version")),
        "recorded_training_gpu": str(manifests[SEEDS[0]]["run_metadata"].get("gpu_name")),
        "methodology_note": "CDLinear family selected using saved three-seed held-out test metrics; test comparison is selection-biased.",
    }

    child_ids: dict[str, str] = {}
    with mlflow.start_run(
        experiment_id=experiment.experiment_id, run_name=PARENT_RUN_NAME
    ) as parent_run:
        parent_id = parent_run.info.run_id
        mlflow.set_tags(parent_tags)
        mlflow.log_params({k: str(v) for k, v in parent_params.items() if v is not None})
        mlflow.log_dict(
            {
                "source_experiment_id": SOURCE_ID,
                "source_fingerprint": fingerprint,
                "source_artifact_fingerprints": source_fingerprints,
                "training_environment": _safe_json(manifests[SEEDS[0]]["run_metadata"]),
                "evaluation_artifact_references": {
                    name: paths[name].name for name in (
                        "evaluation_report", "overall_metrics_csv", "per_zone_metrics_csv",
                        "per_horizon_metrics_csv", "three_seed_summary_csv",
                    )
                },
                "historical_import": True,
                "training_executed_under_mlflow": False,
                "training_git_commit": None,
                "test_selection_bias": True,
                "selected_model_by_saved_test_report": "CDLinear",
                "inference_checkpoint_seed": inference_seed,
                "imported_at_utc": datetime.now(timezone.utc).isoformat(),
            },
            "metadata/import_provenance.json",
        )

        # Parent-only three-seed aggregates and baseline results from the report.
        for row in report.get("three_seed_summary", []):
            model = row["model"]
            for field, value in row.items():
                if field != "model" and isinstance(value, (int, float)):
                    _log_metric(mlflow, _metric_name("test", "three_seed_summary", model, field), value)
        for row in report.get("overall_metrics", []):
            if row.get("model") == "Persistence":
                for metric, field in (("mae", "MAE"), ("rmse", "RMSE"), ("macro_zone_mae", "macro_zone_MAE")):
                    _log_metric(mlflow, _metric_name("test", "baseline", "persistence", metric), row[field])
        for row in report.get("per_zone_metrics", []):
            if row.get("model") == "Persistence":
                zone = re.sub(r"\W+", "_", row["zone"].lower()).strip("_")
                for metric in ("MAE", "RMSE"):
                    _log_metric(mlflow, _metric_name("test", "baseline", "persistence", zone, metric.lower()), row[metric])
        for row in report.get("per_horizon_metrics", []):
            if row.get("model") == "Persistence":
                step = int(row["horizon_step"])
                minutes = int(row["minutes_ahead"])
                for metric in ("MAE", "RMSE"):
                    _log_metric(
                        mlflow, _metric_name("test", "baseline", "persistence", f"horizon_{minutes}m", metric.lower()),
                        row[metric], step=step,
                    )

        # Log source reports/metadata as privacy-safe JSON copies and saved CSV summaries.
        for seed in SEEDS:
            mlflow.log_dict(_safe_json(manifests[seed]), f"metadata/manifests/{seed}.json")
            mlflow.log_dict(_safe_json(histories[seed]), f"metadata/histories/{seed}.json")
        mlflow.log_dict(_safe_json(report), "metadata/evaluation_report.json")
        mlflow.log_dict(_safe_json(dataset), "metadata/dataset_metadata.json")
        mlflow.log_dict(
            {
                model: manifests[SEEDS[0]]["models"][model]["configuration"]
                for model in MODELS
            },
            "metadata/model_configurations.json",
        )
        for key in ("overall_metrics_csv", "per_zone_metrics_csv", "per_horizon_metrics_csv", "three_seed_summary_csv"):
            mlflow.log_artifact(str(paths[key]), artifact_path="evaluation")
        # Every child is a single model/seed run; no metric history is shared.
        for seed in SEEDS:
            manifest = manifests[seed]
            history = histories[seed]
            training = manifest["run_metadata"]["training"]
            env = manifest["run_metadata"]
            for model in MODELS:
                model_record = manifest["models"][model]
                model_config = model_record["configuration"]
                model_history = history["models"][model]
                best_epoch = int(model_record["best_validation_epoch"])
                best_loss = _number(model_record["best_validation_loss"], "best_validation_loss")
                name = _child_name(seed, model)
                with mlflow.start_run(
                    experiment_id=experiment.experiment_id, run_name=name, nested=True
                ) as child_run:
                    child_ids[name] = child_run.info.run_id
                    mlflow.set_tags({
                        "source_experiment_id": SOURCE_ID,
                        "import_type": "historical",
                        "run_role": "model_seed",
                        "historical_import": "true",
                        "training_executed_under_mlflow": "false",
                        "model_name": model,
                        "seed": str(seed),
                        "metric_scope": "training_validation_and_held_out_test",
                        "held_out_test_selection_bias": "true",
                        "training_validation_loss_scale": "normalized MSE",
                        "model_role_in_saved_report": "selected_by_test_report" if model == "CDLinear" else "compared_alternative",
                        "import_parent_run_id": parent_id,
                        "source_fingerprint": fingerprint,
                    })
                    params = {
                        "model.name": model,
                        "seed": seed,
                        "training.epochs": training.get("epochs"),
                        "training.batch_size": training.get("batch_size"),
                        "training.num_workers": training.get("num_workers"),
                        "training.learning_rate": training.get("learning_rate"),
                        "training.weight_decay": training.get("weight_decay"),
                        "training.gradient_clip_norm": training.get("grad_clip_norm"),
                        "training.optimizer": env.get("optimizer"),
                        "training.loss": env.get("loss"),
                        "model.best_validation_epoch": best_epoch,
                        "model.best_validation_loss": best_loss,
                        "dataset.name": env.get("dataset", {}).get("dataset_name"),
                        "dataset.source_sha256": env.get("dataset", {}).get("source_sha256"),
                        "dataset.lookback": env.get("dataset", {}).get("lookback"),
                        "dataset.horizon": env.get("dataset", {}).get("horizon"),
                        "dataset.cadence_minutes": env.get("dataset", {}).get("cadence_minutes"),
                        "dataset.feature_order": json.dumps(env.get("dataset", {}).get("feature_names")),
                        "dataset.target_order": json.dumps(env.get("dataset", {}).get("target_names")),
                        "dataset.normalization": json.dumps(env.get("dataset", {}).get("normalization"), sort_keys=True),
                        "dataset.split_ranges": json.dumps(env.get("dataset", {}).get("used_partitions"), sort_keys=True),
                        "environment.python": env.get("python_version"),
                        "environment.pytorch": env.get("pytorch_version"),
                        "environment.cuda": env.get("cuda_version"),
                        "environment.gpu": env.get("gpu_name"),
                        "training.git_commit": env.get("git_commit"),
                    }
                    params.update({f"model.config.{key}": value for key, value in model_config.items()})
                    mlflow.log_params({k: str(v) for k, v in params.items() if v is not None})
                    for row in model_history["history"]:
                        epoch = int(row["epoch"])
                        if epoch < 1 or epoch > 50:
                            raise ValueError(f"Unexpected epoch {epoch} in {name}.")
                        _log_metric(mlflow, "training.loss", row["train_loss"], step=epoch)
                        _log_metric(mlflow, "validation.loss", row["validation_loss"], step=epoch)
                    _log_metric(mlflow, "validation.best_loss", best_loss, step=best_epoch)

                    for row in report.get("overall_metrics", []):
                        if row.get("model") == model and int(row.get("seed")) == seed:
                            _log_metric(mlflow, "test.mae", row["MAE"])
                            _log_metric(mlflow, "test.rmse", row["RMSE"])
                            _log_metric(mlflow, "test.macro_zone_mae", row["macro_zone_MAE"])
                    for row in report.get("per_zone_metrics", []):
                        if row.get("model") == model and int(row.get("seed")) == seed:
                            zone = re.sub(r"\W+", "_", row["zone"].lower()).strip("_")
                            _log_metric(mlflow, f"test.zone.{zone}.mae", row["MAE"])
                            _log_metric(mlflow, f"test.zone.{zone}.rmse", row["RMSE"])
                    for row in report.get("per_horizon_metrics", []):
                        if row.get("model") == model and int(row.get("seed")) == seed:
                            step = int(row["horizon_step"])
                            minutes = int(row["minutes_ahead"])
                            _log_metric(mlflow, f"test.horizon_{minutes}m.mae", row["MAE"], step=step)
                            _log_metric(mlflow, f"test.horizon_{minutes}m.rmse", row["RMSE"], step=step)

                    mlflow.log_dict(_safe_json(manifest), "metadata/seed_manifest.json")
                    mlflow.log_dict(_safe_json(history), "metadata/training_history.json")
                    checkpoint_key = f"checkpoint_{model}_seed_{seed}"
                    source_checkpoint = paths[checkpoint_key]
                    mlflow.log_artifact(str(source_checkpoint), artifact_path="checkpoints")
                    mlflow.set_tag("checkpoint_source_sha256", _sha256(source_checkpoint))

        if set(child_ids) != {_child_name(seed, model) for seed in SEEDS for model in MODELS}:
            raise RuntimeError("Importer did not create the expected six model/seed child runs.")

    return {
        "status": "created",
        "experiment_id": experiment.experiment_id,
        "parent_run_id": parent_id,
        "child_run_ids": child_ids,
        "source_fingerprint": fingerprint,
    }


def main(argv: list[str] | None = None) -> int:
    defaults = get_config()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tracking-uri", default=defaults.tracking_uri)
    parser.add_argument("--experiment-dir", type=Path, default=EXTENDED_EXPERIMENT_DIR)
    parser.add_argument("--checkpoint-dir", type=Path, default=EXTENDED_CHECKPOINT_DIR)
    parser.add_argument("--processed-dir", type=Path, default=PROCESSED_DATA_DIR)
    parser.add_argument("--evaluation-dir", type=Path, default=EXTENDED_EVALUATION_DIR)
    args = parser.parse_args(argv)
    result = import_extended_experiment(
        tracking_uri=args.tracking_uri,
        experiment_dir=args.experiment_dir,
        checkpoint_dir=args.checkpoint_dir,
        processed_dir=args.processed_dir,
        evaluation_dir=args.evaluation_dir,
    )
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
