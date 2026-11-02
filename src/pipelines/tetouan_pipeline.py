"""Explicit, isolated multi-seed Tetouan training/evaluation pipeline.

Dry-run is the default. Execute only with ``--execute``; every invocation
requires a unique run name and writes into a new output subtree. Test metrics
are reported after training and are never used for configuration or selection.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
import traceback
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _portable(value: Any) -> Any:
    """Strip machine-specific absolute paths before sending metadata to MLflow."""
    if isinstance(value, dict):
        return {key: _portable(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_portable(item) for item in value]
    if isinstance(value, tuple):
        return [_portable(item) for item in value]
    if isinstance(value, str) and Path(value).is_absolute():
        return Path(value).name
    return value


def _log_sanitized_checkpoint(mlflow_module: Any, path: Path) -> dict[str, Any]:
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        payload = torch.load(path, map_location="cpu")
    with tempfile.TemporaryDirectory(prefix="tetouan-checkpoint-log-") as temp:
        safe_path = Path(temp) / path.name
        torch.save(_portable(payload), safe_path)
        mlflow_module.log_artifact(str(safe_path), artifact_path="checkpoints")
    return payload


def validate_artifacts(artifact_dir: Path, source_csv: Path | None = None) -> dict[str, Any]:
    metadata_path, arrays_path = artifact_dir / "metadata.json", artifact_dir / "series.npz"
    if not metadata_path.is_file() or not arrays_path.is_file():
        raise FileNotFoundError(f"Processed Tetouan artifacts missing under {artifact_dir}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("row_count") != 52416 or metadata.get("features", {}).get("count") != 8 or metadata.get("targets", {}).get("count") != 3:
        raise ValueError("Dataset metadata does not match the approved Tetouan contract.")
    if metadata.get("windows", {}).get("lookback") != 144 or metadata.get("windows", {}).get("horizon") != 6:
        raise ValueError("Dataset window contract must remain 144-to-6.")
    with np.load(arrays_path, allow_pickle=False) as arrays:
        if arrays["features"].shape != (52416, 8) or arrays["targets"].shape != (52416, 3):
            raise ValueError("Processed array dimensions do not match metadata.")
        if not np.isfinite(arrays["features"]).all() or not np.isfinite(arrays["targets"]).all():
            raise ValueError("Processed series contains non-finite values.")
    result = {"metadata": sha256(metadata_path), "series": sha256(arrays_path), "row_count": 52416}
    if source_csv is not None:
        if not source_csv.is_file():
            raise FileNotFoundError(f"Raw source CSV missing: {source_csv}")
        actual_hash = sha256(source_csv)
        if actual_hash != metadata.get("source", {}).get("sha256"):
            raise ValueError("Raw CSV fingerprint does not match processed dataset provenance.")
        result["source_csv"] = actual_hash
    return result


def plan(run_name: str, seeds: list[int], artifact_dir: Path, root: Path, source_csv: Path | None = None) -> dict[str, Any]:
    fingerprints = validate_artifacts(artifact_dir, source_csv)
    if not run_name or any(ch not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-" for ch in run_name):
        raise ValueError("run name must contain only letters, digits, dot, underscore or hyphen")
    if not seeds or len(set(seeds)) != len(seeds) or any(seed < 0 for seed in seeds):
        raise ValueError("seeds must be distinct non-negative integers")
    if (root / run_name).exists():
        raise FileExistsError(f"Refusing to overwrite pipeline output: {root / run_name}")
    return {"run_name": run_name, "seeds": seeds, "artifact_dir": str(artifact_dir), "artifact_fingerprints": fingerprints,
            "outputs": {"checkpoints": str(root / run_name / "checkpoints"), "results": str(root / run_name / "results"), "processed": str(root / run_name / "processed")},
            "test_policy": "held-out evaluation after all training; reporting only, never selection", "device": "CUDA" if torch.cuda.is_available() else "CPU/MPS auto-detect"}


def run_pipeline(*, run_name: str, seeds: list[int], artifact_dir: Path, output_root: Path, source_csv: Path | None = None, epochs: int = 50, batch_size: int = 32, learning_rate: float = 1e-3, weight_decay: float = 1e-2, device: str | None = None, dry_run: bool = True, register: bool = True) -> dict[str, Any]:
    if epochs <= 0 or batch_size <= 0 or learning_rate <= 0 or weight_decay < 0:
        raise ValueError("epochs/batch_size/learning_rate must be positive and weight_decay non-negative")
    plan_data = plan(run_name, seeds, artifact_dir, output_root, source_csv)
    if dry_run:
        return {"status": "dry_run", "plan": plan_data}
    import mlflow
    from mlflow.tracking import MlflowClient
    from src.mlops.mlflow_config import EXPERIMENT_NAME, ensure_local_storage, get_config, get_or_create_experiment
    from src.training.evaluate_tetouan import evaluate_checkpoints
    from src.training.train_tetouan import TrainConfig, run_training
    from src.mlops.model_registry import register_checkpoint

    output = output_root / run_name
    output.mkdir(parents=True, exist_ok=False)
    training_artifact_dir = output / "processed"
    tracking_config = get_config()
    ensure_local_storage(tracking_config)
    mlflow.set_tracking_uri(tracking_config.tracking_uri)
    client = MlflowClient(tracking_uri=tracking_config.tracking_uri)
    experiment_id = get_or_create_experiment(client, EXPERIMENT_NAME, tracking_config)
    parent_id: str | None = None
    child_ids: dict[str, dict[str, str]] = {}
    manifest: dict[str, Any] = {"schema_version": 1, "run_name": run_name, "started_utc": datetime.now(timezone.utc).isoformat(), "status": "running", "plan": plan_data, "seeds": {}, "python": platform.python_version(), "torch": str(torch.__version__)}
    try:
        if source_csv is None:
            raise ValueError("--source-csv is required for an execution run so preprocessing is reproducible.")
        from src.data.tetouan_preprocess import preprocess_tetouan
        processed_metadata = preprocess_tetouan(source_csv, training_artifact_dir)
        if processed_metadata["source"]["sha256"] != plan_data["artifact_fingerprints"].get("source_csv"):
            raise ValueError("Preprocessed source hash changed after initial validation.")
        processed_fingerprints = validate_artifacts(training_artifact_dir, source_csv)
        manifest["preprocessed_artifact_fingerprints"] = processed_fingerprints
        with mlflow.start_run(experiment_id=experiment_id, run_name=run_name, tags={"pipeline": "tetouan", "run_role": "pipeline_parent", "test_metrics_used_for_selection": "false"}) as parent:
            parent_id = parent.info.run_id
            for seed in seeds:
                seed_name = f"{run_name}-s{seed}"
                cp_dir = output / "checkpoints" / f"seed-{seed}"
                rs_dir = output / "results" / f"seed-{seed}"
                cp_dir.mkdir(parents=True, exist_ok=False)
                rs_dir.mkdir(parents=True, exist_ok=False)
                result = run_training(TrainConfig(epochs=epochs, batch_size=batch_size, learning_rate=learning_rate, weight_decay=weight_decay, seed=seed, device=device), checkpoint_dir=cp_dir, results_dir=rs_dir, artifact_dir=training_artifact_dir, run_name=seed_name)
                manifest["seeds"][str(seed)] = {"training": result}
                child_ids[str(seed)] = {}
                for model_name, record in result["models"].items():
                    with mlflow.start_run(experiment_id=experiment_id, run_name=f"{seed_name}-{model_name}", nested=True, tags={"run_role": "model_seed_child", "model_name": model_name, "seed": str(seed), "test_metrics_used_for_selection": "false"}) as child:
                        child_ids[str(seed)][model_name] = child.info.run_id
                        mlflow.log_params({"seed": seed, "epochs": epochs, "batch_size": batch_size, "learning_rate": learning_rate, "weight_decay": weight_decay, "model_name": model_name, "dataset_metadata_sha256": processed_fingerprints["metadata"], "dataset_series_sha256": processed_fingerprints["series"]})
                        for row in record["history"]:
                            mlflow.log_metrics({"training.loss": float(row["train_loss"]), "validation.loss": float(row["validation_loss"])}, step=int(row["epoch"]))
                        mlflow.log_metric("validation.best_loss", float(record["best_validation_loss"]), step=int(record["best_epoch"]))
                        logged_payload = _log_sanitized_checkpoint(mlflow, Path(record["checkpoint"]))
                        mlflow.log_params({f"model.{key}": value for key, value in logged_payload.get("model_config", {}).items()})
                        for suffix in ("history", "manifest"):
                            source_json = rs_dir / f"{seed_name}_{suffix}.json"
                            mlflow.log_dict(_portable(json.loads(source_json.read_text(encoding="utf-8"))), f"run_metadata/{source_json.name}")
                manifest["seeds"][str(seed)]["child_run_ids"] = child_ids[str(seed)]
            # All configurations and best-validation checkpoints are now fixed.
            # Test reports are logged for reporting only; they do not select models.
            for seed in seeds:
                seed_name = f"{run_name}-s{seed}"
                cp_dir = output / "checkpoints" / f"seed-{seed}"
                rs_dir = output / "results" / f"seed-{seed}"
                evaluation = evaluate_checkpoints(cp_dir / f"{seed_name}_lineargridtransformer.pt", cp_dir / f"{seed_name}_cdlinear.pt", artifact_dir=training_artifact_dir, results_dir=rs_dir, output_name=f"{seed_name}_test.json", device=device)
                manifest["seeds"][str(seed)]["test_report"] = str(rs_dir / f"{seed_name}_test.json")
                for model_name in ("LinearGridTransformer", "CDLinear"):
                    run_id = child_ids[str(seed)][model_name]
                    for metric, value in evaluation["metrics"]["models"][model_name]["macro_average_across_zones"].items():
                        client.log_metric(run_id, f"test.macro.{metric}", float(value))
                    for metric, value in evaluation["metrics"]["models"][model_name]["per_zone"].items():
                        for metric_name, metric_value in value.items():
                            client.log_metric(run_id, f"test.zone.{metric}.{metric_name}", float(metric_value))
                    pooled = evaluation["metrics"]["models"][model_name].get("pooled_across_all_samples_horizons_and_zones", {})
                    for metric, value in pooled.items():
                        client.log_metric(run_id, f"test.pooled.{metric}", float(value))
                    report_data = json.loads((rs_dir / f"{seed_name}_test.json").read_text(encoding="utf-8"))
                    client.log_dict(run_id, _portable(report_data), f"heldout_evaluation/{seed_name}_test.json")
                for baseline_name, baseline_metrics in evaluation["metrics"]["baselines"].items():
                    for metric_name, metric_value in baseline_metrics.get("macro_average_across_zones", {}).items():
                        client.log_metric(parent_id, f"test.baseline.{baseline_name}.macro.{metric_name}", float(metric_value), step=seed)
                    for metric_name, metric_value in baseline_metrics.get("pooled_across_all_samples_horizons_and_zones", {}).items():
                        client.log_metric(parent_id, f"test.baseline.{baseline_name}.pooled.{metric_name}", float(metric_value), step=seed)
            registered = []
            if register:
                for seed in seeds:
                    seed_name = f"{run_name}-s{seed}"
                    cp_dir = output / "checkpoints" / f"seed-{seed}"
                    for model_name, suffix in (("LinearGridTransformer", "lineargridtransformer"), ("CDLinear", "cdlinear")):
                        registered.append(register_checkpoint(cp_dir / f"{seed_name}_{suffix}.pt", child_ids[str(seed)][model_name], artifact_dir=training_artifact_dir))
            manifest["registrations"] = registered
            manifest["status"] = "completed"
            manifest["completed_utc"] = datetime.now(timezone.utc).isoformat()
        manifest["parent_run_id"] = parent_id
        (output / "pipeline_manifest.json").write_text(json.dumps(manifest, indent=2, allow_nan=False), encoding="utf-8")
        if parent_id:
            client.log_dict(parent_id, _portable(manifest), "pipeline_manifest.json")
        return manifest
    except Exception as exc:
        manifest.update({"status": "failed", "failed_utc": datetime.now(timezone.utc).isoformat(), "error_type": type(exc).__name__, "error": str(exc), "traceback": traceback.format_exc(), "parent_run_id": parent_id})
        (output / "pipeline_manifest.json").write_text(json.dumps(manifest, indent=2, allow_nan=False), encoding="utf-8")
        if parent_id:
            try:
                client.log_dict(parent_id, _portable(manifest), "pipeline_manifest.json")
            finally:
                client.set_terminated(parent_id, status="FAILED")
        raise


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 123, 2026])
    parser.add_argument("--artifact-dir", type=Path, default=REPO_ROOT / "runtime-artifacts/data/processed/tetouan")
    parser.add_argument("--source-csv", type=Path, default=REPO_ROOT / "data/raw/tetouan/Tetuan City power consumption.csv")
    parser.add_argument("--output-root", type=Path, default=REPO_ROOT / "results/pipelines/tetouan")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--device")
    parser.add_argument("--execute", action="store_true", help="Actually train/evaluate; default validates and prints a dry-run plan")
    parser.add_argument("--no-register", action="store_true")
    args = parser.parse_args(argv)
    result = run_pipeline(run_name=args.run_name, seeds=args.seeds, artifact_dir=args.artifact_dir, output_root=args.output_root, source_csv=args.source_csv, epochs=args.epochs, batch_size=args.batch_size, learning_rate=args.learning_rate, weight_decay=args.weight_decay, device=args.device, dry_run=not args.execute, register=not args.no_register)
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
