"""Register validated Tetouan checkpoints as versioned MLflow PyTorch models.

Historical source runs are read-only. Each registry version is logged from a
new run and carries source-run and content-hash lineage tags.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from src.mlops.mlflow_config import EXPERIMENT_NAME, ensure_local_storage, get_config, get_or_create_experiment
from src.training.evaluate_tetouan import dataset_contract, load_best_model

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ARTIFACT_DIR = REPO_ROOT / "runtime-artifacts" / "data" / "processed" / "tetouan"
DEFAULT_CHECKPOINTS = REPO_ROOT / ".mlflow" / "artifacts"
MODEL_NAMES = {"CDLinear": "Tetouan-CDLinear", "LinearGridTransformer": "Tetouan-LinearGridTransformer"}


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _dataset_fingerprints(artifact_dir: Path) -> dict[str, str]:
    return {name: sha256_file(artifact_dir / name) for name in ("metadata.json", "series.npz")}


def _verify_checkpoint_dataset(payload: dict[str, Any], fingerprints: dict[str, str], metadata: dict[str, Any]) -> None:
    run = payload.get("run_metadata", {})
    recorded = run.get("processed_artifacts", {})
    for filename, digest in fingerprints.items():
        if recorded.get(filename, {}).get("sha256") != digest:
            raise ValueError(f"Checkpoint dataset fingerprint mismatch for {filename}.")
    if run.get("dataset", {}).get("source_sha256") != metadata.get("source", {}).get("sha256"):
        raise ValueError("Checkpoint source dataset SHA-256 does not match processed metadata.")


def register_checkpoint(
    checkpoint_path: str | Path,
    source_run_id: str,
    *,
    artifact_dir: str | Path = DEFAULT_ARTIFACT_DIR,
    tracking_uri: str | None = None,
) -> dict[str, Any]:
    """Register one checkpoint, reusing an exact existing version on repeats."""
    import mlflow
    from mlflow.models import ModelSignature
    from mlflow.types.schema import Schema, TensorSpec
    from mlflow.tracking import MlflowClient

    checkpoint = Path(checkpoint_path).resolve()
    processed = Path(artifact_dir).resolve()
    metadata = json.loads((processed / "metadata.json").read_text(encoding="utf-8"))
    fingerprints = _dataset_fingerprints(processed)
    expected = dataset_contract(metadata)
    device = torch.device("cpu")
    try:
        payload = torch.load(checkpoint, map_location=device, weights_only=True)
    except TypeError:
        payload = torch.load(checkpoint, map_location=device)
    model_name = payload.get("model_name")
    if model_name not in MODEL_NAMES:
        raise ValueError("Unsupported checkpoint model name.")
    model, validated = load_best_model(checkpoint, model_name, expected, device)
    _verify_checkpoint_dataset(validated, fingerprints, metadata)
    digest = sha256_file(checkpoint)
    config = validated["model_config"]
    checkpoint_meta = validated["checkpoint_metadata"]
    run_meta = validated["run_metadata"]
    seed = int(run_meta["training"]["seed"])

    config_object = get_config()
    ensure_local_storage(config_object)
    client = MlflowClient(tracking_uri=tracking_uri or config_object.tracking_uri)
    source_run = client.get_run(source_run_id)
    experiment = client.get_experiment(source_run.info.experiment_id)
    if experiment.name != EXPERIMENT_NAME:
        raise ValueError("Source run does not belong to the Tetouan Energy Forecasting experiment.")
    if source_run.data.tags.get("model_name") not in (None, model_name):
        raise ValueError("Source run model lineage does not match the checkpoint.")

    registered_name = MODEL_NAMES[model_name]
    try:
        client.get_registered_model(registered_name)
    except Exception:
        try:
            client.create_registered_model(registered_name, tags={"dataset": "Tetouan City Power Consumption"})
        except Exception:
            client.get_registered_model(registered_name)

    for version in client.search_model_versions(f"name='{registered_name}'"):
        if version.tags.get("checkpoint_sha256") == digest and version.tags.get("source_run_id") == source_run_id:
            if version.status != "READY":
                raise RuntimeError(f"Matching registry version {registered_name}/{version.version} is not READY.")
            return {"registered_model": registered_name, "version": version.version, "run_id": version.run_id, "reused": True, "checkpoint_sha256": digest}

    mlflow.set_tracking_uri(tracking_uri or config_object.tracking_uri)
    experiment_id = get_or_create_experiment(client, EXPERIMENT_NAME, config_object)
    mlflow.set_experiment(experiment_id=experiment_id)
    x_example = np.zeros((1, 144, 8), dtype=np.float32)
    model.eval()
    with torch.no_grad():
        y_example = model(torch.from_numpy(x_example)).cpu().numpy()
    signature = ModelSignature(
        inputs=Schema([TensorSpec(np.dtype(np.float32), (-1, 144, 8), "input")]),
        outputs=Schema([TensorSpec(np.dtype(np.float32), (-1, 6, 3), "output")]),
    )
    tags = {
        "registration_kind": "historical_checkpoint_version",
        "source_run_id": source_run_id,
        "source_experiment_id": source_run.data.tags.get("source_experiment_id", "tetouan_t4_20261101_v1"),
        "source_checkpoint": checkpoint.name,
        "checkpoint_sha256": digest,
        "dataset_metadata_sha256": fingerprints["metadata.json"],
        "dataset_series_sha256": fingerprints["series.npz"],
        "model_name": model_name,
        "seed": str(seed),
        "best_validation_epoch": str(checkpoint_meta["best_epoch"]),
        "best_validation_loss": str(checkpoint_meta["best_validation_loss"]),
        "source_code_commit": str(run_meta.get("git_commit") or "unknown"),
        "feature_order": json.dumps(metadata["features"]["names"]),
        "target_order": json.dumps(metadata["targets"]["names"]),
        "normalization": "model signature consumes training-normalized float32 inputs and returns normalized targets; clients must use saved train statistics for raw-scale conversion",
    }
    with mlflow.start_run(run_name=f"register-{model_name}-seed-{seed}", tags=tags) as run:
        mlflow.log_params({"source_run_id": source_run_id, "seed": seed, "model_name": model_name, "lookback": 144, "horizon": 6, "input_channels": 8, "output_channels": 3, **{f"model.{key}": value for key, value in config.items()}})
        mlflow.log_metric("validation.best_loss", float(checkpoint_meta["best_validation_loss"]), step=int(checkpoint_meta["best_epoch"]))
        mlflow.log_dict({"dataset": {"name": "Power Consumption of Tetouan City", "source": metadata["source"], "row_count": metadata["row_count"], "split": metadata["split"], "timestamps": metadata["timestamps"], "units_status": metadata.get("units_status")}, "features": metadata["features"]["names"], "targets": metadata["targets"]["names"], "normalization": metadata["normalization"], "model_config": config, "tensor_signature": {"input": [None, 144, 8], "output": [None, 6, 3], "input_scale": "training-normalized", "output_scale": "training-normalized"}, "dataset_sha256": fingerprints}, "model_metadata.json")
        mlflow.pytorch.log_model(model, name="model", signature=signature, serialization_format="pickle", code_paths=[str(REPO_ROOT / "src"), str(REPO_ROOT / "test_attention.py")])
        model_uri = f"runs:/{run.info.run_id}/model"
    version = mlflow.register_model(model_uri, registered_name, await_registration_for=0)
    client.set_model_version_tag(registered_name, version.version, "source_run_id", source_run_id)
    for key, value in tags.items():
        client.set_model_version_tag(registered_name, version.version, key, value)
    return {"registered_model": registered_name, "version": version.version, "run_id": run.info.run_id, "reused": False, "checkpoint_sha256": digest}


def historical_checkpoint_inventory() -> list[tuple[str, Path]]:
    """Return the 2 original and 6 extended checkpoints, excluding smoke runs."""
    entries: list[tuple[str, Path]] = []
    base = DEFAULT_CHECKPOINTS
    for run_id in ("616cc34d387248c785a773976bd6df5e",):
        root = base / run_id / "artifacts" / "checkpoints"
        for name in ("tetouan_t4_20261101_v1_lineargridtransformer.pt", "tetouan_t4_20261101_v1_cdlinear.pt"):
            entries.append((run_id, root / name))
    for run_id in ("4d69c63080be4830a4a33f6d7ab8abea", "c24ee0b136a44a4d8766d1dbab555a3c", "fdb8c618092c4f2db0c6cfceb1e903a1", "b49adc47cbaf42e39b7c8d6ed16f0655", "4eb95e37b19a492884d3951cf43c807a", "f31875436b514b6fa701c5f54af299d2"):
        root = base / run_id / "artifacts" / "checkpoints"
        files = list(root.glob("*.pt"))
        entries.extend((run_id, file) for file in files)
    return entries


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-dir", default=str(DEFAULT_ARTIFACT_DIR))
    parser.add_argument("--checkpoint", help="Register one checkpoint")
    parser.add_argument("--source-run-id")
    parser.add_argument("--all-historical", action="store_true", help="Register the 2 original plus 6 extended logged checkpoints")
    args = parser.parse_args(argv)
    if args.all_historical:
        inventory = historical_checkpoint_inventory()
        missing = [str(path) for _, path in inventory if not path.is_file()]
        if missing:
            parser.error("Missing verified historical checkpoints: " + ", ".join(missing))
    elif args.checkpoint and args.source_run_id:
        inventory = [(args.source_run_id, Path(args.checkpoint))]
    else:
        parser.error("Choose --all-historical or provide both --checkpoint and --source-run-id")
    for run_id, checkpoint in inventory:
        result = register_checkpoint(checkpoint, run_id, artifact_dir=args.artifact_dir)
        print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
