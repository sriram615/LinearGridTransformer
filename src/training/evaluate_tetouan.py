"""Final held-out evaluation for best-validation Tetouan experiment checkpoints.

Example::

    .venv311/bin/python3 -m src.training.evaluate_tetouan \\
        --lgt-checkpoint checkpoints/tetouan/<run>_lineargridtransformer.pt \\
        --cdlinear-checkpoint checkpoints/tetouan/<run>_cdlinear.pt

Only non-smoke experiment checkpoints selected by validation loss are accepted.
Reports are isolated under results/tetouan and are never overwritten.
"""
from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from src.data.tetouan_dataset import TetouanSequenceDataset
from src.data.tetouan_preprocess import DEFAULT_OUTPUT_DIR, TARGET_NAMES
from src.models.cdlinear import CDLinearBaseline
from src.training.train import LinearGridForecaster

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RESULTS_DIR = REPO_ROOT / "results" / "tetouan"
MODEL_NAMES = ("LinearGridTransformer", "CDLinear")


def dataset_contract(metadata: dict[str, Any]) -> dict[str, Any]:
    """Match the dataset contract stored by train_tetouan.py."""
    source = metadata["source"]
    timestamps = metadata["timestamps"]
    partitions = metadata["split"]["partitions"]
    normalization = metadata["normalization"]
    return {
        "dataset_name": "Power Consumption of Tetouan City",
        "source_url": source.get("dataset_url"),
        "source_file": source.get("file_name"),
        "source_sha256": source.get("sha256"),
        "doi": source.get("doi"),
        "dataset_row_count": metadata["row_count"],
        "feature_names": tuple(metadata["features"]["names"]),
        "target_names": tuple(metadata["targets"]["names"]),
        "lookback": metadata["windows"]["lookback"],
        "horizon": metadata["windows"]["horizon"],
        "cadence_minutes": timestamps.get("cadence_minutes"),
        "timezone_status": timestamps.get("timezone_status"),
        "units_status": metadata.get("units_status"),
        "normalization": {
            "method": normalization.get("method"),
            "fit_partition": normalization.get("fit_partition"),
            "fit_row_range_half_open": normalization.get("fit_row_range_half_open"),
            "std_ddof": normalization.get("std_ddof"),
        },
        "used_partitions": {"train": partitions["train"], "validation": partitions["validation"]},
        "test_partition_used": False,
    }


def target_normalization_stats(metadata: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    """Read and validate training-only target stats; never refit them."""
    normalization = metadata["normalization"]
    if normalization.get("fit_partition") != "train":
        raise ValueError("Target normalization must be fitted on the training partition.")
    train_partition = metadata["split"]["partitions"]["train"]
    expected_fit_rows = [train_partition["start_index"], train_partition["end_index_exclusive"]]
    if normalization.get("fit_row_range_half_open") != expected_fit_rows:
        raise ValueError("Normalization fit rows do not match the recorded training partition.")
    if tuple(normalization.get("target_names", ())) != tuple(TARGET_NAMES):
        raise ValueError("Target normalization channel ordering does not match the target contract.")
    target_mean = np.asarray(normalization["target_mean"], dtype=np.float64)
    target_std = np.asarray(normalization["target_std"], dtype=np.float64)
    feature_mean = np.asarray(normalization["feature_mean"], dtype=np.float64)
    feature_std = np.asarray(normalization["feature_std"], dtype=np.float64)
    if target_mean.shape != (3,) or target_std.shape != (3,):
        raise ValueError("Expected three target normalization means and standard deviations.")
    if feature_mean.shape != (8,) or feature_std.shape != (8,):
        raise ValueError("Expected eight feature normalization means and standard deviations.")
    if (not np.isfinite(target_mean).all() or not np.isfinite(target_std).all()
            or np.any(target_std <= 0)):
        raise ValueError("Target normalization statistics must be finite with positive standard deviations.")
    if not np.allclose(feature_mean[:3], target_mean, rtol=1e-12, atol=1e-12):
        raise ValueError("Zone input means do not match target training means for persistence.")
    if not np.allclose(feature_std[:3], target_std, rtol=1e-12, atol=1e-12):
        raise ValueError("Zone input standard deviations do not match target training statistics.")
    return target_mean, target_std


def _validate_model_config(model_name: str, config: dict[str, Any]) -> None:
    if model_name == "LinearGridTransformer":
        required = {"in_channels": 8, "out_channels": 3, "lookback": 144, "horizon": 6}
        for key, expected in required.items():
            if config.get(key) != expected:
                raise ValueError(f"Transformer checkpoint requires {key}={expected}; got {config.get(key)!r}.")
        for key in ("d_model", "num_heads", "ffn_dim"):
            if not isinstance(config.get(key), int) or config[key] <= 0:
                raise ValueError(f"Transformer checkpoint has invalid {key}.")
        if config["d_model"] % config["num_heads"]:
            raise ValueError("Transformer d_model must be divisible by num_heads.")
        if not isinstance(config.get("dropout"), (int, float)) or not 0 <= config["dropout"] < 1:
            raise ValueError("Transformer checkpoint has invalid dropout.")
    elif model_name == "CDLinear":
        required = {"in_channels": 8, "out_channels": 3, "seq_len": 144, "pred_len": 6}
        for key, expected in required.items():
            if config.get(key) != expected:
                raise ValueError(f"CDLinear checkpoint requires {key}={expected}; got {config.get(key)!r}.")
    else:
        raise ValueError(f"Unsupported model name {model_name!r}.")


def validate_checkpoint_payload(
    payload: dict[str, Any], model_name: str, expected_dataset: dict[str, Any]
) -> dict[str, Any]:
    """Validate best-validation provenance and dimensions before test access."""
    if payload.get("model_name") != model_name:
        raise ValueError(f"Expected {model_name}, got checkpoint {payload.get('model_name')!r}.")
    run = payload.get("run_metadata")
    if not isinstance(run, dict) or run.get("run_type") != "experiment":
        raise ValueError("Only full experiment checkpoints may be used for held-out evaluation.")
    if run.get("test_partition_used") is not False:
        raise ValueError("Checkpoint does not confirm test-set isolation during training.")
    if run.get("dataset") != expected_dataset:
        raise ValueError("Checkpoint dataset provenance, split, or normalization contract does not match.")
    selection = payload.get("checkpoint_metadata")
    if not isinstance(selection, dict):
        raise ValueError("Checkpoint has no best-validation metadata; it is not an approved best checkpoint.")
    best_epoch, best_loss = selection.get("best_epoch"), selection.get("best_validation_loss")
    if (selection.get("selection_metric") != "validation_loss"
            or not isinstance(best_epoch, int) or best_epoch <= 0
            or not isinstance(best_loss, (int, float)) or not math.isfinite(best_loss)):
        raise ValueError("Checkpoint best-validation metadata is invalid.")
    config = payload.get("model_config")
    if not isinstance(config, dict):
        raise ValueError("Checkpoint is missing its model configuration.")
    _validate_model_config(model_name, config)
    if run.get("models", {}).get(model_name) != config:
        raise ValueError("Checkpoint model configuration conflicts with run metadata.")
    history = payload.get("history")
    if not isinstance(history, list) or not history:
        raise ValueError("Checkpoint is missing its training history.")
    best_rows = [row for row in history if row.get("is_best") is True]
    if not best_rows or best_rows[-1].get("epoch") != best_epoch:
        raise ValueError("Checkpoint history does not agree with its best epoch.")
    history_best_loss = float(best_rows[-1].get("validation_loss", math.nan))
    if not math.isclose(history_best_loss, float(best_loss), rel_tol=1e-7):
        raise ValueError("Checkpoint history does not agree with its best validation loss.")
    if not isinstance(payload.get("model_state_dict"), dict):
        raise ValueError("Checkpoint has no model state dictionary.")
    return config


def build_model(model_name: str, config: dict[str, Any]) -> nn.Module:
    _validate_model_config(model_name, config)
    if model_name == "LinearGridTransformer":
        return LinearGridForecaster(
            in_channels=config["in_channels"], out_channels=config["out_channels"],
            t_forecast=config["horizon"], d_model=config["d_model"],
            num_heads=config["num_heads"], ffn_dim=config["ffn_dim"],
            dropout=config["dropout"],
        )
    return CDLinearBaseline(
        seq_len=config["seq_len"], pred_len=config["pred_len"],
        in_channels=config["in_channels"], out_channels=config["out_channels"],
    )


def _torch_load(path: Path, map_location: str | torch.device) -> dict[str, Any]:
    try:
        return torch.load(path, map_location=map_location, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=map_location)


def load_best_model(
    checkpoint_path: str | Path,
    model_name: str,
    expected_dataset: dict[str, Any],
    device: str | torch.device = "cpu",
) -> tuple[nn.Module, dict[str, Any]]:
    path = Path(checkpoint_path)
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint does not exist: {path}")
    try:
        payload = _torch_load(path, device)
    except Exception as exc:
        raise ValueError(f"Could not load checkpoint {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"Checkpoint {path} must contain a mapping payload.")
    config = validate_checkpoint_payload(payload, model_name, expected_dataset)
    model = build_model(model_name, config)
    try:
        model.load_state_dict(payload["model_state_dict"], strict=True)
    except (RuntimeError, TypeError) as exc:
        raise ValueError(f"Checkpoint weights are incompatible with {model_name}: {exc}") from exc
    model.to(device)
    model.eval()
    return model, payload


def inverse_normalize_targets(
    values: np.ndarray | torch.Tensor, mean: Any, std: Any
) -> np.ndarray | torch.Tensor:
    """Invert target z-scores using the training partition's statistics."""
    if isinstance(values, torch.Tensor):
        mean_value = torch.as_tensor(mean, dtype=values.dtype, device=values.device)
        std_value = torch.as_tensor(std, dtype=values.dtype, device=values.device)
        return values * std_value + mean_value
    values64 = np.asarray(values, dtype=np.float64)
    return values64 * np.asarray(std, dtype=np.float64) + np.asarray(mean, dtype=np.float64)


def calculate_metrics(
    predictions: np.ndarray, targets: np.ndarray, zone_names: tuple[str, ...] = TARGET_NAMES
) -> dict[str, Any]:
    """Compute per-zone metrics, macro averages, and distinct pooled metrics."""
    predictions = np.asarray(predictions, dtype=np.float64)
    targets = np.asarray(targets, dtype=np.float64)
    if predictions.shape != targets.shape or predictions.ndim != 3:
        raise ValueError("Predictions and targets must have matching [samples, horizon, zones] shapes.")
    if predictions.shape[-1] != len(zone_names) or predictions.size == 0:
        raise ValueError("Zone names must match a non-empty prediction channel dimension.")
    if not np.isfinite(predictions).all() or not np.isfinite(targets).all():
        raise ValueError("Metrics require finite predictions and targets.")
    errors = predictions - targets
    per_zone: dict[str, dict[str, float]] = {}
    for channel, name in enumerate(zone_names):
        error = errors[..., channel]
        per_zone[name] = {
            "mae": float(np.mean(np.abs(error))),
            "rmse": float(np.sqrt(np.mean(np.square(error)))),
        }
    macro = {
        metric: float(np.mean([entry[metric] for entry in per_zone.values()]))
        for metric in ("mae", "rmse")
    }
    pooled = {
        "mae": float(np.mean(np.abs(errors))),
        "rmse": float(np.sqrt(np.mean(np.square(errors)))),
    }
    return {
        "per_zone": per_zone,
        "macro_average_across_zones": macro,
        "pooled_across_all_samples_horizons_and_zones": pooled,
    }


def training_mean_baseline(sample_count: int, horizon: int, target_mean: Any) -> np.ndarray:
    """Repeat each train-only zone mean across samples and horizon."""
    means = np.asarray(target_mean, dtype=np.float64)
    if sample_count <= 0 or horizon <= 0 or means.shape != (3,):
        raise ValueError("Expected positive sample/horizon counts and three target means.")
    return np.broadcast_to(means, (sample_count, horizon, 3)).copy()


def persistence_baseline(
    normalized_last_zone_values: np.ndarray | torch.Tensor,
    horizon: int,
    target_mean: Any,
    target_std: Any,
) -> np.ndarray:
    """Repeat each final input zone value at raw numeric scale."""
    if normalized_last_zone_values.ndim != 2 or normalized_last_zone_values.shape[-1] != 3:
        raise ValueError("Final zone values must have shape [samples, 3].")
    if horizon <= 0:
        raise ValueError("horizon must be positive.")
    raw_last = inverse_normalize_targets(normalized_last_zone_values, target_mean, target_std)
    if isinstance(raw_last, torch.Tensor):
        raw_last = raw_last.detach().cpu().numpy()
    return np.repeat(np.asarray(raw_last, dtype=np.float64)[:, None, :], horizon, axis=1)


def write_report(report: dict[str, Any], output_path: str | Path) -> Path:
    path = Path(output_path)
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite existing evaluation report: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(report, indent=2, allow_nan=False)
    with path.open("x", encoding="utf-8") as stream:
        stream.write(serialized)
    return path


def _select_device(requested: str | None) -> torch.device:
    if requested:
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def evaluate_checkpoints(
    lgt_checkpoint: str | Path,
    cdlinear_checkpoint: str | Path,
    *,
    artifact_dir: str | Path = DEFAULT_OUTPUT_DIR,
    results_dir: str | Path = DEFAULT_RESULTS_DIR,
    output_name: str | None = None,
    batch_size: int = 64,
    num_workers: int = 0,
    device: str | None = None,
) -> dict[str, Any]:
    if batch_size <= 0 or num_workers < 0:
        raise ValueError("batch_size must be positive and num_workers non-negative.")
    device_value = _select_device(device)
    metadata_path = Path(artifact_dir) / "metadata.json"
    if not metadata_path.is_file():
        raise FileNotFoundError(f"Tetouan metadata not found: {metadata_path}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    expected_dataset = dataset_contract(metadata)
    target_mean, target_std = target_normalization_stats(metadata)

    # Both selected experiment checkpoints are validated before creating the test dataset.
    models: dict[str, tuple[nn.Module, dict[str, Any], Path]] = {}
    for name, checkpoint_path in (
        ("LinearGridTransformer", lgt_checkpoint), ("CDLinear", cdlinear_checkpoint)
    ):
        model, payload = load_best_model(checkpoint_path, name, expected_dataset, device_value)
        models[name] = (model, payload, Path(checkpoint_path).resolve())

    if output_name is None:
        output_name = f"evaluation-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.json"
    if Path(output_name).name != output_name or not output_name.endswith(".json"):
        raise ValueError("output_name must be a .json filename without directory components.")
    report_path = Path(results_dir) / output_name
    if report_path.exists():
        raise FileExistsError(f"Refusing to overwrite existing evaluation report: {report_path}")

    # This is the first and only test-split dataset construction in this entry point.
    test_dataset = TetouanSequenceDataset(split="test", artifact_dir=artifact_dir)
    if dataset_contract(test_dataset.metadata) != expected_dataset:
        raise ValueError("Test dataset metadata changed while preparing evaluation.")
    loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers,
        drop_last=False, pin_memory=device_value.type == "cuda",
    )
    targets_normalized: list[np.ndarray] = []
    last_zones_normalized: list[np.ndarray] = []
    predictions_normalized: dict[str, list[np.ndarray]] = {name: [] for name in MODEL_NAMES}
    with torch.inference_mode():
        for x, y in loader:
            targets_normalized.append(y.cpu().numpy())
            last_zones_normalized.append(x[:, -1, :3].cpu().numpy())
            for name, (model, _payload, _path) in models.items():
                prediction = model(x.to(device_value, non_blocking=device_value.type == "cuda"))
                if prediction.shape != y.shape:
                    raise ValueError(
                        f"{name} prediction shape {tuple(prediction.shape)} does not match "
                        f"target shape {tuple(y.shape)}."
                    )
                predictions_normalized[name].append(prediction.cpu().numpy())

    y_raw = inverse_normalize_targets(np.concatenate(targets_normalized), target_mean, target_std)
    last_zones = np.concatenate(last_zones_normalized)
    mean_prediction = training_mean_baseline(
        len(test_dataset), test_dataset.horizon, target_mean
    )
    persistence_prediction = persistence_baseline(
        last_zones, test_dataset.horizon, target_mean, target_std
    )
    model_results: dict[str, Any] = {}
    checkpoint_info: dict[str, Any] = {}
    for name, (_model, payload, path) in models.items():
        predictions_raw = inverse_normalize_targets(
            np.concatenate(predictions_normalized[name]), target_mean, target_std
        )
        model_results[name] = calculate_metrics(predictions_raw, y_raw)
        checkpoint_info[name] = {
            "path": str(path),
            "model_config": payload["model_config"],
            "best_epoch": payload["checkpoint_metadata"]["best_epoch"],
            "best_validation_loss": payload["checkpoint_metadata"]["best_validation_loss"],
            "selection_metric": payload["checkpoint_metadata"]["selection_metric"],
            "run_name": payload["run_metadata"].get("run_name"),
        }

    baselines = {
        "training_mean": calculate_metrics(mean_prediction, y_raw),
        "persistence": calculate_metrics(persistence_prediction, y_raw),
    }
    first_window = test_dataset.window_timestamps(0)
    last_window = test_dataset.window_timestamps(len(test_dataset) - 1)
    target_stats = metadata["normalization"]
    report = {
        "report_schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": {
            "name": expected_dataset["dataset_name"],
            "source_url": expected_dataset["source_url"],
            "source_file": expected_dataset["source_file"],
            "source_sha256": expected_dataset["source_sha256"],
            "doi": expected_dataset["doi"],
            "attribution": metadata["source"].get("attribution"),
            "license": metadata["source"].get("license"),
            "units_status": expected_dataset["units_status"],
            "timezone_status": expected_dataset["timezone_status"],
        },
        "split": {
            "name": "test",
            "method": metadata["split"]["method"],
            "partition": metadata["split"]["partitions"]["test"],
            "evaluation_sample_count": len(test_dataset),
            "target_timestamp_range": {
                "start": first_window["target_start"], "end": last_window["target_end"],
            },
            "partition_timestamp_range": {
                "start": test_dataset.partition_boundaries["start_timestamp"],
                "end": test_dataset.partition_boundaries["last_timestamp"],
            },
        },
        "normalization": {
            "method": target_stats["method"],
            "fit_partition": target_stats["fit_partition"],
            "fit_row_range_half_open": target_stats["fit_row_range_half_open"],
            "target_channel_names": target_stats["target_names"],
            "target_training_mean": target_mean.tolist(),
            "target_training_std": target_std.tolist(),
            "inverse_transform": "raw = normalized * training_target_std + training_target_mean",
            "units": "undocumented; metrics are raw numeric values without physical unit assignment",
        },
        "checkpoints": checkpoint_info,
        "metrics": {
            "aggregation_note": (
                "Per-zone metrics aggregate across all test windows and forecast steps. "
                "Macro-average is the arithmetic mean of zone metrics; pooled metrics "
                "aggregate samples, horizons, and zones together."
            ),
            "models": model_results,
            "baselines": baselines,
        },
        "test_usage": {
            "statement": "The test set was used only for this final evaluation; it was not used for training, checkpoint selection, or model configuration.",
            "used_for_training": False,
            "used_for_checkpoint_selection": False,
            "used_for_model_configuration": False,
        },
    }
    write_report(report, report_path)
    print(f"Tetouan held-out evaluation report: {report_path}")
    return report


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def _nonnegative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be non-negative")
    return parsed


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lgt-checkpoint", required=True)
    parser.add_argument("--cdlinear-checkpoint", required=True)
    parser.add_argument("--artifact-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--results-dir", default=str(DEFAULT_RESULTS_DIR))
    parser.add_argument("--output-name", default=None)
    parser.add_argument("--batch-size", type=_positive_int, default=64)
    parser.add_argument("--num-workers", type=_nonnegative_int, default=0)
    parser.add_argument("--device", default=None, help="cpu, cuda, or mps; default auto-detect")
    args = parser.parse_args(argv)
    evaluate_checkpoints(
        args.lgt_checkpoint, args.cdlinear_checkpoint,
        artifact_dir=args.artifact_dir, results_dir=args.results_dir,
        output_name=args.output_name, batch_size=args.batch_size,
        num_workers=args.num_workers, device=args.device,
    )


if __name__ == "__main__":
    main()
