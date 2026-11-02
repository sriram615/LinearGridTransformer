"""Reproducible comparative training entry point for Tetouan City data.

Full run::

    .venv311/bin/python3 -m src.training.train_tetouan --epochs 10

Clearly marked CPU-friendly loop smoke test (one epoch, two train and two
validation batches per model)::

    .venv311/bin/python3 -m src.training.train_tetouan --smoke

Every invocation writes uniquely named artifacts under checkpoints/tetouan/
and results/tetouan/. Existing run artifacts are never overwritten.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import platform
import random
import re
import subprocess
import sys
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CHECKPOINT_DIR = REPO_ROOT / "checkpoints" / "tetouan"
DEFAULT_RESULTS_DIR = REPO_ROOT / "results" / "tetouan"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data.tetouan_dataset import TetouanSequenceDataset
from src.data.tetouan_preprocess import DEFAULT_OUTPUT_DIR
from src.models.cdlinear import CDLinearBaseline
from src.training.train import LinearGridForecaster


@dataclass(frozen=True)
class TrainConfig:
    epochs: int = 10
    batch_size: int = 32
    num_workers: int = 0
    learning_rate: float = 1e-3
    weight_decay: float = 1e-2
    seed: int = 42
    grad_clip_norm: float = 1.0
    max_train_batches: int | None = None
    max_validation_batches: int | None = None
    smoke: bool = False
    device: str | None = None
    d_model: int = 64
    num_heads: int = 4
    ffn_dim: int = 256
    dropout: float = 0.1

    def validate(self) -> None:
        if self.epochs <= 0 or self.batch_size <= 0:
            raise ValueError("epochs and batch_size must be positive.")
        if self.num_workers < 0:
            raise ValueError("num_workers cannot be negative.")
        if self.learning_rate <= 0 or self.weight_decay < 0:
            raise ValueError("learning_rate must be positive and weight_decay non-negative.")
        if self.grad_clip_norm <= 0:
            raise ValueError("grad_clip_norm must be positive.")
        if self.seed < 0:
            raise ValueError("seed cannot be negative.")
        for name, count in (("max_train_batches", self.max_train_batches),
                            ("max_validation_batches", self.max_validation_batches)):
            if count is not None and count <= 0:
                raise ValueError(f"{name} must be positive when supplied.")


def seed_everything(seed: int) -> None:
    """Reset Python, NumPy, and PyTorch RNGs before each model is initialized."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def select_device(requested: str | None = None) -> torch.device:
    if requested:
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def initialize_model(name: str, config: TrainConfig) -> nn.Module:
    """Build an independent model after resetting the configured seed."""
    seed_everything(config.seed)
    if name == "LinearGridTransformer":
        return LinearGridForecaster(
            in_channels=8,
            out_channels=3,
            t_forecast=6,
            d_model=config.d_model,
            num_heads=config.num_heads,
            ffn_dim=config.ffn_dim,
            dropout=config.dropout,
        )
    if name == "CDLinear":
        return CDLinearBaseline(
            seq_len=144,
            pred_len=6,
            in_channels=8,
            out_channels=3,
        )
    raise ValueError(f"Unknown model name: {name!r}")


def _worker_seed(worker_id: int) -> None:
    del worker_id
    worker_seed = torch.initial_seed() % (2**32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def make_loaders(
    train_dataset: TetouanSequenceDataset,
    validation_dataset: TetouanSequenceDataset,
    config: TrainConfig,
    device: torch.device,
) -> tuple[DataLoader, DataLoader]:
    """Create reproducibly shuffled loaders (same seed for each model)."""
    generator = torch.Generator()
    generator.manual_seed(config.seed)
    common = {
        "batch_size": config.batch_size,
        "num_workers": config.num_workers,
        "pin_memory": device.type == "cuda",
        "worker_init_fn": _worker_seed,
    }
    train_loader = DataLoader(
        train_dataset, shuffle=True, drop_last=False, generator=generator, **common
    )
    validation_loader = DataLoader(
        validation_dataset, shuffle=False, drop_last=False, **common
    )
    return train_loader, validation_loader


def _loss_for_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    *,
    training: bool,
    optimizer: torch.optim.Optimizer | None = None,
    grad_clip_norm: float = 1.0,
    max_batches: int | None = None,
) -> tuple[float, int, int]:
    if training and optimizer is None:
        raise ValueError("An optimizer is required for a training epoch.")
    model.train(training)
    weighted_loss = 0.0
    total_samples = 0
    batches = 0
    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for batch_index, (x, y) in enumerate(loader):
            if max_batches is not None and batch_index >= max_batches:
                break
            x = x.to(device, non_blocking=device.type == "cuda")
            y = y.to(device, non_blocking=device.type == "cuda")
            if training:
                assert optimizer is not None
                optimizer.zero_grad(set_to_none=True)
            prediction = model(x)
            loss = criterion(prediction, y)
            if not torch.isfinite(loss):
                raise FloatingPointError("Encountered a non-finite loss.")
            if training:
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip_norm)
                optimizer.step()
            batch_samples = int(x.shape[0])
            weighted_loss += float(loss.detach().item()) * batch_samples
            total_samples += batch_samples
            batches += 1
    if batches == 0:
        raise RuntimeError("The selected loader produced no batches.")
    return weighted_loss / total_samples, batches, total_samples


class BestValidationCheckpoint:
    """Persist the best state for one model, replacing it only on improvement."""

    def __init__(
        self,
        checkpoint_path: str | Path,
        model_name: str,
        model_config: dict[str, Any],
        run_metadata: dict[str, Any],
    ) -> None:
        self.checkpoint_path = Path(checkpoint_path)
        self.model_name = model_name
        self.model_config = copy.deepcopy(model_config)
        self.run_metadata = copy.deepcopy(run_metadata)
        self.best_epoch: int | None = None
        self.best_validation_loss = math.inf
        self._best_model_state: dict[str, torch.Tensor] | None = None
        self._best_optimizer_state: dict[str, Any] | None = None

    def consider(
        self,
        epoch: int,
        validation_loss: float,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        history: list[dict[str, Any]],
    ) -> bool:
        """Save the epoch state immediately if its validation loss is lower."""
        validation_loss = float(validation_loss)
        if not math.isfinite(validation_loss):
            raise ValueError("Best-checkpoint validation loss must be finite.")
        if self.best_epoch is not None and validation_loss >= self.best_validation_loss:
            return False

        self.best_epoch = int(epoch)
        self.best_validation_loss = validation_loss
        self._best_model_state = {
            key: value.detach().cpu().clone()
            for key, value in model.state_dict().items()
        }
        self._best_optimizer_state = copy.deepcopy(optimizer.state_dict())
        self._save(history)
        return True

    def finalize(self, history: list[dict[str, Any]]) -> None:
        """Rewrite this run's checkpoint with full history and best-state weights."""
        if self.best_epoch is None or self._best_model_state is None:
            raise RuntimeError("No validation checkpoint was selected.")
        self._save(history)

    def _save(self, history: list[dict[str, Any]]) -> None:
        payload = {
            "model_name": self.model_name,
            "model_state_dict": self._best_model_state,
            "optimizer_state_dict": self._best_optimizer_state,
            "model_config": self.model_config,
            "run_metadata": self.run_metadata,
            "checkpoint_metadata": {
                "selection_metric": "validation_loss",
                "best_epoch": self.best_epoch,
                "best_validation_loss": self.best_validation_loss,
            },
            "history": copy.deepcopy(history),
        }
        temporary_path = self.checkpoint_path.with_suffix(
            self.checkpoint_path.suffix + ".tmp"
        )
        torch.save(payload, temporary_path)
        os.replace(temporary_path, self.checkpoint_path)


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    validation_loader: DataLoader,
    config: TrainConfig,
    device: torch.device,
    checkpoint_tracker: BestValidationCheckpoint | None = None,
) -> tuple[list[dict[str, float | int | bool]], dict[str, Any]]:
    """Train one model, retaining full epoch history and its best validation state."""
    model.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    criterion = nn.MSELoss()
    history: list[dict[str, float | int | bool]] = []
    best_validation_loss = math.inf
    best_epoch: int | None = None
    for epoch in range(1, config.epochs + 1):
        train_loss, train_batches, train_samples = _loss_for_epoch(
            model, train_loader, criterion, device,
            training=True, optimizer=optimizer,
            grad_clip_norm=config.grad_clip_norm,
            max_batches=config.max_train_batches,
        )
        validation_loss, validation_batches, validation_samples = _loss_for_epoch(
            model, validation_loader, criterion, device,
            training=False, max_batches=config.max_validation_batches,
        )
        is_best = validation_loss < best_validation_loss
        if is_best:
            best_validation_loss = validation_loss
            best_epoch = epoch
        record: dict[str, float | int | bool] = {
            "epoch": epoch,
            "train_loss": train_loss,
            "validation_loss": validation_loss,
            "train_batches": train_batches,
            "validation_batches": validation_batches,
            "train_samples": train_samples,
            "validation_samples": validation_samples,
            "optimizer_updates": train_batches,
            "is_best": is_best,
            "best_epoch": best_epoch if best_epoch is not None else epoch,
            "best_validation_loss": best_validation_loss,
        }
        history.append(record)
        if is_best and checkpoint_tracker is not None:
            checkpoint_tracker.consider(
                epoch, validation_loss, model, optimizer, history
            )
    return history, optimizer.state_dict()


def load_model_checkpoint(
    model: nn.Module, checkpoint_path: str | Path, map_location: str | torch.device = "cpu"
) -> dict[str, Any]:
    """Load a saved model state and return its checkpoint metadata."""
    try:
        payload = torch.load(checkpoint_path, map_location=map_location, weights_only=True)
    except TypeError:  # compatibility with PyTorch versions before weights_only
        payload = torch.load(checkpoint_path, map_location=map_location)
    model.load_state_dict(payload["model_state_dict"])
    return payload


def _dataset_metadata(dataset: TetouanSequenceDataset) -> dict[str, Any]:
    source = dataset.metadata["source"]
    timestamps = dataset.metadata["timestamps"]
    split = dataset.metadata["split"]["partitions"]
    normalization = dataset.metadata["normalization"]
    return {
        "dataset_name": "Power Consumption of Tetouan City",
        "source_url": source.get("dataset_url"),
        "source_file": source.get("file_name"),
        "source_sha256": source.get("sha256"),
        "doi": source.get("doi"),
        "dataset_row_count": dataset.metadata["row_count"],
        "feature_names": dataset.input_channel_names,
        "target_names": dataset.target_channel_names,
        "lookback": dataset.lookback,
        "horizon": dataset.horizon,
        "cadence_minutes": timestamps.get("cadence_minutes"),
        "timezone_status": timestamps.get("timezone_status"),
        "units_status": dataset.metadata.get("units_status"),
        "normalization": {
            "method": normalization.get("method"),
            "fit_partition": normalization.get("fit_partition"),
            "fit_row_range_half_open": normalization.get("fit_row_range_half_open"),
            "std_ddof": normalization.get("std_ddof"),
        },
        "used_partitions": {
            "train": split["train"],
            "validation": split["validation"],
        },
        "test_partition_used": False,
    }


def _git_commit() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, check=True,
            capture_output=True, text=True,
        )
        return result.stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _artifact_fingerprints(artifact_dir: str | Path) -> dict[str, Any]:
    """Fingerprint the processed inputs without changing them."""
    root = Path(artifact_dir)
    result: dict[str, Any] = {}
    for name in ("metadata.json", "series.npz"):
        path = root / name
        if not path.is_file():
            result[name] = None
            continue
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        result[name] = {
            "path": str(path.resolve()),
            "size_bytes": path.stat().st_size,
            "sha256": digest.hexdigest(),
        }
    return result


def _runtime_metadata(device: torch.device) -> dict[str, Any]:
    cuda_version = getattr(torch.version, "cuda", None)
    gpu_name = None
    if torch.cuda.is_available():
        try:
            gpu_name = torch.cuda.get_device_name(device)
        except (RuntimeError, AssertionError):
            gpu_name = None
    return {
        "python_version": platform.python_version(),
        "pytorch_version": str(torch.__version__),
        "device": str(device),
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_version": cuda_version,
        "gpu_name": gpu_name,
    }


def _run_name(value: str | None, smoke: bool) -> str:
    if value is None:
        prefix = "smoke" if smoke else "run"
        value = f"{prefix}-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", value):
        raise ValueError("run_name may contain only letters, digits, dot, underscore and hyphen.")
    if smoke and not value.startswith("smoke-"):
        value = f"smoke-{value}"
    return value


def run_training(
    config: TrainConfig,
    *,
    checkpoint_dir: str | Path = DEFAULT_CHECKPOINT_DIR,
    results_dir: str | Path = DEFAULT_RESULTS_DIR,
    artifact_dir: str | Path = DEFAULT_OUTPUT_DIR,
    run_name: str | None = None,
) -> dict[str, Any]:
    """Run both models on train/validation only, saving isolated run artifacts."""
    config.validate()
    if config.smoke:
        config = replace(
            config, epochs=1,
            max_train_batches=2, max_validation_batches=2,
        )
    name = _run_name(run_name, config.smoke)
    checkpoint_dir, results_dir = Path(checkpoint_dir), Path(results_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "LinearGridTransformer": checkpoint_dir / f"{name}_lineargridtransformer.pt",
        "CDLinear": checkpoint_dir / f"{name}_cdlinear.pt",
    }
    history_path = results_dir / f"{name}_history.json"
    manifest_path = results_dir / f"{name}_manifest.json"
    collisions = [path for path in (*paths.values(), history_path, manifest_path) if path.exists()]
    if collisions:
        raise FileExistsError("Refusing to overwrite existing run artifacts: "
                              + ", ".join(str(path) for path in collisions))

    # Deliberately construct only the training and validation datasets. The
    # test partition is never requested, iterated, or used for decisions.
    train_dataset = TetouanSequenceDataset(split="train", artifact_dir=artifact_dir)
    validation_dataset = TetouanSequenceDataset(split="validation", artifact_dir=artifact_dir)
    dataset_info = _dataset_metadata(train_dataset)
    if dataset_info["feature_names"] != validation_dataset.input_channel_names:
        raise ValueError("Train and validation feature channels differ.")
    if dataset_info["target_names"] != validation_dataset.target_channel_names:
        raise ValueError("Train and validation target channels differ.")

    device = select_device(config.device)
    model_configs = {
        "LinearGridTransformer": {
            "in_channels": 8, "out_channels": 3, "lookback": 144,
            "horizon": 6, "d_model": config.d_model,
            "num_heads": config.num_heads, "ffn_dim": config.ffn_dim,
            "dropout": config.dropout,
        },
        "CDLinear": {
            "in_channels": 8, "out_channels": 3,
            "seq_len": 144, "pred_len": 6,
        },
    }
    runtime_info = _runtime_metadata(device)
    run_metadata = {
        "run_name": name,
        "run_type": "smoke" if config.smoke else "experiment",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        **runtime_info,
        "dataset": dataset_info,
        "processed_artifacts": _artifact_fingerprints(artifact_dir),
        "training": asdict(config),
        "seed_policy": {
            "seed_everything_before_each_model_initialization": True,
            "data_loader_generator_seed": config.seed,
            "same_shuffled_order_seed_per_model": True,
        },
        "gradient_clipping_max_norm": config.grad_clip_norm,
        "loss": "mean squared error",
        "optimizer": "AdamW",
        "test_partition_used": False,
        "models": model_configs,
    }

    outputs: dict[str, Any] = {}
    for model_name in ("LinearGridTransformer", "CDLinear"):
        # Initialize with the same configured seed and give each model an
        # independent loader generator initialized to that seed.
        model = initialize_model(model_name, config)
        train_loader, validation_loader = make_loaders(
            train_dataset, validation_dataset, config, device
        )
        checkpoint_tracker = BestValidationCheckpoint(
            paths[model_name], model_name, model_configs[model_name], run_metadata
        )
        epoch_history, _optimizer_state = train_model(
            model, train_loader, validation_loader, config, device,
            checkpoint_tracker=checkpoint_tracker,
        )
        checkpoint_tracker.finalize(epoch_history)
        outputs[model_name] = {
            "checkpoint": str(paths[model_name]),
            "history": epoch_history,
            "best_epoch": checkpoint_tracker.best_epoch,
            "best_validation_loss": checkpoint_tracker.best_validation_loss,
        }
        del model, train_loader, validation_loader

    result_payload = {
        "run_metadata": run_metadata,
        "models": outputs,
    }
    history_path.write_text(json.dumps(result_payload, indent=2), encoding="utf-8")
    manifest = {
        "manifest_version": 1,
        "run_metadata": run_metadata,
        "models": {
            model_name: {
                "configuration": model_configs[model_name],
                "checkpoint": record["checkpoint"],
                "best_validation_epoch": record["best_epoch"],
                "best_validation_loss": record["best_validation_loss"],
            }
            for model_name, record in outputs.items()
        },
        "history_path": str(history_path.resolve()),
    }
    with manifest_path.open("x", encoding="utf-8") as stream:
        json.dump(manifest, stream, indent=2)
    print(f"Run type: {run_metadata['run_type']} | run name: {name} | device: {device}")
    for model_name, record in outputs.items():
        last = record["history"][-1]
        print(
            f"{model_name}: train_loss={last['train_loss']:.6f} "
            f"validation_loss={last['validation_loss']:.6f} "
            f"updates={last['optimizer_updates']} checkpoint={record['checkpoint']}"
        )
    print(f"Training history: {history_path}")
    print(f"Experiment manifest: {manifest_path}")
    return result_payload


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


def _positive_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive finite number")
    return parsed


def _nonnegative_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or parsed < 0:
        raise argparse.ArgumentTypeError("must be a non-negative finite number")
    return parsed


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=_positive_int, default=10)
    parser.add_argument("--batch-size", type=_positive_int, default=32)
    parser.add_argument("--num-workers", type=_nonnegative_int, default=0)
    parser.add_argument("--learning-rate", type=_positive_float, default=1e-3)
    parser.add_argument("--weight-decay", type=_nonnegative_float, default=1e-2)
    parser.add_argument("--seed", type=_nonnegative_int, default=42)
    parser.add_argument("--device", default=None, help="cpu, cuda, or mps; default auto-detect")
    parser.add_argument("--checkpoint-dir", default=str(DEFAULT_CHECKPOINT_DIR))
    parser.add_argument("--results-dir", default=str(DEFAULT_RESULTS_DIR))
    parser.add_argument("--artifact-dir", default=str(DEFAULT_OUTPUT_DIR),
                        help="Directory containing series.npz and metadata.json")
    parser.add_argument("--run-name", default=None, help="Unique name; existing output names are refused")
    parser.add_argument("--smoke", action="store_true",
                        help="one epoch, two train and two validation batches per model; not an experiment")
    parser.add_argument("--max-train-batches", type=_positive_int, default=None)
    parser.add_argument("--max-validation-batches", type=_positive_int, default=None)
    args = parser.parse_args(argv)
    config = TrainConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        seed=args.seed,
        max_train_batches=args.max_train_batches,
        max_validation_batches=args.max_validation_batches,
        smoke=args.smoke,
        device=args.device,
    )
    run_training(
        config,
        checkpoint_dir=args.checkpoint_dir,
        results_dir=args.results_dir,
        artifact_dir=args.artifact_dir,
        run_name=args.run_name,
    )


if __name__ == "__main__":
    main()
