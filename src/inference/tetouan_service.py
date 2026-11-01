from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import torch
from torch import nn

from src.data.tetouan_preprocess import (
    DEFAULT_OUTPUT_DIR,
    FEATURE_NAMES,
    HORIZON,
    LOOKBACK,
    TARGET_NAMES,
)
from src.training.evaluate_tetouan import (
    MODEL_NAMES,
    dataset_contract,
    inverse_normalize_targets,
    load_best_model,
    target_normalization_stats,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RUN_NAME = "tetouan_t4_20261101_v1"
DEFAULT_CHECKPOINT_DIR = REPO_ROOT / "checkpoints" / "tetouan"
DEFAULT_EXPERIMENT_DIR = REPO_ROOT / "results" / "tetouan" / DEFAULT_RUN_NAME


class TetouanInferenceService:
    """Load and cache approved Tetouan checkpoints for raw-input inference.

    DataFrame input must use canonical feature columns in order. NumPy input
    must include ordered feature_names. Values are raw, unnormalized values.
    Optional timestamps (or a DataFrame DatetimeIndex) are checked for
    timezone-naive, chronological ten-minute cadence.
    """

    def __init__(
        self,
        *,
        artifact_dir: str | Path = DEFAULT_OUTPUT_DIR,
        checkpoint_dir: str | Path = DEFAULT_CHECKPOINT_DIR,
        experiment_dir: str | Path = DEFAULT_EXPERIMENT_DIR,
        run_name: str = DEFAULT_RUN_NAME,
        device: str | torch.device | None = None,
        checkpoint_paths: Mapping[str, str | Path] | None = None,
        manifest_path: str | Path | None = None,
    ) -> None:
        self.artifact_dir = Path(artifact_dir)
        self.checkpoint_dir = Path(checkpoint_dir)
        self.experiment_dir = Path(experiment_dir)
        self.run_name = run_name
        self.device = self._select_device(device)

        metadata_path = self.artifact_dir / "metadata.json"
        if not metadata_path.is_file():
            raise FileNotFoundError(f"Tetouan metadata not found: {metadata_path}")
        self.metadata: dict[str, Any] = json.loads(metadata_path.read_text(encoding="utf-8"))
        self._validate_dataset_metadata()
        self.expected_dataset = dataset_contract(self.metadata)
        self.feature_mean = np.asarray(
            self.metadata["normalization"]["feature_mean"], dtype=np.float64
        )
        self.feature_std = np.asarray(
            self.metadata["normalization"]["feature_std"], dtype=np.float64
        )
        self.target_mean, self.target_std = target_normalization_stats(self.metadata)

        resolved_manifest = Path(manifest_path) if manifest_path else (
            self.experiment_dir / f"{run_name}_manifest.json"
        )
        if not resolved_manifest.is_file():
            raise FileNotFoundError(f"Experiment manifest not found: {resolved_manifest}")
        self.manifest_path = resolved_manifest
        self.manifest = json.loads(resolved_manifest.read_text(encoding="utf-8"))
        self._validate_manifest_and_artifacts()

        paths = {
            "LinearGridTransformer": self.checkpoint_dir
            / f"{run_name}_lineargridtransformer.pt",
            "CDLinear": self.checkpoint_dir / f"{run_name}_cdlinear.pt",
        }
        if checkpoint_paths is not None:
            unknown = set(checkpoint_paths) - set(MODEL_NAMES)
            if unknown:
                raise ValueError(f"Unknown checkpoint model names: {sorted(unknown)}")
            paths.update({name: Path(path) for name, path in checkpoint_paths.items()})

        # Instances are loaded once and retained for subsequent predictions.
        self.models: dict[str, nn.Module] = {}
        self.checkpoints: dict[str, dict[str, Any]] = {}
        for model_name in MODEL_NAMES:
            model, payload = load_best_model(
                paths[model_name],
                model_name,
                self.expected_dataset,
                device=self.device,
            )
            self._validate_loaded_checkpoint(model_name, payload)
            model.eval()
            self.models[model_name] = model
            self.checkpoints[model_name] = payload

        self.input_channel_names = tuple(FEATURE_NAMES)
        self.target_channel_names = tuple(TARGET_NAMES)

    @staticmethod
    def _select_device(device: str | torch.device | None) -> torch.device:
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        result = torch.device(device)
        if result.type not in ("cpu", "cuda"):
            raise ValueError("Inference device must be CPU or CUDA.")
        if result.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested, but CUDA is not available.")
        return result

    def _validate_dataset_metadata(self) -> None:
        if self.metadata.get("features", {}).get("names") != list(FEATURE_NAMES):
            raise ValueError("Dataset metadata feature order does not match the inference contract.")
        if self.metadata.get("targets", {}).get("names") != list(TARGET_NAMES):
            raise ValueError("Dataset metadata target order does not match the inference contract.")
        windows = self.metadata.get("windows", {})
        if windows.get("lookback") != LOOKBACK or windows.get("horizon") != HORIZON:
            raise ValueError("Dataset metadata lookback/horizon does not match the inference contract.")
        if self.metadata.get("timestamps", {}).get("cadence_minutes") != 10:
            raise ValueError("Dataset metadata cadence does not match ten-minute inference cadence.")
        normalization = self.metadata.get("normalization", {})
        if normalization.get("fit_partition") != "train":
            raise ValueError("Normalization statistics must be fitted on the training partition.")
        self._check_stats(
            normalization.get("feature_mean"), normalization.get("feature_std"),
            len(FEATURE_NAMES), "feature",
        )
        self._check_stats(
            normalization.get("target_mean"), normalization.get("target_std"),
            len(TARGET_NAMES), "target",
        )

    @staticmethod
    def _check_stats(mean: Any, std: Any, size: int, label: str) -> None:
        mean_array = np.asarray(mean, dtype=np.float64)
        std_array = np.asarray(std, dtype=np.float64)
        if (
            mean_array.shape != (size,)
            or std_array.shape != (size,)
            or not np.isfinite(mean_array).all()
            or not np.isfinite(std_array).all()
            or np.any(std_array <= 0)
        ):
            raise ValueError(f"Invalid {label} normalization statistics.")

    def _validate_manifest_and_artifacts(self) -> None:
        run = self.manifest.get("run_metadata")
        if not isinstance(run, dict):
            raise ValueError("Experiment manifest is missing run_metadata.")
        if run.get("run_name") != self.run_name or run.get("run_type") != "experiment":
            raise ValueError("Manifest run name/type is incompatible with this inference service.")
        if run.get("test_partition_used") is not False:
            raise ValueError("Manifest does not establish test-partition isolation.")
        expected_contract = json.loads(json.dumps(self.expected_dataset))
        if run.get("dataset") != expected_contract:
            raise ValueError("Manifest dataset contract does not match local Tetouan metadata.")

        manifest_models = run.get("models")
        records = self.manifest.get("models")
        if not isinstance(manifest_models, dict) or not isinstance(records, dict):
            raise ValueError("Manifest is missing model configurations.")
        for model_name in MODEL_NAMES:
            record = records.get(model_name)
            if not isinstance(record, dict) or record.get("configuration") != manifest_models.get(model_name):
                raise ValueError(f"Manifest {model_name} configurations are inconsistent.")

        fingerprints = run.get("processed_artifacts")
        if not isinstance(fingerprints, dict):
            raise ValueError("Manifest has no processed artifact fingerprints.")
        for filename in ("metadata.json", "series.npz"):
            expected = fingerprints.get(filename)
            path = self.artifact_dir / filename
            if not isinstance(expected, dict) or not path.is_file():
                raise ValueError(f"Missing processed artifact fingerprint/file: {filename}")
            digest = hashlib.sha256()
            with path.open("rb") as stream:
                for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(chunk)
            if path.stat().st_size != expected.get("size_bytes"):
                raise ValueError(f"Processed artifact size differs from manifest: {filename}")
            if digest.hexdigest() != expected.get("sha256"):
                raise ValueError(f"Processed artifact hash differs from manifest: {filename}")

    def _validate_loaded_checkpoint(self, model_name: str, payload: dict[str, Any]) -> None:
        run = payload.get("run_metadata", {})
        if run.get("run_name") != self.run_name:
            raise ValueError(f"{model_name} checkpoint belongs to a different run.")
        record = self.manifest["models"][model_name]
        if record.get("configuration") != payload.get("model_config"):
            raise ValueError(f"{model_name} checkpoint configuration differs from manifest.")
        selection = payload.get("checkpoint_metadata", {})
        try:
            loss_matches = np.isclose(
                record["best_validation_loss"],
                selection["best_validation_loss"],
                rtol=1e-7,
                atol=0.0,
            )
        except (KeyError, TypeError, ValueError):
            loss_matches = False
        if (
            record.get("best_validation_epoch") != selection.get("best_epoch")
            or not loss_matches
        ):
            raise ValueError(f"{model_name} checkpoint selection differs from manifest.")

    def _validate_timestamps(
        self, timestamps: Sequence[Any] | pd.Index, expected_count: int
    ) -> None:
        try:
            parsed = pd.DatetimeIndex(pd.to_datetime(list(timestamps), errors="coerce"))
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("Timestamps must be valid timezone-naive datetime values.") from exc
        if len(parsed) != expected_count or parsed.hasnans:
            raise ValueError(f"Expected {expected_count} valid timestamps.")
        if parsed.tz is not None:
            raise ValueError("Timezone-aware timestamps are not accepted; dataset timestamps are naive.")
        if not parsed.is_monotonic_increasing or parsed.has_duplicates:
            raise ValueError("Timestamps must be strictly chronological without duplicates.")
        if len(parsed) > 1:
            deltas = parsed[1:] - parsed[:-1]
            if not bool((deltas == pd.Timedelta(minutes=10)).all()):
                raise ValueError("Timestamps must have exact ten-minute cadence.")

    def prepare_input(
        self,
        observations: pd.DataFrame | np.ndarray,
        *,
        feature_names: Sequence[str] | None = None,
        timestamps: Sequence[Any] | None = None,
    ) -> torch.Tensor:
        """Validate raw observations and return normalized shape [1, 144, 8]."""
        dataframe_timestamps = None
        if isinstance(observations, pd.DataFrame):
            if tuple(observations.columns) != tuple(FEATURE_NAMES):
                raise ValueError(
                    f"DataFrame columns must exactly match canonical feature order {FEATURE_NAMES}."
                )
            if feature_names is not None and tuple(feature_names) != tuple(FEATURE_NAMES):
                raise ValueError("feature_names do not match the canonical feature order.")
            if isinstance(observations.index, pd.DatetimeIndex):
                dataframe_timestamps = observations.index
            values_object = observations.to_numpy(copy=True)
        elif isinstance(observations, np.ndarray):
            if feature_names is None:
                raise ValueError("feature_names are required when observations is a NumPy array.")
            if tuple(feature_names) != tuple(FEATURE_NAMES):
                raise ValueError("feature_names do not match the canonical feature order.")
            values_object = observations
        else:
            raise TypeError("observations must be a pandas DataFrame or NumPy array.")

        if timestamps is not None and dataframe_timestamps is not None:
            raise ValueError("Supply timestamps either as the DataFrame index or as an argument, not both.")
        timestamp_values = timestamps if timestamps is not None else dataframe_timestamps
        try:
            values = np.asarray(values_object, dtype=np.float64)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("Observations must contain numeric values.") from exc
        if values.shape != (LOOKBACK, len(FEATURE_NAMES)):
            raise ValueError(
                f"Expected exactly [{LOOKBACK}, {len(FEATURE_NAMES)}] observations/features; "
                f"got {values.shape}."
            )
        if not np.isfinite(values).all():
            raise ValueError("Observations must contain only finite values.")
        if timestamp_values is not None:
            self._validate_timestamps(timestamp_values, LOOKBACK)

        normalized = (values - self.feature_mean) / self.feature_std
        if not np.isfinite(normalized).all():
            raise ValueError("Normalized observations contain non-finite values.")
        return torch.as_tensor(normalized, dtype=torch.float32).unsqueeze(0)

    def _predict_prepared(self, model_name: str, x: torch.Tensor) -> np.ndarray:
        if model_name not in self.models:
            raise ValueError(f"Unsupported model {model_name!r}; expected one of {MODEL_NAMES}.")
        with torch.inference_mode():
            prediction = self.models[model_name](x.to(self.device))
        expected_shape = (1, HORIZON, len(TARGET_NAMES))
        if tuple(prediction.shape) != expected_shape:
            raise RuntimeError(
                f"{model_name} returned {tuple(prediction.shape)}, expected {expected_shape}."
            )
        normalized_values = prediction[0].detach().cpu().numpy().astype(np.float64, copy=False)
        raw_scale = inverse_normalize_targets(
            normalized_values, self.target_mean, self.target_std
        )
        result = np.asarray(raw_scale, dtype=np.float64)
        if not np.isfinite(result).all():
            raise FloatingPointError(f"{model_name} returned non-finite forecasts.")
        return result

    def predict(
        self,
        observations: pd.DataFrame | np.ndarray,
        model_name: str,
        *,
        feature_names: Sequence[str] | None = None,
        timestamps: Sequence[Any] | None = None,
    ) -> np.ndarray:
        """Return one model's original-scale forecast with shape [6, 3]."""
        x = self.prepare_input(
            observations, feature_names=feature_names, timestamps=timestamps
        )
        return self._predict_prepared(model_name, x)

    def predict_both(
        self,
        observations: pd.DataFrame | np.ndarray,
        *,
        feature_names: Sequence[str] | None = None,
        timestamps: Sequence[Any] | None = None,
    ) -> dict[str, np.ndarray]:
        """Forecast with both cached models using one shared validated input."""
        x = self.prepare_input(
            observations, feature_names=feature_names, timestamps=timestamps
        )
        return {
            model_name: self._predict_prepared(model_name, x)
            for model_name in MODEL_NAMES
        }
