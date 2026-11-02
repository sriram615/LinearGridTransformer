"""Inference service for the 2023 Multi-Zone Regional Grid (NYISO).

Loads approved grid checkpoints, validates raw 144-step observations,
applies train-only normalization, and produces multi-step zonal forecasts in MW.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import torch
from torch import nn

from src.data.tetouan_preprocess import (
    FEATURE_NAMES,
    HORIZON,
    LOOKBACK,
    TARGET_NAMES,
)
from src.training.evaluate_tetouan import build_model, inverse_normalize_targets

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RUN_NAME = "grid_colab_s42"
DEFAULT_CHECKPOINT_DIR = REPO_ROOT / "checkpoints" / "grid"
DEFAULT_RESULTS_DIR = REPO_ROOT / "results" / "grid"
DEFAULT_PROCESSED_DIR = REPO_ROOT / "data" / "processed" / "grid"
MODEL_NAMES = ("LinearGridTransformer", "CDLinear")

ZONE_DISPLAY_NAMES = (
    "New York City (N.Y.C.)",
    "Long Island (LONGIL)",
    "Hudson Valley (HUD VL)",
)


class GridInferenceService:
    """Load and cache regional grid checkpoints for raw-input forecasting."""

    def __init__(
        self,
        *,
        processed_dir: str | Path = DEFAULT_PROCESSED_DIR,
        checkpoint_dir: str | Path = DEFAULT_CHECKPOINT_DIR,
        results_dir: str | Path = DEFAULT_RESULTS_DIR,
        run_name: str = DEFAULT_RUN_NAME,
        device: str | torch.device | None = None,
    ) -> None:
        self.processed_dir = Path(processed_dir)
        self.checkpoint_dir = Path(checkpoint_dir)
        self.results_dir = Path(results_dir)
        self.run_name = run_name
        self.device = self._select_device(device)

        metadata_path = self.processed_dir / "metadata.json"
        if not metadata_path.is_file():
            raise FileNotFoundError(f"Grid metadata not found: {metadata_path}")
        self.metadata: dict[str, Any] = json.loads(metadata_path.read_text(encoding="utf-8"))

        norm = self.metadata["normalization"]
        self.feature_mean = np.asarray(norm["feature_mean"], dtype=np.float64)
        self.feature_std = np.asarray(norm["feature_std"], dtype=np.float64)
        self.target_mean = np.asarray(norm["target_mean"], dtype=np.float64)
        self.target_std = np.asarray(norm["target_std"], dtype=np.float64)

        self.input_channel_names = tuple(self.metadata["features"]["names"])
        self.target_channel_names = tuple(self.metadata["targets"]["names"])
        self.target_display_names = ZONE_DISPLAY_NAMES

        # Load models
        self.models: dict[str, nn.Module] = {}
        self.checkpoints: dict[str, dict[str, Any]] = {}
        self._load_checkpoints()

    def _select_device(self, requested: str | torch.device | None) -> torch.device:
        if isinstance(requested, torch.device):
            return requested
        if requested:
            return torch.device(requested)
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    def _load_checkpoints(self) -> None:
        paths = {
            "LinearGridTransformer": self.checkpoint_dir / f"{self.run_name}_lineargridtransformer.pt",
            "CDLinear": self.checkpoint_dir / f"{self.run_name}_cdlinear.pt",
        }
        for name in MODEL_NAMES:
            pt_path = paths[name]
            if not pt_path.is_file():
                raise FileNotFoundError(f"Checkpoint not found for {name}: {pt_path}")
            payload = torch.load(pt_path, map_location=self.device)
            model = build_model(name, payload["model_config"]).to(self.device)
            model.load_state_dict(payload["model_state_dict"])
            model.eval()
            self.models[name] = model
            self.checkpoints[name] = payload

    def _validate_and_normalize(
        self,
        observations: Any,
        feature_names: Sequence[str] | None = None,
        timestamps: Sequence[str] | None = None,
    ) -> torch.Tensor:
        """Validate shape (144, 8) and normalize inputs."""
        if isinstance(observations, pd.DataFrame):
            if feature_names is None:
                feature_names = list(observations.columns)
            values = observations.to_numpy(dtype=np.float64)
        else:
            values = np.asarray(observations, dtype=np.float64)

        if values.shape != (LOOKBACK, len(FEATURE_NAMES)):
            raise ValueError(
                f"Expected input observations of shape ({LOOKBACK}, {len(FEATURE_NAMES)}), "
                f"got {values.shape}"
            )
        if not np.isfinite(values).all():
            raise ValueError("Observations contain non-finite or NaN values.")

        if feature_names is not None and tuple(feature_names) != self.input_channel_names:
            raise ValueError(
                f"Feature ordering mismatch. Expected {self.input_channel_names}, got {tuple(feature_names)}"
            )

        # Standardize (value - mean) / std
        normalized = (values - self.feature_mean) / self.feature_std
        tensor = torch.from_numpy(normalized.astype(np.float32)).unsqueeze(0).to(self.device)
        return tensor

    def predict(
        self,
        observations: Any,
        model_name: str,
        *,
        feature_names: Sequence[str] | None = None,
        timestamps: Sequence[str] | None = None,
    ) -> np.ndarray:
        """Predict 6-step horizon (60 minutes) in raw Megawatts (MW)."""
        if model_name not in self.models:
            raise ValueError(f"Unknown model name {model_name!r}; choose from {MODEL_NAMES}")

        tensor = self._validate_and_normalize(observations, feature_names, timestamps)
        model = self.models[model_name]
        with torch.inference_mode():
            pred_norm = model(tensor).cpu().numpy().squeeze(0)  # [6, 3]

        # Inverse transform to real Megawatts
        pred_raw = inverse_normalize_targets(pred_norm, self.target_mean, self.target_std)
        return pred_raw

    def predict_both(
        self,
        observations: Any,
        *,
        feature_names: Sequence[str] | None = None,
        timestamps: Sequence[str] | None = None,
    ) -> dict[str, np.ndarray]:
        """Produce simultaneous forecasts for both CDLinear and LinearGridTransformer."""
        tensor = self._validate_and_normalize(observations, feature_names, timestamps)
        results: dict[str, np.ndarray] = {}
        with torch.inference_mode():
            for name in MODEL_NAMES:
                pred_norm = self.models[name](tensor).cpu().numpy().squeeze(0)
                results[name] = inverse_normalize_targets(pred_norm, self.target_mean, self.target_std)
        return results
