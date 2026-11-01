"""Split-aware sliding-window Dataset for Tetouan multi-zone forecasting."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

from src.data.tetouan_preprocess import (
    DEFAULT_OUTPUT_DIR,
    FEATURE_NAMES,
    HORIZON,
    LOOKBACK,
    TARGET_NAMES,
    TetouanDataError,
)


class TetouanSequenceDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    """Return normalized ``(X, Y)`` windows for one chronological partition.

    Validation and test history may include rows from earlier partitions;
    each target window is constrained to its own assigned partition.
    """

    SPLITS = ("train", "validation", "test")

    def __init__(
        self,
        split: str,
        artifact_dir: str | Path = DEFAULT_OUTPUT_DIR,
        lookback: int = LOOKBACK,
        horizon: int = HORIZON,
    ) -> None:
        if split not in self.SPLITS:
            raise ValueError(f"split must be one of {self.SPLITS}, got {split!r}")
        if lookback <= 0 or horizon <= 0:
            raise ValueError("lookback and horizon must be positive integers")
        self.split = split
        self.lookback = int(lookback)
        self.horizon = int(horizon)
        self.artifact_dir = Path(artifact_dir)
        arrays_path = self.artifact_dir / "series.npz"
        metadata_path = self.artifact_dir / "metadata.json"
        if not arrays_path.is_file() or not metadata_path.is_file():
            raise FileNotFoundError(
                f"Tetouan artifacts not found under {self.artifact_dir}; "
                "run src.data.tetouan_preprocess first."
            )

        import json
        self.metadata: dict[str, Any] = json.loads(metadata_path.read_text(encoding="utf-8"))
        with np.load(arrays_path, allow_pickle=False) as arrays:
            self._timestamps = arrays["timestamps"].astype("datetime64[ns]", copy=False)
            self._features = arrays["features"].astype(np.float64, copy=False)
            self._targets = arrays["targets"].astype(np.float64, copy=False)

        n_rows = int(self.metadata["row_count"])
        if self._features.shape != (n_rows, len(FEATURE_NAMES)):
            raise TetouanDataError(f"features array has unexpected shape {self._features.shape}")
        if self._targets.shape != (n_rows, len(TARGET_NAMES)):
            raise TetouanDataError(f"targets array has unexpected shape {self._targets.shape}")
        if self._timestamps.shape != (n_rows,):
            raise TetouanDataError(f"timestamps array has unexpected shape {self._timestamps.shape}")
        if self.metadata["features"]["names"] != list(FEATURE_NAMES):
            raise TetouanDataError("Metadata feature ordering does not match the dataset contract.")
        if self.metadata["targets"]["names"] != list(TARGET_NAMES):
            raise TetouanDataError("Metadata target ordering does not match the dataset contract.")

        normalization = self.metadata["normalization"]
        self._feature_mean = np.asarray(normalization["feature_mean"], dtype=np.float64)
        self._feature_std = np.asarray(normalization["feature_std"], dtype=np.float64)
        self._target_mean = np.asarray(normalization["target_mean"], dtype=np.float64)
        self._target_std = np.asarray(normalization["target_std"], dtype=np.float64)
        self._validate_stats()

        self.partition = dict(self.metadata["split"]["partitions"][split])
        self.start_index = int(self.partition["start_index"])
        self.end_index = int(self.partition["end_index_exclusive"])
        self.n_rows = n_rows
        first_target = max(self.start_index, self.lookback)
        last_target_start = self.end_index - self.horizon
        self._target_starts = np.arange(first_target, last_target_start + 1, dtype=np.int64)
        if len(self._target_starts) == 0:
            raise TetouanDataError(f"Partition {split!r} cannot provide a full target window.")

        self.input_channel_names = tuple(self.metadata["features"]["names"])
        self.target_channel_names = tuple(self.metadata["targets"]["names"])
        self.partition_boundaries = {
            **self.partition,
            "start_timestamp": self._timestamp_string(self.start_index),
            "end_timestamp_exclusive": self._timestamp_string(self.end_index)
            if self.end_index < self.n_rows else None,
            "last_timestamp": self._timestamp_string(self.end_index - 1),
        }

    def _validate_stats(self) -> None:
        expected_shapes = (
            (self._feature_mean, (len(FEATURE_NAMES),), "feature_mean"),
            (self._feature_std, (len(FEATURE_NAMES),), "feature_std"),
            (self._target_mean, (len(TARGET_NAMES),), "target_mean"),
            (self._target_std, (len(TARGET_NAMES),), "target_std"),
        )
        for values, shape, name in expected_shapes:
            if values.shape != shape or not np.isfinite(values).all():
                raise TetouanDataError(f"Invalid normalization metadata: {name} must have shape {shape} and finite values.")
        if np.any(self._feature_std <= 0) or np.any(self._target_std <= 0):
            raise TetouanDataError("Normalization standard deviations must be positive.")

    def _timestamp_string(self, index: int) -> str:
        if index < 0 or index >= self.n_rows:
            raise IndexError(f"row index {index} has no observed timestamp")
        return self._timestamps[index].astype("datetime64[m]").astype(str).replace("T", " ")

    def __len__(self) -> int:
        return int(len(self._target_starts))

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        if not isinstance(index, (int, np.integer)):
            raise TypeError(f"index must be an integer, got {type(index).__name__}")
        if index < 0 or index >= len(self):
            raise IndexError(f"index {index} out of range for {self.split} dataset of size {len(self)}")
        target_start = int(self._target_starts[index])
        input_start = target_start - self.lookback
        target_end = target_start + self.horizon
        x = (self._features[input_start:target_start] - self._feature_mean) / self._feature_std
        y = (self._targets[target_start:target_end] - self._target_mean) / self._target_std
        return (
            torch.from_numpy(np.asarray(x, dtype=np.float32).copy()),
            torch.from_numpy(np.asarray(y, dtype=np.float32).copy()),
        )

    def window_timestamps(self, index: int) -> dict[str, str]:
        """Return inclusive ISO timestamp ranges for the sample's X and Y."""
        if not isinstance(index, (int, np.integer)) or index < 0 or index >= len(self):
            raise IndexError(f"index {index!r} out of range for {self.split} dataset of size {len(self)}")
        target_start = int(self._target_starts[index])
        return {
            "input_start": self._timestamp_string(target_start - self.lookback),
            "input_end": self._timestamp_string(target_start - 1),
            "target_start": self._timestamp_string(target_start),
            "target_end": self._timestamp_string(target_start + self.horizon - 1),
        }

    @property
    def target_start_indices(self) -> np.ndarray:
        """Read-only copy of global row indices where target windows begin."""
        result = self._target_starts.copy()
        result.flags.writeable = False
        return result

    @property
    def input_shape(self) -> tuple[int, int]:
        return self.lookback, len(self.input_channel_names)

    @property
    def target_shape(self) -> tuple[int, int]:
        return self.horizon, len(self.target_channel_names)

    def __repr__(self) -> str:
        return (
            f"TetouanSequenceDataset(split={self.split!r}, n_samples={len(self)}, "
            f"x={self.input_shape}, y={self.target_shape})"
        )
