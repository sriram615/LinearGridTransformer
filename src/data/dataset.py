"""
dataset.py
----------
Core Deep Learning Engineer — LinearGridTransformer project.

Provides GridSequenceDataset: a PyTorch Dataset that serves
(lookback, forecast) window pairs from the household power
consumption data without materialising the full 3-D tensor in RAM.

Memory strategy
---------------
  • The cleaned DataFrame is loaded once and its 7-channel block is
    converted to a contiguous float32 NumPy array (C-order).
  • __getitem__ slices only the rows it needs; PyTorch converts the
    resulting (T, C) sub-array to a Tensor on the fly — O(window) not
    O(N) per call.
  • Normalisation is applied at slice-time using pre-computed scalars,
    so no normalised copy of the full dataset ever exists in memory.

Parameters
----------
T_LOOKBACK : int   — history window length  (default 1440 min = 24 h)
T_FORECAST : int   — forecast horizon length (default   60 min =  1 h)
"""

from __future__ import annotations

import pathlib
from typing import Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

# ── Default window sizes (minutes = rows, 1 row ≡ 1 min) ─────────────────────
T_LOOKBACK: int = 1440   # 24 h of history
T_FORECAST: int =   60   # 1 h ahead to predict

# ── Feature columns — must match clean_chunks.py / compute_stats.py order ────
FEATURE_COLS: list[str] = [
    "Global_active_power",
    "Global_reactive_power",
    "Voltage",
    "Global_intensity",
    "Sub_metering_1",
    "Sub_metering_2",
    "Sub_metering_3",
]
N_CHANNELS: int = len(FEATURE_COLS)   # 7

# ── Default artifact paths (relative to repo root) ────────────────────────────
_REPO_ROOT  = pathlib.Path(__file__).resolve().parents[2]
_DEFAULT_PKL  = _REPO_ROOT / "data" / "processed" / "clean_grid.pkl"
_DEFAULT_MEAN = _REPO_ROOT / "src"  / "data"      / "global_mean.npy"
_DEFAULT_STD  = _REPO_ROOT / "src"  / "data"      / "global_std.npy"


class GridSequenceDataset(Dataset):
    """
    On-demand sliding-window dataset for the LinearGridTransformer.

    Each sample is a pair:
        x : float32 Tensor of shape (T_lookback, N_channels)   — input history
        y : float32 Tensor of shape (T_forecast, N_channels)   — forecast target

    Only the raw data array is kept in RAM; individual windows are
    sliced and normalised lazily inside __getitem__.

    Parameters
    ----------
    pkl_path  : path to data/processed/clean_grid.pkl
    mean_path : path to src/data/global_mean.npy  (shape 7,)
    std_path  : path to src/data/global_std.npy   (shape 7,)
    t_lookback: history window in rows (default T_LOOKBACK)
    t_forecast: forecast horizon in rows (default T_FORECAST)
    split     : 'train' | 'val' | 'all'
                'train' → first 80 % of rows (matches compute_stats.py)
                'val'   → last  20 % of rows
                'all'   → all rows
    """

    def __init__(
        self,
        pkl_path:   str | pathlib.Path = _DEFAULT_PKL,
        mean_path:  str | pathlib.Path = _DEFAULT_MEAN,
        std_path:   str | pathlib.Path = _DEFAULT_STD,
        t_lookback: int = T_LOOKBACK,
        t_forecast: int = T_FORECAST,
        split:      str = "train",
        sub_sample: bool = True,
    ) -> None:
        super().__init__()

        self.t_lookback = t_lookback
        self.t_forecast = t_forecast
        self._window    = t_lookback + t_forecast
        self.sub_sample = sub_sample

        # ── Load and validate normalisation scalars ───────────────────────────
        mean_path = pathlib.Path(mean_path)
        std_path  = pathlib.Path(std_path)

        if not mean_path.exists():
            raise FileNotFoundError(f"global_mean.npy not found: {mean_path}\n"
                                    "Run src/utils/compute_stats.py first.")
        if not std_path.exists():
            raise FileNotFoundError(f"global_std.npy not found: {std_path}\n"
                                    "Run src/utils/compute_stats.py first.")

        # shape (7,) float64 → float32 for tensor ops
        self._mean: np.ndarray = np.load(str(mean_path)).astype(np.float32)  # (C,)
        self._std:  np.ndarray = np.load(str(std_path )).astype(np.float32)  # (C,)

        assert self._mean.shape == (N_CHANNELS,), \
            f"mean.npy shape mismatch: expected ({N_CHANNELS},), got {self._mean.shape}"
        assert self._std.shape  == (N_CHANNELS,), \
            f"std.npy shape mismatch:  expected ({N_CHANNELS},), got {self._std.shape}"

        # ── Load cleaned data ─────────────────────────────────────────────────
        import pandas as pd

        pkl_path = pathlib.Path(pkl_path)
        if not pkl_path.exists():
            raise FileNotFoundError(f"Cleaned pickle not found: {pkl_path}\n"
                                    "Run src/utils/clean_chunks.py first.")

        df = pd.read_pickle(str(pkl_path))

        # ── Apply split and handle subsampling ────────────────────────────────
        n_total    = len(df)
        split_idx  = int(n_total * 0.80)

        if self.sub_sample:
            if split not in ("train", "val"):
                raise ValueError(
                    f"sub_sample=True is only supported for split='train' and split='val', got '{split}'"
                )

            # Determine the subset DataFrame for the split
            df_split = df.iloc[:split_idx] if split == "train" else df.iloc[split_idx:]

            # Extract year and month to group by
            df_split = df_split.copy()
            df_split["_dt"] = pd.to_datetime(df_split["Date"], format="%d/%m/%Y")
            df_split["_year"] = df_split["_dt"].dt.year
            df_split["_month"] = df_split["_dt"].dt.month

            blocks = []
            valid_indices = []
            current_offset = 0

            # Sort groups chronologically
            for _, group in df_split.groupby(["_year", "_month"], sort=True):
                # Target length: 1 week (10,080 minutes) for train, 2 days (2,880 minutes) for val
                block_len_target = 10080 if split == "train" else 2880
                block_df = group.head(block_len_target)
                block_len = len(block_df)
                if block_len == 0:
                    continue

                block_arr = block_df[FEATURE_COLS].to_numpy(dtype=np.float32, copy=False)
                blocks.append(block_arr)

                # Crossover Boundary Protection Mapper:
                # Map sliding windows strictly within the boundaries of each independent slice.
                # Valid starting positions for block i of length L_i at a given current_offset:
                # [current_offset, current_offset + L_i - self._window]
                # which translates directly to:
                # [offset, offset + L_i - (lookback + forecast)]
                if block_len >= self._window:
                    for i in range(block_len - self._window + 1):
                        valid_indices.append(current_offset + i)

                current_offset += block_len

            self._data = np.concatenate(blocks, axis=0)
            self._valid_indices = np.array(valid_indices, dtype=np.int64)
            self._n_samples = len(self._valid_indices)

        else:
            # Select only the 7 feature columns and convert to a contiguous
            # float32 NumPy array — the *only* large allocation in this class.
            raw: np.ndarray = (
                df[FEATURE_COLS]
                .to_numpy(dtype=np.float32, copy=False)   # (N_total, 7), C-contiguous
            )

            if split == "train":
                self._data = raw[:split_idx]
            elif split == "val":
                self._data = raw[split_idx:]
            elif split == "all":
                self._data = raw
            else:
                raise ValueError(f"split must be 'train', 'val', or 'all', got '{split}'")

            self._n_samples = max(0, len(self._data) - self._window + 1)
            self._valid_indices = np.arange(self._n_samples, dtype=np.int64)

    # ── Dataset interface ─────────────────────────────────────────────────────

    def __len__(self) -> int:
        return self._n_samples

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns
        -------
        x : Tensor  shape (T_lookback, 7)  — normalised input window
        y : Tensor  shape (T_forecast, 7)  — normalised forecast target
        """
        if idx < 0 or idx >= self._n_samples:
            raise IndexError(
                f"Index {idx} out of range for dataset of size {self._n_samples}."
            )

        # Slice using the pre-filtered valid indices to avoid boundary crossing
        start = self._valid_indices[idx]
        mid   = start + self.t_lookback
        end   = mid   + self.t_forecast

        x_raw: np.ndarray = self._data[start:mid]   # (T_lookback, 7)
        y_raw: np.ndarray = self._data[mid:end]      # (T_forecast, 7)

        # Z-score normalisation: (x - mean) / std  — broadcast over rows
        x_norm = (x_raw - self._mean) / self._std    # (T_lookback, 7)
        y_norm = (y_raw - self._mean) / self._std    # (T_forecast, 7)

        # Convert to Tensors — shares memory with the NumPy arrays via
        # torch.from_numpy, so no additional allocation occurs.
        return (
            torch.from_numpy(x_norm.copy()),   # .copy() required: slice is a view
            torch.from_numpy(y_norm.copy()),
        )

    # ── Convenience properties ────────────────────────────────────────────────

    @property
    def shape(self) -> dict:
        """Report dataset shape metadata."""
        return {
            "n_samples":  self._n_samples,
            "n_rows_raw": len(self._data),
            "t_lookback": self.t_lookback,
            "t_forecast": self.t_forecast,
            "n_channels": N_CHANNELS,
            "x_shape":    (self.t_lookback, N_CHANNELS),
            "y_shape":    (self.t_forecast, N_CHANNELS),
            "sub_sample": self.sub_sample,
        }

    def __repr__(self) -> str:
        s = self.shape
        return (
            f"GridSequenceDataset("
            f"n_samples={s['n_samples']:,}, "
            f"x={s['x_shape']}, "
            f"y={s['y_shape']},"
            f" sub_sample={s['sub_sample']})"
        )
