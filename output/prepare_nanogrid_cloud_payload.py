"""
prepare_nanogrid_cloud_payload.py
---------------------------------
Prepares the raw continuous time-series 2D array and sliding window index registries
for Google Colab deployment. This minimizes the data transfer footprint to mere
megabytes instead of gigabytes by keeping the dataset unexpanded and slicing windows
dynamically on-the-fly.
"""

from __future__ import annotations

import os
import sys
import time
import pathlib
import numpy as np
import pandas as pd

# Add repo root to path so we can import from src
REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.data.dataset import (
    FEATURE_COLS,
    _DEFAULT_PKL,
    _DEFAULT_MEAN,
    _DEFAULT_STD,
    T_LOOKBACK,
    T_FORECAST,
    N_CHANNELS,
)

def _fmt_size(path: pathlib.Path) -> str:
    n = path.stat().st_size
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024:
            return f"{n:.2f} {unit}"
        n /= 1024
    return f"{n:.2f} TB"

def main() -> None:
    print("=" * 70)
    print("  Nanogrid Cloud Payload Preparation Pipeline")
    print("=" * 70)

    # 1. Ingestion & Unexpanded Binary Serialization
    t0 = time.perf_counter()
    
    if not _DEFAULT_PKL.exists():
        print(f"[ERROR] Cleaned pickle not found at {_DEFAULT_PKL}", file=sys.stderr)
        print("Please run 'python3 src/utils/clean_chunks.py' first.", file=sys.stderr)
        sys.exit(1)

    print(f"Ingesting continuous raw dataset from {_DEFAULT_PKL}...")
    df = pd.read_pickle(str(_DEFAULT_PKL))
    
    print(f"Total rows in DataFrame: {len(df):,}")
    print(f"Channels to extract ({N_CHANNELS}): {FEATURE_COLS}")
    
    # Enforce strict float32 formatting
    raw_sequence = df[FEATURE_COLS].to_numpy(dtype=np.float32, copy=False)
    assert raw_sequence.dtype == np.float32, "Target datatype must be float32!"
    print(f"Matrix shape: {raw_sequence.shape} | dtype: {raw_sequence.dtype}")

    # Set up output paths
    out_dir = REPO_ROOT / "output" / "cloud_ready"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    npy_path = out_dir / "nanogrid_raw_continuous.npy"
    npz_path = out_dir / "nanogrid_index_registry.npz"

    print(f"Saving raw continuous 2D matrix directly to {npy_path}...")
    np.save(str(npy_path), raw_sequence)
    print(f"Saved: {npy_path} ({_fmt_size(npy_path)})")
    print("-" * 70)

    # 2. Dataset Boundary Registry Generation
    print("Computing index ranges for train, validation, and test splits...")
    df = df.copy()
    df["_dt"] = pd.to_datetime(df["Date"], format="%d/%m/%Y")
    df["_year"] = df["_dt"].dt.year
    df["_month"] = df["_dt"].dt.month

    split_idx = int(len(df) * 0.80)
    train_df = df.iloc[:split_idx]
    val_df = df.iloc[split_idx:]

    window_size = T_LOOKBACK + T_FORECAST  # 1440 + 60 = 1500

    # Train Split: first 10,080 minutes of each month (39 months)
    train_indices = []
    for _, group in train_df.groupby(["_year", "_month"], sort=True):
        block_start = group.index[0]
        block_len = min(len(group), 10080)
        if block_len >= window_size:
            train_indices.extend(range(block_start, block_start + block_len - window_size + 1))

    # Val Split: first 2,880 minutes of each month (10 months)
    val_indices = []
    for _, group in val_df.groupby(["_year", "_month"], sort=True):
        block_start = group.index[0]
        block_len = min(len(group), 2880)
        if block_len >= window_size:
            val_indices.extend(range(block_start, block_start + block_len - window_size + 1))

    # Test Split: next 2,880 minutes of each month (10 months)
    test_indices = []
    for _, group in val_df.groupby(["_year", "_month"], sort=True):
        block_start = group.index[0]
        test_block_start = block_start + 2880
        test_block_len = min(len(group) - 2880, 2880)
        if test_block_len >= window_size:
            test_indices.extend(range(test_block_start, test_block_start + test_block_len - window_size + 1))

    train_indices = np.array(train_indices, dtype=np.int64)
    val_indices = np.array(val_indices, dtype=np.int64)
    test_indices = np.array(test_indices, dtype=np.int64)

    print(f"Generated train_indices: {len(train_indices):,} samples")
    print(f"Generated val_indices:   {len(val_indices):,} samples")
    print(f"Generated test_indices:  {len(test_indices):,} samples")

    # Load mean and std for self-contained normalization
    if not _DEFAULT_MEAN.exists() or not _DEFAULT_STD.exists():
        print("[WARNING] Global mean/std not found. Dataloader might fail without them.", file=sys.stderr)
        mean = np.zeros(N_CHANNELS, dtype=np.float32)
        std = np.ones(N_CHANNELS, dtype=np.float32)
    else:
        mean = np.load(str(_DEFAULT_MEAN)).astype(np.float32)
        std = np.load(str(_DEFAULT_STD)).astype(np.float32)

    print(f"Saving compressed registry key file to {npz_path}...")
    np.savez_compressed(
        str(npz_path),
        train_indices=train_indices,
        val_indices=val_indices,
        test_indices=test_indices,
        mean=mean,
        std=std,
    )
    print(f"Saved: {npz_path} ({_fmt_size(npz_path)})")
    
    elapsed = time.perf_counter() - t0
    print("=" * 70)
    print(f"  ✔  Payload prep pipeline complete in {elapsed:.2f} s")
    print("=" * 70)

if __name__ == "__main__":
    main()

# ==============================================================================
# 3. GOOGLE COLAB PYTORCH DATALOADER & TRAINING LOOP TEMPLATE (COPY-PASTE READY)
# ==============================================================================
# 
# import numpy as np
# import torch
# from torch.utils.data import Dataset, DataLoader
# from torch.cuda.amp import autocast, GradScaler
# 
# class ColabDynamicNanogridDataset(Dataset):
#     """
#     Memory-safe, high-performance PyTorch Dataset for Google Colab.
#     Ingests the raw 2D continuous matrix via memmap (mmap_mode='r')
#     and slices lookback/forecast windows dynamically on-the-fly.
#     """
#     def __init__(
#         self,
#         npy_path: str,
#         npz_path: str,
#         split: str = "train",
#         t_lookback: int = 1440,
#         t_forecast: int = 60,
#     ) -> None:
#         super().__init__()
#         self.t_lookback = t_lookback
#         self.t_forecast = t_forecast
#         
#         # Ingest unexpanded raw data via memory mapping (read-only, zero copy memory consumption)
#         self.data = np.load(npy_path, mmap_mode="r")
#         
#         # Load split registry and statistics
#         registry = np.load(npz_path)
#         if split == "train":
#             self.indices = registry["train_indices"]
#         elif split == "val":
#             self.indices = registry["val_indices"]
#         elif split == "test":
#             self.indices = registry["test_indices"]
#         else:
#             raise ValueError(f"Unknown split: {split}")
#             
#         self.mean = registry["mean"].astype(np.float32)
#         self.std = registry["std"].astype(np.float32)
#         
#     def __len__(self) -> int:
#         return len(self.indices)
#         
#     def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
#         # Resolve indices
#         start = self.indices[idx]
#         mid = start + self.t_lookback
#         end = mid + self.t_forecast
#         
#         # Dynamic window slicing (reads from disk buffer on-the-fly)
#         x_raw = self.data[start:mid]  # Shape: [T_lookback, 7]
#         y_raw = self.data[mid:end]    # Shape: [T_forecast, 7]
#         
#         # Apply Z-score normalization on-the-fly
#         x_norm = (x_raw - self.mean) / self.std
#         y_norm = (y_raw - self.mean) / self.std
#         
#         # Return torch float32 tensors (copies data to prevent garbage collection problems)
#         return torch.from_numpy(x_norm.copy()), torch.from_numpy(y_norm.copy())
# 
# 
# def train_colab_amp(
#     model: torch.nn.Module,
#     dataloader: DataLoader,
#     epochs: int = 1,
#     lr: float = 1e-4,
#     max_norm: float = 1.0,
# ) -> None:
#     """
#     Custom FP16 Automatic Mixed Precision (AMP) training loop template
#     to optimize Tensor Core utilization on T4/L4 GPUs.
#     """
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     print(f"Training on device: {device}")
#     
#     model.to(device)
#     optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
#     criterion = torch.nn.MSELoss()
#     
#     # FP16 gradient scaler
#     scaler = GradScaler()
#     
#     for epoch in range(1, epochs + 1):
#         model.train()
#         running_loss = 0.0
#         
#         for batch_idx, (x, y) in enumerate(dataloader):
#             x = x.to(device, non_blocking=True)
#             y = y.to(device, non_blocking=True)
#             
#             optimizer.zero_grad(set_to_none=True)
#             
#             # Autocast context to run ops in FP16 precision
#             with autocast():
#                 pred = model(x)
#                 loss = criterion(pred, y)
#                 
#             # Scale the loss and compute gradients
#             scaler.scale(loss).backward()
#             
#             # Unscale gradients before clipping to ensure correct thresholds
#             scaler.unscale_(optimizer)
#             torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_norm)
#             
#             # Perform weight updates and step the scaler
#             scaler.step(optimizer)
#             scaler.update()
#             
#             running_loss += loss.item()
#             if (batch_idx + 1) % 100 == 0:
#                 print(f"Epoch [{epoch}/{epochs}] | Batch {batch_idx + 1}/{len(dataloader)} | Loss: {loss.item():.6f}")
#                 
#         avg_loss = running_loss / len(dataloader)
#         print(f"Epoch [{epoch}/{epochs}] Complete | Average MSE Loss: {avg_loss:.6f}")
# ==============================================================================
