#!/usr/bin/env python3
"""
colab_tune_sweep.py
------------------
Automated Hyperparameter Tuning and Cognitive Sweep Script for Google Colab GPU backends.
Ingests seasonally-strided GridSequenceDataset and optimizes LinearGridTransformer (stacked layers)
to bridge the validation MSE gap against the CDLinearBaseline.

Designed specifically to run efficiently on T4, L4, or A100 GPUs in Google Colab.
"""

from __future__ import annotations

import os
import sys
import time
import json
import gc
import pathlib
from typing import Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# ── Robust Path Setup for Google Colab and Local Execution ────────────────────
drive_root = "/content/drive/MyDrive/LinearGridTransformer"
if drive_root not in sys.path:
    sys.path.insert(0, drive_root)

local_root = str(pathlib.Path(__file__).resolve().parents[2])
if local_root not in sys.path:
    sys.path.insert(0, local_root)

# Try imports and guide user if paths are missing
try:
    from src.data.dataset import GridSequenceDataset, T_FORECAST, N_CHANNELS
    from src.models.linear_attention import LinearKernelAttention
except ImportError as err:
    print("❌ Import error. Please verify that the repository is located at:")
    print("   /content/drive/MyDrive/LinearGridTransformer (if running in Colab)")
    print(f"   Current sys.path: {sys.path}")
    raise err

# ── Self-healing Directory Initialization ─────────────────────────────────────
raw_data_path = os.path.join(drive_root, "data/raw/")
metrics_dir = os.path.join(drive_root, "output/metrics/")
checkpoint_dir = os.path.join(drive_root, "output/checkpoints/")

os.makedirs(metrics_dir, exist_ok=True)
os.makedirs(checkpoint_dir, exist_ok=True)

trajectory_path = os.path.join(metrics_dir, "colab_trajectory.json")
best_checkpoint_path = os.path.join(checkpoint_dir, "best_colab_transformer.pt")


# ═══════════════════════════════════════════════════════════════════════════════
#  Stacked Multi-Layer Linear Transformer Encoder Architecture
# ═══════════════════════════════════════════════════════════════════════════════

class ForecastHead(nn.Module):
    """
    Slices the last T_forecast time-steps from the encoder output and
    projects from d_model → N_channels via a single linear layer.
    """
    def __init__(self, d_model: int, t_forecast: int, n_channels: int) -> None:
        super().__init__()
        self.t_forecast = t_forecast
        self.proj = nn.Linear(d_model, n_channels)
        nn.init.xavier_uniform_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x : [B, T, d_model]
        tail = x[:, -self.t_forecast:, :]
        return self.proj(tail)


class GridTransformerEncoderLayer(nn.Module):
    """
    Single-layer block that operates entirely within the d_model feature space.
    Does not project back to in_channels inside the layer.
    """
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        ffn_dim: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        assert d_model % num_heads == 0, f"d_model ({d_model}) must divide by num_heads ({num_heads})"
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        # Linear projections for Multi-Head Attention
        self.W_Q = nn.Linear(d_model, d_model, bias=False)
        self.W_K = nn.Linear(d_model, d_model, bias=False)
        self.W_V = nn.Linear(d_model, d_model, bias=False)
        self.W_O = nn.Linear(d_model, d_model, bias=False)

        # O(T) linear kernel attention
        self.attn = LinearKernelAttention(head_dim=self.head_dim, num_heads=num_heads)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.ffn = nn.Sequential(
            nn.Linear(d_model, ffn_dim),
            nn.GELU(),
            nn.Dropout(p=dropout),
            nn.Linear(ffn_dim, d_model),
            nn.Dropout(p=dropout),
        )
        self._init_weights()

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        B, T, _ = x.shape
        return x.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

    def _merge_heads(self, x: torch.Tensor) -> torch.Tensor:
        B, H, T, _ = x.shape
        return x.transpose(1, 2).contiguous().view(B, T, self.d_model)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        # Multi-head attention with residual connection
        Q = self._split_heads(self.W_Q(h))
        K = self._split_heads(self.W_K(h))
        V = self._split_heads(self.W_V(h))

        attn_out, _ = self.attn(Q, K, V)
        attn_out = self._merge_heads(attn_out)
        attn_out = self.W_O(attn_out)

        h = self.norm1(h + attn_out)

        # Feed-forward network with residual connection
        ffn_out = self.ffn(h)
        h = self.norm2(h + ffn_out)
        return h


class GridTransformerEncoderMultiLayer(nn.Module):
    """
    Stacks multiple GridTransformerEncoderLayer blocks inside d_model latent space.
    Takes [B, T, in_channels] -> projects to d_model -> stacks layers -> projects back to in_channels.
    """
    def __init__(
        self,
        in_channels: int = 7,
        d_model: int = 64,
        num_heads: int = 4,
        ffn_dim: int = 256,
        dropout: float = 0.0,
        num_layers: int = 2,
    ) -> None:
        super().__init__()
        self.input_proj = nn.Linear(in_channels, d_model, bias=False)
        self.layers = nn.ModuleList([
            GridTransformerEncoderLayer(
                d_model=d_model,
                num_heads=num_heads,
                ffn_dim=ffn_dim,
                dropout=dropout,
            ) for _ in range(num_layers)
        ])
        self.output_proj = nn.Linear(d_model, in_channels, bias=False)
        nn.init.xavier_uniform_(self.input_proj.weight)
        nn.init.xavier_uniform_(self.output_proj.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.input_proj(x)
        for layer in self.layers:
            h = layer(h)
        return self.output_proj(h)


class ColabLinearGridForecaster(nn.Module):
    """
    Full Wrapper model for Colab training (Encoder + Head).
    """
    def __init__(
        self,
        in_channels: int = 7,
        d_model: int = 64,
        num_heads: int = 4,
        ffn_dim: int = 256,
        t_forecast: int = 60,
        dropout: float = 0.4,
        num_layers: int = 2,
    ) -> None:
        super().__init__()
        self.encoder = GridTransformerEncoderMultiLayer(
            in_channels=in_channels,
            d_model=d_model,
            num_heads=num_heads,
            ffn_dim=ffn_dim,
            dropout=dropout,
            num_layers=num_layers,
        )
        self.head = ForecastHead(
            d_model=in_channels,
            t_forecast=t_forecast,
            n_channels=in_channels,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        enc = self.encoder(x)
        return self.head(enc)


# ═══════════════════════════════════════════════════════════════════════════════
#  Validation Evaluation Loop
# ═══════════════════════════════════════════════════════════════════════════════

def evaluate(model: nn.Module, val_loader: DataLoader, criterion: nn.Module, device: torch.device) -> float:
    model.eval()
    val_loss = 0.0
    total_samples = 0
    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(device), y.to(device)
            pred = model(x)
            loss = criterion(pred, y)
            b_size = x.size(0)
            val_loss += loss.item() * b_size
            total_samples += b_size
    return val_loss / max(total_samples, 1)


# ═══════════════════════════════════════════════════════════════════════════════
#  Main Tuning Orchestrator
# ═══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    print("=" * 70)
    print("    COLAB HYPERPARAMETER SWEEP: LINEAR GRID TRANSFORMER")
    print("=" * 70)

    # 1. Hardware device selection
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Hardware Backend: {device} (CUDA is {'AVAILABLE 🚀' if torch.cuda.is_available() else 'UNAVAILABLE ⚠️'})")

    # Dynamic multi-process data loading settings
    num_workers = 4 if torch.cuda.is_available() else 0
    pin_memory = torch.cuda.is_available()
    batch_size = 512

    # 2. Ingest Seasonally-Strided Dataset
    # Select path based on environment
    pkl_filename = "data/processed/clean_grid.pkl"
    mean_filename = "src/data/global_mean.npy"
    std_filename = "src/data/global_std.npy"

    if os.path.exists(os.path.join(drive_root, pkl_filename)):
        root_dir = drive_root
        print("Using Google Drive dataset paths.")
    else:
        root_dir = local_root
        print("Google Drive paths not detected. Falling back to local directory structures.")

    pkl_path = os.path.join(root_dir, pkl_filename)
    mean_path = os.path.join(root_dir, mean_filename)
    std_path = os.path.join(root_dir, std_filename)

    print(f"Loading files:\n - PKL:  {pkl_path}\n - Mean: {mean_path}\n - Std:  {std_path}")

    try:
        train_dataset = GridSequenceDataset(pkl_path=pkl_path, mean_path=mean_path, std_path=std_path, split="train")
        val_dataset = GridSequenceDataset(pkl_path=pkl_path, mean_path=mean_path, std_path=std_path, split="val")
    except FileNotFoundError as e:
        print(f"❌ Failed to find dataset files: {e}")
        print("Make sure you have processed the raw data locally or mounted Google Drive correctly.")
        sys.exit(1)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False
    )

    print(f"Train Dataset size: {len(train_dataset):,} samples ({len(train_loader)} batches)")
    print(f"Val Dataset size:   {len(val_dataset):,} samples ({len(val_loader)} batches)")
    print("-" * 70)

    # 3. Load Trajectory or Initialize new
    if os.path.exists(trajectory_path):
        try:
            with open(trajectory_path, "r") as f:
                trajectory = json.load(f)
            print(f"Loaded existing sweep trajectory with {len(trajectory)} runs.")
        except Exception:
            trajectory = []
    else:
        trajectory = []

    # Restore best global loss from existing runs
    global_best_val_loss = float('inf')
    for run in trajectory:
        if run.get("best_val_loss", float('inf')) < global_best_val_loss:
            global_best_val_loss = run["best_val_loss"]
    print(f"Baseline Target: CDLinearBaseline (Val MSE: 0.6436)")
    print(f"Current best sweep validation loss: {global_best_val_loss:.6f}")
    print("-" * 70)

    # 4. Sweep Over Architectural Configurations Grid
    d_model_grid = [64, 128]
    nhead_grid = [4, 8]
    num_layers_grid = [2, 3]

    configs = []
    for d_model in d_model_grid:
        for nhead in nhead_grid:
            for num_layers in num_layers_grid:
                # Assert perfect head division alignment
                if d_model % nhead == 0:
                    configs.append((d_model, nhead, num_layers))

    print(f"Prepared grid sweep of {len(configs)} candidate configurations.")

    for idx, (d_model, nhead, num_layers) in enumerate(configs, 1):
        print("\n" + "=" * 70)
        print(f" RUN {idx}/{len(configs)}: d_model={d_model} | nhead={nhead} | num_layers={num_layers}")
        print("=" * 70)

        # Check if already completed in trajectory
        already_run = False
        for run in trajectory:
            if run["d_model"] == d_model and run["num_heads"] == nhead and run["num_layers"] == num_layers:
                print("Configuration already evaluated in trajectory. Skipping.")
                already_run = True
                break
        if already_run:
            continue

        # Instantiate Multi-Layer Forecaster
        model = ColabLinearGridForecaster(
            in_channels=N_CHANNELS,
            d_model=d_model,
            num_heads=nhead,
            ffn_dim=4 * d_model,
            t_forecast=T_FORECAST,
            dropout=0.4,
            num_layers=num_layers,
        ).to(device)

        # Optimization Setup
        optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-2)
        
        # Step Capping logic
        steps_per_epoch = min(len(train_loader), 3000)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=5e-4,
            steps_per_epoch=steps_per_epoch,
            epochs=15,
        )

        criterion = nn.MSELoss()

        train_losses = []
        val_losses = []
        best_val_loss = float('inf')

        # Train Loop (15 Epochs)
        for epoch in range(15):
            t_start = time.time()
            model.train()
            epoch_loss = 0.0
            total_samples = 0

            for batch_idx, (x, y) in enumerate(train_loader):
                if batch_idx >= 3000:
                    break

                x, y = x.to(device), y.to(device)

                # Forward pass
                pred = model(x)
                loss = criterion(pred, y)

                # Backprop
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                optimizer.step()
                scheduler.step()

                b_size = x.size(0)
                epoch_loss += loss.item() * b_size
                total_samples += b_size

            avg_train_loss = epoch_loss / max(total_samples, 1)
            avg_val_loss = evaluate(model, val_loader, criterion, device)

            train_losses.append(avg_train_loss)
            val_losses.append(avg_val_loss)

            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss

            epoch_time = time.time() - t_start
            print(f"  Epoch {epoch+1:02d}/15 | Train Loss: {avg_train_loss:.6f} | Val Loss: {avg_val_loss:.6f} | Time: {epoch_time:.1f}s")

            # Check and Save Global Checkpoint
            if avg_val_loss < global_best_val_loss:
                global_best_val_loss = avg_val_loss
                torch.save({
                    "epoch": epoch + 1,
                    "d_model": d_model,
                    "num_heads": nhead,
                    "num_layers": num_layers,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss": avg_val_loss,
                }, best_checkpoint_path)
                print(f"  ✨ Global Best Model updated! Saving to: {best_checkpoint_path}")

        # Save configuration trajectory
        trajectory.append({
            "d_model": d_model,
            "num_heads": nhead,
            "num_layers": num_layers,
            "train_losses": train_losses,
            "val_losses": val_losses,
            "best_val_loss": best_val_loss,
            "time_completed": time.asctime()
        })

        with open(trajectory_path, "w") as f:
            json.dump(trajectory, f, indent=4)
        print(f"  Saved trajectory trace to {trajectory_path}")

        # ═══════════════════════════════════════════════════════════════════════════
        #  VOLATILE MEMORY SANITIZATION (VRAM RAILS)
        # ═══════════════════════════════════════════════════════════════════════════
        # Clean up current model variables
        if 'model' in locals():
            del model
        if 'optimizer' in locals():
            del optimizer

        # Clear PyTorch CUDA Cache and run garbage collection
        torch.cuda.empty_cache()
        gc.collect()

    print("\n" + "=" * 70)
    print("🎉 HYPERPARAMETER SWEEP SWEEP COMPLETE!")
    print(f"Best Validation Loss achieved: {global_best_val_loss:.6f}")
    print(f"Target Baseline (CDLinear): 0.6436")
    if global_best_val_loss < 0.6436:
        print("🚀 Successfully bridged the validation convergence gap against CDLinear!")
    else:
        print("⚠️ Validation MSE still above CDLinear. Consider widening the search space.")
    print("=" * 70)


if __name__ == "__main__":
    main()
