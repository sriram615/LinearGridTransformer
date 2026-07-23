"""
train_comparative.py
--------------------
Parallel vector-optimized comparative training routine for the
LinearGridTransformer and the CDLinearBaseline models.
"""

from __future__ import annotations

import os
import gc
import sys
import time
import json
import math
import pathlib
import logging
import psutil

# Enforce hardware-level vectorization configurations for Intel x86 architectures first
cores = os.cpu_count() or 4
import torch
torch.set_num_threads(cores)
torch.set_num_interop_threads(2)

# Resolve repo root so src.* imports work regardless of CWD
REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

import torch.nn as nn
from torch.utils.data import DataLoader

from src.data.dataset import GridSequenceDataset
from src.training.train import LinearGridForecaster
from src.models.cdlinear import CDLinearBaseline
from src.models.kernel import TaylorFeatureMap

# ── Monkey-Patch TaylorFeatureMap to avoid huge 5D outer-product and slow advanced indexing ──
def patched_forward(self, x: torch.Tensor) -> torch.Tensor:
    if x.dim() != 4:
        raise ValueError(f"TaylorFeatureMap expects a 4-D tensor [B, H, T, d], got shape {tuple(x.shape)}.")
    B, H, T, d = x.shape
    if d != self.d:
        raise ValueError(f"head_dim mismatch: model was built with d={self.d}, but input last dim is {d}.")

    if self.scale_input:
        x = x / math.sqrt(d)

    ones = x.new_ones(B, H, T, 1)
    linear = x

    # Memory-efficient selection (no outer product constructed)
    quad = (x[..., self.triu_rows] * x[..., self.triu_cols]) * self.quad_scale

    return torch.cat([ones, linear, quad], dim=-1)

TaylorFeatureMap.forward = patched_forward

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("train_comparative")

def get_rss_mb() -> float:
    return psutil.Process(os.getpid()).memory_info().rss / 1e6

def train_model(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    epochs: int,
    max_batches: int = 3000,
) -> tuple[list[float], list[float]]:
    """
    Train a model for a specified number of epochs.
    Returns (epoch_losses, batch_losses).
    """
    epoch_losses = []
    batch_losses = []
    
    device = torch.device("cpu")
    model.to(device)
    
    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        total_samples = 0
        t0 = time.perf_counter()
        
        for batch_idx, (x, y) in enumerate(loader):
            if max_batches is not None and batch_idx >= max_batches:
                break
                
            x = x.to(device)
            y = y.to(device)
            
            optimizer.zero_grad(set_to_none=True)
            pred = model(x)
            loss = criterion(pred, y)
            loss.backward()
            
            # Gradient clipping to prevent explosion
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            b_size = x.size(0)
            loss_val = loss.item()
            running_loss += loss_val * b_size
            total_samples += b_size
            batch_losses.append(loss_val)
            
            # Explicitly delete loss inside batch loop to save memory
            del loss, pred, x, y
            
        epoch_loss = running_loss / max(total_samples, 1)
        epoch_losses.append(epoch_loss)
        elapsed = time.perf_counter() - t0
        
        log.info(
            "Epoch %2d/%d | Loss: %.6f | Time: %.1fs | RAM: %.1f MB",
            epoch,
            epochs,
            epoch_loss,
            elapsed,
            get_rss_mb(),
        )
        
        # End of epoch: clear transient allocations and collect garbage
        gc.collect()
        
    return epoch_losses, batch_losses

def main() -> None:
    log.info("Starting comparative training script...")
    log.info(f"Dynamic CPU cores detected: {cores}")
    log.info(f"Intra-op threads set to: {torch.get_num_threads()}")
    log.info("Inter-op threads set to: 2")
    
    # Instantiate dataset and loader
    log.info("Loading sub-sampled train dataset...")
    # shuffle=True forces random index selection across all months/years
    ds_train = GridSequenceDataset(split="train", sub_sample=True)
    loader = DataLoader(
        ds_train,
        batch_size=32,
        shuffle=True,
        num_workers=0,
        drop_last=True,
    )
    log.info(f"Dataset loaded: {ds_train}")
    
    dry_run = os.environ.get("DRY_RUN", "False").lower() in ("true", "1", "yes")
    epochs = 1 if dry_run else 10
    max_batches = 2 if dry_run else 3000  # Hard step cap processing exactly 3,000 steps (96,000 samples) per epoch
    
    log.info(f"Training parameters: epochs={epochs}, batch_size=32, max_batches_per_epoch={max_batches} (DRY_RUN={dry_run})")
    
    criterion = nn.MSELoss()
    
    # ── 1. Train LinearGridTransformer ──────────────────────────────────────────
    log.info("=" * 60)
    log.info("Training LinearGridTransformer...")
    log.info("=" * 60)
    
    torch.manual_seed(42)
    transformer = LinearGridForecaster(
        in_channels=7,
        d_model=64,
        num_heads=4,
        ffn_dim=256,
    )
    optimizer_tf = torch.optim.AdamW(
        transformer.parameters(),
        lr=1e-4,
        weight_decay=1e-2,
    )
    
    tf_epoch_losses, tf_batch_losses = train_model(
        model=transformer,
        loader=loader,
        optimizer=optimizer_tf,
        criterion=criterion,
        epochs=epochs,
        max_batches=max_batches,
    )
    
    # ── Memory management rails: clean RAM completely before baselines ────────
    log.info("Cleaning memory from Transformer allocation...")
    del transformer, optimizer_tf
    gc.collect()
    
    # ── 2. Train CDLinearBaseline ──────────────────────────────────────────────
    log.info("=" * 60)
    log.info("Training CDLinearBaseline...")
    log.info("=" * 60)
    
    torch.manual_seed(42)
    cdlinear = CDLinearBaseline(
        seq_len=1440,
        pred_len=60,
        channels=7,
    )
    optimizer_cdl = torch.optim.AdamW(
        cdlinear.parameters(),
        lr=1e-4,
        weight_decay=1e-2,
    )
    
    cdl_epoch_losses, cdl_batch_losses = train_model(
        model=cdlinear,
        loader=loader,
        optimizer=optimizer_cdl,
        criterion=criterion,
        epochs=epochs,
        max_batches=max_batches,
    )
    
    # Clean baseline allocations
    del cdlinear, optimizer_cdl
    gc.collect()
    
    # ── 3. Save report ────────────────────────────────────────────────────────
    report = {
        "config": {
            "epochs": epochs,
            "batch_size": 32,
            "max_batches_per_epoch": max_batches,
            "lr": 1e-4,
            "weight_decay": 1e-2,
            "hardware": {
                "cpu_cores": cores,
                "num_threads": torch.get_num_threads(),
            }
        },
        "LinearGridTransformer": {
            "epoch_losses": tf_epoch_losses,
            "batch_losses": tf_batch_losses,
        },
        "CDLinearBaseline": {
            "epoch_losses": cdl_epoch_losses,
            "batch_losses": cdl_batch_losses,
        }
    }
    
    report_dir = REPO_ROOT / "docs"
    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / "generalized_comparative_report.json"
    
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
        
    log.info("=" * 60)
    log.info(f"Generalized comparative report successfully saved to {report_path}")
    log.info("=" * 60)

if __name__ == "__main__":
    main()
