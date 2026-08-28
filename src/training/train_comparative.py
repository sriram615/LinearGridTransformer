"""
train_comparative.py
--------------------
GPU-ready comparative training routine for the LinearGridTransformer and the
CDLinearBaseline model, with real per-epoch validation metrics.

Changes from the CPU-only smoke-test version
----------------------------------------------
The original version of this script hardcoded ``device = torch.device("cpu")``
and only ever reported *training* loss — there was no validation-set signal
for either model, so "which model is better" could only be read off train
loss curves, which is not the number that matters. This version:

  1. Auto-detects CUDA (falls back to MPS, then CPU) so it actually uses a
     Kaggle/Colab GPU when one is available.
  2. Runs a real validation pass after every epoch, via the same
     ``run_evaluation`` used by evaluate.py — so you get per-channel MAE/RMSE
     and the naive-mean-baseline comparison for *both* models, not just a
     training-loss number.
  3. Saves a checkpoint per model at the end of training.
  4. Removes the hard per-epoch batch cap (was 3000, or 15 in the earlier
     smoke-test config) in favour of running the full sub-sampled training
     set each epoch — use --max-batches to reintroduce a cap if a run is
     taking too long.

Usage
-----
    python -m src.training.train_comparative                    # full run
    python -m src.training.train_comparative --epochs 3          # quick check
    DRY_RUN=true python -m src.training.train_comparative         # 2-batch sanity check
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import math
import os
import pathlib
import sys
import time

import psutil

cores = os.cpu_count() or 4
import torch

torch.set_num_threads(cores)
torch.set_num_interop_threads(2)

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

import torch.nn as nn
from torch.utils.data import DataLoader

from src.data.dataset import GridSequenceDataset
from src.training.train import LinearGridForecaster
from src.training.evaluate import run_evaluation, _load_norm_stats
from src.models.cdlinear import CDLinearBaseline
from src.models.kernel import TaylorFeatureMap


# ── Monkey-patch TaylorFeatureMap to avoid the 5D outer-product / advanced
#    indexing path — verified numerically and gradient-equivalent to the
#    original in src/utils/test_gradient.py. ─────────────────────────────────
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

    quad = (x[..., self.triu_rows] * x[..., self.triu_cols]) * self.quad_scale

    return torch.cat([ones, linear, quad], dim=-1)


TaylorFeatureMap.forward = patched_forward

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("train_comparative")


def get_rss_mb() -> float:
    return psutil.Process(os.getpid()).memory_info().rss / 1e6


def pick_device(forced: str | None = None) -> torch.device:
    if forced:
        return torch.device(forced)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    max_batches: int | None,
    name: str = "",
    log_every: int = 200,
) -> float:
    model.train()
    running_loss, total_samples = 0.0, 0
    n_batches = len(loader) if max_batches is None else min(max_batches, len(loader))
    t_start = time.perf_counter()

    for batch_idx, (x, y) in enumerate(loader):
        if max_batches is not None and batch_idx >= max_batches:
            break

        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        pred = model(x)
        loss = criterion(pred, y)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        b_size = x.size(0)
        running_loss += loss.item() * b_size
        total_samples += b_size

        # ── Per-batch progress — without this, a ~10k-batch epoch on a real
        # dataset prints nothing for several minutes, which looks identical
        # to a hang even when the GPU panel shows 100% utilization. ─────────
        if (batch_idx + 1) % log_every == 0 or (batch_idx + 1) == n_batches:
            elapsed = time.perf_counter() - t_start
            samples_per_s = total_samples / max(elapsed, 1e-9)
            log.info(
                "  [%s] batch %5d/%d  running_loss=%.6f  %.1f samples/s  elapsed=%.0fs",
                name, batch_idx + 1, n_batches, running_loss / max(total_samples, 1),
                samples_per_s, elapsed,
            )

        del loss, pred, x, y

    return running_loss / max(total_samples, 1)


def train_and_validate(
    name: str,
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    mean: torch.Tensor,
    std: torch.Tensor,
    device: torch.device,
    epochs: int,
    max_batches: int | None,
    val_max_batches: int | None,
) -> dict:
    """Train `model` for `epochs`, running a full validation pass after each
    epoch. Returns a dict of per-epoch train loss and val metrics."""
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-2)
    criterion = nn.MSELoss()

    history: dict = {"epoch_train_loss": [], "epoch_val": []}

    for epoch in range(1, epochs + 1):
        t0 = time.perf_counter()
        train_loss = train_one_epoch(
            model, train_loader, optimizer, criterion, device, max_batches,
            name=f"{name} epoch {epoch}/{epochs}",
        )
        train_elapsed = time.perf_counter() - t0

        t1 = time.perf_counter()
        val_metrics = run_evaluation(model, val_loader, mean, std, device, max_batches=val_max_batches)
        val_elapsed = time.perf_counter() - t1

        history["epoch_train_loss"].append(train_loss)
        history["epoch_val"].append({
            "mae_global": val_metrics["mae_global"],
            "rmse_global": val_metrics["rmse_global"],
            "mae_global_naive": val_metrics["mae_global_naive"],
            "beats_naive": val_metrics["mae_global"] < val_metrics["mae_global_naive"],
        })

        log.info(
            "[%s] epoch %2d/%d | train_loss=%.6f | val_MAE=%.4f (naive=%.4f, %s) | "
            "val_RMSE=%.4f | train=%.1fs val=%.1fs | RAM=%.0fMB",
            name, epoch, epochs, train_loss,
            val_metrics["mae_global"], val_metrics["mae_global_naive"],
            "beats naive" if val_metrics["mae_global"] < val_metrics["mae_global_naive"] else "LOSES to naive",
            val_metrics["rmse_global"], train_elapsed, val_elapsed, get_rss_mb(),
        )

        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    return history


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--max-batches", type=int, default=None,
                         help="Cap on training batches per epoch (default: full sub-sampled set).")
    parser.add_argument("--val-max-batches", type=int, default=None,
                         help="Cap on validation batches per epoch (default: full val set).")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    dry_run = os.environ.get("DRY_RUN", "False").lower() in ("true", "1", "yes")
    if dry_run:
        args.epochs = 1
        args.max_batches = 2
        args.val_max_batches = 2
        log.info("DRY_RUN=true — running a 2-batch sanity check only.")

    # Dry-run output goes to clearly-separate filenames so a sanity check can
    # NEVER silently overwrite a real training run's report or checkpoints
    # (this is exactly what happened the first time: re-running the dry run
    # after a real run clobbered the real results because both wrote to the
    # same paths).
    suffix = "_dryrun" if dry_run else ""

    device = pick_device(args.device)
    log.info("Device: %s", device)
    log.info("CPU cores: %d, intra-op threads: %d", cores, torch.get_num_threads())

    log.info("Loading train/val datasets (sub_sample=True) …")
    ds_train = GridSequenceDataset(split="train", sub_sample=True)
    ds_val = GridSequenceDataset(split="val", sub_sample=True)
    log.info("Train: %s", ds_train)
    log.info("Val:   %s", ds_val)

    pin_memory = device.type == "cuda"
    train_loader = DataLoader(
        ds_train, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, drop_last=True, pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        ds_val, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, drop_last=False, pin_memory=pin_memory,
    )

    mean, std = _load_norm_stats()

    results: dict = {
        "config": {
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "max_batches_per_epoch": args.max_batches,
            "val_max_batches_per_epoch": args.val_max_batches,
            "num_workers": args.num_workers,
            "device": str(device),
            "lr": 1e-4,
            "weight_decay": 1e-2,
        }
    }

    ckpt_dir = REPO_ROOT / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # ── 1. LinearGridTransformer ────────────────────────────────────────────
    log.info("=" * 70)
    log.info("Training LinearGridTransformer …")
    log.info("=" * 70)
    torch.manual_seed(42)
    transformer = LinearGridForecaster(in_channels=7, d_model=64, num_heads=4, ffn_dim=256)
    results["LinearGridTransformer"] = train_and_validate(
        "LinearGridTransformer", transformer, train_loader, val_loader,
        mean, std, device, args.epochs, args.max_batches, args.val_max_batches,
    )
    torch.save(
        {"model_state": transformer.state_dict(), "config": results["config"]},
        str(ckpt_dir / f"comparative_lgt{suffix}.pt"),
    )
    log.info("Saved checkpoint → %s", ckpt_dir / f"comparative_lgt{suffix}.pt")
    del transformer
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    # ── 2. CDLinearBaseline ──────────────────────────────────────────────────
    log.info("=" * 70)
    log.info("Training CDLinearBaseline …")
    log.info("=" * 70)
    torch.manual_seed(42)
    cdlinear = CDLinearBaseline(seq_len=1440, pred_len=60, channels=7)
    results["CDLinearBaseline"] = train_and_validate(
        "CDLinearBaseline", cdlinear, train_loader, val_loader,
        mean, std, device, args.epochs, args.max_batches, args.val_max_batches,
    )
    torch.save(
        {"model_state": cdlinear.state_dict(), "config": results["config"]},
        str(ckpt_dir / f"comparative_cdlinear{suffix}.pt"),
    )
    log.info("Saved checkpoint → %s", ckpt_dir / f"comparative_cdlinear{suffix}.pt")
    del cdlinear
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    # ── 3. Save report ────────────────────────────────────────────────────────
    report_dir = REPO_ROOT / "docs"
    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / f"gpu_comparative_report{suffix}.json"
    with open(report_path, "w") as f:
        json.dump(results, f, indent=2)
    log.info("=" * 70)
    log.info("Report saved → %s", report_path)

    # ── 4. Final summary ───────────────────────────────────────────────────────
    lgt_final = results["LinearGridTransformer"]["epoch_val"][-1]
    cdl_final = results["CDLinearBaseline"]["epoch_val"][-1]
    log.info("=" * 70)
    log.info("FINAL (epoch %d) validation MAE — LinearGridTransformer: %.4f  |  CDLinearBaseline: %.4f",
              args.epochs, lgt_final["mae_global"], cdl_final["mae_global"])
    log.info("Naive baseline MAE: %.4f", lgt_final["mae_global_naive"])
    log.info("=" * 70)


if __name__ == "__main__":
    main()
