"""
evaluate.py
-----------
Principal Applied AI Scientist — LinearGridTransformer project.

Production evaluation pipeline for the LinearGridForecaster.

Pipeline
--------
  1. Load the saved model checkpoint and reconstruct the full model
     (GridTransformerEncoder + ForecastHead).
  2. Switch to model.eval() + torch.no_grad() — no gradient graph ever built.
  3. Pull every batch exclusively from the 20 % validation split
     (GridSequenceDataset with split='val').
  4. Collect all predicted tensors and ground-truth targets, then
     *reverse* the Z-score normalisation using the training-set statistics
     stored in src/data/global_mean.npy and src/data/global_std.npy:

         x_physical = x_normalised × σ + μ

     This converts both predictions and targets back into their original
     physical units:
       • Global_active_power    → kilowatts  (kW)
       • Global_reactive_power  → kilowatts  (kW)
       • Voltage                → volts      (V)
       • Global_intensity       → amperes    (A)
       • Sub_metering_1/2/3     → watt-hours (Wh)

  5. Compute per-channel global metrics over the entire validation set:
       • MAE  = mean |ŷ − y|
       • RMSE = √mean (ŷ − y)²

  6. Print a formatted summary table and export
     docs/evaluation_report.json.

Checkpoint priority
-------------------
  epoch_01.pt  (full training run)  →  smoke_test.pt  →  any *.pt file
  Override with --checkpoint <path>.

Usage
-----
    python src/training/evaluate.py                          # auto-detect
    python src/training/evaluate.py --checkpoint checkpoints/epoch_01.pt
    python src/training/evaluate.py --batch-size 64 --num-workers 0
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import pathlib
import sys
import time
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# ── Repo root bootstrap ───────────────────────────────────────────────────────
_REPO = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO))

from src.data.dataset import (
    GridSequenceDataset,
    FEATURE_COLS,
    N_CHANNELS,
    T_FORECAST,
    T_LOOKBACK,
)
from src.training.train import LinearGridForecaster   # reuses the model class

# ── Output path ───────────────────────────────────────────────────────────────
_REPORT_PATH = _REPO / "docs" / "evaluation_report.json"

# ── Normalisation artifact paths ──────────────────────────────────────────────
_MEAN_PATH = _REPO / "src" / "data" / "global_mean.npy"
_STD_PATH  = _REPO / "src" / "data" / "global_std.npy"

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("evaluate")

SEP = "═" * 72
sep = "─" * 72

# ── Physical unit labels per channel (matches FEATURE_COLS order) ────────────
CHANNEL_UNITS: list[str] = ["kW", "kW", "V", "A", "Wh", "Wh", "Wh"]


# ═══════════════════════════════════════════════════════════════════════════════
#  CLI
# ═══════════════════════════════════════════════════════════════════════════════

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Evaluation pipeline for LinearGridForecaster.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to a .pt checkpoint file. Auto-detected if omitted.",
    )
    p.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="DataLoader batch size for the validation loop.",
    )
    p.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="DataLoader worker processes (0 = main process only, safe on Mac).",
    )
    p.add_argument(
        "--device",
        type=str,
        default=None,
        help="Force device (e.g. 'cpu', 'cuda', 'mps'). Auto-detected if omitted.",
    )
    p.add_argument(
        "--max-batches",
        type=int,
        default=None,
        metavar="N",
        help=(
            "Stop after N validation batches (default: None = full val set). "
            "Useful for a fast representative evaluation on CPU; metrics are "
            "still computed over all samples seen within those N batches."
        ),
    )
    return p.parse_args()


# ═══════════════════════════════════════════════════════════════════════════════
#  Checkpoint discovery
# ═══════════════════════════════════════════════════════════════════════════════

def _find_checkpoint(override: Optional[str] = None) -> pathlib.Path:
    """
    Resolve a checkpoint path using the following priority:

      1. CLI --checkpoint argument (if provided)
      2. checkpoints/epoch_01.pt   (full training run)
      3. checkpoints/smoke_test.pt (quick verification run)
      4. Any *.pt file found under checkpoints/

    Raises FileNotFoundError if nothing is found.
    """
    if override:
        p = pathlib.Path(override)
        if not p.exists():
            raise FileNotFoundError(f"Specified checkpoint not found: {p}")
        return p

    ckpt_dir = _REPO / "checkpoints"

    for candidate in ["epoch_01.pt", "smoke_test.pt"]:
        p = ckpt_dir / candidate
        if p.exists():
            return p

    # Fallback: any .pt file
    all_pts = sorted(ckpt_dir.glob("*.pt"))
    if all_pts:
        return all_pts[0]

    raise FileNotFoundError(
        f"No checkpoint found in {ckpt_dir}. "
        "Run src/training/train.py first."
    )


# ═══════════════════════════════════════════════════════════════════════════════
#  Model loading
# ═══════════════════════════════════════════════════════════════════════════════

def _load_model(
    ckpt_path: pathlib.Path,
    device: torch.device,
) -> tuple[nn.Module, dict]:
    """
    Load LinearGridForecaster from a checkpoint.

    Reads the 'cfg' key saved by train.py to reconstruct the exact
    architecture that was used during training.

    Returns
    -------
    (model, checkpoint_dict)
    """
    log.info("Loading checkpoint: %s", ckpt_path)
    ckpt = torch.load(str(ckpt_path), map_location=device, weights_only=False)

    # Extract training config saved inside the checkpoint
    saved_cfg: dict = ckpt.get("cfg", {})

    model = LinearGridForecaster(
        in_channels=N_CHANNELS,
        d_model=saved_cfg.get("d_model",    64),
        num_heads=saved_cfg.get("num_heads",  4),
        ffn_dim=saved_cfg.get("ffn_dim",   256),
        t_forecast=T_FORECAST,
        dropout=0.0,   # always 0 during eval — no stochastic dropout
    ).to(device)

    model.load_state_dict(ckpt["model_state"])
    model.eval()   # disables BatchNorm, Dropout, etc.

    total_params = model.param_count()
    log.info(
        "Model loaded — epoch=%s  train_avg_loss=%.6f  params=%s",
        ckpt.get("epoch", "?"),
        ckpt.get("avg_loss", float("nan")),
        f"{total_params:,}",
    )
    return model, ckpt


# ═══════════════════════════════════════════════════════════════════════════════
#  Normalisation stats loader
# ═══════════════════════════════════════════════════════════════════════════════

def _load_norm_stats() -> tuple[torch.Tensor, torch.Tensor]:
    """
    Load per-channel μ and σ from the .npy files produced by compute_stats.py.

    Returns
    -------
    (mean, std)  — float32 Tensors of shape (N_CHANNELS,) = (7,)
    """
    for path, name in [(_MEAN_PATH, "global_mean.npy"), (_STD_PATH, "global_std.npy")]:
        if not path.exists():
            raise FileNotFoundError(
                f"{name} not found at {path}.\n"
                "Run src/utils/compute_stats.py first."
            )

    mean = torch.from_numpy(np.load(str(_MEAN_PATH)).astype(np.float32))  # (7,)
    std  = torch.from_numpy(np.load(str(_STD_PATH )).astype(np.float32))  # (7,)

    assert mean.shape == (N_CHANNELS,), f"mean shape mismatch: {mean.shape}"
    assert std.shape  == (N_CHANNELS,), f"std  shape mismatch: {std.shape}"

    log.info("Norm stats loaded — μ: %s", mean.tolist())
    log.info("                   σ: %s", std.tolist())
    return mean, std


# ═══════════════════════════════════════════════════════════════════════════════
#  Denormalisation
# ═══════════════════════════════════════════════════════════════════════════════

def _denormalise(
    z: torch.Tensor,
    mean: torch.Tensor,
    std: torch.Tensor,
) -> torch.Tensor:
    """
    Reverse Z-score normalisation: x_physical = z × σ + μ

    Parameters
    ----------
    z    : Tensor [..., N_CHANNELS]   — normalised values
    mean : Tensor [N_CHANNELS]        — per-channel training mean
    std  : Tensor [N_CHANNELS]        — per-channel training std

    Returns
    -------
    Tensor [..., N_CHANNELS] in original physical units.
    """
    # Both mean and std broadcast over leading batch/time dimensions.
    return z * std + mean


# ═══════════════════════════════════════════════════════════════════════════════
#  Evaluation loop
# ═══════════════════════════════════════════════════════════════════════════════

def run_evaluation(
    model:      nn.Module,
    loader:     DataLoader,
    mean:       torch.Tensor,
    std:        torch.Tensor,
    device:     torch.device,
    max_batches: Optional[int] = None,
) -> dict:
    """
    Full validation loop — no gradient computation at any point.

    For each batch:
      • Forward pass model on normalised input x
      • Collect predicted output and ground-truth target y
      • Denormalise both to physical units

    Accumulates per-channel squared-errors and absolute-errors over all
    validation samples × forecast steps to compute global MAE and RMSE.

    Alongside the model's own error, this also accumulates the error of a
    trivial "always predict the training-set per-channel mean" baseline on
    the *exact same* validation samples — computing this is nearly free once
    we're already iterating the loader, and it answers the question a
    reviewer asks first: is the model actually better than guessing the
    mean, per channel, or only in aggregate?

    Returns
    -------
    dict with keys:
        'n_samples'        : int      — total validation samples processed
        'n_forecast'       : int      — T_FORECAST horizon (60)
        'n_channels'       : int      — number of feature channels (7)
        'mae_per_ch'       : list[float]  — per-channel model MAE, physical units
        'rmse_per_ch'      : list[float]  — per-channel model RMSE, physical units
        'mae_global'       : float        — mean model MAE across all channels
        'rmse_global'      : float        — mean model RMSE across all channels
        'mae_per_ch_naive' : list[float]  — per-channel naive-mean-baseline MAE
        'rmse_per_ch_naive': list[float]  — per-channel naive-mean-baseline RMSE
        'mae_global_naive' : float        — mean naive-baseline MAE
        'rmse_global_naive': float        — mean naive-baseline RMSE
    """
    model.eval()

    # Move normalisation tensors to the evaluation device once
    mean = mean.to(device)
    std  = std.to(device)

    # Accumulators: sum of squared errors and absolute errors per channel
    # Shape (N_CHANNELS,) — accumulated over all samples × forecast steps
    sum_sq_err  = torch.zeros(N_CHANNELS, device=device)
    sum_abs_err = torch.zeros(N_CHANNELS, device=device)

    # Same accumulators for the naive "predict the training mean" baseline
    sum_sq_err_naive  = torch.zeros(N_CHANNELS, device=device)
    sum_abs_err_naive = torch.zeros(N_CHANNELS, device=device)

    total_elements = 0   # scalar count of (samples × T_FORECAST) elements

    n_batches  = len(loader)
    cap        = min(max_batches, n_batches) if max_batches else n_batches
    t_start    = time.perf_counter()

    if max_batches and max_batches < n_batches:
        log.warning(
            "--max-batches=%d  (full val set = %d batches). "
            "Metrics are computed over the first %d batches only.",
            max_batches, n_batches, cap,
        )
    log.info("Starting evaluation over %d / %d validation batches …", cap, n_batches)

    with torch.no_grad():
        for batch_idx, (x, y) in enumerate(loader):
            if max_batches and batch_idx >= max_batches:
                break
            # x : [B, T_LOOKBACK, 7]  — normalised input history
            # y : [B, T_FORECAST,  7]  — normalised ground-truth targets
            x = x.to(device)
            y = y.to(device)

            # ── Forward pass (no gradient) ────────────────────────────────────
            pred: torch.Tensor = model(x)   # [B, T_FORECAST, 7] — normalised

            # ── Denormalise both prediction and target ────────────────────────
            # Inverse of:  z = (raw - mean) / std
            # Therefore:   raw = z * std + mean
            pred_phys = _denormalise(pred, mean, std)   # [B, T_FORECAST, 7]
            y_phys    = _denormalise(y,    mean, std)   # [B, T_FORECAST, 7]

            # ── Per-element residuals ─────────────────────────────────────────
            # Shapes are [B, T_FORECAST, N_CHANNELS]
            residual = pred_phys - y_phys               # [B, T_F, C]

            # Reduce over batch and time dimensions (dim 0 and 1),
            # keep the channel axis so we get per-channel accumulators.
            sum_sq_err  += (residual ** 2).sum(dim=(0, 1))   # (C,)
            sum_abs_err += residual.abs().sum(dim=(0, 1))     # (C,)

            # ── Naive baseline: predict the training-set mean for every step ──
            # `mean` has shape (C,) and broadcasts against y_phys (B, T_F, C).
            residual_naive = mean - y_phys                    # [B, T_F, C]
            sum_sq_err_naive  += (residual_naive ** 2).sum(dim=(0, 1))  # (C,)
            sum_abs_err_naive += residual_naive.abs().sum(dim=(0, 1))    # (C,)

            # Number of (sample, timestep) pairs this batch contributed
            B = x.size(0)
            total_elements += B * T_FORECAST

            # Progress log every 50 batches
            if (batch_idx + 1) % 50 == 0 or batch_idx == 0:
                elapsed = time.perf_counter() - t_start
                log.info(
                    "  batch %4d / %d   samples=%d   elapsed=%.1f s",
                    batch_idx + 1, cap, total_elements, elapsed,
                )

    elapsed = time.perf_counter() - t_start
    log.info(
        "Evaluation complete — %d samples (%d/%d batches) in %.1f s",
        total_elements, min(batch_idx + 1, cap), n_batches, elapsed,
    )

    # ── Global metric computation ─────────────────────────────────────────────
    # Divide accumulated sums by total element count to get per-channel means
    n = float(total_elements)

    mae_per_ch  = (sum_abs_err / n).cpu().tolist()           # mean |err| per ch
    rmse_per_ch = ((sum_sq_err / n).sqrt()).cpu().tolist()   # √mean(err²) per ch

    mae_global  = float(np.mean(mae_per_ch))
    rmse_global = float(np.mean(rmse_per_ch))

    mae_per_ch_naive  = (sum_abs_err_naive / n).cpu().tolist()
    rmse_per_ch_naive = ((sum_sq_err_naive / n).sqrt()).cpu().tolist()

    mae_global_naive  = float(np.mean(mae_per_ch_naive))
    rmse_global_naive = float(np.mean(rmse_per_ch_naive))

    batches_run = min(batch_idx + 1, cap) if total_elements > 0 else 0
    return {
        "n_samples":       total_elements,
        "n_forecast":      T_FORECAST,
        "n_channels":      N_CHANNELS,
        "mae_per_ch":      mae_per_ch,
        "rmse_per_ch":     rmse_per_ch,
        "mae_global":      mae_global,
        "rmse_global":     rmse_global,
        "mae_per_ch_naive":  mae_per_ch_naive,
        "rmse_per_ch_naive": rmse_per_ch_naive,
        "mae_global_naive":  mae_global_naive,
        "rmse_global_naive": rmse_global_naive,
        "batches_evaluated": batches_run,
        "batches_total":     n_batches,
        "partial_eval":      batches_run < n_batches,
        "max_batches_cap":   max_batches,
    }


# ═══════════════════════════════════════════════════════════════════════════════
#  Report printing
# ═══════════════════════════════════════════════════════════════════════════════

def _print_report(
    metrics: dict,
    ckpt_meta: dict,
    mean: torch.Tensor,
    std: torch.Tensor,
) -> None:
    """Print a formatted per-channel metrics table to stdout."""
    print()
    print(SEP)
    print("  LinearGridForecaster — Validation Evaluation Report")
    print(SEP)
    print(f"  Checkpoint  : epoch={ckpt_meta.get('epoch', '?')}  "
          f"train_avg_loss={ckpt_meta.get('avg_loss', float('nan')):.6f}")
    print(f"  Val samples : {metrics['n_samples']:,}  "
          f"(T_lookback={T_LOOKBACK}, T_forecast={T_FORECAST})")
    print()

    # ── Per-channel table ─────────────────────────────────────────────────────
    # Includes the naive "always predict the training mean" baseline next to
    # the model's own numbers — beating this per channel is the minimum bar
    # for "the model learned something," not just beating it in aggregate.
    col_w = [28, 8, 12, 12, 14, 8]
    header = (
        f"  {'Channel':<{col_w[0]}} "
        f"{'Unit':<{col_w[1]}} "
        f"{'MAE':>{col_w[2]}} "
        f"{'RMSE':>{col_w[3]}} "
        f"{'Naive MAE':>{col_w[4]}} "
        f"{'Beats?':>{col_w[5]}}"
    )
    print(header)
    print("  " + sep[2:])

    mean_list = mean.tolist()
    std_list  = std.tolist()

    for ch_idx, (col, unit) in enumerate(zip(FEATURE_COLS, CHANNEL_UNITS)):
        mae        = metrics["mae_per_ch"][ch_idx]
        rmse       = metrics["rmse_per_ch"][ch_idx]
        mae_naive  = metrics["mae_per_ch_naive"][ch_idx]
        beats      = "yes" if mae < mae_naive else "NO"
        mu   = mean_list[ch_idx]
        sigma = std_list[ch_idx]
        print(
            f"  {col:<{col_w[0]}} "
            f"{unit:<{col_w[1]}} "
            f"{mae:>{col_w[2]}.4f} "
            f"{rmse:>{col_w[3]}.4f} "
            f"{mae_naive:>{col_w[4]}.4f} "
            f"{beats:>{col_w[5]}}"
            f"   (μ={mu:.4f}, σ={sigma:.4f})"
        )

    print("  " + sep[2:])
    print(
        f"  {'GLOBAL MEAN':<{col_w[0]}} "
        f"{'':>{col_w[1]}} "
        f"{metrics['mae_global']:>{col_w[2]}.4f} "
        f"{metrics['rmse_global']:>{col_w[3]}.4f} "
        f"{metrics['mae_global_naive']:>{col_w[4]}.4f} "
        f"{'yes' if metrics['mae_global'] < metrics['mae_global_naive'] else 'NO':>{col_w[5]}}"
    )
    print()
    print(f"  Report saved → {_REPORT_PATH}")
    print(SEP)
    print()


# ═══════════════════════════════════════════════════════════════════════════════
#  JSON export
# ═══════════════════════════════════════════════════════════════════════════════

def _write_json(
    metrics: dict,
    ckpt_path: pathlib.Path,
    ckpt_meta: dict,
    mean: torch.Tensor,
    std: torch.Tensor,
    device: torch.device,
) -> None:
    """Serialise all evaluation results to docs/evaluation_report.json."""
    _REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)

    mean_list = mean.tolist()
    std_list  = std.tolist()

    # Build structured per-channel breakdown
    channels = []
    for ch_idx, (col, unit) in enumerate(zip(FEATURE_COLS, CHANNEL_UNITS)):
        model_mae  = metrics["mae_per_ch"][ch_idx]
        naive_mae  = metrics["mae_per_ch_naive"][ch_idx]
        channels.append({
            "index":       ch_idx,
            "name":        col,
            "unit":        unit,
            "train_mean":  round(mean_list[ch_idx], 6),
            "train_std":   round(std_list[ch_idx],  6),
            "mae":         round(model_mae,  6),
            "rmse":        round(metrics["rmse_per_ch"][ch_idx], 6),
            "naive_baseline_mae":  round(naive_mae, 6),
            "naive_baseline_rmse": round(metrics["rmse_per_ch_naive"][ch_idx], 6),
            "beats_naive_baseline": bool(model_mae < naive_mae),
        })

    report = {
        "meta": {
            "checkpoint":        str(ckpt_path),
            "epoch":             ckpt_meta.get("epoch"),
            "train_avg_loss":    ckpt_meta.get("avg_loss"),
            "train_cfg":         ckpt_meta.get("cfg", {}),
            "device":            str(device),
            "max_batches_cap":   metrics.get("max_batches_cap"),
        },
        "dataset": {
            "split":             "val",
            "split_fraction":    0.20,
            "n_samples":         metrics["n_samples"],
            "batches_evaluated": metrics.get("batches_evaluated"),
            "batches_total":     metrics.get("batches_total"),
            "partial_eval":      metrics.get("partial_eval", False),
            "t_lookback":        T_LOOKBACK,
            "t_forecast":        T_FORECAST,
            "n_channels":        N_CHANNELS,
        },
        "normalisation": {
            "method":   "z-score: z = (x - mean) / std",
            "inverse":  "x = z * std + mean",
            "mean_per_channel": mean_list,
            "std_per_channel":  std_list,
        },
        "metrics": {
            "description":   "All metrics computed on denormalised (physical-unit) outputs",
            "global_mae":    round(metrics["mae_global"],  6),
            "global_rmse":   round(metrics["rmse_global"], 6),
            "naive_baseline": {
                "description": (
                    "Trivial baseline: predict the training-set per-channel "
                    "mean for every forecast step, evaluated on the exact "
                    "same validation samples as the model above."
                ),
                "global_mae":  round(metrics["mae_global_naive"],  6),
                "global_rmse": round(metrics["rmse_global_naive"], 6),
            },
            "beats_naive_baseline_global": bool(
                metrics["mae_global"] < metrics["mae_global_naive"]
            ),
            "per_channel":   channels,
        },
    }

    with _REPORT_PATH.open("w") as f:
        json.dump(report, f, indent=2)

    log.info("JSON report written → %s", _REPORT_PATH)


# ═══════════════════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    args = _parse_args()

    print(SEP)
    print("  LinearGridForecaster — Production Evaluation Pipeline")
    print(SEP)

    # ── Device ────────────────────────────────────────────────────────────────
    if args.device:
        device = torch.device(args.device)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    log.info("Device: %s", device)

    # ── 1. Checkpoint ─────────────────────────────────────────────────────────
    ckpt_path = _find_checkpoint(args.checkpoint)
    model, ckpt_meta = _load_model(ckpt_path, device)

    # ── 2. Normalisation statistics ───────────────────────────────────────────
    mean, std = _load_norm_stats()

    # ── 3. Validation dataset (20 % split, exclusive) ─────────────────────────
    log.info("Initialising validation dataset (split='val') …")
    val_dataset = GridSequenceDataset(split="val")
    log.info("Validation dataset: %s", val_dataset)

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,       # deterministic ordering — required for evaluation
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=False,     # evaluate every sample, including partial last batch
    )
    log.info(
        "DataLoader: %d batches × batch_size=%d  (drop_last=False)",
        len(val_loader), args.batch_size,
    )

    # ── 4 & 5. Evaluation loop → denormalise → compute metrics ───────────────
    metrics = run_evaluation(
        model=model,
        loader=val_loader,
        mean=mean,
        std=std,
        device=device,
        max_batches=args.max_batches,
    )

    # ── 6. Report & export ────────────────────────────────────────────────────
    _print_report(metrics, ckpt_meta, mean, std)
    _write_json(metrics, ckpt_path, ckpt_meta, mean, std, device)


if __name__ == "__main__":
    main()
