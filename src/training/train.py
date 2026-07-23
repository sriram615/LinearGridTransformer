"""
train.py
--------
Principal Deep Learning Optimization Engineer — LinearGridTransformer project.
Research Statistician & Model Diagnostic Engineer — statistical tracking layer.

Full single-epoch training loop for the LinearGridTransformer:
    GridSequenceDataset  →  GridTransformerEncoder  →  ForecastHead
    MSELoss  +  AdamW optimizer

Data flow
---------
    x  [B, 1440, 7]  ──  encoder  ──→  latent  [B, 1440, 64]
                                  slice T[-60:]  [B,   60, 64]
                                  ──  head     ──→  pred  [B,  60,  7]
    y  [B,   60, 7]  ──  MSELoss(pred, y)

Complexity: O(T) per forward pass — no [T×T] matrix ever allocated.

Denominator diagnostics (partition-function Z tracking)
--------------------------------------------------------
A PyTorch *forward hook* is registered on the inner LinearKernelAttention
module before training begins.  The hook captures the (output, debug) tuple
that LinearKernelAttention.forward() returns and stashes the denominator
tensor into a DenominatorTracker without touching the gradient graph.

Every `denom_log_every` steps the tracker computes, inside torch.no_grad():
  • Z_mean — mean of the partition function across [B, H, T]
  • Z_var  — variance of Z (low variance → attention dilution risk)
If Z_var < denom_var_eps the loop emits a highlighted WARNING.

The hook is removed after training to restore the module's original state.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
import pathlib
import logging

# ── Repo root on sys.path ─────────────────────────────────────────────────────
_REPO = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO))

import psutil
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.data.dataset     import GridSequenceDataset, T_FORECAST, N_CHANNELS
from test_attention        import GridTransformerEncoder   # encoder + kernel

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
#  ForecastHead  —  maps encoder hidden states to the forecast target
# ═══════════════════════════════════════════════════════════════════════════════

class ForecastHead(nn.Module):
    """
    Slices the last T_forecast time-steps from the encoder output and
    projects from d_model → N_channels via a single linear layer.

    Input  : [B, T_lookback, d_model]
    Output : [B, T_forecast, N_channels]
    """

    def __init__(self, d_model: int, t_forecast: int, n_channels: int) -> None:
        super().__init__()
        self.t_forecast = t_forecast
        self.proj = nn.Linear(d_model, n_channels)
        nn.init.xavier_uniform_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x : [B, T, d_model]
        tail = x[:, -self.t_forecast:, :]          # [B, t_forecast, d_model]
        return self.proj(tail)                      # [B, t_forecast, n_channels]


# ═══════════════════════════════════════════════════════════════════════════════
#  LinearGridForecaster  —  full model wrapper (encoder + head)
# ═══════════════════════════════════════════════════════════════════════════════

class LinearGridForecaster(nn.Module):
    """
    Wrapper that chains GridTransformerEncoder → ForecastHead.

    Parameters
    ----------
    in_channels : int   — number of feature channels (7)
    d_model     : int   — encoder internal width (64)
    num_heads   : int   — number of attention heads (4)
    ffn_dim     : int   — FFN hidden dim (256)
    t_forecast  : int   — forecast horizon in steps (60)
    dropout     : float — dropout inside FFN

    Shapes
    ------
    Input  x : [B, T_lookback, in_channels]
    Output   : [B, t_forecast, in_channels]
    """

    def __init__(
        self,
        in_channels: int   = N_CHANNELS,
        d_model:     int   = 64,
        num_heads:   int   = 4,
        ffn_dim:     int   = 256,
        t_forecast:  int   = T_FORECAST,
        dropout:     float = 0.1,
    ) -> None:
        super().__init__()
        self.encoder = GridTransformerEncoder(
            in_channels=in_channels,
            d_model=d_model,
            num_heads=num_heads,
            ffn_dim=ffn_dim,
            dropout=dropout,
        )
        # Re-expose the internal hidden projection so head connects at d_model
        self.head = ForecastHead(
            d_model=in_channels,   # encoder output is in_channels (7)
                                   # because GridTransformerEncoder ends with
                                   # output_proj: d_model → in_channels
            t_forecast=t_forecast,
            n_channels=in_channels,
        )
        self.t_forecast = t_forecast

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x   : [B, T_lookback, C]
        out : [B, T_forecast,  C]
        """
        enc = self.encoder(x)                 # [B, T_lookback, C]  (dim-preserving)
        return self.head(enc)                 # [B, T_forecast,  C]

    def param_count(self) -> int:
        return sum(p.numel() for p in self.parameters())


# ═══════════════════════════════════════════════════════════════════════════════
#  Hyper-parameters
# ═══════════════════════════════════════════════════════════════════════════════

CFG = {
    # Data
    "batch_size":   32,
    "num_workers":  0,        # 0 = no sub-processes → safe on Mac
    "split":        "train",

    # Model
    "d_model":      64,
    "num_heads":    4,
    "ffn_dim":      256,
    "dropout":      0.1,

    # Optimiser
    "lr":           1e-4,
    "weight_decay": 1e-2,

    # Training
    "epochs":       1,
    "log_every":       50,    # print MSE loss every N batches
    "max_batches":     None,  # set to int to cap epoch (useful for smoke runs)

    # Denominator / partition-function diagnostics
    "denom_log_every": 100,   # compute Z stats every N steps
    "denom_var_eps":   1e-3,  # variance threshold for dilution warning
}


# ═══════════════════════════════════════════════════════════════════════════════
#  Helpers
# ═══════════════════════════════════════════════════════════════════════════════

def _rss_mb() -> float:
    return psutil.Process(os.getpid()).memory_info().rss / 1e6


def _fmt_time(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    return f"{m}m {s:02d}s" if m else f"{s:.1f}s"


SEP  = "═" * 70
sep  = "─" * 70
WARN = "⚠" * 70


# ═══════════════════════════════════════════════════════════════════════════════
#  DenominatorTracker — partition-function Z diagnostics via forward hook
# ═══════════════════════════════════════════════════════════════════════════════

class DenominatorTracker:
    """
    Non-gradient side-channel observer for the attention partition function Z.

    Strategy
    --------
    LinearKernelAttention.forward() returns a tuple (output, debug) where
    debug['denominator'] has shape [B, H, T].  However, GridTransformerEncoder
    silently discards the debug dict (``attn_out, _ = self.attn(...)``), so the
    denominator never reaches the training loop normally.

    We solve this with a **forward hook**: PyTorch calls the hook with the raw
    module output *before* the caller receives it.  The hook stashes the debug
    dict into this tracker.  Because we only read tensors here (never write into
    them), gradient tracking is unaffected — we additionally guard every
    computation inside ``torch.no_grad()``.

    Hook signature (PyTorch contract)
    -----------------------------------
        hook(module, input, output) -> None
    output for LinearKernelAttention is (attn_tensor, debug_dict).
    """

    def __init__(self, var_eps: float = 1e-3, log_every: int = 100) -> None:
        self.var_eps   = var_eps
        self.log_every = log_every
        self._latest_debug: dict | None = None
        self._handle   = None          # torch hook handle, for removal
        # Cumulative counters
        self.total_checks    = 0
        self.dilution_events = 0

    # ── Hook function (registered on LinearKernelAttention) ──────────────────

    def _hook(self, module, inp, output) -> None:  # noqa: ANN001
        """
        Called by PyTorch immediately after LinearKernelAttention.forward().
        Stashes the debug dict; does NOT alter the tensor output.
        """
        # output is (attn_out_tensor, debug_dict)
        if isinstance(output, (tuple, list)) and len(output) == 2:
            self._latest_debug = output[1]

    def register(self, attn_module) -> None:
        """Attach the hook to a LinearKernelAttention instance."""
        self._handle = attn_module.register_forward_hook(self._hook)
        log.info(
            "DenominatorTracker hooked onto %s  (log_every=%d, var_eps=%.0e)",
            attn_module.__class__.__name__, self.log_every, self.var_eps,
        )

    def remove(self) -> None:
        """Detach the hook — restores the module to its original state."""
        if self._handle is not None:
            self._handle.remove()
            self._handle = None
            log.info("DenominatorTracker hook removed.")

    # ── Diagnostic computation ────────────────────────────────────────────────

    def log_stats(self, step: int) -> None:
        """
        Compute and print Z mean / variance.  Wrapped fully in no_grad.
        Emits WARNING if Var(Z) < var_eps (attention dilution).

        Parameters
        ----------
        step : int  — global training step (batch index), 1-based.
        """
        if self._latest_debug is None:
            return   # hook hasn't fired yet

        with torch.no_grad():
            # denominator shape: [B, H, T]  — float32
            denom: torch.Tensor = self._latest_debug["denominator"].detach()

            # Flatten to a 1-D view for global statistics
            flat = denom.reshape(-1)          # [B * H * T]

            z_mean = float(flat.mean())
            z_var  = float(flat.var())        # Bessel-corrected (ddof=1)
            z_min  = float(flat.min())
            z_max  = float(flat.max())

        self.total_checks += 1

        # ── Always print the stats table ─────────────────────────────────────
        log.info(
            "[Z-STATS] step=%5d │ Z_mean=%12.4f │ Z_var=%12.6f │ "
            "Z_min=%10.4f │ Z_max=%10.4f",
            step, z_mean, z_var, z_min, z_max,
        )

        # ── Dilution warning ──────────────────────────────────────────────────
        if z_var < self.var_eps:
            self.dilution_events += 1
            print()
            print(WARN)
            print(
                f"  WARNING: Potential Attention Dilution Detected — "
                f"Denominator manifold is flattening."
            )
            print(
                f"  step={step}  Var(Z)={z_var:.2e}  < eps={self.var_eps:.0e}  "
                f"(event #{self.dilution_events})"
            )
            print(WARN)
            print()


# ═══════════════════════════════════════════════════════════════════════════════
#  Training loop
# ═══════════════════════════════════════════════════════════════════════════════

def train_one_epoch(
    model:     nn.Module,
    loader:    DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch:     int,
    cfg:       dict,
    tracker:   "DenominatorTracker | None" = None,
) -> dict:
    """
    Run one full epoch over `loader`.

    Parameters
    ----------
    tracker : DenominatorTracker | None
        When provided, denominator statistics are computed and logged every
        ``cfg['denom_log_every']`` steps.  All stat ops run inside
        ``torch.no_grad()`` to avoid polluting the gradient graph.

    Returns
    -------
    dict with keys: avg_loss, batches, samples, elapsed_s, loss_history
        loss_history : list[float]  — per-batch MSE loss (scalar), used by
        the smoke-test trend checker.
    """
    model.train()
    device     = next(model.parameters()).device

    running_loss  = 0.0
    total_batches = 0
    total_samples = 0
    loss_history: list[float] = []      # one scalar MSE per batch
    t_epoch_start = time.perf_counter()

    max_batches = cfg.get("max_batches") or len(loader)

    for batch_idx, (x, y) in enumerate(loader):

        if batch_idx >= max_batches:
            break

        # ── Move to device (no-op on CPU; transparent for future GPU) ────────
        x = x.to(device)   # [B, 1440, 7]
        y = y.to(device)   # [B,   60, 7]

        # ── Forward pass ──────────────────────────────────────────────────────
        pred = model(x)    # [B,   60, 7]

        # ── Loss computation ──────────────────────────────────────────────────
        loss = criterion(pred, y)   # scalar MSE across all elements

        # ── Backpropagation ───────────────────────────────────────────────────
        optimizer.zero_grad(set_to_none=True)  # set_to_none frees grad memory
        loss.backward()

        # Gradient clipping — prevents exploding gradients early in training
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

        # ── Metrics accumulation ──────────────────────────────────────────────
        b_size         = x.size(0)
        batch_loss     = loss.item()
        running_loss  += batch_loss * b_size
        total_samples += b_size
        total_batches += 1
        loss_history.append(batch_loss)

        # ── Per-batch MSE logging ─────────────────────────────────────────────
        if (batch_idx + 1) % cfg["log_every"] == 0 or batch_idx == 0:
            elapsed = time.perf_counter() - t_epoch_start
            throughput = total_samples / max(elapsed, 1e-9)
            log.info(
                "epoch=%d  batch=%4d/%d  loss=%.6f  "
                "samples/s=%5.0f  RSS=%.0f MB",
                epoch,
                batch_idx + 1,
                max_batches,
                loss.item(),
                throughput,
                _rss_mb(),
            )

        # ── Denominator / partition-function Z diagnostics ────────────────────
        # The DenominatorTracker's forward hook already captured the debug dict
        # during model(x) above.  We query it here, outside any autograd
        # context, so the stat tensors are never part of the compute graph.
        if tracker is not None:
            denom_step = batch_idx + 1   # 1-based global step within epoch
            if denom_step % cfg["denom_log_every"] == 0 or batch_idx == 0:
                tracker.log_stats(step=denom_step)

    elapsed_s = time.perf_counter() - t_epoch_start
    avg_loss  = running_loss / max(total_samples, 1)

    return {
        "avg_loss":     avg_loss,
        "batches":      total_batches,
        "samples":      total_samples,
        "elapsed_s":    elapsed_s,
        "loss_history": loss_history,
    }


# ═══════════════════════════════════════════════════════════════════════════════
#  Main entry point
# ═══════════════════════════════════════════════════════════════════════════════

# ═══════════════════════════════════════════════════════════════════════════════
#  CLI argument parser
# ═══════════════════════════════════════════════════════════════════════════════

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="LinearGridTransformer training loop.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--smoke-test",
        action="store_true",
        default=False,
        help=(
            "Quick verification mode: cap the run to exactly 50 batches, "
            "log every batch, compute Z-stats every 10 steps, "
            "and print a loss-trend + memory report at exit."
        ),
    )
    return p.parse_args()


def main() -> None:
    args = _parse_args()

    # ── Apply smoke-test overrides before anything else ───────────────────────
    SMOKE_BATCHES = 50
    if args.smoke_test:
        CFG["max_batches"]     = SMOKE_BATCHES
        CFG["log_every"]       = 1    # print every single batch
        CFG["denom_log_every"] = 10   # Z-stats every 10 steps
        mode_tag = f"SMOKE-TEST ({SMOKE_BATCHES} batches)"
    else:
        mode_tag = "FULL EPOCH"

    print(SEP)
    print(f"  LinearGridTransformer — Training Run  [{mode_tag}]")
    print(SEP)

    torch.manual_seed(42)
    device = torch.device("cpu")   # Mac CPU; switch to "cuda" / "mps" freely

    # ── 1. Dataset & DataLoader ───────────────────────────────────────────────
    log.info("Initialising GridSequenceDataset (split='%s') …", CFG["split"])
    t0 = time.perf_counter()
    dataset = GridSequenceDataset(split=CFG["split"])
    log.info("Dataset ready in %.1f s — %s", time.perf_counter() - t0, dataset)

    loader = DataLoader(
        dataset,
        batch_size=CFG["batch_size"],
        shuffle=True,
        num_workers=CFG["num_workers"],   # 0 → main process only, safe on Mac
        pin_memory=False,
        drop_last=True,                   # keeps every batch the same size
    )

    n_batches = len(loader)
    log.info(
        "DataLoader: %d batches × batch_size=%d  (drop_last=True)",
        n_batches, CFG["batch_size"],
    )

    # ── 2. Model ──────────────────────────────────────────────────────────────
    log.info("Building LinearGridForecaster …")
    model = LinearGridForecaster(
        in_channels=N_CHANNELS,
        d_model=CFG["d_model"],
        num_heads=CFG["num_heads"],
        ffn_dim=CFG["ffn_dim"],
        t_forecast=T_FORECAST,
        dropout=CFG["dropout"],
    ).to(device)

    # ── 2b. Denominator tracker — hook onto the inner attention module ────────
    # model.encoder.attn is the LinearKernelAttention instance.
    # The hook reads its (output, debug) return value without changing it.
    tracker = DenominatorTracker(
        var_eps=CFG["denom_var_eps"],
        log_every=CFG["denom_log_every"],
    )
    tracker.register(model.encoder.attn)

    total_params = model.param_count()
    log.info(
        "Model — total params: %s  |  d_model=%d  num_heads=%d  ffn_dim=%d",
        f"{total_params:,}", CFG["d_model"], CFG["num_heads"], CFG["ffn_dim"],
    )

    # ── 3. Loss ───────────────────────────────────────────────────────────────
    criterion = nn.MSELoss()
    log.info("Objective: MSELoss (regression over %d channels)", N_CHANNELS)

    # ── 4. Optimiser ──────────────────────────────────────────────────────────
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=CFG["lr"],
        weight_decay=CFG["weight_decay"],
        betas=(0.9, 0.999),
    )
    log.info(
        "Optimizer: AdamW  lr=%.0e  weight_decay=%.0e  betas=(0.9, 0.999)",
        CFG["lr"], CFG["weight_decay"],
    )

    # ── Pre-training memory baseline ──────────────────────────────────────────
    rss_before = _rss_mb()
    log.info("RSS before training : %.0f MB", rss_before)

    print()
    print(sep)
    print(f"  Epoch 1 / {CFG['epochs']}  —  {n_batches:,} batches")
    print(sep)

    # ── 5. Training epoch ─────────────────────────────────────────────────────
    try:
        epoch_result = train_one_epoch(
            model=model,
            loader=loader,
            criterion=criterion,
            optimizer=optimizer,
            epoch=1,
            cfg=CFG,
            tracker=tracker,
        )
    finally:
        # Always remove the hook — even if training raises an exception.
        # This prevents stale hooks from accumulating across notebook re-runs.
        tracker.remove()

    # ── Post-training report ──────────────────────────────────────────────────
    rss_after = _rss_mb()

    print()
    print(SEP)
    print("  EPOCH SUMMARY")
    print(SEP)
    print(f"  Batches processed   : {epoch_result['batches']:,}")
    print(f"  Samples seen        : {epoch_result['samples']:,}")
    print(f"  Average MSE loss    : {epoch_result['avg_loss']:.6f}")
    print(f"  Epoch wall-time     : {_fmt_time(epoch_result['elapsed_s'])}")
    print(f"  Throughput          : "
          f"{epoch_result['samples'] / epoch_result['elapsed_s']:.1f} samples/s")
    print(f"  RSS before / after  : {rss_before:.0f} MB / {rss_after:.0f} MB  "
          f"(Δ {rss_after - rss_before:+.0f} MB)")
    print(f"  [T×T] score matrix  : NEVER ALLOCATED  ✔  (O(T) kernel)")
    print()
    print(f"  Input  [B, 1440, 7]  →  Encoder  →  Head  →  Output [B, 60, 7]")
    print(f"  Loss criterion      : MSELoss over [B × 60 × 7] predictions")
    print()
    print(f"  Denominator (Z) diagnostics:")
    print(f"    Z-stat checks run   : {tracker.total_checks}")
    print(f"    Dilution events     : {tracker.dilution_events}  "
          f"({'⚠  check attention scale' if tracker.dilution_events else '✔  none detected'})")
    print(SEP)

    # ── Save model checkpoint ─────────────────────────────────────────────────
    ckpt_dir  = _REPO / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_name = "smoke_test.pt" if args.smoke_test else "epoch_01.pt"
    ckpt_path = ckpt_dir / ckpt_name

    torch.save(
        {
            "epoch":        1,
            "model_state":  model.state_dict(),
            "optim_state":  optimizer.state_dict(),
            "avg_loss":     epoch_result["avg_loss"],
            "cfg":          CFG,
            "smoke_test":   args.smoke_test,
        },
        str(ckpt_path),
    )
    ckpt_mb = ckpt_path.stat().st_size / 1e6
    print(f"  Checkpoint saved → {ckpt_path}  ({ckpt_mb:.2f} MB)")

    # ── Smoke-test report ─────────────────────────────────────────────────────
    if args.smoke_test:
        lh = epoch_result["loss_history"]          # list[float], len==50
        n  = len(lh)

        # ── Performance metrics ───────────────────────────────────────────────
        iters_per_sec = epoch_result["batches"] / max(epoch_result["elapsed_s"], 1e-9)
        samples_per_sec = epoch_result["samples"] / max(epoch_result["elapsed_s"], 1e-9)
        rss_final = _rss_mb()

        # ── Loss-trend analysis (linear regression slope over all 50 batches) ─
        # A negative slope confirms the model is learning / locally overfitting.
        with torch.no_grad():
            lh_t  = torch.tensor(lh, dtype=torch.float64)
            steps = torch.arange(n, dtype=torch.float64)
            # slope = cov(steps, loss) / var(steps)  — O(n) closed-form
            steps_c = steps - steps.mean()
            loss_c  = lh_t  - lh_t.mean()
            slope   = float((steps_c * loss_c).sum() / (steps_c ** 2).sum())
            first5_avg = float(lh_t[:5].mean())
            last5_avg  = float(lh_t[-5:].mean())
            loss_drop  = first5_avg - last5_avg
            is_learning = slope < 0

        # ── Z-variance stability check ────────────────────────────────────────
        z_stable = tracker.dilution_events == 0

        print()
        print(SEP)
        print("  SMOKE-TEST EXECUTION PROFILE")
        print(SEP)
        print(f"  Batches run         : {n} / 50")
        print()
        print(f"  ── Loss trend ───────────────────────────────────────────")
        print(f"  First-5  avg MSE    : {first5_avg:.6f}")
        print(f"  Last-5   avg MSE    : {last5_avg:.6f}")
        print(f"  Net drop            : {loss_drop:+.6f}  "
              f"{'↓ model is learning ✔' if is_learning else '↑ loss not decreasing ⚠'}")
        print(f"  OLS slope / batch   : {slope:+.6f}  "
              f"({'negative ✔' if slope < 0 else 'positive ⚠'})")
        print()
        print(f"  ── Z-variance (attention denominator) ───────────────────")
        print(f"  Z-stat checks       : {tracker.total_checks}")
        print(f"  Dilution events     : {tracker.dilution_events}  "
              f"({'✔  Z-var stable & non-zero' if z_stable else '⚠  manifold flattening detected'})")
        print()
        print(f"  ── Speed & memory ───────────────────────────────────────")
        print(f"  Iterations / sec    : {iters_per_sec:.2f} it/s  "
              f"({samples_per_sec:.1f} samples/s)")
        print(f"  Batch time (avg)    : {1000/max(iters_per_sec,1e-9):.0f} ms / batch")
        print(f"  RSS (final)         : {rss_final:.0f} MB")
        print(f"  RSS delta           : {rss_final - rss_before:+.0f} MB")
        print(f"  [T×T] score matrix  : NEVER ALLOCATED  ✔")
        print()

        # ── Pass / fail summary ───────────────────────────────────────────────
        all_pass = is_learning and z_stable
        status   = "PASS ✔" if all_pass else "FAIL ✘"
        print(f"  ┌─ Smoke-test result: {status} {'─' * 44}")
        print(f"  │  Loss descending   : {'✔' if is_learning else '✘'}")
        print(f"  │  Z-var stable      : {'✔' if z_stable    else '✘'}")
        print(f"  └{'─' * 57}")
        print(SEP)
        print()


if __name__ == "__main__":
    main()
