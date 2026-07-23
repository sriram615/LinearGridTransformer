"""
test_pipeline.py
----------------
Core Deep Learning Engineer — LinearGridTransformer project.

Integration smoke-test for GridSequenceDataset + DataLoader.

Checks
------
  1. Dataset initialises without error.
  2. DataLoader yields exactly one batch of shape [32, 1440, 7] for X
     and [32, 60, 7] for Y.
  3. All values are finite (no NaN / Inf leaking through).
  4. Prints a concise diagnostic table and exits immediately to free RAM.

Usage
-----
    python3 test_pipeline.py
"""

from __future__ import annotations

import sys
import time

import torch
from torch.utils.data import DataLoader

# Add repo root to path so src.data.dataset is importable without install
import pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from src.data.dataset import GridSequenceDataset, T_LOOKBACK, T_FORECAST, N_CHANNELS

# ── Test configuration ────────────────────────────────────────────────────────
BATCH_SIZE      = 32
EXPECTED_X_SHAPE = torch.Size([BATCH_SIZE, T_LOOKBACK, N_CHANNELS])  # [32, 1440, 7]
EXPECTED_Y_SHAPE = torch.Size([BATCH_SIZE, T_FORECAST, N_CHANNELS])  # [32,   60, 7]

SEP = "─" * 62


def _pass(msg: str) -> None:
    print(f"  ✔  {msg}")


def _fail(msg: str) -> None:
    print(f"  ✘  {msg}", file=sys.stderr)
    sys.exit(1)


def main() -> None:
    print(SEP)
    print("  GridSequenceDataset — integration smoke-test")
    print(SEP)

    # ── 1. Dataset construction ───────────────────────────────────────────────
    t0 = time.perf_counter()
    print("\n[1] Initialising dataset (split='train') …")
    try:
        ds = GridSequenceDataset(split="train")
    except Exception as exc:
        _fail(f"Dataset init failed: {exc}")

    t_init = time.perf_counter() - t0
    _pass(f"Dataset created in {t_init:.2f} s")
    print(f"     {ds}")
    for k, v in ds.shape.items():
        print(f"       {k:<14}: {v}")

    # ── 2. Single-item sanity check ───────────────────────────────────────────
    print("\n[2] Fetching item 0 directly …")
    x0, y0 = ds[0]
    if x0.shape != torch.Size([T_LOOKBACK, N_CHANNELS]):
        _fail(f"Item x shape wrong: {x0.shape}")
    if y0.shape != torch.Size([T_FORECAST, N_CHANNELS]):
        _fail(f"Item y shape wrong: {y0.shape}")
    _pass(f"ds[0] x: {tuple(x0.shape)}   y: {tuple(y0.shape)}")

    # ── 3. DataLoader — pull exactly ONE batch ────────────────────────────────
    print(f"\n[3] Building DataLoader (batch_size={BATCH_SIZE}, num_workers=0) …")
    loader = DataLoader(
        ds,
        batch_size=BATCH_SIZE,
        shuffle=False,       # deterministic for test
        num_workers=0,       # avoid forking overhead in smoke-test
        pin_memory=False,
    )

    print(f"     Pulling 1 batch …")
    t1 = time.perf_counter()
    x_batch, y_batch = next(iter(loader))
    t_batch = time.perf_counter() - t1

    # ── 4. Shape assertions ───────────────────────────────────────────────────
    print("\n[4] Shape verification …")
    if x_batch.shape != EXPECTED_X_SHAPE:
        _fail(
            f"X batch shape mismatch!\n"
            f"       expected : {tuple(EXPECTED_X_SHAPE)}\n"
            f"       got      : {tuple(x_batch.shape)}"
        )
    _pass(f"X batch shape : {tuple(x_batch.shape)}  ← [batch × lookback × channels]")

    if y_batch.shape != EXPECTED_Y_SHAPE:
        _fail(
            f"Y batch shape mismatch!\n"
            f"       expected : {tuple(EXPECTED_Y_SHAPE)}\n"
            f"       got      : {tuple(y_batch.shape)}"
        )
    _pass(f"Y batch shape : {tuple(y_batch.shape)}  ← [batch × forecast × channels]")

    # ── 5. Numerical checks ───────────────────────────────────────────────────
    print("\n[5] Numerical health checks …")
    if not torch.isfinite(x_batch).all():
        _fail("X batch contains NaN or Inf values!")
    _pass("X batch: all values finite")

    if not torch.isfinite(y_batch).all():
        _fail("Y batch contains NaN or Inf values!")
    _pass("Y batch: all values finite")

    # Check z-scores are in a plausible range (bulk of data within ±6σ)
    x_max_abs = x_batch.abs().max().item()
    if x_max_abs > 50:
        print(f"  ⚠  X max |z| = {x_max_abs:.2f} — unusually large, check outliers")
    else:
        _pass(f"X max |z-score| = {x_max_abs:.4f}  (within ±50 σ)")

    # ── 6. Summary report ─────────────────────────────────────────────────────
    print()
    print(SEP)
    print("  BATCH DIAGNOSTIC SUMMARY")
    print(SEP)
    print(f"  X (lookback) shape : {list(x_batch.shape)}   dtype={x_batch.dtype}")
    print(f"  Y (forecast) shape : {list(y_batch.shape)}   dtype={y_batch.dtype}")
    print(f"  X — mean={x_batch.mean().item():.4f}  std={x_batch.std().item():.4f}"
          f"  min={x_batch.min().item():.4f}  max={x_batch.max().item():.4f}")
    print(f"  Y — mean={y_batch.mean().item():.4f}  std={y_batch.std().item():.4f}"
          f"  min={y_batch.min().item():.4f}  max={y_batch.max().item():.4f}")
    print(f"  Batch load time    : {t_batch*1000:.1f} ms")
    print(SEP)
    print()
    print("  All assertions PASSED  ✔")
    print()

    # ── 7. Explicit cleanup before exit ──────────────────────────────────────
    del x_batch, y_batch, loader, ds
    print("  Memory released. Exiting.")
    sys.exit(0)


if __name__ == "__main__":
    main()
