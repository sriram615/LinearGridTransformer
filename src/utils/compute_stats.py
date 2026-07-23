"""
compute_stats.py
----------------
Mathematical Optimization Engineer — LinearGridTransformer project.

Reads the cleaned pickle, isolates the first 80 % of the timeline
(training split), and computes channel-wise mean and standard deviation
across the 7 numeric feature columns.

Outputs
-------
    src/data/global_mean.npy   shape (7,)  float64
    src/data/global_std.npy    shape (7,)  float64

Usage
-----
    python3 src/utils/compute_stats.py
"""

from __future__ import annotations

import pathlib
import sys
import textwrap

import numpy as np
import pandas as pd

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT      = pathlib.Path(__file__).resolve().parents[2]
PKL_PATH  = ROOT / "data" / "processed" / "clean_grid.pkl"
OUT_DIR   = ROOT / "src" / "data"
MEAN_PATH = OUT_DIR / "global_mean.npy"
STD_PATH  = OUT_DIR / "global_std.npy"

# ── Feature columns (same order as clean_chunks.py) ──────────────────────────
FEATURE_COLS: list[str] = [
    "Global_active_power",
    "Global_reactive_power",
    "Voltage",
    "Global_intensity",
    "Sub_metering_1",
    "Sub_metering_2",
    "Sub_metering_3",
]

TRAIN_RATIO = 0.80   # first 80 % of the timeline = training split
STD_EPS     = 1e-8   # guard against zero-std channels


def _fmt_bytes(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024:
            return f"{n:.1f} {unit}"
        n /= 1024  # type: ignore[assignment]
    return f"{n:.1f} TB"


def main() -> None:
    # ── 1. Load cleaned data ──────────────────────────────────────────────────
    if not PKL_PATH.exists():
        print(f"[ERROR] Cleaned pickle not found: {PKL_PATH}", file=sys.stderr)
        print("        Run  src/utils/clean_chunks.py  first.", file=sys.stderr)
        sys.exit(1)

    print(f"Loading  : {PKL_PATH}  ({_fmt_bytes(PKL_PATH.stat().st_size)})")
    df = pd.read_pickle(str(PKL_PATH))
    total_rows = len(df)
    print(f"Total rows: {total_rows:,}")

    # ── 2. Carve out training split (first 80 % by position) ─────────────────
    split_idx = int(total_rows * TRAIN_RATIO)
    df_train  = df.iloc[:split_idx][FEATURE_COLS]

    print(f"Train rows: {len(df_train):,}  ({TRAIN_RATIO*100:.0f} %  of timeline, "
          f"rows 0 – {split_idx-1:,})")
    print(f"Val rows  : {total_rows - split_idx:,}  (held-out, rows {split_idx:,} – {total_rows-1:,})")
    print()

    # ── 3. Compute channel-wise statistics (float64 precision) ────────────────
    # Cast to float64 so accumulation is exact (stored as float32 on disk).
    arr = df_train.to_numpy(dtype=np.float64)

    global_mean = arr.mean(axis=0)          # shape (7,)
    global_std  = arr.std(axis=0, ddof=0)   # population std — consistent with z-score normalisation

    # Safety: clamp near-zero std to avoid division-by-zero during normalisation
    global_std = np.where(global_std < STD_EPS, STD_EPS, global_std)

    # ── 4. Persist ────────────────────────────────────────────────────────────
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    np.save(str(MEAN_PATH), global_mean)
    np.save(str(STD_PATH),  global_std)
    print(f"Saved: {MEAN_PATH}")
    print(f"Saved: {STD_PATH}")
    print()

    # ── 5. Pretty-print verification table ───────────────────────────────────
    col_w = max(len(c) for c in FEATURE_COLS) + 2
    header = f"{'Channel':<{col_w}}  {'Mean':>14}  {'Std':>14}  {'Min':>14}  {'Max':>14}"
    rule   = "─" * len(header)

    print(rule)
    print(header)
    print(rule)

    col_min = arr.min(axis=0)
    col_max = arr.max(axis=0)

    for i, col in enumerate(FEATURE_COLS):
        print(
            f"{col:<{col_w}}  "
            f"{global_mean[i]:>14.6f}  "
            f"{global_std[i]:>14.6f}  "
            f"{col_min[i]:>14.6f}  "
            f"{col_max[i]:>14.6f}"
        )
    print(rule)

    # Quick sanity: normalise one value and verify it is O(1)
    z_check = (arr[0] - global_mean) / global_std
    print()
    print("Sanity check — z-score of row 0 (should be O(1)):")
    for i, col in enumerate(FEATURE_COLS):
        print(f"  {col:<{col_w - 2}} z = {z_check[i]:+.4f}")

    # Read-back verification
    mean_rb = np.load(str(MEAN_PATH))
    std_rb  = np.load(str(STD_PATH))
    assert np.allclose(mean_rb, global_mean), "Mean read-back mismatch!"
    assert np.allclose(std_rb,  global_std),  "Std  read-back mismatch!"
    print()
    print(textwrap.dedent(f"""\
        ╔══════════════════════════════════════════════╗
        ║  global_mean.npy  shape={mean_rb.shape}  dtype={mean_rb.dtype}  ║
        ║  global_std.npy   shape={std_rb.shape}  dtype={std_rb.dtype}  ║
        ║  Read-back assertion: PASSED  ✔              ║
        ╚══════════════════════════════════════════════╝"""))


if __name__ == "__main__":
    main()
