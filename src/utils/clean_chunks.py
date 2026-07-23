"""
clean_chunks.py
---------------
Data Infrastructure Engineer — LinearGridTransformer project.

Loads data/raw/data.txt in 100,000-row chunks (minimises Mac RAM footprint),
coerces the 7 numeric target channels to float64, replaces '?' / corrupted
values via forward-fill within each chunk, then writes the combined result to:
    data/processed/clean_grid.pkl  (compressed pickle, protocol 4)

Usage
-----
    python src/utils/clean_chunks.py

Post-run validation is printed automatically (row count, file size).
"""

from __future__ import annotations

import os
import sys
import time
import pathlib
import logging

import pandas as pd

# ── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT        = pathlib.Path(__file__).resolve().parents[2]   # repo root
RAW_FILE    = ROOT / "data" / "raw"  / "data.txt"
OUT_DIR     = ROOT / "data" / "processed"
OUT_PKL     = OUT_DIR / "clean_grid.pkl"

# ── Column configuration ──────────────────────────────────────────────────────
DELIMITER   = ";"
CHUNKSIZE   = 100_000

# The 7 numeric target channels (columns 2–8, 0-indexed).
NUMERIC_COLS: list[str] = [
    "Global_active_power",
    "Global_reactive_power",
    "Voltage",
    "Global_intensity",
    "Sub_metering_1",
    "Sub_metering_2",
    "Sub_metering_3",
]

# Columns to keep in output (Date + Time retained for traceability).
KEEP_COLS: list[str] = ["Date", "Time"] + NUMERIC_COLS


# ── Helpers ───────────────────────────────────────────────────────────────────

def _fmt_bytes(n: int) -> str:
    """Human-readable byte size."""
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024:
            return f"{n:.1f} {unit}"
        n /= 1024  # type: ignore[assignment]
    return f"{n:.1f} TB"


def clean_chunk(chunk: pd.DataFrame) -> pd.DataFrame:
    """
    Per-chunk cleaning pipeline:
      1. Replace literal '?' (and empty strings) with NaN.
      2. Coerce each target column to float64 (errors → NaN).
      3. Forward-fill NaN within the chunk; backfill any leading NaNs.
      4. Drop rows that are still fully NaN across all numeric cols
         (only possible when the *entire* chunk is corrupted — extremely rare).
    """
    # Step 1 — mark corrupted sentinels as NaN
    chunk = chunk.replace("?", pd.NA).replace("", pd.NA)

    # Step 2 — coerce target columns to numeric
    for col in NUMERIC_COLS:
        if col in chunk.columns:
            chunk[col] = pd.to_numeric(chunk[col], errors="coerce")

    # Step 3 — forward-fill then backward-fill (handles leading NaNs in chunk)
    chunk[NUMERIC_COLS] = (
        chunk[NUMERIC_COLS]
        .ffill()
        .bfill()
    )

    return chunk


# ── Main pipeline ─────────────────────────────────────────────────────────────

def main() -> None:
    if not RAW_FILE.exists():
        log.error("Raw file not found: %s", RAW_FILE)
        sys.exit(1)

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    log.info("Source  : %s  (%s)", RAW_FILE, _fmt_bytes(RAW_FILE.stat().st_size))
    log.info("Chunk   : %d rows", CHUNKSIZE)
    log.info("Output  : %s", OUT_PKL)
    log.info("─" * 60)

    reader = pd.read_csv(
        RAW_FILE,
        sep=DELIMITER,
        chunksize=CHUNKSIZE,
        dtype=str,          # read everything as str first — safest for '?' detection
        na_values=["?", ""],
        low_memory=False,
    )

    cleaned_chunks: list[pd.DataFrame] = []
    total_raw_rows  = 0
    total_bad_cells = 0
    t_start         = time.perf_counter()

    for chunk_idx, chunk in enumerate(reader, start=1):
        raw_rows   = len(chunk)
        total_raw_rows += raw_rows

        # Count corrupted / missing cells before cleaning (for reporting)
        bad_before = chunk[NUMERIC_COLS].isna().sum().sum()
        total_bad_cells += int(bad_before)

        cleaned = clean_chunk(chunk)

        # Keep only the desired columns
        cleaned = cleaned[[c for c in KEEP_COLS if c in cleaned.columns]]

        cleaned_chunks.append(cleaned)

        log.info(
            "Chunk %3d | rows=%7d | NaN-before=%5d | after-fill NaN=%d",
            chunk_idx,
            raw_rows,
            int(bad_before),
            int(cleaned[NUMERIC_COLS].isna().sum().sum()),
        )

    # ── Concatenate all chunks ────────────────────────────────────────────────
    log.info("─" * 60)
    log.info("Concatenating %d chunk(s)…", len(cleaned_chunks))
    df_all = pd.concat(cleaned_chunks, ignore_index=True)

    # Final dtype tightening — numeric cols to float32 saves ~40 % RAM vs float64
    df_all[NUMERIC_COLS] = df_all[NUMERIC_COLS].astype("float32")

    # ── Persist ───────────────────────────────────────────────────────────────
    log.info("Writing pickle (protocol=4, compression=None)…")
    df_all.to_pickle(str(OUT_PKL), protocol=4)

    elapsed = time.perf_counter() - t_start

    # ── Validation report ─────────────────────────────────────────────────────
    file_size_bytes = OUT_PKL.stat().st_size
    log.info("─" * 60)
    log.info("✔  Pipeline complete in %.1f s", elapsed)
    log.info("   Total rows (raw)      : %s", f"{total_raw_rows:,}")
    log.info("   Total rows (cleaned)  : %s", f"{len(df_all):,}")
    log.info("   Corrupt cells fixed   : %s", f"{total_bad_cells:,}")
    log.info("   Output file size      : %s", _fmt_bytes(file_size_bytes))
    log.info("   Output path           : %s", OUT_PKL)

    # ── Quick sanity read-back ────────────────────────────────────────────────
    log.info("─" * 60)
    log.info("Sanity read-back…")
    df_check = pd.read_pickle(str(OUT_PKL))
    assert len(df_check) == len(df_all), "Row count mismatch after read-back!"
    log.info("   Read-back rows        : %s  ✔", f"{len(df_check):,}")
    log.info("   Columns               : %s", list(df_check.columns))
    log.info("   dtypes:\n%s", df_check.dtypes.to_string())
    log.info("─" * 60)

    # ── Print summary to stdout (easy to capture in CI/CD) ───────────────────
    print("\n" + "=" * 60)
    print(f"  clean_grid.pkl  │  rows={len(df_all):,}  │  size={_fmt_bytes(file_size_bytes)}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
