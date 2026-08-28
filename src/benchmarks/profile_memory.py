"""
profile_memory.py
-----------------
Principal AI Infrastructure Performance Engineer — LinearGridTransformer project.

Rigorous hardware memory profiling script that compares the peak Resident Set
Size (RSS) of two attention mechanisms across expanding sequence lengths:

    1. Vanilla Softmax Attention  (torch.nn.MultiheadAttention)   — O(T²) memory
    2. Custom LinearKernelAttention                                — O(T)  memory

Method
------
For every sequence length T in [128, 256, 512, 1024, 2048, 4096, 8192], each
(mechanism, T) pair runs in its own **fresh subprocess** (see
``_memory_worker.py``), which reports its own peak RSS via
``resource.getrusage(RUSAGE_SELF).ru_maxrss``.

  Vanilla attention is wrapped in try/except inside the worker — OOM or any
  RuntimeError is caught, reported back as an error, and profiling continues
  for the linear layer.

Why a subprocess per data point, instead of measuring RSS before/after a
forward pass in one long-lived process
-------------------------------------------------------------------------
An earlier version of this script measured RSS immediately before and after
each forward pass in the same process, with an untimed "warmup" pass run
first using the exact same tensor shapes. That warmup silently defeated the
measurement: PyTorch's CPU allocator keeps freed buffers in an internal pool
rather than returning them to the OS, so the warmup pass already grew the
process to (approximately) its peak size, and the *timed* pass then reused
that cached memory — producing a near-zero, noisy "delta" regardless of T
(this is exactly what the old data/benchmarks/memory_scale.csv showed: sub-MB
deltas at every T up to 8192, non-monotonic, with vanilla_oom never True).
Running each data point in an isolated process removes that cross-call
contamination entirely, and reading peak RSS (rather than a before/after
snapshot) captures the transient [T×T] score-matrix allocation even though
PyTorch frees it again before the process exits.

Outputs
-------
  data/benchmarks/memory_scale.csv   — raw data points (T, rss_vanilla_mb,
                                        rss_linear_mb, vanilla_oom flag)
  docs/memory_vs_sequence.png        — publication-ready line graph

Usage
-----
    python src/benchmarks/profile_memory.py

Dependencies: torch, psutil, matplotlib, numpy (all standard ML deps)
"""

from __future__ import annotations

import csv
import json
import logging
import pathlib
import subprocess
import sys
from typing import Optional

import matplotlib
matplotlib.use("Agg")   # headless backend — safe in any environment
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import torch

# ── Path bootstrap — allow running from any working directory ─────────────────
_ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT))

# ── Output paths ──────────────────────────────────────────────────────────────
_CSV_PATH = _ROOT / "data" / "benchmarks" / "memory_scale.csv"
_PNG_PATH = _ROOT / "docs" / "memory_vs_sequence.png"

# ── Per-data-point subprocess timeout (seconds) — a hang past this is treated
#    as an OOM/thrash condition rather than left to block the whole sweep ────
_WORKER_TIMEOUT_S = 180

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("profile_memory")

# ── Profiling configuration ───────────────────────────────────────────────────
SEQUENCE_LENGTHS: list[int] = [128, 256, 512, 1024, 2048, 4096, 8192]

# Shared attention hyperparameters (identical for both layers)
BATCH_SIZE: int = 1       # single sample — isolates T scaling cleanly
NUM_HEADS:  int = 4
HEAD_DIM:   int = 16      # per-head dimension; embed_dim = NUM_HEADS * HEAD_DIM
EMBED_DIM:  int = NUM_HEADS * HEAD_DIM   # = 64

# ═══════════════════════════════════════════════════════════════════════════════
# Isolated-subprocess measurement
# ═══════════════════════════════════════════════════════════════════════════════

def _run_worker(mode: str, T: int, device: torch.device) -> tuple[float, Optional[str]]:
    """
    Run one (mode, T) measurement in a fresh subprocess via ``_memory_worker.py``
    and return (peak_rss_mb, error_or_None).

    See the module docstring for why this runs out-of-process rather than
    measuring a before/after RSS delta in the long-lived parent process.
    """
    cmd = [
        sys.executable, "-m", "src.benchmarks._memory_worker",
        "--mode", mode, "--T", str(T),
        "--device", device.type,
    ]
    try:
        result = subprocess.run(
            cmd, cwd=str(_ROOT), capture_output=True, text=True,
            timeout=_WORKER_TIMEOUT_S,
        )
    except subprocess.TimeoutExpired:
        return float("nan"), f"OOM (worker exceeded {_WORKER_TIMEOUT_S}s — likely thrashing)"

    stdout = result.stdout.strip()
    if not stdout:
        # Process died (e.g. OS OOM-killer) before it could print its report.
        return float("nan"), (
            f"OOM (subprocess exit code {result.returncode}, "
            f"stderr tail: {result.stderr[-300:]!r})"
        )

    try:
        payload = json.loads(stdout.splitlines()[-1])
    except (json.JSONDecodeError, IndexError):
        return float("nan"), f"OOM (unparseable worker output: {stdout!r})"

    if payload.get("error"):
        return float("nan"), payload["error"]
    return float(payload["peak_rss_mb"]), None


# ═══════════════════════════════════════════════════════════════════════════════
# Per-length profiling logic
# ═══════════════════════════════════════════════════════════════════════════════

def _profile_vanilla(T: int, device: torch.device) -> tuple[float, Optional[str]]:
    """
    Profile vanilla torch.nn.MultiheadAttention for sequence length T, in an
    isolated subprocess.

    Returns
    -------
    (peak_rss_mb, error_or_None)
    """
    log.info("  [Vanilla  MHA] T=%-5d  running isolated subprocess …", T)

    peak_mb, err = _run_worker("vanilla", T, device)

    if err is None:
        log.info("  [Vanilla  MHA] T=%-5d  peak RSS = %.2f MB", T, peak_mb)
    else:
        log.warning("  [Vanilla  MHA] T=%-5d  %s", T, err)

    return peak_mb, err


def _profile_linear(T: int, device: torch.device) -> tuple[float, Optional[str]]:
    """
    Profile LinearKernelAttention for sequence length T, in an isolated
    subprocess.

    Returns
    -------
    (peak_rss_mb, error_or_None)
    """
    log.info("  [LinearKernel] T=%-5d  running isolated subprocess …", T)

    peak_mb, err = _run_worker("linear", T, device)

    if err is None:
        log.info("  [LinearKernel] T=%-5d  peak RSS = %.2f MB", T, peak_mb)
    else:
        log.warning("  [LinearKernel] T=%-5d  %s", T, err)

    return peak_mb, err


# ═══════════════════════════════════════════════════════════════════════════════
# CSV export
# ═══════════════════════════════════════════════════════════════════════════════

# NOTE: *_rss_mb columns are each mechanism's peak process RSS while running
# a single isolated forward pass in its own subprocess (see _memory_worker.py)
# — not a before/after delta in a shared process. See module docstring.
_CSV_HEADER = [
    "sequence_length",
    "vanilla_rss_mb",
    "linear_rss_mb",
    "vanilla_oom",
]


def _write_csv(rows: list[dict]) -> None:
    _CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    with _CSV_PATH.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=_CSV_HEADER)
        writer.writeheader()
        writer.writerows(rows)
    log.info("CSV written → %s", _CSV_PATH)


# ═══════════════════════════════════════════════════════════════════════════════
# Publication-ready plot
# ═══════════════════════════════════════════════════════════════════════════════

# Colour palette — colourblind-safe, print-friendly
_VANILLA_COLOR  = "#E05C5C"   # warm red
_LINEAR_COLOR   = "#3A86FF"   # electric blue
_OOM_COLOR      = "#FF9F1C"   # amber for OOM marker


def _make_plot(rows: list[dict]) -> None:
    """Generate a publication-ready memory vs sequence-length figure."""

    Ts          = [r["sequence_length"]  for r in rows]
    vanilla_rss = [r["vanilla_rss_mb"]   for r in rows]
    linear_rss  = [r["linear_rss_mb"]    for r in rows]
    oom_flags   = [r["vanilla_oom"]       for r in rows]

    # ── Figure layout ─────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 6))
    fig.patch.set_facecolor("#0D1117")
    ax.set_facecolor("#161B22")

    # ── Grid ──────────────────────────────────────────────────────────────────
    ax.grid(color="#30363D", linewidth=0.6, linestyle="--", zorder=0)
    ax.set_axisbelow(True)

    # ── Separate valid / OOM vanilla points ───────────────────────────────────
    Ts_vanilla_valid = [T for T, oom in zip(Ts, oom_flags) if not oom]
    rss_vanilla_valid = [
        r for r, oom in zip(vanilla_rss, oom_flags) if not oom
    ]
    Ts_oom = [T for T, oom in zip(Ts, oom_flags) if oom]

    # ── Plot lines ────────────────────────────────────────────────────────────
    ax.plot(
        Ts_vanilla_valid, rss_vanilla_valid,
        color=_VANILLA_COLOR,
        linewidth=2.5,
        marker="o", markersize=7,
        markerfacecolor="white", markeredgewidth=2,
        label="Softmax MHA  — O(T²)",
        zorder=3,
    )

    ax.plot(
        Ts, linear_rss,
        color=_LINEAR_COLOR,
        linewidth=2.5,
        marker="s", markersize=7,
        markerfacecolor="white", markeredgewidth=2,
        label="LinearKernelAttention  — O(T)",
        zorder=3,
    )

    # ── OOM annotation markers ────────────────────────────────────────────────
    for T_oom in Ts_oom:
        ax.axvline(
            x=T_oom, color=_OOM_COLOR, linewidth=1.2,
            linestyle=":", alpha=0.7, zorder=2,
        )
        ax.annotate(
            "OOM",
            xy=(T_oom, ax.get_ylim()[1] if ax.get_ylim()[1] > 0 else 1),
            xycoords=("data", "axes fraction"),
            xytext=(4, -18), textcoords="offset points",
            fontsize=8.5, color=_OOM_COLOR,
            fontweight="bold",
            rotation=90,
            va="top",
        )

    # ── Axes styling ──────────────────────────────────────────────────────────
    ax.set_xscale("log", base=2)
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
    ax.set_xticks(Ts)

    ax.tick_params(colors="#C9D1D9", labelsize=10)
    for spine in ax.spines.values():
        spine.set_edgecolor("#30363D")

    ax.set_xlabel("Sequence Length  (T)", color="#C9D1D9", fontsize=12, labelpad=10)
    ax.set_ylabel("Peak Process RSS  (MB)", color="#C9D1D9", fontsize=12, labelpad=10)

    # ── Title & legend ────────────────────────────────────────────────────────
    ax.set_title(
        "Memory Scaling: Softmax MHA vs LinearKernelAttention",
        color="#F0F6FC", fontsize=14, fontweight="bold", pad=16,
    )

    legend = ax.legend(
        framealpha=0.2,
        facecolor="#161B22",
        edgecolor="#30363D",
        labelcolor="#C9D1D9",
        fontsize=11,
        loc="upper left",
    )

    # ── Annotation box with model specs ───────────────────────────────────────
    spec_text = (
        f"Config: B={BATCH_SIZE}, H={NUM_HEADS}, head_dim={HEAD_DIM}\n"
        f"embed_dim={EMBED_DIM}, device={'CUDA' if torch.cuda.is_available() else 'CPU'}\n"
        f"OOM threshold indicated by dotted vertical lines"
    )
    ax.text(
        0.98, 0.04, spec_text,
        transform=ax.transAxes,
        fontsize=8.5, color="#8B949E",
        va="bottom", ha="right",
        bbox=dict(
            boxstyle="round,pad=0.5",
            facecolor="#0D1117",
            edgecolor="#30363D",
            alpha=0.85,
        ),
    )

    # ── Save ──────────────────────────────────────────────────────────────────
    _PNG_PATH.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(_PNG_PATH, dpi=200, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    log.info("Plot saved → %s", _PNG_PATH)


# ═══════════════════════════════════════════════════════════════════════════════
# Main profiling loop
# ═══════════════════════════════════════════════════════════════════════════════

def run_profiling() -> None:
    """
    Main entry point.

    Iterates over SEQUENCE_LENGTHS, profiles both attention mechanisms,
    accumulates results, writes CSV and PNG.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    SEP = "═" * 66
    log.info(SEP)
    log.info("  Memory Profiling — Softmax MHA vs LinearKernelAttention")
    log.info("  Device   : %s", device)
    log.info("  Config   : B=%d  H=%d  head_dim=%d  embed_dim=%d",
             BATCH_SIZE, NUM_HEADS, HEAD_DIM, EMBED_DIM)
    log.info("  T range  : %s", SEQUENCE_LENGTHS)
    log.info(SEP)

    rows: list[dict] = []

    for T in SEQUENCE_LENGTHS:
        log.info("")
        log.info("━━  T = %d  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━", T)

        # ── 1. Vanilla softmax MHA (may OOM) ──────────────────────────────────
        vanilla_delta, vanilla_err = _profile_vanilla(T, device)
        vanilla_oom = vanilla_err is not None

        # ── 2. LinearKernelAttention (always runs) ────────────────────────────
        linear_delta, linear_err = _profile_linear(T, device)

        row = {
            "sequence_length": T,
            "vanilla_rss_mb":  round(vanilla_delta, 4) if not vanilla_oom else float("nan"),
            "linear_rss_mb":   round(linear_delta, 4),
            "vanilla_oom":     vanilla_oom,
        }
        rows.append(row)

        log.info(
            "  Summary T=%-5d | Vanilla: %s MB | Linear: %.2f MB%s",
            T,
            f"{vanilla_delta:.2f}" if not vanilla_oom else "OOM",
            linear_delta,
            " ← linear survives!" if vanilla_oom else "",
        )

    # ── Write outputs ─────────────────────────────────────────────────────────
    log.info("")
    log.info(SEP)
    _write_csv(rows)
    _make_plot(rows)

    # ── Print summary table ───────────────────────────────────────────────────
    log.info("")
    log.info("  %-12s  %-16s  %-16s  %s",
             "T", "Vanilla RSS (MB)", "Linear RSS (MB)", "Vanilla OOM?")
    log.info("  %s", "-" * 58)
    for r in rows:
        vanilla_str = f"{r['vanilla_rss_mb']:.2f}" if not r["vanilla_oom"] else "OOM"
        log.info("  %-12d  %-16s  %-16.2f  %s",
                 r["sequence_length"], vanilla_str, r["linear_rss_mb"],
                 "YES" if r["vanilla_oom"] else "no")
    log.info(SEP)


# ═══════════════════════════════════════════════════════════════════════════════
# Entry point
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    run_profiling()
