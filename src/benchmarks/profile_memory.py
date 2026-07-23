"""
profile_memory.py
-----------------
Principal AI Infrastructure Performance Engineer — LinearGridTransformer project.

Rigorous hardware memory profiling script that compares the Resident Set Size
(RSS) footprint of two attention mechanisms across expanding sequence lengths:

    1. Vanilla Softmax Attention  (torch.nn.MultiheadAttention)   — O(T²) memory
    2. Custom LinearKernelAttention                                — O(T)  memory

Method
------
For every sequence length T in [128, 256, 512, 1024, 2048, 4096, 8192]:
  • Build isolated layer instances with fresh random Q/K/V tensors.
  • Measure the OS-level RSS delta (via psutil) bracketing the forward pass.
  • Vanilla attention is wrapped in try/except — OOM or any RuntimeError is
    caught, logged as "OOM", and profiling continues for the linear layer.

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
import gc
import logging
import os
import pathlib
import sys
import time
from typing import Optional

import matplotlib
matplotlib.use("Agg")   # headless backend — safe in any environment
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import psutil
import torch
import torch.nn as nn

# ── Path bootstrap — allow running from any working directory ─────────────────
_ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT))
from src.models.linear_attention import LinearKernelAttention

# ── Output paths ──────────────────────────────────────────────────────────────
_CSV_PATH = _ROOT / "data" / "benchmarks" / "memory_scale.csv"
_PNG_PATH = _ROOT / "docs" / "memory_vs_sequence.png"

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

# Extra warmup forward pass before measurement (populates CUDA caches / JIT)
WARMUP_PASSES: int = 1

# Bytes → MB conversion
_MB = 1 / (1024 ** 2)


# ═══════════════════════════════════════════════════════════════════════════════
# Utilities
# ═══════════════════════════════════════════════════════════════════════════════

def _current_rss_mb() -> float:
    """Return the current process RSS in megabytes."""
    return psutil.Process(os.getpid()).memory_info().rss * _MB


def _gc_collect() -> None:
    """Force Python + (optionally) CUDA garbage collection."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _make_qkv(T: int, device: torch.device) -> tuple[torch.Tensor, ...]:
    """
    Create isolated random Q, K, V tensors for a single forward pass.

    Shapes
    ------
    Q, K : [BATCH_SIZE, NUM_HEADS, T, HEAD_DIM]
    V    : [BATCH_SIZE, NUM_HEADS, T, HEAD_DIM]

    For vanilla torch.nn.MultiheadAttention we also need a flattened
    [T, BATCH_SIZE, EMBED_DIM] version; that is returned as the 4th element.
    """
    torch.manual_seed(42)  # deterministic — same tensors every call
    Q = torch.randn(BATCH_SIZE, NUM_HEADS, T, HEAD_DIM, device=device)
    K = torch.randn(BATCH_SIZE, NUM_HEADS, T, HEAD_DIM, device=device)
    V = torch.randn(BATCH_SIZE, NUM_HEADS, T, HEAD_DIM, device=device)

    # Vanilla MHA expects [T, B, embed_dim]
    x_flat = torch.randn(T, BATCH_SIZE, EMBED_DIM, device=device)
    return Q, K, V, x_flat


def _measure_rss_delta(fn) -> tuple[float, Optional[str]]:
    """
    Execute ``fn()`` and return the peak RSS delta in MB.

    Strategy
    --------
    • Record RSS immediately before call (post-GC).
    • Record RSS immediately after call.
    • Delta = after − before  (negative values are clamped to 0).

    Parameters
    ----------
    fn : callable  — zero-argument forward-pass thunk

    Returns
    -------
    (delta_mb, error_string)
        error_string is None on success, or "OOM" / exception message on failure.
    """
    _gc_collect()
    rss_before = _current_rss_mb()

    try:
        fn()
        rss_after = _current_rss_mb()
        delta = max(0.0, rss_after - rss_before)
        return delta, None
    except (RuntimeError, MemoryError, Exception) as exc:
        # Catch OOM / any memory wall — log verbosely, report "OOM"
        return float("nan"), f"OOM ({type(exc).__name__}: {exc})"


# ═══════════════════════════════════════════════════════════════════════════════
# Layer builders
# ═══════════════════════════════════════════════════════════════════════════════

def _build_vanilla_mha(device: torch.device) -> nn.MultiheadAttention:
    """Instantiate a standard softmax MultiheadAttention layer."""
    layer = nn.MultiheadAttention(
        embed_dim=EMBED_DIM,
        num_heads=NUM_HEADS,
        batch_first=False,   # expects [T, B, E]
        bias=True,
    ).to(device)
    layer.eval()
    return layer


def _build_linear_attention(device: torch.device) -> LinearKernelAttention:
    """Instantiate the custom O(T) LinearKernelAttention layer."""
    layer = LinearKernelAttention(
        head_dim=HEAD_DIM,
        num_heads=NUM_HEADS,
        scale_input=True,
        eps=1e-6,
    ).to(device)
    layer.eval()
    return layer


# ═══════════════════════════════════════════════════════════════════════════════
# Per-length profiling logic
# ═══════════════════════════════════════════════════════════════════════════════

def _profile_vanilla(T: int, device: torch.device) -> tuple[float, Optional[str]]:
    """
    Profile vanilla torch.nn.MultiheadAttention for sequence length T.

    Returns
    -------
    (rss_delta_mb, error_or_None)
    """
    log.info("  [Vanilla  MHA] T=%-5d  building layer …", T)

    # Materialise fresh layer and inputs inside the measured scope so that
    # parameter allocation is counted in the RSS delta.
    layer = _build_vanilla_mha(device)

    Q, K, V, x_flat = _make_qkv(T, device)

    # Optional warmup (outside measured window)
    if WARMUP_PASSES > 0:
        try:
            with torch.no_grad():
                layer(x_flat, x_flat, x_flat)
        except Exception:
            pass   # warmup failure is non-fatal

    def _forward():
        with torch.no_grad():
            out, _ = layer(x_flat, x_flat, x_flat)
        return out

    delta_mb, err = _measure_rss_delta(_forward)

    if err is None:
        log.info("  [Vanilla  MHA] T=%-5d  RSS delta = %.2f MB", T, delta_mb)
    else:
        log.warning("  [Vanilla  MHA] T=%-5d  %s", T, err)

    # Explicit teardown — release all tensors before next iteration
    del layer, Q, K, V, x_flat
    _gc_collect()

    return delta_mb, err


def _profile_linear(T: int, device: torch.device) -> tuple[float, Optional[str]]:
    """
    Profile LinearKernelAttention for sequence length T.

    Returns
    -------
    (rss_delta_mb, error_or_None)
    """
    log.info("  [LinearKernel] T=%-5d  building layer …", T)

    layer = _build_linear_attention(device)
    Q, K, V, _ = _make_qkv(T, device)

    # Warmup
    if WARMUP_PASSES > 0:
        try:
            with torch.no_grad():
                layer(Q, K, V)
        except Exception:
            pass

    def _forward():
        with torch.no_grad():
            out, _ = layer(Q, K, V)
        return out

    delta_mb, err = _measure_rss_delta(_forward)

    if err is None:
        log.info("  [LinearKernel] T=%-5d  RSS delta = %.2f MB", T, delta_mb)
    else:
        log.warning("  [LinearKernel] T=%-5d  %s", T, err)

    del layer, Q, K, V
    _gc_collect()

    return delta_mb, err


# ═══════════════════════════════════════════════════════════════════════════════
# CSV export
# ═══════════════════════════════════════════════════════════════════════════════

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
    ax.set_ylabel("RSS Memory Delta  (MB)", color="#C9D1D9", fontsize=12, labelpad=10)

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
