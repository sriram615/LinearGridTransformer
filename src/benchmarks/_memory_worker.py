"""
_memory_worker.py
------------------
Runs a single attention forward pass (vanilla softmax MHA or
LinearKernelAttention) for one sequence length, in an isolated process, and
reports the process's true peak RSS.

Why this exists
----------------
profile_memory.py used to measure memory in one long-lived process across an
increasing sweep of T, taking a before/after RSS snapshot around each forward
pass — with an untimed warmup pass run first using the *same* tensor shapes.
That warmup silently defeated the measurement: PyTorch's CPU allocator keeps
freed buffers in an internal pool rather than returning them to the OS, so by
the time the "measured" pass ran, the process had already grown to its peak
size during warmup, and the timed pass just reused that cached memory —
producing a near-zero, noisy delta regardless of T.

Running each (mechanism, T) data point in its own fresh process removes that
cross-call contamination entirely: this process only ever does one thing, so
its peak RSS (via ``resource.getrusage(RUSAGE_SELF).ru_maxrss``, which is a
true running maximum, not a snapshot) directly reflects the memory that one
forward pass required — including the transient [T×T] score matrix, even
though PyTorch frees it again before this process exits.

Usage (invoked by profile_memory.py via subprocess; not meant to be run by
hand, though it works fine standalone for debugging)
-----------------------------------------------------
    python -m src.benchmarks._memory_worker --mode vanilla --T 4096
    python -m src.benchmarks._memory_worker --mode linear  --T 4096 --device cpu

Prints exactly one JSON line to stdout:
    {"peak_rss_mb": <float>, "cuda_peak_mb": <float|null>, "error": <str|null>}
"""

from __future__ import annotations

import argparse
import json
import pathlib
import resource
import sys

_ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT))


def _peak_rss_mb() -> float:
    """
    Peak resident set size of this process so far, in MB.

    ru_maxrss units differ by platform: Linux reports KiB, macOS/BSD report
    bytes. Get this wrong and every number on macOS is off by 1024x.
    """
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if sys.platform == "darwin":
        return peak / (1024 ** 2)
    return peak / 1024  # Linux: KiB → MiB


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["vanilla", "linear"], required=True)
    parser.add_argument("--T", type=int, required=True)
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    args = parser.parse_args()

    error: str | None = None
    cuda_peak_mb: float | None = None

    try:
        import torch
        import torch.nn as nn

        from src.models.linear_attention import LinearKernelAttention

        BATCH_SIZE, NUM_HEADS, HEAD_DIM = 1, 4, 16
        EMBED_DIM = NUM_HEADS * HEAD_DIM

        device = torch.device(
            args.device if (args.device == "cpu" or torch.cuda.is_available()) else "cpu"
        )
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)

        torch.manual_seed(42)

        if args.mode == "vanilla":
            layer = nn.MultiheadAttention(
                embed_dim=EMBED_DIM, num_heads=NUM_HEADS,
                batch_first=False, bias=True,
            ).to(device).eval()
            x = torch.randn(args.T, BATCH_SIZE, EMBED_DIM, device=device)
            with torch.no_grad():
                layer(x, x, x)
        else:
            layer = LinearKernelAttention(
                head_dim=HEAD_DIM, num_heads=NUM_HEADS,
                scale_input=True, eps=1e-6,
            ).to(device).eval()
            Q = torch.randn(BATCH_SIZE, NUM_HEADS, args.T, HEAD_DIM, device=device)
            K = torch.randn(BATCH_SIZE, NUM_HEADS, args.T, HEAD_DIM, device=device)
            V = torch.randn(BATCH_SIZE, NUM_HEADS, args.T, HEAD_DIM, device=device)
            with torch.no_grad():
                layer(Q, K, V)

        if device.type == "cuda":
            cuda_peak_mb = torch.cuda.max_memory_allocated(device) / (1024 ** 2)

    except Exception as exc:  # deliberately broad — OOM and any RuntimeError both count
        error = f"{type(exc).__name__}: {exc}"

    print(json.dumps({
        "peak_rss_mb": _peak_rss_mb(),
        "cuda_peak_mb": cuda_peak_mb,
        "error": error,
    }))


if __name__ == "__main__":
    main()
