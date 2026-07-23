"""
test_gradient.py
----------------
Deep Learning Core Autograd Engineer — LinearGridTransformer project.

Verifies the mathematical and autograd gradient equivalence between:
  Path A: Original 5D tensor outer-product advanced indexing
  Path B: Patched 4D memory-efficient direct index-selection
"""

from __future__ import annotations

import os
import sys
import time
import math
import torch

# ── 1. Enforce Intel CPU multi-threading configurations ────────────────────────
cores = os.cpu_count() or 4
torch.set_num_threads(cores)
torch.set_num_interop_threads(2)


def main() -> None:
    print("═" * 70)
    print("  Autograd Gradient Verification Harness: 5D vs 4D Feature Maps")
    print("═" * 70)
    print(f"  CPU Cores: {cores}")
    print(f"  Intra-op threads: {torch.get_num_threads()}")
    print(f"  Inter-op threads: 2")
    print("-" * 70)

    # Configuration
    B, H, T, d = 2, 4, 128, 16
    rows, cols = torch.triu_indices(d, d)

    # Scale vector: 1.0 on diagonal, sqrt(2) on off-diagonal
    scale = torch.where(
        rows == cols,
        torch.ones(len(rows)),
        torch.full((len(rows),), math.sqrt(2.0))
    ).double()

    # ── Path A: Original 5D ───────────────────────────────────────────────────
    # Using float64 (double precision) to eliminate float32 round-off differences
    x_a = torch.randn(B, H, T, d, dtype=torch.float64, requires_grad=True)

    # Forward Path A
    x_outer_a = x_a.unsqueeze(-1) * x_a.unsqueeze(-2)
    quad_5d = x_outer_a[..., rows, cols] * scale
    loss_5d = quad_5d.sum()

    # Benchmark Backward Path A
    t_start_a = time.perf_counter()
    loss_5d.backward()
    t_back_a = time.perf_counter() - t_start_a

    # ── Path B: Patched 4D ────────────────────────────────────────────────────
    x_b = x_a.detach().clone()
    x_b.requires_grad = True

    # Forward Path B
    quad_4d = (x_b[..., rows] * x_b[..., cols]) * scale
    loss_4d = quad_4d.sum()

    # Benchmark Backward Path B
    t_start_b = time.perf_counter()
    loss_4d.backward()
    t_back_b = time.perf_counter() - t_start_b

    # ── Gradient Equivalence & Metrics ───────────────────────────────────────
    # Check shape compatibility
    assert quad_5d.shape == quad_4d.shape, f"Forward shape mismatch: {quad_5d.shape} vs {quad_4d.shape}"

    # Check forward values equivalence
    forward_diff = (quad_5d - quad_4d).abs().max().item()
    print(f"  Forward Max Absolute Difference : {forward_diff:.4e}")
    assert forward_diff < 1e-6, f"Forward values differ! Max diff: {forward_diff:.4e}"

    # Check backward gradient equivalence
    grad_diff = (x_a.grad - x_b.grad).abs().max().item()
    print(f"  Gradient Max Absolute Difference: {grad_diff:.4e}")
    assert grad_diff < 1e-6, f"Gradients differ! Max diff: {grad_diff:.4e}"

    # Verify non-zero gradients (anti-underflow/stagnation)
    mean_grad_mag = x_a.grad.abs().mean().item()
    print(f"  Mean Gradient Magnitude         : {mean_grad_mag:.4e}")
    assert mean_grad_mag > 1e-6, "Warning: Gradients are extremely small or zero (underflow risk)!"

    print("-" * 70)
    print("  Performance Profile (Backward Pass in Float64):")
    print(f"    Path A (Original 5D) time   : {t_back_a * 1000:.3f} ms")
    print(f"    Path B (Patched 4D) time    : {t_back_b * 1000:.3f} ms")

    speedup = t_back_a / max(t_back_b, 1e-9)
    print(f"    Speedup on CPU              : {speedup:.2f}x")
    print("═" * 70)
    print("  ALL VERIFICATION TESTS PASSED  ✔")
    print("═" * 70)


if __name__ == "__main__":
    main()
