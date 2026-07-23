"""
kernel.py
---------
Principal Theoretical ML Researcher — LinearGridTransformer project.

Implements a deterministic 2nd-degree Taylor Series feature map that
approximates the softmax exponential kernel:

    exp(qᵀk) ≈ φ(q)ᵀ φ(k)

where the feature map for a d-dimensional vector x is:

    φ(x) = [ 1,
              x₁, …, x_d,
              x₁², …, x_d²,
              √2·x₁x₂, √2·x₁x₃, …, √2·x_{d-1}·x_d ]

Inner-product verification
--------------------------
    φ(x)ᵀφ(y)
      = 1
      + Σᵢ xᵢyᵢ                     (≡ xᵀy)
      + Σᵢ xᵢ²yᵢ²  +  2Σᵢ<ⱼ xᵢxⱼyᵢyⱼ
      = 1 + xᵀy + (xᵀy)²

    which matches the 2nd-order Taylor expansion of exp(xᵀy) up to the
    1/2! scaling factor on the quadratic term (common in linearised-
    attention literature where relative scale is absorbed by layer norm).

Output dimension
----------------
    d_prime = 1          (constant)
            + d          (linear monomials)
            + d*(d+1)//2 (symmetric 2nd-degree monomials: d diagonal
                          + d*(d-1)//2 off-diagonal)

    Example  d = 16:
        d_prime = 1 + 16 + (16 * 17 // 2) = 1 + 16 + 136 = 153

Memory strategy
---------------
    The outer product x ⊗ x has shape [B, H, T, d, d].
    Only the upper triangle (d*(d+1)//2 entries) is extracted via
    pre-registered buffer indices — no Python loops, fully vectorised.
"""

from __future__ import annotations

import math
import torch
import torch.nn as nn


class TaylorFeatureMap(nn.Module):
    """
    Deterministic 2nd-degree polynomial feature map for linearised attention.

    Maps each token vector x ∈ ℝᵈ →  φ(x) ∈ ℝᵈ′  where
        d_prime = 1 + d + d*(d+1)//2.

    Parameters
    ----------
    head_dim : int
        Dimensionality of each attention head (d).
    scale_input : bool
        When True, divides the input by √d before mapping — prevents
        polynomial components from growing with head_dim and keeps
        activations in a numerically stable range (recommended).

    Shape
    -----
    Input  : [B, H, T, d]
    Output : [B, H, T, d_prime]
    """

    def __init__(self, head_dim: int, scale_input: bool = True) -> None:
        super().__init__()
        self.d           = head_dim
        self.scale_input = scale_input
        self.d_prime     = 1 + head_dim + head_dim * (head_dim + 1) // 2

        # ── Precompute upper-triangular index pairs (incl. diagonal) ──────────
        # triu_rows[k], triu_cols[k]  →  (i, j) pair for the k-th quadratic
        # monomial, ordered row-major through the upper triangle.
        rows, cols = torch.triu_indices(head_dim, head_dim, offset=0)
        self.register_buffer("triu_rows", rows)   # (n_quad,)
        self.register_buffer("triu_cols", cols)   # (n_quad,)

        # ── Scale vector for the quadratic block ──────────────────────────────
        # Diagonal entries (i == j):  coefficient = 1  → monomial xᵢ²
        # Off-diagonal (i  < j):      coefficient = √2 → monomial √2·xᵢxⱼ
        # This ensures φ(x)ᵀφ(y) reconstructs (xᵀy)² exactly (see module doc).
        quad_scale = torch.where(
            rows == cols,
            torch.ones(len(rows)),
            torch.full((len(rows),), math.sqrt(2.0)),
        )
        self.register_buffer("quad_scale", quad_scale)  # (n_quad,)

    # ── Properties ────────────────────────────────────────────────────────────

    @property
    def input_dim(self) -> int:
        return self.d

    @property
    def output_dim(self) -> int:
        return self.d_prime

    # ── Forward pass ──────────────────────────────────────────────────────────

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : Tensor  [B, H, T, d]
            Projected Q or K block from a multi-head attention layer.

        Returns
        -------
        phi : Tensor  [B, H, T, d_prime]
            2nd-degree polynomial feature vectors, ready for linearised
            attention via  A ≈ φ(Q) · [φ(K)ᵀ V] / [φ(Q) · φ(K)ᵀ 1].
        """
        if x.dim() != 4:
            raise ValueError(
                f"TaylorFeatureMap expects a 4-D tensor [B, H, T, d], "
                f"got shape {tuple(x.shape)}."
            )
        B, H, T, d = x.shape
        if d != self.d:
            raise ValueError(
                f"head_dim mismatch: model was built with d={self.d}, "
                f"but input last dim is {d}."
            )

        # Optional input normalisation  (keeps polynomial magnitudes O(1))
        if self.scale_input:
            x = x / math.sqrt(d)

        # ── 1. Constant block ─────────────────────────────────────────────────
        # Shape: [B, H, T, 1]
        ones = x.new_ones(B, H, T, 1)

        # ── 2. Linear block ───────────────────────────────────────────────────
        # Shape: [B, H, T, d]   (identity — just the raw features)
        linear = x

        # ── 3. Quadratic block ────────────────────────────────────────────────
        # Outer product  x_outer[b,h,t,i,j] = x[b,h,t,i] * x[b,h,t,j]
        # Shape: [B, H, T, d, d]
        #
        # unsqueeze(-1) → [B, H, T, d, 1]
        # unsqueeze(-2) → [B, H, T, 1, d]
        # broadcast multiply → [B, H, T, d, d]
        x_outer = x.unsqueeze(-1) * x.unsqueeze(-2)

        # Extract upper triangle (row-major), scaled by quad_scale.
        # triu_rows / triu_cols are registered buffers → live on same device.
        # Result shape: [B, H, T, n_quad]  where n_quad = d*(d+1)//2
        quad = x_outer[..., self.triu_rows, self.triu_cols]   # [B,H,T,n_quad]
        quad = quad * self.quad_scale                          # broadcast (n_quad,)

        # ── 4. Concatenate all blocks ─────────────────────────────────────────
        # Output: [B, H, T, 1 + d + n_quad]  ≡  [B, H, T, d_prime]
        return torch.cat([ones, linear, quad], dim=-1)

    def __repr__(self) -> str:
        return (
            f"TaylorFeatureMap("
            f"d={self.d}, "
            f"d_prime={self.d_prime}, "
            f"scale_input={self.scale_input})"
        )


# ── Self-test ─────────────────────────────────────────────────────────────────

def _run_self_test() -> None:
    """
    Instantiate the model with d=16, pass a [2, 4, 1440, 16] tensor,
    and verify:
      1. Output shape is exactly [2, 4, 1440, 153].
      2. d_prime matches the closed-form formula.
      3. No NaN / Inf in the output.
      4. The inner-product identity φ(x)ᵀφ(y) ≈ 1 + xᵀy + (xᵀy)² holds
         for a small random pair.
    """
    SEP = "─" * 62
    print(SEP)
    print("  TaylorFeatureMap — self-test")
    print(SEP)

    d      = 16
    B, H, T = 2, 4, 1440

    # Expected output dimension (hard-coded for verification)
    expected_d_prime = 1 + d + (d * (d + 1) // 2)   # = 1 + 16 + 136 = 153
    print(f"\n  d            = {d}")
    print(f"  d_prime (formula) = 1 + {d} + ({d}*{d+1}//2) "
          f"= 1 + {d} + {d*(d+1)//2} = {expected_d_prime}")

    # ── Build model ───────────────────────────────────────────────────────────
    model = TaylorFeatureMap(head_dim=d, scale_input=True)
    print(f"\n  {model}")
    assert model.d_prime == expected_d_prime, (
        f"d_prime attribute mismatch: {model.d_prime} ≠ {expected_d_prime}"
    )
    print(f"  d_prime attribute matches formula  ✔")

    # ── Shape test ────────────────────────────────────────────────────────────
    torch.manual_seed(42)
    x = torch.randn(B, H, T, d)
    phi_x = model(x)

    expected_shape = torch.Size([B, H, T, expected_d_prime])
    assert phi_x.shape == expected_shape, (
        f"Output shape mismatch: {phi_x.shape} ≠ {expected_shape}"
    )
    print(f"\n  Input  shape : {list(x.shape)}")
    print(f"  Output shape : {list(phi_x.shape)}  ✔  [B × H × T × d_prime]")

    # ── NaN / Inf check ───────────────────────────────────────────────────────
    assert torch.isfinite(phi_x).all(), "Output contains NaN or Inf!"
    print(f"  All values finite  ✔")
    print(f"  Output  min={phi_x.min().item():.4f}  "
          f"max={phi_x.max().item():.4f}  "
          f"mean={phi_x.mean().item():.4f}")

    # ── Inner-product identity  φ(x)ᵀφ(y) = 1 + xᵀy + (xᵀy)² ──────────────
    # Use a single token pair for clarity (B=H=T=1).
    print(f"\n  Inner-product identity check (single token pair):")
    xa = torch.randn(1, 1, 1, d)
    xb = torch.randn(1, 1, 1, d)

    phi_a = model(xa)   # [1, 1, 1, d_prime]
    phi_b = model(xb)

    # Kernel value via feature map inner product
    k_approx = (phi_a * phi_b).sum(dim=-1).item()

    # Ground-truth: 1 + xᵀy + (xᵀy)²  (using scaled inputs as model does)
    xa_s = xa / math.sqrt(d)
    xb_s = xb / math.sqrt(d)
    dot   = (xa_s * xb_s).sum().item()
    k_ref = 1.0 + dot + dot ** 2

    err = abs(k_approx - k_ref)
    print(f"    φ(x)ᵀφ(y)           = {k_approx:.8f}")
    print(f"    1 + xᵀy + (xᵀy)²   = {k_ref:.8f}")
    print(f"    absolute error       = {err:.2e}  ", end="")
    assert err < 1e-5, f"Inner-product identity failed! err={err:.2e}"
    print("✔")

    # ── Memory footprint estimate ─────────────────────────────────────────────
    # Outer product intermediate: [B, H, T, d, d] float32
    outer_bytes = B * H * T * d * d * 4
    out_bytes   = B * H * T * expected_d_prime * 4
    print(f"\n  Peak intermediate  (outer product) : "
          f"{outer_bytes / 1e6:.2f} MB  [B×H×T×d×d float32]")
    print(f"  Output tensor size                 : "
          f"{out_bytes / 1e6:.2f} MB  [B×H×T×d_prime float32]")

    print()
    print(SEP)
    print("  All assertions PASSED  ✔")
    print(SEP + "\n")


if __name__ == "__main__":
    import math   # already imported at module level; explicit for clarity
    _run_self_test()
