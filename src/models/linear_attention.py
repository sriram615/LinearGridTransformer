"""
linear_attention.py
-------------------
Foundational Core Deep Learning Architect — LinearGridTransformer project.

Implements LinearKernelAttention: a drop-in multi-head attention module whose
computational complexity scales at O(T) in sequence length rather than the
standard O(T²) of softmax attention.

Theory
------
Standard softmax attention computes:

    Attn(Q, K, V) = softmax(QKᵀ / √d) · V

which requires materialising an [N × N] score matrix.  We instead approximate
the exponential kernel via the 2nd-degree Taylor feature map φ:

    exp(qᵀk) ≈ φ(q)ᵀ φ(k)

This lets us factor the attention as:

    Output[t] = φ(q_t)ᵀ · (Σₜ' φ(k_t') ⊗ v_t') / (φ(q_t)ᵀ · Σₜ' φ(k_t'))

Associativity trick (O(T) cost)
--------------------------------
Rather than expanding Σ φ(k_t')ᵀ φ(q_t) for each query (O(T²)), we group:

    KV_state = Σ_t  φ(K_t)ᵀ ⊗ V_t     →  shape [B, H, d_prime, d_v]
    Context  = φ(Q) · KV_state          →  shape [B, H, T, d_v]

Both einsum contractions are O(T · d_prime · d_v), independent of T² — the
NxN score matrix never exists in memory.

Normalisation
-------------
Each query's output is divided by its per-row denominator:

    D_t = φ(q_t)ᵀ · (Σₜ' φ(k_t'))

which is the partition function of the approximated softmax, ensuring the
attention weights sum to ~1 per query token.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn

# Import the Taylor kernel from sibling module
import sys
import pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
from src.models.kernel import TaylorFeatureMap


class LinearKernelAttention(nn.Module):
    """
    Multi-head linearised attention using the 2nd-degree Taylor feature map.

    Replaces the O(T²) softmax attention with an O(T) associative formulation.
    No [T × T] score matrix is ever computed or stored.

    Parameters
    ----------
    head_dim  : int   — dimensionality of each attention head (d_v = head_dim).
    num_heads : int   — number of parallel attention heads.
    scale_input : bool — passed through to TaylorFeatureMap (divide by √d
                        before mapping; strongly recommended).
    eps : float       — safety scalar added to the denominator to prevent
                        division by zero (default 1e-6).

    Shapes  (all tensors are pre-projected; this module does NOT contain
             the input linear projections W_Q / W_K / W_V)
    ------
    Q, K, V input  : [B, H, T, head_dim]
    Output         : [B, H, T, head_dim]

    Complexity
    ----------
    Time   : O(T · d_prime · d_v)   — linear in T
    Memory : O(B · H · d_prime · d_v)  for KV_state (no T² term)
    """

    def __init__(
        self,
        head_dim:    int,
        num_heads:   int,
        scale_input: bool  = True,
        eps:         float = 1e-6,
    ) -> None:
        super().__init__()

        self.head_dim  = head_dim
        self.num_heads = num_heads
        self.eps       = eps

        # One shared feature map (stateless except for index buffers).
        # Buffers auto-migrate with .to(device) — no extra bookkeeping needed.
        self.phi = TaylorFeatureMap(head_dim=head_dim, scale_input=scale_input)
        self.d_prime = self.phi.d_prime   # 1 + d + d*(d+1)//2

    # ── Core attention logic ───────────────────────────────────────────────────

    def forward(
        self,
        Q: torch.Tensor,
        K: torch.Tensor,
        V: torch.Tensor,
    ) -> Tuple[torch.Tensor, dict]:
        """
        Compute linearised kernel attention.

        Parameters
        ----------
        Q : Tensor  [B, H, T, d]   — query projections
        K : Tensor  [B, H, T, d]   — key   projections
        V : Tensor  [B, H, T, d_v] — value projections  (d_v == head_dim here)

        Returns
        -------
        output : Tensor  [B, H, T, d_v]
            Normalised attention output for each query position.
        debug  : dict
            Metadata tensors for inspection:
                'Q_phi'      [B, H, T, d_prime]
                'K_phi'      [B, H, T, d_prime]
                'KV_state'   [B, H, d_prime, d_v]
                'K_phi_sum'  [B, H, d_prime]
                'denominator'[B, H, T]
        """
        self._validate_inputs(Q, K, V)
        B, H, T, d = Q.shape
        d_v = V.shape[-1]

        # ── Step 1: Apply feature map to Q and K ─────────────────────────────
        # Both calls to phi share the same stateless map; no weight update.
        #
        # Q_phi : [B, H, T, d_prime]
        # K_phi : [B, H, T, d_prime]
        Q_phi: torch.Tensor = self.phi(Q)   # φ(Q)
        K_phi: torch.Tensor = self.phi(K)   # φ(K)

        # ── Step 2: KV memory state — O(d_prime · d_v) per token ─────────────
        #
        # KV_state[b,h,p,v] = Σ_t  K_phi[b,h,t,p] · V[b,h,t,v]
        #
        # einsum legend:
        #   b = batch,  h = head,  t = time,
        #   p = d_prime feature index,  v = value dim
        #
        # This is the "compressed memory" of all key-value pairs.
        # Shape: [B, H, d_prime, d_v]
        #
        # Cost: O(T · d_prime · d_v)  — linear in T
        # Memory: O(B · H · d_prime · d_v) — NO T² term ever allocated
        KV_state: torch.Tensor = torch.einsum("bhtp, bhtv -> bhpv", K_phi, V)

        # ── Step 3: Context — retrieve for every query in parallel ────────────
        #
        # Context[b,h,t,v] = Σ_p  Q_phi[b,h,t,p] · KV_state[b,h,p,v]
        #
        # Shape: [B, H, T, d_v]
        # Cost: O(T · d_prime · d_v)  — linear in T
        Context: torch.Tensor = torch.einsum("bhtp, bhpv -> bhtv", Q_phi, KV_state)

        # ── Step 4: Normalisation denominator ─────────────────────────────────
        #
        # The partition function for the approximated softmax:
        #   D_t = φ(q_t)ᵀ · Σ_{t'} φ(k_{t'})
        #
        # Step 4a: Sum K_phi over the time axis → compressed key mass
        #          K_phi_sum[b,h,p] = Σ_t  K_phi[b,h,t,p]
        #          Shape: [B, H, d_prime]
        K_phi_sum: torch.Tensor = K_phi.sum(dim=2)   # Σ_t φ(K)

        # Step 4b: Inner product of each Q feature vector with the key mass
        #          D[b,h,t] = Σ_p  Q_phi[b,h,t,p] · K_phi_sum[b,h,p]
        #          Shape: [B, H, T]
        # unsqueeze(-1) to broadcast [B,H,d_prime] over T for einsum clarity:
        denominator: torch.Tensor = torch.einsum(
            "bhtp, bhp -> bht", Q_phi, K_phi_sum
        )

        # ── Step 5: Safe normalisation ────────────────────────────────────────
        #
        # Divide context by denominator (with ε guard against zero).
        # unsqueeze(-1) broadcasts [B,H,T] → [B,H,T,1] to divide [B,H,T,d_v].
        output: torch.Tensor = Context / (denominator.unsqueeze(-1) + self.eps)

        debug = {
            "Q_phi":       Q_phi,
            "K_phi":       K_phi,
            "KV_state":    KV_state,
            "K_phi_sum":   K_phi_sum,
            "denominator": denominator,
        }
        return output, debug

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _validate_inputs(
        self,
        Q: torch.Tensor,
        K: torch.Tensor,
        V: torch.Tensor,
    ) -> None:
        for name, t in [("Q", Q), ("K", K), ("V", V)]:
            if t.dim() != 4:
                raise ValueError(
                    f"{name} must be 4-D [B, H, T, d], got shape {tuple(t.shape)}."
                )
        B, H, T, d = Q.shape
        if d != self.head_dim:
            raise ValueError(
                f"Q last dim ({d}) ≠ head_dim ({self.head_dim})."
            )
        if K.shape != Q.shape:
            raise ValueError(
                f"K shape {tuple(K.shape)} must match Q shape {tuple(Q.shape)}."
            )
        if V.shape[:3] != Q.shape[:3]:
            raise ValueError(
                f"V first 3 dims {tuple(V.shape[:3])} must match Q {tuple(Q.shape[:3])}."
            )
        if H != self.num_heads:
            raise ValueError(
                f"Number of heads in input ({H}) ≠ num_heads ({self.num_heads})."
            )

    def complexity_report(self, B: int, T: int) -> dict:
        """Return theoretical FLOPs and memory estimates for a given batch/length."""
        d, d_p, d_v = self.head_dim, self.d_prime, self.head_dim
        H = self.num_heads
        phi_flops   = T * (1 + d + d * d)          # per head: triu outer prod
        kv_flops    = T * d_p * d_v                  # einsum bhtp,bhtv->bhpv
        ctx_flops   = T * d_p * d_v                  # einsum bhtp,bhpv->bhtv
        denom_flops = T * d_p                         # einsum bhtp,bhp->bht
        total_flops = H * B * (2 * phi_flops + kv_flops + ctx_flops + denom_flops)

        kv_bytes    = B * H * d_p * d_v * 4          # float32, no T
        return {
            "O_complexity": "O(T)",
            "total_flops":  total_flops,
            "KV_state_MB":  kv_bytes / 1e6,
            "T_quadratic_term": "absent — [T×T] matrix never allocated",
        }

    def __repr__(self) -> str:
        return (
            f"LinearKernelAttention("
            f"head_dim={self.head_dim}, "
            f"num_heads={self.num_heads}, "
            f"d_prime={self.d_prime}, "
            f"eps={self.eps})"
        )


# ── Integration test ─────────────────────────────────────────────────────────

def _run_test() -> None:
    SEP = "─" * 66

    print(SEP)
    print("  LinearKernelAttention — integration test")
    print(SEP)

    # Configuration matching the project's dataset output
    B, H, T, d = 2, 4, 1440, 16

    torch.manual_seed(0)
    Q = torch.randn(B, H, T, d)
    K = torch.randn(B, H, T, d)
    V = torch.randn(B, H, T, d)

    # ── Build module ──────────────────────────────────────────────────────────
    model = LinearKernelAttention(head_dim=d, num_heads=H)
    print(f"\n  {model}")
    print(f"  Feature map d_prime = {model.d_prime}")

    # ── Forward pass ──────────────────────────────────────────────────────────
    import time
    t0 = time.perf_counter()
    output, debug = model(Q, K, V)
    elapsed_ms = (time.perf_counter() - t0) * 1000

    print(f"\n  Forward pass completed in {elapsed_ms:.1f} ms")

    # ── Shape verification ────────────────────────────────────────────────────
    print(f"\n  Intermediate tensor shapes:")
    print(f"    Q_phi        {list(debug['Q_phi'].shape)}"
          f"  ← [B, H, T, d_prime]")
    print(f"    K_phi        {list(debug['K_phi'].shape)}"
          f"  ← [B, H, T, d_prime]")
    print(f"    KV_state     {list(debug['KV_state'].shape)}"
          f"  ← [B, H, d_prime, d_v]  (NO T dim — O(d²) memory)")
    print(f"    K_phi_sum    {list(debug['K_phi_sum'].shape)}"
          f"  ← [B, H, d_prime]")
    print(f"    denominator  {list(debug['denominator'].shape)}"
          f"  ← [B, H, T]")
    print(f"    output       {list(output.shape)}"
          f"  ← [B, H, T, d_v]")

    # Strict shape assertions
    assert output.shape == torch.Size([B, H, T, d]), \
        f"Output shape wrong: {output.shape}"
    assert debug['KV_state'].shape  == torch.Size([B, H, model.d_prime, d])
    assert debug['K_phi_sum'].shape == torch.Size([B, H, model.d_prime])
    assert debug['denominator'].shape == torch.Size([B, H, T])
    print(f"\n  All shape assertions PASSED  ✔")

    # ── Numerical checks ──────────────────────────────────────────────────────
    assert torch.isfinite(output).all(), "Output contains NaN / Inf!"
    print(f"  Output is finite  ✔")

    denom = debug['denominator']
    assert (denom > 0).all(), "Denominator has non-positive entries!"
    print(f"  Denominator strictly positive  ✔  "
          f"(min={denom.min().item():.4f})")

    # ── Complexity report ─────────────────────────────────────────────────────
    report = model.complexity_report(B=B, T=T)
    print(f"\n  Complexity report:")
    for k, v in report.items():
        print(f"    {k:<28}: {v}")

    # ── Memory footprint ──────────────────────────────────────────────────────
    out_mb  = output.numel()  * 4 / 1e6
    kv_mb   = debug['KV_state'].numel() * 4 / 1e6
    phi_mb  = debug['Q_phi'].numel()    * 4 / 1e6
    print(f"\n  Runtime tensor sizes (float32):")
    print(f"    Q_phi / K_phi  each : {phi_mb:.2f} MB   [B×H×T×d_prime]")
    print(f"    KV_state            : {kv_mb:.4f} MB   [B×H×d_prime×d_v]  "
          f"← constant in T")
    print(f"    output              : {out_mb:.2f} MB   [B×H×T×d_v]")
    print(f"    [T×T] score matrix  : NEVER ALLOCATED  ✔")

    print()
    print(SEP)
    print("  All assertions PASSED  ✔")
    print(SEP + "\n")


if __name__ == "__main__":
    _run_test()
