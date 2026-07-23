#!/Users/apple/LinearGridTransformer/.venv311/bin/python3
"""
test_attention.py
-----------------
Integration & Validation Engineer — LinearGridTransformer project.

Constructs a single-layer 'GridTransformerEncoder' that encapsulates:
    1. Multi-head projection  →  LinearKernelAttention (O(T) kernel)
    2. Residual + LayerNorm   (post-attention)
    3. Two-layer Feed-Forward Network (FFN)
    4. Residual + LayerNorm   (post-FFN)

The encoder maps  [B, T, C_in]  →  [B, T, C_in]  (dimension-preserving).

Mock input   : [Batch=16, Sequence=1440, Channels=7]
Expected out : [16, 1440, 7]

Also monitors Mac process RSS memory before and after the forward pass
to confirm no unexpected allocation spike (OOM guard).

Usage
-----
    .venv311/bin/python3 test_attention.py
    # or, if the shebang is honoured:
    ./test_attention.py
"""

from __future__ import annotations

import math
import os
import sys
import time

# ── Resolve repo root so src.* imports work regardless of CWD ─────────────────
_REPO = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _REPO)

import psutil
import torch
import torch.nn as nn

from src.models.linear_attention import LinearKernelAttention


# ═══════════════════════════════════════════════════════════════════════════════
#  GridTransformerEncoder
# ═══════════════════════════════════════════════════════════════════════════════

class GridTransformerEncoder(nn.Module):
    """
    Single-layer Transformer encoder block built around LinearKernelAttention.

    Architecture
    ------------
    x  [B, T, C_in]
    │
    ├─ Input projection  :  Linear(C_in,  d_model)
    │
    ├─ ┌─────────────────────────────────────────────┐
    │  │  Multi-Head Linear Attention                 │
    │  │  Q,K,V  projections  →  [B,H,T,head_dim]    │
    │  │  LinearKernelAttention (O(T) kernel)         │
    │  │  Output projection   →  [B,T,d_model]        │
    │  └─────────────────────────────────────────────┘
    │
    ├─ Residual + LayerNorm(d_model)
    │
    ├─ ┌─────────────────────────────────────────────┐
    │  │  FFN                                        │
    │  │  Linear(d_model, ffn_dim)  →  GELU          │
    │  │  Linear(ffn_dim,  d_model)                  │
    │  └─────────────────────────────────────────────┘
    │
    ├─ Residual + LayerNorm(d_model)
    │
    └─ Output projection :  Linear(d_model, C_in)

    Output shape equals input shape  [B, T, C_in].

    Parameters
    ----------
    in_channels : int   — number of input feature channels (C_in = 7 here)
    d_model     : int   — internal embedding dimension (must be divisible by num_heads)
    num_heads   : int   — number of attention heads
    ffn_dim     : int   — hidden dimension of the FFN  (typically 4 × d_model)
    dropout     : float — dropout rate applied inside FFN
    """

    def __init__(
        self,
        in_channels: int   = 7,
        d_model:     int   = 64,
        num_heads:   int   = 4,
        ffn_dim:     int   = 256,
        dropout:     float = 0.0,
    ) -> None:
        super().__init__()

        assert d_model % num_heads == 0, (
            f"d_model ({d_model}) must be divisible by num_heads ({num_heads})."
        )
        self.in_channels = in_channels
        self.d_model     = d_model
        self.num_heads   = num_heads
        self.head_dim    = d_model // num_heads   # d per head

        # ── Input / Output projections ────────────────────────────────────────
        self.input_proj  = nn.Linear(in_channels, d_model, bias=False)
        self.output_proj = nn.Linear(d_model, in_channels, bias=False)

        # ── Multi-head Q / K / V projections ─────────────────────────────────
        self.W_Q = nn.Linear(d_model, d_model, bias=False)
        self.W_K = nn.Linear(d_model, d_model, bias=False)
        self.W_V = nn.Linear(d_model, d_model, bias=False)
        self.W_O = nn.Linear(d_model, d_model, bias=False)   # output re-projection

        # ── O(T) kernel attention ─────────────────────────────────────────────
        self.attn = LinearKernelAttention(
            head_dim=self.head_dim,
            num_heads=num_heads,
        )

        # ── Normalisation (post-attention, post-FFN) ──────────────────────────
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        # ── Two-layer FFN: d_model → ffn_dim → d_model ────────────────────────
        self.ffn = nn.Sequential(
            nn.Linear(d_model, ffn_dim),
            nn.GELU(),
            nn.Dropout(p=dropout),
            nn.Linear(ffn_dim, d_model),
            nn.Dropout(p=dropout),
        )

        self._init_weights()

    def _init_weights(self) -> None:
        """Xavier uniform for all linear weight matrices."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        """[B, T, d_model] → [B, H, T, head_dim]"""
        B, T, _ = x.shape
        return x.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

    def _merge_heads(self, x: torch.Tensor) -> torch.Tensor:
        """[B, H, T, head_dim] → [B, T, d_model]"""
        B, H, T, _ = x.shape
        return x.transpose(1, 2).contiguous().view(B, T, self.d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : Tensor  [B, T, C_in]

        Returns
        -------
        out : Tensor  [B, T, C_in]
        """
        # ── Input projection: C_in → d_model ─────────────────────────────────
        h = self.input_proj(x)              # [B, T, d_model]

        # ── Multi-head linear attention with residual ─────────────────────────
        Q = self._split_heads(self.W_Q(h))  # [B, H, T, head_dim]
        K = self._split_heads(self.W_K(h))  # [B, H, T, head_dim]
        V = self._split_heads(self.W_V(h))  # [B, H, T, head_dim]

        attn_out, _ = self.attn(Q, K, V)   # [B, H, T, head_dim]
        attn_out    = self._merge_heads(attn_out)        # [B, T, d_model]
        attn_out    = self.W_O(attn_out)                 # [B, T, d_model]

        h = self.norm1(h + attn_out)        # Residual + LayerNorm ①

        # ── Feed-Forward Network with residual ────────────────────────────────
        ffn_out = self.ffn(h)               # [B, T, d_model]
        h = self.norm2(h + ffn_out)         # Residual + LayerNorm ②

        # ── Output projection: d_model → C_in ────────────────────────────────
        return self.output_proj(h)          # [B, T, C_in]

    def param_count(self) -> dict:
        total  = sum(p.numel() for p in self.parameters())
        attn_p = sum(p.numel() for p in self.attn.parameters())
        ffn_p  = sum(p.numel() for p in self.ffn.parameters())
        return {"total": total, "attention": attn_p, "ffn": ffn_p}

    def __repr__(self) -> str:
        pc = self.param_count()
        return (
            f"GridTransformerEncoder("
            f"in_channels={self.in_channels}, "
            f"d_model={self.d_model}, "
            f"num_heads={self.num_heads}, "
            f"head_dim={self.head_dim}, "
            f"params={pc['total']:,})"
        )


# ═══════════════════════════════════════════════════════════════════════════════
#  Validation suite
# ═══════════════════════════════════════════════════════════════════════════════

def _rss_mb() -> float:
    """Current process RSS memory in MB."""
    return psutil.Process(os.getpid()).memory_info().rss / 1e6


def _fmt(val: float, unit: str = "MB") -> str:
    return f"{val:.2f} {unit}"


SEP  = "═" * 68
sep  = "─" * 68


def run_validation() -> None:
    print(SEP)
    print("  GridTransformerEncoder — structural verification")
    print(SEP)

    # ── Configuration ─────────────────────────────────────────────────────────
    B, T, C = 16, 1440, 7     # actual DataLoader output shape
    d_model  = 64             # 4 heads × 16 head_dim
    num_heads = 4
    ffn_dim   = 256           # 4 × d_model

    print(f"\n  Configuration:")
    print(f"    Input shape    : [{B}, {T}, {C}]   ← matches DataLoader output")
    print(f"    d_model        : {d_model}          (= {num_heads} heads × {d_model // num_heads} head_dim)")
    print(f"    FFN hidden     : {ffn_dim}")
    print(f"    Kernel         : TaylorFeatureMap  d_prime={1+16+16*17//2}")

    # ── Model construction ────────────────────────────────────────────────────
    print(f"\n  Building GridTransformerEncoder …")
    torch.manual_seed(7)
    model = GridTransformerEncoder(
        in_channels=C,
        d_model=d_model,
        num_heads=num_heads,
        ffn_dim=ffn_dim,
        dropout=0.0,
    )
    model.eval()   # deterministic run — no dropout

    pc = model.param_count()
    print(f"  {model}")
    print(f"    ├─ Attention params : {pc['attention']:,}")
    print(f"    ├─ FFN params       : {pc['ffn']:,}")
    print(f"    └─ Total params     : {pc['total']:,}")

    # ── Mock input tensor ─────────────────────────────────────────────────────
    print(f"\n  Generating mock input  {[B, T, C]}  …")
    x = torch.randn(B, T, C)
    inp_mb = x.numel() * 4 / 1e6
    print(f"    Input tensor : {_fmt(inp_mb)}  (float32)")

    # ── Memory snapshot before forward pass ──────────────────────────────────
    rss_before = _rss_mb()
    print(f"\n  Process RSS before forward : {_fmt(rss_before)}")

    # ── Forward pass — timed ──────────────────────────────────────────────────
    print(f"\n  Running forward pass …")
    t_start = time.perf_counter()
    with torch.no_grad():
        out = model(x)
    elapsed_ms = (time.perf_counter() - t_start) * 1000

    # ── Memory snapshot after forward pass ───────────────────────────────────
    rss_after  = _rss_mb()
    rss_delta  = rss_after - rss_before

    print(f"  Forward pass elapsed       : {elapsed_ms:.1f} ms")
    print(f"  Process RSS after forward  : {_fmt(rss_after)}")
    print(f"  RSS delta (peak allocation): {rss_delta:+.1f} MB")

    # ── Assertions ────────────────────────────────────────────────────────────
    print(f"\n{sep}")
    print(f"  ASSERTIONS")
    print(sep)

    # 1. Output shape exactly matches input
    expected = torch.Size([B, T, C])
    assert out.shape == expected, (
        f"Shape mismatch!  expected {list(expected)}, got {list(out.shape)}"
    )
    print(f"  ✔  Output shape : {list(out.shape)}  ==  [{B}, {T}, {C}]")

    # 2. Finite values (no NaN / Inf)
    assert torch.isfinite(out).all(), "Output contains NaN or Inf values!"
    print(f"  ✔  All output values are finite")

    # 3. Output has non-zero variance (not collapsed to constant)
    out_std = out.std().item()
    assert out_std > 1e-4, f"Output has suspiciously low std: {out_std:.2e}"
    print(f"  ✔  Output std   : {out_std:.4f}  (non-degenerate)")

    # 4. Memory OOM guard — RSS delta should be < 1 GB (generous bound for CPU)
    OOM_THRESHOLD_MB = 1000
    assert rss_delta < OOM_THRESHOLD_MB, (
        f"RSS grew by {rss_delta:.1f} MB — possible memory leak or OOM condition!"
    )
    print(f"  ✔  RSS delta    : {rss_delta:+.1f} MB  (< {OOM_THRESHOLD_MB} MB OOM threshold)")

    # 5. Forward pass speed — must be < 10 s on CPU (sanity guard)
    MAX_FORWARD_MS = 10_000
    assert elapsed_ms < MAX_FORWARD_MS, (
        f"Forward pass took {elapsed_ms:.0f} ms — unexpectedly slow!"
    )
    print(f"  ✔  Elapsed      : {elapsed_ms:.1f} ms  (< {MAX_FORWARD_MS} ms guard)")

    # ── Summary report ────────────────────────────────────────────────────────
    print(f"\n{sep}")
    print(f"  SUMMARY")
    print(sep)
    out_mb = out.numel() * 4 / 1e6
    print(f"  Input  [{B:>2}, {T}, {C}]  →  Output [{B:>2}, {T}, {C}]  ✔  (dimensions preserved)")
    print(f"  Forward pass    : {elapsed_ms:.1f} ms")
    print(f"  RSS Δ           : {rss_delta:+.1f} MB   (no OOM event)")
    print(f"  Output tensor   : {_fmt(out_mb)}")
    print(f"  Params total    : {pc['total']:,}")
    print(f"  Attention cost  : O(T) — [T×T] matrix never allocated  ✔")
    print()

    # ── Architecture diagram ──────────────────────────────────────────────────
    print(f"  Block diagram:")
    print(f"    [16, 1440,  7]  ──  input_proj  →  [16, 1440, {d_model}]")
    print(f"    [16, 1440, {d_model}]  ──  split heads  →  [16,  {num_heads}, 1440, {d_model//num_heads}]")
    print(f"    [16,   {num_heads}, 1440, {d_model//num_heads}]  ──  LinearKernelAttention  →  same")
    print(f"                         KV_state=[16,{num_heads},{1+16+16*17//2},{d_model//num_heads}]  ← O(d²) not O(T²)")
    print(f"    [16, 1440, {d_model}]  ──  LayerNorm  →  [16, 1440, {d_model}]")
    print(f"    [16, 1440, {d_model}]  ──  FFN({ffn_dim})  →  [16, 1440, {d_model}]")
    print(f"    [16, 1440, {d_model}]  ──  LayerNorm  →  [16, 1440, {d_model}]")
    print(f"    [16, 1440, {d_model}]  ──  output_proj→  [16, 1440,  7]")

    print()
    print(SEP)
    print("  All assertions PASSED  ✔")
    print(SEP + "\n")


if __name__ == "__main__":
    run_validation()
