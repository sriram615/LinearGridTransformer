"""
dlinear.py
----------
Principal Deep Learning Infrastructure Engineer — LinearGridTransformer project.

Implements a self-contained DLinear baseline model for direct comparison
against the LinearGridTransformer.

Reference
---------
Zeng et al., "Are Transformers Effective for Time Series Forecasting?" (2023)
https://arxiv.org/abs/2205.13504

Architecture
------------
DLinear decomposes a multivariate time-series into a *trend* component and a
*seasonal* (residual) component via a moving-average filter, then independently
maps each component from the lookback window to the forecast horizon using a
separate set of per-channel linear layers.

    x  [B, T, C]
    │
    ├─ SeriesDecomposition(kernel_size)
    │     ├─ trend    [B, T, C]   — moving-average smoothed signal
    │     └─ seasonal [B, T, C]   — x − trend  (residual)
    │
    ├─ trend_layers    : ModuleList of C × Linear(T → pred_len)   (channel-wise)
    ├─ seasonal_layers : ModuleList of C × Linear(T → pred_len)   (channel-wise)
    │
    └─ output = trend_pred + seasonal_pred                [B, pred_len, C]

Channel-Independent Design
--------------------------
A separate nn.Linear is used for each of the C channels in both the trend and
seasonal branches.  This deliberate architecture prevents cross-channel
contamination: the model evaluates each physical variable (e.g. Voltage in
Volts, active power in kW, sub-metering in Wh) as a fully isolated forecasting
problem.  This mirrors the channel-independence finding in the original paper
and makes the baseline a rigorous, fair comparison point against the
LinearGridTransformer's multi-head attention (which is also channel-aware via
independent head projections).

Boundary Handling
-----------------
nn.AvgPool1d with kernel_size k and stride 1 naturally shortens the sequence:
    L_out = L_in − k + 1

SeriesDecomposition restores the original length T using edge-padding
duplication before pooling (torch.nn.functional.pad with 'replicate' mode),
so L_in → L_out = L_in always.  This ensures the decomposition is a true
identity-preserving split:
    x = trend + seasonal

"""

from __future__ import annotations

import math
import sys
import pathlib

import torch
import torch.nn as nn
import torch.nn.functional as F


# ══════════════════════════════════════════════════════════════════════════════
#  SeriesDecomposition
# ══════════════════════════════════════════════════════════════════════════════

class SeriesDecomposition(nn.Module):
    """
    Moving-average trend–seasonal decomposition.

    Applies a 1-D average-pool filter along the time axis to extract a smooth
    trend signal, then derives the seasonal component as the point-wise
    residual.

    Boundary handling
    -----------------
    nn.AvgPool1d(kernel_size=k, stride=1) shortens a length-T sequence to
    T − k + 1.  To preserve the original length we replicate-pad both ends
    before pooling:
        • left  padding  = k // 2
        • right padding  = k − 1 − k // 2      (= k // 2 for odd k)

    Total padded length: T + k − 1  →  after pool: T.

    For even k the asymmetry shifts the moving average by half a step to the
    right, which is the standard practice in the DLinear literature.

    Parameters
    ----------
    kernel_size : int
        Length of the averaging window (default 25 — same as original paper).
        Odd values produce a symmetric filter; even values produce a
        right-shifted one.  Must be ≥ 1.

    Input / Output
    --------------
    x        : Tensor  [B, T, C]
    trend    : Tensor  [B, T, C]   — low-frequency component
    seasonal : Tensor  [B, T, C]   — high-frequency residual (x − trend)
    """

    def __init__(self, kernel_size: int = 25) -> None:
        super().__init__()
        if kernel_size < 1:
            raise ValueError(
                f"kernel_size must be ≥ 1, got {kernel_size}."
            )
        self.kernel_size = kernel_size

        # Stride=1 keeps every position; no learnable parameters.
        self.avg_pool = nn.AvgPool1d(kernel_size=kernel_size, stride=1, padding=0)

        # Pre-compute asymmetric pad widths (work for both odd and even k)
        self._pad_left  = kernel_size // 2
        self._pad_right = kernel_size - 1 - self._pad_left

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        x : Tensor  [B, T, C]

        Returns
        -------
        trend    : Tensor  [B, T, C]
        seasonal : Tensor  [B, T, C]   (= x − trend)
        """
        # ── AvgPool1d expects [B, C, T] — transpose ─────────────────────────
        # x: [B, T, C]  →  x_t: [B, C, T]
        x_t = x.transpose(1, 2)    # [B, C, T]

        # ── Edge-replicate pad to preserve sequence length ───────────────────
        # F.pad pads the last dim: (left, right) applied to dim=-1 (T axis)
        # 'replicate' copies the edge value — avoids zero-injection artifacts
        # at boundaries that would distort the trend signal.
        x_padded = F.pad(x_t, (self._pad_left, self._pad_right), mode="replicate")
        # x_padded shape: [B, C, T + kernel_size − 1]

        # ── Average pool → trend ─────────────────────────────────────────────
        trend_t = self.avg_pool(x_padded)   # [B, C, T]

        # ── Transpose back to [B, T, C] ──────────────────────────────────────
        trend    = trend_t.transpose(1, 2)  # [B, T, C]

        # ── Seasonal residual ─────────────────────────────────────────────────
        seasonal = x - trend                # [B, T, C]

        return trend, seasonal

    def __repr__(self) -> str:
        return (
            f"SeriesDecomposition("
            f"kernel_size={self.kernel_size}, "
            f"pad_left={self._pad_left}, "
            f"pad_right={self._pad_right})"
        )


# ══════════════════════════════════════════════════════════════════════════════
#  DLinearBaseline
# ══════════════════════════════════════════════════════════════════════════════

class DLinearBaseline(nn.Module):
    """
    DLinear: Decomposition + channel-independent Linear forecaster.

    Each of the C physical channels is forecast independently using a pair of
    dedicated Linear layers (one for trend, one for seasonal), preventing
    cross-channel contamination between physically distinct variables such as
    Voltage (V) and Sub-metering (Wh).

    Parameters
    ----------
    seq_len   : int   — input lookback window length  (default 1440 = 24 h)
    pred_len  : int   — forecast horizon length       (default   60 =  1 h)
    channels  : int   — number of feature channels   (default    7)
    kernel_size : int — moving-average window for decomposition (default 25)

    Shapes
    ------
    Input  : [B, seq_len,  channels]
    Output : [B, pred_len, channels]

    Parameter count
    ---------------
    trend_layers    : channels × (seq_len × pred_len + pred_len bias)
    seasonal_layers : channels × (seq_len × pred_len + pred_len bias)
    decomposition   : 0  (stateless avg-pool filter)

    For the default config (seq_len=1440, pred_len=60, channels=7):
        Each Linear: 1440 × 60 + 60 = 86,460 parameters
        Total: 2 × 7 × 86,460 = 1,210,440 parameters
    """

    def __init__(
        self,
        seq_len:     int = 1440,
        pred_len:    int = 60,
        channels:    int = 7,
        kernel_size: int = 25,
    ) -> None:
        super().__init__()

        self.seq_len    = seq_len
        self.pred_len   = pred_len
        self.channels   = channels

        # ── Decomposition block ───────────────────────────────────────────────
        self.decomposition = SeriesDecomposition(kernel_size=kernel_size)

        # ── Channel-independent linear layers ─────────────────────────────────
        # ModuleList ensures all layers are registered as proper submodules so
        # their parameters appear in model.parameters() and state_dict().
        #
        # trend_layers[c]    maps the trend    sequence of channel c:  T → pred_len
        # seasonal_layers[c] maps the seasonal sequence of channel c:  T → pred_len
        #
        # Using bias=True (default) provides a per-channel learned offset —
        # important because each physical variable has a different mean level.
        self.trend_layers = nn.ModuleList(
            [nn.Linear(seq_len, pred_len) for _ in range(channels)]
        )
        self.seasonal_layers = nn.ModuleList(
            [nn.Linear(seq_len, pred_len) for _ in range(channels)]
        )

        self._init_weights()

    def _init_weights(self) -> None:
        """
        Xavier uniform initialisation for all Linear weight matrices;
        biases zeroed.  Consistent with the rest of the LinearGridTransformer
        project's initialisation convention.
        """
        for layer in (*self.trend_layers, *self.seasonal_layers):
            nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : Tensor  [B, seq_len, channels]

        Returns
        -------
        out : Tensor  [B, pred_len, channels]
            Sum of per-channel trend and seasonal linear predictions.
        """
        # ── Step 1: Decompose ─────────────────────────────────────────────────
        trend, seasonal = self.decomposition(x)
        # trend    : [B, T, C]
        # seasonal : [B, T, C]

        # ── Step 2: Channel-independent linear projection ─────────────────────
        # For each channel c, extract the time series [B, T], apply Linear,
        # and stack the results back to [B, pred_len, C].
        #
        # trend[..., c]    : [B, T]
        # trend_layers[c]  : Linear(T → pred_len)
        # output per ch    : [B, pred_len]
        #
        # torch.stack over the channel dimension gives [B, pred_len, C].
        trend_pred = torch.stack(
            [self.trend_layers[c](trend[..., c]) for c in range(self.channels)],
            dim=-1,                              # stack on channel dim
        )   # [B, pred_len, C]

        seasonal_pred = torch.stack(
            [self.seasonal_layers[c](seasonal[..., c]) for c in range(self.channels)],
            dim=-1,
        )   # [B, pred_len, C]

        # ── Step 3: Combine ────────────────────────────────────────────────────
        return trend_pred + seasonal_pred        # [B, pred_len, C]

    def param_count(self) -> dict:
        """Return parameter count breakdown."""
        trend_p    = sum(p.numel() for l in self.trend_layers    for p in l.parameters())
        seasonal_p = sum(p.numel() for l in self.seasonal_layers for p in l.parameters())
        decomp_p   = 0  # AvgPool1d has no parameters
        total_p    = trend_p + seasonal_p
        return {
            "trend_layers":    trend_p,
            "seasonal_layers": seasonal_p,
            "decomposition":   decomp_p,
            "total":           total_p,
        }

    def __repr__(self) -> str:
        pc = self.param_count()
        return (
            f"DLinearBaseline("
            f"seq_len={self.seq_len}, "
            f"pred_len={self.pred_len}, "
            f"channels={self.channels}, "
            f"params={pc['total']:,})"
        )


# ══════════════════════════════════════════════════════════════════════════════
#  Self-contained unit execution check
# ══════════════════════════════════════════════════════════════════════════════

def _run_unit_check() -> None:
    """
    Instantiate DLinearBaseline with project defaults and verify:

      1. Output shape is exactly [B, pred_len, channels] = [4, 60, 7].
      2. All output values are finite (no NaN / Inf).
      3. Output is non-degenerate (std > 0 — not collapsed to a constant).
      4. Decomposition identity holds: trend + seasonal ≡ x (within float32
         numerical tolerance).
      5. Parameter count matches the closed-form formula:
             2 × C × (seq_len × pred_len + pred_len)

    The mock input shape [4, 1440, 7] matches the project's DataLoader output
    with batch_size=4 (a single gradient step's worth of data).
    """
    SEP  = "═" * 66
    sep  = "─" * 66

    print(SEP)
    print("  DLinearBaseline — unit execution check")
    print(SEP)

    # ── Configuration ─────────────────────────────────────────────────────────
    B, T, C       = 4, 1440, 7
    PRED_LEN      = 60
    KERNEL_SIZE   = 25

    print(f"\n  Mock input shape : [{B}, {T}, {C}]  ← matches DataLoader output")
    print(f"  Expected output  : [{B}, {PRED_LEN}, {C}]")
    print(f"  Decomp kernel    : {KERNEL_SIZE} (moving-average window)")

    # ── Build model ────────────────────────────────────────────────────────────
    torch.manual_seed(42)
    model = DLinearBaseline(
        seq_len=T,
        pred_len=PRED_LEN,
        channels=C,
        kernel_size=KERNEL_SIZE,
    )
    model.eval()

    print(f"\n  {model}")
    pc = model.param_count()
    print(f"  Parameter breakdown:")
    print(f"    trend_layers    : {pc['trend_layers']:>10,}  ({C} × Linear({T}→{PRED_LEN}))")
    print(f"    seasonal_layers : {pc['seasonal_layers']:>10,}  ({C} × Linear({T}→{PRED_LEN}))")
    print(f"    decomposition   : {pc['decomposition']:>10,}  (AvgPool1d — no params)")
    print(f"    total           : {pc['total']:>10,}")

    # Closed-form expected count: 2 × C × (T × pred_len + pred_len)
    expected_params = 2 * C * (T * PRED_LEN + PRED_LEN)
    assert pc["total"] == expected_params, (
        f"Parameter count mismatch: got {pc['total']}, "
        f"expected {expected_params}."
    )
    print(f"\n  ✔  Parameter count matches formula  "
          f"2 × {C} × ({T} × {PRED_LEN} + {PRED_LEN}) = {expected_params:,}")

    # ── Mock input ─────────────────────────────────────────────────────────────
    x = torch.randn(B, T, C)

    # ── Forward pass ───────────────────────────────────────────────────────────
    with torch.no_grad():
        out = model(x)

    # ── Assertion 1: Output shape ─────────────────────────────────────────────
    expected_shape = torch.Size([B, PRED_LEN, C])
    assert out.shape == expected_shape, (
        f"Shape mismatch: got {tuple(out.shape)}, expected {tuple(expected_shape)}."
    )
    print(f"\n  ✔  Output shape : {list(out.shape)} == [{B}, {PRED_LEN}, {C}]")

    # ── Assertion 2: Finite values ────────────────────────────────────────────
    assert torch.isfinite(out).all(), "Output contains NaN or Inf!"
    print(f"  ✔  All output values finite")
    print(f"     min={out.min().item():.4f}  "
          f"max={out.max().item():.4f}  "
          f"mean={out.mean().item():.4f}")

    # ── Assertion 3: Non-degenerate output ────────────────────────────────────
    out_std = out.std().item()
    assert out_std > 1e-6, f"Output std too low ({out_std:.2e}) — possible collapse!"
    print(f"  ✔  Output std : {out_std:.6f}  (non-degenerate)")

    # ── Assertion 4: Decomposition identity x = trend + seasonal ─────────────
    with torch.no_grad():
        trend, seasonal = model.decomposition(x)
    residual = (x - (trend + seasonal)).abs().max().item()
    assert residual < 1e-5, (
        f"Decomposition identity failed: max|x − (trend + seasonal)| = {residual:.2e}"
    )
    print(f"  ✔  Decomposition identity  |x − (trend + seasonal)|_max = {residual:.2e}  ≈ 0")

    # ── Decomposition magnitude report ────────────────────────────────────────
    print(f"\n  Decomposition stats (over mock input):")
    print(f"    trend    std : {trend.std().item():.6f}")
    print(f"    seasonal std : {seasonal.std().item():.6f}")
    print(f"    input    std : {x.std().item():.6f}")

    # ── Assertion 5: Gradient flow (training mode) ────────────────────────────
    model.train()
    x_grad = torch.randn(B, T, C, requires_grad=False)
    pred   = model(x_grad)
    loss   = pred.mean()
    loss.backward()

    grads_non_none = [
        (n, p) for n, p in model.named_parameters()
        if p.grad is not None
    ]
    assert len(grads_non_none) == len(list(model.parameters())), (
        "Some parameters did not receive gradients!"
    )
    print(f"\n  ✔  Gradient flow: all {len(grads_non_none)} parameter tensors "
          f"received non-None gradients")

    # ── Channel-independence sanity: perturbing channel 0 must not change output[..., 1] ──
    model.eval()
    with torch.no_grad():
        x_base  = torch.zeros(1, T, C)
        x_perturb = x_base.clone()
        x_perturb[..., 0] += 1.0   # perturb only channel 0

        out_base    = model(x_base)
        out_perturb = model(x_perturb)

    for c in range(1, C):
        diff = (out_perturb[..., c] - out_base[..., c]).abs().max().item()
        assert diff < 1e-6, (
            f"Channel independence violated: perturbing ch-0 changed ch-{c} "
            f"output by {diff:.2e}"
        )
    print(f"  ✔  Channel independence: perturbing channel 0 has zero effect "
          f"on channels 1–{C-1}  (max diff < 1e-6)")

    # ── Summary ────────────────────────────────────────────────────────────────
    print()
    print(sep)
    print("  SUMMARY")
    print(sep)
    print(f"  Input  [{B}, {T}, {C}]  →  Output [{B}, {PRED_LEN}, {C}]  ✔")
    print(f"  Architecture : SeriesDecomposition(k={KERNEL_SIZE})"
          f"  +  2 × {C} channel-independent Linears")
    print(f"  Parameters   : {pc['total']:,}  "
          f"(trend {pc['trend_layers']:,}  +  seasonal {pc['seasonal_layers']:,})")
    print()
    print(SEP)
    print("  All assertions PASSED  ✔")
    print(SEP + "\n")


if __name__ == "__main__":
    _run_unit_check()
