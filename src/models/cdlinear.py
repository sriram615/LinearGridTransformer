"""
cdlinear.py
-----------
Principal Research Scientist & Framework Architect — LinearGridTransformer project.

Implements CDLinearBaseline: a cross-channel-aware linear forecaster that
flattens the full [B, T, C] input into a joint spatial-temporal matrix [B, T*C]
and maps it to the forecast target [B, pred_len, C] through a single unified
nn.Linear layer.

This forces every output channel token to be a linear combination of ALL
historical cross-channel tokens, providing a fair comparison to the
Transformer's joint cross-channel capacity.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class CDLinearBaseline(nn.Module):
    """
    CDLinear: A joint spatial-temporal linear forecaster.

    Unlike DLinear (channel-independent), CDLinear completely flattens the
    input sequence across both time and channel axes and applies a single
    unified linear projection, modelling all cross-channel interactions.

    Parameters
    ----------
    seq_len  : int — input lookback window length (default 1440 = 24 h)
    pred_len : int — forecast horizon length      (default   60 =  1 h)
    channels : int — number of feature channels   (default    7)
    in_channels : int | None — input feature channels (defaults to channels)
    out_channels : int | None — forecast target channels (defaults to channels)

    Shapes
    ------
    Input  : [B, seq_len,  channels]
    Output : [B, pred_len, out_channels]

    Parameter count
    ---------------
    For the default config (seq_len=1440, pred_len=60, channels=7):
        in_features  = 1440 × 7  = 10,080
        out_features =   60 × 7  =    420
        Total        = 10,080 × 420 + 420 = 4,234,020 parameters
    """

    def __init__(
        self,
        seq_len:  int = 1440,
        pred_len: int = 60,
        channels: int = 7,
        in_channels: int | None = None,
        out_channels: int | None = None,
    ) -> None:
        super().__init__()

        for name, value in (("seq_len", seq_len), ("pred_len", pred_len),
                            ("channels", channels)):
            if not isinstance(value, int):
                raise TypeError(f"{name} must be an integer, got {type(value).__name__}.")
            if value <= 0:
                raise ValueError(f"{name} must be positive, got {value}.")

        in_channels = channels if in_channels is None else in_channels
        out_channels = channels if out_channels is None else out_channels
        for name, value in (("in_channels", in_channels), ("out_channels", out_channels)):
            if not isinstance(value, int):
                raise TypeError(f"{name} must be an integer, got {type(value).__name__}.")
            if value <= 0:
                raise ValueError(f"{name} must be positive, got {value}.")

        self.seq_len  = seq_len
        self.pred_len = pred_len
        self.channels = channels
        self.in_channels = in_channels
        self.out_channels = out_channels

        # Single unified projection: (T × input channels) → (pred_len × output channels)
        self.linear = nn.Linear(seq_len * in_channels, pred_len * out_channels)

        self._init_weights()

    def _init_weights(self) -> None:
        """Xavier uniform on weight; zeros on bias — matches project convention."""
        nn.init.xavier_uniform_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : Tensor  [B, seq_len, channels]

        Returns
        -------
        out : Tensor  [B, pred_len, out_channels]
        """
        if x.ndim != 3:
            raise ValueError(
                f"Expected input rank 3 [B, {self.seq_len}, {self.in_channels}], "
                f"got rank {x.ndim} with shape {tuple(x.shape)}."
            )
        B, T, C = x.shape
        if T != self.seq_len:
            raise ValueError(f"Expected sequence length {self.seq_len}, got {T}.")
        if C != self.in_channels:
            raise ValueError(f"Expected {self.in_channels} input channels, got {C}.")

        # ── Step 1: Flatten to 2D spatial-temporal matrix ─────────────────────
        # [B, T, C] → [B, T * C]
        x_flat = x.reshape(B, -1)

        # ── Step 2: Unified joint linear projection ───────────────────────────
        # [B, T * input_channels] → [B, pred_len * output_channels]
        out_flat = self.linear(x_flat)

        # ── Step 3: Reshape back to standard 3D pipeline format ───────────────
        # [B, pred_len * output_channels] → [B, pred_len, output_channels]
        return out_flat.reshape(B, self.pred_len, self.out_channels)

    def param_count(self) -> int:
        """Return total learnable parameter count."""
        return sum(p.numel() for p in self.parameters())

    def __repr__(self) -> str:
        return (
            f"CDLinearBaseline("
            f"seq_len={self.seq_len}, "
            f"pred_len={self.pred_len}, "
            f"channels={self.channels}, "
            f"in_channels={self.in_channels}, "
            f"out_channels={self.out_channels}, "
            f"params={self.param_count():,})"
        )


if __name__ == "__main__":
    SEP = "═" * 68
    sep = "─" * 68

    print(SEP)
    print("  CDLinearBaseline — execution check")
    print(SEP)

    B, T, C  = 4, 1440, 7
    PRED_LEN = 60

    print(f"\n  Mock input shape : [{B}, {T}, {C}]")
    print(f"  Expected output  : [{B}, {PRED_LEN}, {C}]")

    # ── Build model ───────────────────────────────────────────────────────────
    torch.manual_seed(42)
    model = CDLinearBaseline(seq_len=T, pred_len=PRED_LEN, channels=C)
    model.train()
    print(f"\n  {model}")

    # ── Assertion 1: Output shape ─────────────────────────────────────────────
    x   = torch.randn(B, T, C)
    out = model(x)

    expected_shape = torch.Size([B, PRED_LEN, C])
    assert out.shape == expected_shape, (
        f"Shape mismatch: got {tuple(out.shape)}, expected {tuple(expected_shape)}."
    )
    print(f"\n  ✔  Output shape : {list(out.shape)} == [{B}, {PRED_LEN}, {C}]")

    # ── Backward pass via .sum().backward() ───────────────────────────────────
    loss = out.sum()
    loss.backward()

    # ── Assertion 2: Gradient flow — weight ───────────────────────────────────
    w_grad = model.linear.weight.grad
    assert w_grad is not None,               "linear.weight received no gradient!"
    assert not torch.isnan(w_grad).any(),    "linear.weight gradient contains NaN!"
    assert not torch.isinf(w_grad).any(),    "linear.weight gradient contains Inf!"
    assert w_grad.abs().max().item() > 0.0,  "linear.weight gradient is exactly zero (stagnation)!"
    print(
        f"  ✔  linear.weight  grad — shape {list(w_grad.shape)} | "
        f"max_abs={w_grad.abs().max().item():.4e} | "
        f"NaN-free ✔  Inf-free ✔"
    )

    # ── Assertion 3: Gradient flow — bias ────────────────────────────────────
    b_grad = model.linear.bias.grad
    assert b_grad is not None,               "linear.bias received no gradient!"
    assert not torch.isnan(b_grad).any(),    "linear.bias gradient contains NaN!"
    assert not torch.isinf(b_grad).any(),    "linear.bias gradient contains Inf!"
    assert b_grad.abs().max().item() > 0.0,  "linear.bias gradient is exactly zero (stagnation)!"
    print(
        f"  ✔  linear.bias    grad — shape {list(b_grad.shape)} | "
        f"max_abs={b_grad.abs().max().item():.4e} | "
        f"NaN-free ✔  Inf-free ✔"
    )

    print()
    print(sep)
    print("  SUMMARY")
    print(sep)
    print(f"  Input  [{B}, {T}, {C}]  →  reshape → linear → reshape  →  Output [{B}, {PRED_LEN}, {C}]")
    print(f"  Gradients flow seamlessly into weight and bias without NaNs  ✔")
    print(SEP)
    print("  All assertions PASSED  ✔")
    print(SEP + "\n")
