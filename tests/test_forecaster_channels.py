from __future__ import annotations

import unittest

import torch

from src.data.dataset import N_CHANNELS, T_FORECAST
from src.training.train import LinearGridForecaster


class LinearGridForecasterChannelTests(unittest.TestCase):
    def test_default_household_configuration_output_and_backward_compatibility(self) -> None:
        model = LinearGridForecaster()
        self.assertEqual(model.in_channels, 7)
        self.assertEqual(model.out_channels, 7)
        self.assertEqual(model.t_forecast, 60)
        self.assertEqual(N_CHANNELS, 7)
        self.assertEqual(T_FORECAST, 60)

        model.eval()
        with torch.no_grad():
            result = model(torch.randn(2, 1440, 7))
        self.assertEqual(tuple(result.shape), (2, 60, 7))
        self.assertTrue(torch.isfinite(result).all().item())

    def test_output_width_defaults_to_selected_input_width(self) -> None:
        model = LinearGridForecaster(in_channels=8, t_forecast=6)
        self.assertEqual(model.out_channels, 8)

    def test_tetouan_eight_input_three_output_contract(self) -> None:
        model = LinearGridForecaster(
            in_channels=8,
            out_channels=3,
            t_forecast=6,
            dropout=0.0,
        )
        model.eval()
        with torch.no_grad():
            result = model(torch.randn(2, 144, 8))
        self.assertEqual(tuple(result.shape), (2, 6, 3))
        self.assertTrue(torch.isfinite(result).all().item())

    def test_gradients_propagate_for_independent_output_width(self) -> None:
        model = LinearGridForecaster(
            in_channels=8,
            out_channels=3,
            t_forecast=6,
            dropout=0.0,
        )
        x = torch.randn(2, 144, 8, requires_grad=True)
        result = model(x)
        result.square().mean().backward()

        self.assertIsNotNone(x.grad)
        self.assertTrue(torch.isfinite(x.grad).all().item())
        self.assertIsNotNone(model.head.proj.weight.grad)
        self.assertTrue(torch.isfinite(model.head.proj.weight.grad).all().item())
        self.assertEqual(tuple(model.head.proj.weight.shape), (3, 8))

    def test_non_positive_input_or_output_channels_are_rejected(self) -> None:
        for kwargs in ({"in_channels": 0}, {"in_channels": -1},
                       {"out_channels": 0}, {"out_channels": -1}):
            with self.subTest(kwargs=kwargs), self.assertRaisesRegex(ValueError, "positive"):
                LinearGridForecaster(**kwargs)

    def test_invalid_input_rank_and_channel_width_are_rejected(self) -> None:
        model = LinearGridForecaster(in_channels=8, out_channels=3, t_forecast=6)
        with self.assertRaisesRegex(ValueError, "expects \\[B, T, C\\]"):
            model(torch.randn(144, 8))
        with self.assertRaisesRegex(ValueError, "expected 8 input channels"):
            model(torch.randn(2, 144, 7))

    def test_sequence_shorter_than_horizon_is_rejected(self) -> None:
        model = LinearGridForecaster(in_channels=8, out_channels=3, t_forecast=6)
        with self.assertRaisesRegex(ValueError, "shorter than forecast horizon"):
            model(torch.randn(2, 5, 8))


if __name__ == "__main__":
    unittest.main()
