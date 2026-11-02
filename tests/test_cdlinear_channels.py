import unittest

import torch

from src.models.cdlinear import CDLinearBaseline


class CDLinearChannelTests(unittest.TestCase):
    def test_default_household_shape_and_legacy_channels(self):
        model = CDLinearBaseline()
        self.assertEqual(tuple(model(torch.randn(2, 1440, 7)).shape), (2, 60, 7))

    def test_legacy_channels_argument_sets_both_dimensions(self):
        model = CDLinearBaseline(seq_len=12, pred_len=3, channels=5)
        self.assertEqual((model.in_channels, model.out_channels), (5, 5))
        self.assertEqual(tuple(model(torch.randn(2, 12, 5)).shape), (2, 3, 5))

    def test_tetouan_shape_projection_finiteness_and_gradients(self):
        model = CDLinearBaseline(seq_len=144, pred_len=6, in_channels=8, out_channels=3)
        x = torch.randn(2, 144, 8, requires_grad=True)
        y = model(x)
        self.assertEqual(tuple(y.shape), (2, 6, 3))
        self.assertEqual(model.linear.in_features, 1152)
        self.assertEqual(model.linear.out_features, 18)
        self.assertTrue(torch.isfinite(y).all().item())
        y.sum().backward()
        self.assertIsNotNone(x.grad)
        self.assertIsNotNone(model.linear.weight.grad)
        self.assertTrue(torch.isfinite(x.grad).all().item())

    def test_rejects_invalid_input_rank_and_dimensions(self):
        model = CDLinearBaseline(seq_len=12, pred_len=3, channels=2)
        for value in (torch.randn(12, 2), torch.randn(1, 12, 2, 1)):
            with self.subTest(shape=tuple(value.shape)), self.assertRaisesRegex(ValueError, "rank 3"):
                model(value)
        with self.assertRaisesRegex(ValueError, "sequence length"):
            model(torch.randn(1, 11, 2))
        with self.assertRaisesRegex(ValueError, "input channels"):
            model(torch.randn(1, 12, 3))

    def test_rejects_invalid_configuration_values(self):
        for kwargs in (
            {"seq_len": 0}, {"pred_len": -1}, {"channels": 0},
            {"in_channels": 0}, {"out_channels": -2},
        ):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                CDLinearBaseline(**kwargs)
        with self.assertRaises(TypeError):
            CDLinearBaseline(in_channels=2.5)


if __name__ == "__main__":
    unittest.main()
