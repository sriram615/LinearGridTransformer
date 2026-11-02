from __future__ import annotations

import unittest

import torch
from torch.utils.data import DataLoader

from src.data.tetouan_dataset import TetouanSequenceDataset
from src.models.cdlinear import CDLinearBaseline
from src.training.train import LinearGridForecaster


class TetouanModelIntegrationTests(unittest.TestCase):
    def test_real_training_batch_runs_through_both_forecasters(self) -> None:
        dataset = TetouanSequenceDataset(split="train")
        loader = DataLoader(dataset, batch_size=4, shuffle=False, num_workers=0)
        x, target = next(iter(loader))

        expected_x_shape = (4, 144, 8)
        expected_y_shape = (4, 6, 3)
        self.assertEqual(tuple(x.shape), expected_x_shape)
        self.assertEqual(tuple(target.shape), expected_y_shape)
        self.assertEqual(x.dtype, torch.float32)
        self.assertEqual(target.dtype, torch.float32)
        self.assertTrue(torch.isfinite(x).all().item())
        self.assertTrue(torch.isfinite(target).all().item())

        transformer = LinearGridForecaster(
            in_channels=8, out_channels=3, t_forecast=6, dropout=0.0
        ).eval()
        cdlinear = CDLinearBaseline(
            seq_len=144, pred_len=6, in_channels=8, out_channels=3
        ).eval()

        transformer_input = x.detach().requires_grad_(True)
        transformer_prediction = transformer(transformer_input)
        self.assertEqual(tuple(transformer_prediction.shape), tuple(target.shape))
        self.assertEqual(transformer_prediction.dtype, target.dtype)
        self.assertTrue(torch.isfinite(transformer_prediction).all().item())
        transformer_prediction.square().mean().backward()
        self.assertIsNotNone(transformer_input.grad)
        self.assertTrue(torch.isfinite(transformer_input.grad).all().item())
        self.assertIsNotNone(transformer.head.proj.weight.grad)
        self.assertTrue(torch.isfinite(transformer.head.proj.weight.grad).all().item())

        cdlinear_input = x.detach().requires_grad_(True)
        cdlinear_prediction = cdlinear(cdlinear_input)
        self.assertEqual(tuple(cdlinear_prediction.shape), tuple(target.shape))
        self.assertEqual(cdlinear_prediction.dtype, target.dtype)
        self.assertTrue(torch.isfinite(cdlinear_prediction).all().item())
        cdlinear_prediction.square().mean().backward()
        self.assertIsNotNone(cdlinear_input.grad)
        self.assertTrue(torch.isfinite(cdlinear_input.grad).all().item())
        self.assertIsNotNone(cdlinear.linear.weight.grad)
        self.assertTrue(torch.isfinite(cdlinear.linear.weight.grad).all().item())

        self.assertEqual(dataset.split, "train")
        self.assertEqual(dataset.input_shape, (144, 8))
        self.assertEqual(dataset.target_shape, (6, 3))
        self.assertEqual(x.shape[0], 4)
        self.assertEqual(target.shape[0], 4)


if __name__ == "__main__":
    unittest.main()
