from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from src.data.tetouan_dataset import TetouanSequenceDataset
from src.data.tetouan_preprocess import (
    DEFAULT_SOURCE,
    FEATURE_NAMES,
    TARGET_NAMES,
    preprocess_tetouan,
)


class TetouanDatasetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls._tmp = tempfile.TemporaryDirectory()
        preprocess_tetouan(DEFAULT_SOURCE, cls._tmp.name)
        cls.datasets = {
            split: TetouanSequenceDataset(split=split, artifact_dir=cls._tmp.name)
            for split in ("train", "validation", "test")
        }
        source = pd.read_csv(DEFAULT_SOURCE)
        cls.timestamps = pd.to_datetime(source["DateTime"], format="%m/%d/%Y %H:%M").to_numpy(dtype="datetime64[ns]")

    @classmethod
    def tearDownClass(cls) -> None:
        cls._tmp.cleanup()

    def test_channel_order_shapes_and_metadata(self) -> None:
        for dataset in self.datasets.values():
            self.assertEqual(dataset.input_channel_names, FEATURE_NAMES)
            self.assertEqual(dataset.target_channel_names, TARGET_NAMES)
            self.assertEqual(dataset.input_shape, (144, 8))
            self.assertEqual(dataset.target_shape, (6, 3))
            x, y = dataset[0]
            self.assertEqual(tuple(x.shape), (144, 8))
            self.assertEqual(tuple(y.shape), (6, 3))
            self.assertEqual(x.dtype, torch.float32)
            self.assertEqual(y.dtype, torch.float32)
            self.assertTrue(torch.isfinite(x).all().item())
            self.assertTrue(torch.isfinite(y).all().item())
            self.assertIn("first_timestamp", dataset.partition_boundaries)
            self.assertIn("last_timestamp", dataset.partition_boundaries)

    def test_window_counts_first_last_and_timestamp_ranges(self) -> None:
        expected_counts = {"train": 36542, "validation": 7857, "test": 7858}
        expected_first = {"train": 144, "validation": 36691, "test": 44553}
        for split, ds in self.datasets.items():
            with self.subTest(split=split):
                self.assertEqual(len(ds), expected_counts[split])
                starts = ds.target_start_indices
                self.assertEqual(int(starts[0]), expected_first[split])
                self.assertEqual(int(starts[-1]), ds.end_index - 6)
                first = ds.window_timestamps(0)
                last = ds.window_timestamps(len(ds) - 1)
                np.testing.assert_array_equal(
                    np.array([first[k] for k in ("input_start", "input_end", "target_start", "target_end")]),
                    np.array([str(self.timestamps[expected_first[split] - 144].astype("datetime64[m]")).replace("T", " "), str(self.timestamps[expected_first[split] - 1].astype("datetime64[m]")).replace("T", " "), str(self.timestamps[expected_first[split]].astype("datetime64[m]")).replace("T", " "), str(self.timestamps[expected_first[split] + 5].astype("datetime64[m]")).replace("T", " ")]),
                )
                self.assertLessEqual(last["target_end"], ds.partition_boundaries["last_timestamp"])
                self.assertEqual(last["target_end"], str(self.timestamps[ds.end_index - 1].astype("datetime64[m]")).replace("T", " "))

    def test_validation_and_test_context_can_precede_partition(self) -> None:
        validation = self.datasets["validation"]
        test = self.datasets["test"]
        self.assertEqual(validation.window_timestamps(0)["input_end"], str(self.timestamps[36690].astype("datetime64[m]")).replace("T", " "))
        self.assertEqual(validation.window_timestamps(0)["target_start"], str(self.timestamps[36691].astype("datetime64[m]")).replace("T", " "))
        self.assertEqual(test.window_timestamps(0)["input_end"], str(self.timestamps[44552].astype("datetime64[m]")).replace("T", " "))
        self.assertEqual(test.window_timestamps(0)["target_start"], str(self.timestamps[44553].astype("datetime64[m]")).replace("T", " "))

    def test_no_targets_cross_partition_boundaries(self) -> None:
        for name, ds in self.datasets.items():
            starts = ds.target_start_indices
            ends = starts + ds.horizon
            with self.subTest(split=name):
                self.assertTrue(np.all(starts >= ds.start_index))
                self.assertTrue(np.all(ends <= ds.end_index))
                if name == "train":
                    self.assertTrue(np.all(starts - ds.lookback >= 0))
                if name == "validation":
                    self.assertLessEqual(int(ends.max()), 44553)
                if name == "test":
                    self.assertLessEqual(int(ends.max()), 52416)

    def test_training_normalization_and_inverse_target(self) -> None:
        ds = self.datasets["train"]
        train_end = ds.end_index
        source = pd.read_csv(DEFAULT_SOURCE)
        features = source.loc[:, [
            "Zone 1 Power Consumption", "Zone 2  Power Consumption", "Zone 3  Power Consumption",
            "Temperature", "Humidity", "Wind Speed", "general diffuse flows", "diffuse flows",
        ]].to_numpy(dtype=np.float64)
        targets = source.loc[:, [
            "Zone 1 Power Consumption", "Zone 2  Power Consumption", "Zone 3  Power Consumption",
        ]].to_numpy(dtype=np.float64)
        stats = ds.metadata["normalization"]
        np.testing.assert_allclose(stats["feature_mean"], features[:train_end].mean(0))
        np.testing.assert_allclose(stats["feature_std"], features[:train_end].std(0, ddof=0))
        np.testing.assert_allclose(stats["target_mean"], targets[:train_end].mean(0))
        np.testing.assert_allclose(stats["target_std"], targets[:train_end].std(0, ddof=0))
        raw_x = features[144:288]
        raw_y = targets[288:294]
        x, y = ds[144]
        np.testing.assert_allclose(x.numpy(), (raw_x - stats["feature_mean"]) / stats["feature_std"], rtol=1e-6, atol=1e-6)
        np.testing.assert_allclose(y.numpy(), (raw_y - stats["target_mean"]) / stats["target_std"], rtol=1e-6, atol=1e-6)
        restored = y.numpy() * np.asarray(stats["target_std"]) + np.asarray(stats["target_mean"])
        np.testing.assert_allclose(restored, raw_y, rtol=1e-6, atol=1e-3)

    def test_batch_contract_and_invalid_access(self) -> None:
        loader = DataLoader(self.datasets["validation"], batch_size=4, shuffle=False)
        x, y = next(iter(loader))
        self.assertEqual(tuple(x.shape), (4, 144, 8))
        self.assertEqual(tuple(y.shape), (4, 6, 3))
        self.assertEqual(x.dtype, torch.float32)
        self.assertEqual(y.dtype, torch.float32)
        self.assertTrue(torch.isfinite(x).all().item())
        self.assertTrue(torch.isfinite(y).all().item())
        with self.assertRaises(ValueError):
            TetouanSequenceDataset(split="val", artifact_dir=self._tmp.name)
        with self.assertRaises(IndexError):
            self.datasets["train"][-1]
        with self.assertRaises(IndexError):
            self.datasets["train"][len(self.datasets["train"])]
        with self.assertRaises(ValueError):
            TetouanSequenceDataset(split="train", artifact_dir=self._tmp.name, lookback=0)


if __name__ == "__main__":
    unittest.main()
