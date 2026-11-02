from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.tetouan_preprocess import (
    DEFAULT_SOURCE,
    FEATURE_MAPPING,
    FEATURE_NAMES,
    FEATURE_SOURCE_COLUMNS,
    SOURCE_COLUMNS,
    TARGET_MAPPING,
    TARGET_NAMES,
    TARGET_SOURCE_COLUMNS,
    TetouanDataError,
    calculate_split_indices,
    fit_normalization,
    inverse_normalize_targets,
    load_and_validate_csv,
    preprocess_tetouan,
)


class TetouanPreprocessingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.frame, cls.timestamps = load_and_validate_csv(DEFAULT_SOURCE)

    def test_exact_column_mapping_and_whitespace(self) -> None:
        self.assertEqual(tuple(self.frame.columns), SOURCE_COLUMNS)
        self.assertEqual(FEATURE_MAPPING[:3], (
            ("zone_1_power_consumption", "Zone 1 Power Consumption"),
            ("zone_2_power_consumption", "Zone 2  Power Consumption"),
            ("zone_3_power_consumption", "Zone 3  Power Consumption"),
        ))
        self.assertEqual(len(FEATURE_NAMES), 8)
        self.assertEqual(FEATURE_SOURCE_COLUMNS[:3], TARGET_SOURCE_COLUMNS)
        self.assertEqual(len(TARGET_NAMES), 3)
        self.assertEqual(tuple(name for name, _ in TARGET_MAPPING), TARGET_NAMES)

    def test_actual_source_quality_and_timestamp_cadence(self) -> None:
        self.assertEqual(len(self.frame), 52416)
        self.assertFalse(self.frame.isna().any().any())
        self.assertEqual(int(self.frame.duplicated().sum()), 0)
        self.assertEqual(self.timestamps[0], pd.Timestamp("2017-01-01 00:00"))
        self.assertEqual(self.timestamps[-1], pd.Timestamp("2017-12-30 23:50"))
        self.assertTrue(self.timestamps.is_monotonic_increasing)
        self.assertFalse(self.timestamps.has_duplicates)
        times_ns = self.timestamps.to_numpy(dtype="datetime64[ns]").astype(np.int64)
        self.assertTrue(np.all(np.diff(times_ns) == 10 * 60 * 1_000_000_000))
        self.assertNotIn(pd.Timestamp("2017-12-31 00:00"), self.timestamps)

    def test_split_indices_counts_and_timestamp_boundaries(self) -> None:
        bounds = calculate_split_indices(52416)
        expected = {
            "train": {"start_index": 0, "end_index_exclusive": 36691},
            "validation": {"start_index": 36691, "end_index_exclusive": 44553},
            "test": {"start_index": 44553, "end_index_exclusive": 52416},
        }
        self.assertEqual(bounds, expected)
        self.assertEqual([v["end_index_exclusive"] - v["start_index"] for v in bounds.values()], [36691, 7862, 7863])
        self.assertEqual(str(self.timestamps[36690]), "2017-09-12 19:00:00")
        self.assertEqual(str(self.timestamps[36691]), "2017-09-12 19:10:00")
        self.assertEqual(str(self.timestamps[44552]), "2017-11-06 09:20:00")
        self.assertEqual(str(self.timestamps[44553]), "2017-11-06 09:30:00")

    def test_training_only_population_normalization(self) -> None:
        bounds = calculate_split_indices(len(self.frame))
        end = bounds["train"]["end_index_exclusive"]
        features = self.frame.loc[:, FEATURE_SOURCE_COLUMNS].to_numpy(dtype=np.float64)
        targets = self.frame.loc[:, TARGET_SOURCE_COLUMNS].to_numpy(dtype=np.float64)
        stats = fit_normalization(features, targets, end)
        np.testing.assert_allclose(stats["feature_mean"], features[:end].mean(0))
        np.testing.assert_allclose(stats["feature_std"], features[:end].std(0, ddof=0))
        np.testing.assert_allclose(stats["target_mean"], targets[:end].mean(0))
        np.testing.assert_allclose(stats["target_std"], targets[:end].std(0, ddof=0))
        altered = features.copy()
        altered[end:] = 1e12
        altered_targets = targets.copy()
        altered_targets[end:] = -1e12
        altered_stats = fit_normalization(altered, altered_targets, end)
        for key in stats:
            np.testing.assert_allclose(stats[key], altered_stats[key], rtol=1e-13, atol=1e-10)

    def test_inverse_normalization(self) -> None:
        raw = self.frame.loc[:, TARGET_SOURCE_COLUMNS].to_numpy(dtype=np.float64)[:20]
        mean, std = raw.mean(0), raw.std(0)
        normalized = (raw - mean) / std
        np.testing.assert_allclose(inverse_normalize_targets(normalized, mean, std), raw, rtol=1e-12, atol=1e-10)

    def test_preprocessing_artifact_schema_and_provenance(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            metadata = preprocess_tetouan(DEFAULT_SOURCE, tmp)
            arrays_path = Path(tmp) / "series.npz"
            metadata_path = Path(tmp) / "metadata.json"
            self.assertTrue(arrays_path.is_file())
            self.assertTrue(metadata_path.is_file())
            saved = json.loads(metadata_path.read_text())
            self.assertEqual(metadata, saved)
            self.assertEqual(saved["row_count"], 52416)
            self.assertIsNone(saved["timestamps"]["timezone"])
            self.assertEqual(saved["split"]["partitions"]["train"]["end_index_exclusive"], 36691)
            with np.load(arrays_path, allow_pickle=False) as arrays:
                self.assertEqual(arrays["features"].shape, (52416, 8))
                self.assertEqual(arrays["targets"].shape, (52416, 3))
                self.assertEqual(arrays["timestamps"].shape, (52416,))
                np.testing.assert_array_equal(arrays["features"][:5], self.frame.loc[:4, FEATURE_SOURCE_COLUMNS].to_numpy())
                np.testing.assert_array_equal(arrays["targets"][:5], self.frame.loc[:4, TARGET_SOURCE_COLUMNS].to_numpy())
            with self.assertRaises(FileExistsError):
                preprocess_tetouan(DEFAULT_SOURCE, tmp)

    def test_rejects_wrong_columns_malformed_timestamps_duplicates_and_gaps(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = self.frame.iloc[:4].copy()
            cases = []
            wrong = base.rename(columns={"Zone 2  Power Consumption": "Zone 2 Power Consumption"})
            cases.append((wrong, "column"))
            malformed = base.copy()
            malformed.loc[malformed.index[1], "DateTime"] = "2017-01-01 00:10"
            cases.append((malformed, "format"))
            duplicate_ts = base.copy()
            duplicate_ts.loc[duplicate_ts.index[1], "DateTime"] = duplicate_ts.iloc[0]["DateTime"]
            cases.append((duplicate_ts, "Duplicate timestamps"))
            gap = base.iloc[[0, 2, 3]].copy()
            cases.append((gap, "10 minutes"))
            duplicate_row = pd.concat([base.iloc[:1], base.iloc[:1]], ignore_index=True)
            cases.append((duplicate_row, "Duplicate rows"))
            out_of_order = base.iloc[[0, 2, 1, 3]].copy()
            cases.append((out_of_order, "chronological"))
            non_finite = base.copy()
            non_finite.loc[non_finite.index[0], "Temperature"] = np.inf
            cases.append((non_finite, "non-finite"))
            missing = base.copy()
            missing.loc[missing.index[0], "Temperature"] = np.nan
            cases.append((missing, "Missing values"))
            for idx, (frame, expected) in enumerate(cases):
                path = Path(tmp) / f"case_{idx}.csv"
                frame.to_csv(path, index=False)
                with self.subTest(case=idx), self.assertRaisesRegex(TetouanDataError, expected):
                    load_and_validate_csv(path)

    def test_rejects_unusable_split_and_zero_scale(self) -> None:
        with self.assertRaises(TetouanDataError):
            calculate_split_indices(100)
        with self.assertRaises(TetouanDataError):
            fit_normalization(np.ones((200, 8)), np.ones((200, 3)), 140)


if __name__ == "__main__":
    unittest.main()
