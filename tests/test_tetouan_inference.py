from __future__ import annotations

import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch

from src.data.tetouan_dataset import TetouanSequenceDataset
from src.data.tetouan_preprocess import (
    DEFAULT_SOURCE,
    FEATURE_MAPPING,
    FEATURE_NAMES,
    load_and_validate_csv,
)
from src.inference.tetouan_service import (
    DEFAULT_RUN_NAME,
    TetouanInferenceService,
)


class TetouanInferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.service = TetouanInferenceService(device="cpu")
        frame, timestamps = load_and_validate_csv(DEFAULT_SOURCE)
        start = 1000
        end = start + 144
        cls.raw_input = pd.DataFrame(
            {
                canonical: frame[source].iloc[start:end].to_numpy(copy=True)
                for canonical, source in FEATURE_MAPPING
            },
            index=timestamps[start:end],
        )
        cls.train_dataset = TetouanSequenceDataset(split="train")

    def test_both_real_checkpoints_load_and_produce_raw_scale_forecasts(self) -> None:
        self.assertEqual(set(self.service.models), {"LinearGridTransformer", "CDLinear"})
        self.assertTrue(all(not model.training for model in self.service.models.values()))
        predictions = self.service.predict_both(self.raw_input)
        self.assertEqual(set(predictions), {"LinearGridTransformer", "CDLinear"})
        for values in predictions.values():
            self.assertEqual(values.shape, (6, 3))
            self.assertTrue(np.isfinite(values).all())
        self.assertEqual(self.service.device.type, "cpu")

    def test_preprocessing_matches_training_dataset_normalization(self) -> None:
        normalized = self.service.prepare_input(self.raw_input)
        self.assertEqual(tuple(normalized.shape), (1, 144, 8))
        self.assertEqual(normalized.dtype, torch.float32)
        # Dataset sample 1000 uses raw input rows [1000, 1144), matching this window.
        expected_x, _ = self.train_dataset[1000]
        torch.testing.assert_close(normalized[0], expected_x, rtol=1e-6, atol=1e-6)

    def test_inverse_normalization_uses_saved_target_training_statistics(self) -> None:
        expected = np.broadcast_to(self.service.target_mean, (6, 3))
        for name, model in self.service.models.items():
            with patch.object(
                model,
                "forward",
                return_value=torch.zeros((1, 6, 3), device=self.service.device),
            ):
                prediction = self.service.predict(self.raw_input, name)
            np.testing.assert_allclose(prediction, expected, rtol=1e-12, atol=1e-10)

    def test_numpy_input_requires_and_checks_feature_names(self) -> None:
        values = self.raw_input.to_numpy()
        with self.assertRaisesRegex(ValueError, "feature_names are required"):
            self.service.prepare_input(values)
        self.service.prepare_input(values, feature_names=FEATURE_NAMES)
        with self.assertRaisesRegex(ValueError, "canonical feature order"):
            self.service.prepare_input(values, feature_names=tuple(reversed(FEATURE_NAMES)))

    def test_invalid_row_and_feature_counts_are_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "Expected exactly"):
            self.service.prepare_input(self.raw_input.iloc[:-1])
        with self.assertRaisesRegex(ValueError, "DataFrame columns"):
            self.service.prepare_input(self.raw_input.iloc[:, :-1])
        with self.assertRaisesRegex(ValueError, "DataFrame columns"):
            self.service.prepare_input(self.raw_input.assign(extra=1.0))

    def test_missing_additional_and_reordered_features_are_rejected(self) -> None:
        frames = (
            self.raw_input.drop(columns=[FEATURE_NAMES[0]]),
            self.raw_input.assign(unexpected=0.0),
            self.raw_input.loc[:, list(reversed(FEATURE_NAMES))],
        )
        for frame in frames:
            with self.assertRaisesRegex(ValueError, "DataFrame columns"):
                self.service.prepare_input(frame)

    def test_non_numeric_nan_and_infinite_values_are_rejected(self) -> None:
        nonnumeric = self.raw_input.copy()
        column = FEATURE_NAMES[0]
        nonnumeric[column] = nonnumeric[column].astype(object)
        nonnumeric.iloc[0, 0] = "not-a-number"
        with self.assertRaisesRegex(ValueError, "numeric"):
            self.service.prepare_input(nonnumeric)
        for invalid in (np.nan, np.inf, -np.inf):
            values = self.raw_input.copy()
            values.iloc[0, 0] = invalid
            with self.assertRaisesRegex(ValueError, "finite"):
                self.service.prepare_input(values)

    def test_timestamp_validity_order_and_cadence(self) -> None:
        self.service.prepare_input(self.raw_input)
        with self.assertRaisesRegex(ValueError, "strictly chronological"):
            self.service.prepare_input(self.raw_input.iloc[::-1])
        irregular_index = self.raw_input.index.to_list()
        irregular_index[10] = irregular_index[10] + pd.Timedelta(minutes=1)
        irregular = self.raw_input.copy()
        irregular.index = pd.DatetimeIndex(irregular_index)
        with self.assertRaisesRegex(ValueError, "ten-minute cadence"):
            self.service.prepare_input(irregular)
        values_without_index = self.raw_input.reset_index(drop=True)
        with self.assertRaisesRegex(ValueError, "valid timestamps"):
            self.service.prepare_input(
                values_without_index, timestamps=["not-a-timestamp"] * 144
            )
        aware = self.raw_input.index.tz_localize("UTC")
        with self.assertRaisesRegex(ValueError, "Timezone-aware"):
            self.service.prepare_input(values_without_index, timestamps=aware)

    def test_both_models_use_same_input_for_comparison(self) -> None:
        combined = self.service.predict_both(self.raw_input)
        for name in ("LinearGridTransformer", "CDLinear"):
            separate = self.service.predict(self.raw_input, name)
            np.testing.assert_allclose(combined[name], separate, rtol=0.0, atol=0.0)

    def test_cuda_inference_when_available(self) -> None:
        if not torch.cuda.is_available():
            self.skipTest("CUDA is unavailable in this environment; CUDA inference is untested.")
        service = TetouanInferenceService(device="cuda")
        result = service.predict_both(self.raw_input)
        self.assertTrue(all(values.shape == (6, 3) for values in result.values()))

    def test_manifest_and_run_name_are_consistent(self) -> None:
        self.assertEqual(self.service.run_name, DEFAULT_RUN_NAME)
        self.assertEqual(set(self.service.checkpoints), {"LinearGridTransformer", "CDLinear"})
        for name, payload in self.service.checkpoints.items():
            self.assertEqual(payload["run_metadata"]["run_name"], DEFAULT_RUN_NAME)
            self.assertEqual(
                payload["checkpoint_metadata"]["best_epoch"],
                self.service.manifest["models"][name]["best_validation_epoch"],
            )


if __name__ == "__main__":
    unittest.main()
