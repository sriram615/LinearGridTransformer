from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from src.data.tetouan_preprocess import TARGET_NAMES
from src.training.horizon_evaluation import (
    build_horizon_evaluation_report,
    calculate_horizon_metrics,
    create_horizon_evaluation_artifacts,
)


class TetouanHorizonEvaluationTests(unittest.TestCase):
    def test_known_per_horizon_zone_macro_and_pooled_metrics(self) -> None:
        actual = np.zeros((2, 6, 3), dtype=np.float64)
        predictions = np.zeros_like(actual)
        # Zone errors by horizon: [h+1, 2(h+1), 3(h+1)] on both windows.
        for h in range(6):
            predictions[:, h, :] = [h + 1, 2 * (h + 1), 3 * (h + 1)]
        metrics = calculate_horizon_metrics(actual, predictions)
        self.assertEqual(len(metrics), 6)
        self.assertEqual(metrics[0]["horizon_step"], 1)
        self.assertEqual(metrics[5]["horizon_minutes"], 60)
        self.assertEqual(list(metrics[0]["per_zone"]), list(TARGET_NAMES))
        self.assertEqual(
            [metrics[0]["per_zone"][z]["mae"] for z in TARGET_NAMES],
            [1.0, 2.0, 3.0],
        )
        self.assertEqual(
            [metrics[0]["per_zone"][z]["rmse"] for z in TARGET_NAMES],
            [1.0, 2.0, 3.0],
        )
        self.assertEqual(metrics[0]["macro_average_across_zones"]["mae"], 2.0)
        self.assertEqual(metrics[0]["macro_average_across_zones"]["rmse"], 2.0)
        self.assertAlmostEqual(
            metrics[0]["pooled_across_test_windows_and_zones"]["rmse"],
            np.sqrt(14 / 3),
        )

    def test_invalid_shapes_zones_and_nonfinite_values_are_rejected(self) -> None:
        actual = np.zeros((3, 6, 3))
        with self.assertRaisesRegex(ValueError, "matching"):
            calculate_horizon_metrics(actual, np.zeros((3, 5, 3)))
        with self.assertRaisesRegex(ValueError, "Zone names"):
            calculate_horizon_metrics(actual, actual, ("z1", "z2"))
        bad = actual.copy()
        bad[0, 0, 0] = np.nan
        with self.assertRaisesRegex(ValueError, "finite"):
            calculate_horizon_metrics(bad, actual)
        with self.assertRaisesRegex(ValueError, "finite"):
            calculate_horizon_metrics(actual, np.full_like(actual, np.inf))

    def test_saved_predictions_reproduce_overall_report_and_are_reproducible(self) -> None:
        root = Path(__file__).resolve().parents[1] / "results/tetouan/tetouan_t4_20261101_v1"
        evaluation = json.loads((root / "tetouan_t4_20261101_v1_test_evaluation.json").read_text())
        with np.load(root / "visualisations/test_predictions.npz", allow_pickle=False) as arrays:
            prediction_arrays = {key: arrays[key] for key in arrays.files}
        self.assertEqual(set(prediction_arrays), {"actual", "cdlinear", "lineargridtransformer"})
        self.assertEqual(prediction_arrays["actual"].shape, (7858, 6, 3))
        report1 = build_horizon_evaluation_report(
            evaluation,
            prediction_arrays,
            source_evaluation_reference="source.json",
            prediction_artifact_reference="predictions.npz",
        )
        report2 = build_horizon_evaluation_report(
            evaluation,
            prediction_arrays,
            source_evaluation_reference="source.json",
            prediction_artifact_reference="predictions.npz",
        )
        self.assertEqual(report1, report2)
        self.assertEqual(report1["target_zone_names"], list(TARGET_NAMES))
        self.assertEqual(report1["test_window_count"], 7858)
        for model, record in report1["models"].items():
            self.assertEqual(len(record["per_horizon"]), 6)
            self.assertTrue(record["consistency_with_existing_overall_metrics"]["all_match"])
            for h in record["per_horizon"]:
                self.assertEqual(list(h["per_zone"]), list(TARGET_NAMES))
                self.assertEqual(set(h["per_zone"][TARGET_NAMES[0]]), {"mae", "rmse"})
            for zone, check in record["consistency_with_existing_overall_metrics"]["per_zone"].items():
                self.assertTrue(check["mae_matches_saved"], (model, zone, "MAE"))
                self.assertTrue(check["rmse_matches_saved"], (model, zone, "RMSE"))

    def test_report_and_visualization_creation_refuses_overwrite(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = Path(__file__).resolve().parents[1] / "results/tetouan/tetouan_t4_20261101_v1"
            # The pure report builder is tested above; here verify output naming/collision safety
            # using copied source inputs in a temporary experiment directory.
            (root / "visualisations").mkdir()
            for filename in (
                "tetouan_t4_20261101_v1_test_evaluation.json",
            ):
                (root / filename).write_bytes((source / filename).read_bytes())
            (root / "visualisations/test_predictions.npz").write_bytes(
                (source / "visualisations/test_predictions.npz").read_bytes()
            )
            report_path, plot_path, _report = create_horizon_evaluation_artifacts(root)
            self.assertTrue(report_path.is_file())
            self.assertTrue(plot_path.is_file())
            self.assertEqual(report_path.name, "tetouan_t4_20261101_v1_horizon_evaluation.json")
            self.assertEqual(plot_path.name, "horizon_wise_mae.png")
            with self.assertRaises(FileExistsError):
                create_horizon_evaluation_artifacts(root)


if __name__ == "__main__":
    unittest.main()
