"""Unit tests for GridExperimentStore and GridInferenceService."""

import unittest
import numpy as np

from src.data.grid_experiment_store import GridExperimentStore
from src.inference.grid_service import GridInferenceService


class TestGridServiceAndStore(unittest.TestCase):
    """Test suite for Phase 1 grid store and inference foundation."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.store = GridExperimentStore()
        cls.service = GridInferenceService(device="cpu")

    def test_store_artifact_verification(self) -> None:
        status = self.store.verify_artifacts()
        self.assertEqual(status["status"], "ok")
        self.assertEqual(status["verified_seeds"], [42, 123, 2026])

    def test_store_dataset_info(self) -> None:
        info = self.store.get_dataset_info()
        self.assertEqual(info["units"], "Megawatts (MW)")
        self.assertEqual(info["timezone"], "UTC")
        self.assertEqual(info["row_count"], 52555)
        self.assertEqual(len(info["zones"]), 3)
        self.assertIn("ZONE_1", info["zones"])

    def test_store_experiment_info(self) -> None:
        exp = self.store.get_experiment_info()
        self.assertEqual(exp["run_identifier"], "grid_colab_v1")
        self.assertEqual(exp["model_selection"]["selected_model"], "CDLinear")
        self.assertIn("econometric_significance", exp["model_selection"])
        self.assertEqual(exp["model_selection"]["econometric_significance"]["dm_statistic"], 72.63)

        models = exp["models"]
        self.assertTrue(models["CDLinear"]["selected"])
        self.assertFalse(models["LinearGridTransformer"]["selected"])
        self.assertLess(models["CDLinear"]["validation_loss_mean"], models["LinearGridTransformer"]["validation_loss_mean"])

    def test_store_training_history(self) -> None:
        history = self.store.get_training_history()
        self.assertIn("CDLinear", history)
        self.assertIn("LinearGridTransformer", history)
        self.assertEqual(len(history["CDLinear"]), 50)
        self.assertEqual(len(history["LinearGridTransformer"]), 50)

        seed_hist = self.store.get_seed_training_history()
        self.assertIn("42", seed_hist["CDLinear"])
        self.assertIn("123", seed_hist["CDLinear"])
        self.assertIn("2026", seed_hist["CDLinear"])

    def test_store_evaluation_info(self) -> None:
        eval_info = self.store.get_evaluation_info()
        self.assertEqual(eval_info["physical_units"], "Megawatts (MW)")
        macros = eval_info["overall_macro_averages"]
        self.assertLess(macros["CDLinear"]["mae_mean"], macros["LinearGridTransformer"]["mae_mean"])
        self.assertLess(macros["CDLinear"]["mae_mean"], macros["Persistence"]["mae"])

    def test_inference_service_predict_single(self) -> None:
        dummy_input = np.ones((144, 8), dtype=np.float64) * 1000.0
        # Dummy weather channels
        dummy_input[:, 3] = 15.0   # temp
        dummy_input[:, 4] = 60.0   # humidity
        dummy_input[:, 5] = 10.0   # wind
        dummy_input[:, 6] = 50.0   # diffuse
        dummy_input[:, 7] = 200.0  # direct

        pred_cdl = self.service.predict(dummy_input, "CDLinear")
        self.assertEqual(pred_cdl.shape, (6, 3))
        self.assertTrue(np.isfinite(pred_cdl).all())
        # Predictions should be positive MW
        self.assertTrue((pred_cdl > 0).all())

        pred_lgt = self.service.predict(dummy_input, "LinearGridTransformer")
        self.assertEqual(pred_lgt.shape, (6, 3))
        self.assertTrue(np.isfinite(pred_lgt).all())

    def test_inference_service_predict_both(self) -> None:
        dummy_input = np.ones((144, 8), dtype=np.float64) * 2000.0
        preds = self.service.predict_both(dummy_input)
        self.assertIn("CDLinear", preds)
        self.assertIn("LinearGridTransformer", preds)
        self.assertEqual(preds["CDLinear"].shape, (6, 3))
        self.assertEqual(preds["LinearGridTransformer"].shape, (6, 3))

    def test_inference_service_shape_validation(self) -> None:
        bad_input = np.ones((100, 8), dtype=np.float64)
        with self.assertRaises(ValueError):
            self.service.predict(bad_input, "CDLinear")


if __name__ == "__main__":
    unittest.main()
