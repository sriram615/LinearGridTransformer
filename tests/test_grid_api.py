from __future__ import annotations

import unittest
from fastapi.testclient import TestClient
import numpy as np

from src.api.config import ApiConfig
from src.api.main import create_app
from src.data.tetouan_preprocess import FEATURE_NAMES


class GridApiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.config = ApiConfig(active_dataset="grid")
        cls.app = create_app(cls.config)
        cls.client = TestClient(cls.app)
        cls.client.__enter__()

    @classmethod
    def tearDownClass(cls) -> None:
        cls.client.__exit__(None, None, None)

    def test_list_datasets(self) -> None:
        response = self.client.get("/api/v1/datasets")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("datasets", data)
        dataset_ids = [d["id"] for d in data["datasets"]]
        self.assertIn("grid", dataset_ids)
        self.assertIn("tetouan", dataset_ids)

        grid_item = next(d for d in data["datasets"] if d["id"] == "grid")
        self.assertEqual(grid_item["physical_units"], "Megawatts (MW)")
        self.assertEqual(len(grid_item["zones"]), 3)
        self.assertTrue(grid_item["active"])

    def test_grid_dataset_info(self) -> None:
        response = self.client.get("/api/v1/dataset?dataset=grid")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["row_count"], 52555)
        self.assertEqual(data["timezone"], "UTC")
        self.assertEqual(data["physical_units"], "Megawatts (MW)")
        self.assertEqual(len(data["features"]["names"]), 8)
        self.assertEqual(len(data["targets"]["names"]), 3)

    def test_grid_models_roles(self) -> None:
        response = self.client.get("/api/v1/models?dataset=grid")
        self.assertEqual(response.status_code, 200)
        models = response.json()
        self.assertEqual(len(models), 2)
        model_map = {m["name"]: m for m in models}

        # Zero-selection bias validation winner
        self.assertTrue(model_map["CDLinear"]["selected"])
        self.assertEqual(model_map["CDLinear"]["role"], "primary_selected_model")
        self.assertFalse(model_map["LinearGridTransformer"]["selected"])
        self.assertEqual(model_map["LinearGridTransformer"]["role"], "comparative_baseline")

    def test_grid_experiment_info(self) -> None:
        response = self.client.get("/api/v1/experiments/grid_colab_v1")
        self.assertEqual(response.status_code, 200)
        exp = response.json()
        self.assertEqual(exp["run_identifier"], "grid_colab_v1")
        self.assertEqual(exp["physical_units"], "Megawatts (MW)")
        self.assertIn("model_selection", exp)
        self.assertEqual(exp["model_selection"]["selected_model"], "CDLinear")
        self.assertEqual(exp["model_selection"]["econometric_significance"]["dm_statistic"], 72.63)

        models = exp["models"]
        self.assertTrue(models["CDLinear"]["selected"])
        self.assertLess(models["CDLinear"]["best_validation_loss"], models["LinearGridTransformer"]["best_validation_loss"])

    def test_grid_training_history(self) -> None:
        response = self.client.get("/api/v1/experiments/grid_colab_v1/history")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["experiment_id"], "grid_colab_v1")
        self.assertIn("CDLinear", data["models"])
        self.assertIn("LinearGridTransformer", data["models"])
        self.assertEqual(len(data["models"]["CDLinear"]), 50)
        self.assertIn("per_seed", data)

    def test_grid_evaluation_metrics(self) -> None:
        response = self.client.get("/api/v1/experiments/grid_colab_v1/metrics")
        self.assertEqual(response.status_code, 200)
        metrics = response.json()
        self.assertEqual(metrics["experiment_id"], "grid_colab_v1")
        self.assertIn("overall", metrics)
        self.assertIn("horizon_wise", metrics)
        self.assertIn("overall_macro_averages", metrics)
        self.assertIn("per_zone_summary", metrics)
        self.assertIn("econometric_significance", metrics)
        self.assertEqual(metrics["econometric_significance"]["test"], "Diebold-Mariano")

    def test_grid_forecast_compare(self) -> None:
        body = {
            "feature_names": list(FEATURE_NAMES),
            "observations": np.full((144, 8), 5000.0, dtype=np.float64).tolist(),
        }
        response = self.client.post("/api/v1/forecasts/compare?dataset=grid", json=body)
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["experiment_id"], "grid_colab_v1")
        self.assertEqual(data["input_dimensions"], [144, 8])
        self.assertIn("CDLinear", data["models"])
        self.assertIn("LinearGridTransformer", data["models"])
        self.assertEqual(data["models"]["CDLinear"]["output_dimensions"], [6, 3])
        self.assertEqual(len(data["models"]["CDLinear"]["forecast_steps"]), 6)
        self.assertIn("Megawatts (MW)", data["units_status"])

    def test_grid_forecast_single_model(self) -> None:
        body = {
            "feature_names": list(FEATURE_NAMES),
            "observations": np.full((144, 8), 5000.0, dtype=np.float64).tolist(),
        }
        response = self.client.post("/api/v1/forecasts/CDLinear?dataset=grid", json=body)
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["model_name"], "CDLinear")
        self.assertEqual(data["output_dimensions"], [6, 3])
        self.assertIn("Megawatts (MW)", data["units_status"])

    def test_grid_telemetry_latest(self) -> None:
        response = self.client.get("/api/v1/grid/telemetry/latest?limit=144")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["row_count"], 144)
        self.assertEqual(len(data["observations"]), 144)
        self.assertEqual(len(data["observations"][0]), 8)
        self.assertEqual(len(data["timestamps"]), 144)
        self.assertEqual(data["units"], "Megawatts (MW)")
        self.assertIn("start", data["timestamp_range"])
        self.assertIn("end", data["timestamp_range"])


if __name__ == "__main__":
    unittest.main()
