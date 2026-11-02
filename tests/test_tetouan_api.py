from __future__ import annotations

import unittest
import json
import math
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from fastapi.testclient import TestClient
import numpy as np
import pandas as pd

from src.api.config import ApiConfig
from src.api.main import create_app
from src.data.tetouan_experiment_store import (
    REPO_ROOT,
    RUN_NAME,
    InvalidArtifactError,
    MissingArtifactError,
)
from src.data.tetouan_extended_experiment_store import (
    DEFAULT_CHECKPOINT_DIR as EXTENDED_CHECKPOINT_DIR,
    DEFAULT_EVALUATION_DIR as EXTENDED_EVALUATION_DIR,
    DEFAULT_EXPERIMENT_DIR as EXTENDED_EXPERIMENT_DIR,
    EXPERIMENT_ID as EXTENDED_EXPERIMENT_ID,
    TetouanExtendedExperimentStore,
)
from src.data.tetouan_preprocess import (
    DEFAULT_SOURCE,
    FEATURE_MAPPING,
    FEATURE_NAMES,
    TARGET_NAMES,
)
from src.inference.tetouan_service import TetouanInferenceService


class FakeStore:
    def __init__(self, *, status: str = "ok") -> None:
        self.status = status
        self.verify_calls = 0

    def verify_artifacts(self):
        self.verify_calls += 1
        return {"status": self.status, "issues": [] if self.status == "ok" else ["/private/data/secret"]}


class FakeInference:
    def __init__(self) -> None:
        self.device = "cpu"
        self.models = {"LinearGridTransformer": object(), "CDLinear": object()}
        self.checkpoints = {
            "LinearGridTransformer": {"model_config": {"in_channels": 8, "out_channels": 3,
                                                             "lookback": 144, "horizon": 6}},
            "CDLinear": {"model_config": {"in_channels": 8, "out_channels": 3,
                                             "seq_len": 144, "pred_len": 6}},
        }
        self.prediction_calls = 0

    def predict(self, *args, **kwargs):
        self.prediction_calls += 1
        raise AssertionError("Historical endpoints must not invoke prediction.")


class HistoricalFakeStore(FakeStore):
    def __init__(self):
        super().__init__()
        self.calls = []
        self.history_rows = [{"epoch": 1, "train_loss": 0.25, "validation_loss": 0.5}]

    def get_dataset_info(self):
        self.calls.append("dataset")
        return {
            "dataset_name": "Power Consumption of Tetouan City",
            "source": {"source_url": "https://archive.ics.uci.edu/dataset/849",
                       "file_name": "Tetuan City power consumption.csv"},
            "source_path": "/Users/private/raw/source.csv",
            "row_count": 52416,
            "date_range": {"first": "2017-01-01 00:00:00", "last": "2017-12-30 23:50:00"},
            "cadence_minutes": 10,
            "timezone": None,
            "timezone_status": "undocumented",
            "features": {"names": list(FEATURE_NAMES)},
            "targets": {"names": list(TARGET_NAMES)},
            "splits": {"partitions": {"train": {"row_count": 36691}}},
            "windows": {"lookback": 144, "horizon": 6, "samples": {"train": 36542}},
            "normalization": {"fit_partition": "train", "method": "z-score"},
            "data_quality": {"source": "/Users/private/raw/INSPECTION_REPORT.md",
                             "missing_values_per_column": 0},
            "coverage_uncertainties": {"units": None, "units_status": "unknown"},
            "metadata_artifact": "/content/private/metadata.json",
        }

    def get_experiment_info(self):
        self.calls.append("experiment")
        models = {}
        for name in ("LinearGridTransformer", "CDLinear"):
            config = self._config(name)
            models[name] = {
                "configuration": config,
                "checkpoint": {"path": f"/Users/private/checkpoints/{name}.pt",
                               "exists": True, "size_bytes": 1024,
                               "sha256_observed": "abc", "recorded_manifest_reference": "/content/checkpoint.pt"},
                "best_validation_epoch": 8,
                "best_validation_loss": 0.125,
                "history": self.history_rows,
            }
        return {
            "run_identifier": RUN_NAME,
            "run_type": "experiment",
            "created_utc": "2026-11-01T08:21:14Z",
            "provenance": {"git_commit": None,
                           "training_environment": {"python_version": "3.13.16",
                                                     "device": "cuda", "gpu_name": "Tesla T4"}},
            "training_configuration": {"epochs": 10, "batch_size": 32},
            "optimizer": "AdamW",
            "loss": "mean squared error",
            "seed_policy": {"seed": 42},
            "test_partition_used": False,
            "models": models,
            "artifact_references": {"manifest": "/content/run/manifest.json",
                                    "checkpoint_CDLinear": "/Users/private/checkpoints/CDLinear.pt"},
            "manifest_artifact": "/content/run/manifest.json",
        }

    def get_training_history(self):
        self.calls.append("history")
        return {name: self.history_rows for name in ("LinearGridTransformer", "CDLinear")}

    def get_evaluation_info(self):
        self.calls.append("metrics")
        return {
            "experiment_id": RUN_NAME,
            "test_split": {"name": "test", "evaluation_sample_count": 7858},
            "test_usage": {"used_for_training": False, "used_for_model_selection": False},
            "overall": {
                "models": {"LinearGridTransformer": {"per_zone": {TARGET_NAMES[0]: {"mae": 1.0}}},
                           "CDLinear": {"per_zone": {TARGET_NAMES[0]: {"mae": 0.5}}}},
                "baselines": {"training_mean": {"macro_average": {"mae": 10.0}},
                              "persistence": {"macro_average": {"mae": 5.0}}},
            },
            "horizon_wise": {
                "forecast_horizon_count": 6,
                "target_zone_names": list(TARGET_NAMES),
                "models": {"LinearGridTransformer": [{"horizon_step": 1, "per_zone": {TARGET_NAMES[0]: {"mae": 1.0}}}],
                           "CDLinear": [{"horizon_step": 1, "per_zone": {TARGET_NAMES[0]: {"mae": 0.5}}}]},
            },
            "overall_evaluation_artifact": "/Users/private/evaluation.json",
        }

    @staticmethod
    def _config(name):
        if name == "LinearGridTransformer":
            return {"in_channels": 8, "out_channels": 3, "lookback": 144, "horizon": 6}
        return {"in_channels": 8, "out_channels": 3, "seq_len": 144, "pred_len": 6}


class ForecastFakeInference(FakeInference):
    def __init__(self):
        super().__init__()
        self.single_calls = []
        self.both_calls = []
        self.metadata = {"units_status": "unknown; numeric dataset values"}
        self.target_channel_names = TARGET_NAMES
        self.single_values = {
            "LinearGridTransformer": np.arange(18, dtype=np.float64).reshape(6, 3),
            "CDLinear": np.arange(18, dtype=np.float64).reshape(6, 3) + 100.0,
        }

    def predict(self, observations, model_name, *, feature_names, timestamps=None):
        self.single_calls.append((observations, model_name, feature_names, timestamps))
        return self.single_values[model_name]

    def predict_both(self, observations, *, feature_names, timestamps=None):
        self.both_calls.append((observations, feature_names, timestamps))
        return {name: self.single_values[name] for name in ("LinearGridTransformer", "CDLinear")}


def api_config_for_test(root: Path) -> ApiConfig:
    return ApiConfig(
        artifact_dir=root / "processed",
        experiment_dir=root / "experiment",
        checkpoint_dir=root / "checkpoints",
        raw_dir=root / "raw",
        device="cpu",
    )


def valid_forecast_body() -> dict:
    return {
        "feature_names": list(FEATURE_NAMES),
        "observations": np.ones((144, 8), dtype=np.float64).tolist(),
    }


class TetouanApiTests(unittest.TestCase):
    def test_defaults_preserve_repository_artifact_layout(self):
        config = ApiConfig()
        self.assertEqual(config.run_name, RUN_NAME)
        self.assertEqual(config.artifact_dir, REPO_ROOT / "data" / "processed" / "tetouan")
        self.assertEqual(config.experiment_dir, REPO_ROOT / "results" / "tetouan" / RUN_NAME)
        self.assertEqual(config.checkpoint_dir, REPO_ROOT / "checkpoints" / "tetouan")
        self.assertEqual(config.latest_experiment_id, EXTENDED_EXPERIMENT_ID)
        self.assertEqual(config.latest_experiment_dir, EXTENDED_EXPERIMENT_DIR)
        self.assertEqual(config.latest_checkpoint_dir, EXTENDED_CHECKPOINT_DIR)
        self.assertEqual(config.latest_evaluation_dir, EXTENDED_EVALUATION_DIR)

    def test_vite_local_origins_can_read_health_and_preflight_forecast(self):
        with TemporaryDirectory() as temp:
            app = create_app(api_config_for_test(Path(temp)),
                             store_factory=lambda **_: FakeStore(),
                             inference_factory=lambda **_: FakeInference())
            with TestClient(app) as client:
                response = client.get("/health", headers={"Origin": "http://localhost:8443"})
                preflight = client.options(
                    "/api/v1/forecasts/compare",
                    headers={
                        "Origin": "http://localhost:8443",
                        "Access-Control-Request-Method": "POST",
                        "Access-Control-Request-Headers": "content-type",
                    },
                )
            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.headers["access-control-allow-origin"],
                             "http://localhost:8443")
            self.assertEqual(preflight.status_code, 200)
            self.assertEqual(preflight.headers["access-control-allow-origin"],
                             "http://localhost:8443")

    def test_injected_services_health_and_single_initialization_per_lifecycle(self):
        with TemporaryDirectory() as temp:
            store = FakeStore()
            inference = FakeInference()
            store_calls, inference_calls = [], []

            def make_store(**kwargs):
                store_calls.append(kwargs)
                return store

            def make_inference(**kwargs):
                inference_calls.append(kwargs)
                return inference

            app = create_app(api_config_for_test(Path(temp)), store_factory=make_store,
                             inference_factory=make_inference)
            self.assertIsNone(app.state.inference_service)
            with TestClient(app) as client:
                self.assertIs(app.state.experiment_store, store)
                self.assertIs(app.state.inference_service, inference)
                for _ in range(3):
                    response = client.get("/health")
                    self.assertEqual(response.status_code, 200)
                    self.assertEqual(response.json(), {
                        "status": "ready",
                        "inference_service_available": True,
                        "experiment_store_available": True,
                        "selected_device": "cpu",
                        "experiment_id": EXTENDED_EXPERIMENT_ID,
                    })
                self.assertEqual(len(store_calls), 1)
                self.assertEqual(len(inference_calls), 1)
                self.assertEqual(store.verify_calls, 1)
            self.assertIsNone(app.state.experiment_store)
            self.assertIsNone(app.state.inference_service)

    def test_latest_evaluation_directory_is_passed_to_extended_store(self):
        with TemporaryDirectory() as temp:
            evaluation_dir = Path(temp) / "mounted-evaluation"
            extended_calls = []

            def make_extended_store(**kwargs):
                extended_calls.append(kwargs)
                return FakeStore()

            config = replace(api_config_for_test(Path(temp)), latest_evaluation_dir=evaluation_dir)
            app = create_app(
                config,
                store_factory=lambda **_: FakeStore(),
                extended_store_factory=make_extended_store,
                inference_factory=lambda **_: FakeInference(),
            )
            with TestClient(app):
                self.assertEqual(extended_calls[0]["evaluation_dir"], evaluation_dir)

    def test_artifact_verification_failure_is_degraded_and_sanitized(self):
        with TemporaryDirectory() as temp:
            store = FakeStore(status="issues")
            inference_calls = []
            app = create_app(
                api_config_for_test(Path(temp)), store_factory=lambda **_: store,
                inference_factory=lambda **kw: inference_calls.append(kw),
            )
            with TestClient(app) as client:
                response = client.get("/health")
                self.assertEqual(response.status_code, 503)
                self.assertEqual(response.json()["status"], "degraded")
                self.assertFalse(response.json()["experiment_store_available"])
                self.assertFalse(response.json()["inference_service_available"])
                self.assertNotIn("/private/data/secret", response.text)
                self.assertEqual(inference_calls, [])
                self.assertEqual(app.state.startup_error_type, "RuntimeError")

    def test_missing_artifact_constructor_failure_is_degraded_and_sanitized(self):
        with TemporaryDirectory() as temp:
            def missing_store(**_):
                raise FileNotFoundError("/private/root/missing/metadata.json")

            app = create_app(api_config_for_test(Path(temp)), store_factory=missing_store,
                             inference_factory=lambda **_: FakeInference())
            with TestClient(app) as client:
                response = client.get("/health")
                self.assertEqual(response.status_code, 503)
                self.assertNotIn("/private/root", response.text)
                self.assertNotIn("metadata.json", response.text)
                self.assertEqual(app.state.startup_error_type, "FileNotFoundError")

    def test_missing_checkpoint_failure_is_degraded_and_sanitized(self):
        with TemporaryDirectory() as temp:
            store = FakeStore()

            def missing_checkpoint(**_):
                raise FileNotFoundError("/private/root/checkpoints/secret.pt")

            app = create_app(api_config_for_test(Path(temp)), store_factory=lambda **_: store,
                             inference_factory=missing_checkpoint)
            with TestClient(app) as client:
                response = client.get("/health")
                self.assertEqual(response.status_code, 503)
                self.assertTrue(response.json()["experiment_store_available"])
                self.assertFalse(response.json()["inference_service_available"])
                self.assertNotIn("/private/root", response.text)
                self.assertNotIn("secret.pt", response.text)
                self.assertEqual(app.state.startup_error_type, "FileNotFoundError")

    def test_unhandled_request_exception_is_sanitized(self):
        with TemporaryDirectory() as temp:
            app = create_app(
                api_config_for_test(Path(temp)),
                store_factory=lambda **_: FakeStore(),
                inference_factory=lambda **_: FakeInference(),
            )

            @app.get("/internal-error-test")
            def raise_internal_error():
                raise RuntimeError("/private/path/secret.pt")

            with TestClient(app, raise_server_exceptions=False) as client:
                response = client.get("/internal-error-test")
            self.assertEqual(response.status_code, 500)
            self.assertEqual(response.json(), {
                "error": {"code": "internal_error", "message": "Internal server error."}
            })
            self.assertNotIn("/private/path", response.text)
            self.assertNotIn("secret.pt", response.text)

    def test_historical_endpoints_preserve_order_data_and_sanitize_paths(self):
        with TemporaryDirectory() as temp:
            store = HistoricalFakeStore()
            inference = FakeInference()
            app = create_app(api_config_for_test(Path(temp)),
                             store_factory=lambda **_: store,
                             inference_factory=lambda **_: inference)
            with TestClient(app) as client:
                model_response = client.get("/api/v1/models")
                dataset_response = client.get("/api/v1/dataset")
                experiment_response = client.get(f"/api/v1/experiments/{RUN_NAME}")
                history_response = client.get(f"/api/v1/experiments/{RUN_NAME}/history")
                metrics_response = client.get(f"/api/v1/experiments/{RUN_NAME}/metrics")

            for response in (model_response, dataset_response, experiment_response,
                             history_response, metrics_response):
                self.assertEqual(response.status_code, 200, response.text)
                self.assertNotIn("/Users/", response.text)
                self.assertNotIn("/content/", response.text)

            models = model_response.json()
            self.assertEqual([item["name"] for item in models],
                             ["LinearGridTransformer", "CDLinear"])
            self.assertEqual(models[0]["input_shape"], [144, 8])
            self.assertEqual(models[0]["forecast_output_shape"], [6, 3])
            self.assertEqual(models[0]["configuration"]["out_channels"], 3)
            self.assertTrue(all(item["loaded"] and item["available"] for item in models))

            dataset = dataset_response.json()
            self.assertEqual(dataset["row_count"], 52416)
            self.assertEqual(dataset["features"]["names"], list(FEATURE_NAMES))
            self.assertEqual(dataset["targets"]["names"], list(TARGET_NAMES))
            self.assertEqual(dataset["cadence_minutes"], 10)
            self.assertEqual(dataset["coverage_uncertainties"]["units_status"], "unknown")
            self.assertNotIn("source_path", dataset)

            experiment = experiment_response.json()
            self.assertEqual(experiment["run_identifier"], RUN_NAME)
            self.assertEqual(experiment["provenance"]["training_environment"]["gpu_name"], "Tesla T4")
            self.assertEqual(experiment["models"]["CDLinear"]["best_validation_epoch"], 8)
            self.assertEqual(experiment["models"]["CDLinear"]["checkpoint"]["name"], "CDLinear.pt")
            self.assertEqual(experiment["artifact_references"]["manifest"]["name"], "manifest.json")

            history = history_response.json()
            self.assertEqual(list(history["models"]), ["LinearGridTransformer", "CDLinear"])
            self.assertEqual(history["models"]["LinearGridTransformer"], store.history_rows)
            metrics = metrics_response.json()
            self.assertIn("per_zone", metrics["overall"]["models"]["CDLinear"])
            self.assertIn("macro_average", metrics["overall"]["baselines"]["training_mean"])
            self.assertEqual(metrics["horizon_wise"]["target_zone_names"], list(TARGET_NAMES))
            self.assertEqual(store.calls, ["dataset", "experiment", "history", "metrics"])
            self.assertEqual(inference.prediction_calls, 0)

    def test_unknown_experiment_id_returns_404(self):
        with TemporaryDirectory() as temp:
            store = HistoricalFakeStore()
            app = create_app(api_config_for_test(Path(temp)),
                             store_factory=lambda **_: store,
                             inference_factory=lambda **_: FakeInference())
            with TestClient(app) as client:
                response = client.get("/api/v1/experiments/other-run")
            self.assertEqual(response.status_code, 404)
            self.assertEqual(response.json()["error"]["code"], "experiment_not_found")
            self.assertEqual(store.calls, [])

    def test_missing_and_malformed_artifacts_map_to_sanitized_errors(self):
        with TemporaryDirectory() as temp:
            store = HistoricalFakeStore()

            def missing():
                raise MissingArtifactError("missing: /Users/private/history.json")

            store.get_training_history = missing
            app = create_app(api_config_for_test(Path(temp)),
                             store_factory=lambda **_: store,
                             inference_factory=lambda **_: FakeInference())
            with TestClient(app) as client:
                response = client.get(f"/api/v1/experiments/{RUN_NAME}/history")
            self.assertEqual(response.status_code, 503)
            self.assertEqual(response.json()["error"]["code"], "artifact_unavailable")
            self.assertNotIn("/Users/", response.text)
            self.assertNotIn("history.json", response.text)

        with TemporaryDirectory() as temp:
            store = HistoricalFakeStore()

            def malformed():
                raise InvalidArtifactError("Malformed report at /content/private/metrics.json")

            store.get_evaluation_info = malformed
            app = create_app(api_config_for_test(Path(temp)),
                             store_factory=lambda **_: store,
                             inference_factory=lambda **_: FakeInference())
            with TestClient(app) as client:
                response = client.get(f"/api/v1/experiments/{RUN_NAME}/metrics")
            self.assertEqual(response.status_code, 500)
            self.assertEqual(response.json()["error"]["code"], "invalid_artifact")
            self.assertNotIn("/content/", response.text)
            self.assertNotIn("metrics.json", response.text)

    def test_single_model_forecasts_and_comparison_response_contract(self):
        with TemporaryDirectory() as temp:
            inference = ForecastFakeInference()
            app = create_app(api_config_for_test(Path(temp)),
                             store_factory=lambda **_: FakeStore(),
                             inference_factory=lambda **_: inference)
            body = valid_forecast_body()
            with TestClient(app) as client:
                transformer = client.post(
                    "/api/v1/forecasts/LinearGridTransformer", json=body
                )
                cdlinear = client.post("/api/v1/forecasts/CDLinear", json=body)
                comparison = client.post("/api/v1/forecasts/compare", json=body)

            for response in (transformer, cdlinear, comparison):
                self.assertEqual(response.status_code, 200, response.text)
            result = transformer.json()
            self.assertEqual(result["experiment_id"], EXTENDED_EXPERIMENT_ID)
            self.assertEqual(result["model_name"], "LinearGridTransformer")
            self.assertEqual(result["input_dimensions"], [144, 8])
            self.assertEqual(result["output_dimensions"], [6, 3])
            self.assertEqual(result["target_zone_names"], list(TARGET_NAMES))
            self.assertEqual([step["minutes_ahead"] for step in result["forecast_steps"]],
                             [10, 20, 30, 40, 50, 60])
            self.assertTrue(np.isfinite(np.asarray(
                [step["values"] for step in result["forecast_steps"]]
            )).all())
            self.assertEqual(result["units_status"], "unknown; numeric dataset values")
            self.assertEqual(cdlinear.json()["model_name"], "CDLinear")

            compared = comparison.json()
            self.assertEqual(list(compared["models"]), ["LinearGridTransformer", "CDLinear"])
            self.assertEqual(compared["input_dimensions"], [144, 8])
            for model in compared["models"].values():
                self.assertEqual(model["output_dimensions"], [6, 3])
                self.assertEqual(len(model["forecast_steps"]), 6)
                self.assertTrue(np.isfinite(np.asarray(
                    [step["values"] for step in model["forecast_steps"]]
                )).all())
            self.assertEqual(len(inference.single_calls), 2)
            self.assertEqual(len(inference.both_calls), 1)
            self.assertEqual(inference.both_calls[0][1], list(FEATURE_NAMES))

    def test_forecast_schema_rejects_bad_dimensions_order_features_and_nonfinite_values(self):
        with TemporaryDirectory() as temp:
            app = create_app(api_config_for_test(Path(temp)),
                             store_factory=lambda **_: FakeStore(),
                             inference_factory=lambda **_: ForecastFakeInference())
            cases = []

            short = valid_forecast_body()
            short["observations"] = short["observations"][:-1]
            cases.append(short)
            long = valid_forecast_body()
            long["observations"].append([1.0] * 8)
            cases.append(long)
            reordered = valid_forecast_body()
            reordered["feature_names"].reverse()
            cases.append(reordered)
            incorrect_name = valid_forecast_body()
            incorrect_name["feature_names"][3] = "air_temperature"
            cases.append(incorrect_name)
            missing_feature = valid_forecast_body()
            missing_feature["feature_names"] = missing_feature["feature_names"][:-1]
            cases.append(missing_feature)
            extra_feature = valid_forecast_body()
            extra_feature["feature_names"].append("extra")
            cases.append(extra_feature)
            bad_row = valid_forecast_body()
            bad_row["observations"][7] = [1.0] * 7
            cases.append(bad_row)
            extra_value = valid_forecast_body()
            extra_value["observations"][7] = [1.0] * 9
            cases.append(extra_value)
            wrong_time_count = valid_forecast_body()
            wrong_time_count["timestamps"] = ["2017-01-01 00:00:00"] * 143
            cases.append(wrong_time_count)
            nonnumeric = valid_forecast_body()
            nonnumeric["observations"][0][0] = "not-a-number"
            cases.append(nonnumeric)
            for invalid in (float("nan"), float("inf"), float("-inf")):
                nonfinite = valid_forecast_body()
                nonfinite["observations"][0][0] = invalid
                cases.append(nonfinite)

            with TestClient(app) as client:
                for body in cases:
                    contains_nonfinite = False
                    for row in body["observations"]:
                        for value in row:
                            try:
                                contains_nonfinite |= not math.isfinite(float(value))
                            except (TypeError, ValueError):
                                continue
                    if contains_nonfinite:
                        response = client.post(
                            "/api/v1/forecasts/CDLinear",
                            content=json.dumps(body, allow_nan=True),
                            headers={"content-type": "application/json"},
                        )
                    else:
                        response = client.post("/api/v1/forecasts/CDLinear", json=body)
                    self.assertEqual(response.status_code, 422, response.text)
                    self.assertEqual(response.json()["error"]["code"], "invalid_request")
                    self.assertNotIn("observations", response.text)

    def test_forecast_unknown_model_unavailable_service_and_failure_are_sanitized(self):
        with TemporaryDirectory() as temp:
            app = create_app(api_config_for_test(Path(temp)),
                             store_factory=lambda **_: FakeStore(),
                             inference_factory=lambda **_: ForecastFakeInference())
            with TestClient(app) as client:
                unknown = client.post("/api/v1/forecasts/unknown", json=valid_forecast_body())
            self.assertEqual(unknown.status_code, 404)
            self.assertEqual(unknown.json()["error"]["code"], "model_not_found")

        with TemporaryDirectory() as temp:
            def missing_inference(**_):
                raise FileNotFoundError("/private/checkpoints/model.pt")

            app = create_app(api_config_for_test(Path(temp)),
                             store_factory=lambda **_: FakeStore(),
                             inference_factory=missing_inference)
            with TestClient(app) as client:
                unavailable = client.post("/api/v1/forecasts/CDLinear", json=valid_forecast_body())
            self.assertEqual(unavailable.status_code, 503)
            self.assertNotIn("/private", unavailable.text)

        with TemporaryDirectory() as temp:
            inference = ForecastFakeInference()

            def fail(*args, **kwargs):
                raise RuntimeError("/private/internal/model failure")

            inference.predict = fail
            app = create_app(api_config_for_test(Path(temp)),
                             store_factory=lambda **_: FakeStore(),
                             inference_factory=lambda **_: inference)
            with TestClient(app) as client:
                failed = client.post("/api/v1/forecasts/CDLinear", json=valid_forecast_body())
            self.assertEqual(failed.status_code, 500)
            self.assertNotIn("/private", failed.text)
            self.assertNotIn("model failure", failed.text)

    def test_actual_cpu_startup_loads_both_checkpoints_without_test_dataset(self):
        app = create_app(replace(ApiConfig(), device="cpu"))
        with TestClient(app) as client:
            response = client.get("/health")
            self.assertEqual(response.status_code, 200, response.text)
            self.assertEqual(response.json()["selected_device"], "cpu")
            service = app.state.inference_service
            self.assertIsInstance(service, TetouanInferenceService)
            self.assertEqual(set(service.models), {"LinearGridTransformer", "CDLinear"})
            self.assertEqual(set(service.checkpoints), {"LinearGridTransformer", "CDLinear"})
            self.assertTrue(all(not model.training for model in service.models.values()))
            self.assertEqual(app.state.experiment_store.verify_artifacts()["status"], "ok")
            actual_responses = {
                "models": client.get("/api/v1/models"),
                "dataset": client.get("/api/v1/dataset"),
                "experiment": client.get(f"/api/v1/experiments/{RUN_NAME}"),
                "history": client.get(f"/api/v1/experiments/{RUN_NAME}/history"),
                "metrics": client.get(f"/api/v1/experiments/{RUN_NAME}/metrics"),
            }
            for name, result in actual_responses.items():
                self.assertEqual(result.status_code, 200, f"{name}: {result.text}")
                self.assertNotIn("/Users/", result.text)
                self.assertNotIn("/content/", result.text)
            self.assertEqual(actual_responses["dataset"].json()["row_count"], 52416)
            self.assertEqual(actual_responses["history"].json()["models"].keys(),
                             {"LinearGridTransformer", "CDLinear"})
            metrics = actual_responses["metrics"].json()
            self.assertIn("baselines", metrics["overall"])
            self.assertIn("horizon_wise", metrics)

    def test_latest_three_seed_experiment_api_metadata_and_metrics(self):
        store = TetouanExtendedExperimentStore()
        self.assertEqual(store.verify_artifacts()["status"], "ok")
        roles = store.get_model_roles()
        self.assertEqual(roles["CDLinear"], {"role": "selected_forecaster", "selected": True})
        self.assertEqual(roles["LinearGridTransformer"], {
            "role": "experimental_alternative", "selected": False
        })
        direct_metrics = store.get_evaluation_info()
        summary = direct_metrics["overall"]["models"]["CDLinear"]["three_seed_summary"]
        self.assertAlmostEqual(summary["MAE_mean"], 525.1009881813567)
        self.assertAlmostEqual(summary["RMSE_mean"], 713.7357624882283)
        self.assertEqual(len(direct_metrics["overall"]["models"]["CDLinear"]["per_seed"]), 3)
        transformer_summary = direct_metrics["overall"]["models"]["LinearGridTransformer"]["three_seed_summary"]
        self.assertAlmostEqual(transformer_summary["MAE_mean"], 1904.0782611114637)
        self.assertAlmostEqual(transformer_summary["RMSE_mean"], 2386.8759484967554)
        persistence = direct_metrics["overall"]["baselines"]["persistence"]
        self.assertAlmostEqual(persistence["MAE"], 881.2094120374148)
        self.assertAlmostEqual(persistence["RMSE"], 1424.0505543265078)
        self.assertEqual(len(direct_metrics["overall"]["models"]["CDLinear"]["per_zone"]), 3)
        self.assertEqual(len(direct_metrics["horizon_wise"]["models"]["CDLinear"]), 18)

        app = create_app(replace(ApiConfig(), device="cpu"))
        with TestClient(app) as client:
            model_response = client.get("/api/v1/models")
            experiment_response = client.get(f"/api/v1/experiments/{EXTENDED_EXPERIMENT_ID}")
            history_response = client.get(f"/api/v1/experiments/{EXTENDED_EXPERIMENT_ID}/history")
            metrics_response = client.get(f"/api/v1/experiments/{EXTENDED_EXPERIMENT_ID}/metrics")
        for response in (model_response, experiment_response, history_response, metrics_response):
            self.assertEqual(response.status_code, 200, response.text)
            self.assertNotIn("/Users/", response.text)
            self.assertNotIn("/content/", response.text)
        models = {row["name"]: row for row in model_response.json()}
        self.assertEqual(models["CDLinear"]["role"], "selected_forecaster")
        self.assertTrue(models["CDLinear"]["selected"])
        self.assertEqual(models["LinearGridTransformer"]["role"], "experimental_alternative")
        record = experiment_response.json()
        self.assertEqual(record["run_identifier"], EXTENDED_EXPERIMENT_ID)
        self.assertEqual(record["models"]["CDLinear"]["checkpoint_seed"], 2026)
        self.assertEqual(len(record["models"]["CDLinear"]["seed_runs"]), 3)
        self.assertEqual(record["baselines"]["Persistence"]["trained"], False)
        history = history_response.json()
        self.assertEqual(len(history["models"]["CDLinear"]), 50)
        self.assertEqual(len(history["per_seed"]["CDLinear"]), 3)
        latest_metrics = metrics_response.json()
        self.assertEqual(latest_metrics["experiment_id"], EXTENDED_EXPERIMENT_ID)
        self.assertEqual(latest_metrics["test_split"]["seeds"], [42, 123, 2026])
        self.assertAlmostEqual(
            latest_metrics["overall"]["models"]["CDLinear"]["three_seed_summary"]["MAE_mean"],
            525.1009881813567,
        )

    def test_extended_store_reports_missing_artifacts_without_claiming_availability(self):
        with TemporaryDirectory() as temp:
            store = TetouanExtendedExperimentStore(
                experiment_dir=Path(temp) / "experiment",
                checkpoint_dir=Path(temp) / "checkpoints",
                evaluation_dir=Path(temp) / "evaluation",
                processed_dir=Path(temp) / "processed",
            )
            verification = store.verify_artifacts()
            self.assertEqual(verification["status"], "issues")
            self.assertTrue(verification["issues"])
            self.assertFalse(verification["checks"]["evaluation_report"]["available"])

    def test_actual_cpu_http_forecasts_from_training_rows_and_shared_comparison_input(self):
        # Read only the first 144 raw source rows; these timestamps are in train.
        raw = pd.read_csv(DEFAULT_SOURCE, nrows=144)
        source_columns = [source for _, source in FEATURE_MAPPING]
        observations = raw.loc[:, source_columns].to_numpy(dtype=np.float64)
        timestamps = pd.DatetimeIndex(
            pd.to_datetime(raw["DateTime"], format="%m/%d/%Y %H:%M")
        )
        body = {
            "feature_names": list(FEATURE_NAMES),
            "observations": observations.tolist(),
            "timestamps": timestamps.strftime("%Y-%m-%d %H:%M:%S").tolist(),
        }
        app = create_app(replace(ApiConfig(), device="cpu"))
        with TestClient(app) as client:
            service = app.state.inference_service
            for name in ("LinearGridTransformer", "CDLinear"):
                response = client.post(f"/api/v1/forecasts/{name}", json=body)
                self.assertEqual(response.status_code, 200, response.text)
                self.assertEqual(response.json()["experiment_id"], EXTENDED_EXPERIMENT_ID)
                self.assertNotIn("/Users/", response.text)
                self.assertNotIn("/content/", response.text)
                result = response.json()
                self.assertEqual(result["input_dimensions"], [144, 8])
                self.assertEqual(result["output_dimensions"], [6, 3])
                values = np.asarray([step["values"] for step in result["forecast_steps"]])
                self.assertEqual(values.shape, (6, 3))
                self.assertTrue(np.isfinite(values).all())
                self.assertEqual(result["target_zone_names"], list(TARGET_NAMES))

            for invalid_timestamps in (
                ["not-a-timestamp"] * 144,
                timestamps[::-1].strftime("%Y-%m-%d %H:%M:%S").tolist(),
                [*timestamps.strftime("%Y-%m-%d %H:%M:%S").tolist()[:20],
                 "2017-01-01 03:21:00", *timestamps.strftime("%Y-%m-%d %H:%M:%S").tolist()[21:]],
                timestamps.tz_localize("UTC").strftime("%Y-%m-%d %H:%M:%S%z").tolist(),
            ):
                bad = {**body, "timestamps": invalid_timestamps}
                response = client.post("/api/v1/forecasts/CDLinear", json=bad)
                self.assertEqual(response.status_code, 422, response.text)
                self.assertNotIn("observations", response.text)

            prepared = []
            original_predict_prepared = service._predict_prepared

            def record_prepared(model_name, x):
                prepared.append((model_name, x))
                return original_predict_prepared(model_name, x)

            with patch.object(service, "_predict_prepared", side_effect=record_prepared):
                compared = client.post("/api/v1/forecasts/compare", json=body)
            self.assertEqual(compared.status_code, 200, compared.text)
            self.assertEqual(compared.json()["experiment_id"], EXTENDED_EXPERIMENT_ID)
            self.assertNotIn("/Users/", compared.text)
            self.assertNotIn("/content/", compared.text)
            comparison_result = compared.json()
            self.assertEqual(list(comparison_result["models"]),
                             ["LinearGridTransformer", "CDLinear"])
            self.assertEqual(comparison_result["input_dimensions"], [144, 8])
            self.assertEqual(comparison_result["target_zone_names"], list(TARGET_NAMES))
            self.assertEqual([step for step in range(1, 7)],
                             [item["horizon_step"] for item in
                              comparison_result["models"]["CDLinear"]["forecast_steps"]])
            self.assertEqual([10, 20, 30, 40, 50, 60],
                             [item["minutes_ahead"] for item in
                              comparison_result["models"]["CDLinear"]["forecast_steps"]])
            for model_result in comparison_result["models"].values():
                self.assertEqual(model_result["output_dimensions"], [6, 3])
                self.assertEqual(len(model_result["forecast_steps"]), 6)
                self.assertEqual([10, 20, 30, 40, 50, 60],
                                 [item["minutes_ahead"] for item in
                                  model_result["forecast_steps"]])
                values = np.asarray([step["values"] for step in model_result["forecast_steps"]])
                self.assertEqual(values.shape, (6, 3))
                self.assertTrue(np.isfinite(values).all())
            self.assertEqual(len(prepared), 2)
            self.assertIs(prepared[0][1], prepared[1][1])


if __name__ == "__main__":
    unittest.main()
