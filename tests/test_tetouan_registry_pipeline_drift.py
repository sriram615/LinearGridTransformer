from __future__ import annotations

import hashlib
import json
import os
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path

import numpy as np
import torch

from src.monitoring.tetouan_drift import feature_drift, log_monitoring_report, main as drift_main, performance_report, prediction_drift, repeated_alert
from src.pipelines.tetouan_pipeline import plan, run_pipeline

ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "runtime-artifacts" / "data" / "processed" / "tetouan"


class PipelineTests(unittest.TestCase):
    def test_dry_run_validates_fingerprints_and_does_not_write(self):
        with tempfile.TemporaryDirectory() as temp:
            result = run_pipeline(run_name="audit-dry-run", seeds=[42, 123], artifact_dir=ARTIFACTS, output_root=Path(temp), dry_run=True)
            self.assertEqual(result["status"], "dry_run")
            self.assertEqual(result["plan"]["artifact_fingerprints"]["row_count"], 52416)
            self.assertEqual(result["plan"]["seeds"], [42, 123])
            self.assertFalse((Path(temp) / "audit-dry-run").exists())

    def test_mock_pipeline_records_parent_children_test_metrics_and_manifest(self):
        import mlflow
        from mlflow.tracking import MlflowClient
        from src.training import train_tetouan, evaluate_tetouan
        from src.mlops import model_registry

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            uri = f"sqlite:///{root / 'tracking.db'}"
            env = {"MLFLOW_TRACKING_URI": uri, "MLFLOW_ARTIFACT_ROOT": str(root / "mlflow-artifacts")}
            calls = []
            events = []
            def fake_train(config, *, checkpoint_dir, results_dir, artifact_dir, run_name):
                calls.append((config.seed, str(artifact_dir)))
                events.append(("train", config.seed))
                outputs = {}
                for name, suffix in (("LinearGridTransformer", "lineargridtransformer"), ("CDLinear", "cdlinear")):
                    checkpoint = checkpoint_dir / f"{run_name}_{suffix}.pt"
                    torch.save({"run_metadata": {"local_path": str(root / "local")} , "model_state_dict": {"w": torch.ones(1)}}, checkpoint)
                    outputs[name] = {"checkpoint": str(checkpoint), "history": [{"epoch": 1, "train_loss": .2, "validation_loss": .3}], "best_epoch": 1, "best_validation_loss": .3}
                (results_dir / f"{run_name}_history.json").write_text(json.dumps({"checkpoint": str(root / "private/path")}))
                (results_dir / f"{run_name}_manifest.json").write_text(json.dumps({"run": run_name}))
                return {"models": outputs}
            def fake_evaluate(lgt, cdlinear, *, artifact_dir, results_dir, output_name, device):
                events.append(("evaluate", output_name))
                report = {"metrics": {"models": {name: {"macro_average_across_zones": {"mae": 1., "rmse": 2.}, "per_zone": {zone: {"mae": 1., "rmse": 2.} for zone in ("zone_1", "zone_2", "zone_3")}} for name in ("LinearGridTransformer", "CDLinear")}, "baselines": {baseline: {"macro_average_across_zones": {"mae": 3., "rmse": 4.}, "pooled_across_all_samples_horizons_and_zones": {"mae": 3., "rmse": 4.}} for baseline in ("training_mean", "persistence")}}}
                (results_dir / output_name).write_text(json.dumps(report))
                return report
            mlflow.set_tracking_uri(uri)
            mlflow.create_experiment("Tetouan Energy Forecasting", artifact_location=(root / "mlflow-artifacts").as_uri())
            def fake_register(checkpoint, source_run_id, *, artifact_dir):
                events.append(("register", Path(checkpoint).name))
                return {"registered_model": "test", "version": "1", "source_run_id": source_run_id}
            with patch.dict(os.environ, env), patch.object(train_tetouan, "run_training", side_effect=fake_train), patch.object(evaluate_tetouan, "evaluate_checkpoints", side_effect=fake_evaluate), patch.object(model_registry, "register_checkpoint", side_effect=fake_register):
                result = run_pipeline(run_name="mock-run", seeds=[42, 123], artifact_dir=ARTIFACTS, output_root=root / "runs", source_csv=ROOT / "data/raw/tetouan/Tetuan City power consumption.csv", epochs=1, dry_run=False, register=True)
            self.assertEqual(result["status"], "completed")
            self.assertEqual([seed for seed, _ in calls], [42, 123])
            first_registration = next(index for index, event in enumerate(events) if event[0] == "register")
            last_evaluation = max(index for index, event in enumerate(events) if event[0] == "evaluate")
            self.assertGreater(first_registration, last_evaluation)
            self.assertEqual(sum(event[0] == "register" for event in events), 4)
            self.assertTrue((root / "runs/mock-run/pipeline_manifest.json").is_file())
            client = MlflowClient(tracking_uri=uri)
            runs = client.search_runs(["1"], max_results=20)
            parent = next(run for run in runs if run.data.tags.get("run_role") == "pipeline_parent")
            children = [run for run in runs if run.data.tags.get("run_role") == "model_seed_child"]
            self.assertEqual(len(children), 4)
            self.assertTrue(all(run.data.tags.get("mlflow.parentRunId") == parent.info.run_id for run in children))
            self.assertIn("training.loss", {metric.key for metric in client.get_metric_history(children[0].info.run_id, "training.loss")})
            self.assertIn("test.macro.mae", set(client.get_run(children[0].info.run_id).data.metrics))
            import tempfile as _tempfile
            with _tempfile.TemporaryDirectory() as download_dir:
                downloaded = Path(client.download_artifacts(parent.info.run_id, "pipeline_manifest.json", dst_path=download_dir))
                logged_manifest = downloaded.read_text()
                self.assertNotIn(str(root), logged_manifest)
                self.assertNotIn("/Users/apple", logged_manifest)

    def test_collision_and_duplicate_seeds_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "existing").mkdir()
            with self.assertRaises(FileExistsError):
                plan("existing", [42], ARTIFACTS, root)
            with self.assertRaisesRegex(ValueError, "distinct"):
                plan("fresh", [42, 42], ARTIFACTS, root)


class DriftTests(unittest.TestCase):
    def test_identical_and_shifted_feature_distributions(self):
        rng = np.random.default_rng(8)
        reference = rng.normal(size=(1000, 8))
        identical = feature_drift(reference, reference.copy())
        self.assertFalse(identical["drift_detected"])
        shifted = feature_drift(reference, reference + np.array([3, 0, 0, 0, 0, 0, 0, 0]))
        self.assertEqual(shifted["features"]["zone_1_power_consumption"]["level"], "alert")
        self.assertEqual(shifted["sample_counts"], {"reference": 1000, "current": 1000})
        constant_shift = feature_drift(np.zeros((30, 8)), np.column_stack([np.ones(30), np.zeros((30, 7))]))
        self.assertNotEqual(constant_shift["features"]["zone_1_power_consumption"]["psi"], 0.0)

    def test_prediction_drift_zone_horizon_and_interpretation(self):
        rng = np.random.default_rng(3)
        reference = rng.normal(size=(120, 6, 3))
        current = reference.copy()
        current[:, 2, 1] += 4
        report = prediction_drift(reference, current, model_version="CDLinear/2", experiment_id="exp")
        self.assertEqual(len(report["zone_horizon"]), 18)
        self.assertEqual(report["zone_order"], ["zone_1_power_consumption", "zone_2_power_consumption", "zone_3_power_consumption"])
        self.assertEqual(report["zone_horizon"]["zone_2_power_consumption__h3"]["level"], "alert")
        self.assertIn("does not establish", report["interpretation"])

    def test_performance_requires_aligned_ground_truth(self):
        prediction = np.ones((4, 6, 3))
        actual = np.zeros_like(prediction)
        names = ("zone_1_power_consumption", "zone_2_power_consumption", "zone_3_power_consumption")
        report = performance_report(prediction, actual, {"per_zone": {name: {"mae": .5, "rmse": .5} for name in names}, "per_horizon_minutes": {str(i * 10): {"mae": .5, "rmse": .5} for i in range(1, 7)}})
        self.assertEqual(report["per_zone"][names[1]], {"mae": 1.0, "rmse": 1.0})
        self.assertEqual(report["per_horizon_minutes"]["60"], {"mae": 1.0, "rmse": 1.0})
        self.assertEqual(report["comparison_to_validation_reference"]["per_zone"][names[1]], {"mae_higher_than_reference": True, "rmse_higher_than_reference": True})
        self.assertEqual(report["comparison_to_validation_reference"]["per_horizon_minutes"]["60"], {"mae_higher_than_reference": True, "rmse_higher_than_reference": True})

    def test_invalid_values_shapes_and_repeated_alerts(self):
        with self.assertRaisesRegex(ValueError, "non-finite"):
            feature_drift(np.zeros((5, 8)), np.full((5, 8), np.nan))
        with self.assertRaisesRegex(ValueError, "shape"):
            prediction_drift(np.zeros((5, 5, 3)), np.zeros((5, 5, 3)), model_version="v1", experiment_id="exp")
        with self.assertRaisesRegex(ValueError, "match"):
            performance_report(np.zeros((2, 6, 3)), np.zeros((3, 6, 3)))
        self.assertFalse(repeated_alert([{"drift_detected": True}] * 2, consecutive=3))
        self.assertTrue(repeated_alert([{"drift_detected": False}] + [{"drift_detected": True}] * 3, consecutive=3))


    def test_monitor_cli_reads_training_reference_and_applies_prior_alerts_without_logging_rows(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            current = np.full((12, 8), 100000.0, dtype=np.float64)
            current_path = root / "incoming.npy"
            np.save(current_path, current)
            history_dir = root / "history"
            history_dir.mkdir()
            for index in (1, 2):
                (history_dir / f"window-{index}.json").write_text(json.dumps({"any_distribution_drift": True}))
            output = root / "report.json"
            drift_main(["--current-features", str(current_path), "--output", str(output), "--alert-history-dir", str(history_dir)])
            report = json.loads(output.read_text())
            self.assertEqual(report["reference_partition"]["name"], "train")
            self.assertTrue(report["any_distribution_drift"])
            self.assertTrue(report["repeated_alert"])
            self.assertEqual(report["alert_policy"]["prior_reports_considered"], 2)
            self.assertNotIn("100000", output.read_text())

    def test_monitoring_metrics_and_aggregate_only_artifacts_logged_to_mlflow(self):
        import mlflow
        from mlflow.tracking import MlflowClient
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            uri = f"sqlite:///{root / 'monitoring.db'}"
            mlflow.set_tracking_uri(uri)
            experiment_id = mlflow.create_experiment("Tetouan Energy Forecasting", artifact_location=(root / "artifacts").as_uri())
            report = feature_drift(np.zeros((20, 8)), np.zeros((20, 8)))
            report.update({"reference_fingerprint": "abc123", "model_version": "CDLinear/4", "experiment_id": "tetouan-run"})
            run_id = log_monitoring_report(report, tracking_uri=uri)
            client = MlflowClient(tracking_uri=uri)
            run = client.get_run(run_id)
            self.assertEqual(run.data.tags["contains_raw_observations"], "false")
            self.assertEqual(run.data.params["reference_fingerprint"], "abc123")
            artifacts = client.list_artifacts(run_id, "monitoring")
            self.assertEqual({item.path.rsplit("/", 1)[-1] for item in artifacts}, {"monitoring_report.json", "monitoring_report.md"})


class RegistryTests(unittest.TestCase):
    def test_hash_lineage_idempotency_and_prediction_round_trip(self):
        try:
            import mlflow
            from mlflow.tracking import MlflowClient
        except ImportError:
            self.skipTest("MLflow is unavailable")
        from src.mlops.model_registry import register_checkpoint, sha256_file
        from src.training.evaluate_tetouan import dataset_contract
        from src.training.train import LinearGridForecaster

        metadata = json.loads((ARTIFACTS / "metadata.json").read_text())
        fingerprints = {name: hashlib.sha256((ARTIFACTS / name).read_bytes()).hexdigest() for name in ("metadata.json", "series.npz")}
        config = {"in_channels": 8, "out_channels": 3, "lookback": 144, "horizon": 6, "d_model": 64, "num_heads": 4, "ffn_dim": 256, "dropout": 0.1}
        model = LinearGridForecaster(in_channels=8, out_channels=3, t_forecast=6, d_model=64, num_heads=4, ffn_dim=256, dropout=0.1)
        dataset = dataset_contract(metadata)
        payload = {"model_name": "LinearGridTransformer", "model_state_dict": model.state_dict(), "model_config": config,
                   "checkpoint_metadata": {"selection_metric": "validation_loss", "best_epoch": 1, "best_validation_loss": .5},
                   "history": [{"epoch": 1, "is_best": True, "validation_loss": .5}],
                   "run_metadata": {"run_type": "experiment", "test_partition_used": False, "dataset": dataset,
                                    "models": {"LinearGridTransformer": config}, "training": {"seed": 42},
                                    "processed_artifacts": {name: {"sha256": digest} for name, digest in fingerprints.items()}}}
        with tempfile.TemporaryDirectory() as temp:
            temp_path = Path(temp)
            checkpoint = temp_path / "model.pt"
            torch.save(payload, checkpoint)
            uri = f"sqlite:///{temp_path / 'tracking.db'}"
            mlflow.set_tracking_uri(uri)
            exp_id = mlflow.create_experiment("Tetouan Energy Forecasting", artifact_location=(temp_path / "artifacts").as_uri())
            with mlflow.start_run(experiment_id=exp_id) as source:
                source_id = source.info.run_id
                mlflow.set_tag("model_name", "LinearGridTransformer")
            first = register_checkpoint(checkpoint, source_id, artifact_dir=ARTIFACTS, tracking_uri=uri)
            second = register_checkpoint(checkpoint, source_id, artifact_dir=ARTIFACTS, tracking_uri=uri)
            self.assertEqual(first["version"], second["version"])
            self.assertTrue(second["reused"])
            self.assertEqual(first["checkpoint_sha256"], sha256_file(checkpoint))
            client = MlflowClient(tracking_uri=uri)
            version = client.get_model_version(first["registered_model"], first["version"])
            self.assertEqual(version.tags["source_run_id"], source_id)
            self.assertEqual(version.tags["seed"], "42")
            loaded = mlflow.pytorch.load_model(version.source)
            model.eval()
            x = torch.zeros(2, 144, 8)
            with torch.no_grad():
                self.assertTrue(torch.allclose(loaded(x), model(x)))
            self.assertEqual(client.get_run(source_id).data.tags["model_name"], "LinearGridTransformer")


if __name__ == "__main__":
    unittest.main()
