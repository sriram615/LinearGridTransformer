from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import mlflow
from mlflow.tracking import MlflowClient

from src.mlops.import_extended_tetouan_to_mlflow import (
    MODELS, SEEDS, ImportConflictError, import_extended_experiment,
)
from src.mlops.mlflow_config import EXPERIMENT_NAME


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def source_tree(root: Path):
    experiment_dir = root / "experiment"
    checkpoint_dir = root / "checkpoints"
    processed_dir = root / "processed"
    evaluation_dir = experiment_dir / "evaluation_test_v1"
    processed_dir.mkdir(parents=True)
    evaluation_dir.mkdir(parents=True)
    write_json(processed_dir / "metadata.json", {
        "row_count": 52416, "source": {"sha256": "dataset-fingerprint"},
        "features": {"count": 8}, "targets": {"count": 3},
        "windows": {"lookback": 144, "horizon": 6},
        "timestamps": {"cadence_minutes": 10},
    })
    overall, zones, horizons = [], [], []
    for seed in SEEDS:
        name = f"tetouan_extended_v2_s{seed}"
        configs = {
            "LinearGridTransformer": {"in_channels": 8, "out_channels": 3, "lookback": 144, "horizon": 6, "d_model": 64},
            "CDLinear": {"in_channels": 8, "out_channels": 3, "seq_len": 144, "pred_len": 6},
        }
        records, model_histories = {}, {}
        for model in MODELS:
            rows = [
                {"epoch": epoch, "train_loss": 1.0 / epoch, "validation_loss": 2.0 / epoch}
                for epoch in range(1, 51)
            ]
            records[model] = {
                "configuration": configs[model], "best_validation_epoch": 50,
                "best_validation_loss": 0.04,
            }
            model_histories[model] = {"best_epoch": 50, "best_validation_loss": 0.04, "history": rows}
            suffix = "lineargridtransformer" if model == "LinearGridTransformer" else "cdlinear"
            path = checkpoint_dir / f"{name}_{suffix}.pt"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(f"checkpoint {seed} {model}".encode())
        write_json(experiment_dir / f"{name}_manifest.json", {
            "run_metadata": {
                "run_name": name, "run_type": "experiment", "test_partition_used": False,
                "training": {"seed": seed, "epochs": 50, "batch_size": 32, "num_workers": 0,
                             "learning_rate": 0.001, "weight_decay": 0.01, "grad_clip_norm": 1.0},
                "optimizer": "AdamW", "loss": "MSE", "python_version": "3.13.16",
                "pytorch_version": "2.11.0+cu130", "cuda_version": "13.0",
                "gpu_name": "Tesla T4", "git_commit": None,
                "dataset": {"source_sha256": "dataset-fingerprint", "lookback": 144, "horizon": 6},
            },
            "models": records,
        })
        write_json(experiment_dir / f"{name}_history.json", {
            "run_metadata": {"run_name": name}, "models": model_histories,
        })
    for model in MODELS:
        for seed in SEEDS:
            overall.append({"model": model, "seed": seed, "MAE": 10.0, "RMSE": 12.0, "macro_zone_MAE": 10.0})
            for zone in ("Zone 1", "Zone 2", "Zone 3"):
                zones.append({"model": model, "seed": seed, "zone": zone, "MAE": 10.0, "RMSE": 12.0})
            for step in range(1, 7):
                horizons.append({"model": model, "seed": seed, "horizon_step": step,
                                 "minutes_ahead": step * 10, "MAE": 10.0 + step, "RMSE": 12.0 + step})
    overall.append({"model": "Persistence", "seed": "baseline", "MAE": 20.0, "RMSE": 22.0, "macro_zone_MAE": 20.0})
    for zone in ("Zone 1", "Zone 2", "Zone 3"):
        zones.append({"model": "Persistence", "seed": "baseline", "zone": zone, "MAE": 20.0, "RMSE": 22.0})
    for step in range(1, 7):
        horizons.append({"model": "Persistence", "seed": "baseline", "horizon_step": step,
                         "minutes_ahead": step * 10, "MAE": 20.0 + step, "RMSE": 22.0 + step})
    summary = [{"model": model, "MAE_mean": 10.0, "MAE_std": 0.5,
                "RMSE_mean": 12.0, "RMSE_std": 0.6} for model in MODELS]
    write_json(evaluation_dir / "evaluation_report.json", {
        "experiment": "extended_training_v2", "test_windows": 7858,
        "three_seed_summary": summary, "overall_metrics": overall,
        "per_zone_metrics": zones, "per_horizon_metrics": horizons,
    })
    for name in ("overall_metrics_by_seed.csv", "per_zone_metrics_by_seed.csv",
                 "per_horizon_metrics_by_seed.csv", "three_seed_summary.csv"):
        (evaluation_dir / name).write_text("model,metric\nCDLinear,1.0\n", encoding="utf-8")
    return experiment_dir, checkpoint_dir, processed_dir, evaluation_dir


class ExtendedTetouanImportTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        root = Path(self.temp.name)
        self.experiment_dir, self.checkpoint_dir, self.processed_dir, self.evaluation_dir = source_tree(root / "source")
        self.tracking_uri = f"sqlite:///{root / 'tracking.db'}"
        mlflow.set_tracking_uri(self.tracking_uri)
        self.client = MlflowClient(tracking_uri=self.tracking_uri)
        self.experiment_id = self.client.create_experiment(
            EXPERIMENT_NAME, artifact_location=(root / "mlflow-artifacts").as_uri()
        )
        self.kwargs = {
            "tracking_uri": self.tracking_uri,
            "experiment_dir": self.experiment_dir,
            "checkpoint_dir": self.checkpoint_dir,
            "processed_dir": self.processed_dir,
            "evaluation_dir": self.evaluation_dir,
        }

    def test_parent_children_history_metrics_and_artifact_hashes(self):
        result = import_extended_experiment(**self.kwargs)
        self.assertEqual(result["status"], "created")
        self.assertEqual(len(result["child_run_ids"]), 6)
        parent = self.client.get_run(result["parent_run_id"])
        self.assertEqual(parent.data.tags["run_role"], "parent")
        self.assertEqual(parent.data.tags["test_selection_bias"], "true")
        self.assertEqual(parent.data.tags["local_mlflow_environment_differs_from_training"], "true")
        self.assertEqual(parent.data.tags["local_python_version"], "3.11.13")
        self.assertEqual(parent.data.tags["local_pytorch_version"], "2.2.2")
        children = self.client.search_runs(
            [self.experiment_id],
            filter_string=f"tags.mlflow.parentRunId = '{parent.info.run_id}'",
            max_results=20,
        )
        self.assertEqual(len(children), 6)
        for child in children:
            self.assertEqual(child.data.tags["historical_import"], "true")
            self.assertEqual(child.data.params["training.epochs"], "50")
            self.assertIn(child.data.params["model.name"], MODELS)
            self.assertEqual([p.step for p in self.client.get_metric_history(child.info.run_id, "training.loss")], list(range(1, 51)))
            self.assertEqual([p.step for p in self.client.get_metric_history(child.info.run_id, "validation.loss")], list(range(1, 51)))
            self.assertAlmostEqual(child.data.metrics["test.mae"], 10.0)
            seed = int(child.data.tags["seed"])
            model = child.data.params["model.name"]
            suffix = "lineargridtransformer" if model == "LinearGridTransformer" else "cdlinear"
            source = self.checkpoint_dir / f"tetouan_extended_v2_s{seed}_{suffix}.pt"
            downloaded = Path(self.client.download_artifacts(child.info.run_id, f"checkpoints/{source.name}"))
            digest = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
            self.assertEqual(digest(downloaded), digest(source))
        self.assertAlmostEqual(parent.data.metrics["test.three_seed_summary.CDLinear.MAE_mean"], 10.0)
        self.assertAlmostEqual(parent.data.metrics["test.baseline.persistence.mae"], 20.0)
        for artifact in (
            "metadata/evaluation_report.json",
            "metadata/import_provenance.json",
            "evaluation/overall_metrics_by_seed.csv",
            "evaluation/per_zone_metrics_by_seed.csv",
            "evaluation/per_horizon_metrics_by_seed.csv",
            "evaluation/three_seed_summary.csv",
        ):
            self.assertTrue(Path(self.client.download_artifacts(parent.info.run_id, artifact)).is_file())
        cdlinear = self.client.get_run(result["child_run_ids"]["tetouan_extended_v2_s42_cdlinear"])
        self.assertAlmostEqual(cdlinear.data.metrics["test.horizon_10m.mae"], 11.0)

    def test_repeat_reuses_runs_and_changed_fingerprint_conflicts(self):
        first = import_extended_experiment(**self.kwargs)
        before = self.client.search_runs([self.experiment_id], max_results=100)
        second = import_extended_experiment(**self.kwargs)
        after = self.client.search_runs([self.experiment_id], max_results=100)
        self.assertEqual(second["status"], "reused")
        self.assertEqual(second["parent_run_id"], first["parent_run_id"])
        self.assertEqual(second["child_run_ids"], first["child_run_ids"])
        self.assertEqual(len(before), len(after))
        metadata_path = self.processed_dir / "metadata.json"
        metadata = json.loads(metadata_path.read_text())
        metadata["row_count"] += 1
        write_json(metadata_path, metadata)
        with self.assertRaisesRegex(ImportConflictError, "Source artifacts changed"):
            import_extended_experiment(**self.kwargs)
        self.assertEqual(len(self.client.search_runs([self.experiment_id], max_results=100)), len(before))


if __name__ == "__main__":
    unittest.main()
