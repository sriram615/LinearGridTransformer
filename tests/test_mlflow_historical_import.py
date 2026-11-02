"""Focused tests for idempotent historical Tetouan run import."""

from __future__ import annotations

import json
from pathlib import Path

import unittest
from mlflow.tracking import MlflowClient

from src.mlops.import_tetouan_to_mlflow import import_historical_run
from src.mlops.mlflow_config import EXPERIMENT_NAME, RUN_NAME


def _write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data), encoding="utf-8")


def _source_tree(root: Path) -> tuple[Path, Path, Path]:
    experiment_dir = root / "results" / RUN_NAME
    processed_dir = root / "processed"
    checkpoint_dir = root / "checkpoints"
    manifest = {
        "run_metadata": {
            "run_name": RUN_NAME,
            "training": {"epochs": 2, "batch_size": 4, "learning_rate": 0.001, "seed": 12},
            "optimizer": "AdamW",
            "loss": "mean squared error",
            "gradient_clipping_max_norm": 1.0,
            "test_partition_used": False,
        },
        "models": {
            "LinearGridTransformer": {
                "configuration": {"in_channels": 8, "out_channels": 3, "lookback": 144, "horizon": 6},
                "best_validation_epoch": 2,
                "best_validation_loss": 0.2,
            }
        },
        "history_path": "/private/remote/history.json",
    }
    history = {"models": {"LinearGridTransformer": {
        "best_epoch": 2,
        "best_validation_loss": 0.2,
        "history": [
            {"epoch": 1, "train_loss": 0.4, "validation_loss": 0.3},
            {"epoch": 2, "train_loss": 0.25, "validation_loss": 0.2},
        ],
    }}}
    evaluation = {
        "dataset": {"name": "Tetouan", "source_url": "https://archive.ics.uci.edu/", "source_sha256": "abc"},
        "metrics": {
            "models": {"LinearGridTransformer": {"macro_average_across_zones": {"mae": 5.0, "rmse": 6.0}}},
            "baselines": {"persistence": {"macro_average_across_zones": {"mae": 7.0, "rmse": 8.0}}},
        },
    }
    horizon = {"models": {"LinearGridTransformer": {
        "per_horizon": [{"horizon_step": 1, "horizon_minutes": 10,
                          "per_zone": {"zone_1": {"mae": 4.0, "rmse": 5.0}}}],
        "consistency_with_existing_overall_metrics": {"all_match": True},
    }}}
    metadata = {
        "row_count": 100,
        "features": {"count": 8},
        "targets": {"count": 3},
        "windows": {"lookback": 144, "horizon": 6},
        "timestamps": {"cadence_minutes": 10},
        "normalization": {"method": "training z-score", "fit_partition": "train"},
    }
    for filename, data in (
        (f"{RUN_NAME}_manifest.json", manifest),
        (f"{RUN_NAME}_history.json", history),
        (f"{RUN_NAME}_test_evaluation.json", evaluation),
        (f"{RUN_NAME}_horizon_evaluation.json", horizon),
    ):
        _write_json(experiment_dir / filename, data)
    _write_json(processed_dir / "metadata.json", metadata)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    (checkpoint_dir / f"{RUN_NAME}_lineargridtransformer.pt").write_bytes(b"fixture transformer checkpoint")
    (checkpoint_dir / f"{RUN_NAME}_cdlinear.pt").write_bytes(b"fixture cdlinear checkpoint")
    return experiment_dir, processed_dir, checkpoint_dir


class TestMlflowHistoricalImport(unittest.TestCase):
    def setUp(self) -> None:
        import tempfile
        self._temporary = tempfile.TemporaryDirectory()
        self.tmp_path = Path(self._temporary.name)

    def tearDown(self) -> None:
        self._temporary.cleanup()

    def test_import_logs_historical_metadata_metrics_and_artifacts(self) -> None:
        experiment_dir, processed_dir, checkpoint_dir = _source_tree(self.tmp_path / "source")
        tracking_uri = f"sqlite:///{(self.tmp_path / 'tracking.db').as_posix()}"
        result = import_historical_run(
            tracking_uri=tracking_uri,
            artifact_root=self.tmp_path / "artifact-store",
            experiment_dir=experiment_dir,
            processed_dir=processed_dir,
            checkpoint_dir=checkpoint_dir,
        )
        self.assertEqual(result["status"], "created")

        client = MlflowClient(tracking_uri=tracking_uri)
        run = client.get_run(result["run_id"])
        self.assertEqual(run.data.tags["original_experiment_id"], RUN_NAME)
        self.assertEqual(run.data.tags["imported_historical"], "true")
        self.assertEqual(run.data.params["model.LinearGridTransformer.in_channels"], "8")
        self.assertEqual(run.data.metrics["training.LinearGridTransformer.train_loss"], 0.25)
        self.assertEqual(run.data.metrics["training.LinearGridTransformer.validation_loss"], 0.2)
        self.assertEqual(run.data.metrics["test.LinearGridTransformer.macro_average_across_zones.mae"], 5.0)
        self.assertEqual(run.data.metrics["test.baseline.persistence.macro_average_across_zones.mae"], 7.0)
        self.assertEqual(client.get_metric_history(result["run_id"], "training.LinearGridTransformer.train_loss")[-1].step, 2)
        self.assertEqual(client.get_metric_history(result["run_id"], "test.horizon.LinearGridTransformer.per_zone.zone_1.mae")[0].step, 1)
        artifacts = {item.path for item in client.list_artifacts(result["run_id"])}
        self.assertIn("metadata", artifacts)
        self.assertIn("checkpoints", artifacts)
        downloaded_manifest = Path(client.download_artifacts(
            result["run_id"], "metadata/manifest.json", str(self.tmp_path)
        ))
        logged_manifest = json.loads(downloaded_manifest.read_text(encoding="utf-8"))
        self.assertEqual(logged_manifest["history_path"], "history.json")
        self.assertNotIn("/private/remote", json.dumps(logged_manifest))

    def test_repeated_import_reuses_original_mlflow_run(self) -> None:
        experiment_dir, processed_dir, checkpoint_dir = _source_tree(self.tmp_path / "source")
        args = {
            "tracking_uri": f"sqlite:///{(self.tmp_path / 'tracking.db').as_posix()}",
            "artifact_root": self.tmp_path / "artifact-store",
            "experiment_dir": experiment_dir,
            "processed_dir": processed_dir,
            "checkpoint_dir": checkpoint_dir,
        }
        first = import_historical_run(**args)
        second = import_historical_run(**args)
        self.assertEqual(first["status"], "created")
        self.assertEqual(second, {"run_id": first["run_id"], "status": "reused", "experiment_id": first["experiment_id"]})
        client = MlflowClient(tracking_uri=args["tracking_uri"])
        runs = client.search_runs([first["experiment_id"]], max_results=10)
        self.assertEqual(len(runs), 1)


if __name__ == "__main__":
    unittest.main()

