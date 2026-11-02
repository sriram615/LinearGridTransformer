from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from src.data.tetouan_experiment_store import (
    InvalidArtifactError,
    MissingArtifactError,
    TetouanExperimentStore,
)


REPO = Path(__file__).resolve().parents[1]
RUN = "tetouan_t4_20261101_v1"
REAL_EXPERIMENT = REPO / "results/tetouan" / RUN


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class TetouanExperimentStoreTests(unittest.TestCase):
    def test_dataset_metadata_and_quality_are_retrieved(self) -> None:
        store = TetouanExperimentStore()
        dataset = store.get_dataset_info()
        self.assertEqual(dataset["dataset_name"], "Power Consumption of Tetouan City")
        self.assertEqual(dataset["row_count"], 52416)
        self.assertEqual(dataset["date_range"]["first"], "2017-01-01 00:00:00")
        self.assertEqual(dataset["date_range"]["last"], "2017-12-30 23:50:00")
        self.assertEqual(dataset["cadence_minutes"], 10)
        self.assertEqual(dataset["features"]["names"][0], "zone_1_power_consumption")
        self.assertEqual(dataset["features"]["names"][-1], "diffuse_flows")
        self.assertEqual(dataset["targets"]["names"], dataset["features"]["names"][:3])
        self.assertEqual(dataset["splits"]["partitions"]["test"]["row_count"], 7863)
        self.assertEqual(dataset["normalization"]["fit_partition"], "train")
        self.assertIsNone(dataset["coverage_uncertainties"]["units"])
        self.assertIn("undocumented", dataset["timezone_status"])
        quality = dataset["data_quality"]
        self.assertEqual(quality["missing_values_per_column"], 0)
        self.assertEqual(quality["exact_duplicate_rows"], 0)
        self.assertEqual(quality["duplicate_timestamps"], 0)
        self.assertEqual(quality["timestamp_parse_failures"], 0)
        self.assertEqual(quality["irregular_intervals"], 0)

    def test_manifest_history_models_and_local_checkpoint_references(self) -> None:
        store = TetouanExperimentStore()
        info = store.get_experiment_info()
        self.assertEqual(info["run_identifier"], RUN)
        self.assertIsNone(info["provenance"]["git_commit"])
        self.assertEqual(list(info["models"]), ["LinearGridTransformer", "CDLinear"])
        self.assertEqual(info["models"]["LinearGridTransformer"]["configuration"]["in_channels"], 8)
        self.assertEqual(info["models"]["CDLinear"]["configuration"]["out_channels"], 3)
        for model in info["models"].values():
            self.assertTrue(model["checkpoint"]["exists"])
            self.assertGreater(model["checkpoint"]["size_bytes"], 0)
            self.assertEqual(len(model["history"]), 10)
        self.assertNotIn("/content/", info["models"]["CDLinear"]["checkpoint"]["path"])

    def test_training_history_matches_saved_json(self) -> None:
        store = TetouanExperimentStore()
        expected = json.loads((REAL_EXPERIMENT / f"{RUN}_history.json").read_text())
        actual = store.get_training_history()
        for name in ("LinearGridTransformer", "CDLinear"):
            self.assertEqual(actual[name], expected["models"][name]["history"])

    def test_evaluation_data_matches_saved_reports_without_recomputation(self) -> None:
        store = TetouanExperimentStore()
        result = store.get_evaluation_info()
        overall = json.loads((REAL_EXPERIMENT / f"{RUN}_test_evaluation.json").read_text())
        horizon = json.loads((REAL_EXPERIMENT / f"{RUN}_horizon_evaluation.json").read_text())
        self.assertEqual(
            result["overall"]["models"],
            {name: overall["metrics"]["models"][name] for name in ("LinearGridTransformer", "CDLinear")},
        )
        self.assertEqual(
            result["overall"]["baselines"],
            {name: overall["metrics"]["baselines"][name] for name in ("training_mean", "persistence")},
        )
        self.assertEqual(result["horizon_wise"]["target_zone_names"], horizon["target_zone_names"])
        self.assertEqual(
            result["horizon_wise"]["models"]["CDLinear"],
            horizon["models"]["CDLinear"]["per_horizon"],
        )
        self.assertTrue(result["horizon_wise"]["consistency_with_overall"]["LinearGridTransformer"]["all_match"])

    def test_verify_artifacts_checks_required_files_and_fingerprints(self) -> None:
        result = TetouanExperimentStore().verify_artifacts()
        self.assertEqual(result["status"], "ok", result["issues"])
        self.assertTrue(result["checks"]["fingerprints"]["metadata.json"]["valid"])
        self.assertTrue(result["checks"]["fingerprints"]["series.npz"]["valid"])
        self.assertTrue(result["checks"]["checkpoint_CDLinear"]["exists"])

    def test_read_operations_do_not_change_source_artifacts(self) -> None:
        store = TetouanExperimentStore()
        files = [
            store.metadata_path, store.series_path, store.manifest_path, store.history_path,
            store.evaluation_path, store.horizon_evaluation_path,
            store.checkpoint_path("LinearGridTransformer"), store.checkpoint_path("CDLinear"),
        ]
        before = {path: sha256(path) for path in files}
        store.get_dataset_info()
        store.get_experiment_info()
        store.get_training_history()
        store.get_evaluation_info()
        store.verify_artifacts()
        after = {path: sha256(path) for path in files}
        self.assertEqual(before, after)

    def test_custom_paths_and_temp_fixture(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            data_dir = base / "data"
            exp_dir = base / "exp"
            ckpt_dir = base / "ckpt"
            raw_dir = base / "raw"
            data_dir.mkdir()
            exp_dir.mkdir()
            ckpt_dir.mkdir()
            raw_dir.mkdir()
            metadata = json.loads((REPO / "data/processed/tetouan/metadata.json").read_text())
            (data_dir / "metadata.json").write_text(json.dumps(metadata))
            (data_dir / "series.npz").write_bytes(b"opaque-array-artifact")
            run = RUN + "-fixture"
            # A custom run identifier and explicit directories are supported.
            manifest = json.loads((REAL_EXPERIMENT / f"{RUN}_manifest.json").read_text())
            history = json.loads((REAL_EXPERIMENT / f"{RUN}_history.json").read_text())
            manifest["run_metadata"]["run_name"] = run
            history["run_metadata"]["run_name"] = run
            for model_name, value in manifest["models"].items():
                value["checkpoint"] = f"/nonportable/{model_name}.pt"
                (ckpt_dir / f"{run}_{'lineargridtransformer' if model_name == 'LinearGridTransformer' else 'cdlinear'}.pt").write_bytes(
                    f"opaque-{model_name}".encode()
                )
            (exp_dir / f"{run}_manifest.json").write_text(json.dumps(manifest))
            (exp_dir / f"{run}_history.json").write_text(json.dumps(history))
            store = TetouanExperimentStore(
                run_name=run,
                artifact_dir=data_dir,
                experiment_dir=exp_dir,
                checkpoint_dir=ckpt_dir,
                raw_dir=raw_dir,
                inspection_report_path=None,
            )
            self.assertEqual(store.get_dataset_info()["row_count"], 52416)
            info = store.get_experiment_info()
            self.assertTrue(
                Path(info["models"]["CDLinear"]["checkpoint"]["path"]).parent == ckpt_dir.resolve()
            )
            self.assertNotEqual(
                info["models"]["CDLinear"]["checkpoint"]["path"],
                info["models"]["CDLinear"]["checkpoint"]["recorded_manifest_reference"],
            )

    def test_missing_artifact_has_clear_error(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = TetouanExperimentStore(artifact_dir=tmp, inspection_report_path=None)
            with self.assertRaisesRegex(MissingArtifactError, "dataset metadata is missing"):
                store.get_dataset_info()

    def test_malformed_json_and_missing_fields_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "metadata.json").write_text("{broken")
            store = TetouanExperimentStore(artifact_dir=root, inspection_report_path=None)
            with self.assertRaisesRegex(InvalidArtifactError, "Could not read dataset metadata"):
                store.get_dataset_info()
            (root / "metadata.json").write_text("{}")
            with self.assertRaisesRegex(InvalidArtifactError, "missing required fields"):
                store.get_dataset_info()


if __name__ == "__main__":
    unittest.main()
