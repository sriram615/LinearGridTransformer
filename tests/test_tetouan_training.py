from __future__ import annotations

import math
import inspect
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

import src.training.train_tetouan as training


class TetouanTrainingTests(unittest.TestCase):
    def test_updates_validation_checkpoint_roundtrip_and_test_isolation(self) -> None:
        real_dataset = training.TetouanSequenceDataset
        requested_splits: list[str] = []

        def tracked_dataset(*, split: str, **kwargs):
            requested_splits.append(split)
            if split == "test":
                raise AssertionError("The test partition must never be instantiated.")
            return real_dataset(split=split, **kwargs)

        config = training.TrainConfig(
            epochs=1,
            batch_size=4,
            num_workers=0,
            learning_rate=1e-3,
            weight_decay=1e-2,
            seed=23,
            max_train_batches=1,
            max_validation_batches=1,
            device="cpu",
            dropout=0.0,
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            checkpoints = root / "checkpoints"
            results = root / "results"
            with patch.object(training, "TetouanSequenceDataset", side_effect=tracked_dataset):
                result = training.run_training(
                    config,
                    checkpoint_dir=checkpoints,
                    results_dir=results,
                    run_name="unit-test",
                )

            self.assertEqual(requested_splits, ["train", "validation"])
            self.assertEqual(result["run_metadata"]["test_partition_used"], False)
            self.assertEqual(result["run_metadata"]["dataset"]["test_partition_used"], False)
            self.assertEqual(set(result["models"]), {"LinearGridTransformer", "CDLinear"})
            transformer_path = Path(result["models"]["LinearGridTransformer"]["checkpoint"])
            cdlinear_path = Path(result["models"]["CDLinear"]["checkpoint"])
            self.assertNotEqual(transformer_path, cdlinear_path)
            self.assertTrue(transformer_path.is_file())
            self.assertTrue(cdlinear_path.is_file())
            self.assertTrue((results / "unit-test_history.json").is_file())
            manifest_path = results / "unit-test_manifest.json"
            self.assertTrue(manifest_path.is_file())
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(manifest["run_metadata"]["dataset"]["feature_names"],
                             list(real_dataset(split="train").input_channel_names))
            self.assertEqual(manifest["run_metadata"]["dataset"]["lookback"], 144)
            self.assertEqual(manifest["run_metadata"]["dataset"]["horizon"], 6)
            self.assertEqual(manifest["run_metadata"]["training"]["batch_size"], 4)
            self.assertEqual(manifest["run_metadata"]["python_version"], training.platform.python_version())
            self.assertEqual(manifest["run_metadata"]["pytorch_version"], str(torch.__version__))
            self.assertIn("metadata.json", manifest["run_metadata"]["processed_artifacts"])
            self.assertIn("series.npz", manifest["run_metadata"]["processed_artifacts"])
            self.assertEqual(manifest["models"]["LinearGridTransformer"]["best_validation_epoch"], 1)
            self.assertEqual(manifest["models"]["CDLinear"]["best_validation_epoch"], 1)

            for model_name in ("LinearGridTransformer", "CDLinear"):
                record = result["models"][model_name]
                epoch = record["history"][0]
                self.assertEqual(epoch["train_batches"], 1)
                self.assertEqual(epoch["optimizer_updates"], 1)
                self.assertEqual(epoch["validation_batches"], 1)
                self.assertTrue(math.isfinite(epoch["train_loss"]))
                self.assertTrue(math.isfinite(epoch["validation_loss"]))
                self.assertTrue(epoch["is_best"])
                self.assertEqual(epoch["best_epoch"], 1)
                self.assertEqual(epoch["best_validation_loss"], epoch["validation_loss"])
                fresh = training.initialize_model(model_name, config)
                payload = training.load_model_checkpoint(fresh, record["checkpoint"])
                self.assertEqual(payload["model_name"], model_name)
                self.assertEqual(payload["checkpoint_metadata"]["best_epoch"], 1)
                self.assertEqual(
                    payload["checkpoint_metadata"]["best_validation_loss"],
                    epoch["validation_loss"],
                )
                self.assertEqual(payload["model_config"]["in_channels"], 8)
                self.assertEqual(payload["model_config"]["out_channels"], 3)
                self.assertEqual(len(payload["history"]), 1)
                self.assertTrue(payload["run_metadata"]["dataset"]["normalization"]["fit_partition"] == "train")
                self.assertTrue(all(torch.isfinite(p).all().item() for p in fresh.parameters()))

            self.assertEqual(result["run_metadata"]["dataset"]["lookback"], 144)
            self.assertEqual(result["run_metadata"]["dataset"]["horizon"], 6)
            self.assertEqual(result["run_metadata"]["training"]["seed"], 23)


    def test_best_checkpoint_updates_only_on_improvement_and_is_independent(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            model_a = torch.nn.Linear(1, 1)
            model_b = torch.nn.Linear(1, 1)
            optimizer_a = torch.optim.AdamW(model_a.parameters())
            optimizer_b = torch.optim.AdamW(model_b.parameters())
            tracker_a = training.BestValidationCheckpoint(
                root / "model_a.pt", "ModelA", {"width": 1}, {"seed": 13}
            )
            tracker_b = training.BestValidationCheckpoint(
                root / "model_b.pt", "ModelB", {"width": 1}, {"seed": 13}
            )

            with torch.no_grad():
                model_a.weight.fill_(1.0)
                model_a.bias.fill_(0.1)
            history_a = [{
                "epoch": 1, "validation_loss": 0.4, "is_best": True,
                "best_epoch": 1, "best_validation_loss": 0.4,
            }]
            self.assertTrue(tracker_a.consider(1, 0.4, model_a, optimizer_a, history_a))

            with torch.no_grad():
                model_a.weight.fill_(9.0)
                model_a.bias.fill_(9.0)
            history_a.append({
                "epoch": 2, "validation_loss": 0.7, "is_best": False,
                "best_epoch": 1, "best_validation_loss": 0.4,
            })
            self.assertFalse(tracker_a.consider(2, 0.7, model_a, optimizer_a, history_a))
            after_worse_epoch = torch.nn.Linear(1, 1)
            worse_payload = training.load_model_checkpoint(
                after_worse_epoch, root / "model_a.pt"
            )
            self.assertEqual(worse_payload["checkpoint_metadata"]["best_epoch"], 1)
            self.assertEqual(float(after_worse_epoch.weight.item()), 1.0)

            # The other model tracks its own best score and checkpoint state.
            with torch.no_grad():
                model_b.weight.fill_(-2.0)
                model_b.bias.fill_(-0.2)
            history_b = [{
                "epoch": 1, "validation_loss": 0.3, "is_best": True,
                "best_epoch": 1, "best_validation_loss": 0.3,
            }]
            self.assertTrue(tracker_b.consider(1, 0.3, model_b, optimizer_b, history_b))

            # A's later improvement replaces only A's checkpoint.
            with torch.no_grad():
                model_a.weight.fill_(3.0)
                model_a.bias.fill_(0.3)
            history_a.append({
                "epoch": 3, "validation_loss": 0.2, "is_best": True,
                "best_epoch": 3, "best_validation_loss": 0.2,
            })
            self.assertTrue(tracker_a.consider(3, 0.2, model_a, optimizer_a, history_a))
            tracker_a.finalize(history_a)
            tracker_b.finalize(history_b)

            restored_a = torch.nn.Linear(1, 1)
            restored_b = torch.nn.Linear(1, 1)
            checkpoint_a = training.load_model_checkpoint(restored_a, root / "model_a.pt")
            checkpoint_b = training.load_model_checkpoint(restored_b, root / "model_b.pt")
            self.assertEqual(checkpoint_a["checkpoint_metadata"]["best_epoch"], 3)
            self.assertEqual(checkpoint_a["checkpoint_metadata"]["best_validation_loss"], 0.2)
            self.assertEqual(checkpoint_a["history"][-1]["best_epoch"], 3)
            self.assertEqual(checkpoint_a["history"][-1]["best_validation_loss"], 0.2)
            self.assertEqual(checkpoint_b["checkpoint_metadata"]["best_epoch"], 1)
            self.assertEqual(checkpoint_b["checkpoint_metadata"]["best_validation_loss"], 0.3)
            self.assertEqual(float(restored_a.weight.item()), 3.0)
            self.assertEqual(float(restored_b.weight.item()), -2.0)
            self.assertNotEqual(checkpoint_a["model_name"], checkpoint_b["model_name"])

    def test_smoke_configuration_is_explicitly_bounded(self) -> None:
        config = training.TrainConfig(smoke=True, epochs=9, max_train_batches=5)
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            result = training.run_training(
                config,
                checkpoint_dir=root / "checkpoints",
                results_dir=root / "results",
                run_name="smoke-test",
            )
        metadata = result["run_metadata"]
        self.assertEqual(metadata["run_type"], "smoke")
        self.assertEqual(metadata["training"]["epochs"], 1)
        self.assertEqual(metadata["training"]["max_train_batches"], 2)
        self.assertEqual(metadata["training"]["max_validation_batches"], 2)
        for record in result["models"].values():
            self.assertLessEqual(record["history"][0]["train_batches"], 2)
            self.assertLessEqual(record["history"][0]["validation_batches"], 2)

    def test_default_output_directories_remain_unchanged(self) -> None:
        signature = inspect.signature(training.run_training)
        self.assertEqual(signature.parameters["checkpoint_dir"].default,
                         training.REPO_ROOT / "checkpoints" / "tetouan")
        self.assertEqual(signature.parameters["results_dir"].default,
                         training.REPO_ROOT / "results" / "tetouan")
        self.assertEqual(training.DEFAULT_CHECKPOINT_DIR,
                         training.REPO_ROOT / "checkpoints" / "tetouan")
        self.assertEqual(training.DEFAULT_RESULTS_DIR,
                         training.REPO_ROOT / "results" / "tetouan")

    def test_cli_routes_custom_output_directories_and_run_name(self) -> None:
        with patch.object(training, "run_training") as mocked:
            training.main([
                "--checkpoint-dir", "/tmp/custom-ckpts",
                "--results-dir", "/tmp/custom-results",
                "--run-name", "colab-run",
            ])
        _, kwargs = mocked.call_args
        self.assertEqual(kwargs["checkpoint_dir"], "/tmp/custom-ckpts")
        self.assertEqual(kwargs["results_dir"], "/tmp/custom-results")
        self.assertEqual(kwargs["run_name"], "colab-run")


if __name__ == "__main__":
    unittest.main()
