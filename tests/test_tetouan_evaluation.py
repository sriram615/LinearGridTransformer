from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from src.training.evaluate_tetouan import (
    build_model,
    calculate_metrics,
    dataset_contract,
    inverse_normalize_targets,
    load_best_model,
    persistence_baseline,
    target_normalization_stats,
    training_mean_baseline,
    validate_checkpoint_payload,
    write_report,
)


def synthetic_metadata() -> dict:
    train = {"start_index": 0, "end_index_exclusive": 100, "row_count": 100}
    validation = {"start_index": 100, "end_index_exclusive": 120, "row_count": 20}
    return {
        "source": {
            "dataset_url": "https://example.invalid/tetouan",
            "file_name": "tetouan.csv",
            "sha256": "a" * 64,
            "doi": "10.example/test",
        },
        "row_count": 140,
        "features": {"names": [
            "zone_1_power_consumption", "zone_2_power_consumption",
            "zone_3_power_consumption", "temperature", "humidity",
            "wind_speed", "general_diffuse_flows", "diffuse_flows",
        ]},
        "targets": {"names": [
            "zone_1_power_consumption", "zone_2_power_consumption",
            "zone_3_power_consumption",
        ]},
        "windows": {"lookback": 144, "horizon": 6},
        "timestamps": {"cadence_minutes": 10, "timezone_status": "undocumented"},
        "units_status": "undocumented",
        "normalization": {
            "method": "z-score: (value - mean) / std",
            "fit_partition": "train",
            "fit_row_range_half_open": [0, 100],
            "std_ddof": 0,
            "target_names": [
                "zone_1_power_consumption", "zone_2_power_consumption",
                "zone_3_power_consumption",
            ],
            "feature_mean": [10.0, 20.0, 30.0, 1.0, 2.0, 3.0, 4.0, 5.0],
            "feature_std": [2.0, 4.0, 5.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            "target_mean": [10.0, 20.0, 30.0],
            "target_std": [2.0, 4.0, 5.0],
        },
        "split": {
            "method": "chronological",
            "partitions": {"train": train, "validation": validation,
                           "test": {"start_index": 120, "end_index_exclusive": 140,
                                    "row_count": 20}},
        },
        "source_attribution": "Test Author (2020)",
    }


def make_payload(model_name: str, metadata: dict) -> dict:
    if model_name == "LinearGridTransformer":
        config = {
            "in_channels": 8, "out_channels": 3, "lookback": 144, "horizon": 6,
            "d_model": 8, "num_heads": 2, "ffn_dim": 16, "dropout": 0.0,
        }
    else:
        config = {"in_channels": 8, "out_channels": 3, "seq_len": 144, "pred_len": 6}
    model = build_model(model_name, config)
    return {
        "model_name": model_name,
        "model_state_dict": model.state_dict(),
        "model_config": config,
        "run_metadata": {
            "run_type": "experiment",
            "test_partition_used": False,
            "dataset": dataset_contract(metadata),
            "models": {model_name: config},
            "run_name": "unit-experiment",
        },
        "checkpoint_metadata": {
            "selection_metric": "validation_loss",
            "best_epoch": 2,
            "best_validation_loss": 0.125,
        },
        "history": [
            {"epoch": 1, "validation_loss": 0.2, "is_best": True,
             "best_epoch": 1, "best_validation_loss": 0.2},
            {"epoch": 2, "validation_loss": 0.125, "is_best": True,
             "best_epoch": 2, "best_validation_loss": 0.125},
        ],
    }


class TetouanEvaluationTests(unittest.TestCase):
    def test_best_checkpoint_strict_load_and_incompatible_rejection(self) -> None:
        metadata = synthetic_metadata()
        expected = dataset_contract(metadata)
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "best.pt"
            payload = make_payload("CDLinear", metadata)
            torch.save(payload, path)
            model, loaded = load_best_model(path, "CDLinear", expected, "cpu")
            self.assertEqual(loaded["checkpoint_metadata"]["best_epoch"], 2)
            self.assertEqual(model.linear.in_features, 1152)
            self.assertEqual(model.linear.out_features, 18)

            wrong_dataset = dict(expected)
            wrong_dataset["source_sha256"] = "b" * 64
            with self.assertRaisesRegex(ValueError, "dataset provenance"):
                validate_checkpoint_payload(payload, "CDLinear", wrong_dataset)

            wrong_dimensions = make_payload("CDLinear", metadata)
            wrong_dimensions["model_config"]["out_channels"] = 2
            wrong_dimensions["run_metadata"]["models"]["CDLinear"]["out_channels"] = 2
            with self.assertRaisesRegex(ValueError, "out_channels=3"):
                validate_checkpoint_payload(wrong_dimensions, "CDLinear", expected)

            bad_state_payload = dict(payload)
            bad_state_payload["model_state_dict"] = dict(payload["model_state_dict"])
            bad_state_payload["model_state_dict"]["unexpected.weight"] = torch.ones(1)
            torch.save(bad_state_payload, path)
            with self.assertRaisesRegex(ValueError, "weights are incompatible"):
                load_best_model(path, "CDLinear", expected, "cpu")

    def test_transformer_checkpoint_reconstructs_from_saved_configuration(self) -> None:
        metadata = synthetic_metadata()
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "transformer_best.pt"
            torch.save(make_payload("LinearGridTransformer", metadata), path)
            model, payload = load_best_model(
                path, "LinearGridTransformer", dataset_contract(metadata), "cpu"
            )
            self.assertEqual(payload["model_config"]["d_model"], 8)
            self.assertEqual(model.in_channels, 8)
            self.assertEqual(model.out_channels, 3)
            self.assertEqual(model.t_forecast, 6)

    def test_missing_checkpoint_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            with self.assertRaises(FileNotFoundError):
                load_best_model(
                    Path(temp_dir) / "missing.pt", "CDLinear",
                    dataset_contract(synthetic_metadata()), "cpu",
                )

    def test_rejects_smoke_or_non_best_checkpoint(self) -> None:
        metadata = synthetic_metadata()
        expected = dataset_contract(metadata)
        payload = make_payload("CDLinear", metadata)
        payload["run_metadata"]["run_type"] = "smoke"
        with self.assertRaisesRegex(ValueError, "full experiment"):
            validate_checkpoint_payload(payload, "CDLinear", expected)
        payload = make_payload("CDLinear", metadata)
        payload.pop("checkpoint_metadata")
        with self.assertRaisesRegex(ValueError, "best-validation metadata"):
            validate_checkpoint_payload(payload, "CDLinear", expected)

    def test_train_only_normalization_stats_are_reused_and_validated(self) -> None:
        metadata = synthetic_metadata()
        mean, std = target_normalization_stats(metadata)
        np.testing.assert_array_equal(mean, [10.0, 20.0, 30.0])
        np.testing.assert_array_equal(std, [2.0, 4.0, 5.0])
        metadata["normalization"]["fit_partition"] = "test"
        with self.assertRaisesRegex(ValueError, "training partition"):
            target_normalization_stats(metadata)

    def test_inverse_normalization_and_known_zone_macro_and_pooled_metrics(self) -> None:
        normalized = np.array([[[0.0, 1.0, -1.0]]])
        raw = inverse_normalize_targets(normalized, [10.0, 20.0, 30.0], [2.0, 4.0, 5.0])
        np.testing.assert_array_equal(raw, np.array([[[10.0, 24.0, 25.0]]]))

        targets = np.array([[[1.0, 2.0, 3.0], [3.0, 4.0, 5.0]]])
        predictions = np.array([[[2.0, 2.0, 1.0], [2.0, 6.0, 5.0]]])
        metrics = calculate_metrics(predictions, targets, ("z1", "z2", "z3"))
        self.assertEqual([metrics["per_zone"][z]["mae"] for z in ("z1", "z2", "z3")], [1.0, 1.0, 1.0])
        self.assertEqual(metrics["macro_average_across_zones"]["mae"], 1.0)
        self.assertAlmostEqual(metrics["macro_average_across_zones"]["rmse"], (1.0 + 2 * np.sqrt(2)) / 3)
        self.assertAlmostEqual(metrics["pooled_across_all_samples_horizons_and_zones"]["mae"], 1.0)
        self.assertAlmostEqual(metrics["pooled_across_all_samples_horizons_and_zones"]["rmse"], np.sqrt(10 / 6))

    def test_training_mean_and_persistence_baselines(self) -> None:
        mean_prediction = training_mean_baseline(2, 6, [10.0, 20.0, 30.0])
        self.assertEqual(mean_prediction.shape, (2, 6, 3))
        np.testing.assert_array_equal(mean_prediction[1, -1], [10.0, 20.0, 30.0])

        # Final input zone values have normalized values [1, -1, 0].
        normalized_final_values = np.array([[1.0, -1.0, 0.0]], dtype=np.float32)
        prediction = persistence_baseline(
            normalized_final_values, 6, [10.0, 20.0, 30.0], [2.0, 4.0, 5.0]
        )
        self.assertEqual(prediction.shape, (1, 6, 3))
        np.testing.assert_array_equal(prediction[0], np.tile([12.0, 16.0, 30.0], (6, 1)))

    def test_report_serialization_and_no_overwrite(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "evaluation.json"
            report = {
                "metrics": calculate_metrics(
                    np.ones((1, 2, 3)), np.zeros((1, 2, 3))
                ),
                "test_usage": {"used_for_training": False},
            }
            write_report(report, path)
            loaded = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(loaded["test_usage"]["used_for_training"], False)
            self.assertEqual(loaded["metrics"]["macro_average_across_zones"]["mae"], 1.0)
            with self.assertRaises(FileExistsError):
                write_report(report, path)


if __name__ == "__main__":
    unittest.main()
