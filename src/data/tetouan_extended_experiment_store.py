"""Read-only store for the three-seed Tetouan extended-training experiment."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from statistics import mean, pstdev
from typing import Any

from src.data.tetouan_preprocess import DEFAULT_OUTPUT_DIR
from src.training.evaluate_tetouan import dataset_contract


REPO_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT_ID = "extended_training_v2"
SEEDS = (42, 123, 2026)
MODEL_NAMES = ("LinearGridTransformer", "CDLinear")
DEFAULT_EXPERIMENT_DIR = REPO_ROOT / "results" / "tetouan" / EXPERIMENT_ID
DEFAULT_CHECKPOINT_DIR = REPO_ROOT / "checkpoints" / "tetouan" / EXPERIMENT_ID
DEFAULT_EVALUATION_DIR = DEFAULT_EXPERIMENT_DIR / "evaluation_test_v1"
DEFAULT_PROCESSED_DIR = DEFAULT_OUTPUT_DIR


class ExtendedExperimentError(RuntimeError):
    """Base error for extended experiment artifacts."""


class MissingExtendedArtifactError(ExtendedExperimentError, FileNotFoundError):
    """A required extended experiment artifact is missing."""


class InvalidExtendedArtifactError(ExtendedExperimentError, ValueError):
    """An extended experiment artifact is malformed or inconsistent."""


class TetouanExtendedExperimentStore:
    """Read saved manifests, histories, checkpoints and held-out report only."""

    def __init__(
        self,
        *,
        experiment_dir: str | Path = DEFAULT_EXPERIMENT_DIR,
        checkpoint_dir: str | Path = DEFAULT_CHECKPOINT_DIR,
        evaluation_dir: str | Path = DEFAULT_EVALUATION_DIR,
        processed_dir: str | Path = DEFAULT_PROCESSED_DIR,
    ) -> None:
        self.experiment_dir = Path(experiment_dir)
        self.checkpoint_dir = Path(checkpoint_dir)
        self.evaluation_dir = Path(evaluation_dir)
        self.processed_dir = Path(processed_dir)

    def manifest_path(self, seed: int) -> Path:
        return self.experiment_dir / f"tetouan_extended_v2_s{seed}_manifest.json"

    def history_path(self, seed: int) -> Path:
        return self.experiment_dir / f"tetouan_extended_v2_s{seed}_history.json"

    def checkpoint_path(self, model: str, seed: int) -> Path:
        suffix = "lineargridtransformer" if model == "LinearGridTransformer" else "cdlinear"
        return self.checkpoint_dir / f"tetouan_extended_v2_s{seed}_{suffix}.pt"

    @property
    def report_path(self) -> Path:
        return self.evaluation_dir / "evaluation_report.json"

    @property
    def predictions_path(self) -> Path:
        return self.evaluation_dir / "test_predictions_all_models.npz"

    def _read_json(self, path: Path, label: str) -> dict[str, Any]:
        if not path.is_file():
            raise MissingExtendedArtifactError(f"Required {label} is missing.")
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise InvalidExtendedArtifactError(f"Could not read {label}.") from exc
        if not isinstance(value, dict):
            raise InvalidExtendedArtifactError(f"{label} must contain a JSON object.")
        return value

    @staticmethod
    def _sha256(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    def _load_sources(self) -> tuple[dict[int, dict[str, Any]], dict[int, dict[str, Any]], dict[str, Any]]:
        manifests = {seed: self._read_json(self.manifest_path(seed), f"seed {seed} manifest") for seed in SEEDS}
        histories = {seed: self._read_json(self.history_path(seed), f"seed {seed} history") for seed in SEEDS}
        report = self._read_json(self.report_path, "extended evaluation report")
        return manifests, histories, report

    def _validate(self) -> tuple[dict[int, dict[str, Any]], dict[int, dict[str, Any]], dict[str, Any]]:
        manifests, histories, report = self._load_sources()
        metadata_path = self.processed_dir / "metadata.json"
        series_path = self.processed_dir / "series.npz"
        if not metadata_path.is_file() or not series_path.is_file():
            raise MissingExtendedArtifactError("Processed Tetouan data artifacts are missing.")
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise InvalidExtendedArtifactError("Processed Tetouan metadata is invalid.") from exc
        # JSON round-trip canonicalizes list/tuple fields in manifests.
        expected_dataset = json.loads(json.dumps(dataset_contract(metadata)))
        for seed in SEEDS:
            manifest = manifests[seed]
            history = histories[seed]
            run_name = f"tetouan_extended_v2_s{seed}"
            run = manifest.get("run_metadata", {})
            if run.get("run_name") != run_name or run.get("run_type") != "experiment":
                raise InvalidExtendedArtifactError(f"Seed {seed} manifest identity is inconsistent.")
            if run.get("test_partition_used") is not False:
                raise InvalidExtendedArtifactError(f"Seed {seed} does not establish test isolation.")
            if run.get("dataset") != expected_dataset:
                raise InvalidExtendedArtifactError(f"Seed {seed} dataset/split/normalization contract differs.")
            training = run.get("training", {})
            if training.get("seed") != seed or training.get("epochs") != 50:
                raise InvalidExtendedArtifactError(f"Seed {seed} training metadata is inconsistent.")
            if history.get("run_metadata", {}).get("run_name") != run_name:
                raise InvalidExtendedArtifactError(f"Seed {seed} history belongs to another run.")
            fingerprints = run.get("processed_artifacts", {})
            for filename, path in (("metadata.json", metadata_path), ("series.npz", series_path)):
                recorded = fingerprints.get(filename, {})
                if (
                    recorded.get("size_bytes") != path.stat().st_size
                    or recorded.get("sha256") != self._sha256(path)
                ):
                    raise InvalidExtendedArtifactError(
                        f"Seed {seed} processed-data fingerprint mismatch for {filename}."
                    )
            for model_name in MODEL_NAMES:
                config = manifest.get("models", {}).get(model_name, {}).get("configuration")
                if config != run.get("models", {}).get(model_name):
                    raise InvalidExtendedArtifactError(f"Seed {seed} {model_name} config mismatch.")
                model_history = history.get("models", {}).get(model_name, {})
                rows = model_history.get("history")
                checkpoint_record = manifest["models"][model_name]
                if not isinstance(rows, list) or len(rows) != 50:
                    raise InvalidExtendedArtifactError(f"Seed {seed} {model_name} history is not 50 epochs.")
                if (
                    model_history.get("best_epoch") != checkpoint_record.get("best_validation_epoch")
                    or model_history.get("best_validation_loss") != checkpoint_record.get("best_validation_loss")
                ):
                    raise InvalidExtendedArtifactError(f"Seed {seed} {model_name} best-validation record mismatch.")
                selection = checkpoint_record.get("best_validation_loss")
                if not isinstance(selection, (int, float)) or not math.isfinite(selection):
                    raise InvalidExtendedArtifactError(f"Seed {seed} {model_name} validation loss is invalid.")
                checkpoint = self.checkpoint_path(model_name, seed)
                if not checkpoint.is_file() or checkpoint.stat().st_size <= 0:
                    raise MissingExtendedArtifactError(f"Seed {seed} {model_name} checkpoint is missing.")

        if report.get("experiment") != EXPERIMENT_ID or report.get("seeds") != list(SEEDS):
            raise InvalidExtendedArtifactError("Evaluation report experiment ID or seed list is inconsistent.")
        if report.get("test_windows") != 7858 or report.get("lookback") != 144:
            raise InvalidExtendedArtifactError("Evaluation report dimensions or sample count are inconsistent.")
        if report.get("forecast_horizon") != 6 or report.get("cadence_minutes") != 10:
            raise InvalidExtendedArtifactError("Evaluation report horizon/cadence is inconsistent.")
        if report.get("zones") != ["Zone 1", "Zone 2", "Zone 3"]:
            raise InvalidExtendedArtifactError("Evaluation report target zone ordering is inconsistent.")
        summary_rows = {row.get("model"): row for row in report.get("three_seed_summary", [])}
        for model_name in MODEL_NAMES:
            rows = [row for row in report.get("overall_metrics", []) if row.get("model") == model_name]
            if len(rows) != len(SEEDS) or {row.get("seed") for row in rows} != set(SEEDS):
                raise InvalidExtendedArtifactError(f"Evaluation report is missing per-seed metrics for {model_name}.")
            summary = summary_rows.get(model_name)
            if not isinstance(summary, dict):
                raise InvalidExtendedArtifactError(f"Evaluation report is missing the {model_name} summary.")
            for metric, summary_key in (("MAE", "MAE_mean"), ("RMSE", "RMSE_mean")):
                expected_mean = mean(float(row[metric]) for row in rows)
                recorded_mean = summary.get(summary_key)
                if not isinstance(recorded_mean, (int, float)) or not math.isclose(
                    float(recorded_mean), expected_mean, rel_tol=1e-10, abs_tol=1e-10
                ):
                    raise InvalidExtendedArtifactError(f"{model_name} {summary_key} disagrees with per-seed metrics.")
        selected_model = self._selected_model(report)
        if selected_model != "CDLinear":
            raise InvalidExtendedArtifactError("The verified three-seed evaluation no longer selects CDLinear.")
        if not self.predictions_path.is_file():
            raise MissingExtendedArtifactError("All-model saved test predictions are missing.")
        return manifests, histories, report

    def verify_artifacts(self) -> dict[str, Any]:
        paths: dict[str, Path] = {
            "evaluation_report": self.report_path,
            "all_model_test_predictions": self.predictions_path,
            "overall_metrics_by_seed": self.evaluation_dir / "overall_metrics_by_seed.csv",
            "per_zone_metrics_by_seed": self.evaluation_dir / "per_zone_metrics_by_seed.csv",
            "per_horizon_metrics_by_seed": self.evaluation_dir / "per_horizon_metrics_by_seed.csv",
            "three_seed_summary": self.evaluation_dir / "three_seed_summary.csv",
            "evaluated_checkpoints": self.evaluation_dir / "evaluated_checkpoints.csv",
        }
        for seed in SEEDS:
            paths[f"manifest_seed_{seed}"] = self.manifest_path(seed)
            paths[f"history_seed_{seed}"] = self.history_path(seed)
            for model in MODEL_NAMES:
                paths[f"checkpoint_{model}_seed_{seed}"] = self.checkpoint_path(model, seed)
        checks: dict[str, Any] = {}
        issues: list[str] = []
        for name, path in paths.items():
            exists = path.is_file()
            checks[name] = {
                "name": path.name,
                "available": exists,
                "size_bytes": path.stat().st_size if exists else None,
                "sha256": self._sha256(path) if exists else None,
            }
            if not exists:
                issues.append(f"Missing {name}.")
        try:
            self._validate()
        except ExtendedExperimentError as exc:
            issues.append(str(exc))
        return {"status": "ok" if not issues else "issues", "checks": checks, "issues": issues}

    def get_dataset_info(self) -> dict[str, Any]:
        # Dataset metadata is unchanged from the original Tetouan pipeline.
        from src.data.tetouan_experiment_store import TetouanExperimentStore

        return TetouanExperimentStore().get_dataset_info()

    @staticmethod
    def _selected_model(report: dict[str, Any]) -> str:
        summaries = {row["model"]: row for row in report["three_seed_summary"]}
        candidates = [name for name in MODEL_NAMES if name in summaries]
        if not candidates:
            raise InvalidExtendedArtifactError("Evaluation report has no model summaries.")
        return min(
            candidates,
            key=lambda name: (
                float(summaries[name]["MAE_mean"]),
                float(summaries[name]["RMSE_mean"]),
            ),
        )

    @staticmethod
    def _inference_seed(manifests: dict[int, dict[str, Any]]) -> int:
        return min(
            SEEDS,
            key=lambda seed: float(
                manifests[seed]["models"]["CDLinear"]["best_validation_loss"]
            ),
        )

    def get_inference_run_name(self) -> str:
        manifests, _, _ = self._validate()
        return f"tetouan_extended_v2_s{self._inference_seed(manifests)}"

    def get_model_roles(self, report: dict[str, Any] | None = None) -> dict[str, dict[str, Any]]:
        if report is None:
            _, _, report = self._validate()
        selected = self._selected_model(report)
        return {
            "LinearGridTransformer": {
                "role": "selected_forecaster" if selected == "LinearGridTransformer" else "experimental_alternative",
                "selected": selected == "LinearGridTransformer",
            },
            "CDLinear": {
                "role": "selected_forecaster" if selected == "CDLinear" else "experimental_alternative",
                "selected": selected == "CDLinear",
            },
        }

    def get_experiment_info(self) -> dict[str, Any]:
        manifests, _, report = self._validate()
        selected_seed = self._inference_seed(manifests)
        selected_manifest = manifests[selected_seed]
        roles = self.get_model_roles(report)
        model_records: dict[str, Any] = {}
        for name in MODEL_NAMES:
            checkpoint_path = self.checkpoint_path(name, selected_seed)
            selected_record = selected_manifest["models"][name]
            seed_runs = []
            for seed in SEEDS:
                record = manifests[seed]["models"][name]
                path = self.checkpoint_path(name, seed)
                seed_runs.append({
                    "seed": seed,
                    "best_validation_epoch": record["best_validation_epoch"],
                    "best_validation_loss": record["best_validation_loss"],
                    "checkpoint": {
                        "name": path.name,
                        "available": path.is_file(),
                        "size_bytes": path.stat().st_size if path.is_file() else None,
                        "sha256": self._sha256(path) if path.is_file() else None,
                    },
                })
            model_records[name] = {
                "configuration": selected_record["configuration"],
                **roles[name],
                "checkpoint_seed": selected_seed,
                "checkpoint": {
                    "name": checkpoint_path.name,
                    "available": checkpoint_path.is_file(),
                    "size_bytes": checkpoint_path.stat().st_size,
                    "sha256": self._sha256(checkpoint_path),
                },
                "best_validation_epoch": selected_record["best_validation_epoch"],
                "best_validation_loss": selected_record["best_validation_loss"],
                "seed_runs": seed_runs,
            }
        summary = {row["model"]: row for row in report["three_seed_summary"]}
        return {
            "run_identifier": EXPERIMENT_ID,
            "run_type": "three_seed_50_epoch_experiment",
            "created_utc": max(manifests[seed]["run_metadata"]["created_utc"] for seed in SEEDS),
            "provenance": {
                "dataset_source": selected_manifest["run_metadata"]["dataset"]["source_url"],
                "dataset_source_sha256": selected_manifest["run_metadata"]["dataset"]["source_sha256"],
                "training_environment": {
                    key: selected_manifest["run_metadata"].get(key)
                    for key in ("python_version", "pytorch_version", "device", "cuda_available", "cuda_version", "gpu_name")
                },
                "git_commit": selected_manifest["run_metadata"].get("git_commit"),
            },
            "training_configuration": {
                "seeds": list(SEEDS),
                "epochs": 50,
                "per_seed": {str(seed): manifests[seed]["run_metadata"]["training"] for seed in SEEDS},
                "shared": {
                    "batch_size": selected_manifest["run_metadata"]["training"]["batch_size"],
                    "learning_rate": selected_manifest["run_metadata"]["training"]["learning_rate"],
                    "weight_decay": selected_manifest["run_metadata"]["training"]["weight_decay"],
                    "optimizer": selected_manifest["run_metadata"]["optimizer"],
                    "loss": selected_manifest["run_metadata"]["loss"],
                    "gradient_clipping_max_norm": selected_manifest["run_metadata"]["gradient_clipping_max_norm"],
                },
            },
            "optimizer": selected_manifest["run_metadata"]["optimizer"],
            "loss": selected_manifest["run_metadata"]["loss"],
            "seed_policy": {str(seed): manifests[seed]["run_metadata"]["seed_policy"] for seed in SEEDS},
            "test_partition_used": False,
            "model_selection": {
                "selected_model": self._selected_model(report),
                "selection_basis": "lowest three-seed mean held-out test MAE and RMSE among the compared forecasting architectures",
                "evidence": summary,
                "persistence_role": "evaluation_baseline_not_trained_model",
                "inference_checkpoint_seed": selected_seed,
                "inference_checkpoint_selection_metric": "validation_loss",
            },
            "models": model_records,
            "baselines": {"Persistence": {"role": "evaluation_baseline", "trained": False}},
            "artifact_references": {
                "evaluation_report": str(self.report_path.resolve()),
                "three_seed_summary": str((self.evaluation_dir / "three_seed_summary.csv").resolve()),
                "overall_metrics_by_seed": str((self.evaluation_dir / "overall_metrics_by_seed.csv").resolve()),
                "per_zone_metrics_by_seed": str((self.evaluation_dir / "per_zone_metrics_by_seed.csv").resolve()),
                "per_horizon_metrics_by_seed": str((self.evaluation_dir / "per_horizon_metrics_by_seed.csv").resolve()),
                "test_predictions": str(self.predictions_path.resolve()),
                **{f"manifest_seed_{seed}": str(self.manifest_path(seed).resolve()) for seed in SEEDS},
                **{f"history_seed_{seed}": str(self.history_path(seed).resolve()) for seed in SEEDS},
            },
        }

    def get_training_history(self) -> dict[str, list[dict[str, Any]]]:
        _, histories, _ = self._validate()
        result: dict[str, list[dict[str, Any]]] = {}
        for model in MODEL_NAMES:
            rows = [histories[seed]["models"][model]["history"] for seed in SEEDS]
            averaged = []
            for epoch_index in range(50):
                record = {"epoch": epoch_index + 1, "seeds": list(SEEDS)}
                for key in ("train_loss", "validation_loss"):
                    values = [float(seed_rows[epoch_index][key]) for seed_rows in rows]
                    record[key] = mean(values)
                    record[f"{key}_std"] = pstdev(values)
                averaged.append(record)
            result[model] = averaged
        return result

    def get_seed_training_history(self) -> dict[str, dict[str, list[dict[str, Any]]]]:
        _, histories, _ = self._validate()
        return {
            model: {str(seed): list(histories[seed]["models"][model]["history"]) for seed in SEEDS}
            for model in MODEL_NAMES
        }

    def get_evaluation_info(self) -> dict[str, Any]:
        _, _, report = self._validate()
        per_seed = [row for row in report["overall_metrics"] if row["model"] != "Persistence"]
        persistence = next(row for row in report["overall_metrics"] if row["model"] == "Persistence")
        models: dict[str, Any] = {}
        for model in MODEL_NAMES:
            rows = [row for row in per_seed if row["model"] == model]
            zone_rows = [row for row in report["per_zone_metrics"] if row["model"] == model]
            horizon_rows = [row for row in report["per_horizon_metrics"] if row["model"] == model]
            model_summary = next(row for row in report["three_seed_summary"] if row["model"] == model)
            by_zone: dict[str, Any] = {}
            for zone in report["zones"]:
                by_zone[zone] = {
                    "per_seed": [row for row in zone_rows if row["zone"] == zone],
                    "mae_mean": mean(row["MAE"] for row in zone_rows if row["zone"] == zone),
                    "rmse_mean": mean(row["RMSE"] for row in zone_rows if row["zone"] == zone),
                }
            models[model] = {
                "three_seed_summary": model_summary,
                "per_seed": rows,
                "per_zone": by_zone,
                "per_horizon": horizon_rows,
            }
        baseline_zones = {
            zone: next(row for row in report["per_zone_metrics"] if row["model"] == "Persistence" and row["zone"] == zone)
            for zone in report["zones"]
        }
        baseline_horizons = [row for row in report["per_horizon_metrics"] if row["model"] == "Persistence"]
        baseline = {
            "name": "Persistence",
            "role": "evaluation_baseline",
            "trained": False,
            "seed": "baseline",
            "MAE": persistence["MAE"],
            "RMSE": persistence["RMSE"],
            "macro_zone_MAE": persistence.get("macro_zone_MAE"),
            "per_zone": baseline_zones,
            "per_horizon": baseline_horizons,
        }
        return {
            "experiment_id": EXPERIMENT_ID,
            "test_split": {
                "name": report["evaluation"],
                "evaluation_sample_count": report["test_windows"],
                "lookback": report["lookback"],
                "forecast_horizon": report["forecast_horizon"],
                "cadence_minutes": report["cadence_minutes"],
                "seeds": report["seeds"],
                "target_zones": report["zones"],
            },
            "test_usage": {
                "statement": "Held-out chronological test data was used for final reporting only; it was not used for training or validation checkpoint selection.",
                "used_for_training": False,
                "used_for_checkpoint_selection": False,
                "used_for_model_configuration": False,
            },
            "overall": {
                "aggregation_note": "Model MAE/RMSE means and standard deviations summarize the three independent seed runs. Persistence is a single deterministic baseline.",
                "models": models,
                "baselines": {"persistence": baseline},
            },
            "horizon_wise": {
                "forecast_horizon_count": report["forecast_horizon"],
                "target_zone_names": [name.lower().replace(" ", "_") for name in report["zones"]],
                "models": {
                    model: [row for row in report["per_horizon_metrics"] if row["model"] == model]
                    for model in MODEL_NAMES
                },
                "baselines": {"persistence": baseline_horizons},
            },
            "source_report": report,
        }
