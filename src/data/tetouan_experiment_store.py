"""Read-only access to Tetouan dataset and experiment artifacts."""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any, Mapping

from src.data.tetouan_preprocess import DEFAULT_OUTPUT_DIR, FEATURE_NAMES, TARGET_NAMES

REPO_ROOT = Path(__file__).resolve().parents[2]
RUN_NAME = "tetouan_t4_20261101_v1"
MODEL_ORDER = ("LinearGridTransformer", "CDLinear")
BASELINE_ORDER = ("training_mean", "persistence")
DEFAULT_CHECKPOINT_DIR = REPO_ROOT / "checkpoints" / "tetouan"
DEFAULT_EXPERIMENT_DIR = REPO_ROOT / "results" / "tetouan" / RUN_NAME
DEFAULT_RAW_DIR = REPO_ROOT / "data" / "raw" / "tetouan"
DEFAULT_INSPECTION_REPORT = DEFAULT_RAW_DIR / "INSPECTION_REPORT.md"


class ArtifactStoreError(RuntimeError):
    """Base error for invalid or unavailable experiment artifacts."""


class MissingArtifactError(ArtifactStoreError, FileNotFoundError):
    """Raised when a required artifact is missing."""


class InvalidArtifactError(ArtifactStoreError, ValueError):
    """Raised when an artifact cannot be parsed or fails schema validation."""


class TetouanExperimentStore:
    """Read-only accessor for one dataset and one completed Tetouan run.

    All locations are configurable. Checkpoints are inspected as files only;
    this store never deserializes model weights or loads the NPZ arrays.
    """

    def __init__(
        self,
        *,
        run_name: str = RUN_NAME,
        artifact_dir: str | Path = DEFAULT_OUTPUT_DIR,
        experiment_dir: str | Path | None = None,
        checkpoint_dir: str | Path = DEFAULT_CHECKPOINT_DIR,
        raw_dir: str | Path = DEFAULT_RAW_DIR,
        inspection_report_path: str | Path | None = DEFAULT_INSPECTION_REPORT,
    ) -> None:
        if not run_name or Path(run_name).name != run_name or run_name in (".", ".."):
            raise ValueError("run_name must be a non-empty filename-safe identifier.")
        self.run_name = run_name
        self.artifact_dir = Path(artifact_dir)
        self.experiment_dir = (
            Path(experiment_dir)
            if experiment_dir is not None
            else REPO_ROOT / "results" / "tetouan" / run_name
        )
        self.checkpoint_dir = Path(checkpoint_dir)
        self.raw_dir = Path(raw_dir)
        self.inspection_report_path = (
            Path(inspection_report_path) if inspection_report_path is not None else None
        )

    @property
    def metadata_path(self) -> Path:
        return self.artifact_dir / "metadata.json"

    @property
    def series_path(self) -> Path:
        return self.artifact_dir / "series.npz"

    @property
    def manifest_path(self) -> Path:
        return self.experiment_dir / f"{self.run_name}_manifest.json"

    @property
    def history_path(self) -> Path:
        return self.experiment_dir / f"{self.run_name}_history.json"

    @property
    def evaluation_path(self) -> Path:
        return self.experiment_dir / f"{self.run_name}_test_evaluation.json"

    @property
    def horizon_evaluation_path(self) -> Path:
        return self.experiment_dir / f"{self.run_name}_horizon_evaluation.json"

    def checkpoint_path(self, model_name: str) -> Path:
        if model_name not in MODEL_ORDER:
            raise ValueError(f"Unknown model {model_name!r}; expected one of {MODEL_ORDER}.")
        suffix = "lineargridtransformer" if model_name == "LinearGridTransformer" else "cdlinear"
        return self.checkpoint_dir / f"{self.run_name}_{suffix}.pt"

    @staticmethod
    def _read_json(path: Path, label: str) -> dict[str, Any]:
        if not path.is_file():
            raise MissingArtifactError(f"Required {label} is missing: {path}")
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise InvalidArtifactError(f"Could not read {label} at {path}: {exc}") from exc
        if not isinstance(value, dict):
            raise InvalidArtifactError(f"{label} at {path} must contain a JSON object.")
        return value

    @staticmethod
    def _require_keys(value: Mapping[str, Any], keys: tuple[str, ...], label: str) -> None:
        missing = [key for key in keys if key not in value]
        if missing:
            raise InvalidArtifactError(f"{label} is missing required fields: {', '.join(missing)}.")

    def _read_metadata(self) -> dict[str, Any]:
        metadata = self._read_json(self.metadata_path, "dataset metadata")
        self._require_keys(
            metadata,
            ("source", "row_count", "features", "targets", "timestamps", "split",
             "windows", "normalization", "units_status"),
            "dataset metadata",
        )
        if not isinstance(metadata["source"], dict):
            raise InvalidArtifactError("dataset metadata source must be an object.")
        if not isinstance(metadata["features"], dict) or not isinstance(metadata["targets"], dict):
            raise InvalidArtifactError("dataset feature/target metadata must be objects.")
        if metadata["features"].get("names") != list(FEATURE_NAMES):
            raise InvalidArtifactError("Dataset feature names/order do not match the Tetouan contract.")
        if metadata["targets"].get("names") != list(TARGET_NAMES):
            raise InvalidArtifactError("Dataset target names/order do not match the Tetouan contract.")
        return metadata

    def _read_manifest(self) -> dict[str, Any]:
        manifest = self._read_json(self.manifest_path, "experiment manifest")
        self._require_keys(manifest, ("run_metadata", "models"), "experiment manifest")
        run = manifest["run_metadata"]
        if not isinstance(run, dict):
            raise InvalidArtifactError("Manifest run_metadata must be an object.")
        if run.get("run_name") != self.run_name:
            raise InvalidArtifactError(
                f"Manifest run_name {run.get('run_name')!r} does not match {self.run_name!r}."
            )
        if not isinstance(manifest["models"], dict):
            raise InvalidArtifactError("Manifest models must be an object.")
        for name in MODEL_ORDER:
            if name not in manifest["models"]:
                raise InvalidArtifactError(f"Manifest is missing model {name}.")
        return manifest

    def _read_history(self) -> dict[str, Any]:
        history = self._read_json(self.history_path, "training history")
        self._require_keys(history, ("run_metadata", "models"), "training history")
        run_metadata = history.get("run_metadata")
        models = history.get("models")
        if not isinstance(run_metadata, dict) or run_metadata.get("run_name") != self.run_name:
            raise InvalidArtifactError("Training history run identifier does not match the store.")
        if not isinstance(models, dict):
            raise InvalidArtifactError("Training history models must be an object.")
        for name in MODEL_ORDER:
            model = models.get(name)
            if not isinstance(model, dict) or not isinstance(model.get("history"), list):
                raise InvalidArtifactError(f"Training history is missing valid rows for {name}.")
        return history

    def _read_evaluation(self) -> dict[str, Any]:
        report = self._read_json(self.evaluation_path, "overall evaluation report")
        self._require_keys(
            report, ("dataset", "split", "normalization", "checkpoints", "metrics", "test_usage"),
            "overall evaluation report",
        )
        target_order = report.get("normalization", {}).get("target_channel_names")
        if target_order != list(TARGET_NAMES):
            raise InvalidArtifactError("Overall evaluation target ordering is invalid.")
        metrics = report["metrics"]
        if not isinstance(metrics, dict) or not isinstance(metrics.get("models"), dict):
            raise InvalidArtifactError("Overall evaluation metrics.models must be an object.")
        if not isinstance(metrics.get("baselines"), dict):
            raise InvalidArtifactError("Overall evaluation metrics.baselines must be an object.")
        for name in MODEL_ORDER:
            if name not in metrics["models"]:
                raise InvalidArtifactError(f"Overall evaluation is missing model metrics for {name}.")
        for name in BASELINE_ORDER:
            if name not in metrics["baselines"]:
                raise InvalidArtifactError(f"Overall evaluation is missing baseline {name}.")
        return report

    def _read_horizon_evaluation(self) -> dict[str, Any]:
        report = self._read_json(self.horizon_evaluation_path, "horizon evaluation report")
        self._require_keys(
            report,
            ("experiment_id", "test_window_count", "forecast_horizon_count",
             "target_zone_names", "metric_definitions", "models", "derivation"),
            "horizon evaluation report",
        )
        if report["experiment_id"] != self.run_name:
            raise InvalidArtifactError("Horizon evaluation experiment_id does not match the store.")
        if report["target_zone_names"] != list(TARGET_NAMES):
            raise InvalidArtifactError("Horizon evaluation target ordering is invalid.")
        if not isinstance(report["models"], dict) or report["forecast_horizon_count"] != 6:
            raise InvalidArtifactError("Horizon evaluation model/horizon structure is invalid.")
        for name in MODEL_ORDER:
            record = report["models"].get(name)
            horizons = record.get("per_horizon") if isinstance(record, dict) else None
            if not isinstance(horizons, list) or len(horizons) != report["forecast_horizon_count"]:
                raise InvalidArtifactError(f"Horizon evaluation is incomplete for {name}.")
        return report

    @staticmethod
    def _sha256(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    def _quality_findings(self) -> dict[str, Any] | None:
        path = self.inspection_report_path
        if path is None or not path.is_file():
            return None
        try:
            text = path.read_text(encoding="utf-8")
        except (OSError, UnicodeError):
            return {
                "source": str(path),
                "available": False,
                "issue": "Inspection report could not be read.",
            }

        def count(label: str) -> int | None:
            prefix = f"- {label}: **"
            for line in text.splitlines():
                if line.startswith(prefix):
                    raw = line[len(prefix):].split("**", 1)[0]
                    match = re.match(r"\s*(\d+)\b", raw)
                    return int(match.group(1)) if match else None
            return None

        ordered = any(
            line.startswith("- Chronological ordering: **yes**")
            for line in text.splitlines()
        )
        return {
            "source": str(path.resolve()),
            "available": True,
            "missing_values_per_column": count("Missing values"),
            "exact_duplicate_rows": count("Exact duplicate rows"),
            "duplicate_timestamps": count("Duplicate parsed timestamps"),
            "timestamp_parse_failures": count("Timestamp parse failures"),
            "irregular_intervals": count("Irregular intervals within the observed range"),
            "chronological_ordering": "strictly increasing" if ordered else None,
        }

    def get_dataset_info(self) -> dict[str, Any]:
        """Return provenance, schema, coverage, partitions, and saved normalization."""
        metadata = self._read_metadata()
        timestamps = metadata["timestamps"]
        source = metadata["source"]
        source_path = self.raw_dir / source.get("file_name", "")
        return {
            "dataset_name": "Power Consumption of Tetouan City",
            "source": dict(source),
            "source_path": str(source_path.resolve()),
            "source_file_available": source_path.is_file(),
            "row_count": metadata["row_count"],
            "date_range": {
                "first": timestamps.get("first_timestamp"),
                "last": timestamps.get("last_timestamp"),
            },
            "cadence_minutes": timestamps.get("cadence_minutes"),
            "timezone": timestamps.get("timezone"),
            "timezone_status": timestamps.get("timezone_status"),
            "features": {
                "names": list(metadata["features"]["names"]),
                "source_columns": metadata["features"].get("source_columns"),
            },
            "targets": {
                "names": list(metadata["targets"]["names"]),
                "source_columns": metadata["targets"].get("source_columns"),
            },
            "splits": {
                "method": metadata["split"].get("method"),
                "partitions": metadata["split"].get("partitions"),
            },
            "windows": dict(metadata["windows"]),
            "normalization": dict(metadata["normalization"]),
            "data_quality": self._quality_findings(),
            "coverage_uncertainties": {
                "missing_final_calendar_day": timestamps.get("missing_final_calendar_day"),
                "handling": timestamps.get("handling"),
                "units": metadata.get("units"),
                "units_status": metadata.get("units_status"),
            },
            "metadata_artifact": str(self.metadata_path.resolve()),
        }

    def get_training_history(self) -> dict[str, list[dict[str, Any]]]:
        """Return saved epoch rows in stable model order; no recomputation."""
        data = self._read_history()
        return {
            name: list(data["models"][name]["history"])
            for name in MODEL_ORDER
        }

    def get_experiment_info(self) -> dict[str, Any]:
        """Return manifest provenance/configuration and local checkpoint references."""
        manifest = self._read_manifest()
        history_file = self._read_history()
        run = manifest["run_metadata"]
        history_models = history_file["models"]
        manifest_run_models = run.get("models")
        if not isinstance(manifest_run_models, dict):
            raise InvalidArtifactError("Manifest run_metadata.models must be an object.")
        models: dict[str, Any] = {}
        for name in MODEL_ORDER:
            record = manifest["models"][name]
            history = history_models[name]
            config = record.get("configuration")
            if not isinstance(config, dict):
                raise InvalidArtifactError(f"Manifest model config for {name} must be an object.")
            if manifest_run_models.get(name) != config:
                raise InvalidArtifactError(f"Run/model configuration mismatch for {name}.")
            if history.get("best_epoch") != record.get("best_validation_epoch"):
                raise InvalidArtifactError(f"Best epoch mismatch between manifest and history for {name}.")
            if history.get("best_validation_loss") != record.get("best_validation_loss"):
                raise InvalidArtifactError(f"Best validation loss mismatch for {name}.")
            checkpoint = self.checkpoint_path(name)
            if not checkpoint.is_file():
                raise MissingArtifactError(f"Required checkpoint for {name} is missing: {checkpoint}")
            models[name] = {
                "configuration": dict(config),
                "checkpoint": {
                    "path": str(checkpoint.resolve()),
                    "exists": True,
                    "size_bytes": checkpoint.stat().st_size,
                    "sha256_observed": self._sha256(checkpoint),
                    "recorded_manifest_reference": record.get("checkpoint"),
                },
                "best_validation_epoch": record.get("best_validation_epoch"),
                "best_validation_loss": record.get("best_validation_loss"),
                "history": list(history["history"]),
            }
        return {
            "run_identifier": run.get("run_name"),
            "run_type": run.get("run_type"),
            "created_utc": run.get("created_utc"),
            "provenance": {
                "git_commit": run.get("git_commit"),
                "training_environment": {
                    key: run.get(key)
                    for key in (
                        "python_version", "pytorch_version", "device", "cuda_available",
                        "cuda_version", "gpu_name",
                    )
                },
            },
            "training_configuration": run.get("training"),
            "optimizer": run.get("optimizer"),
            "loss": run.get("loss"),
            "seed_policy": run.get("seed_policy"),
            "test_partition_used": run.get("test_partition_used"),
            "models": models,
            "artifact_references": self._artifact_references(),
            "manifest_artifact": str(self.manifest_path.resolve()),
            "history_artifact": str(self.history_path.resolve()),
        }

    def _artifact_references(self) -> dict[str, str]:
        references = {
            "dataset_metadata": self.metadata_path,
            "processed_series": self.series_path,
            "manifest": self.manifest_path,
            "training_history": self.history_path,
            "overall_evaluation": self.evaluation_path,
            "horizon_evaluation": self.horizon_evaluation_path,
        }
        for name in MODEL_ORDER:
            references[f"checkpoint_{name}"] = self.checkpoint_path(name)
        visual_dir = self.experiment_dir / "visualisations"
        for name in (
            "actual_vs_predicted_three_zones.png",
            "training_validation_loss.png",
            "test_model_comparison.png",
            "per_zone_mae.png",
            "horizon_wise_mae.png",
            "test_predictions.npz",
        ):
            references[f"visualisation_{name}"] = visual_dir / name
        return {key: str(path.resolve()) for key, path in references.items()}

    def get_evaluation_info(self) -> dict[str, Any]:
        """Return saved overall and horizon metrics, preserving model/baseline groups."""
        overall = self._read_evaluation()
        horizon = self._read_horizon_evaluation()
        sample_count = overall["split"].get("evaluation_sample_count")
        if sample_count != horizon.get("test_window_count"):
            raise InvalidArtifactError("Overall and horizon reports have different test-window counts.")
        return {
            "experiment_id": horizon["experiment_id"],
            "test_split": dict(overall["split"]),
            "test_usage": dict(overall["test_usage"]),
            "overall": {
                "aggregation_note": overall["metrics"].get("aggregation_note"),
                "models": {
                    name: overall["metrics"]["models"][name]
                    for name in MODEL_ORDER
                },
                "baselines": {
                    name: overall["metrics"]["baselines"][name]
                    for name in BASELINE_ORDER
                },
            },
            "horizon_wise": {
                "forecast_horizon_count": horizon["forecast_horizon_count"],
                "target_zone_names": list(horizon["target_zone_names"]),
                "metric_definitions": dict(horizon["metric_definitions"]),
                "models": {
                    name: horizon["models"][name]["per_horizon"]
                    for name in MODEL_ORDER
                },
                "consistency_with_overall": {
                    name: horizon["models"][name]["consistency_with_existing_overall_metrics"]
                    for name in MODEL_ORDER
                },
            },
            "overall_evaluation_artifact": str(self.evaluation_path.resolve()),
            "horizon_evaluation_artifact": str(self.horizon_evaluation_path.resolve()),
        }

    def verify_artifacts(self) -> dict[str, Any]:
        """Check required artifact presence, JSON structure, and recorded data hashes."""
        paths: dict[str, Path] = {
            "dataset_metadata": self.metadata_path,
            "processed_series": self.series_path,
            "manifest": self.manifest_path,
            "training_history": self.history_path,
            "overall_evaluation": self.evaluation_path,
            "horizon_evaluation": self.horizon_evaluation_path,
        }
        paths.update({f"checkpoint_{name}": self.checkpoint_path(name) for name in MODEL_ORDER})
        visual_dir = self.experiment_dir / "visualisations"
        for name in (
            "actual_vs_predicted_three_zones.png",
            "training_validation_loss.png",
            "test_model_comparison.png",
            "per_zone_mae.png",
            "horizon_wise_mae.png",
            "test_predictions.npz",
        ):
            paths[f"visualisation_{name}"] = visual_dir / name
        checks: dict[str, Any] = {}
        issues: list[str] = []
        for label, path in paths.items():
            exists = path.is_file()
            checks[label] = {
                "path": str(path.resolve()),
                "exists": exists,
                "size_bytes": path.stat().st_size if exists else None,
            }
            if exists:
                checks[label]["sha256_observed"] = self._sha256(path)
            else:
                issues.append(f"Missing {label}: {path}")

        try:
            metadata = self._read_metadata()
            manifest = self._read_manifest()
            self.get_experiment_info()
            self.get_training_history()
            self.get_evaluation_info()
            fingerprints = manifest["run_metadata"].get("processed_artifacts", {})
            for filename in ("metadata.json", "series.npz"):
                path = self.artifact_dir / filename
                expected = fingerprints.get(filename)
                if not isinstance(expected, dict):
                    issues.append(f"Manifest has no processed artifact fingerprint for {filename}.")
                    checks.setdefault("fingerprints", {})[filename] = {
                        "valid": False,
                        "issue": "fingerprint missing",
                    }
                    continue
                if not path.is_file():
                    checks.setdefault("fingerprints", {})[filename] = {
                        "valid": False,
                        "issue": "file missing",
                    }
                    continue
                observed_hash = self._sha256(path)
                valid = (
                    path.stat().st_size == expected.get("size_bytes")
                    and observed_hash == expected.get("sha256")
                )
                checks.setdefault("fingerprints", {})[filename] = {
                    "path": str(path.resolve()),
                    "size_bytes": path.stat().st_size,
                    "sha256_observed": observed_hash,
                    "sha256_expected": expected.get("sha256"),
                    "valid": valid,
                }
                if not valid:
                    issues.append(f"Processed artifact fingerprint mismatch: {filename}.")
            if metadata.get("source", {}).get("sha256") != manifest["run_metadata"].get(
                "dataset", {}
            ).get("source_sha256"):
                issues.append("Dataset source SHA-256 differs between metadata and manifest.")
            source_path = self.raw_dir / metadata["source"].get("file_name", "")
            if source_path.is_file():
                observed_source_hash = self._sha256(source_path)
                checks["raw_source_csv"] = {
                    "path": str(source_path.resolve()),
                    "exists": True,
                    "sha256_observed": observed_source_hash,
                    "sha256_expected": metadata["source"].get("sha256"),
                    "valid": observed_source_hash == metadata["source"].get("sha256"),
                }
                if not checks["raw_source_csv"]["valid"]:
                    issues.append("Raw source CSV SHA-256 differs from dataset metadata.")
        except ArtifactStoreError as exc:
            issues.append(str(exc))
        return {
            "status": "ok" if not issues else "issues",
            "checks": checks,
            "issues": issues,
        }
