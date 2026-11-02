"""Derive horizon-wise Tetouan metrics from frozen test predictions.

This module reads the saved evaluation JSON and test_predictions.npz only. It
never loads a model, constructs a test Dataset, or runs inference.
"""
from __future__ import annotations

import io
import json
from pathlib import Path
from typing import Any, Mapping

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.data.tetouan_preprocess import TARGET_NAMES
from src.training.evaluate_tetouan import calculate_metrics

REPO_ROOT = Path(__file__).resolve().parents[2]
RUN_NAME = "tetouan_t4_20261101_v1"
EXPERIMENT_DIR = REPO_ROOT / "results" / "tetouan" / RUN_NAME
EVALUATION_FILENAME = f"{RUN_NAME}_test_evaluation.json"
PREDICTIONS_FILENAME = "test_predictions.npz"
HORIZON_REPORT_FILENAME = f"{RUN_NAME}_horizon_evaluation.json"
HORIZON_PLOT_FILENAME = "horizon_wise_mae.png"
PREDICTION_KEYS = {
    "LinearGridTransformer": "lineargridtransformer",
    "CDLinear": "cdlinear",
}


def calculate_horizon_metrics(
    actual: np.ndarray,
    predictions: np.ndarray,
    zone_names: tuple[str, ...] = TARGET_NAMES,
) -> list[dict[str, Any]]:
    """Calculate zone, macro, and pooled RMSE metrics for each lead step."""
    actual = np.asarray(actual, dtype=np.float64)
    predictions = np.asarray(predictions, dtype=np.float64)
    if actual.ndim != 3 or predictions.shape != actual.shape:
        raise ValueError(
            "Actual and prediction arrays must have matching [windows, horizons, zones] shapes."
        )
    if actual.shape[0] == 0 or actual.shape[1] == 0:
        raise ValueError("Prediction arrays must contain at least one window and horizon.")
    if actual.shape[2] != len(zone_names) or not zone_names:
        raise ValueError("Zone names must match the prediction target dimension.")
    if not np.isfinite(actual).all() or not np.isfinite(predictions).all():
        raise ValueError("Actual values and predictions must all be finite.")

    results: list[dict[str, Any]] = []
    for horizon_index in range(actual.shape[1]):
        # Reuse the evaluator's established per-zone, macro, and pooled formulas.
        result = calculate_metrics(
            predictions[:, horizon_index : horizon_index + 1, :],
            actual[:, horizon_index : horizon_index + 1, :],
            zone_names,
        )
        results.append(
            {
                "horizon_step": horizon_index + 1,
                "horizon_minutes": (horizon_index + 1) * 10,
                "per_zone": result["per_zone"],
                "macro_average_across_zones": result["macro_average_across_zones"],
                "pooled_across_test_windows_and_zones": {
                    "rmse": result["pooled_across_all_samples_horizons_and_zones"]["rmse"]
                },
            }
        )
    return results


def _verify_overall_consistency(
    actual: np.ndarray,
    predictions: np.ndarray,
    saved_overall: Mapping[str, Any],
    zone_names: tuple[str, ...],
    *,
    rtol: float = 1e-7,
    atol: float = 1e-7,
) -> dict[str, Any]:
    """Check the full-array metrics and horizon aggregation against the saved report."""
    recalculated = calculate_metrics(predictions, actual, zone_names)["per_zone"]
    horizons = calculate_horizon_metrics(actual, predictions, zone_names)
    checks: dict[str, Any] = {}
    for zone in zone_names:
        saved = saved_overall.get(zone)
        if not isinstance(saved, Mapping):
            raise ValueError(f"Saved evaluation report has no overall metrics for {zone}.")
        aggregate_mae = float(
            np.mean([row["per_zone"][zone]["mae"] for row in horizons])
        )
        aggregate_rmse = float(
            np.sqrt(np.mean([row["per_zone"][zone]["rmse"] ** 2 for row in horizons]))
        )
        checks[zone] = {
            "mae_matches_saved": bool(
                np.isclose(recalculated[zone]["mae"], saved["mae"], rtol=rtol, atol=atol)
                and np.isclose(aggregate_mae, saved["mae"], rtol=rtol, atol=atol)
            ),
            "rmse_matches_saved": bool(
                np.isclose(recalculated[zone]["rmse"], saved["rmse"], rtol=rtol, atol=atol)
                and np.isclose(aggregate_rmse, saved["rmse"], rtol=rtol, atol=atol)
            ),
            "recalculated_overall": recalculated[zone],
            "aggregated_from_horizons": {
                "mae": aggregate_mae,
                "rmse": aggregate_rmse,
            },
            "saved_overall": dict(saved),
        }
        if not checks[zone]["mae_matches_saved"] or not checks[zone]["rmse_matches_saved"]:
            raise ValueError(
                f"Horizon metrics do not reproduce saved overall metrics for {zone}."
            )
    return {
        "tolerance": {"rtol": rtol, "atol": atol},
        "all_match": all(
            row["mae_matches_saved"] and row["rmse_matches_saved"]
            for row in checks.values()
        ),
        "per_zone": checks,
    }


def build_horizon_evaluation_report(
    evaluation_report: Mapping[str, Any],
    prediction_arrays: Mapping[str, np.ndarray],
    *,
    source_evaluation_reference: str,
    prediction_artifact_reference: str,
) -> dict[str, Any]:
    """Build a serializable report from saved predictions and existing metrics."""
    split = evaluation_report.get("split", {})
    expected_windows = split.get("evaluation_sample_count")
    if not isinstance(expected_windows, int) or expected_windows <= 0:
        raise ValueError("Source evaluation report has an invalid test-window count.")
    zone_names = tuple(evaluation_report.get("normalization", {}).get("target_channel_names", ()))
    if zone_names != tuple(TARGET_NAMES):
        raise ValueError("Source report target ordering does not match the Tetouan target contract.")
    source_models = evaluation_report.get("metrics", {}).get("models", {})
    source_checkpoints = evaluation_report.get("checkpoints", {})
    if not isinstance(source_models, Mapping) or not isinstance(source_checkpoints, Mapping):
        raise ValueError("Source evaluation report is missing model metrics/checkpoint metadata.")

    required_keys = {"actual", *PREDICTION_KEYS.values()}
    if set(prediction_arrays) != required_keys:
        raise ValueError(
            f"Prediction arrays must have exactly keys {sorted(required_keys)}; "
            f"got {sorted(prediction_arrays)}."
        )
    models: dict[str, Any] = {}
    for model_name, key in PREDICTION_KEYS.items():
        if key not in prediction_arrays:
            raise ValueError(f"Prediction artifact is missing NPZ key {key!r}.")
        if model_name not in source_models or model_name not in source_checkpoints:
            raise ValueError(f"Source evaluation report is missing {model_name}.")
        actual = np.asarray(prediction_arrays.get("actual"), dtype=np.float64)
        predicted = np.asarray(prediction_arrays[key], dtype=np.float64)
        expected_shape = (expected_windows, 6, len(zone_names))
        if actual.shape != expected_shape or predicted.shape != expected_shape:
            raise ValueError(
                f"{model_name} arrays must both have shape {expected_shape}; "
                f"got actual={actual.shape}, prediction={predicted.shape}."
            )
        if not np.isfinite(actual).all() or not np.isfinite(predicted).all():
            raise ValueError("Prediction artifact contains non-finite values.")

        saved_per_zone = source_models[model_name].get("per_zone", {})
        consistency = _verify_overall_consistency(
            actual, predicted, saved_per_zone, zone_names
        )
        models[model_name] = {
            "prediction_array_key": key,
            "per_horizon": calculate_horizon_metrics(actual, predicted, zone_names),
            "existing_overall_per_zone_metrics": saved_per_zone,
            "consistency_with_existing_overall_metrics": consistency,
        }

    run_ids = {
        source_checkpoints[name].get("run_name")
        for name in PREDICTION_KEYS
    }
    if len(run_ids) != 1 or None in run_ids:
        raise ValueError("Source evaluation checkpoints do not share one experiment identifier.")
    return {
        "report_schema_version": 1,
        "experiment_id": run_ids.pop(),
        "source_evaluation_report": source_evaluation_reference,
        "prediction_artifact": prediction_artifact_reference,
        "test_window_count": expected_windows,
        "forecast_horizon_count": 6,
        "target_zone_names": list(zone_names),
        "metric_definitions": {
            "scale": "Original dataset numeric values; physical units are undocumented.",
            "per_horizon_per_zone_mae": (
                "Mean absolute error across all test windows for one zone at one forecast lead."
            ),
            "per_horizon_per_zone_rmse": (
                "Square root of mean squared error across all test windows for one zone at one forecast lead."
            ),
            "macro_average": (
                "Arithmetic mean of the three zone-specific MAE or RMSE values at that horizon."
            ),
            "pooled_rmse": (
                "Square root of mean squared error pooled across all test windows and all three zones "
                "at that horizon."
            ),
            "horizon_step": "Step 1 is 10 minutes ahead; step 6 is 60 minutes ahead.",
            "window_order": (
                "Arrays retain the source evaluator's sequential, non-shuffled test-loader order; "
                "actuals and predictions were appended from the same batches."
            ),
        },
        "models": models,
        "derivation": {
            "source": "Saved test_predictions.npz arrays only.",
            "training_rerun": False,
            "held_out_evaluation_rerun": False,
            "test_dataset_instantiated": False,
            "test_metrics_recalculated_from_models": False,
        },
    }


def create_horizon_mae_plot(report: Mapping[str, Any], output_path: str | Path) -> Path:
    """Write a new six-step, per-zone MAE plot, refusing to overwrite."""
    path = Path(output_path)
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite existing visualization: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    x = np.arange(1, 7)
    figure, axes = plt.subplots(1, 2, figsize=(12, 5.25), sharex=True, sharey=True,
                                constrained_layout=True)
    try:
        colors = ("#1f77b4", "#ff7f0e", "#2ca02c")
        for axis, model_name in zip(axes, PREDICTION_KEYS):
            for zone_index, zone in enumerate(report["target_zone_names"]):
                label = f"Zone {zone_index + 1} consumption"
                values = [
                    report["models"][model_name]["per_horizon"][step]["per_zone"][zone]["mae"]
                    for step in range(6)
                ]
                axis.plot(x, values, marker="o", linewidth=2, color=colors[zone_index],
                          label=label)
            axis.set_xticks(x)
            axis.set_xlabel("Forecast horizon (step; 10 minutes per step)")
            axis.set_title(model_name)
            axis.grid(True, alpha=0.3)
        axes[0].set_ylabel(
            "MAE (original dataset numeric units; physical units undocumented)"
        )
        handles, labels = axes[0].get_legend_handles_labels()
        figure.legend(handles, labels, title="Target zone", loc="outside lower center",
                      ncol=3)
        figure.suptitle("Tetouan Test MAE by Forecast Horizon and Zone")
        buffer = io.BytesIO()
        figure.savefig(buffer, format="png", dpi=160)
        payload = buffer.getvalue()
    finally:
        plt.close(figure)
    with path.open("xb") as stream:
        stream.write(payload)
    return path


def create_horizon_evaluation_artifacts(
    experiment_dir: str | Path = EXPERIMENT_DIR,
) -> tuple[Path, Path, dict[str, Any]]:
    """Derive and save the separate JSON report and MAE visualization."""
    root = Path(experiment_dir)
    evaluation_path = root / EVALUATION_FILENAME
    predictions_path = root / "visualisations" / PREDICTIONS_FILENAME
    report_path = root / HORIZON_REPORT_FILENAME
    plot_path = root / "visualisations" / HORIZON_PLOT_FILENAME
    collisions = [str(path) for path in (report_path, plot_path) if path.exists()]
    if collisions:
        raise FileExistsError("Refusing to overwrite existing horizon artifacts: " + ", ".join(collisions))
    if not evaluation_path.is_file() or not predictions_path.is_file():
        raise FileNotFoundError("Saved source evaluation report or prediction NPZ is missing.")

    evaluation = json.loads(evaluation_path.read_text(encoding="utf-8"))
    with np.load(predictions_path, allow_pickle=False) as archive:
        arrays = {name: archive[name] for name in archive.files}
    def reference(path: Path) -> str:
        try:
            return str(path.resolve().relative_to(REPO_ROOT))
        except ValueError:
            return str(path.resolve())

    report = build_horizon_evaluation_report(
        evaluation,
        arrays,
        source_evaluation_reference=reference(evaluation_path),
        prediction_artifact_reference=reference(predictions_path),
    )

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_bytes = json.dumps(report, indent=2, allow_nan=False).encode("utf-8")
    create_horizon_mae_plot(report, plot_path)
    with report_path.open("xb") as stream:
        stream.write(report_bytes)
    return report_path, plot_path, report


if __name__ == "__main__":
    report_file, plot_file, _report = create_horizon_evaluation_artifacts()
    print(f"Horizon evaluation report: {report_file}")
    print(f"Horizon-wise MAE visualization: {plot_file}")
