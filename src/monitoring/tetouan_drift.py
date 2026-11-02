"""NumPy PSI monitoring for features, predictions, and observed performance.

Drift is a distributional signal, not proof of accuracy degradation. Reports
contain aggregate statistics only; input rows and uploaded observations are
never copied into MLflow artifacts.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

FEATURES = ("zone_1_power_consumption", "zone_2_power_consumption", "zone_3_power_consumption", "temperature", "humidity", "wind_speed", "general_diffuse_flows", "diffuse_flows")
ZONES = FEATURES[:3]


def _array(value: Any, name: str, ndim: int) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64)
    if result.ndim != ndim or result.size == 0:
        raise ValueError(f"{name} must be a non-empty {ndim}-dimensional array")
    if not np.isfinite(result).all():
        raise ValueError(f"{name} contains non-finite values")
    return result


def _psi(reference: np.ndarray, current: np.ndarray, bins: int = 10) -> tuple[float, list[float]]:
    if bins < 2:
        raise ValueError("bins must be at least two")
    edges = np.unique(np.quantile(reference, np.linspace(0, 1, bins + 1)))
    if len(edges) < 2:
        center = float(reference[0])
        margin = max(abs(center) * 1e-6, 1e-6)
        edges = np.asarray([center - 2 * margin, center - margin, center + margin, center + 2 * margin], dtype=np.float64)
    edges[0], edges[-1] = -np.inf, np.inf
    ref_count = np.histogram(reference, bins=edges)[0].astype(np.float64)
    cur_count = np.histogram(current, bins=edges)[0].astype(np.float64)
    eps = 1e-6
    ref_rate = np.maximum(ref_count / len(reference), eps)
    cur_rate = np.maximum(cur_count / len(current), eps)
    return float(np.sum((cur_rate - ref_rate) * np.log(cur_rate / ref_rate))), [float(x) for x in edges[1:-1]]


def _level(psi: float, warn: float = 0.1, alert: float = 0.25) -> str:
    if warn <= 0 or alert <= warn:
        raise ValueError("thresholds must satisfy 0 < warn < alert")
    return "alert" if psi >= alert else "warning" if psi >= warn else "stable"


def feature_drift(reference: Any, current: Any, feature_names: tuple[str, ...] = FEATURES, *, bins: int = 10, warn: float = 0.1, alert: float = 0.25) -> dict[str, Any]:
    ref, cur = _array(reference, "reference features", 2), _array(current, "current features", 2)
    if ref.shape[1] != len(feature_names) or cur.shape[1] != len(feature_names):
        raise ValueError("feature arrays must match the canonical eight-feature order")
    details = {}
    for index, name in enumerate(feature_names):
        psi, cuts = _psi(ref[:, index], cur[:, index], bins)
        details[name] = {"psi": psi, "level": _level(psi, warn, alert), "reference_count": int(len(ref)), "current_count": int(len(cur)), "reference_quantile_cuts": cuts}
    return {"kind": "feature_drift", "reference": "training partition", "feature_order": list(feature_names), "thresholds": {"warning_psi": warn, "alert_psi": alert, "status": "provisional heuristics; calibrate on validation windows"}, "sample_counts": {"reference": int(len(ref)), "current": int(len(cur))}, "features": details, "drift_detected": any(item["level"] != "stable" for item in details.values())}


def prediction_drift(reference: Any, current: Any, *, model_version: str, experiment_id: str, warn: float = 0.1, alert: float = 0.25) -> dict[str, Any]:
    ref, cur = _array(reference, "reference predictions", 3), _array(current, "current predictions", 3)
    if ref.shape[1:] != (6, 3) or cur.shape[1:] != (6, 3):
        raise ValueError("prediction arrays must have shape [N, 6, 3]")
    values = {}
    for zone in range(3):
        for horizon in range(6):
            psi, _ = _psi(ref[:, horizon, zone], cur[:, horizon, zone])
            values[f"{ZONES[zone]}__h{horizon + 1}"] = {"psi": psi, "level": _level(psi, warn, alert)}
    return {"kind": "prediction_distribution_drift", "model_version": model_version, "experiment_id": experiment_id, "zone_order": list(ZONES), "horizon_minutes": [10, 20, 30, 40, 50, 60], "sample_counts": {"reference": len(ref), "current": len(cur)}, "zone_horizon": values, "interpretation": "Prediction distribution movement does not establish forecasting accuracy degradation.", "drift_detected": any(v["level"] != "stable" for v in values.values())}


def performance_report(predictions: Any, actuals: Any, reference_metrics: dict[str, Any] | None = None) -> dict[str, Any]:
    pred, actual = _array(predictions, "predictions", 3), _array(actuals, "actuals", 3)
    if pred.shape != actual.shape or pred.shape[1:] != (6, 3):
        raise ValueError("predictions and actuals must match [N, 6, 3]")
    err = pred - actual
    zones = {}
    horizons = {}
    for z, name in enumerate(ZONES):
        zones[name] = {"mae": float(np.mean(np.abs(err[..., z]))), "rmse": float(np.sqrt(np.mean(err[..., z] ** 2)))}
    for h in range(6):
        horizons[str((h + 1) * 10)] = {"mae": float(np.mean(np.abs(err[:, h, :]))), "rmse": float(np.sqrt(np.mean(err[:, h, :] ** 2)))}
    zone_comparison = horizon_comparison = None
    if reference_metrics is not None:
        zone_comparison, horizon_comparison = {}, {}
        for name, metrics in zones.items():
            base = reference_metrics.get("per_zone", {}).get(name, {})
            zone_comparison[name] = {f"{metric}_higher_than_reference": (value > float(base[metric])) if metric in base else None for metric, value in metrics.items()}
        for horizon, metrics in horizons.items():
            base = reference_metrics.get("per_horizon_minutes", {}).get(horizon, {})
            horizon_comparison[horizon] = {f"{metric}_higher_than_reference": (value > float(base[metric])) if metric in base else None for metric, value in metrics.items()}
    return {"kind": "observed_performance", "sample_count": len(pred), "per_zone": zones, "per_horizon_minutes": horizons, "comparison_to_validation_reference": {"per_zone": zone_comparison, "per_horizon_minutes": horizon_comparison} if reference_metrics is not None else None, "interpretation": "Metrics require aligned observed ground truth. Higher errors than validation are monitoring signals, not significance tests; feature/prediction drift alone is not an accuracy claim."}


def repeated_alert(history: list[dict[str, Any]], *, consecutive: int = 3) -> bool:
    if consecutive <= 0:
        raise ValueError("consecutive must be positive")
    return len(history) >= consecutive and all(bool(item.get("drift_detected")) for item in history[-consecutive:])


def log_monitoring_report(report: dict[str, Any], *, experiment_name: str = "Tetouan Energy Forecasting", tracking_uri: str | None = None) -> str:
    import mlflow
    from mlflow.tracking import MlflowClient
    from src.mlops.mlflow_config import ensure_local_storage, get_config, get_or_create_experiment
    config_object = get_config()
    uri = tracking_uri or config_object.tracking_uri
    ensure_local_storage(config_object)
    mlflow.set_tracking_uri(uri)
    client = MlflowClient(tracking_uri=uri)
    experiment_id = get_or_create_experiment(client, experiment_name, config_object)
    mlflow.set_experiment(experiment_id=experiment_id)
    with mlflow.start_run(experiment_id=experiment_id, run_name=f"monitor-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}", tags={"run_role": "drift_monitoring", "contains_raw_observations": "false", "test_metrics_used_for_selection": "false"}) as run:
        mlflow.log_params({"monitor_kind": report.get("kind"), "reference_fingerprint": report.get("reference_fingerprint", "unknown"), "model_version": report.get("model_version", "unknown"), "experiment_id": report.get("experiment_id", "unknown")})
        mlflow.log_metric("drift.detected", float(bool(report.get("drift_detected", False))))
        details = report.get("features", report.get("zone_horizon", {}))
        for name, item in details.items():
            if isinstance(item, dict) and "psi" in item:
                mlflow.log_metric(f"psi.{name}", float(item["psi"]))
        with tempfile.TemporaryDirectory(prefix="tetouan-monitor-") as temp_dir:
            json_path = Path(temp_dir) / "monitoring_report.json"
            md_path = Path(temp_dir) / "monitoring_report.md"
            json_path.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
            md_path.write_text("# Tetouan monitoring report\n\n" + "```json\n" + json.dumps(report, indent=2, allow_nan=False) + "\n```\n", encoding="utf-8")
            mlflow.log_artifact(str(json_path), artifact_path="monitoring")
            mlflow.log_artifact(str(md_path), artifact_path="monitoring")
        return run.info.run_id


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    repo_root = Path(__file__).resolve().parents[2]
    parser.add_argument("--artifact-dir", type=Path, default=repo_root / "runtime-artifacts/data/processed/tetouan")
    parser.add_argument("--current-features", required=True, type=Path, help="NPY incoming batch [N,8]; never copied into report")
    parser.add_argument("--reference-predictions", type=Path, help="NPY validation prediction reference [N,6,3]")
    parser.add_argument("--current-predictions", type=Path, help="NPY incoming predictions [N,6,3]")
    parser.add_argument("--current-actuals", type=Path, help="Optional aligned observed ground truth [N,6,3]")
    parser.add_argument("--validation-metrics", type=Path, help="Optional JSON with validation per_zone MAE/RMSE")
    parser.add_argument("--model-version", default="unversioned")
    parser.add_argument("--experiment-id", default="tetouan_t4_20261101_v1")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--alert-history-dir", type=Path, help="Optional directory of prior aggregate monitor JSON reports")
    parser.add_argument("--consecutive-alerts", type=int, default=3)
    parser.add_argument("--log-mlflow", action="store_true")
    args = parser.parse_args(argv)
    metadata_path, series_path = args.artifact_dir / "metadata.json", args.artifact_dir / "series.npz"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    train_end = int(metadata["split"]["partitions"]["train"]["end_index_exclusive"])
    with np.load(series_path, allow_pickle=False) as stored:
        training_reference = np.asarray(stored["features"][:train_end], dtype=np.float64)
    current_features = np.load(args.current_features, allow_pickle=False)
    report = feature_drift(training_reference, current_features)
    fingerprint = hashlib.sha256((metadata_path.read_bytes() + series_path.read_bytes())).hexdigest()
    report["reference_fingerprint"] = fingerprint
    report["model_version"] = args.model_version
    report["experiment_id"] = args.experiment_id
    report["reference_partition"] = {"name": "train", "start_index": 0, "end_index_exclusive": train_end}
    report["prediction_drift"] = None
    report["observed_performance"] = None
    if args.reference_predictions or args.current_predictions or args.current_actuals:
        if not (args.reference_predictions and args.current_predictions):
            parser.error("reference and current prediction arrays must be provided together")
        current_predictions = np.load(args.current_predictions, allow_pickle=False)
        report["prediction_drift"] = prediction_drift(np.load(args.reference_predictions, allow_pickle=False), current_predictions, model_version=args.model_version, experiment_id=args.experiment_id)
        if args.current_actuals:
            validation_metrics = None
            if args.validation_metrics:
                validation_metrics = json.loads(args.validation_metrics.read_text(encoding="utf-8"))
            report["observed_performance"] = performance_report(current_predictions, np.load(args.current_actuals, allow_pickle=False), validation_metrics)
    any_distribution_drift = bool(report["drift_detected"] or (report["prediction_drift"] or {}).get("drift_detected", False))
    report["any_distribution_drift"] = any_distribution_drift
    prior_reports: list[dict[str, Any]] = []
    if args.alert_history_dir:
        if args.consecutive_alerts <= 0:
            parser.error("--consecutive-alerts must be positive")
        if not args.alert_history_dir.is_dir():
            parser.error("--alert-history-dir must refer to an existing directory")
        for history_path in sorted(args.alert_history_dir.glob("*.json")):
            try:
                prior_reports.append(json.loads(history_path.read_text(encoding="utf-8")))
            except (OSError, json.JSONDecodeError) as exc:
                parser.error(f"cannot read prior alert report {history_path.name}: {exc}")
    prior_alerts = [bool(item.get("any_distribution_drift", item.get("drift_detected", False))) for item in prior_reports]
    report["repeated_alert"] = repeated_alert(
        [{"drift_detected": value} for value in prior_alerts + [any_distribution_drift]],
        consecutive=args.consecutive_alerts,
    )
    report["alert_policy"] = {
        "consecutive_reports_required": args.consecutive_alerts,
        "prior_reports_considered": len(prior_reports),
        "status": "repeated alert only when the current report and preceding reports meet the configured consecutive threshold",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
    if args.log_mlflow:
        print("MLflow monitoring run:", log_monitoring_report(report))
    print(args.output)


if __name__ == "__main__":
    main()
