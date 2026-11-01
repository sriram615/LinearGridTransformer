"""Validation and artifact preparation for the Tetouan City dataset.

The raw CSV is read-only. Derived arrays and metadata are written to
``data/processed/tetouan/`` by default; no row is imputed, resampled, or
reordered.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SOURCE = REPO_ROOT / "data/raw/tetouan/Tetuan City power consumption.csv"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "data/processed/tetouan"
SOURCE_URL = "https://archive.ics.uci.edu/static/public/849/power%2Bconsumption%2Bof%2Btetouan%2Bcity.zip"
DATASET_URL = "https://archive.ics.uci.edu/dataset/849/power+consumption+of+tetouan+city"
DATASET_DOI = "10.24432/C5B034"
TIMESTAMP_FORMAT = "%m/%d/%Y %H:%M"
CADENCE = pd.Timedelta(minutes=10)
LOOKBACK = 144
HORIZON = 6

SOURCE_COLUMNS: tuple[str, ...] = (
    "DateTime",
    "Temperature",
    "Humidity",
    "Wind Speed",
    "general diffuse flows",
    "diffuse flows",
    "Zone 1 Power Consumption",
    "Zone 2  Power Consumption",
    "Zone 3  Power Consumption",
)

# Canonical names are stable machine identifiers; source names remain exact.
FEATURE_MAPPING: tuple[tuple[str, str], ...] = (
    ("zone_1_power_consumption", "Zone 1 Power Consumption"),
    ("zone_2_power_consumption", "Zone 2  Power Consumption"),
    ("zone_3_power_consumption", "Zone 3  Power Consumption"),
    ("temperature", "Temperature"),
    ("humidity", "Humidity"),
    ("wind_speed", "Wind Speed"),
    ("general_diffuse_flows", "general diffuse flows"),
    ("diffuse_flows", "diffuse flows"),
)
TARGET_MAPPING: tuple[tuple[str, str], ...] = FEATURE_MAPPING[:3]
FEATURE_NAMES: tuple[str, ...] = tuple(name for name, _ in FEATURE_MAPPING)
TARGET_NAMES: tuple[str, ...] = tuple(name for name, _ in TARGET_MAPPING)
FEATURE_SOURCE_COLUMNS: tuple[str, ...] = tuple(source for _, source in FEATURE_MAPPING)
TARGET_SOURCE_COLUMNS: tuple[str, ...] = tuple(source for _, source in TARGET_MAPPING)


class TetouanDataError(ValueError):
    """Raised when the source data violates the approved data contract."""


def load_and_validate_csv(source_path: str | Path = DEFAULT_SOURCE) -> tuple[pd.DataFrame, pd.DatetimeIndex]:
    """Load the exact source schema and validate values and 10-minute time order."""
    path = Path(source_path)
    if not path.is_file():
        raise FileNotFoundError(f"Tetouan source CSV not found: {path}")

    frame = pd.read_csv(path)
    actual = tuple(frame.columns)
    if actual != SOURCE_COLUMNS:
        raise TetouanDataError(
            "Source columns/order mismatch. "
            f"Expected {SOURCE_COLUMNS!r}; received {actual!r}."
        )
    if frame.empty:
        raise TetouanDataError("Source CSV contains no data rows.")
    if frame.isna().any().any():
        counts = frame.isna().sum()
        details = {str(k): int(v) for k, v in counts[counts > 0].items()}
        raise TetouanDataError(f"Missing values are not allowed: {details}")

    if frame.duplicated().any():
        raise TetouanDataError("Duplicate rows are not allowed.")

    try:
        timestamps = pd.DatetimeIndex(
            pd.to_datetime(frame["DateTime"], format=TIMESTAMP_FORMAT, errors="raise")
        )
    except (TypeError, ValueError) as exc:
        raise TetouanDataError(
            f"DateTime must parse with explicit format {TIMESTAMP_FORMAT!r}."
        ) from exc
    if timestamps.hasnans:
        raise TetouanDataError("DateTime contains an unparseable timestamp.")
    if timestamps.has_duplicates:
        raise TetouanDataError("Duplicate timestamps are not allowed.")
    if not timestamps.is_monotonic_increasing:
        raise TetouanDataError("Rows must already be in strictly chronological order.")
    if len(timestamps) > 1:
        deltas = timestamps[1:] - timestamps[:-1]
        bad = np.flatnonzero(deltas != CADENCE)
        if len(bad):
            i = int(bad[0])
            raise TetouanDataError(
                "Every adjacent timestamp must be exactly 10 minutes apart; "
                f"rows {i} and {i + 1} are {timestamps[i]} and {timestamps[i + 1]} "
                f"({deltas[i]} apart)."
            )
    numeric_columns = [column for column in SOURCE_COLUMNS if column != "DateTime"]
    for column in numeric_columns:
        try:
            values = pd.to_numeric(frame[column], errors="raise").to_numpy(dtype=np.float64)
        except (TypeError, ValueError) as exc:
            raise TetouanDataError(f"Column {column!r} must contain numeric values.") from exc
        if not np.isfinite(values).all():
            raise TetouanDataError(f"Column {column!r} contains non-finite values.")
        frame[column] = values

    return frame, timestamps


def calculate_split_indices(row_count: int) -> dict[str, dict[str, int]]:
    """Return half-open row bounds using floor(70%) and floor(85%)."""
    if row_count < LOOKBACK + HORIZON:
        raise TetouanDataError(
            f"At least {LOOKBACK + HORIZON} rows are required; received {row_count}."
        )
    train_end = int(row_count * 0.70)
    val_end = int(row_count * 0.85)
    if train_end < LOOKBACK + HORIZON:
        raise TetouanDataError(
            f"Training partition must contain at least {LOOKBACK + HORIZON} rows; "
            f"it would contain {train_end}."
        )
    bounds = {
        "train": {"start_index": 0, "end_index_exclusive": train_end},
        "validation": {"start_index": train_end, "end_index_exclusive": val_end},
        "test": {"start_index": val_end, "end_index_exclusive": row_count},
    }
    if any(v["end_index_exclusive"] <= v["start_index"] for v in bounds.values()):
        raise TetouanDataError("All three chronological partitions must contain rows.")
    if any(
        bounds[name]["end_index_exclusive"] - bounds[name]["start_index"] < HORIZON
        for name in bounds
    ):
        raise TetouanDataError("Each partition must contain at least one full target horizon.")
    return bounds


def fit_normalization(
    feature_values: np.ndarray,
    target_values: np.ndarray,
    train_end: int,
) -> dict[str, np.ndarray]:
    """Fit float64 population statistics using rows [0, train_end) only."""
    if feature_values.ndim != 2 or feature_values.shape[1] != len(FEATURE_NAMES):
        raise TetouanDataError(f"Features must have shape [N, {len(FEATURE_NAMES)}].")
    if target_values.ndim != 2 or target_values.shape[1] != len(TARGET_NAMES):
        raise TetouanDataError(f"Targets must have shape [N, {len(TARGET_NAMES)}].")
    if feature_values.shape[0] != target_values.shape[0]:
        raise TetouanDataError("Feature and target row counts must match.")
    if not 0 < train_end <= feature_values.shape[0]:
        raise TetouanDataError("train_end must select a non-empty training prefix.")

    train_features = np.asarray(feature_values[:train_end], dtype=np.float64)
    train_targets = np.asarray(target_values[:train_end], dtype=np.float64)
    result = {
        "feature_mean": train_features.mean(axis=0),
        "feature_std": train_features.std(axis=0, ddof=0),
        "target_mean": train_targets.mean(axis=0),
        "target_std": train_targets.std(axis=0, ddof=0),
    }
    for name, values in result.items():
        if not np.isfinite(values).all():
            raise TetouanDataError(f"Normalization statistic {name} is non-finite.")
        if name.endswith("_std") and np.any(values <= 0):
            raise TetouanDataError(f"Normalization statistic {name} must be positive.")
    return result


def inverse_normalize_targets(values: Any, mean: Any, std: Any) -> Any:
    """Invert target z-scores; supports NumPy arrays and PyTorch tensors."""
    return values * std + mean


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _partition_metadata(
    bounds: dict[str, dict[str, int]], timestamps: pd.DatetimeIndex
) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for name, pair in bounds.items():
        start, end = pair["start_index"], pair["end_index_exclusive"]
        result[name] = {
            **pair,
            "row_count": end - start,
            "first_timestamp": timestamps[start].isoformat(sep=" "),
            "last_timestamp": timestamps[end - 1].isoformat(sep=" "),
        }
    return result


def preprocess_tetouan(
    source_path: str | Path = DEFAULT_SOURCE,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
) -> dict[str, Any]:
    """Validate raw CSV and write ``series.npz`` and ``metadata.json``.

    Existing artifact paths are never overwritten. Raw CSV content is not
    normalized or changed; normalization statistics are recorded separately.
    """
    source = Path(source_path).resolve()
    destination = Path(output_dir).resolve()
    frame, timestamps = load_and_validate_csv(source)
    n_rows = len(frame)
    splits = calculate_split_indices(n_rows)
    train_end = splits["train"]["end_index_exclusive"]

    feature_values = frame.loc[:, FEATURE_SOURCE_COLUMNS].to_numpy(dtype=np.float64, copy=True)
    target_values = frame.loc[:, TARGET_SOURCE_COLUMNS].to_numpy(dtype=np.float64, copy=True)
    stats = fit_normalization(feature_values, target_values, train_end)
    timestamps_ns = timestamps.to_numpy(dtype="datetime64[ns]")

    destination.mkdir(parents=True, exist_ok=True)
    arrays_path = destination / "series.npz"
    metadata_path = destination / "metadata.json"
    for artifact in (arrays_path, metadata_path):
        if artifact.exists():
            raise FileExistsError(f"Refusing to overwrite existing artifact: {artifact}")

    metadata: dict[str, Any] = {
        "schema_version": 1,
        "source": {
            "file_name": source.name,
            "source_url": SOURCE_URL,
            "dataset_url": DATASET_URL,
            "doi": DATASET_DOI,
            "attribution": "Salam, A. & El Hibaoui, A. (2018), Power Consumption of Tetouan City, UCI Machine Learning Repository.",
            "license": "CC BY 4.0",
            "sha256": _sha256(source),
        },
        "row_count": n_rows,
        "source_columns": list(SOURCE_COLUMNS),
        "features": {
            "names": list(FEATURE_NAMES),
            "source_columns": list(FEATURE_SOURCE_COLUMNS),
            "count": len(FEATURE_NAMES),
        },
        "targets": {
            "names": list(TARGET_NAMES),
            "source_columns": list(TARGET_SOURCE_COLUMNS),
            "count": len(TARGET_NAMES),
        },
        "timestamps": {
            "source_column": "DateTime",
            "format": TIMESTAMP_FORMAT,
            "timezone": None,
            "timezone_status": "undocumented; timestamps are timezone-naive",
            "cadence_minutes": 10,
            "first_timestamp": timestamps[0].isoformat(sep=" "),
            "last_timestamp": timestamps[-1].isoformat(sep=" "),
            "missing_final_calendar_day": "2017-12-31",
            "handling": "No records appended or fabricated.",
        },
        "split": {
            "method": "chronological row count; floor(0.70*N), floor(0.85*N)",
            "partitions": _partition_metadata(splits, timestamps),
        },
        "windows": {
            "lookback": LOOKBACK,
            "horizon": HORIZON,
            "stride": 1,
            "input_contract": [LOOKBACK, len(FEATURE_NAMES)],
            "target_contract": [HORIZON, len(TARGET_NAMES)],
            "validation_test_context_may_precede_partition": True,
            "targets_must_remain_in_partition": True,
            "future_weather_features": False,
        },
        "normalization": {
            "method": "z-score: (value - mean) / std",
            "fit_partition": "train",
            "fit_row_range_half_open": [0, train_end],
            "std_ddof": 0,
            "feature_names": list(FEATURE_NAMES),
            "feature_mean": stats["feature_mean"].tolist(),
            "feature_std": stats["feature_std"].tolist(),
            "target_names": list(TARGET_NAMES),
            "target_mean": stats["target_mean"].tolist(),
            "target_std": stats["target_std"].tolist(),
            "inverse": "raw_target = normalized_target * target_std + target_mean",
        },
        "units": None,
        "units_status": "Not documented by the source metadata; preserved as unknown.",
        "arrays_file": "series.npz",
        "arrays": {
            "timestamps": "datetime64[ns], timezone-naive, length N",
            "features": "float64 raw values, shape [N, 8], canonical feature order",
            "targets": "float64 raw values, shape [N, 3], canonical target order",
        },
    }

    np.savez_compressed(
        arrays_path,
        timestamps=timestamps_ns,
        features=feature_values,
        targets=target_values,
    )
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return metadata


if __name__ == "__main__":
    result = preprocess_tetouan()
    print(f"Wrote {DEFAULT_OUTPUT_DIR / 'series.npz'}")
    print(f"Wrote {DEFAULT_OUTPUT_DIR / 'metadata.json'}")
    print(f"Rows: {result['row_count']}; features: {result['features']['count']}; targets: {result['targets']['count']}")
