"""Export PostgreSQL-Preprocessed Grid Telemetry to PyTorch Artifacts.

Reads normalized & raw multi-zone telemetry from TimescaleDB/PostgreSQL,
formats the arrays into canonical ML contracts, and exports:
- `data/processed/grid/series.npz`
- `data/processed/grid/metadata.json`
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import pandas as pd

from src.data.db.postgres_manager import DEFAULT_PG_URL, PostgresManager
from src.data.tetouan_preprocess import (
    FEATURE_NAMES,
    HORIZON,
    LOOKBACK,
    TARGET_NAMES,
)


logger = logging.getLogger("export_processed_grid")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_DIR = REPO_ROOT / "data/processed/grid"


def export_grid_dataset(
    db_url: str = DEFAULT_PG_URL,
    output_dir: Path | str = DEFAULT_OUTPUT_DIR,
    force: bool = False,
) -> dict[str, Any]:
    """Export preprocessed grid data from PostgreSQL to ML disk artifacts."""
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    arrays_path = output_dir / "series.npz"
    metadata_path = output_dir / "metadata.json"

    if not force and (arrays_path.exists() or metadata_path.exists()):
        raise FileExistsError(
            f"Artifacts already exist at {output_dir}. Use force=True to overwrite."
        )

    logger.info("Connecting to PostgreSQL and verifying preprocessing...")
    pm = PostgresManager(connection_url=db_url)
    pm.run_preprocessing()

    split_bounds = pm.get_split_bounds()
    stats = pm.get_train_stats()
    wide_df = pm.fetch_wide_telemetry()

    n_rows = len(wide_df)
    logger.info(f"Retrieved {n_rows} wide rows from PostgreSQL.")

    # 1. Format timestamps, features, and targets
    timestamps = pd.to_datetime(wide_df["time"]).to_numpy(dtype="datetime64[ns]")
    feature_cols = list(FEATURE_NAMES)
    target_cols = list(TARGET_NAMES)

    features = wide_df[feature_cols].to_numpy(dtype=np.float64, copy=True)
    targets = wide_df[target_cols].to_numpy(dtype=np.float64, copy=True)

    # 2. Save compressed NPZ
    logger.info(f"Saving arrays to {arrays_path}...")
    np.savez_compressed(
        arrays_path,
        timestamps=timestamps,
        features=features,
        targets=targets,
    )

    # 3. Compute SHA256 of the generated arrays
    hasher = hashlib.sha256()
    with arrays_path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            hasher.update(chunk)
    arrays_sha256 = hasher.hexdigest()

    # 4. Construct metadata
    metadata: dict[str, Any] = {
        "schema_version": 1,
        "source": {
            "name": "NYISO Regional Multi-Zone Power Consumption + Open-Meteo Reanalysis",
            "iso": "New York Independent System Operator (NYISO)",
            "zones": {
                "ZONE_1": "New York City (N.Y.C.)",
                "ZONE_2": "Long Island (LONGIL)",
                "ZONE_3": "Hudson Valley (HUD VL)",
            },
            "source_urls": [
                "http://mis.nyiso.com/public/csv/pal/",
                "https://archive-api.open-meteo.com/v1/archive",
            ],
            "year": 2023,
            "arrays_sha256": arrays_sha256,
        },
        "row_count": n_rows,
        "source_columns": [
            "time",
            "zone_1_power_consumption",
            "zone_2_power_consumption",
            "zone_3_power_consumption",
            "temperature",
            "humidity",
            "wind_speed",
            "general_diffuse_flows",
            "diffuse_flows",
        ],
        "features": {
            "names": list(FEATURE_NAMES),
            "source_columns": list(FEATURE_NAMES),
            "count": len(FEATURE_NAMES),
        },
        "targets": {
            "names": list(TARGET_NAMES),
            "source_columns": list(TARGET_NAMES),
            "count": len(TARGET_NAMES),
        },
        "timestamps": {
            "source_column": "time",
            "format": "%Y-%m-%d %H:%M:%S%z",
            "timezone": "UTC",
            "timezone_status": "UTC timestamp with timezone",
            "cadence_minutes": 10,
            "first_timestamp": wide_df["time"].iloc[0].strftime("%Y-%m-%d %H:%M:%S"),
            "last_timestamp": wide_df["time"].iloc[-1].strftime("%Y-%m-%d %H:%M:%S"),
            "handling": "Strictly contiguous 10-minute intervals verified in TimescaleDB.",
        },
        "split": {
            "method": "chronological row count; floor(0.70*N), floor(0.85*N)",
            "partitions": split_bounds,
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
            "fit_row_range_half_open": [
                split_bounds["train"]["start_index"],
                split_bounds["train"]["end_index_exclusive"],
            ],
            "std_ddof": 0,
            "feature_names": list(FEATURE_NAMES),
            "feature_mean": stats["feature_mean"],
            "feature_std": stats["feature_std"],
            "target_names": list(TARGET_NAMES),
            "target_mean": stats["target_mean"],
            "target_std": stats["target_std"],
        },
    }

    logger.info(f"Writing metadata to {metadata_path}...")
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    logger.info(f"Export complete. Artifacts written to {output_dir}")
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description="Export preprocessed grid data from PostgreSQL")
    parser.add_argument("--db-url", type=str, default=DEFAULT_PG_URL, help="PostgreSQL connection URL")
    parser.add_argument("--output-dir", type=str, default=str(DEFAULT_OUTPUT_DIR), help="Output directory")
    parser.add_argument("--force", action="store_true", help="Overwrite existing artifacts")
    args = parser.parse_args()

    export_grid_dataset(db_url=args.db_url, output_dir=args.output_dir, force=args.force)


if __name__ == "__main__":
    main()
