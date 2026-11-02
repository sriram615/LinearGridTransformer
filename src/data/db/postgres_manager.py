"""PostgreSQL / TimescaleDB Manager for Multi-Zone Grid Telemetry."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

import pandas as pd
import psycopg
from psycopg import sql

logger = logging.getLogger("postgres_manager")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

DEFAULT_PG_URL = os.environ.get(
    "GRID_DB_URL", "postgresql://postgres:postgres@localhost:5433/grid_forecasting"
)
SCHEMA_FILE = Path(__file__).resolve().parent / "schema.sql"


class PostgresManager:
    """Manages database connection, schema migration, and telemetry batch loading."""

    def __init__(self, connection_url: str = DEFAULT_PG_URL) -> None:
        self.url = connection_url

    def init_db(self, schema_path: Path | str = SCHEMA_FILE) -> None:
        """Run DDL migration script to set up tables and hypertable."""
        schema_path = Path(schema_path)
        if not schema_path.is_file():
            raise FileNotFoundError(f"Schema file not found at {schema_path}")

        ddl = schema_path.read_text(encoding="utf-8")
        with psycopg.connect(self.url, autocommit=True) as conn:
            with conn.cursor() as cur:
                cur.execute(ddl)
        logger.info("Database schema initialized successfully.")

    def insert_telemetry_batch(self, df: pd.DataFrame) -> int:
        """Fast batch copy of telemetry records into PostgreSQL.
        
        Expected DataFrame columns:
        - time (datetime)
        - zone_id (str)
        - load_mw (float)
        - temperature_c (float)
        - relative_humidity_pct (float)
        - wind_speed_ms (float)
        - diffuse_solar_wm2 (float)
        - direct_solar_wm2 (float)
        """
        required_cols = [
            "time", "zone_id", "load_mw", "temperature_c",
            "relative_humidity_pct", "wind_speed_ms",
            "diffuse_solar_wm2", "direct_solar_wm2"
        ]
        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            raise ValueError(f"DataFrame is missing required columns: {missing}")

        data = df[required_cols].copy()
        # Convert timestamp to ISO string
        data["time"] = pd.to_datetime(data["time"]).dt.strftime("%Y-%m-%d %H:%M:%S%z")

        query = sql.SQL(
            """
            INSERT INTO grid_telemetry (
                time, zone_id, load_mw, temperature_c,
                relative_humidity_pct, wind_speed_ms,
                diffuse_solar_wm2, direct_solar_wm2
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (time, zone_id) DO UPDATE SET
                load_mw = EXCLUDED.load_mw,
                temperature_c = EXCLUDED.temperature_c,
                relative_humidity_pct = EXCLUDED.relative_humidity_pct,
                wind_speed_ms = EXCLUDED.wind_speed_ms,
                diffuse_solar_wm2 = EXCLUDED.diffuse_solar_wm2,
                direct_solar_wm2 = EXCLUDED.direct_solar_wm2;
            """
        )

        records = [tuple(row) for row in data.to_numpy()]
        with psycopg.connect(self.url) as conn:
            with conn.cursor() as cur:
                cur.executemany(query, records)
            conn.commit()

        logger.info(f"Inserted/updated {len(records)} telemetry records.")
        return len(records)

    def fetch_window(
        self, start_time: str, end_time: str, zone_ids: list[str] | None = None
    ) -> pd.DataFrame:
        """Fetch sliding-window telemetry for given time range."""
        query = """
            SELECT time, zone_id, load_mw, temperature_c, relative_humidity_pct,
                   wind_speed_ms, diffuse_solar_wm2, direct_solar_wm2
            FROM grid_telemetry
            WHERE time >= %s AND time <= %s
        """
        params: list[Any] = [start_time, end_time]
        if zone_ids:
            query += " AND zone_id = ANY(%s)"
            params.append(zone_ids)
        query += " ORDER BY time ASC, zone_id ASC;"

        with psycopg.connect(self.url) as conn:
            df = pd.read_sql_query(query, conn, params=params)
        return df

    def export_to_parquet(self, output_dir: Path | str, year: int) -> Path:
        """Export one calendar year of telemetry to Apache Parquet for PyTorch training."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        out_file = output_dir / f"grid_telemetry_{year}.parquet"

        start_time = f"{year}-01-01 00:00:00+00"
        end_time = f"{year}-12-31 23:59:59+00"

        logger.info(f"Querying telemetry for year {year}...")
        df = self.fetch_window(start_time, end_time)
        if df.empty:
            raise ValueError(f"No telemetry records found for year {year}")

        df.to_parquet(out_file, index=False, engine="pyarrow")
        logger.info(f"Saved {len(df)} rows to {out_file} ({out_file.stat().st_size / 1e6:.2f} MB)")
        return out_file

    def run_preprocessing(self, script_path: Path | str | None = None) -> None:
        """Execute the SQL preprocessing pipeline (split bounds, wide view, train stats)."""
        if script_path is None:
            script_path = Path(__file__).resolve().parent / "preprocessing.sql"
        script_path = Path(script_path)
        if not script_path.is_file():
            raise FileNotFoundError(f"Preprocessing SQL script not found: {script_path}")

        sql_code = script_path.read_text(encoding="utf-8")
        with psycopg.connect(self.url, autocommit=True) as conn:
            with conn.cursor() as cur:
                cur.execute(sql_code)
        logger.info("SQL preprocessing pipeline executed successfully.")

    def get_split_bounds(self) -> dict[str, dict[str, Any]]:
        """Return the recorded split bounds from grid_split_bounds table."""
        with psycopg.connect(self.url) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT split_name, start_index, end_index_exclusive, row_count,
                           start_time, end_time_exclusive
                    FROM grid_split_bounds
                    ORDER BY start_index ASC;
                    """
                )
                rows = cur.fetchall()
        bounds = {}
        for row in rows:
            bounds[row[0]] = {
                "start_index": int(row[1]),
                "end_index_exclusive": int(row[2]),
                "row_count": int(row[3]),
                "first_timestamp": row[4].strftime("%Y-%m-%d %H:%M:%S"),
                "last_timestamp": (row[5] - pd.Timedelta(minutes=10)).strftime("%Y-%m-%d %H:%M:%S"),
            }
        return bounds

    def get_train_stats(self) -> dict[str, list[float]]:
        """Return train-only population mean and std for features and targets."""
        with psycopg.connect(self.url) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT
                        zone_1_mean, zone_2_mean, zone_3_mean,
                        temp_mean, humidity_mean, wind_mean, diffuse_mean, direct_mean,
                        zone_1_std, zone_2_std, zone_3_std,
                        temp_std, humidity_std, wind_std, diffuse_std, direct_std
                    FROM grid_train_stats
                    WHERE stat_id = 1;
                    """
                )
                row = cur.fetchone()
        if not row:
            raise RuntimeError("grid_train_stats table is empty; run run_preprocessing() first.")

        feature_mean = [float(x) for x in row[:8]]
        feature_std = [float(x) for x in row[8:16]]
        target_mean = [float(x) for x in row[:3]]
        target_std = [float(x) for x in row[8:11]]

        return {
            "feature_mean": feature_mean,
            "feature_std": feature_std,
            "target_mean": target_mean,
            "target_std": target_std,
        }

    def fetch_wide_telemetry(self) -> pd.DataFrame:
        """Fetch all wide rows (time + 8 channels) from grid_telemetry_wide materialized view."""
        query = """
            SELECT time,
                   zone_1_power_consumption, zone_2_power_consumption, zone_3_power_consumption,
                   temperature, humidity, wind_speed, general_diffuse_flows, diffuse_flows
            FROM grid_telemetry_wide
            ORDER BY time ASC;
        """
        with psycopg.connect(self.url) as conn:
            with conn.cursor() as cur:
                cur.execute(query)
                rows = cur.fetchall()
                cols = [desc[0] for desc in cur.description]
        return pd.DataFrame(rows, columns=cols)

    def fetch_latest_telemetry(self, limit: int = 144) -> pd.DataFrame:
        """Fetch the latest `limit` rows (time + 8 channels) from grid_telemetry_wide."""
        query = """
            SELECT time,
                   zone_1_power_consumption, zone_2_power_consumption, zone_3_power_consumption,
                   temperature, humidity, wind_speed, general_diffuse_flows, diffuse_flows
            FROM (
                SELECT time,
                       zone_1_power_consumption, zone_2_power_consumption, zone_3_power_consumption,
                       temperature, humidity, wind_speed, general_diffuse_flows, diffuse_flows
                FROM grid_telemetry_wide
                ORDER BY time DESC
                LIMIT %s
            ) sub
            ORDER BY time ASC;
        """
        with psycopg.connect(self.url) as conn:
            with conn.cursor() as cur:
                cur.execute(query, (limit,))
                rows = cur.fetchall()
                cols = [desc[0] for desc in cur.description]
        return pd.DataFrame(rows, columns=cols)



