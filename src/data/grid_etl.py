"""Multi-Zone Grid Ingestion & ETL Pipeline.

Downloads open transmission load data and matching Open-Meteo weather,
formats into the exact canonical 8-channel contract expected by
LinearGridTransformer, and optionally ingests into PostgreSQL / TimescaleDB.
"""

from __future__ import annotations

import argparse
import io
import logging
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import requests

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("grid_etl")

# Canonical 8-channel feature order matching Tetouan dataset contract
CANONICAL_COLUMNS = [
    "DateTime",
    "Temperature",
    "Humidity",
    "Wind Speed",
    "general diffuse flows",
    "diffuse flows",
    "Zone 1 Power Consumption",
    "Zone 2  Power Consumption",
    "Zone 3  Power Consumption",
]

DEFAULT_ZONES = {
    "N.Y.C.": "Zone 1 Power Consumption",
    "LONGIL": "Zone 2  Power Consumption",   # Preserves exact doubled-space source mapping
    "HUD VL": "Zone 3  Power Consumption",   # Preserves exact doubled-space source mapping
}


def fetch_nyiso_month(year: int, month: int) -> pd.DataFrame:
    """Download and extract one month of NYISO 5-minute zonal load."""
    url = f"http://mis.nyiso.com/public/csv/pal/{year:04d}{month:02d}01pal_csv.zip"
    logger.info(f"Downloading NYISO archive: {url}")
    res = requests.get(url, timeout=30)
    if res.status_code != 200:
        raise RuntimeError(f"Failed to download NYISO archive for {year}-{month:02d}: HTTP {res.status_code}")

    with zipfile.ZipFile(io.BytesIO(res.content)) as z:
        dfs = []
        for name in z.namelist():
            if name.endswith(".csv"):
                dfs.append(pd.read_csv(z.open(name)))
        raw = pd.concat(dfs, ignore_index=True)

    # Filter target zones
    filtered = raw[raw["Name"].isin(DEFAULT_ZONES.keys())].copy()
    filtered["timestamp"] = pd.to_datetime(filtered["Time Stamp"], format="%m/%d/%Y %H:%M:%S", utc=True)
    
    # Pivot so each zone becomes a column
    pivoted = filtered.pivot_table(index="timestamp", columns="Name", values="Load", aggfunc="mean")
    return pivoted


def fetch_open_meteo_weather(start_date: str, end_date: str, lat: float = 40.71, lon: float = -74.00) -> pd.DataFrame:
    """Fetch hourly weather reanalysis from Open-Meteo for grid zone centroid."""
    url = (
        "https://archive-api.open-meteo.com/v1/archive?"
        f"latitude={lat}&longitude={lon}&start_date={start_date}&end_date={end_date}&"
        "hourly=temperature_2m,relative_humidity_2m,wind_speed_10m,diffuse_radiation,direct_normal_irradiance"
    )
    logger.info(f"Downloading Open-Meteo weather: {start_date} to {end_date}")
    res = requests.get(url, timeout=30).json()
    if "hourly" not in res:
        raise RuntimeError(f"Open-Meteo error: {res.get('reason', 'Unknown')}")

    df_weather = pd.DataFrame(res["hourly"])
    df_weather["timestamp"] = pd.to_datetime(df_weather["time"], utc=True)
    df_weather = df_weather.set_index("timestamp").drop(columns=["time"])
    return df_weather


def run_etl(
    year: int,
    output_csv: Path | str,
    cadence: str = "10min",
    load_to_postgres: bool = False,
    pg_url: str | None = None,
) -> pd.DataFrame:
    """Run full ETL: download 1 year, align load and weather, validate and save."""
    output_path = Path(output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # 1. Fetch all 12 months of load
    month_dfs = []
    for month in range(1, 13):
        try:
            m_df = fetch_nyiso_month(year, month)
            month_dfs.append(m_df)
        except Exception as e:
            logger.warning(f"Could not fetch month {year}-{month:02d}: {e}")

    if not month_dfs:
        raise RuntimeError(f"No load records retrieved for year {year}")

    df_load_raw = pd.concat(month_dfs).sort_index()
    # Remove duplicates
    df_load_raw = df_load_raw[~df_load_raw.index.duplicated(keep="first")]

    # 2. Resample load to target cadence (10min or 1h)
    df_load = df_load_raw.resample(cadence).mean().interpolate(method="time")

    # 3. Fetch matching weather
    start_str = df_load.index.min().strftime("%Y-%m-%d")
    end_str = df_load.index.max().strftime("%Y-%m-%d")
    df_weather_hourly = fetch_open_meteo_weather(start_str, end_str)

    # Resample weather to target cadence (interpolating hourly to 10min)
    df_weather = df_weather_hourly.resample(cadence).interpolate(method="time")

    # 4. Merge load and weather
    df_merged = df_load.join(df_weather, how="inner").dropna()

    # 5. Build canonical DataFrame
    df_canonical = pd.DataFrame({
        "DateTime": df_merged.index.strftime("%m/%d/%Y %H:%M"),
        "Temperature": df_merged["temperature_2m"],
        "Humidity": df_merged["relative_humidity_2m"],
        "Wind Speed": df_merged["wind_speed_10m"],
        "general diffuse flows": df_merged["diffuse_radiation"],
        "diffuse flows": df_merged["direct_normal_irradiance"],
        "Zone 1 Power Consumption": df_merged["N.Y.C."],
        "Zone 2  Power Consumption": df_merged["LONGIL"],
        "Zone 3  Power Consumption": df_merged["HUD VL"],
    })

    # Validate finite values
    for col in df_canonical.columns:
        if col != "DateTime":
            assert np.isfinite(df_canonical[col]).all(), f"Non-finite values found in {col}"

    # Save to disk
    df_canonical.to_csv(output_path, index=False)
    logger.info(f"Saved canonical dataset: {output_path} ({len(df_canonical)} rows, {len(df_canonical.columns)} columns)")

    # 6. Optional: Load into PostgreSQL
    if load_to_postgres:
        from src.data.db.postgres_manager import PostgresManager
        pm = PostgresManager(connection_url=pg_url or "postgresql://postgres:postgres@localhost:5433/grid_forecasting")
        try:
            pm.init_db()
            # Prepare rows in normalized format for DB
            records = []
            for zone_key, zone_id in [("N.Y.C.", "ZONE_1"), ("LONGIL", "ZONE_2"), ("HUD VL", "ZONE_3")]:
                sub = pd.DataFrame({
                    "time": df_merged.index,
                    "zone_id": zone_id,
                    "load_mw": df_merged[zone_key],
                    "temperature_c": df_merged["temperature_2m"],
                    "relative_humidity_pct": df_merged["relative_humidity_2m"],
                    "wind_speed_ms": df_merged["wind_speed_10m"],
                    "diffuse_solar_wm2": df_merged["diffuse_radiation"],
                    "direct_solar_wm2": df_merged["direct_normal_irradiance"],
                })
                records.append(sub)
            all_records = pd.concat(records)
            pm.insert_telemetry_batch(all_records)
            logger.info("Successfully ingested into PostgreSQL!")
        except Exception as e:
            logger.error(f"PostgreSQL ingestion skipped/failed: {e}")

    return df_canonical


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Multi-Zone Grid Ingestion & ETL")
    parser.add_argument("--year", type=int, default=2023, help="Calendar year to ingest")
    parser.add_argument("--output", type=str, default="data/raw/grid/grid_power_consumption.csv", help="Target CSV output path")
    parser.add_argument("--cadence", type=str, default="10min", help="Resampling cadence (e.g. 10min, 1h)")
    parser.add_argument("--load-db", action="store_true", help="Ingest into PostgreSQL")
    args = parser.parse_args()

    run_etl(args.year, args.output, cadence=args.cadence, load_to_postgres=args.load_db)
