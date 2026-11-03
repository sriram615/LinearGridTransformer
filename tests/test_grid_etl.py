"""Unit tests for the Multi-Zone Grid Ingestion & ETL pipeline and Data Quality Checks."""

from __future__ import annotations

import io
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd

from src.data.grid_etl import (
    CANONICAL_COLUMNS,
    DEFAULT_ZONES,
    DEFAULT_VALUE_RANGES,
    DataQualityReport,
    GridDataQualityError,
    check_duplicate_records,
    check_expected_zones,
    check_timestamp_intervals,
    check_value_integrity_and_types,
    fetch_nyiso_month,
    fetch_open_meteo_weather,
    generate_etl_quality_report,
    run_etl,
    validate_grid_dataset,
)


def _make_mock_nyiso_zip_bytes(zones: list[str] | None = None) -> bytes:
    """Helper to generate in-memory NYISO zip payload for mocking."""
    if zones is None:
        zones = ["N.Y.C.", "LONGIL", "HUD VL"]
    records = []
    for z in zones:
        for m in [0, 5, 10]:
            records.append({
                "Time Stamp": f"01/01/2023 00:{m:02d}:00",
                "Name": z,
                "Load": 5000.0 if z == "N.Y.C." else (2000.0 if z == "LONGIL" else 1000.0),
            })
    df = pd.DataFrame(records)
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("20230101pal.csv", df.to_csv(index=False))
    return buf.getvalue()


def _make_valid_grid_df(rows: int = 10) -> pd.DataFrame:
    """Helper to generate a strictly valid synthetic grid DataFrame."""
    dates = pd.date_range("2023-01-01 00:00", periods=rows, freq="10min")
    return pd.DataFrame({
        "DateTime": dates.strftime("%m/%d/%Y %H:%M"),
        "Temperature": np.linspace(10.0, 15.0, rows),
        "Humidity": np.linspace(50.0, 60.0, rows),
        "Wind Speed": np.linspace(5.0, 10.0, rows),
        "general diffuse flows": np.linspace(0.0, 100.0, rows),
        "diffuse flows": np.linspace(0.0, 200.0, rows),
        "Zone 1 Power Consumption": np.linspace(5000.0, 6000.0, rows),
        "Zone 2  Power Consumption": np.linspace(2000.0, 2500.0, rows),
        "Zone 3  Power Consumption": np.linspace(900.0, 1100.0, rows),
    })


class GridETLTests(unittest.TestCase):
    def test_canonical_schema_order_and_channel_counts(self) -> None:
        self.assertEqual(len(CANONICAL_COLUMNS), 9)
        self.assertEqual(CANONICAL_COLUMNS[0], "DateTime")
        self.assertEqual(CANONICAL_COLUMNS[1], "Temperature")
        self.assertEqual(CANONICAL_COLUMNS[2], "Humidity")
        self.assertEqual(CANONICAL_COLUMNS[3], "Wind Speed")
        self.assertEqual(CANONICAL_COLUMNS[4], "general diffuse flows")
        self.assertEqual(CANONICAL_COLUMNS[5], "diffuse flows")
        self.assertEqual(CANONICAL_COLUMNS[6], "Zone 1 Power Consumption")
        self.assertEqual(CANONICAL_COLUMNS[7], "Zone 2  Power Consumption")
        self.assertEqual(CANONICAL_COLUMNS[8], "Zone 3  Power Consumption")

    def test_valid_dataset_passes_audit(self) -> None:
        df = _make_valid_grid_df(20)
        report = validate_grid_dataset(df, cadence="10min", raise_on_error=True)
        self.assertTrue(report.passed)
        self.assertEqual(report.total_rows, 20)
        self.assertEqual(len(report.errors), 0)
        self.assertIn("PASSED", report.summary())

    def test_detect_missing_timestamps_and_intervals(self) -> None:
        # Create timestamps with a 30-minute jump (2 missing 10-minute intervals)
        t_part1 = pd.date_range("2023-01-01 00:00", periods=5, freq="10min")
        t_part2 = pd.date_range("2023-01-01 01:10", periods=5, freq="10min")
        timestamps = t_part1.append(t_part2)

        res = check_timestamp_intervals(timestamps, expected_cadence="10min")
        self.assertTrue(res["is_monotonic"])
        self.assertGreater(res["cadence_violations"], 0)
        self.assertEqual(res["missing_interval_count"], 2)  # 00:50 and 01:00 are missing
        self.assertEqual(len(res["missing_intervals"]), 1)
        self.assertEqual(res["missing_intervals"][0]["missing_steps"], 2)

        # Test non-monotonic timestamps
        reversed_ts = timestamps[::-1]
        res_rev = check_timestamp_intervals(reversed_ts, expected_cadence="10min")
        self.assertFalse(res_rev["is_monotonic"])

        # Test failure when validating dataset
        df = _make_valid_grid_df(10)
        df.loc[5, "DateTime"] = "01/01/2023 01:40"  # gap
        with self.assertRaises(GridDataQualityError) as ctx:
            validate_grid_dataset(df, cadence="10min", raise_on_error=True)
        self.assertIn("missing timestamp step", str(ctx.exception))

    def test_detect_duplicate_records(self) -> None:
        df = _make_valid_grid_df(10)
        # Duplicate timestamp
        df.loc[3, "DateTime"] = df.loc[2, "DateTime"]

        res = check_duplicate_records(df)
        self.assertEqual(res["duplicate_timestamps"], 1)

        # Test duplicate entire row
        df_row_dupe = _make_valid_grid_df(5)
        df_row_dupe = pd.concat([df_row_dupe, df_row_dupe.iloc[[0]]], ignore_index=True)
        res_row = check_duplicate_records(df_row_dupe)
        self.assertEqual(res_row["duplicate_rows"], 1)

        with self.assertRaises(GridDataQualityError) as ctx:
            validate_grid_dataset(df, raise_on_error=True)
        self.assertIn("duplicate", str(ctx.exception).lower())

    def test_detect_missing_values_and_type_errors(self) -> None:
        # Missing NaN value
        df_nan = _make_valid_grid_df(10)
        df_nan.loc[2, "Temperature"] = np.nan
        val_res = check_value_integrity_and_types(df_nan)
        self.assertEqual(val_res["missing_counts"]["Temperature"], 1)

        with self.assertRaises(GridDataQualityError) as ctx:
            validate_grid_dataset(df_nan, raise_on_error=True)
        self.assertIn("Missing values found", str(ctx.exception))

        # Non-numeric type error
        df_type = _make_valid_grid_df(10)
        df_type["Humidity"] = df_type["Humidity"].astype(str)
        df_type.loc[4, "Humidity"] = "not_a_number"
        val_type = check_value_integrity_and_types(df_type)
        self.assertEqual(val_type["type_errors"]["Humidity"], 1)

        with self.assertRaises(GridDataQualityError) as ctx:
            validate_grid_dataset(df_type, raise_on_error=True)
        self.assertIn("Data type parsing errors", str(ctx.exception))

    def test_detect_invalid_values_and_physical_bounds(self) -> None:
        # Infinite values
        df_inf = _make_valid_grid_df(10)
        df_inf.loc[1, "Wind Speed"] = np.inf
        val_inf = check_value_integrity_and_types(df_inf)
        self.assertEqual(val_inf["non_finite_counts"]["Wind Speed"], 1)

        with self.assertRaises(GridDataQualityError):
            validate_grid_dataset(df_inf, raise_on_error=True)

        # Physical range violation: Negative power consumption
        df_neg_load = _make_valid_grid_df(10)
        df_neg_load.loc[0, "Zone 1 Power Consumption"] = -50.0
        val_range = check_value_integrity_and_types(df_neg_load)
        self.assertIn("Zone 1 Power Consumption", val_range["range_violations"])

        with self.assertRaises(GridDataQualityError) as ctx:
            validate_grid_dataset(df_neg_load, raise_on_error=True)
        self.assertIn("Physical range violations detected", str(ctx.exception))

        # Physical range violation: Humidity > 100%
        df_humidity = _make_valid_grid_df(10)
        df_humidity.loc[0, "Humidity"] = 105.0
        val_hum = check_value_integrity_and_types(df_humidity)
        self.assertIn("Humidity", val_hum["range_violations"])

    def test_check_expected_zones_presence(self) -> None:
        # Valid canonical format
        df = _make_valid_grid_df(5)
        res_canon = check_expected_zones(df)
        self.assertTrue(res_canon["all_present"])
        self.assertEqual(len(res_canon["missing_zones"]), 0)

        # Missing a zone column
        df_missing_zone = df.drop(columns=["Zone 3  Power Consumption"])
        res_missing = check_expected_zones(df_missing_zone)
        self.assertFalse(res_missing["all_present"])
        self.assertIn("HUD VL", res_missing["missing_zones"])

        with self.assertRaises(GridDataQualityError) as ctx:
            validate_grid_dataset(df_missing_zone, raise_on_error=True)
        self.assertIn("Missing expected grid zone", str(ctx.exception))

        # Raw NYISO unpivoted format check
        raw_df = pd.DataFrame({
            "Name": ["N.Y.C.", "LONGIL", "N.Y.C."],
            "Load": [5000.0, 2000.0, 4800.0],
            "Time Stamp": ["01/01/2023 00:00:00", "01/01/2023 00:00:00", "01/01/2023 00:05:00"],
        })
        raw_check = check_expected_zones(raw_df)
        self.assertFalse(raw_check["all_present"])
        self.assertIn("HUD VL", raw_check["missing_zones"])

    def test_downloaded_csv_integrity(self) -> None:
        csv_file = Path("data/raw/grid/grid_power_consumption.csv")
        if not csv_file.is_file():
            self.skipTest("Downloaded grid CSV not found in workspace.")

        df = pd.read_csv(csv_file)
        self.assertEqual(df.columns.tolist(), CANONICAL_COLUMNS)
        self.assertGreater(len(df), 50000)

        # Run complete data-quality audit on official downloaded CSV
        report = validate_grid_dataset(df, cadence="10min", raise_on_error=True)
        self.assertTrue(report.passed)
        self.assertEqual(report.timestamp_continuity["cadence_violations"], 0)
        self.assertEqual(report.timestamp_continuity["missing_interval_count"], 0)
        self.assertEqual(report.duplicate_records["duplicate_timestamps"], 0)
        self.assertEqual(report.duplicate_records["duplicate_rows"], 0)
        self.assertEqual(len(report.missing_and_invalid_values["missing_counts"]), 0)
        self.assertEqual(len(report.missing_and_invalid_values["non_finite_counts"]), 0)

    def test_generate_etl_quality_report(self) -> None:
        csv_file = Path("data/raw/grid/grid_power_consumption.csv")
        if not csv_file.is_file():
            self.skipTest("Downloaded grid CSV not found in workspace.")

        with tempfile.TemporaryDirectory() as tmpdir:
            out_json = Path(tmpdir) / "quality_report.json"
            report_data = generate_etl_quality_report(
                source_csv=csv_file,
                output_path=out_json,
                cadence="10min",
            )
            self.assertTrue(out_json.is_file())
            self.assertEqual(report_data["report_metadata"]["overall_status"], "PASSED")
            self.assertEqual(report_data["row_counts"]["total_rows"], 52555)
            self.assertEqual(len(report_data["failed_checks"]), 0)
            self.assertEqual(report_data["source_coverage"]["physical_units"], "Megawatts (MW)")
            self.assertIn("ZONE_1", report_data["source_coverage"]["zones"])
            self.assertTrue(report_data["ingestion_status"]["raw_csv_available"])

    # -------------------------------------------------------------------------
    # Mocking External API Responses & Pipeline Ingestion Tests
    # -------------------------------------------------------------------------

    def test_mock_fetch_nyiso_month_success(self) -> None:
        zip_bytes = _make_mock_nyiso_zip_bytes()
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.content = zip_bytes

        with patch("requests.get", return_value=mock_response):
            pivoted = fetch_nyiso_month(2023, 1)
            self.assertIsInstance(pivoted.index, pd.DatetimeIndex)
            self.assertEqual(sorted(pivoted.columns.tolist()), ["HUD VL", "LONGIL", "N.Y.C."])
            self.assertEqual(len(pivoted), 3)

    def test_mock_fetch_nyiso_month_http_error(self) -> None:
        mock_response = MagicMock()
        mock_response.status_code = 404
        with patch("requests.get", return_value=mock_response):
            with self.assertRaises(RuntimeError) as ctx:
                fetch_nyiso_month(2023, 1)
            self.assertIn("HTTP 404", str(ctx.exception))

    def test_mock_fetch_nyiso_month_missing_zone_failure(self) -> None:
        # Missing HUD VL
        zip_bytes = _make_mock_nyiso_zip_bytes(zones=["N.Y.C.", "LONGIL"])
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.content = zip_bytes

        with patch("requests.get", return_value=mock_response):
            with self.assertRaises(GridDataQualityError) as ctx:
                fetch_nyiso_month(2023, 1)
            self.assertIn("missing expected zone", str(ctx.exception).lower())

    def test_mock_fetch_open_meteo_weather_success(self) -> None:
        times = ["2023-01-01T00:00", "2023-01-01T01:00"]
        mock_json = {
            "hourly": {
                "time": times,
                "temperature_2m": [8.5, 9.0],
                "relative_humidity_2m": [80.0, 78.0],
                "wind_speed_10m": [12.0, 11.5],
                "diffuse_radiation": [0.0, 0.0],
                "direct_normal_irradiance": [0.0, 0.0],
            }
        }
        mock_response = MagicMock()
        mock_response.json.return_value = mock_json

        with patch("requests.get", return_value=mock_response):
            df_weather = fetch_open_meteo_weather("2023-01-01", "2023-01-02")
            self.assertIsInstance(df_weather.index, pd.DatetimeIndex)
            self.assertEqual(len(df_weather), 2)
            self.assertIn("temperature_2m", df_weather.columns)

    def test_mock_fetch_open_meteo_weather_api_error(self) -> None:
        mock_response = MagicMock()
        mock_response.json.return_value = {"error": True, "reason": "Rate limit exceeded"}

        with patch("requests.get", return_value=mock_response):
            with self.assertRaises(RuntimeError) as ctx:
                fetch_open_meteo_weather("2023-01-01", "2023-01-02")
            self.assertIn("Rate limit exceeded", str(ctx.exception))

    def test_mock_successful_run_etl_with_postgres_ingestion(self) -> None:
        def mock_nyiso(year: int, month: int) -> pd.DataFrame:
            if month > 1:
                raise ValueError("1 month only for fast unit test")
            idx = pd.date_range("2023-01-01 00:00", periods=6, freq="10min", tz="UTC")
            return pd.DataFrame({
                "N.Y.C.": [5000.0 + i for i in range(6)],
                "LONGIL": [2000.0 + i for i in range(6)],
                "HUD VL": [1000.0 + i for i in range(6)],
            }, index=idx)

        def mock_weather(start_str: str, end_str: str) -> pd.DataFrame:
            idx = pd.date_range("2023-01-01 00:00", periods=2, freq="1h", tz="UTC")
            return pd.DataFrame({
                "temperature_2m": [10.0, 12.0],
                "relative_humidity_2m": [50.0, 55.0],
                "wind_speed_10m": [5.0, 6.0],
                "diffuse_radiation": [0.0, 10.0],
                "direct_normal_irradiance": [0.0, 20.0],
            }, index=idx)

        with tempfile.TemporaryDirectory() as tmpdir:
            out_csv = Path(tmpdir) / "output.csv"
            with patch("src.data.grid_etl.fetch_nyiso_month", side_effect=mock_nyiso), \
                 patch("src.data.grid_etl.fetch_open_meteo_weather", side_effect=mock_weather), \
                 patch("src.data.db.postgres_manager.PostgresManager") as mock_pm_cls:
                mock_pm = MagicMock()
                mock_pm_cls.return_value = mock_pm

                df_canonical = run_etl(
                    year=2023,
                    output_csv=out_csv,
                    cadence="10min",
                    load_to_postgres=True,
                    pg_url="mock://pg_url",
                )

                self.assertTrue(out_csv.is_file())
                self.assertEqual(df_canonical.columns.tolist(), CANONICAL_COLUMNS)
                self.assertEqual(len(df_canonical), 6)
                self.assertTrue(mock_pm.init_db.called)
                self.assertTrue(mock_pm.insert_telemetry_batch.called)

    def test_mock_run_etl_validation_failure_on_corrupt_data(self) -> None:
        def mock_corrupt_nyiso(year: int, month: int) -> pd.DataFrame:
            if month > 1:
                raise ValueError("1 month only")
            idx = pd.date_range("2023-01-01 00:00", periods=6, freq="10min", tz="UTC")
            return pd.DataFrame({
                "N.Y.C.": [-500.0] * 6,  # Corrupt negative power
                "LONGIL": [2000.0] * 6,
                "HUD VL": [1000.0] * 6,
            }, index=idx)

        def mock_weather(start_str: str, end_str: str) -> pd.DataFrame:
            idx = pd.date_range("2023-01-01 00:00", periods=2, freq="1h", tz="UTC")
            return pd.DataFrame({
                "temperature_2m": [10.0, 12.0],
                "relative_humidity_2m": [50.0, 55.0],
                "wind_speed_10m": [5.0, 6.0],
                "diffuse_radiation": [0.0, 10.0],
                "direct_normal_irradiance": [0.0, 20.0],
            }, index=idx)

        with tempfile.TemporaryDirectory() as tmpdir:
            out_csv = Path(tmpdir) / "corrupt.csv"
            with patch("src.data.grid_etl.fetch_nyiso_month", side_effect=mock_corrupt_nyiso), \
                 patch("src.data.grid_etl.fetch_open_meteo_weather", side_effect=mock_weather):
                with self.assertRaises(GridDataQualityError) as ctx:
                    run_etl(year=2023, output_csv=out_csv, cadence="10min")
                self.assertIn("Physical range violations detected", str(ctx.exception))
                # Corrupt output file should not exist
                self.assertFalse(out_csv.is_file())

    def test_mock_run_etl_no_months_retrieved_raises(self) -> None:
        with patch("src.data.grid_etl.fetch_nyiso_month", side_effect=RuntimeError("Download failed")):
            with self.assertRaises(RuntimeError) as ctx:
                run_etl(year=2023, output_csv="dummy.csv")
            self.assertIn("No load records retrieved", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
