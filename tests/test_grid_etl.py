"""Unit tests for the Multi-Zone Grid Ingestion & ETL pipeline."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.grid_etl import CANONICAL_COLUMNS


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

    def test_downloaded_csv_integrity(self) -> None:
        csv_file = Path("data/raw/grid/grid_power_consumption.csv")
        if not csv_file.is_file():
            self.skipTest("Downloaded grid CSV not found in workspace.")

        df = pd.read_csv(csv_file)
        self.assertEqual(df.columns.tolist(), CANONICAL_COLUMNS)
        self.assertGreater(len(df), 50000)
        self.assertEqual(df.isna().sum().sum(), 0)

        # Check numeric finite values
        for col in CANONICAL_COLUMNS[1:]:
            values = pd.to_numeric(df[col], errors="raise").to_numpy()
            self.assertTrue(np.isfinite(values).all(), f"Non-finite values in {col}")
            if "Zone" in col:
                self.assertTrue((values > 0).all(), f"Non-positive load values in {col}")


if __name__ == "__main__":
    unittest.main()
