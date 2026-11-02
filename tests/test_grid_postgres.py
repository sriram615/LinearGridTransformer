"""Unit tests for PostgreSQL / TimescaleDB Preprocessing and Dataset Export."""

import json
import unittest
from pathlib import Path

import numpy as np

from src.data.db.postgres_manager import DEFAULT_PG_URL, PostgresManager
from src.data.tetouan_dataset import TetouanSequenceDataset

REPO_ROOT = Path(__file__).resolve().parents[1]
GRID_PROCESSED_DIR = REPO_ROOT / "data/processed/grid"


class TestGridPostgresPreprocessing(unittest.TestCase):
    """Test suite verifying database preprocessing and PyTorch dataset loading."""

    @classmethod
    def setUpClass(cls) -> None:
        try:
            cls.pm = PostgresManager(connection_url=DEFAULT_PG_URL)
            cls.split_bounds = cls.pm.get_split_bounds()
            cls.stats = cls.pm.get_train_stats()
            cls.has_db = True
        except Exception:
            cls.has_db = False

    def test_database_split_bounds(self) -> None:
        if not self.has_db:
            self.skipTest("TimescaleDB not available")

        self.assertIn("train", self.split_bounds)
        self.assertIn("validation", self.split_bounds)
        self.assertIn("test", self.split_bounds)

        self.assertEqual(self.split_bounds["train"]["start_index"], 0)
        self.assertEqual(self.split_bounds["train"]["end_index_exclusive"], 36788)
        self.assertEqual(self.split_bounds["validation"]["start_index"], 36788)
        self.assertEqual(self.split_bounds["validation"]["end_index_exclusive"], 44671)
        self.assertEqual(self.split_bounds["test"]["start_index"], 44671)
        self.assertEqual(self.split_bounds["test"]["end_index_exclusive"], 52555)

    def test_database_train_stats_no_leakage(self) -> None:
        if not self.has_db:
            self.skipTest("TimescaleDB not available")

        self.assertEqual(len(self.stats["feature_mean"]), 8)
        self.assertEqual(len(self.stats["feature_std"]), 8)
        self.assertEqual(len(self.stats["target_mean"]), 3)
        self.assertEqual(len(self.stats["target_std"]), 3)

        # Zone loads must be positive and within reasonable MW ranges (NYC: ~5600 MW)
        self.assertGreater(self.stats["feature_mean"][0], 3000.0)
        self.assertLess(self.stats["feature_mean"][0], 8000.0)
        for std in self.stats["feature_std"] + self.stats["target_std"]:
            self.assertGreater(std, 0.0)

    def test_exported_artifacts_exist_and_valid(self) -> None:
        npz_path = GRID_PROCESSED_DIR / "series.npz"
        meta_path = GRID_PROCESSED_DIR / "metadata.json"

        self.assertTrue(npz_path.is_file(), f"Missing {npz_path}")
        self.assertTrue(meta_path.is_file(), f"Missing {meta_path}")

        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        self.assertEqual(meta["row_count"], 52555)
        self.assertEqual(meta["windows"]["input_contract"], [144, 8])
        self.assertEqual(meta["windows"]["target_contract"], [6, 3])

        with np.load(npz_path) as data:
            self.assertEqual(data["features"].shape, (52555, 8))
            self.assertEqual(data["targets"].shape, (52555, 3))
            self.assertEqual(data["timestamps"].shape, (52555,))

    def test_dataset_loader_compatibility(self) -> None:
        for split in ("train", "validation", "test"):
            dataset = TetouanSequenceDataset(split=split, artifact_dir=GRID_PROCESSED_DIR)
            self.assertGreater(len(dataset), 7000)

            x, y = dataset[0]
            self.assertEqual(x.shape, (144, 8))
            self.assertEqual(y.shape, (6, 3))
            self.assertTrue(np.isfinite(x.numpy()).all())
            self.assertTrue(np.isfinite(y.numpy()).all())


if __name__ == "__main__":
    unittest.main()
