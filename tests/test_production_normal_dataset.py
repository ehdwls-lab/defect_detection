from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from src.patch_dataset import PatchDataset
from src.infer_anomaly import select_patch_positions
from src.tools.build_production_normal_manifest import build_manifests


class ProductionNormalDatasetTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.image = self.root / "rgb.png"
        cv2.imwrite(str(self.image), np.full((128, 128, 3), 120, dtype=np.uint8))
        self.mask = self.root / "mask.png"
        mask = np.zeros((128, 128), dtype=np.uint8)
        mask[0:64, 0:128] = 255
        cv2.imwrite(str(self.mask), mask)

    def tearDown(self):
        self.temp.cleanup()

    def write_manifest(self, rows: list[dict[str, str]]) -> Path:
        path = self.root / "manifest.csv"
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
        return path

    def test_mask_mode_accepts_only_full_coverage_patches(self):
        manifest = self.write_manifest([{
            "path": str(self.image), "split": "train", "label": "normal",
            "mask": str(self.mask), "regions": "",
        }])
        dataset = PatchDataset(manifest_path=manifest, split="train", return_metadata=True)
        self.assertEqual(len(dataset), 3)
        _, metadata = dataset[0]
        self.assertEqual(metadata["mask_path"], str(self.mask))

    def test_one_outside_pixel_rejects_patch(self):
        mask = np.full((128, 128), 255, dtype=np.uint8)
        mask[63, 63] = 0
        cv2.imwrite(str(self.mask), mask)
        positions = select_patch_positions(
            (128, 128, 3), 64, 32, surface_mask=mask, min_surface_coverage=1.0,
        )
        self.assertNotIn((0, 0), positions)

    def test_mask_takes_priority_over_regions(self):
        regions = self.root / "regions.json"
        regions.write_text('{"regions": [{"x1": 0, "y1": 0, "x2": 128, "y2": 128}]}')
        manifest = self.write_manifest([{
            "path": str(self.image), "split": "train", "label": "normal",
            "mask": str(self.mask), "regions": str(regions),
        }])
        dataset = PatchDataset(manifest_path=manifest, split="train")
        self.assertEqual(len(dataset), 3)

    def test_region_mode_remains_backward_compatible(self):
        regions = self.root / "regions.json"
        regions.write_text('{"regions": [{"x1": 0, "y1": 0, "x2": 128, "y2": 128}]}')
        manifest = self.write_manifest([{
            "path": str(self.image), "split": "train", "label": "normal",
            "mask": "", "regions": str(regions),
        }])
        self.assertEqual(len(PatchDataset(manifest_path=manifest, split="train")), 9)

    def test_builder_rejects_same_run_in_train_and_val(self):
        with self.assertRaisesRegex(ValueError, "same run"):
            build_manifests([self.root], [self.root], self.root / "out", session="gray_01", material="gray")

    def test_builder_rejects_missing_mask(self):
        run = self.root / "run_01" / "plane_00" / "final_capture"
        run.mkdir(parents=True)
        cv2.imwrite(str(run / "final_rgb.png"), np.full((128, 128, 3), 120, dtype=np.uint8))
        with self.assertRaises(FileNotFoundError):
            build_manifests([self.root / "run_01"], [self.root / "run_02"], self.root / "out", session="gray_01", material="gray")

    def test_latest_run_matches_production_counts(self):
        run = Path("results/integrated_hardware/run_20260904_075229")
        if not run.is_dir():
            self.skipTest("latest hardware run artifact is unavailable")
        rows = []
        for plane in ("plane_00", "plane_01"):
            image = cv2.imread(str(run / plane / "final_capture" / "final_rgb.png"))
            mask = cv2.imread(str(run / plane / "anomaly" / "inspection_mask.png"), cv2.IMREAD_GRAYSCALE)
            from src.infer_anomaly import select_patch_positions
            positions = select_patch_positions(image.shape, 64, 32, surface_mask=mask, min_surface_coverage=1.0)
            rows.append(len(positions))
        self.assertEqual(rows, [105, 60])


if __name__ == "__main__":
    unittest.main()
