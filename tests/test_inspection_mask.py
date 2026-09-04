from __future__ import annotations

import unittest
from dataclasses import replace

import numpy as np

from src.config import InspectionConfig
from src.core.inspection_mask import build_inspection_mask


def config(*, close_size: int = 3, erosion: int = 0) -> InspectionConfig:
    base = InspectionConfig.default()
    return replace(base, surface_roi=replace(
        base.surface_roi,
        boundary_margin_px=erosion,
        inspection_close_size_px=close_size,
        inspection_close_iterations=1,
    ))


class InspectionMaskTests(unittest.TestCase):
    def test_enclosed_depth_hole_is_filled(self):
        mask = np.zeros((80, 80), dtype=np.uint8)
        mask[10:70, 10:70] = 255
        mask[35:40, 35:40] = 0
        result = build_inspection_mask(mask, mask, config())
        self.assertTrue(np.all(result.mask[35:40, 35:40] == 255))

    def test_background_gap_connected_to_frame_is_not_filled(self):
        mask = np.zeros((80, 80), dtype=np.uint8)
        mask[10:70, 10:70] = 255
        mask[10:55, 32:47] = 0
        result = build_inspection_mask(mask, mask, config())
        self.assertEqual(int(result.mask[20, 40]), 0)

    def test_inspection_mask_can_restore_area_missing_from_surface_mask(self):
        object_mask = np.zeros((80, 80), dtype=np.uint8)
        object_mask[10:70, 10:70] = 255
        surface_mask = np.zeros_like(object_mask)
        surface_mask[20:60, 20:60] = 255
        result = build_inspection_mask(object_mask, surface_mask, config())
        self.assertGreater(result.inspection_area_px, np.count_nonzero(surface_mask))
        self.assertGreater(result.inspection_to_surface_ratio, 1.0)

    def test_full_frame_object_mask_is_rejected(self):
        mask = np.ones((80, 80), dtype=np.uint8) * 255
        with self.assertRaisesRegex(ValueError, "full-frame"):
            build_inspection_mask(mask, mask, config())


if __name__ == "__main__":
    unittest.main()
