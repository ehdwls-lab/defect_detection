from __future__ import annotations

import unittest
from dataclasses import replace

import numpy as np

from src.config import InspectionConfig
from src.core.aruco_board import detect_markers, get_board_outer_quad, polygon_mask
from src.integration.hybrid_inspection_roi import build_hybrid_inspection_roi
from src.integration.hybrid_inspection_roi import HybridROIError
from src.integration.metric_pose import CameraIntrinsics


def marker_map() -> dict[int, np.ndarray]:
    return {
        0: np.array([[10, 10], [20, 10], [20, 20], [10, 20]], np.float32),
        1: np.array([[80, 10], [90, 10], [90, 20], [80, 20]], np.float32),
        2: np.array([[10, 80], [20, 80], [20, 90], [10, 90]], np.float32),
        3: np.array([[80, 80], [90, 80], [90, 90], [80, 90]], np.float32),
    }


class FakeDetector:
    def __init__(self, markers: dict[int, np.ndarray]):
        self.markers = markers

    def detectMarkers(self, gray):
        del gray
        ids = np.asarray(sorted(self.markers), dtype=np.int32).reshape(-1, 1)
        corners = [self.markers[int(marker_id)][None, ...] for marker_id in ids[:, 0]]
        return corners, ids, []


def inspection_config() -> InspectionConfig:
    base = InspectionConfig.default()
    return replace(
        base,
        surface_roi=replace(base.surface_roi, boundary_margin_px=0),
        hybrid_roi=replace(
            base.hybrid_roi,
            marker_ignore_px=2,
            unknown_recovery_radius_px=16,
            close_size_px=3,
        ),
    )


def intrinsics() -> CameraIntrinsics:
    return CameraIntrinsics(200, 200, 50, 50, 100, 100, "synthetic", "color")


def plane_depth(*, tilted: bool = False) -> np.ndarray:
    vv, uu = np.mgrid[:100, :100]
    if not tilted:
        return np.full((100, 100), 500, dtype=np.float32)
    normal = np.array([0.12, -0.06, -1.0], dtype=np.float64)
    normal /= np.linalg.norm(normal)
    center = np.array([0.0, 0.0, 500.0])
    ray_x = (uu - 50) / 200
    ray_y = (vv - 50) / 200
    denominator = normal[0] * ray_x + normal[1] * ray_y + normal[2]
    return ((normal @ center) / denominator).astype(np.float32)


class HybridInspectionROITests(unittest.TestCase):
    def test_existing_marker_contract_builds_outer_board_polygon(self):
        markers = detect_markers(
            np.zeros((100, 100, 3), dtype=np.uint8), FakeDetector(marker_map()),
        )
        quad = get_board_outer_quad(markers)
        np.testing.assert_array_equal(
            quad, np.array([[10, 10], [90, 10], [90, 90], [10, 90]], np.float32),
        )
        self.assertGreater(np.count_nonzero(polygon_mask((100, 100, 3), quad)), 0)

    def test_tilted_board_and_raised_object_are_classified_in_metric_space(self):
        depth = plane_depth(tilted=True)
        depth[35:65, 35:65] -= 30
        color = np.full((100, 100, 3), 120, dtype=np.uint8)
        result = build_hybrid_inspection_roi(
            color, depth, intrinsics(), get_board_outer_quad(marker_map()),
            marker_map(), inspection_config(),
        )
        self.assertGreater(result.board_plane_inlier_ratio, .9)
        self.assertLess(result.board_plane_residual_mm, 2.0)
        self.assertGreater(np.count_nonzero(result.board_background_mask), 0)
        self.assertGreater(result.depth_object_area_px, 0)
        self.assertEqual(int(result.depth_object_mask[50, 50]), 255)

    def test_invalid_depth_is_unknown_and_enclosed_hole_is_restored(self):
        depth = plane_depth()
        depth[30:70, 30:70] = 460
        depth[45:55, 45:55] = 0
        color = np.full((100, 100, 3), 100, dtype=np.uint8)
        result = build_hybrid_inspection_roi(
            color, depth, intrinsics(), get_board_outer_quad(marker_map()),
            marker_map(), inspection_config(),
        )
        self.assertEqual(int(result.depth_unknown_mask[50, 50]), 255)
        self.assertEqual(int(result.board_background_mask[50, 50]), 0)
        self.assertEqual(int(result.inspection_mask[50, 50]), 255)

    def test_board_external_unknown_is_never_recovered(self):
        depth = plane_depth()
        depth[35:65, 35:65] = 460
        depth[:8, :8] = 0
        color = np.full((100, 100, 3), 100, dtype=np.uint8)
        result = build_hybrid_inspection_roi(
            color, depth, intrinsics(), get_board_outer_quad(marker_map()),
            marker_map(), inspection_config(),
        )
        self.assertEqual(int(result.inspection_mask[4, 4]), 0)
        self.assertFalse(np.any((result.inspection_mask > 0) & (result.board_roi_mask == 0)))

    def test_object_colored_nearby_unknown_is_recovered_but_board_color_is_not(self):
        depth = plane_depth()
        depth[35:65, 35:60] = 460
        depth[35:65, 60:70] = 0
        color = np.full((100, 100, 3), (120, 120, 120), dtype=np.uint8)
        color[35:65, 35:65] = (10, 10, 200)
        result = build_hybrid_inspection_roi(
            color, depth, intrinsics(), get_board_outer_quad(marker_map()),
            marker_map(), inspection_config(),
        )
        self.assertEqual(int(result.rgb_recovered_unknown_mask[50, 62]), 255)
        self.assertEqual(int(result.rgb_recovered_unknown_mask[50, 68]), 0)

    def test_local_defect_color_does_not_punch_enclosed_unknown_hole(self):
        depth = plane_depth()
        depth[30:70, 30:70] = 460
        depth[45:55, 45:55] = 0
        color = np.full((100, 100, 3), (10, 10, 200), dtype=np.uint8)
        color[45:55, 45:55] = (200, 10, 10)
        result = build_hybrid_inspection_roi(
            color, depth, intrinsics(), get_board_outer_quad(marker_map()),
            marker_map(), inspection_config(),
        )
        self.assertEqual(int(result.rgb_recovered_unknown_mask[50, 50]), 0)
        self.assertEqual(int(result.inspection_mask[50, 50]), 255)

    def test_nearly_full_frame_board_polygon_is_rejected(self):
        depth = plane_depth()
        depth[35:65, 35:65] = 460
        full_quad = np.array([[0, 0], [99, 0], [99, 99], [0, 99]], np.float32)
        with self.assertRaisesRegex(HybridROIError, "nearly full-frame"):
            build_hybrid_inspection_roi(
                np.zeros((100, 100, 3), dtype=np.uint8), depth, intrinsics(),
                full_quad, {}, inspection_config(),
            )


if __name__ == "__main__":
    unittest.main()
