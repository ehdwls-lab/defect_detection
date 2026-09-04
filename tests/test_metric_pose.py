from __future__ import annotations

import math
import unittest

import numpy as np

from src.integration.metric_pose import (
    CameraIntrinsics,
    MetricFitConfig,
    PlatformAxisContract,
    build_metric_pose,
    ply_xy_to_depth_pixels,
)


class MetricPoseTests(unittest.TestCase):
    def setUp(self):
        self.intrinsics = CameraIntrinsics(
            fx=500.0, fy=500.0, cx=49.5, cy=39.5,
            width=100, height=80, source="synthetic SDK profile", aligned_to="color",
        )
        self.contract = PlatformAxisContract(
            verified=True, roll_source="camera_roll", pitch_source="camera_pitch",
            reference="synthetic verified mapping",
        )
        self.config = MetricFitConfig(
            min_depth_mm=100, max_depth_mm=2000, ransac_threshold_mm=0.2,
            ransac_iterations=80, min_points=100, max_points=10000,
            min_coverage=0.5, max_tilt_deg=22.5,
        )
        vv, uu = np.mgrid[10:70, 10:90]
        self.pixels = np.column_stack((uu.ravel(), vv.ravel()))

    def depth_for_angles(self, roll_deg=0.0, pitch_deg=0.0):
        # Plane n.x + d = 0, n=(-tan(R), -tan(P), -1), passing through Z=500.
        r = math.tan(math.radians(roll_deg))
        p = math.tan(math.radians(pitch_deg))
        uu, vv = np.meshgrid(np.arange(100), np.arange(80))
        denominator = 1.0 - r * (uu - self.intrinsics.cx) / self.intrinsics.fx
        denominator -= p * (vv - self.intrinsics.cy) / self.intrinsics.fy
        return (500.0 / denominator).astype(np.float32)

    def assert_angles(self, result, roll, pitch, places=2):
        self.assertEqual(result["status"], "REACHABLE")
        self.assertAlmostEqual(result["roll_deg"], roll, places=places)
        self.assertAlmostEqual(result["pitch_deg"], pitch, places=places)

    def test_horizontal_plane(self):
        result = build_metric_pose(
            self.depth_for_angles(), self.pixels, self.intrinsics, self.contract, self.config,
        )
        self.assert_angles(result, 0.0, 0.0)

    def test_known_roll_plane(self):
        result = build_metric_pose(
            self.depth_for_angles(12.0, 0.0), self.pixels,
            self.intrinsics, self.contract, self.config,
        )
        self.assert_angles(result, 12.0, 0.0)

    def test_known_pitch_plane(self):
        result = build_metric_pose(
            self.depth_for_angles(0.0, -8.0), self.pixels,
            self.intrinsics, self.contract, self.config,
        )
        self.assert_angles(result, 0.0, -8.0)

    def test_limit_is_reachable_without_clipping(self):
        result = build_metric_pose(
            self.depth_for_angles(22.5, -22.5), self.pixels,
            self.intrinsics, self.contract, self.config,
        )
        self.assert_angles(result, 22.5, -22.5, places=1)

    def test_over_limit_is_unreachable_and_not_clipped(self):
        result = build_metric_pose(
            self.depth_for_angles(30.0, 0.0), self.pixels,
            self.intrinsics, self.contract, self.config,
        )
        self.assertEqual(result["status"], "UNREACHABLE")
        self.assertFalse(result["reachable"])
        self.assertAlmostEqual(result["roll_deg"], 30.0, places=1)

    def test_invalid_depth_is_metric_invalid(self):
        result = build_metric_pose(
            np.zeros((80, 100), np.float32), self.pixels,
            self.intrinsics, self.contract, self.config,
        )
        self.assertEqual(result["status"], "DETECTED")
        self.assertFalse(result["reachable"])

    def test_unverified_axis_contract_blocks_reachability(self):
        result = build_metric_pose(
            self.depth_for_angles(), self.pixels, self.intrinsics,
            PlatformAxisContract(verified=False), self.config,
        )
        self.assertEqual(result["status"], "METRIC_VALID")
        self.assertFalse(result["reachable"])

    def test_ply_pixel_recovery_applies_180_degree_mapping(self):
        points = np.array([[-40.0, 30.0, 1.0]])  # SL pixel (10, 10)
        pixels = ply_xy_to_depth_pixels(points, 100, 80, transform="rotate_180")
        np.testing.assert_array_equal(pixels, [[89, 69]])


if __name__ == "__main__":
    unittest.main()
