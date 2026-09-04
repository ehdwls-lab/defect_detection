from __future__ import annotations

import math
import tempfile
import unittest
from pathlib import Path

import numpy as np

from src.camera.controller import RGBDepthFrame
from src.integration.metric_pose import CameraIntrinsics, MetricFitConfig, MetricPoseError
from src.platform.types import PlatformTelemetry
from src.tools.measure_depth_plane_angle import (
    DepthPlaneAngleMeasurement,
    DepthPlaneMeasurementError,
    validate_roi,
)


def telemetry(*, stable=True, homing=False):
    return PlatformTelemetry(
        z_cm=20.0, roll_deg=-0.08, pitch_deg=0.01,
        stable=stable, homing=homing, motor1=0, motor2=0, motor3=0,
        imu_mode=1, control_mode=1, roll_rate_deg_s=0, pitch_rate_deg_s=0,
        timestamp=1.0,
    )


class FakePlatform:
    def __init__(self, value):
        self.value = value
        self.reads = 0
        self.writes = []
        self.closed = False

    def connect(self): pass
    def read_fresh_telemetry(self, **kwargs):
        self.reads += 1
        return self.value
    def move_to(self, *args): self.writes.append(args); raise AssertionError("motion forbidden")
    def move_z(self, *args): self.writes.append(args); raise AssertionError("motion forbidden")
    def move_orientation(self, *args): self.writes.append(args); raise AssertionError("motion forbidden")
    def close(self): self.closed = True


class FakeCamera:
    def __init__(self, depth):
        self.depth = depth
        self.starts = 0
        self.captures = 0
        self.closed = False
        self.intrinsics = CameraIntrinsics(
            fx=500, fy=500, cx=49.5, cy=39.5, width=100, height=80,
            source="synthetic rgb_intrinsic", aligned_to="color",
        )

    def start(self): self.starts += 1
    def capture(self):
        self.captures += 1
        return RGBDepthFrame(
            color_bgr=np.zeros((80, 100, 3), np.uint8),
            depth_mm=self.depth, timestamp=1.0,
        )
    def color_intrinsics(self, width, height):
        self.asserted_grid = (width, height)
        return self.intrinsics
    def close(self): self.closed = True


def plane_depth(roll_deg=0.0, pitch_deg=0.0):
    r = math.tan(math.radians(roll_deg))
    p = math.tan(math.radians(pitch_deg))
    uu, vv = np.meshgrid(np.arange(100), np.arange(80))
    denominator = 1 - r * (uu - 49.5) / 500 - p * (vv - 39.5) / 500
    return (500 / denominator).astype(np.float32)


class DepthPlaneAngleMeasurementTests(unittest.TestCase):
    def config(self):
        return MetricFitConfig(
            min_depth_mm=100, max_depth_mm=1000, ransac_threshold_mm=.2,
            ransac_iterations=80, min_points=100, max_points=10000,
        )

    def run_measurement(self, platform, camera, root, roi=(10, 10, 90, 70), save_preview=False):
        measurement = DepthPlaneAngleMeasurement(
            platform=platform, camera=camera, roi=roi,
            output_directory=Path(root) / "run", fit_config=self.config(),
            telemetry_settle_s=0, save_preview=save_preview,
        )
        self.assertFalse(hasattr(measurement, "structured_light_runner"))
        self.assertFalse(hasattr(measurement, "projector"))
        self.assertFalse(hasattr(measurement, "conveyor"))
        self.assertFalse(hasattr(measurement, "automatic_z"))
        return measurement.run()

    def test_synthetic_known_plane_and_read_only_contract(self):
        platform = FakePlatform(telemetry())
        camera = FakeCamera(plane_depth(12, -8))
        with tempfile.TemporaryDirectory() as root:
            result = self.run_measurement(platform, camera, root)
            self.assertAlmostEqual(result["plane"]["camera_roll_deg"], 12, places=2)
            self.assertAlmostEqual(result["plane"]["camera_pitch_deg"], -8, places=2)
            self.assertTrue((Path(root) / "run" / "summary.json").is_file())
        self.assertEqual(platform.reads, 1)
        self.assertEqual(platform.writes, [])
        self.assertEqual(camera.captures, 1)
        self.assertFalse(result["safety"]["structured_light_used"])
        self.assertFalse(result["safety"]["projector_used"])
        self.assertFalse(result["safety"]["conveyor_used"])
        self.assertFalse(result["safety"]["automatic_z_used"])

    def test_roi_bounds(self):
        for roi in ((-1, 0, 10, 10), (0, 0, 101, 10), (5, 5, 5, 10)):
            with self.subTest(roi=roi), self.assertRaises(DepthPlaneMeasurementError):
                validate_roi(roi, 100, 80)

    def test_invalid_depth(self):
        platform = FakePlatform(telemetry())
        camera = FakeCamera(np.zeros((80, 100), np.float32))
        with tempfile.TemporaryDirectory() as root, self.assertRaises(MetricPoseError):
            self.run_measurement(platform, camera, root)
        self.assertEqual(platform.writes, [])

    def test_full_previews_survive_plane_fit_failure(self):
        platform = FakePlatform(telemetry())
        camera = FakeCamera(np.zeros((80, 100), np.float32))
        with tempfile.TemporaryDirectory() as root:
            with self.assertRaises(MetricPoseError):
                self.run_measurement(platform, camera, root, save_preview=True)
            run = Path(root) / "run"
            self.assertTrue((run / "color_full_preview.png").is_file())
            self.assertTrue((run / "depth_full_preview.png").is_file())
            self.assertFalse((run / "color_roi_preview.png").exists())
            self.assertFalse((run / "summary.json").exists())

    def test_full_previews_survive_invalid_roi(self):
        platform = FakePlatform(telemetry())
        camera = FakeCamera(plane_depth())
        with tempfile.TemporaryDirectory() as root:
            with self.assertRaises(DepthPlaneMeasurementError):
                self.run_measurement(
                    platform, camera, root, roi=(0, 0, 101, 80), save_preview=True,
                )
            run = Path(root) / "run"
            self.assertTrue((run / "color_full_preview.png").is_file())
            self.assertTrue((run / "depth_full_preview.png").is_file())

    def test_homing_and_unstable_gate_before_camera(self):
        for value in (telemetry(homing=True), telemetry(stable=False)):
            platform = FakePlatform(value)
            camera = FakeCamera(plane_depth())
            with tempfile.TemporaryDirectory() as root:
                with self.subTest(value=value), self.assertRaises(DepthPlaneMeasurementError):
                    self.run_measurement(platform, camera, root)
            self.assertEqual(platform.reads, 1)
            self.assertEqual(platform.writes, [])
            self.assertEqual(camera.starts, 0)
            self.assertEqual(camera.captures, 0)


if __name__ == "__main__":
    unittest.main()
