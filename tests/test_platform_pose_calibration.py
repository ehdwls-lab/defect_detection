from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from src.integration.platform_pose_calibration import (
    CAMERA_PLATFORM_RP_20260903,
    CameraPlatformRPCalibration,
    PlatformPoseCalibrationError,
    apply_calibration_to_pose_json,
    calibrated_platform_target,
)
from src.integration.real_pose_planner import RealPosePlanner


class PlatformPoseCalibrationTests(unittest.TestCase):
    @staticmethod
    def camera_for_target(roll, pitch):
        k = np.asarray(CAMERA_PLATFORM_RP_20260903.platform_correction_from_camera_K)
        return np.linalg.solve(k, np.asarray([roll, pitch], dtype=float))

    def test_known_camera_angle_correction(self):
        result = calibrated_platform_target(10, 5, 0, 0, CAMERA_PLATFORM_RP_20260903)
        self.assertAlmostEqual(result["platform_delta_roll_deg"], 9.955565, places=6)
        self.assertAlmostEqual(result["platform_delta_pitch_deg"], -4.400105, places=6)

    def test_zero_camera_angle_has_zero_delta(self):
        result = calibrated_platform_target(0, 0, 3, -4, CAMERA_PLATFORM_RP_20260903)
        self.assertEqual(result["platform_delta_roll_deg"], 0)
        self.assertEqual(result["platform_delta_pitch_deg"], 0)
        self.assertEqual(result["target_platform_roll_deg"], 3)
        self.assertEqual(result["target_platform_pitch_deg"], -4)

    def test_nonzero_current_pose_is_added_to_delta(self):
        result = calibrated_platform_target(10, 5, 2, -3, CAMERA_PLATFORM_RP_20260903)
        self.assertAlmostEqual(
            result["target_platform_roll_deg"],
            2 + result["platform_delta_roll_deg"], places=9,
        )
        self.assertAlmostEqual(
            result["target_platform_pitch_deg"],
            -3 + result["platform_delta_pitch_deg"], places=9,
        )

    def test_29_9_degree_targets_are_reachable(self):
        camera = self.camera_for_target(29.9, -29.9)
        result = calibrated_platform_target(*camera, 0, 0, CAMERA_PLATFORM_RP_20260903)
        self.assertTrue(result["reachable"])

    def test_exact_30_degree_targets_are_reachable(self):
        camera = self.camera_for_target(-30.0, 30.0)
        result = calibrated_platform_target(*camera, 0, 0, CAMERA_PLATFORM_RP_20260903)
        self.assertTrue(result["reachable"])

    def test_over_limit_is_unreachable_and_never_clamped(self):
        camera = self.camera_for_target(30.1, -30.2)
        result = calibrated_platform_target(*camera, 0, 0, CAMERA_PLATFORM_RP_20260903)
        self.assertFalse(result["reachable"])
        self.assertAlmostEqual(result["target_platform_roll_deg"], 30.1, places=8)
        self.assertAlmostEqual(result["target_platform_pitch_deg"], -30.2, places=8)
        self.assertIn("roll=+30.100000", result["reason"])
        self.assertIn("pitch=-30.200000", result["reason"])
        self.assertIn("±30", result["reason"])
        self.assertIn("exceeds", result["reason"])

    def test_missing_and_invalid_calibration_block(self):
        with self.assertRaisesRegex(PlatformPoseCalibrationError, "not loaded"):
            calibrated_platform_target(1, 2, 0, 0, None)
        invalid = CameraPlatformRPCalibration(
            "bad", 15, (0, 0, 1, 1),
            ((1, 2), (2, 4)), ((1, 0), (0, 1)),
        )
        with self.assertRaisesRegex(PlatformPoseCalibrationError, "singular"):
            calibrated_platform_target(1, 2, 0, 0, invalid)

    def test_json_uses_camera_metric_angles_not_legacy_phase(self):
        with tempfile.TemporaryDirectory() as raw:
            path = Path(raw) / "pose.json"
            path.write_text(json.dumps({"planes": [{
                "raw_roll_deg": -80, "raw_pitch_deg": 70,
                "metric_pose": {
                    "physical_metric": True, "status": "METRIC_VALID",
                    "camera_roll_deg": 10, "camera_pitch_deg": 5,
                },
            }]}), encoding="utf-8")
            apply_calibration_to_pose_json(
                path, current_platform_roll_deg=2, current_platform_pitch_deg=-3,
                calibration=CAMERA_PLATFORM_RP_20260903,
            )
            metric = json.loads(path.read_text())["planes"][0]["metric_pose"]
            self.assertAlmostEqual(metric["platform_delta_roll_deg"], 9.955565, places=6)
            self.assertAlmostEqual(metric["target_platform_roll_deg"], 11.955565, places=6)
            self.assertEqual(metric["calibration_id"], "camera_platform_rp_20260903")
            self.assertEqual(metric["calibration"]["calibration_z_cm"], 15.0)
            self.assertEqual(metric["calibration"]["roi"], [450, 180, 950, 520])
            self.assertEqual(metric["calibration"]["platform_limit_deg"], 30.0)
            self.assertNotEqual(metric["target_platform_roll_deg"], -80)

    def test_unloaded_calibration_blocks_motion_planning(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            ply = root / "input.ply"
            ply.write_text("ply", encoding="ascii")
            path = root / "pose.json"
            path.write_text(json.dumps({
                "input_ply": str(ply),
                "planes": [{
                    "plane_name": "one", "points_count": 100,
                    "roll_deg": -80, "pitch_deg": 70,
                    "raw_roll_deg": -80, "raw_pitch_deg": 70,
                    "metric_pose": {
                        "physical_metric": True, "status": "METRIC_VALID",
                        "camera_roll_deg": 10, "camera_pitch_deg": 5,
                    },
                }],
            }), encoding="utf-8")
            apply_calibration_to_pose_json(
                path, current_platform_roll_deg=0, current_platform_pitch_deg=0,
                calibration=None,
            )
            plan = RealPosePlanner().plan(path)
            self.assertEqual(plan.poses, [])
            self.assertFalse(plan.metadata["platform_motion_allowed"])
            metric = json.loads(path.read_text())["planes"][0]["metric_pose"]
            self.assertFalse(metric["reachable"])
            self.assertIn("not loaded", metric["reason"])


if __name__ == "__main__":
    unittest.main()
