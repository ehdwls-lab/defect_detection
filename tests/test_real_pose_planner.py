from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from src.integration.coordinate_contract import (
    CloudType, DEFAULT_COORDINATE_CONVENTION, PointCloudMetadata, StructuredLightResult,
)
from src.integration.real_pose_planner import PoseJSONError, RealPosePlanner, parse_pose_json
from src.integration.platform_alignment import predicted_camera_residual_angle_deg


class RealPosePlannerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.ply = self.root / "FINAL.ply"
        self.ply.write_text("ply", encoding="ascii")

    def tearDown(self):
        self.temp.cleanup()

    def payload(self):
        return {
            "schema_version": "structured_light_pose_v1",
            "input_ply": str(self.ply),
            "coordinate_contract": {"xy_unit": "pixel", "z_unit": "phase_relative", "metric_z": False},
            "planes": [
                {"plane_name": "Shot 2", "dominant": False, "points_count": 20,
                 "roll_deg": 2.0, "pitch_deg": 3.0, "raw_roll_deg": 2.5, "raw_pitch_deg": 3.5,
                 "metric_pose": {"source": "orbbec_depth", "physical_metric": True,
                                 "status": "REACHABLE", "reachable": True,
                                 "roll_deg": 2.0, "pitch_deg": 3.0,
                                 "target_platform_roll_deg": 2.0,
                                 "target_platform_pitch_deg": 3.0,
                                 "calibration_id": "camera_platform_rp_20260903",
                                 "depth_points_count": 20, "depth_coverage": 1.0},
                 "legacy_relative_z": {"value_cm": 20.0, "metric": False, "stm_compatible": False}},
                {"plane_name": "Shot 1", "dominant": True, "points_count": 80,
                 "roll_deg": -6.83, "pitch_deg": 6.44, "raw_roll_deg": -6.825, "raw_pitch_deg": 6.438,
                 "metric_pose": {"source": "orbbec_depth", "physical_metric": True,
                                 "status": "REACHABLE", "reachable": True,
                                 "roll_deg": -6.83, "pitch_deg": 6.44,
                                 "target_platform_roll_deg": -6.83,
                                 "target_platform_pitch_deg": 6.44,
                                 "calibration_id": "camera_platform_rp_20260903",
                                 "depth_points_count": 80, "depth_coverage": 1.0},
                 "legacy_relative_z": {"value_cm": 99.0, "metric": False, "stm_compatible": False}},
            ],
            "stm_z_command_allowed": False,
        }

    def write(self, payload=None):
        path = self.root / "FINAL_pose.json"
        path.write_text(json.dumps(self.payload() if payload is None else payload), encoding="utf-8")
        return path

    def test_real_pose_json_parse_preserves_all_planes(self):
        document = parse_pose_json(self.write())
        self.assertEqual(len(document.planes), 2)
        self.assertAlmostEqual(document.planes[1].point_ratio, 0.8)
        self.assertFalse(document.stm_z_command_allowed)

    def test_dominant_selection_and_plan_metadata(self):
        plan = RealPosePlanner().plan(self.write())
        self.assertEqual(len(plan.poses), 1)
        pose = plan.poses[0]
        self.assertEqual(pose.pose_id, "Shot 1")
        self.assertEqual((pose.roll_deg, pose.pitch_deg), (-6.83, 6.44))
        self.assertEqual(pose.metadata["point_count"], 80)
        self.assertEqual(pose.metadata["plane_role"], "dominant")
        self.assertEqual(pose.metadata["source_plane_index"], 1)
        self.assertTrue(pose.metadata["dominant"])
        self.assertEqual(plan.metadata["detected_plane_count"], 2)
        self.assertFalse(plan.metadata["z_provided"])
        self.assertTrue(plan.metadata["platform_motion_allowed"])
        self.assertFalse(plan.metadata["automatic_z_allowed"])

    def test_inspection_operational_limit_clamps_roll_and_preserves_requested(self):
        payload = self.payload()
        metric = payload["planes"][1]["metric_pose"]
        metric.update(
            target_platform_roll_deg=30.0,
            target_platform_pitch_deg=27.0,
            commanded_target_roll_deg=30.0,
            commanded_target_pitch_deg=27.0,
            camera_roll_deg=0.0,
            camera_pitch_deg=0.0,
            current_platform_roll_deg=0.0,
            current_platform_pitch_deg=0.0,
        )
        pose = RealPosePlanner().plan(self.write(payload)).poses[0]
        self.assertEqual((pose.roll_deg, pose.pitch_deg), (28.0, 27.0))
        self.assertEqual(pose.metadata["requested_roll"], 30.0)
        self.assertEqual(pose.metadata["applied_roll"], 28.0)
        self.assertTrue(pose.metadata["clamped"])
        self.assertEqual(pose.metadata["metric_pose"]["commanded_target_roll_deg"], 28.0)
        self.assertAlmostEqual(
            pose.metadata["metric_pose"]["predicted_residual_angle_deg"],
            predicted_camera_residual_angle_deg(
                -1.027901 * 28.0 - 0.053029 * 27.0,
                -0.039609 * 28.0 + 1.046716 * 27.0,
            ),
            places=6,
        )

    def test_inspection_operational_limit_keeps_safe_pose_unclamped(self):
        payload = self.payload()
        metric = payload["planes"][1]["metric_pose"]
        metric.update(
            target_platform_roll_deg=27.0, target_platform_pitch_deg=-27.0,
            commanded_target_roll_deg=27.0, commanded_target_pitch_deg=-27.0,
            camera_roll_deg=0.0, camera_pitch_deg=0.0,
            current_platform_roll_deg=0.0, current_platform_pitch_deg=0.0,
        )
        pose = RealPosePlanner().plan(self.write(payload)).poses[0]
        self.assertEqual((pose.roll_deg, pose.pitch_deg), (27.0, -27.0))
        self.assertFalse(pose.metadata["clamped"])

    def test_structured_light_result_input_discovers_sibling_json(self):
        path = self.write()
        result = StructuredLightResult(
            run_id="run", ply_path=self.ply,
            cloud=PointCloudMetadata(self.ply, 1, False, False, True, False, False,
                                     CloudType.OBJECT_ONLY, DEFAULT_COORDINATE_CONVENTION),
            metadata={"pose_json_path": str(path)},
        )
        self.assertEqual(RealPosePlanner().plan(result).poses[0].pose_id, "Shot 1")

    def test_legacy_z_is_ignored(self):
        plan = RealPosePlanner().plan(self.write())
        self.assertTrue(plan.poses[0].metadata["legacy_z_ignored"])
        self.assertFalse(plan.poses[0].metadata["stm_z_command_allowed"])
        self.assertNotIn("z_cm", plan.poses[0].metadata)

    def test_unsafe_legacy_z_contract_is_rejected(self):
        payload = self.payload()
        payload["planes"][0]["legacy_relative_z"]["metric"] = True
        with self.assertRaises(PoseJSONError):
            parse_pose_json(self.write(payload))

    def test_legacy_phase_pose_is_never_a_motion_fallback(self):
        payload = self.payload()
        for plane in payload["planes"]:
            del plane["metric_pose"]
        plan = RealPosePlanner("all_valid_planes").plan(self.write(payload))
        self.assertEqual(plan.poses, [])
        self.assertFalse(plan.metadata["platform_motion_allowed"])
        self.assertEqual(len(plan.metadata["rejected_planes"]), 2)

    def test_malformed_json(self):
        path = self.root / "bad.json"
        path.write_text("{", encoding="utf-8")
        with self.assertRaises(PoseJSONError):
            parse_pose_json(path)

    def test_no_planes(self):
        payload = self.payload()
        payload["planes"] = []
        with self.assertRaises(PoseJSONError):
            parse_pose_json(self.write(payload))

    def test_missing_roll_or_pitch(self):
        for missing in ("roll_deg", "pitch_deg"):
            payload = self.payload()
            del payload["planes"][0][missing]
            with self.subTest(missing=missing), self.assertRaises(PoseJSONError):
                parse_pose_json(self.write(payload))

    def test_selection_is_deterministic_and_uses_largest_count(self):
        payload = self.payload()
        payload["planes"][0]["points_count"] = 80
        payload["planes"][1]["points_count"] = 80
        payload["planes"][0]["dominant"] = True
        payload["planes"][1]["dominant"] = True
        path = self.write(payload)
        selected = [RealPosePlanner().plan(path).poses[0].pose_id for _ in range(5)]
        self.assertEqual(selected, ["Shot 2"] * 5)

    def test_all_valid_planes_preserves_dominant_first_deterministic_order(self):
        plan = RealPosePlanner("all_valid_planes").plan(self.write())
        self.assertEqual([pose.pose_id for pose in plan.poses], ["Shot 1", "Shot 2"])
        self.assertEqual([pose.metadata["point_count"] for pose in plan.poses], [80, 20])
        self.assertEqual([pose.metadata["source_plane_index"] for pose in plan.poses], [1, 0])
        self.assertEqual([pose.metadata["dominant"] for pose in plan.poses], [True, False])
        self.assertEqual(plan.metadata["selection_policy"], "all_valid_planes")

    def test_multi_plane_alias_is_kept_for_compatibility(self):
        plan = RealPosePlanner("multi_plane").plan(self.write())
        self.assertEqual(len(plan.poses), 2)

    def test_full_poses_are_ordered_before_partial_and_commanded_target_is_used(self):
        payload = self.payload()
        full = payload["planes"][0]["metric_pose"]
        partial = payload["planes"][1]["metric_pose"]
        full.update(
            alignment_mode="FULL", commanded_target_roll_deg=2.0,
            commanded_target_pitch_deg=3.0,
        )
        partial.update(
            alignment_mode="PARTIAL", desired_target_roll_deg=-35.0,
            desired_target_pitch_deg=6.0, commanded_target_roll_deg=-30.0,
            commanded_target_pitch_deg=5.8, predicted_residual_angle_deg=5.0,
        )
        plan = RealPosePlanner("all_valid_planes").plan(self.write(payload))
        self.assertEqual([pose.metadata["alignment_mode"] for pose in plan.poses], ["FULL", "PARTIAL"])
        self.assertEqual((plan.poses[1].roll_deg, plan.poses[1].pitch_deg), (-28.0, 5.8))
        self.assertEqual(plan.poses[1].metadata["requested_roll"], -30.0)
        self.assertTrue(plan.poses[1].metadata["clamped"])


if __name__ == "__main__":
    unittest.main()
