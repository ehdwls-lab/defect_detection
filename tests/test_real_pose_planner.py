from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from src.integration.coordinate_contract import (
    CloudType, DEFAULT_COORDINATE_CONVENTION, PointCloudMetadata, StructuredLightResult,
)
from src.integration.real_pose_planner import PoseJSONError, RealPosePlanner, parse_pose_json


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
                 "legacy_relative_z": {"value_cm": 20.0, "metric": False, "stm_compatible": False}},
                {"plane_name": "Shot 1", "dominant": True, "points_count": 80,
                 "roll_deg": -6.83, "pitch_deg": 6.44, "raw_roll_deg": -6.825, "raw_pitch_deg": 6.438,
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
        self.assertFalse(plan.metadata["z_provided"])
        self.assertFalse(plan.metadata["platform_motion_allowed"])
        self.assertFalse(plan.metadata["automatic_z_allowed"])

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

    def test_multi_plane_is_explicitly_reserved(self):
        with self.assertRaises(NotImplementedError):
            RealPosePlanner("multi_plane").plan(self.write())


if __name__ == "__main__":
    unittest.main()
