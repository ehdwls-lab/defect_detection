from __future__ import annotations

import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

from src.tools.test_real_pose_planner import print_plan_diagnostic


class RealPosePlannerDiagnosticTests(unittest.TestCase):
    def test_zero_reachable_poses_prints_rejection_without_index_error(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            ply = root / "input.ply"
            ply.write_text("ply", encoding="ascii")
            pose = root / "pose.json"
            pose.write_text(json.dumps({
                "input_ply": str(ply), "metric_physical_plane_count": 1,
                "planes": [{
                    "plane_name": "Physical Plane 1", "points_count": 100,
                    "roll_deg": 70.0, "pitch_deg": 0.0,
                    "metric_pose": {
                        "physical_metric": True, "status": "UNREACHABLE",
                        "reachable": False,
                        "reject_reason": "target platform angle exceeds ±30 deg",
                        "camera_roll_deg": 36.0, "camera_pitch_deg": 0.0,
                    },
                }],
            }), encoding="utf-8")
            output = io.StringIO()
            with redirect_stdout(output):
                result = print_plan_diagnostic(pose)
            self.assertEqual(result, 0)
            self.assertIn("Total physical planes = 1", output.getvalue())
            self.assertIn("Reachable poses = 0", output.getvalue())
            self.assertIn("exceeds ±30", output.getvalue())
            self.assertIn("Selected inspection pose = NONE", output.getvalue())


if __name__ == "__main__":
    unittest.main()
