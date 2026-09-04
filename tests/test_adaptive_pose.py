from __future__ import annotations

import unittest

from src.inspection.adaptive_pose import (
    adaptive_pose_for_z, combined_tilt_deg, max_tilt_for_z,
)


class AdaptivePoseTests(unittest.TestCase):
    envelope = ((17.0, 20.0), (18.0, 21.0), (19.0, 22.0), (20.0, 23.0), (21.0, 25.0))

    def test_combined_tilt_uses_surface_normal(self):
        self.assertAlmostEqual(combined_tilt_deg(25, 0), 25.0, places=6)
        self.assertAlmostEqual(combined_tilt_deg(20, 20), 27.9909, places=3)
        self.assertAlmostEqual(combined_tilt_deg(25, 25), 34.7754, places=3)

    def test_z_envelope_and_interpolation(self):
        self.assertEqual(max_tilt_for_z(17, self.envelope), 20)
        self.assertEqual(max_tilt_for_z(18, self.envelope), 21)
        self.assertEqual(max_tilt_for_z(19, self.envelope), 22)
        self.assertEqual(max_tilt_for_z(20, self.envelope), 23)
        self.assertEqual(max_tilt_for_z(21, self.envelope), 25)
        self.assertEqual(max_tilt_for_z(25, self.envelope), 25)
        self.assertEqual(max_tilt_for_z(19.5, self.envelope), 22.5)

    def test_scale_preserves_direction_and_ratio(self):
        pose = adaptive_pose_for_z(
            20, 25, 10, roll_limit_deg=25, pitch_limit_deg=25,
            envelope=self.envelope,
        )
        self.assertLess(pose.tilt_scale, 1.0)
        self.assertLessEqual(pose.combined_tilt_deg, pose.max_combined_tilt_deg + 1e-9)
        self.assertAlmostEqual(pose.applied_roll_deg / pose.applied_pitch_deg, 2.5)
        self.assertGreater(pose.applied_roll_deg, 0)
        self.assertGreater(pose.applied_pitch_deg, 0)

    def test_safe_pose_is_unchanged(self):
        pose = adaptive_pose_for_z(
            25, -20, 10, roll_limit_deg=25, pitch_limit_deg=25,
            envelope=self.envelope,
        )
        self.assertEqual((pose.applied_roll_deg, pose.applied_pitch_deg), (-20, 10))
        self.assertEqual(pose.tilt_scale, 1.0)


if __name__ == "__main__":
    unittest.main()