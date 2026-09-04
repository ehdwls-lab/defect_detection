from __future__ import annotations

import unittest

from src.integration.platform_alignment import plan_platform_alignment


class PlatformAlignmentTests(unittest.TestCase):
    def test_full_target_is_commanded_without_change(self):
        result = plan_platform_alignment(
            camera_roll_deg=10, camera_pitch_deg=5,
            current_platform_roll_deg=0, current_platform_pitch_deg=0,
            desired_target_roll_deg=9.955565, desired_target_pitch_deg=-4.400105,
        )
        self.assertEqual(result["alignment_mode"], "FULL")
        self.assertAlmostEqual(result["commanded_target_roll_deg"], 9.955565)
        self.assertAlmostEqual(result["commanded_target_pitch_deg"], -4.400105)

    def test_partial_target_is_box_constrained_best_effort_not_full_clamp(self):
        result = plan_platform_alignment(
            camera_roll_deg=-36.635014, camera_pitch_deg=2.527686,
            current_platform_roll_deg=0, current_platform_pitch_deg=.01,
            desired_target_roll_deg=-35.446831, desired_target_pitch_deg=-3.746247,
        )
        self.assertEqual(result["alignment_mode"], "PARTIAL")
        self.assertLessEqual(abs(result["commanded_target_roll_deg"]), 30.0)
        self.assertLessEqual(abs(result["commanded_target_pitch_deg"]), 30.0)
        self.assertNotEqual(
            (result["commanded_target_roll_deg"], result["commanded_target_pitch_deg"]),
            (-30.0, -3.746247),
        )
        self.assertGreater(result["predicted_residual_angle_deg"], 0.0)
