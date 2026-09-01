from __future__ import annotations

import unittest

from src.inspection.automatic_z_search import AutomaticZSearch
from src.inspection.z_search_types import InspectionQualitySample


def sample(z: float, passed: bool, score: float | None) -> InspectionQualitySample:
    return InspectionQualitySample(z, 0.8, 0.8, 1.0, 10000, 9000, 20, False,
                                   gate_passed=passed, quality_score=score)


class AutomaticZSearchTests(unittest.TestCase):
    def test_selects_global_best_not_first_pass(self):
        samples = [sample(20, False, None), sample(20.5, True, .68), sample(21, True, .81),
                   sample(21.5, True, .87), sample(22, True, .74)]
        result = AutomaticZSearch.select_best(pose_id="p1", samples=samples)
        self.assertTrue(result.success)
        self.assertEqual(result.best_z_cm, 21.5)
        self.assertEqual(len(result.samples), 5)

    def test_no_valid_candidate(self):
        result = AutomaticZSearch.select_best(pose_id="p1", samples=[sample(20, False, None)])
        self.assertFalse(result.success)
        self.assertEqual(result.failure_reason, "NoValidInspectionZ")

    def test_tie_prefers_lower_z(self):
        result = AutomaticZSearch.select_best(
            pose_id="p1", samples=[sample(22, True, .8), sample(21, True, .8)]
        )
        self.assertEqual(result.best_z_cm, 21)


if __name__ == "__main__":
    unittest.main()
