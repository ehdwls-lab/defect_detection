from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np


SUBSYSTEM_ROOT = Path(__file__).resolve().parents[1] / "서영 파트 파일"
if str(SUBSYSTEM_ROOT) not in sys.path:
    sys.path.insert(0, str(SUBSYSTEM_ROOT))

from structured_light_projector import (  # noqa: E402
    PRODUCTION_PERIOD,
    parse_xrandr_monitors,
    production_phase_patterns,
    select_projector_monitor,
)


class StructuredLightProjectorTests(unittest.TestCase):
    def test_xwayland_secondary_selection_matches_production_policy(self):
        monitors = parse_xrandr_monitors(
            "XWAYLAND1 connected primary 1920x1200+0+0\n"
            "XWAYLAND2 connected 1920x1080+1920+0\n"
        )
        selected = select_projector_monitor(monitors)
        self.assertIsNotNone(selected)
        self.assertEqual(selected["name"], "XWAYLAND2")
        self.assertEqual((selected["x"], selected["y"]), (1920, 0))

    def test_production_four_phase_arrays_have_quarter_period_shift(self):
        height = PRODUCTION_PERIOD * 2
        patterns = production_phase_patterns(64, height)
        self.assertEqual(set(patterns), {"000", "090", "180", "270"})
        for image in patterns.values():
            self.assertEqual(image.shape, (height, 64, 3))
            self.assertLess(int(image.min()), int(image.max()))
        values = list(patterns.values())
        for index, left in enumerate(values):
            for right in values[index + 1:]:
                self.assertFalse(np.array_equal(left, right))
        quarter = PRODUCTION_PERIOD // 4
        # Independently evaluated float32 cosine values can differ by one
        # uint8 level at truncation boundaries while preserving the shift.
        self.assertTrue(np.allclose(
            patterns["090"], np.roll(patterns["000"], -quarter, axis=0), atol=1,
        ))


if __name__ == "__main__":
    unittest.main()
