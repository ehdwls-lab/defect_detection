from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from src.integration.structured_light_runner import (
    ShellStructuredLightConfig,
    ShellStructuredLightRunner,
    StructuredLightStatus,
)


class StructuredLightRunnerTests(unittest.TestCase):
    def test_preflight_detects_broken_hardcoded_base(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            (root / "results").mkdir()
            (root / "물체검사.sh").write_text('BASE="/definitely/not/present"\n', encoding="utf-8")
            runner = ShellStructuredLightRunner(ShellStructuredLightConfig(root, root / "results"))
            issues = runner.preflight()
            self.assertTrue(any("BASE path" in issue for issue in issues))

    def test_calibration_absence_has_distinct_status(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            for name in ShellStructuredLightRunner.REQUIRED_SOURCES:
                (root / name).write_text("# portable source\n", encoding="utf-8")
            result_root = root / "results"
            result_root.mkdir()
            runner = ShellStructuredLightRunner(ShellStructuredLightConfig(
                root, result_root, python_path=Path(__import__("sys").executable),
            ))
            report = runner.preflight_report()
            self.assertTrue(report.source_ready)
            self.assertTrue(report.environment_ready)
            self.assertFalse(report.calibration_ready)
            self.assertEqual(report.overall_status, StructuredLightStatus.CALIBRATION_REQUIRED)


if __name__ == "__main__":
    unittest.main()
