from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from src.integration.structured_light_runner import ShellStructuredLightConfig, ShellStructuredLightRunner


class StructuredLightRunnerTests(unittest.TestCase):
    def test_preflight_detects_broken_hardcoded_base(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            (root / "results").mkdir()
            (root / "물체검사.sh").write_text('BASE="/definitely/not/present"\n', encoding="utf-8")
            runner = ShellStructuredLightRunner(ShellStructuredLightConfig(root, root / "results"))
            issues = runner.preflight()
            self.assertTrue(any("BASE path" in issue for issue in issues))


if __name__ == "__main__":
    unittest.main()
