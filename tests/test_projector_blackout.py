from __future__ import annotations

import unittest
from pathlib import Path

from src.integration.projector_controller import FourPhaseProjectorSession, ProjectorState
from src.integration.structured_light_runner import ShellStructuredLightConfig, ShellStructuredLightRunner


class FakeProjector:
    def __init__(self):
        self.state = ProjectorState.CLOSED
        self.events = []

    def open(self):
        self.state = ProjectorState.BLACK
        self.events.append("OPEN_BLACK")

    def show_black(self):
        self.state = ProjectorState.BLACK
        self.events.append("BLACK")

    def show_phase(self, name):
        self.state = ProjectorState.PHASE
        self.events.append(f"PHASE_{name}")

    def close(self):
        self.show_black()
        self.state = ProjectorState.CLOSED


class ProjectorBlackoutTests(unittest.TestCase):
    def test_projector_starts_black_and_scan_ends_black(self):
        projector = FakeProjector()
        projector.open()
        result = FourPhaseProjectorSession(projector).capture(lambda name: name)
        self.assertEqual(list(result), ["000", "090", "180", "270"])
        self.assertEqual(projector.state, ProjectorState.BLACK)
        self.assertEqual(projector.events, [
            "OPEN_BLACK", "BLACK", "PHASE_000", "PHASE_090",
            "PHASE_180", "PHASE_270", "BLACK",
        ])

    def test_scan_exception_ends_black(self):
        projector = FakeProjector(); projector.open()
        def fail(name):
            if name == "180":
                raise RuntimeError("capture failed")
            return name
        with self.assertRaises(RuntimeError):
            FourPhaseProjectorSession(projector).capture(fail)
        self.assertEqual(projector.state, ProjectorState.BLACK)

    def test_shell_scan_wrapper_keeps_window_and_restores_black(self):
        projector = FakeProjector()
        runner = ShellStructuredLightRunner(
            ShellStructuredLightConfig(Path("."), Path(".")),
            projector=projector,
        )
        runner._run_scan = lambda: "ok"
        self.assertEqual(runner.run_scan(), "ok")
        self.assertEqual(projector.state, ProjectorState.BLACK)
        self.assertNotEqual(projector.state, ProjectorState.CLOSED)

    def test_shell_scan_wrapper_exception_restores_black(self):
        projector = FakeProjector()
        runner = ShellStructuredLightRunner(
            ShellStructuredLightConfig(Path("."), Path(".")),
            projector=projector,
        )
        def fail(): raise RuntimeError("scan")
        runner._run_scan = fail
        with self.assertRaises(RuntimeError):
            runner.run_scan()
        self.assertEqual(projector.state, ProjectorState.BLACK)


if __name__ == "__main__":
    unittest.main()
