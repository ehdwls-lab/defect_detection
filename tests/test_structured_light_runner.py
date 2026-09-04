from __future__ import annotations

import tempfile
import unittest
import os
import signal
import subprocess
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, call, patch
import json
import sys

from src.integration.structured_light_runner import (
    ShellStructuredLightConfig,
    ShellStructuredLightRunner,
    StructuredLightPreflightReport,
    StructuredLightStatus,
)


class StructuredLightRunnerTests(unittest.TestCase):
    @staticmethod
    def ready_report():
        return StructuredLightPreflightReport(
            StructuredLightStatus.READY, True, True, True,
            python_path=Path(sys.executable),
        )

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

    def test_scan_does_not_reuse_an_unchanged_old_run(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            results = root / "results"
            (results / "촬영_old").mkdir(parents=True)
            runner = ShellStructuredLightRunner(ShellStructuredLightConfig(root, results))
            with patch.object(runner, "preflight_report", return_value=self.ready_report()), patch.object(
                runner, "_execute_script",
                return_value=SimpleNamespace(returncode=0, stdout="", stderr=""),
            ):
                with self.assertRaisesRegex(RuntimeError, "no new or updated"):
                    runner._run_scan()

    def test_current_scan_exposes_pose_json_and_forwards_monitor(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            results = root / "results"
            results.mkdir()
            runner = ShellStructuredLightRunner(ShellStructuredLightConfig(
                root, results, projector_monitor="HDMI-0",
            ))
            captured_env = {}

            def fake_run(_root, env):
                captured_env.update(env)
                current = results / "촬영_new"
                current.mkdir()
                (current / "FINAL_DC_MASK_PHASE_z30_SIGN_PLUS_pose.json").write_text(
                    "{}", encoding="utf-8",
                )
                return SimpleNamespace(returncode=0, stdout="ok", stderr="")

            with patch.object(runner, "preflight_report", return_value=self.ready_report()), patch.object(
                runner, "_execute_script", side_effect=fake_run,
            ):
                info = runner._run_scan()
            self.assertEqual(captured_env["STRUCTURED_LIGHT_MONITOR"], "HDMI-0")
            self.assertEqual(captured_env["STRUCTURED_LIGHT_RESULT_ROOT"], str(results.resolve()))
            self.assertEqual(info.result_directory.name, "촬영_new")
            self.assertEqual(info.pose_json_path.name, "FINAL_DC_MASK_PHASE_z30_SIGN_PLUS_pose.json")
            manifest = json.loads(info.manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(manifest["artifacts"]["pose_json"], str(info.pose_json_path))

    def test_manifest_phase_artifact_excludes_floor_and_segmented(self):
        with tempfile.TemporaryDirectory() as raw:
            directory = Path(raw)
            phase = directory / "FINAL_DC_MASK_PHASE_z30_SIGN_PLUS.ply"
            phase.write_text("phase", encoding="ascii")
            (directory / "FINAL_DC_MASK_PHASE_z30_SIGN_PLUS_WITH_FLOOR.ply").write_text(
                "floor", encoding="ascii",
            )
            (directory / "FINAL_DC_MASK_PHASE_z30_SIGN_PLUS_dominant_plane_segmented.ply").write_text(
                "segmented", encoding="ascii",
            )
            pose = directory / "FINAL_DC_MASK_PHASE_z30_SIGN_PLUS_pose.json"
            pose.write_text("{}", encoding="utf-8")
            runner = ShellStructuredLightRunner(ShellStructuredLightConfig(directory, directory))
            manifest_path = runner._write_manifest(
                directory, "start", "finish", 0, pose_json_path=pose,
            )
            artifacts = json.loads(manifest_path.read_text(encoding="utf-8"))["artifacts"]
            self.assertEqual(artifacts["phase_object_only_ply"], str(phase.resolve()))
            self.assertEqual(artifacts["pose_json"], str(pose.resolve()))

    def test_updated_run_cannot_reuse_unchanged_pose_from_same_second(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            results = root / "results"
            current = results / "촬영_same_second"
            current.mkdir(parents=True)
            pose = current / "FINAL_DC_MASK_PHASE_z30_SIGN_PLUS_pose.json"
            pose.write_text("{}", encoding="utf-8")
            runner = ShellStructuredLightRunner(ShellStructuredLightConfig(root, results))

            def update_directory(_root, _env):
                marker = current / "new_scan_marker.txt"
                marker.write_text("new", encoding="utf-8")
                previous = current.stat().st_mtime_ns
                os.utime(current, ns=(previous + 1_000_000_000, previous + 1_000_000_000))
                return SimpleNamespace(returncode=0, stdout="ok", stderr="")

            with patch.object(runner, "preflight_report", return_value=self.ready_report()), patch.object(
                runner, "_execute_script", side_effect=update_directory,
            ):
                with self.assertRaisesRegex(RuntimeError, "created or updated"):
                    runner._run_scan()
            self.assertTrue((results / "_diagnostic" / "failure.json").is_file())

    def test_timeout_terminates_the_whole_process_group(self):
        runner = ShellStructuredLightRunner(ShellStructuredLightConfig(
            Path("/tmp/source"), Path("/tmp/result"), timeout_sec=1.0,
            termination_grace_sec=0.5,
        ))
        process = Mock()
        process.pid = 4321
        process.args = ["bash", "scan.sh"]
        process.returncode = -signal.SIGTERM
        # The leader exits after SIGTERM, but a descendant keeps the inherited
        # pipes open until the whole process group receives SIGKILL.
        process.poll.return_value = None
        process.communicate.side_effect = [
            subprocess.TimeoutExpired(process.args, 1.0),
            subprocess.TimeoutExpired(process.args, 0.5),
            ("partial stdout", "partial stderr"),
        ]
        with patch(
            "src.integration.structured_light_runner.subprocess.Popen",
            return_value=process,
        ) as popen, patch(
            "src.integration.structured_light_runner.os.killpg",
        ) as killpg:
            with self.assertRaisesRegex(RuntimeError, "timed out") as raised:
                runner._execute_script(Path("/tmp/source"), {})
        self.assertEqual(raised.exception.stdout, "partial stdout")
        popen.assert_called_once()
        self.assertTrue(popen.call_args.kwargs["start_new_session"])
        self.assertEqual(killpg.call_args_list, [
            call(4321, signal.SIGTERM),
            call(4321, signal.SIGKILL),
        ])


if __name__ == "__main__":
    unittest.main()
