from __future__ import annotations

import json
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

from src.platform.motion_diagnostic import (
    MotionDiagnosticError, MotionWaitConfig, PlatformMotionDiagnostic,
)
from src.platform.serial_controller import PlatformTelemetryTimeout
from src.platform.types import PlatformTelemetry
from src.tools.read_platform_telemetry import main as read_telemetry_main
from src.tools.test_platform_motion import build_parser, run


def telemetry(*, z=0.0, roll=0.0, pitch=0.0, stable=True):
    return PlatformTelemetry(z, roll, pitch, stable, False, 0, 0, 0, 1, 1, 0.0, 0.0, 1.0)


class FakeController:
    def __init__(self, values, stale_input=()):
        self.values = list(values)
        self.stale_input = list(stale_input)
        self.writes = []
        self.connected = False
        self.discard_count = 0
        self.fresh_count = 0
        self.events = []

    def connect(self): self.connected = True
    def close(self): self.connected = False

    def read_telemetry(self, timeout):
        self.events.append("read")
        if not self.values:
            raise PlatformTelemetryTimeout("fake timeout")
        return self.values.pop(0)

    def read_fresh_telemetry(self, timeout, *, settle_s):
        self.fresh_count += 1
        self.events.append("fresh_boundary")
        return self.read_telemetry(timeout)

    def discard_stale_input(self):
        self.stale_input.clear()
        self.discard_count += 1
        self.events.append("discard")

    def move_z(self, value): self.writes.append(f"Z:{value:.2f}"); self.events.append("move_z")
    def move_orientation(self, roll, pitch): self.writes.append(f"R:{roll:.2f} P:{pitch:.2f}"); self.events.append("move_orientation")


class PostCommandBoundaryFake(FakeController):
    """Model stale packets queued after write but before the fresh boundary."""

    def __init__(self, before, stale_after_command, fresh_after_boundary):
        super().__init__([before])
        self.stale_after_command = list(stale_after_command)
        self.fresh_after_boundary = list(fresh_after_boundary)
        self.discarded_at_post_boundary = []

    def move_z(self, value):
        super().move_z(value)
        self.values.extend(self.stale_after_command)

    def read_fresh_telemetry(self, timeout, *, settle_s):
        if self.fresh_count == 1:
            self.discarded_at_post_boundary = list(self.values)
            self.values = list(self.fresh_after_boundary)
        return super().read_fresh_telemetry(timeout, settle_s=settle_s)


class DelayedMotionFake(FakeController):
    def read_telemetry(self, timeout):
        time.sleep(0.002)
        return super().read_telemetry(timeout)


class PlatformMotionDiagnosticTests(unittest.TestCase):
    def args(self, *extra):
        return build_parser().parse_args([
            "--port", "FAKE", "--timeout", "0.01",
            "--post-command-guard", "0", "--stable-samples", "1",
            "--deadband-observation", "0", "--fresh-settle", "0", *extra,
        ])

    def test_default_is_read_only(self):
        fake = FakeController([telemetry()])
        self.assertEqual(run(self.args(), controller=fake), 0)
        self.assertEqual(fake.writes, [])
        self.assertEqual(fake.fresh_count, 1)

    def test_explicit_snapshot_is_fresh_and_read_only(self):
        fake = FakeController([telemetry(z=20)])
        self.assertEqual(run(self.args("--snapshot"), controller=fake), 0)
        self.assertEqual(fake.writes, [])
        self.assertEqual(fake.fresh_count, 1)

    def test_stream_reader_snapshot_uses_one_fresh_packet(self):
        fake = FakeController([telemetry(z=20)])
        argv = [
            "read_platform_telemetry.py", "--port", "FAKE", "--snapshot",
            "--fresh-settle", "0", "--timeout", "0.01",
        ]
        with patch("sys.argv", argv), patch(
            "src.tools.read_platform_telemetry.SerialPlatformController",
            return_value=fake,
        ):
            self.assertEqual(read_telemetry_main(), 0)
        self.assertEqual(fake.fresh_count, 1)
        self.assertEqual(fake.events.count("read"), 1)
        self.assertEqual(fake.writes, [])

    def test_targets_without_execute_do_not_write(self):
        fake = FakeController([telemetry()])
        run(self.args("--z", "5", "--roll", "2"), controller=fake)
        self.assertEqual(fake.writes, [])

    def test_z_command_and_stable_wait(self):
        fake = FakeController([telemetry(), telemetry(z=2, stable=False), telemetry(z=5)])
        run(self.args("--z", "5", "--execute"), controller=fake, confirm=lambda _: True)
        self.assertEqual(fake.writes, ["Z:5.00"])
        command_index = fake.events.index("move_z")
        self.assertEqual(fake.events[command_index - 1:command_index + 2],
                         ["discard", "move_z", "fresh_boundary"])
        self.assertEqual(fake.events[command_index + 2], "read")
        self.assertEqual(fake.fresh_count, 2)

    def test_z_wait_requires_target_after_stale_stable_and_motion_samples(self):
        fake = FakeController([
            telemetry(z=0),
            telemetry(z=0, stable=True),
            telemetry(z=5, stable=False),
            telemetry(z=10, stable=False),
            telemetry(z=15, stable=True),
        ])
        diagnostic = PlatformMotionDiagnostic(
            fake, timeout_s=0.01, wait_config=MotionWaitConfig(0, 1, 0, 0),
            confirm=lambda _: True,
        )
        diagnostic.read_before()
        completed = diagnostic.execute_z(15)
        self.assertEqual(completed.z_cm, 15)
        self.assertTrue(completed.stable)
        self.assertEqual(fake.writes, ["Z:15.00"])
        self.assertEqual(
            [item["z_cm"] for item in diagnostic.log.records if item["stage"] == "during"],
            [0, 5, 10, 15],
        )

    def test_stale_stable_z_zero_does_not_complete_safe_z_command(self):
        fake = FakeController([
            telemetry(z=0),
            telemetry(z=0, stable=True),
            telemetry(z=0, stable=True),
            telemetry(z=15, stable=True),
        ])
        diagnostic = PlatformMotionDiagnostic(
            fake, timeout_s=0.01, wait_config=MotionWaitConfig(0, 1, 0, 0),
            confirm=lambda _: True,
        )
        diagnostic.read_before()
        completed = diagnostic.execute_z(15)
        self.assertEqual(completed.z_cm, 15)
        self.assertEqual(fake.events.count("read"), 4)

    def test_pose_never_writes_orientation_when_z_target_is_not_reached(self):
        fake = FakeController([
            telemetry(z=0),
            telemetry(z=0, stable=True),
            telemetry(z=5, stable=False),
            telemetry(z=10, stable=False),
            telemetry(z=14.5, stable=True),
        ])
        diagnostic = PlatformMotionDiagnostic(
            fake, timeout_s=0.002, wait_config=MotionWaitConfig(0, 1, 0, 0),
            confirm=lambda _: True,
        )
        diagnostic.read_before()
        with self.assertRaises(PlatformTelemetryTimeout):
            diagnostic.execute_pose(
                safe_z_cm=15, roll_deg=5, pitch_deg=-3, ack_safe_height=True,
            )
        self.assertEqual(fake.writes, ["Z:15.00"])

    def test_pose_orders_completed_safe_z_before_orientation(self):
        fake = FakeController([
            telemetry(z=0),
            telemetry(z=0, stable=True),
            telemetry(z=5, stable=False),
            telemetry(z=10, stable=False),
            telemetry(z=15, stable=True),
            telemetry(z=15, roll=5, pitch=-3, stable=True),
        ])
        diagnostic = PlatformMotionDiagnostic(
            fake, timeout_s=0.01, wait_config=MotionWaitConfig(0, 1, 0, 0),
            confirm=lambda _: True,
        )
        diagnostic.read_before()
        result = diagnostic.execute_pose(
            safe_z_cm=15, roll_deg=5, pitch_deg=-3, ack_safe_height=True,
        )
        self.assertEqual(fake.writes, ["Z:15.00", "R:5.00 P:-3.00"])
        self.assertEqual(result.z_cm, 15)
        command_records = [
            item["command"] for item in diagnostic.log.records
            if item["stage"] == "command"
        ]
        self.assertEqual(command_records, ["Z:15.00", "R:5.00 P:-3.00"])

    def test_log_records_before_command_during_after(self):
        fake = FakeController([telemetry(), telemetry(z=3, stable=False), telemetry(z=5)])
        diagnostic = PlatformMotionDiagnostic(
            fake, timeout_s=0.01, wait_config=MotionWaitConfig(0, 1, 0, 0),
            confirm=lambda _: True,
        )
        diagnostic.read_before()
        diagnostic.execute_z(5)
        self.assertEqual([item["stage"] for item in diagnostic.log.records],
                         ["before", "command", "during", "during", "after_z"])
        self.assertEqual(diagnostic.log.records[1]["command"], "Z:5.00")

    def test_roll_and_pitch_commands(self):
        for option, expected in ((["--roll", "2"], "R:2.00 P:0.00"),
                                 (["--pitch", "2"], "R:0.00 P:2.00")):
            with self.subTest(option=option):
                fake = FakeController([telemetry(), telemetry()])
                run(self.args(*option, "--execute", "--ack-safe-height"),
                    controller=fake, confirm=lambda _: True)
                self.assertEqual(fake.writes, [expected])

    def test_orientation_requires_safe_height_ack(self):
        fake = FakeController([telemetry()])
        with self.assertRaises(MotionDiagnosticError):
            run(self.args("--roll", "2", "--execute"), controller=fake, confirm=lambda _: True)
        self.assertEqual(fake.writes, [])

    def test_confirmation_rejection_writes_nothing(self):
        fake = FakeController([telemetry()])
        with self.assertRaises(MotionDiagnosticError):
            run(self.args("--z", "5", "--execute"), controller=fake, confirm=lambda _: False)
        self.assertEqual(fake.writes, [])

    def test_timeout_sends_no_additional_command(self):
        fake = FakeController([telemetry()])
        diagnostic = PlatformMotionDiagnostic(
            fake, timeout_s=0.001, wait_config=MotionWaitConfig(0, 2, 1.0, 0),
            confirm=lambda _: True,
        )
        diagnostic.read_before()
        with self.assertRaises(PlatformTelemetryTimeout):
            diagnostic.execute_z(5)
        self.assertEqual(fake.writes, ["Z:5.00"])

    def test_motion_can_use_extended_platform_timeout(self):
        values = [telemetry(), telemetry(z=12.01, stable=False), telemetry(z=20)]
        fake = DelayedMotionFake(values)
        diagnostic = PlatformMotionDiagnostic(
            fake, timeout_s=0.02, wait_config=MotionWaitConfig(0, 1, 0, 0),
            confirm=lambda _: True,
        )
        diagnostic.read_before()
        result = diagnostic.execute_z(20)
        self.assertTrue(result.stable)
        self.assertEqual(fake.writes, ["Z:20.00"])

    def test_pre_command_stable_buffer_is_discarded(self):
        fake = FakeController(
            [telemetry(), telemetry(z=2, stable=False), telemetry(z=5), telemetry(z=5)],
            stale_input=[telemetry(stable=True)],
        )
        diagnostic = PlatformMotionDiagnostic(
            fake, timeout_s=0.01, wait_config=MotionWaitConfig(0, 2, 1.0, 0),
            confirm=lambda _: True,
        )
        diagnostic.read_before()
        result = diagnostic.execute_z(5)
        self.assertTrue(result.stable)
        self.assertEqual(fake.discard_count, 1)
        self.assertEqual(fake.fresh_count, 2)
        self.assertEqual(fake.stale_input, [])

    def test_post_command_fresh_boundary_discards_queued_stable_samples(self):
        stale = [telemetry(z=28.77), telemetry(z=28.39), telemetry(z=20)]
        fake = PostCommandBoundaryFake(
            telemetry(z=20),
            stale_after_command=stale,
            fresh_after_boundary=[
                telemetry(z=19, stable=False), telemetry(z=20), telemetry(z=20),
            ],
        )
        diagnostic = PlatformMotionDiagnostic(
            fake, timeout_s=0.01, wait_config=MotionWaitConfig(0, 2, 1.0, 0),
            confirm=lambda _: True,
        )
        diagnostic.read_before()
        result = diagnostic.execute_z(20)
        self.assertEqual(fake.discarded_at_post_boundary, stale)
        self.assertEqual(fake.fresh_count, 2)
        self.assertTrue(result.stable)
        self.assertIn(False, [
            item["stable"] for item in diagnostic.log.records
            if item["stage"] == "during"
        ])

    def test_stale_stable_samples_do_not_complete_before_motion(self):
        fake = FakeController([
            telemetry(),
            telemetry(stable=True), telemetry(stable=True), telemetry(stable=True),
            telemetry(z=2, stable=False), telemetry(z=5), telemetry(z=5),
        ])
        diagnostic = PlatformMotionDiagnostic(
            fake, timeout_s=0.01, wait_config=MotionWaitConfig(0, 2, 1.0, 0),
            confirm=lambda _: True,
        )
        diagnostic.read_before()
        diagnostic.execute_z(5)
        stages = [item["stage"] for item in diagnostic.log.records]
        self.assertEqual(stages.count("after_z"), 1)
        self.assertIn(False, [item["stable"] for item in diagnostic.log.records if item["stage"] == "during"])

    def test_stable_only_deadband_path_waits_for_observation_window(self):
        fake = FakeController([telemetry()] + [telemetry(z=5)] * 10)
        diagnostic = PlatformMotionDiagnostic(
            fake, timeout_s=0.001, wait_config=MotionWaitConfig(0, 2, 1.0, 0),
            confirm=lambda _: True,
        )
        diagnostic.read_before()
        with self.assertRaises(PlatformTelemetryTimeout):
            diagnostic.execute_z(5)

    def test_invalid_numeric_is_rejected(self):
        with self.assertRaises(SystemExit):
            self.args("--z", "nan")

    def test_fresh_settle_must_leave_post_command_timeout_budget(self):
        fake = FakeController([telemetry()])
        with self.assertRaisesRegex(ValueError, "less than timeout_s"):
            PlatformMotionDiagnostic(
                fake, timeout_s=0.1,
                wait_config=MotionWaitConfig(fresh_read_settle_s=0.1),
            )
        self.assertEqual(fake.writes, [])

    def test_pose_json_dry_run_and_legacy_z_never_sent(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            ply = root / "input.ply"
            ply.write_text("ply", encoding="ascii")
            pose_json = root / "pose.json"
            pose_json.write_text(json.dumps({
                "schema_version": "structured_light_pose_v1", "input_ply": str(ply),
                "planes": [{"plane_name": "Shot 1", "dominant": True, "points_count": 10,
                            "roll_deg": -6.8, "pitch_deg": 6.4,
                            "metric_pose": {"source": "orbbec_depth", "physical_metric": True,
                                            "status": "REACHABLE", "reachable": True,
                                            "roll_deg": -6.8, "pitch_deg": 6.4,
                                            "target_platform_roll_deg": -6.8,
                                            "target_platform_pitch_deg": 6.4,
                                            "calibration_id": "camera_platform_rp_20260903",
                                            "depth_points_count": 10, "depth_coverage": 1.0},
                            "legacy_relative_z": {"value_cm": 20, "metric": False,
                                                  "stm_compatible": False}}],
                "stm_z_command_allowed": False,
            }), encoding="utf-8")
            fake = FakeController([telemetry()])
            run(self.args("--pose-json", str(pose_json), "--dry-run"), controller=fake)
            self.assertEqual(fake.writes, [])

    def test_execute_pose_uses_explicit_z_then_orientation(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            ply = root / "input.ply"
            ply.write_text("ply", encoding="ascii")
            pose_json = root / "pose.json"
            pose_json.write_text(json.dumps({
                "input_ply": str(ply), "planes": [{"plane_name": "one", "points_count": 1,
                "roll_deg": -2, "pitch_deg": 3,
                "metric_pose": {"source": "orbbec_depth", "physical_metric": True,
                                "status": "REACHABLE", "reachable": True,
                                "roll_deg": -2, "pitch_deg": 3,
                                "target_platform_roll_deg": -2,
                                "target_platform_pitch_deg": 3,
                                "calibration_id": "camera_platform_rp_20260903",
                                "depth_points_count": 1, "depth_coverage": 1.0}}],
                "stm_z_command_allowed": False,
            }), encoding="utf-8")
            fake = FakeController([telemetry(), telemetry(z=7), telemetry(z=7, roll=-2, pitch=3)])
            run(self.args("--pose-json", str(pose_json), "--z", "7", "--execute-pose",
                          "--ack-safe-height"), controller=fake, confirm=lambda _: True)
            self.assertEqual(fake.writes, ["Z:7.00", "R:-2.00 P:3.00"])


if __name__ == "__main__":
    unittest.main()
