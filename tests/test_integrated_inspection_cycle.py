from __future__ import annotations

import json
import io
import tempfile
import unittest
from contextlib import redirect_stdout
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from src.camera.controller import RGBDepthFrame
from src.config import InspectionConfig
from src.core.depth_contour_roi import DepthExternalContourROIResult
from src.core.surface_geometry import SurfaceGeometryResult
from src.inspection.hardware_z_search import (
    HardwareAutomaticZSearch,
    HardwareZCandidateResult,
    HardwareZSearchResult,
    HardwareZSearchConfig,
)
from src.inspection.adaptive_pose import adaptive_pose_for_z
from src.integration.integrated_inspection_cycle import (
    IntegratedCycleError,
    IntegratedCycleResult,
    IntegratedInspectionCycle,
    resolve_current_pose_json,
)
from src.integration.platform_limits import (
    ORIENTATION_SAFE_Z_MIN_CM,
    PLATFORM_RP_LIMIT_DEG,
)
from src.integration.hybrid_inspection_roi import HybridInspectionROIResult
from src.integration.metric_pose import CameraIntrinsics
from src.integration.projector_controller import ProjectorState
from src.integration.platform_pose_calibration import (
    CAMERA_PLATFORM_RP_20260903,
    apply_calibration_to_pose_json,
)
from src.integration.platform_alignment import apply_alignment_policy_to_pose_json
from src.integration.inspection_failures import (
    AnomalyInputDataError,
    FinalCaptureQualityError,
    FinalRGBCaptureError,
)
from src.integration.final_capture import (
    FinalCaptureAttemptDiagnostic,
    GeometryReadyFinalCapture,
    acquire_geometry_ready_final_frame,
    acquire_warmed_final_rgb_frame,
    save_final_geometry_capture,
    save_final_rgb_capture,
)
from src.integration.real_pose_planner import RealPosePlanner
from src.integration.structured_light_runner import StructuredLightRunInfo, StructuredLightStatus
from src.platform.motion_diagnostic import DiagnosticZMover, TelemetryLog
from src.platform.types import PlatformTelemetry
from src.tools.test_integrated_inspection_cycle import build_parser, run as run_cli
from src.anomaly.detector import AnomalyResult, ProductionAnomalyConfig, ProductionAnomalyDetector, AnomalyModelNotReadyError


def telemetry(*, z=20.0, roll=0.0, pitch=0.0, stable=True,
              homing=False) -> PlatformTelemetry:
    return PlatformTelemetry(
        z, roll, pitch, stable, homing, 0, 0, 0, 1, 1, 0.0, 0.0, 1.0,
    )


def candidate(z: float, accepted: bool) -> HardwareZCandidateResult:
    return HardwareZCandidateResult(
        z, -6.8, 6.4, None, None, 500.0, .9, .9, .1,
        .01, .01, 10.0, 10.0, 0.0, None, accepted,
        () if accepted else ("not_ready",), readiness_pass=accepted,
        readiness_frames=4, plane_inlier_ratio=.8, plane_residual=1.0,
        surface_patch_count=3, object_area_px=1000, surface_area_px=600,
        surface_ratio=.6, usable_patch_count=3,
    )


def assert_subsequence(test: unittest.TestCase, events: list[str], expected: list[str]) -> None:
    position = 0
    for item in expected:
        try:
            position = events.index(item, position) + 1
        except ValueError:
            test.fail(f"event {item!r} missing after index {position}; events={events}")


class FakeProjector:
    def __init__(self, events):
        self.events = events
        self.state = ProjectorState.CLOSED

    def open(self):
        self.state = ProjectorState.BLACK
        self.events.append("PROJECTOR_OPEN_BLACK")

    def show_black(self):
        if self.state is ProjectorState.CLOSED:
            raise RuntimeError("projector closed")
        self.state = ProjectorState.BLACK
        self.events.append("BLACK")

    def show_phase(self, name):
        self.state = ProjectorState.PHASE
        self.events.append(f"PHASE_{name}")

    def close(self):
        if self.state is not ProjectorState.CLOSED:
            self.show_black()
        self.state = ProjectorState.CLOSED
        self.events.append("PROJECTOR_CLOSE")


class FakeConveyor:
    def __init__(self, events, *, fail=False):
        self.events, self.fail = events, fail
        self.move_out_calls = 0
        self.config = type("Config", (), {
            "exit_direction": "F", "exit_steps": 10000,
        })()

    def connect(self): self.events.append("CONVEYOR_CONNECT")
    def move_to_inspection(self): self.events.append("CONVEYOR_F")
    def wait_until_stopped(self):
        self.events.append("CONVEYOR_WAIT")
        if self.fail:
            raise RuntimeError("conveyor failed")
        self.events.append("CONVEYOR_TARGET_REACHED")
    def move_out(self): self.move_out_calls += 1; self.events.append("CONVEYOR_OUT")
    def close(self): self.events.append("CONVEYOR_CLOSE")


class FakeStructuredRunner:
    def __init__(self, events, projector, run_dir, pose_json, *, fail=False):
        self.events, self.projector = events, projector
        self.run_dir, self.pose_json, self.fail = run_dir, pose_json, fail
        self.calls = 0

    def run_scan(self):
        self.calls += 1
        self.events.append("STRUCTURED_SCAN")
        try:
            for phase in ("000", "090", "180", "270"):
                self.projector.show_phase(phase)
                if self.fail and phase == "180":
                    raise RuntimeError("structured-light failed")
        finally:
            self.projector.show_black()
        return StructuredLightRunInfo(
            "scan", self.run_dir, stdout="scan stdout", stderr="",
            pose_json_path=self.pose_json,
        )


class FakePlatform:
    def __init__(self, events):
        self.events = events
        self.connected = False

    def connect(self): self.connected = True; self.events.append("PLATFORM_CONNECT")
    def close(self): self.connected = False; self.events.append("PLATFORM_CLOSE")


class FakeMotion:
    def __init__(self, events, projector, *, initial_telemetry,
                 fail_orientation_call=None, fail_z_call=None):
        self.events, self.projector = events, projector
        self.timeout_s = 1.0
        self.log = TelemetryLog()
        self.z_commands = []
        self.orientation_commands = []
        self.orientation_z_cm = []
        self.current = initial_telemetry
        self.fail_orientation_call = fail_orientation_call
        self.fail_z_call = fail_z_call

    def read_before(self):
        self.events.append("PLATFORM_FRESH_READ")
        self.log.add("before", self.current)
        return self.current

    def execute_z(self, z):
        assert self.projector.state is ProjectorState.BLACK
        self.z_commands.append(z)
        self.events.append(f"Z_{z:g}")
        self.log.add_command(f"Z:{z:.2f}")
        if len(self.z_commands) == self.fail_z_call:
            raise RuntimeError("Z stable timeout")
        self.current = telemetry(
            z=z, roll=self.current.roll_deg, pitch=self.current.pitch_deg,
        )
        self.log.add("after_z", self.current)
        self.events.append(f"Z_SETTLED_{z:g}")
        return self.current

    def execute_orientation(self, *, roll_deg, pitch_deg, before, ack_safe_height):
        assert ack_safe_height
        assert self.projector.state is ProjectorState.BLACK
        self.orientation_commands.append((roll_deg, pitch_deg))
        self.orientation_z_cm.append(float(before.z_cm))
        self.events.append(f"RP_{roll_deg:.1f}_{pitch_deg:.1f}")
        self.log.add_command(f"R:{roll_deg:.2f} P:{pitch_deg:.2f}")
        if len(self.orientation_commands) == self.fail_orientation_call:
            raise RuntimeError("orientation stable timeout")
        self.current = telemetry(
            z=before.z_cm, roll=roll_deg, pitch=pitch_deg,
        )
        self.log.add("after_orientation", self.current)
        return self.current


class FakeCamera:
    def __init__(self, events, projector, *, fail_start=False):
        self.events, self.projector = events, projector
        self.fail_start = fail_start
        self.started = False
        self.captures = 0

    def start(self):
        self.events.append("CAMERA_START")
        if self.fail_start:
            raise RuntimeError("camera start failed")
        self.started = True
    def capture(self):
        assert self.projector.state is ProjectorState.BLACK
        self.captures += 1
        self.events.append("CAPTURE")
        depth = np.full((128, 128), 500, dtype=np.float32)
        depth[36:92, 36:92] = 450
        return RGBDepthFrame(
            np.zeros((128, 128, 3), dtype=np.uint8), depth, 1.0,
        )
    def color_intrinsics(self, width, height):
        return CameraIntrinsics(
            200.0, 200.0, width / 2, height / 2,
            width, height, "synthetic", "color",
        )
    def close(self): self.started = False; self.events.append("CAMERA_CLOSE")


class FakeLighting:
    def __init__(self, events, fail_on=None):
        self.events = events
        self.fail_on = fail_on
        self.connected = False
        self.on = False

    def connect(self):
        self.events.append("LIGHTING_CONNECT")
        if self.fail_on == "connect":
            raise RuntimeError("lighting connect failed")
        self.connected = True

    def inspection_on(self):
        self.events.append("LIGHTING_ON")
        if self.fail_on == "on":
            raise RuntimeError("lighting on failed")
        self.on = True

    def inspection_off(self):
        self.events.append("LIGHTING_OFF")
        if self.fail_on == "off":
            raise RuntimeError("lighting off failed")
        self.on = False

    def projector_cover_open(self): self.events.append("COVER_OPEN")
    def projector_cover_close(self): self.events.append("COVER_CLOSE")
    def projector_cover_cleanup(self): self.events.append("COVER_CLEANUP")

    def close(self): self.connected = False; self.events.append("LIGHTING_CLOSE")


class FakeAnomalyDetector:
    def __init__(self, events, failures=None, classifications=None):
        self.events = events; self.frames = []; self.kwargs = []
        self.failures = list(failures or [])
        self.classifications = list(classifications or [])
    def inspect_frame(self, frame, **kwargs):
        self.events.append("ANOMALY_INFER")
        self.frames.append(frame)
        self.kwargs.append(kwargs)
        if self.failures:
            failure = self.failures.pop(0)
            if failure is not None:
                raise failure
        classification = self.classifications.pop(0) if self.classifications else "NORMAL"
        return AnomalyResult(
            status="ANOMALY_COMPLETE", is_mock=False, score=.1,
            threshold=.2, classification=classification, heatmap_path="heatmap.png",
        )


class FakeEvaluator:
    def __init__(self, events, outcomes):
        self.events, self.outcomes = events, dict(outcomes)

    def evaluate(self, frame, *, z_command, roll, pitch, rgb_path=None, depth_path=None):
        self.events.append(f"EVALUATE_{z_command:g}")
        outcome = self.outcomes[z_command]
        if isinstance(outcome, list):
            outcome = outcome.pop(0)
        return candidate(z_command, outcome)


class IntegratedInspectionCycleTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.scan = self.root / "scan" / "촬영_1"
        self.scan.mkdir(parents=True)
        self.ply = self.scan / "FINAL_DC_MASK_PHASE_z30_SIGN_PLUS.ply"
        self.ply.write_text(
            "ply\nformat ascii 1.0\nelement vertex 1\nproperty float x\n"
            "property float y\nproperty float z\nend_header\n0 0 0\n",
            encoding="ascii",
        )
        self.pose_json = self.scan / "FINAL_DC_MASK_PHASE_z30_SIGN_PLUS_pose.json"
        self.write_pose()

    def tearDown(self): self.temp.cleanup()

    def write_pose(self, *, malformed=False, input_ply=None):
        if malformed:
            self.pose_json.write_text("{broken", encoding="utf-8")
            return
        self.pose_json.write_text(json.dumps({
            "schema_version": "structured_light_pose_v1",
            "input_ply": str(input_ply or self.ply),
            "planes": [
                {"plane_name": "Shot 2", "points_count": 10,
                 "roll_deg": 2.0, "pitch_deg": 1.0,
                 "metric_pose": {"source": "orbbec_depth", "physical_metric": True,
                                 "status": "REACHABLE", "reachable": True,
                                 "roll_deg": 2.0, "pitch_deg": 1.0,
                                 "camera_roll_deg": 2.108830477273099,
                                 "camera_pitch_deg": -0.9674970836853369,
                                 "depth_points_count": 10, "depth_coverage": 1.0}},
                {"plane_name": "Shot 1", "dominant": True, "points_count": 90,
                 "roll_deg": -6.8, "pitch_deg": 6.4,
                 "metric_pose": {"source": "orbbec_depth", "physical_metric": True,
                                 "status": "REACHABLE", "reachable": True,
                                 "roll_deg": -6.8, "pitch_deg": 6.4,
                                 "camera_roll_deg": -6.650339998530714,
                                 "camera_pitch_deg": -6.968332107227335,
                                 "depth_points_count": 90, "depth_coverage": 1.0},
                 "legacy_relative_z": {"value_cm": 99, "metric": False,
                                         "stm_compatible": False}},
            ],
            "stm_z_command_allowed": False,
        }), encoding="utf-8")

    def build(self, *, conveyor_fail=False, scan_fail=False, led=True,
              outcomes=None, malformed_pose=False, camera_start_fail=False,
              initial_homing=False, initial_stable=True,
              initial_z=20.0, initial_roll=-6.83, initial_pitch=6.46,
              fail_orientation_call=None, fail_z_call=None, plan_mode="dominant_only",
              lighting=None, anomaly_detector=None, final_geometry_saver=None,
              final_capture_acquirer=None, final_rgb_acquirer=None,
              final_rgb_saver=None, final_rgb_warmup_frames=3,
              conveyor_out_enabled=False, safe_z=20, scan_z=0):
        if malformed_pose:
            self.write_pose(malformed=True)
        events = []
        projector = FakeProjector(events)
        conveyor = FakeConveyor(events, fail=conveyor_fail)
        runner = FakeStructuredRunner(
            events, projector, self.scan, self.pose_json, fail=scan_fail,
        )
        platform = FakePlatform(events)
        initial = telemetry(
            z=initial_z, roll=initial_roll, pitch=initial_pitch,
            stable=initial_stable, homing=initial_homing,
        )
        motion = FakeMotion(
            events, projector, initial_telemetry=initial,
            fail_orientation_call=fail_orientation_call,
            fail_z_call=fail_z_call,
        )
        camera = FakeCamera(events, projector, fail_start=camera_start_fail)
        evaluator = FakeEvaluator(events, outcomes or {20: True, 25: True, 30: False})
        search = HardwareAutomaticZSearch(
            platform=DiagnosticZMover(motion), camera=camera,
            projector=projector, evaluator=evaluator,
            config=HardwareZSearchConfig(
                (20, 25, 30), 30, 1.0, "highest_passing_readiness",
            ),
        )

        def checkpoint():
            events.append("LED_ON" if led else "LED_NOT_CONFIRMED")
            return led

        def fake_postprocessor(path, *, fresh_telemetry_reader):
            current = fresh_telemetry_reader()
            apply_calibration_to_pose_json(
                path,
                current_platform_roll_deg=float(current.roll_deg),
                current_platform_pitch_deg=float(current.pitch_deg),
                calibration=CAMERA_PLATFORM_RP_20260903,
            )
            apply_alignment_policy_to_pose_json(path)
            return {
                "raw_plane_count": 2, "metric_physical_plane_count": 2,
                "reachable_pose_count": 2,
            }

        def accept_first_final_frame(camera, inspection_config, *, max_attempts):
            del inspection_config, max_attempts
            selected_frame = camera.capture()
            mask = np.zeros(selected_frame.depth_mm.shape, dtype=np.uint8)
            mask[8:-8, 8:-8] = 255
            area = int(np.count_nonzero(mask))
            selected_geometry = SurfaceGeometryResult(
                object_mask=mask.copy(), surface_mask=mask, patches=(),
                object_area_px=area, surface_area_px=area,
                surface_ratio=1.0, depth_valid_ratio=.3, plane_inlier_ratio=.8,
                plane_residual=1.0, fov_edge_contact=False,
            )
            diagnostic = FinalCaptureAttemptDiagnostic(
                attempt=1, depth_valid_ratio=.3, plane_inlier_ratio=.8,
                plane_residual=1.0, object_area_px=area,
                surface_area_px=area, ready=True, reasons=(),
            )
            return GeometryReadyFinalCapture(
                selected_frame, selected_geometry, (diagnostic,), 1,
            )

        roi_base = InspectionConfig.default()
        roi_config = replace(
            roi_base,
            depth=replace(roi_base.depth, plane_min_points=100, object_open_size=1),
            surface_roi=replace(
                roi_base.surface_roi, fallback_workspace_margin_px=8,
                min_object_area=100,
            ),
        )
        cycle = IntegratedInspectionCycle(
            conveyor=conveyor, structured_light_runner=runner,
            pose_planner=RealPosePlanner(plan_mode), projector=projector,
            platform=platform, motion_diagnostic=motion, camera=camera,
            final_capture_inspection_config=roi_config,
            automatic_z_search=search, scan_z=scan_z, safe_z=safe_z,
            run_directory=self.root / "result", led_checkpoint=checkpoint,
            lighting=lighting,
            metric_pose_postprocessor=fake_postprocessor,
            anomaly_detector=anomaly_detector,
            final_capture_acquirer=final_capture_acquirer or accept_first_final_frame,
            final_geometry_saver=final_geometry_saver or save_final_geometry_capture,
            final_rgb_acquirer=final_rgb_acquirer or acquire_warmed_final_rgb_frame,
            final_rgb_saver=final_rgb_saver or save_final_rgb_capture,
            final_rgb_warmup_frames=final_rgb_warmup_frames,
            conveyor_out_enabled=conveyor_out_enabled,
        )
        cycle.final_capture_inspection_config = replace(
            roi_config,
            surface_roi=replace(
                roi_config.surface_roi,
                min_patchable_ratio=0.0,
            ),
        )
        return cycle, events, conveyor, runner, platform, motion, camera, projector

    def test_lighting_brackets_integrated_cycle_and_cleanup(self):
        events = []
        lighting = FakeLighting(events)
        cycle, events, conveyor, runner, _, motion, camera, projector = self.build(
            plan_mode="all_valid_planes", lighting=lighting,
        )
        lighting.events = events
        result = cycle.run()
        self.assertTrue(result.success)
        self.assertTrue(result.lighting_connected)
        self.assertTrue(result.inspection_led_initial_off)
        self.assertTrue(result.inspection_led_on)
        self.assertTrue(result.inspection_led_off_at_end)
        self.assertFalse(result.anomaly_executed)
        self.assertFalse(result.conveyor_out_executed)
        self.assertEqual(events[events.index("LIGHTING_CONNECT") + 1], "LIGHTING_OFF")
        self.assertEqual(events.count("LIGHTING_ON"), 1)
        self.assertLess(events.index("LIGHTING_OFF", events.index("LIGHTING_ON")), events.index("LIGHTING_CLOSE"))
        self.assertNotIn("MANUAL_LED_CHECKPOINT", result.stage_history)
        self.assertIn("READY_FOR_ANOMALY", result.stage_history)

    def test_failure_restores_black_and_turns_lighting_off(self):
        events = []
        lighting = FakeLighting(events)
        cycle, events, _, _, _, _, _, projector = self.build(
            scan_fail=True, lighting=lighting,
        )
        lighting.events = events
        result = cycle.run()
        self.assertFalse(result.success)
        self.assertTrue(result.lighting_connected)
        self.assertTrue(result.inspection_led_off_at_end)
        self.assertEqual(result.projector_final_state, "BLACK")
        self.assertFalse(result.conveyor_out_executed)
        self.assertLess(events.index("LIGHTING_OFF", events.index("LIGHTING_CONNECT")),
                        events.index("LIGHTING_OFF", events.index("LIGHTING_ON"))
                        if "LIGHTING_ON" in events else len(events))
        self.assertEqual(projector.state, ProjectorState.CLOSED)

    def test_automatic_z_failure_after_led_on_turns_lighting_off(self):
        events = []
        lighting = FakeLighting(events)
        cycle, events, conveyor, _, _, _, _, _ = self.build(
            plan_mode="all_valid_planes", outcomes={20: False, 25: True, 30: True},
            lighting=lighting,
        )
        lighting.events = events
        result = cycle.run()
        self.assertFalse(result.success)
        self.assertTrue(result.inspection_led_on)
        self.assertTrue(result.inspection_led_off_at_end)
        self.assertEqual(result.planes_failed, 2)
        self.assertEqual(result.overall_status, "FAILED")
        self.assertFalse(result.conveyor_out_executed)
        self.assertLess(events.index("LIGHTING_ON"), events.index("LIGHTING_OFF", events.index("LIGHTING_ON")))
        self.assertEqual(result.inspection_planes[0]["anomaly_executed"], False)

    def test_all_valid_planes_runs_each_plane_with_safe_transition(self):
        cycle, events, conveyor, runner, _, motion, camera, _ = self.build(
            plan_mode="all_valid_planes", outcomes={20: True, 25: True, 30: False},
        )
        result = cycle.run()
        self.assertTrue(result.success)
        self.assertEqual(result.planes_total, 2)
        self.assertEqual(result.planes_completed, 2)
        self.assertEqual(result.planes_failed, 0)
        self.assertEqual([item["plane_name"] for item in result.inspection_planes], ["Shot 1", "Shot 2"])
        self.assertTrue(all(item["ready_for_anomaly"] for item in result.inspection_planes))
        self.assertFalse(result.anomaly_executed)
        self.assertEqual(runner.events.count("STRUCTURED_SCAN"), 1)
        self.assertEqual(motion.orientation_commands, [(0.0, 0.0), (-6.8, 6.4), (2.0, 1.0)])
        self.assertEqual(
            motion.z_commands,
            [0, 20, 20, 25, 30, 25, 20, 20, 20, 25, 30, 25, 20],
        )
        self.assertEqual(camera.captures, 6)
        self.assertEqual(conveyor.move_out_calls, 0)
        self.assertTrue((cycle.paths.root / "plane_00" / "automatic_z" / "result.json").is_file())
        self.assertTrue((cycle.paths.root / "plane_01" / "automatic_z" / "result.json").is_file())

    def test_all_valid_planes_prints_plan_progress_results_and_final_summary(self):
        self.write_pose()
        cycle, _, _, _, _, _, _, _ = self.build(
            plan_mode="all_valid_planes", outcomes={20: True, 25: True, 30: False},
        )
        output = io.StringIO()
        with redirect_stdout(output):
            result = cycle.run()
        text = output.getvalue()
        self.assertTrue(result.success)
        self.assertEqual(result.detected_planes, 2)
        self.assertIn("[STAGE] PLAN_INSPECTION_POSES", text)
        self.assertIn("pose_plan_mode : all_valid_planes", text)
        self.assertIn("Detected planes : 2", text)
        self.assertIn("Planned poses   : 2", text)
        self.assertIn("original_plane_index = 1", text)
        self.assertIn("original_plane_name = Shot 1", text)
        self.assertIn("INSPECTION POSE 1 / 2", text)
        self.assertIn("INSPECTION POSE 2 / 2", text)
        self.assertEqual(text.count("status = READY_FOR_ANOMALY"), 2)
        self.assertIn("TOTAL INSPECTION POSES = 2", text)
        self.assertIn("Completed poses = 2", text)
        self.assertIn("Failed poses = 0", text)

    def test_dominant_only_prints_selected_plane_and_clamp_provenance(self):
        payload = json.loads(self.pose_json.read_text(encoding="utf-8"))
        payload["planes"][1]["raw_roll_deg"] = -41.9
        payload["planes"][1]["raw_pitch_deg"] = 6.4
        self.pose_json.write_text(json.dumps(payload), encoding="utf-8")
        cycle, _, _, _, _, _, _, _ = self.build()
        output = io.StringIO()
        with redirect_stdout(output):
            result = cycle.run()
        text = output.getvalue()
        self.assertTrue(result.success)
        self.assertEqual(result.detected_planes, 2)
        self.assertEqual(result.planes_total, 1)
        self.assertEqual(result.planes_completed, 1)
        self.assertIn("pose_plan_mode : dominant_only", text)
        self.assertIn("Selected plane", text)
        self.assertIn("legacy_phase_raw_roll = -41.900000", text)
        self.assertIn("applied_roll = -6.800000", text)
        self.assertIn("clamped = False", text)
        self.assertIn("INSPECTION POSE 1 / 1", text)
        self.assertIn("status = READY_FOR_ANOMALY", text)
        self.assertIn("Planned poses = 1", text)

    def test_pose_failure_prints_failed_progress_and_final_summary(self):
        cycle, _, _, _, _, _, _, _ = self.build(fail_orientation_call=2)
        output = io.StringIO()
        with redirect_stdout(output):
            result = cycle.run()
        text = output.getvalue()
        self.assertFalse(result.success)
        self.assertEqual(result.planes_failed, 1)
        self.assertIn("INSPECTION POSE 1 / 1", text)
        self.assertIn("status = FAILED", text)
        self.assertIn("best_z = None", text)
        self.assertIn("Failed poses = 1", text)

    def test_happy_path_order_best_return_and_artifacts(self):
        cycle, events, conveyor, runner, _, motion, camera, projector = self.build()
        result = cycle.run()
        self.assertTrue(result.success)
        self.assertEqual(result.best_z, 25)
        self.assertEqual(result.automatic_z_stop_reason, "next_candidate_failed_readiness")
        self.assertAlmostEqual(result.selected_roll, -6.8)
        self.assertAlmostEqual(result.selected_pitch, 6.4)
        self.assertTrue(result.initialization_success)
        self.assertAlmostEqual(result.initial_z_before, 20.0)
        self.assertAlmostEqual(result.initial_roll_before, -6.83)
        self.assertAlmostEqual(result.initial_pitch_before, 6.46)
        self.assertEqual(result.scan_z_requested, 0)
        self.assertTrue(result.orientation_zeroed)
        self.assertTrue(result.scan_z_reached)
        self.assertEqual(motion.orientation_commands, [(0.0, 0.0), (-6.8, 6.4)])
        self.assertEqual(motion.z_commands, [0, 20, 20, 25, 30, 25])
        self.assertEqual(camera.captures, 3)
        self.assertEqual(conveyor.move_out_calls, 0)
        self.assertFalse(result.conveyor_out_executed)
        self.assertFalse(result.anomaly_executed)
        self.assertEqual(result.projector_final_state, "BLACK")
        self.assertEqual(result.projector_state_after_close, "CLOSED")
        assert_subsequence(self, events, [
            "PROJECTOR_OPEN_BLACK", "PLATFORM_CONNECT", "PLATFORM_FRESH_READ",
            "RP_0.0_0.0", "Z_0", "CONVEYOR_F", "CONVEYOR_TARGET_REACHED",
            "STRUCTURED_SCAN", "PHASE_000", "PHASE_090", "PHASE_180", "PHASE_270",
            "Z_20", "RP_-6.8_6.4",
            "LED_ON", "CAMERA_START", "Z_20", "CAPTURE", "Z_25", "CAPTURE",
            "Z_30", "CAPTURE", "Z_25", "PROJECTOR_CLOSE",
        ])
        self.assertEqual([event for event in events if event.startswith("PHASE_")], [
            "PHASE_000", "PHASE_090", "PHASE_180", "PHASE_270",
        ])
        self.assertNotIn("CONVEYOR_OUT", events)
        self.assertEqual(events[events.index("RP_0.0_0.0") - 1], "BLACK")
        self.assertEqual(events[events.index("Z_0") - 1], "BLACK")
        camera_start = events.index("CAMERA_START")
        for command in ("Z_20", "Z_25", "Z_30"):
            command_index = events.index(command, camera_start)
            self.assertEqual(events[command_index - 1], "BLACK")
        for path in (
            cycle.paths.root / "cycle_result.json",
            cycle.paths.structured_light / "run_info.json",
            cycle.paths.structured_light / self.pose_json.name,
            cycle.paths.automatic_z / "result.json",
            cycle.paths.automatic_z / "candidates.csv",
            cycle.paths.telemetry / "platform.json",
            cycle.paths.telemetry / "platform.csv",
            cycle.paths.logs / "stages.log",
        ):
            self.assertTrue(path.is_file(), path)
        self.assertTrue((cycle.paths.structured_light / "current_run").is_dir())
        saved = json.loads((cycle.paths.root / "cycle_result.json").read_text())
        self.assertTrue(saved["conveyor_complete"])
        self.assertTrue(saved["initialization_success"])
        self.assertEqual(saved["initial_z_before"], 20.0)
        self.assertEqual(saved["initial_roll_before"], -6.83)
        self.assertEqual(saved["initial_pitch_before"], 6.46)
        self.assertEqual(saved["scan_z_requested"], 0.0)
        self.assertTrue(saved["orientation_zeroed"])
        self.assertTrue(saved["scan_z_reached"])
        self.assertTrue(saved["structured_light_success"])
        self.assertTrue(saved["manual_led_confirmed"])
        self.assertTrue(saved["automatic_z_success"])

    def test_led_checkpoint_blocks_camera_and_automatic_z(self):
        cycle, events, conveyor, _, _, motion, camera, _ = self.build(led=False)
        result = cycle.run()
        self.assertFalse(result.success)
        self.assertEqual(result.stage, "MANUAL_LED_CHECKPOINT")
        self.assertFalse(result.manual_led_confirmed)
        self.assertEqual(camera.captures, 0)
        self.assertNotIn("CAMERA_START", events)
        self.assertEqual(motion.z_commands, [0, 20])
        self.assertEqual(conveyor.move_out_calls, 0)
        self.assertEqual(result.projector_final_state, "BLACK")

    def test_conveyor_failure_halts_before_scan(self):
        cycle, _, conveyor, runner, platform, motion, camera, _ = self.build(conveyor_fail=True)
        result = cycle.run()
        self.assertFalse(result.success)
        self.assertEqual(result.stage, "CONVEYOR_TO_INSPECTION")
        self.assertEqual(runner.calls, 0)
        self.assertFalse(platform.connected)
        self.assertEqual(motion.z_commands, [0])
        self.assertEqual(camera.captures, 0)
        self.assertEqual(conveyor.move_out_calls, 0)

    def test_structured_light_failure_restores_black_and_halts(self):
        cycle, _, conveyor, runner, platform, motion, camera, projector = self.build(scan_fail=True)
        result = cycle.run()
        self.assertFalse(result.success)
        self.assertEqual(result.stage, "STRUCTURED_LIGHT_SCAN")
        self.assertEqual(runner.calls, 1)
        self.assertFalse(platform.connected)
        self.assertEqual(motion.z_commands, [0])
        self.assertEqual(camera.captures, 0)
        self.assertEqual(conveyor.move_out_calls, 0)
        self.assertEqual(result.projector_final_state, "BLACK")
        self.assertEqual(projector.state, ProjectorState.CLOSED)

    def test_failure_before_pose_planning_prints_not_reached(self):
        cycle, _, _, _, _, _, _, _ = self.build(scan_fail=True)
        output = io.StringIO()
        with redirect_stdout(output):
            result = cycle.run()
        self.assertFalse(result.pose_planning_reached)
        self.assertIn("Detected planes = NOT REACHED", output.getvalue())
        self.assertIn("Planned poses = NOT REACHED", output.getvalue())

    def test_executed_plan_with_zero_reachable_poses_prints_numeric_zero(self):
        payload = json.loads(self.pose_json.read_text(encoding="utf-8"))
        for plane in payload["planes"]:
            plane["metric_pose"].update(
                status="METRIC_VALID", reachable=False,
                camera_pitch_deg=0.0,
                reject_reason="metric camera roll missing",
            )
            plane["metric_pose"].pop("camera_roll_deg", None)
        self.pose_json.write_text(json.dumps(payload), encoding="utf-8")
        cycle, _, _, _, _, _, _, _ = self.build()
        output = io.StringIO()
        with redirect_stdout(output):
            result = cycle.run()
        self.assertTrue(result.pose_planning_reached)
        self.assertIn("Detected planes = 2", output.getvalue())
        self.assertIn("Planned poses = 0", output.getvalue())

    def test_pose_failure_prevents_inspection_motion_after_initialization(self):
        cycle, _, conveyor, _, platform, motion, camera, _ = self.build(malformed_pose=True)
        result = cycle.run()
        self.assertFalse(result.success)
        self.assertEqual(result.stage, "METRIC_POSE")
        self.assertFalse(platform.connected)
        self.assertEqual(motion.z_commands, [0])
        self.assertEqual(motion.orientation_commands, [(0.0, 0.0)])
        self.assertEqual(camera.captures, 0)
        self.assertEqual(conveyor.move_out_calls, 0)

    def test_automatic_z_failure_does_not_add_return_or_conveyor_out(self):
        cycle, _, conveyor, _, _, motion, camera, _ = self.build(outcomes={20: False})
        result = cycle.run()
        self.assertFalse(result.success)
        self.assertEqual(result.stage, "AUTOMATIC_Z")
        self.assertEqual(result.automatic_z_stop_reason, "first_candidate_failed_readiness")
        self.assertEqual(motion.z_commands, [0, 20, 20])
        self.assertEqual(camera.captures, 1)
        self.assertEqual(conveyor.move_out_calls, 0)
        self.assertTrue((cycle.paths.automatic_z / "result.json").is_file())

    def test_camera_start_failure_is_recorded_as_automatic_z_stage(self):
        cycle, events, conveyor, _, _, motion, camera, _ = self.build(
            camera_start_fail=True,
        )
        result = cycle.run()
        self.assertFalse(result.success)
        self.assertEqual(result.stage, "AUTOMATIC_Z")
        self.assertEqual(result.error_message, "camera start failed")
        self.assertEqual(camera.captures, 0)
        self.assertEqual(motion.z_commands, [0, 20])
        self.assertEqual(conveyor.move_out_calls, 0)
        self.assertEqual(events.count("CAMERA_START"), 1)

    def test_pose_json_and_input_ply_must_belong_to_current_run(self):
        outside = self.root / "outside"
        outside.mkdir()
        outside_pose = outside / self.pose_json.name
        outside_pose.write_text(self.pose_json.read_text(), encoding="utf-8")
        info = StructuredLightRunInfo("scan", self.scan, pose_json_path=outside_pose)
        with self.assertRaisesRegex(RuntimeError, "outside"):
            resolve_current_pose_json(info)

        outside_ply = outside / self.ply.name
        outside_ply.write_text(self.ply.read_text(), encoding="ascii")
        self.write_pose(input_ply=outside_ply)
        cycle, _, _, _, platform, motion, _, _ = self.build()
        result = cycle.run()
        self.assertFalse(result.success)
        self.assertIn("input_ply is outside", result.error_message)
        self.assertFalse(platform.connected)
        self.assertEqual(motion.z_commands, [0])

    def cli_args(self, *extra):
        return build_parser().parse_args([
            "--conveyor-port", "/dev/never-conveyor",
            "--platform-port", "/dev/never-platform",
            "--lighting-port", "/dev/never-lighting",
            "--conveyor-steps", "6325", "--monitor", "HDMI-0",
            "--scan-z", "0", "--safe-z", "20",
            "--z-candidates", "20,25,30", "--z-max", "30",
            "--quality-config", "config/automatic_z_quality.json", *extra,
        ])

    def test_cli_default_is_completely_hardware_free_dry_run(self):
        args = self.cli_args()
        with patch(
            "src.tools.test_integrated_inspection_cycle.OpenCVProjectorController",
        ) as projector, patch(
            "src.tools.test_integrated_inspection_cycle.SerialConveyorController",
        ) as conveyor, patch(
            "src.tools.test_integrated_inspection_cycle.SerialPlatformController",
        ) as platform, patch(
            "src.tools.test_integrated_inspection_cycle.OrbbecCameraController",
        ) as camera:
            self.assertEqual(run_cli(args), 0)
        projector.assert_not_called(); conveyor.assert_not_called()
        platform.assert_not_called(); camera.assert_not_called()

    def test_cli_accepts_explicit_legacy_roi_rollback_without_hardware(self):
        args = self.cli_args("--legacy-inspection-roi")
        self.assertTrue(args.legacy_inspection_roi)
        self.assertEqual(run_cli(args), 0)

    def test_cli_accepts_adaptive_z_and_all_valid_planes_without_hardware(self):
        args = build_parser().parse_args([
            "--conveyor-port", "/dev/never-conveyor", "--platform-port", "/dev/never-platform",
            "--lighting-port", "/dev/never-lighting",
            "--conveyor-steps", "6325", "--monitor", "HDMI-0", "--scan-z", "0",
            "--safe-z", "20", "--z-start", "20", "--z-max", "30",
            "--z-coarse-step", "5", "--z-fine-step", "1",
            "--pose-plan-mode", "all_valid_planes",
            "--quality-config", "config/automatic_z_quality.json",
        ])
        self.assertEqual(run_cli(args), 0)

    def test_production_safe_z_accepts_15_and_20(self):
        for safe_z in (15, 20):
            with self.subTest(safe_z=safe_z):
                self.assertEqual(run_cli(self.cli_args("--safe-z", str(safe_z))), 0)

    def test_production_safe_z_rejects_values_below_15(self):
        for safe_z in (14.99, 0):
            with self.subTest(safe_z=safe_z), self.assertRaisesRegex(
                ValueError, "at least.*15",
            ):
                run_cli(self.cli_args("--safe-z", str(safe_z)))

    def test_orientation_runtime_gate_rejects_fresh_z_below_minimum(self):
        self.assertEqual(ORIENTATION_SAFE_Z_MIN_CM, 15.0)
        IntegratedInspectionCycle._require_orientation_safe_height(telemetry(z=15.0))
        with self.assertRaisesRegex(IntegratedCycleError, "below.*15"):
            IntegratedInspectionCycle._require_orientation_safe_height(telemetry(z=14.99))

    def test_production_structured_light_scan_z_remains_zero_only(self):
        with self.assertRaisesRegex(ValueError, "scan_z must be 0"):
            self.build(scan_z=0.01)

    def test_fresh_scan_pose_uses_startup_fast_path_without_platform_command(self):
        cycle, events, _, _, _, _, _, _ = self.build(
            safe_z=15, initial_z=0, initial_roll=0, initial_pitch=0,
        )
        output = io.StringIO()
        with redirect_stdout(output):
            result = cycle.run()
        before_conveyor = events[:events.index("CONVEYOR_F")]
        self.assertTrue(result.success)
        self.assertTrue(result.startup_fast_path_used)
        self.assertTrue(result.orientation_zeroed)
        self.assertTrue(result.scan_z_reached)
        self.assertFalse(any(
            item.startswith(("Z_", "RP_")) for item in before_conveyor
        ))
        self.assertIn(
            "[STARTUP] already at scan pose; skipping Z15/RP0/Z0 initialization",
            output.getvalue(),
        )

    def test_stale_cached_scan_pose_cannot_enable_fast_path(self):
        cycle, events, _, _, _, motion, _, _ = self.build(
            safe_z=15, initial_z=0, initial_roll=0, initial_pitch=0,
        )
        original_read_before = motion.read_before
        startup_read = False

        def fresh_read_before():
            nonlocal startup_read
            if not startup_read:
                startup_read = True
                # The cached value was Z0, but the fresh boundary observes Z15.
                motion.current = telemetry(z=15, roll=0, pitch=0)
            return original_read_before()

        motion.read_before = fresh_read_before
        result = cycle.run()
        before_conveyor = events[:events.index("CONVEYOR_F")]
        self.assertTrue(result.success)
        self.assertFalse(result.startup_fast_path_used)
        self.assertIn("PLATFORM_FRESH_READ", before_conveyor)
        self.assertIn("RP_0.0_0.0", before_conveyor)
        self.assertIn("Z_0", before_conveyor)

    def test_safe_z_start_uses_existing_zero_and_scan_initialization(self):
        cycle, events, _, _, _, _, _, _ = self.build(
            safe_z=15, initial_z=15, initial_roll=0, initial_pitch=0,
        )
        result = cycle.run()
        before_conveyor = events[:events.index("CONVEYOR_F")]
        self.assertTrue(result.success)
        self.assertFalse(result.startup_fast_path_used)
        self.assertNotIn("Z_15", before_conveyor)
        assert_subsequence(self, before_conveyor, [
            "PLATFORM_FRESH_READ", "RP_0.0_0.0", "Z_0",
        ])

    def test_nonzero_startup_orientation_uses_safe_initialization(self):
        cycle, events, _, _, _, _, _, _ = self.build(
            safe_z=15, initial_z=0, initial_roll=5, initial_pitch=0,
        )
        result = cycle.run()
        before_conveyor = events[:events.index("CONVEYOR_F")]
        self.assertTrue(result.success)
        self.assertFalse(result.startup_fast_path_used)
        assert_subsequence(self, before_conveyor, [
            "PLATFORM_FRESH_READ", "Z_15", "Z_SETTLED_15",
            "RP_0.0_0.0", "Z_0",
        ])

    def test_unstable_startup_forces_safe_z_settle_before_orientation(self):
        cycle, events, _, _, _, _, _, _ = self.build(
            safe_z=15, initial_z=0, initial_roll=0, initial_pitch=0,
            initial_stable=False,
        )
        result = cycle.run()
        before_conveyor = events[:events.index("CONVEYOR_F")]
        self.assertTrue(result.success)
        self.assertFalse(result.startup_fast_path_used)
        assert_subsequence(self, before_conveyor, [
            "PLATFORM_FRESH_READ", "Z_15", "Z_SETTLED_15",
            "RP_0.0_0.0", "Z_0",
        ])

    def test_cli_defaults_to_25_to_17_cm_one_cm_adaptive_search(self):
        args = build_parser().parse_args([
            "--conveyor-port", "/dev/never-conveyor",
            "--platform-port", "/dev/never-platform",
            "--lighting-port", "/dev/never-lighting",
            "--conveyor-steps", "6325", "--monitor", "HDMI-0",
            "--scan-z", "0", "--safe-z", "15", "--z-max", "30",
            "--quality-config", "config/automatic_z_quality.json",
        ])
        self.assertEqual(run_cli(args), 0)
        self.assertEqual((args.z_start, args.z_coarse_step, args.z_fine_step), (25.0, 1.0, 1.0))
        self.assertEqual(args.z_search_min, 17.0)
        self.assertEqual(args.z_selection_policy, "best_surface_coverage")

    def test_safe_z_15_precedes_every_production_orientation(self):
        cycle, events, _, _, _, motion, _, _ = self.build(safe_z=15)
        result = cycle.run()
        self.assertTrue(result.success)
        self.assertEqual(motion.orientation_z_cm, [15.0, 15.0])
        assert_subsequence(self, events, [
            "Z_15", "Z_SETTLED_15", "RP_0.0_0.0",
            "Z_0", "Z_SETTLED_0", "STRUCTURED_SCAN",
            "Z_15", "Z_SETTLED_15", "RP_-6.8_6.4",
        ])
        self.assertEqual(PLATFORM_RP_LIMIT_DEG, 30.0)

    def test_completed_z_below_safety_minimum_blocks_orientation_command(self):
        cycle, _, _, _, _, motion, _, _ = self.build(safe_z=15)
        execute_z = motion.execute_z

        def report_below_minimum(z_cm):
            completed = execute_z(z_cm)
            return telemetry(
                z=14.9, roll=completed.roll_deg, pitch=completed.pitch_deg,
            )

        motion.execute_z = report_below_minimum
        result = cycle.run()
        self.assertFalse(result.success)
        self.assertIn("below the production minimum 15", result.error_message)
        self.assertEqual(motion.orientation_commands, [])

    def test_cli_rejects_mixed_explicit_and_adaptive_z_modes(self):
        args = self.cli_args("--z-start", "20", "--z-coarse-step", "5", "--z-fine-step", "1")
        with self.assertRaisesRegex(ValueError, "cannot be combined"):
            run_cli(args)

    def test_cli_rejects_invalid_anomaly_surface_coverage_before_hardware(self):
        for value in ("0", "0.8", "0.99", "1.01"):
            with self.subTest(value=value), self.assertRaisesRegex(
                ValueError, "anomaly_surface_coverage",
            ):
                run_cli(self.cli_args("--anomaly-surface-coverage", value))

    def test_cli_safe_z_and_ack_gate_before_hardware(self):
        with self.assertRaisesRegex(ValueError, "safe_z"):
            run_cli(self.cli_args("--safe-z", "31"))
        with patch(
            "src.tools.test_integrated_inspection_cycle.OpenCVProjectorController",
        ) as projector, self.assertRaisesRegex(ValueError, "ack-mechanical-range"):
            run_cli(self.cli_args("--execute"))
        projector.assert_not_called()

    def test_execute_confirmation_rejection_opens_no_hardware(self):
        args = self.cli_args(
            "--execute", "--ack-mechanical-range",
            "--cover-open-angle", "0", "--cover-close-angle", "90",
        )
        ready = type("Report", (), {
            "overall_status": StructuredLightStatus.READY,
            "issues": (), "warnings": (),
        })()
        with patch(
            "src.tools.test_integrated_inspection_cycle.ShellStructuredLightRunner.preflight_report",
            return_value=ready,
        ), patch(
            "src.tools.test_integrated_inspection_cycle.OpenCVProjectorController",
        ) as projector, patch(
            "src.tools.test_integrated_inspection_cycle.SerialConveyorController",
        ) as conveyor, patch(
            "src.tools.test_integrated_inspection_cycle.SerialPlatformController",
        ) as platform, patch(
            "src.tools.test_integrated_inspection_cycle.OrbbecCameraController",
        ) as camera:
            with self.assertRaisesRegex(ValueError, "cancelled"):
                run_cli(args, confirmation_input=lambda _: "NO")
        projector.assert_not_called(); conveyor.assert_not_called()
        platform.assert_not_called(); camera.assert_not_called()

    def test_end_to_end_cover_safe_motion_fresh_capture_and_anomaly_order(self):
        events = []
        lighting = FakeLighting(events)
        anomaly = FakeAnomalyDetector(events)
        cycle, events, _, _, _, motion, camera, _ = self.build(
            lighting=lighting, anomaly_detector=anomaly,
            outcomes={20: True, 25: True, 30: False},
        )
        lighting.events = events
        anomaly.events = events
        result = cycle.run()
        self.assertTrue(result.success)
        self.assertTrue(result.anomaly_executed)
        self.assertLess(events.index("COVER_OPEN"), events.index("STRUCTURED_SCAN"))
        close_index = events.index("COVER_CLOSE")
        self.assertEqual(events[close_index - 1], "BLACK")
        inspection_orientation = events.index("RP_-6.8_6.4")
        self.assertLess(events.index("Z_20"), inspection_orientation)
        capture_indices = [index for index, event in enumerate(events) if event == "CAPTURE"]
        geometry_capture = capture_indices[3]
        final_rgb_capture = capture_indices[-1]
        final_led_on = [i for i, event in enumerate(events) if event == "LIGHTING_ON"][-1]
        self.assertLess(geometry_capture, final_led_on)
        self.assertEqual(len([
            index for index in capture_indices
            if final_led_on < index < final_rgb_capture
        ]), 11)
        self.assertLess(final_rgb_capture, events.index("ANOMALY_INFER"))
        self.assertEqual(camera.captures, 23)
        self.assertFalse(result.inspection_planes[0]["candidate_frames_reused_for_anomaly"])
        self.assertEqual(result.inspection_planes[0]["anomaly_score"], .1)
        self.assertEqual(result.inspection_planes[0]["alignment_mode"], "FULL")
        self.assertIn("BEST_Z_SETTLED", result.stage_history)
        self.assertIn("FINAL_GEOMETRY_CAPTURE", result.stage_history)
        self.assertIn("FINAL_RGB_CAPTURE", result.stage_history)
        self.assertIn("ANOMALY_INFERENCE", result.stage_history)
        self.assertIn("COVER_CLEANUP", events)
        self.assertEqual(result.inspection_planes[0]["final_capture_attempts"], 1)
        self.assertEqual(result.inspection_planes[0]["final_capture_accepted_attempt"], 1)
        assert_subsequence(self, result.stage_history, [
            "CONNECT", "CONVEYOR_POSITION", "STRUCTURED_LIGHT", "PROJECTOR_COVER_CLOSE",
            "METRIC_POSE", "MOVE_SAFE_Z", "MOVE_ORIENTATION",
            "AUTOMATIC_Z", "BEST_Z_SETTLED", "FINAL_GEOMETRY_CAPTURE", "LED_ON",
            "FINAL_RGB_CAPTURE",
            "ANOMALY_INFERENCE", "COMPLETE",
        ])

    def test_final_capture_selects_geometry_ready_pair_and_passes_same_geometry(self):
        ratios = iter((.18, .22, .27))
        geometries = []

        def geometry_extractor(depth_mm, image_shape, inspection_config):
            del inspection_config
            mask = np.zeros(tuple(image_shape[:2]), dtype=np.uint8)
            mask[32:96, 32:96] = 255
            ratio = next(ratios)
            result = SurfaceGeometryResult(
                object_mask=mask.copy(), surface_mask=mask, patches=(),
                object_area_px=int(mask.size), surface_area_px=int(mask.size),
                surface_ratio=1.0, depth_valid_ratio=ratio,
                plane_inlier_ratio=.8, plane_residual=1.0,
                fov_edge_contact=False,
            )
            geometries.append(result)
            return result

        def acquire(camera, inspection_config, *, max_attempts):
            return acquire_geometry_ready_final_frame(
                camera, inspection_config, max_attempts=max_attempts,
                geometry_extractor=geometry_extractor,
            )

        events = []
        lighting = FakeLighting(events)
        anomaly = FakeAnomalyDetector(events)
        cycle, events, _, _, _, _, camera, _ = self.build(
            lighting=lighting, anomaly_detector=anomaly,
            final_capture_acquirer=acquire,
        )
        lighting.events = events; anomaly.events = events
        result = cycle.run()
        self.assertTrue(result.success)
        plane = result.inspection_planes[0]
        self.assertEqual(plane["final_capture_attempts"], 3)
        self.assertEqual(plane["final_capture_accepted_attempt"], 3)
        self.assertEqual(plane["final_capture_depth_valid_ratio"], .27)
        self.assertIs(anomaly.kwargs[0]["surface_geometry"], geometries[2])
        self.assertEqual(
            anomaly.kwargs[0]["geometry_capture_metadata"]["geometry_accepted_attempt"], 3,
        )
        geometry_capture_indices = [
            index for index, event in enumerate(events) if event == "CAPTURE"
        ][3:6]
        final_led_on = [i for i, event in enumerate(events) if event == "LIGHTING_ON"][-1]
        self.assertTrue(all(index < final_led_on for index in geometry_capture_indices))
        self.assertEqual(camera.captures, 25)

    def test_end_to_end_partial_pose_commands_best_effort_inside_limit(self):
        payload = json.loads(self.pose_json.read_text(encoding="utf-8"))
        dominant = payload["planes"][1]["metric_pose"]
        dominant.update(camera_roll_deg=-36.635014, camera_pitch_deg=2.527686)
        self.pose_json.write_text(json.dumps(payload), encoding="utf-8")
        events = []
        lighting = FakeLighting(events)
        anomaly = FakeAnomalyDetector(events)
        cycle, events, _, _, _, motion, _, _ = self.build(
            lighting=lighting, anomaly_detector=anomaly,
        )
        lighting.events = events; anomaly.events = events
        result = cycle.run()
        self.assertTrue(result.success)
        plane = result.inspection_planes[0]
        self.assertEqual(plane["alignment_mode"], "FULL")
        self.assertEqual(plane["requested_roll"], -30.0)
        self.assertEqual(plane["applied_roll"], -28.0)
        self.assertLessEqual(abs(plane["commanded_roll"]), 28.0)
        self.assertLessEqual(abs(plane["commanded_pitch"]), 28.0)
        self.assertEqual(motion.orientation_commands[-2], (
            plane["commanded_roll"], plane["commanded_pitch"],
        ))

    def test_first_plane_z_failure_recovers_then_second_plane_completes(self):
        events = []
        lighting = FakeLighting(events)
        anomaly = FakeAnomalyDetector(events)
        cycle, events, _, _, _, motion, _, _ = self.build(
            plan_mode="all_valid_planes", lighting=lighting, anomaly_detector=anomaly,
            outcomes={20: [False, True], 25: [True], 30: [False]},
        )
        lighting.events = events; anomaly.events = events
        result = cycle.run()
        self.assertFalse(result.success)  # backward-compatible: only all-complete is True
        self.assertEqual(result.overall_status, "PARTIAL_COMPLETE")
        self.assertEqual((result.planes_completed, result.planes_failed), (1, 1))
        self.assertEqual([item["status"] for item in result.inspection_planes], ["FAILED", "COMPLETE"])
        self.assertEqual(result.inspection_planes[0]["failure_stage"], "AUTOMATIC_Z")
        self.assertEqual(result.inspection_planes[0]["failure_reason"], "NoValidInspectionZ")
        self.assertIn((2.0, 1.0), motion.orientation_commands)
        first_failure = events.index("EVALUATE_20")
        next_orientation = events.index("RP_2.0_1.0")
        recovery_events = events[first_failure:next_orientation]
        self.assertIn("LIGHTING_OFF", recovery_events)
        self.assertIn("BLACK", recovery_events)
        self.assertIn("Z_20", recovery_events)
        self.assertIn("Z_SETTLED_20", recovery_events)
        saved = json.loads((cycle.paths.root / "plane_00" / "result.json").read_text())
        for key in (
            "object_area_px", "surface_area_px", "surface_ratio",
            "usable_patch_count", "failure_stage", "failure_reason",
        ):
            self.assertIn(key, saved)

    def test_first_plane_pass_second_plane_z_failure_is_partial_complete(self):
        events = []
        lighting = FakeLighting(events)
        anomaly = FakeAnomalyDetector(events)
        cycle, events, _, _, _, _, _, _ = self.build(
            plan_mode="all_valid_planes", lighting=lighting, anomaly_detector=anomaly,
            outcomes={20: [True, False], 25: [True], 30: [False]},
        )
        lighting.events = events; anomaly.events = events
        result = cycle.run()
        self.assertEqual(result.overall_status, "PARTIAL_COMPLETE")
        self.assertEqual((result.planes_completed, result.planes_failed), (1, 1))
        self.assertEqual([item["status"] for item in result.inspection_planes], ["COMPLETE", "FAILED"])

    def test_all_plane_quality_failures_attempt_every_pose_then_fail(self):
        events = []
        lighting = FakeLighting(events)
        anomaly = FakeAnomalyDetector(events)
        cycle, events, _, _, _, motion, _, _ = self.build(
            plan_mode="all_valid_planes", lighting=lighting, anomaly_detector=anomaly,
            outcomes={20: [False, False]},
        )
        lighting.events = events; anomaly.events = events
        result = cycle.run()
        self.assertEqual(result.overall_status, "FAILED")
        self.assertEqual((result.planes_completed, result.planes_failed), (0, 2))
        self.assertEqual(len(result.inspection_planes), 2)
        self.assertEqual(len(motion.orientation_commands), 4)  # initialize + planes + park

    def test_fatal_orientation_timeout_aborts_before_next_plane(self):
        events = []
        lighting = FakeLighting(events)
        anomaly = FakeAnomalyDetector(events)
        cycle, events, _, _, _, motion, _, _ = self.build(
            plan_mode="all_valid_planes", lighting=lighting, anomaly_detector=anomaly,
            fail_orientation_call=2,
        )
        lighting.events = events; anomaly.events = events
        result = cycle.run()
        self.assertEqual(result.overall_status, "FAILED")
        self.assertEqual(result.error_message, "orientation stable timeout")
        self.assertEqual(len(motion.orientation_commands), 2)
        self.assertNotIn("RP_2.0_1.0", events)

    def test_recoverable_anomaly_input_failure_continues_to_next_plane(self):
        events = []
        lighting = FakeLighting(events)
        anomaly = FakeAnomalyDetector(
            events, failures=[AnomalyInputDataError("bad final frame"), None],
        )
        cycle, events, _, _, _, _, _, _ = self.build(
            plan_mode="all_valid_planes", lighting=lighting, anomaly_detector=anomaly,
            outcomes={20: True, 25: True, 30: False},
        )
        lighting.events = events; anomaly.events = events
        result = cycle.run()
        self.assertEqual(result.overall_status, "PARTIAL_COMPLETE")
        self.assertEqual(result.inspection_planes[0]["failure_stage"], "ANOMALY_INFERENCE")
        self.assertEqual(result.inspection_planes[1]["status"], "COMPLETE")
        self.assertEqual(events.count("ANOMALY_INFER"), 2)

    def test_unclassified_anomaly_process_failure_is_fatal(self):
        events = []
        lighting = FakeLighting(events)
        anomaly = FakeAnomalyDetector(events, failures=[RuntimeError("model process failed")])
        cycle, events, _, _, _, motion, _, _ = self.build(
            plan_mode="all_valid_planes", lighting=lighting, anomaly_detector=anomaly,
            outcomes={20: True, 25: True, 30: False},
        )
        lighting.events = events; anomaly.events = events
        result = cycle.run()
        self.assertEqual(result.overall_status, "FAILED")
        self.assertEqual(result.error_message, "model process failed")
        self.assertEqual(len(motion.orientation_commands), 2)

    def test_final_capture_quality_failure_is_recoverable_per_plane(self):
        events = []
        lighting = FakeLighting(events)
        anomaly = FakeAnomalyDetector(events)
        calls = 0

        def saver(capture, output_directory, **kwargs):
            nonlocal calls
            calls += 1
            if calls == 1:
                raise FinalCaptureQualityError("final frame quality rejected")
            return save_final_geometry_capture(capture, output_directory, **kwargs)

        cycle, events, conveyor, _, _, _, _, _ = self.build(
            plan_mode="all_valid_planes", lighting=lighting, anomaly_detector=anomaly,
            final_geometry_saver=saver,
            outcomes={20: True, 25: True, 30: False},
            conveyor_out_enabled=True,
        )
        lighting.events = events; anomaly.events = events
        result = cycle.run()
        self.assertEqual(result.overall_status, "PARTIAL_COMPLETE")
        self.assertEqual(result.inspection_planes[0]["failure_stage"], "FINAL_GEOMETRY_CAPTURE")
        self.assertEqual(result.inspection_planes[1]["status"], "COMPLETE")
        self.assertEqual(result.inspection_status, "PARTIAL_COMPLETE")
        self.assertEqual(result.final_judgement, "RECHECK")
        self.assertEqual(result.conveyor_out, "COMPLETE")
        self.assertTrue(result.cycle_transport_complete)
        self.assertEqual(conveyor.move_out_calls, 1)

    def test_geometry_readiness_exhaustion_recovers_and_next_plane_continues(self):
        scripts = [(0.18,) * 8, (0.27,)]

        def acquire(camera, inspection_config, *, max_attempts):
            ratios = iter(scripts.pop(0))

            def extract(depth_mm, image_shape, config):
                del depth_mm, config
                mask = np.zeros(tuple(image_shape[:2]), dtype=np.uint8)
                mask[32:96, 32:96] = 255
                ratio = next(ratios)
                return SurfaceGeometryResult(
                    object_mask=mask.copy(), surface_mask=mask, patches=(),
                    object_area_px=4096, surface_area_px=4096,
                    surface_ratio=1.0, depth_valid_ratio=ratio,
                    plane_inlier_ratio=.8, plane_residual=1.0,
                    fov_edge_contact=False,
                )

            return acquire_geometry_ready_final_frame(
                camera, inspection_config, max_attempts=max_attempts,
                geometry_extractor=extract,
            )

        events = []
        lighting = FakeLighting(events)
        anomaly = FakeAnomalyDetector(events)
        cycle, events, _, _, _, _, _, _ = self.build(
            plan_mode="all_valid_planes", lighting=lighting,
            anomaly_detector=anomaly, final_capture_acquirer=acquire,
            safe_z=15,
            outcomes={20: [True, True], 25: [True, True], 30: [False, False]},
        )
        lighting.events = events; anomaly.events = events
        result = cycle.run()
        self.assertEqual(result.overall_status, "PARTIAL_COMPLETE")
        first, second = result.inspection_planes
        self.assertEqual(first["failure_stage"], "FINAL_GEOMETRY_CAPTURE")
        self.assertEqual(first["final_capture_attempts"], 8)
        self.assertIsNone(first["final_capture_accepted_attempt"])
        self.assertFalse((cycle.paths.root / "plane_00" / "final_capture" / "final_depth.npy").exists())
        self.assertEqual(second["status"], "COMPLETE")
        self.assertEqual(second["final_capture_accepted_attempt"], 1)
        self.assertEqual(events.count("ANOMALY_INFER"), 1)
        next_orientation = events.index("RP_2.0_1.0")
        preceding = events[:next_orientation]
        self.assertIn("LIGHTING_OFF", preceding)
        self.assertIn("Z_SETTLED_15", preceding)

    def test_geometry_is_frozen_before_led_on_and_no_motion_occurs_until_anomaly(self):
        events = []
        lighting = FakeLighting(events)
        anomaly = FakeAnomalyDetector(events)
        cycle, events, _, _, _, _, _, _ = self.build(
            lighting=lighting, anomaly_detector=anomaly,
        )
        lighting.events = events; anomaly.events = events
        result = cycle.run()
        self.assertEqual(result.quality_judgement, "OK")
        led_on = [i for i, event in enumerate(events) if event == "LIGHTING_ON"][-1]
        geometry_capture = [
            index for index, event in enumerate(events[:led_on]) if event == "CAPTURE"
        ][-1]
        anomaly_index = events.index("ANOMALY_INFER")
        forbidden = [
            event for event in events[geometry_capture + 1:anomaly_index]
            if event.startswith(("Z_", "RP_", "CONVEYOR_"))
        ]
        self.assertEqual(forbidden, [])
        self.assertIs(anomaly.kwargs[0]["surface_geometry"].surface_mask is None, False)
        self.assertIsNotNone(anomaly.kwargs[0]["inspection_mask"])
        plane = result.inspection_planes[0]
        for key in (
            "final_depth_path", "final_rgb_path", "object_mask_path",
            "surface_mask_path", "surface_geometry_overlay_path",
            "inspection_mask_path", "inspection_mask_overlay_path",
        ):
            self.assertTrue(Path(plane[key]).is_file())
        self.assertEqual(plane["anomaly_roi_type"], "depth_external_contour_fill")
        self.assertGreater(plane["inspection_area_px"], 0)

    def test_aruco_detection_failure_falls_back_without_breaking_complete_cycle(self):
        class MissingMarkerDetector:
            def detectMarkers(self, gray):
                del gray
                return [], None, []

        events: list[str] = []
        lighting = FakeLighting(events)
        anomaly = FakeAnomalyDetector(events)
        cycle, events, _, _, _, _, camera, _ = self.build(
            lighting=lighting, anomaly_detector=anomaly,
        )
        lighting.events = events
        anomaly.events = events
        camera.color_intrinsics = lambda width, height: CameraIntrinsics(
            200, 200, width / 2, height / 2, width, height, "synthetic", "color",
        )
        cycle.aruco_detector_factory = MissingMarkerDetector
        result = cycle.run()
        plane = result.inspection_planes[0]
        self.assertTrue(result.success)
        self.assertEqual(plane["aruco_roi_status"], "FALLBACK")
        self.assertEqual(plane["anomaly_roi_type"], "depth_external_contour_fill")
        self.assertIn("required ArUco marker IDs", plane["aruco_fallback_reason"])
        self.assertTrue(Path(
            plane["roi_diagnostic_artifacts"]["aruco_rgb.png"]
        ).is_file())

    def test_hybrid_roi_is_used_and_aruco_depth_sequence_has_no_motion(self):
        markers = {
            0: np.array([[10, 10], [20, 10], [20, 20], [10, 20]], np.float32),
            1: np.array([[100, 10], [110, 10], [110, 20], [100, 20]], np.float32),
            2: np.array([[10, 100], [20, 100], [20, 110], [10, 110]], np.float32),
            3: np.array([[100, 100], [110, 100], [110, 110], [100, 110]], np.float32),
        }

        class Detector:
            def detectMarkers(self, gray):
                del gray
                ids = np.arange(4, dtype=np.int32).reshape(-1, 1)
                return [markers[index][None, ...] for index in range(4)], ids, []

        hybrid_mask = np.zeros((128, 128), dtype=np.uint8)
        hybrid_mask[24:104, 24:104] = 255
        board = np.zeros_like(hybrid_mask)
        board[10:111, 10:111] = 255
        hybrid_result = HybridInspectionROIResult(
            inspection_mask=hybrid_mask,
            board_roi_mask=board,
            board_background_mask=board.copy(),
            depth_object_mask=hybrid_mask.copy(),
            depth_unknown_mask=np.zeros_like(board),
            rgb_recovered_unknown_mask=np.zeros_like(board),
            board_plane_normal=np.array([0.0, 0.0, -1.0]),
            board_plane_center_mm=np.array([0.0, 0.0, 500.0]),
            board_plane_inlier_ratio=.9,
            board_plane_residual_mm=.5,
            board_roi_area_px=int(np.count_nonzero(board)),
            depth_object_area_px=int(np.count_nonzero(hybrid_mask)),
            depth_unknown_area_px=0,
            hybrid_inspection_area_px=int(np.count_nonzero(hybrid_mask)),
            hybrid_to_depth_object_ratio=1.0,
            board_roi_depth_valid_ratio=.9,
            depth_p05_mm=450.0,
            depth_median_mm=500.0,
            depth_p95_mm=505.0,
            intrinsics_source="synthetic",
        )
        events: list[str] = []
        lighting = FakeLighting(events)
        anomaly = FakeAnomalyDetector(events)
        cycle, events, _, _, _, _, camera, _ = self.build(
            lighting=lighting, anomaly_detector=anomaly,
        )
        lighting.events = events
        anomaly.events = events
        camera.color_intrinsics = lambda width, height: CameraIntrinsics(
            200, 200, width / 2, height / 2, width, height, "synthetic", "color",
        )
        cycle.aruco_detector_factory = Detector
        cycle.hybrid_roi_builder = lambda *args, **kwargs: hybrid_result
        default_config = cycle.final_capture_inspection_config
        cycle.final_capture_inspection_config = replace(
            default_config,
            hybrid_roi=replace(default_config.hybrid_roi, use_for_anomaly=True),
        )
        result = cycle.run()
        plane = result.inspection_planes[0]
        self.assertTrue(result.success)
        self.assertEqual(plane["aruco_roi_status"], "SUCCESS")
        self.assertEqual(plane["anomaly_roi_type"], "aruco_depth_rgb_hybrid")
        np.testing.assert_array_equal(
            anomaly.kwargs[0]["inspection_mask"], hybrid_mask,
        )
        first_led_on = events.index("LIGHTING_ON")
        first_led_off = events.index("LIGHTING_OFF", first_led_on)
        motion_during_aruco = [
            event for event in events[first_led_on:first_led_off]
            if event.startswith(("Z_", "RP_", "CONVEYOR_"))
        ]
        self.assertEqual(motion_during_aruco, [])
        self.assertEqual(events.count("LIGHTING_ON"), 2)
        self.assertTrue(Path(plane["inspection_mask_path"]).is_file())
        self.assertTrue(Path(
            plane["roi_diagnostic_artifacts"]["hybrid_inspection_mask.png"]
        ).is_file())

    def test_external_contour_fill_is_default_anomaly_roi_and_saves_evidence(self):
        mask = np.zeros((128, 128), dtype=np.uint8)
        main = np.zeros_like(mask)
        main[24:104, 24:104] = 255
        main[50:78, 50:78] = 0
        filled = np.zeros_like(mask)
        filled[24:104, 24:104] = 255
        inspection = np.zeros_like(mask)
        inspection[34:94, 34:94] = 255
        contour_result = DepthExternalContourROIResult(
            workspace_mask=np.full_like(mask, 255),
            depth_object_candidate_mask=main.copy(),
            depth_main_component_mask=main,
            depth_object_contour_filled=filled,
            inspection_mask=inspection,
            workspace_source="fallback",
            workspace_area_px=mask.size,
            depth_candidate_area_px=int(np.count_nonzero(main)),
            depth_main_component_area_px=int(np.count_nonzero(main)),
            filled_object_area_px=int(np.count_nonzero(filled)),
            inspection_area_px=int(np.count_nonzero(inspection)),
            fill_gain_px=int(np.count_nonzero(filled) - np.count_nonzero(main)),
            fill_gain_ratio=float(
                (np.count_nonzero(filled) - np.count_nonzero(main))
                / np.count_nonzero(main)
            ),
            depth_valid_ratio=.8,
            board_plane_inlier_ratio=.9,
            board_plane_residual_mm=.5,
        )
        events: list[str] = []
        lighting = FakeLighting(events)
        anomaly = FakeAnomalyDetector(events)
        cycle, events, _, _, _, _, _, _ = self.build(
            lighting=lighting, anomaly_detector=anomaly,
        )
        lighting.events = events
        anomaly.events = events
        cycle.depth_contour_roi_builder = lambda *args, **kwargs: contour_result
        cycle.hybrid_roi_builder = lambda *args, **kwargs: self.fail(
            "hybrid ROI must not be the production default"
        )
        result = cycle.run()
        plane = result.inspection_planes[0]
        self.assertTrue(result.success)
        self.assertEqual(plane["anomaly_roi_type"], "depth_external_contour_fill")
        self.assertEqual(plane["workspace_source"], "fallback")
        self.assertGreater(plane["fill_gain_px"], 0)
        np.testing.assert_array_equal(
            anomaly.kwargs[0]["inspection_mask"], inspection,
        )
        np.testing.assert_array_equal(
            anomaly.kwargs[0]["filled_object_mask"], filled,
        )
        for name in (
            "depth_object_candidate_mask.png",
            "depth_main_component_mask.png",
            "depth_object_contour_filled.png",
            "inspection_mask.png",
            "inspection_mask_overlay.png",
        ):
            self.assertTrue(Path(
                plane["depth_contour_roi_artifacts"][name]
            ).is_file(), name)

    def test_invalid_optional_hybrid_keeps_trusted_contour_roi(self):
        markers = {
            0: np.array([[10, 10], [20, 10], [20, 20], [10, 20]], np.float32),
            1: np.array([[100, 10], [110, 10], [110, 20], [100, 20]], np.float32),
            2: np.array([[10, 100], [20, 100], [20, 110], [10, 110]], np.float32),
            3: np.array([[100, 100], [110, 100], [110, 110], [100, 110]], np.float32),
        }

        class Detector:
            def detectMarkers(self, gray):
                del gray
                ids = np.arange(4, dtype=np.int32).reshape(-1, 1)
                return [markers[index][None, ...] for index in range(4)], ids, []

        events: list[str] = []
        lighting = FakeLighting(events)
        anomaly = FakeAnomalyDetector(events)
        cycle, events, _, _, _, _, camera, _ = self.build(
            lighting=lighting, anomaly_detector=anomaly,
        )
        lighting.events = events
        anomaly.events = events
        camera.color_intrinsics = lambda width, height: CameraIntrinsics(
            200, 200, width / 2, height / 2, width, height, "synthetic", "color",
        )
        cycle.aruco_detector_factory = Detector
        cycle.hybrid_roi_builder = lambda *args, **kwargs: (_ for _ in ()).throw(
            ValueError("hybrid inspection mask is empty")
        )
        default_config = cycle.final_capture_inspection_config
        cycle.final_capture_inspection_config = replace(
            default_config,
            hybrid_roi=replace(default_config.hybrid_roi, use_for_anomaly=True),
        )
        result = cycle.run()
        plane = result.inspection_planes[0]
        self.assertTrue(result.success)
        self.assertEqual(plane["aruco_roi_status"], "FALLBACK")
        self.assertEqual(plane["anomaly_roi_type"], "depth_external_contour_fill")
        self.assertIn("hybrid inspection mask is empty", plane["aruco_fallback_reason"])

    def test_untrusted_board_plane_skips_anomaly_and_returns_recheck(self):
        events: list[str] = []
        lighting = FakeLighting(events)
        anomaly = FakeAnomalyDetector(events)
        cycle, events, _, _, _, _, _, _ = self.build(
            lighting=lighting, anomaly_detector=anomaly,
        )
        lighting.events = events
        anomaly.events = events
        cycle.depth_contour_roi_builder = lambda *args, **kwargs: (_ for _ in ()).throw(
            ValueError("all spatial plane hypotheses rejected")
        )
        result = cycle.run()
        plane = result.inspection_planes[0]
        self.assertFalse(result.success)
        self.assertEqual(result.quality_judgement, "RECHECK")
        self.assertEqual(plane["failure_stage"], "FINAL_GEOMETRY_CAPTURE")
        self.assertFalse(plane["anomaly_executed"])
        self.assertNotIn("ANOMALY_INFER", events)

    def test_led_on_invalid_depth_does_not_block_frozen_geometry_anomaly(self):
        def rgb_with_invalid_depth(camera, *, warmup_frames, expected_shape):
            del camera, warmup_frames
            return RGBDepthFrame(
                np.zeros((*expected_shape, 3), dtype=np.uint8),
                np.zeros((1, 1), dtype=np.float32), 99.0,
            )

        events = []
        lighting = FakeLighting(events)
        anomaly = FakeAnomalyDetector(events)
        cycle, events, _, _, _, _, _, _ = self.build(
            lighting=lighting, anomaly_detector=anomaly,
            final_rgb_acquirer=rgb_with_invalid_depth,
        )
        lighting.events = events; anomaly.events = events
        result = cycle.run()
        self.assertEqual(result.inspection_planes[0]["inspection_judgement"], "OK")
        self.assertEqual(events.count("ANOMALY_INFER"), 1)

    def test_final_rgb_mask_shape_mismatch_is_recheck(self):
        def invalid_rgb(camera, *, warmup_frames, expected_shape):
            del camera, warmup_frames, expected_shape
            raise FinalRGBCaptureError("frozen surface mask / final RGB shape mismatch")

        events = []
        lighting = FakeLighting(events)
        anomaly = FakeAnomalyDetector(events)
        cycle, events, _, _, _, _, _, _ = self.build(
            lighting=lighting, anomaly_detector=anomaly,
            final_rgb_acquirer=invalid_rgb,
        )
        lighting.events = events; anomaly.events = events
        result = cycle.run()
        plane = result.inspection_planes[0]
        self.assertEqual(plane["inspection_judgement"], "RECHECK")
        self.assertEqual(plane["failure_stage"], "FINAL_RGB_CAPTURE")
        self.assertEqual(result.quality_judgement, "RECHECK")
        self.assertNotIn("ANOMALY_INFER", events)

    def test_normal_is_ok_and_defect_is_ng_without_changing_execution_status(self):
        for classification, judgement in (("NORMAL", "OK"), ("DEFECT", "NG")):
            with self.subTest(classification=classification):
                events = []
                lighting = FakeLighting(events)
                anomaly = FakeAnomalyDetector(events, classifications=[classification])
                cycle, events, _, _, _, _, _, _ = self.build(
                    lighting=lighting, anomaly_detector=anomaly,
                )
                lighting.events = events; anomaly.events = events
                result = cycle.run()
                self.assertEqual(result.execution_status, "COMPLETE")
                self.assertEqual(result.quality_judgement, judgement)
                self.assertEqual(
                    result.inspection_planes[0]["inspection_judgement"], judgement,
                )
                saved = json.loads(
                    (cycle.paths.root / "cycle_result.json").read_text(encoding="utf-8")
                )
                self.assertEqual(saved["execution_status"], "COMPLETE")
                self.assertEqual(saved["quality_judgement"], judgement)
                self.assertEqual(saved["planned_planes"], 1)
                saved_plane = saved["inspection_planes"][0]
                self.assertEqual(saved_plane["classification"], classification)
                self.assertEqual(saved_plane["quality_judgement"], judgement)
                self.assertIn("commanded_rp", saved_plane)
                self.assertIn("actual_rp", saved_plane)
                if classification == "DEFECT":
                    self.assertTrue(result.success)

    def test_multi_plane_quality_judgement_priority(self):
        cases = (
            (["NORMAL", "NORMAL"], [None, None], "OK", "COMPLETE"),
            (["NORMAL", "DEFECT"], [None, None], "NG", "COMPLETE"),
            (["NORMAL"], [None, AnomalyInputDataError("bad input")],
             "RECHECK", "PARTIAL_COMPLETE"),
            (["DEFECT"], [None, AnomalyInputDataError("bad input")],
             "NG", "PARTIAL_COMPLETE"),
        )
        for classifications, failures, quality, execution in cases:
            with self.subTest(quality=quality, execution=execution):
                events = []
                lighting = FakeLighting(events)
                anomaly = FakeAnomalyDetector(
                    events, failures=failures, classifications=classifications,
                )
                cycle, events, _, _, _, _, _, _ = self.build(
                    plan_mode="all_valid_planes", lighting=lighting,
                    anomaly_detector=anomaly,
                )
                lighting.events = events; anomaly.events = events
                result = cycle.run()
                self.assertEqual(result.quality_judgement, quality)
                self.assertEqual(result.execution_status, execution)
                self.assertEqual(result.planned_planes, 2)
                self.assertEqual(result.completed_planes, result.planes_completed)
                self.assertEqual(result.failed_planes, result.planes_failed)

    def test_optional_conveyor_out_runs_only_when_enabled_after_safe_recovery(self):
        for enabled in (False, True):
            with self.subTest(enabled=enabled):
                events = []
                lighting = FakeLighting(events)
                anomaly = FakeAnomalyDetector(events)
                cycle, events, conveyor, _, _, _, _, _ = self.build(
                    lighting=lighting, anomaly_detector=anomaly,
                    conveyor_out_enabled=enabled, safe_z=15,
                )
                lighting.events = events; anomaly.events = events
                result = cycle.run()
                self.assertEqual(conveyor.move_out_calls, int(enabled))
                self.assertEqual(result.conveyor_out_executed, enabled)
                if enabled:
                    out_index = events.index("CONVEYOR_OUT")
                    self.assertIn("LIGHTING_OFF", events[:out_index])
                    self.assertIn("Z_SETTLED_0", events[:out_index])
                    self.assertLess(events.index("COVER_OPEN", events.index("ANOMALY_INFER")), out_index)
                    self.assertEqual(result.conveyor_out_direction, "F")
                    self.assertEqual(result.conveyor_out_steps, 10000)
                    assert_subsequence(self, result.stage_history, [
                        "ANOMALY_INFERENCE", "MOVE_SAFE_Z", "CONVEYOR_OUT", "COMPLETE",
                    ])

    def test_cli_conveyor_out_is_fixed_to_f10000(self):
        self.assertEqual(run_cli(self.cli_args()), 0)
        with self.assertRaisesRegex(ValueError, "F10000"):
            run_cli(self.cli_args("--conveyor-out-direction", "B"))
        with self.assertRaisesRegex(ValueError, "F10000"):
            run_cli(self.cli_args("--conveyor-out-steps", "9999"))

    def test_next_adaptive_pose_changes_rp_at_best_z_then_raises_to_25(self):
        cycle, events, _, _, _, motion, _, projector = self.build(
            initial_z=19, initial_roll=10, initial_pitch=0, safe_z=15,
        )
        projector.open(); events.clear()
        pose = SimpleNamespace(roll_deg=-25.0, pitch_deg=5.0)
        transition = cycle._adaptive_pose_for_z(pose, 19)
        entry = cycle._adaptive_pose_for_z(pose, 25)
        cycle._apply_adaptive_pose(transition, None)
        cycle._apply_adaptive_pose(entry, transition)
        rp_at_best = next(i for i, event in enumerate(events) if event.startswith("RP_"))
        z25 = events.index("Z_25")
        rp_at_entry = next(
            i for i, event in enumerate(events[z25 + 1:], z25 + 1)
            if event.startswith("RP_")
        )
        self.assertLess(rp_at_best, z25)
        self.assertLess(z25, rp_at_entry)
        self.assertNotIn("Z_15", events)
        self.assertNotIn("RP_0.0_0.0", events)
        self.assertEqual(motion.orientation_z_cm[:2], [19.0, 25.0])

    def test_two_pose_production_transition_skips_level_and_z15(self):
        events = []
        lighting = FakeLighting(events)
        anomaly = FakeAnomalyDetector(events, classifications=["NORMAL", "NORMAL"])
        cycle, events, _, _, _, motion, _, _ = self.build(
            plan_mode="all_valid_planes", lighting=lighting,
            anomaly_detector=anomaly, conveyor_out_enabled=False, safe_z=15,
            initial_z=0, initial_roll=0, initial_pitch=0,
        )
        lighting.events = events; anomaly.events = events

        class FakeAdaptiveSearch:
            config = SimpleNamespace(search_mode="adaptive")

            def run(inner_self, *, pose_id, roll, pitch):
                del inner_self, pose_id, roll, pitch
                events.append("AUTO_Z_RUN")
                self.assertAlmostEqual(motion.current.z_cm, 25.0)
                motion.execute_z(19.0)
                item = replace(candidate(19, True), quality_score=.9)
                return HardwareZSearchResult(
                    True, (item,), 19.0, None, None,
                    item.diagnostic_dict(), selection_policy="best_surface_coverage",
                    search_mode="adaptive", stop_reason="best_surface_coverage",
                )

        cycle.automatic_z_search = FakeAdaptiveSearch()
        result = cycle.run()
        self.assertTrue(result.success)
        first_anomaly = events.index("ANOMALY_INFER")
        second_auto_z = events.index("AUTO_Z_RUN", events.index("AUTO_Z_RUN") + 1)
        transition = events[first_anomaly + 1:second_auto_z]
        rp_events = [event for event in transition if event.startswith("RP_")]
        self.assertTrue(rp_events)
        self.assertLess(transition.index(rp_events[0]), transition.index("Z_25"))
        self.assertNotIn("Z_15", transition)
        self.assertNotIn("RP_0.0_0.0", transition)

    def test_final_park_then_cover_open_then_conveyor_out(self):
        events = []
        lighting = FakeLighting(events)
        cycle, events, conveyor, _, _, _, _, projector = self.build(
            lighting=lighting, conveyor_out_enabled=True,
            initial_z=19, initial_roll=20, initial_pitch=2,
        )
        lighting.events = events; lighting.connected = True
        projector.open(); events.clear()
        result = IntegratedCycleResult(str(self.root), 15, 0, lighting_connected=True)
        cycle._park_and_run_optional_conveyor_out(result)
        rp0 = events.index("RP_0.0_0.0")
        z0 = events.index("Z_0")
        cover = events.index("COVER_OPEN")
        conveyor_out = events.index("CONVEYOR_OUT")
        self.assertLess(rp0, z0); self.assertLess(z0, cover)
        self.assertLess(cover, conveyor_out)
        self.assertTrue(result.platform_parked)
        self.assertEqual(
            (result.final_platform_roll_deg, result.final_platform_pitch_deg,
             result.final_platform_z_cm), (0.0, 0.0, 0.0),
        )
        self.assertTrue(result.cover_open)
        self.assertEqual(result.conveyor_out, "COMPLETE")
        self.assertTrue(result.cycle_transport_complete)
        self.assertEqual(conveyor.move_out_calls, 1)

    def test_park_failure_blocks_cover_and_conveyor_out(self):
        events = []
        lighting = FakeLighting(events)
        cycle, events, conveyor, _, _, motion, _, projector = self.build(
            lighting=lighting, conveyor_out_enabled=True,
            initial_z=19, initial_roll=20, initial_pitch=2, fail_z_call=1,
        )
        lighting.events = events; lighting.connected = True
        projector.open(); events.clear()
        with self.assertRaisesRegex(RuntimeError, "Z stable timeout"):
            cycle._park_and_run_optional_conveyor_out(
                IntegratedCycleResult(str(self.root), 15, 0, lighting_connected=True),
            )
        self.assertNotIn("COVER_OPEN", events)
        self.assertEqual(conveyor.move_out_calls, 0)

    def test_cover_open_failure_blocks_conveyor_out(self):
        events = []
        lighting = FakeLighting(events)
        cycle, events, conveyor, _, _, _, _, projector = self.build(
            lighting=lighting, conveyor_out_enabled=True,
            initial_z=19, initial_roll=20, initial_pitch=2,
        )
        lighting.events = events; lighting.connected = True
        lighting.projector_cover_open = lambda: (_ for _ in ()).throw(
            RuntimeError("cover open failed")
        )
        projector.open(); events.clear()
        with self.assertRaisesRegex(RuntimeError, "cover open failed"):
            cycle._park_and_run_optional_conveyor_out(
                IntegratedCycleResult(str(self.root), 15, 0, lighting_connected=True),
            )
        self.assertEqual(conveyor.move_out_calls, 0)

    def test_conveyor_out_failure_does_not_mark_transport_complete(self):
        events = []
        lighting = FakeLighting(events)
        cycle, events, conveyor, _, _, _, _, projector = self.build(
            lighting=lighting, conveyor_out_enabled=True,
            initial_z=19, initial_roll=20, initial_pitch=2,
        )
        lighting.events = events; lighting.connected = True
        conveyor.fail = True
        projector.open(); events.clear()
        result = IntegratedCycleResult(str(self.root), 15, 0, lighting_connected=True)
        with self.assertRaisesRegex(RuntimeError, "conveyor failed"):
            cycle._park_and_run_optional_conveyor_out(result)
        self.assertEqual(conveyor.move_out_calls, 1)
        self.assertEqual(result.conveyor_out, "FAILED")
        self.assertFalse(result.cycle_transport_complete)

    def test_normal_and_ng_both_park_and_run_conveyor_out(self):
        for classification, judgement in (("NORMAL", "OK"), ("DEFECT", "NG")):
            with self.subTest(classification=classification):
                events = []
                lighting = FakeLighting(events)
                anomaly = FakeAnomalyDetector(events, classifications=[classification])
                cycle, events, conveyor, _, _, _, _, _ = self.build(
                    lighting=lighting, anomaly_detector=anomaly,
                    conveyor_out_enabled=True,
                )
                lighting.events = events; anomaly.events = events
                result = cycle.run()
                self.assertTrue(result.success)
                self.assertEqual(result.quality_judgement, judgement)
                self.assertTrue(result.platform_parked)
                self.assertTrue(result.cover_open)
                self.assertEqual(result.conveyor_out, "COMPLETE")
                self.assertTrue(result.cycle_transport_complete)
                self.assertTrue(result.cycle_complete)
                self.assertEqual(conveyor.move_out_calls, 1)
                saved = json.loads(
                    (cycle.paths.root / "cycle_result.json").read_text(encoding="utf-8")
                )
                self.assertEqual(saved["completed_planes"], 1)
                self.assertEqual(saved["quality_judgement"], judgement)
                self.assertEqual(
                    (saved["final_platform_roll_deg"],
                     saved["final_platform_pitch_deg"],
                     saved["final_platform_z_cm"]),
                    (0.0, 0.0, 0.0),
                )
                self.assertTrue(saved["platform_parked"])
                self.assertTrue(saved["cover_open"])
                self.assertEqual(saved["conveyor_out"], "COMPLETE")
                self.assertTrue(saved["cycle_transport_complete"])
                self.assertTrue(saved["cycle_complete"])

    def test_anomaly_model_missing_reports_not_ready_without_hardware(self):
        detector = ProductionAnomalyDetector(ProductionAnomalyConfig(
            self.root / "missing.pth", self.root / "missing.csv",
        ))
        with self.assertRaisesRegex(AnomalyModelNotReadyError, "ANOMALY_MODEL_NOT_READY"):
            detector.validate_ready()

    def test_final_candidate_descending_transition_reduces_tilt_before_z(self):
        cycle, events, _, _, _, _, _, _ = self.build()
        cfg = cycle.final_capture_inspection_config.quality
        current = adaptive_pose_for_z(
            22, 25, 10, roll_limit_deg=cfg.inspection_roll_limit_deg,
            pitch_limit_deg=cfg.inspection_pitch_limit_deg, envelope=cfg.tilt_envelope,
        )
        target = adaptive_pose_for_z(
            19, 25, 10, roll_limit_deg=cfg.inspection_roll_limit_deg,
            pitch_limit_deg=cfg.inspection_pitch_limit_deg, envelope=cfg.tilt_envelope,
        )
        events.clear()
        cycle.projector.open()
        events.clear()
        cycle._move_to_candidate(target, current)
        rp_index = next(i for i, value in enumerate(events) if value.startswith("RP_"))
        z_index = events.index("Z_19")
        self.assertLess(rp_index, z_index)

    def test_final_candidate_ascending_transition_raises_z_before_tilt(self):
        cycle, events, _, _, _, _, _, _ = self.build()
        cfg = cycle.final_capture_inspection_config.quality
        current = adaptive_pose_for_z(
            18, 25, 10, roll_limit_deg=cfg.inspection_roll_limit_deg,
            pitch_limit_deg=cfg.inspection_pitch_limit_deg, envelope=cfg.tilt_envelope,
        )
        target = adaptive_pose_for_z(
            21, 25, 10, roll_limit_deg=cfg.inspection_roll_limit_deg,
            pitch_limit_deg=cfg.inspection_pitch_limit_deg, envelope=cfg.tilt_envelope,
        )
        events.clear()
        cycle.projector.open()
        events.clear()
        cycle._move_to_candidate(target, current)
        z_index = events.index("Z_21")
        rp_index = next(i for i, value in enumerate(events) if value.startswith("RP_"))
        self.assertLess(z_index, rp_index)

    def test_final_candidate_falls_back_to_second_ranked_pass(self):
        calls = []

        def capture(camera, config, *, max_attempts):
            del config, max_attempts
            calls.append("geometry")
            if len(calls) == 1:
                raise FinalCaptureQualityError("candidate one geometry failed")
            frame = camera.capture()
            mask = np.full(frame.depth_mm.shape, 255, dtype=np.uint8)
            area = int(mask.size)
            geometry = SurfaceGeometryResult(
                object_mask=mask, surface_mask=mask, patches=(),
                object_area_px=area, surface_area_px=area, surface_ratio=1.0,
                depth_valid_ratio=1.0, plane_inlier_ratio=1.0,
                plane_residual=0.1, fov_edge_contact=False,
            )
            return GeometryReadyFinalCapture(frame, geometry, (), 1)

        cycle, events, _, _, _, _, _, _ = self.build(final_capture_acquirer=capture)
        cycle.projector.open()
        contour_mask = np.full((128, 128), 255, dtype=np.uint8)
        contour = SimpleNamespace(
            inspection_mask=contour_mask,
            depth_object_contour_filled=contour_mask,
            workspace_mask=contour_mask,
            depth_main_component_mask=contour_mask,
            board_plane_fit_mask=contour_mask,
            plane_hypothesis_masks=(),
            selected_board_plane_hypothesis_index=None,
        )
        cycle.depth_contour_roi_builder = lambda *args, **kwargs: contour
        ranked = [
            replace(candidate(22, True), quality_score=.9),
            replace(candidate(20, True), quality_score=.8),
        ]
        z_result = SimpleNamespace(candidates=ranked, best_z=22)
        pose = SimpleNamespace(roll_deg=25.0, pitch_deg=0.0)
        result = SimpleNamespace(lighting_connected=False)
        with patch(
            "src.integration.integrated_inspection_cycle.measure_patchability",
            return_value=([(0, 0)], 4096, 1.0),
        ):
            selected = cycle._select_usable_final_candidate(
                z_result, pose,
                {"available": False, "board_quad": None, "marker_map": {}},
                result, self.root / "candidate_final",
            )
        self.assertEqual(selected.z_command, 20)
        self.assertEqual(cycle._active_final_candidate_metadata["selected_candidate_rank"], 2)
        attempts = cycle._active_final_candidate_metadata["final_candidate_attempts"]
        self.assertEqual([item["result"] for item in attempts], ["REJECT", "ACCEPT"])
        self.assertEqual(len(calls), 2)
        self.assertIsNotNone(cycle._active_final_candidate_bundle)

    def test_final_candidate_attempts_are_capped_at_three(self):
        calls = []

        def fail_capture(camera, config, *, max_attempts):
            del camera, config, max_attempts
            calls.append("geometry")
            raise FinalCaptureQualityError("unusable")

        cycle, _, _, _, _, _, _, _ = self.build(final_capture_acquirer=fail_capture)
        cycle.projector.open()
        ranked = [replace(candidate(z, True), quality_score=1.0 - index / 10)
                  for index, z in enumerate((22, 21, 20, 19))]
        with self.assertRaisesRegex(Exception, "all ranked final candidates"):
            cycle._select_usable_final_candidate(
                SimpleNamespace(candidates=ranked, best_z=22),
                SimpleNamespace(roll_deg=25.0, pitch_deg=0.0),
                {"available": False, "board_quad": None, "marker_map": {}},
                SimpleNamespace(lighting_connected=False),
                self.root / "candidate_cap",
            )
        self.assertEqual(len(calls), 3)
        self.assertEqual(
            len(cycle._active_final_candidate_metadata["final_candidate_attempts"]), 3,
        )

    def test_final_candidate_optical_sequence_accept_keeps_led_on_for_final_rgb(self):
        events = []
        lighting = FakeLighting(events)

        def capture(camera, config, *, max_attempts):
            del config, max_attempts
            self.assertFalse(lighting.on)
            events.append("FINAL_GEOMETRY")
            frame = camera.capture()
            mask = np.full(frame.depth_mm.shape, 255, dtype=np.uint8)
            geometry = SurfaceGeometryResult(
                object_mask=mask, surface_mask=mask, patches=(),
                object_area_px=mask.size, surface_area_px=mask.size,
                surface_ratio=1.0, depth_valid_ratio=1.0,
                plane_inlier_ratio=1.0, plane_residual=.1,
                fov_edge_contact=False,
            )
            return GeometryReadyFinalCapture(frame, geometry, (), 1)

        cycle, _, _, _, _, _, _, _ = self.build(
            lighting=lighting, final_capture_acquirer=capture,
        )
        cycle.projector.open()
        lighting.connected = True
        mask = np.full((128, 128), 255, dtype=np.uint8)
        contour = SimpleNamespace(
            inspection_mask=mask, depth_object_contour_filled=mask,
            workspace_mask=mask, depth_main_component_mask=mask,
            board_plane_fit_mask=mask, plane_hypothesis_masks=(),
            selected_board_plane_hypothesis_index=None,
        )

        def build_roi(*args, **kwargs):
            del args, kwargs
            self.assertTrue(lighting.on)
            events.append("ROI_BUILD")
            return contour

        cycle.depth_contour_roi_builder = build_roi
        result = IntegratedCycleResult(str(self.root), 15, 0, lighting_connected=True)
        with patch(
            "src.integration.integrated_inspection_cycle.measure_patchability",
            return_value=([(0, 0)], 4096, 1.0),
        ):
            cycle._select_usable_final_candidate(
                SimpleNamespace(
                    candidates=[replace(candidate(22, True), quality_score=.9)],
                    best_z=22,
                ),
                SimpleNamespace(roll_deg=20.0, pitch_deg=0.0),
                {"available": False, "board_quad": None, "marker_map": {}},
                result, self.root / "candidate_optical_accept",
            )
        self.assertTrue(lighting.on)
        events.append("FINAL_RGB")
        self.assertNotIn(
            "LIGHTING_OFF",
            events[events.index("ROI_BUILD") + 1:events.index("FINAL_RGB")],
        )
        cycle._lighting_off(result)
        self.assertEqual(events[-1], "LIGHTING_OFF")

    def test_rejected_candidate_turns_led_off_before_next_candidate_rgb_fallback_on(self):
        events = []
        lighting = FakeLighting(events)

        def capture(camera, config, *, max_attempts):
            del config, max_attempts
            self.assertFalse(lighting.on)
            events.append("FINAL_GEOMETRY")
            frame = camera.capture()
            mask = np.full(frame.depth_mm.shape, 255, dtype=np.uint8)
            geometry = SurfaceGeometryResult(
                object_mask=mask, surface_mask=mask, patches=(),
                object_area_px=mask.size, surface_area_px=mask.size,
                surface_ratio=1.0, depth_valid_ratio=1.0,
                plane_inlier_ratio=1.0, plane_residual=.1,
                fov_edge_contact=False,
            )
            return GeometryReadyFinalCapture(frame, geometry, (), 1)

        cycle, _, _, _, _, _, _, _ = self.build(
            lighting=lighting, final_capture_acquirer=capture,
        )
        cycle.final_capture_inspection_config = replace(
            cycle.final_capture_inspection_config,
            surface_roi=replace(
                cycle.final_capture_inspection_config.surface_roi,
                min_patchable_ratio=.5,
            ),
        )
        cycle.projector.open(); lighting.connected = True
        mask = np.full((128, 128), 255, dtype=np.uint8)
        contour = SimpleNamespace(
            inspection_mask=mask, depth_object_contour_filled=mask,
            workspace_mask=mask, depth_main_component_mask=mask,
            board_plane_fit_mask=mask, plane_hypothesis_masks=(),
            selected_board_plane_hypothesis_index=None,
        )
        cycle.depth_contour_roi_builder = lambda *args, **kwargs: (
            events.append("ROI_BUILD") or contour
        )
        result = IntegratedCycleResult(str(self.root), 15, 0, lighting_connected=True)

        def rgb_fallback(*args, **kwargs):
            del args, kwargs
            self.assertTrue(lighting.on)
            events.append("RGB_FALLBACK")
            return None

        with patch(
            "src.integration.integrated_inspection_cycle.measure_patchability",
            side_effect=[([], 0, 0.0), ([(0, 0)], 4096, 1.0)],
        ), patch(
            "src.integration.integrated_inspection_cycle.build_rgb_seeded_roi",
            side_effect=rgb_fallback,
        ):
            cycle._select_usable_final_candidate(
                SimpleNamespace(candidates=[
                    replace(candidate(22, True), quality_score=.9),
                    replace(candidate(20, True), quality_score=.8),
                ], best_z=22),
                SimpleNamespace(roll_deg=20.0, pitch_deg=0.0),
                {"available": False, "board_quad": None, "marker_map": {}},
                result, self.root / "candidate_optical_reject",
            )
        first_roi = events.index("ROI_BUILD")
        second_geometry = events.index("FINAL_GEOMETRY", events.index("FINAL_GEOMETRY") + 1)
        self.assertIn("RGB_FALLBACK", events[first_roi:second_geometry])
        self.assertIn("LIGHTING_OFF", events[first_roi:second_geometry])
        self.assertTrue(lighting.on)


if __name__ == "__main__":
    unittest.main()
