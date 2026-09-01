from __future__ import annotations

import logging
from dataclasses import asdict
from datetime import datetime, timezone
from typing import Any

from src.integration.coordinate_contract import StructuredLightPaths
from src.platform.types import PlatformPoseCommand
from src.system.results import PoseInspectionResult, SystemInspectionResult
from src.system.states import SystemState


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class SystemController:
    """Owns all top-level state transitions and coordinates injected subsystems."""

    def __init__(self, *, conveyor: Any, structured_light_runner: Any,
                 structured_light_adapter: Any, pose_planner: Any, platform: Any,
                 automatic_z_search: Any, quality_sampler: Any,
                 surface_inspector: Any, anomaly_detector: Any,
                 safe_z_cm: float, mock: bool = False) -> None:
        self.conveyor = conveyor
        self.structured_light_runner = structured_light_runner
        self.structured_light_adapter = structured_light_adapter
        self.pose_planner = pose_planner
        self.platform = platform
        self.automatic_z_search = automatic_z_search
        self.quality_sampler = quality_sampler
        self.surface_inspector = surface_inspector
        self.anomaly_detector = anomaly_detector
        self.safe_z_cm = safe_z_cm
        self.mock = mock
        self.state = SystemState.STOPPED
        self.state_history: list[SystemState] = []
        self.logger = logging.getLogger(__name__)

    def _transition(self, state: SystemState) -> None:
        self.state = state
        self.state_history.append(state)
        self.logger.info("[STATE] %s", state.value)

    def run_once(self) -> SystemInspectionResult:
        started_at = _now()
        pose_results: list[PoseInspectionResult] = []
        failed_state: SystemState | None = None
        self.state_history = []
        try:
            self._transition(SystemState.INITIALIZING)
            self.conveyor.connect()
            self._transition(SystemState.READY)
            self._transition(SystemState.CONVEYOR_TO_INSPECTION)
            self.conveyor.move_to_inspection()
            self.conveyor.wait_until_stopped()

            self._transition(SystemState.STRUCTURED_LIGHT_SCAN)
            run_info = self.structured_light_runner.run_scan()
            structured = self.structured_light_adapter.from_directory(
                run_info.result_directory,
                paths=StructuredLightPaths(root=run_info.result_directory, current_run_dir=run_info.result_directory),
            )

            self._transition(SystemState.PLAN_POSES)
            plan = self.pose_planner.plan(structured)
            plan.validate()

            for pose in plan.poses:
                self.logger.info("[POSE] %s source=%s", pose.pose_id, pose.source)
                self._transition(SystemState.MOVE_SAFE_POSE)
                safe_command = PlatformPoseCommand(
                    z_cm=self.safe_z_cm,
                    roll_deg=float(pose.roll_deg or 0.0),
                    pitch_deg=float(pose.pitch_deg or 0.0),
                )
                self.platform.move_to(safe_command)
                telemetry = self.platform.wait_until_stable(timeout=5.0)
                if not telemetry.stable:
                    raise TimeoutError(f"Platform did not stabilize for {pose.pose_id}")
                self.logger.info("[PLATFORM] mock stable pose=%s", pose.pose_id)

                self._transition(SystemState.AUTO_Z_SEARCH)
                samples = self.quality_sampler.sample(pose)
                z_result = self.automatic_z_search.select_best(pose_id=pose.pose_id, samples=samples)
                if not z_result.success or z_result.best_z_cm is None:
                    raise RuntimeError(z_result.failure_reason or "NoValidInspectionZ")
                self.logger.info("[Z_SEARCH] best=%.2f", z_result.best_z_cm)
                self.platform.move_to(PlatformPoseCommand(
                    z_cm=z_result.best_z_cm,
                    roll_deg=float(pose.roll_deg or 0.0),
                    pitch_deg=float(pose.pitch_deg or 0.0),
                ))
                telemetry = self.platform.wait_until_stable(timeout=5.0)
                if not telemetry.stable:
                    raise TimeoutError(f"Platform did not stabilize at best Z for {pose.pose_id}")

                self._transition(SystemState.SURFACE_INSPECTION)
                surface = self.surface_inspector.inspect(pose, z_result.best_z_cm)
                if not surface.ready:
                    raise RuntimeError(f"Surface inspection not ready for {pose.pose_id}")

                self._transition(SystemState.ANOMALY_INSPECTION)
                anomaly = self.anomaly_detector.inspect(surface)
                self.logger.info("[ANOMALY] %s", anomaly.status)
                pose_results.append(PoseInspectionResult(
                    pose_id=pose.pose_id,
                    target_roll_deg=pose.roll_deg,
                    target_pitch_deg=pose.pitch_deg,
                    best_z_cm=z_result.best_z_cm,
                    z_search_samples=[asdict(sample) for sample in z_result.samples],
                    anomaly_result=asdict(anomaly),
                ))
                self._transition(SystemState.NEXT_POSE)

            self._transition(SystemState.FINALIZE)
            self._transition(SystemState.CONVEYOR_OUT)
            self.conveyor.move_out()
            self.conveyor.wait_until_stopped()
            self._transition(SystemState.COMPLETE)
            self._transition(SystemState.STOPPED)
            self.conveyor.close()
            self.logger.info("[RESULT] MOCK workflow complete")
            return SystemInspectionResult(
                success=True, started_at=started_at, finished_at=_now(),
                final_status="MOCK_COMPLETE", pose_results=pose_results,
                state_history=[state.value for state in self.state_history], mock=self.mock,
            )
        except Exception as exc:
            failed_state = self.state
            self.logger.error("[ERROR] state=%s %s: %s", failed_state.value, type(exc).__name__, exc)
            self._transition(SystemState.ERROR)
            self._transition(SystemState.STOPPED)
            try:
                self.conveyor.close()
            except Exception:
                self.logger.exception("[CONVEYOR] close failed during error handling")
            return SystemInspectionResult(
                success=False, started_at=started_at, finished_at=_now(),
                final_status="MOCK_FAILED" if self.mock else "FAILED",
                failed_state=failed_state.value, error_type=type(exc).__name__,
                error_message=str(exc), pose_results=pose_results,
                state_history=[state.value for state in self.state_history], mock=self.mock,
            )
