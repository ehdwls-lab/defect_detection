from __future__ import annotations

import json
import math
import os
import shutil
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Callable

import numpy as np

from src.config import InspectionConfig
from src.core.aruco_board import (
    create_aruco_detector,
    detect_markers,
    get_board_outer_quad,
)
from src.core.depth_contour_roi import (
    build_depth_external_contour_roi,
    save_depth_external_contour_roi_artifacts,
)
from src.core.inspection_mask import InspectionMaskResult, build_inspection_mask
from src.core.patch_extractor import measure_patchability
from src.core.rgb_seeded_roi import build_rgb_seeded_roi
from src.core.surface_roi import erode_surface_mask, mask_touches_frame_edge

from .projector_controller import ProjectorState
from .metric_pose_postprocess import postprocess_metric_pose_json
from .structured_light_runner import StructuredLightRunInfo
from .final_capture import (
    FINAL_GEOMETRY_MAX_ATTEMPTS,
    FINAL_RGB_WARMUP_FRAMES,
    acquire_geometry_ready_final_frame,
    acquire_warmed_final_rgb_frame,
    save_final_geometry_capture,
    save_final_rgb_capture,
)
from .inspection_failures import (
    AnomalyInputDataError,
    FinalCaptureQualityError,
    RecoverablePlaneInspectionError,
)
from .hybrid_inspection_roi import (
    HybridROIError,
    build_hybrid_inspection_roi,
    save_hybrid_roi_artifacts,
)
from .platform_limits import ORIENTATION_SAFE_Z_MIN_CM
from src.platform.motion_diagnostic import (
    ORIENTATION_TARGET_REACHED_TOLERANCE_DEG,
    Z_TARGET_REACHED_TOLERANCE_CM,
)
from src.inspection.adaptive_pose import (
    AdaptivePose, adaptive_pose_for_z, apply_adaptive_pose_transition,
)


class IntegratedCycleStage(str, Enum):
    CONNECT = "CONNECT"
    CONVEYOR_POSITION = "CONVEYOR_POSITION"
    STRUCTURED_LIGHT = "STRUCTURED_LIGHT"
    METRIC_POSE = "METRIC_POSE"
    PROJECTOR_COVER_CLOSE = "PROJECTOR_COVER_CLOSE"
    BEST_Z_SETTLED = "BEST_Z_SETTLED"
    LED_ON = "LED_ON"
    FINAL_GEOMETRY_CAPTURE = "FINAL_GEOMETRY_CAPTURE"
    FINAL_RGB_CAPTURE = "FINAL_RGB_CAPTURE"
    CONVEYOR_OUT = "CONVEYOR_OUT"
    # Retained for readers of historical stage logs; new production runs use
    # FINAL_GEOMETRY_CAPTURE and FINAL_RGB_CAPTURE.
    FRESH_FINAL_CAPTURE = "FRESH_FINAL_CAPTURE"
    ANOMALY_INFERENCE = "ANOMALY_INFERENCE"
    INITIALIZING = "INITIALIZING"
    PROJECTOR_BLACK = "PROJECTOR_BLACK"
    LIGHTING_OFF = "LIGHTING_OFF"
    PLATFORM_INITIALIZE = "PLATFORM_INITIALIZE"
    CONVEYOR_TO_INSPECTION = "CONVEYOR_TO_INSPECTION"
    STRUCTURED_LIGHT_SCAN = "STRUCTURED_LIGHT_SCAN"
    PLAN_DOMINANT_POSE = "PLAN_DOMINANT_POSE"
    MOVE_SAFE_Z = "MOVE_SAFE_Z"
    MOVE_ORIENTATION = "MOVE_ORIENTATION"
    MANUAL_LED_CHECKPOINT = "MANUAL_LED_CHECKPOINT"
    AUTOMATIC_Z = "AUTOMATIC_Z"
    LIGHTING_ON = "LIGHTING_ON"
    READY_FOR_ANOMALY = "READY_FOR_ANOMALY"
    COMPLETE = "COMPLETE"
    PARTIAL_COMPLETE = "PARTIAL_COMPLETE"
    FAILED = "FAILED"
    CLEANUP = "CLEANUP"


class IntegratedCycleError(RuntimeError):
    pass


class ManualLEDConfirmationError(IntegratedCycleError):
    pass


class PlatformInitializationError(IntegratedCycleError):
    pass


@dataclass(frozen=True)
class IntegratedCyclePaths:
    root: Path
    structured_light: Path
    automatic_z: Path
    telemetry: Path
    logs: Path

    @classmethod
    def create(cls, root: str | Path) -> "IntegratedCyclePaths":
        resolved = Path(root).expanduser().resolve()
        paths = cls(
            resolved,
            resolved / "structured_light",
            resolved / "automatic_z",
            resolved / "telemetry",
            resolved / "logs",
        )
        for directory in (
            paths.root, paths.structured_light, paths.automatic_z,
            paths.telemetry, paths.logs,
        ):
            directory.mkdir(parents=True, exist_ok=True)
        return paths


@dataclass
class IntegratedCycleResult:
    run_directory: str
    safe_z: float
    scan_z_requested: float
    success: bool = False
    stage: str = IntegratedCycleStage.INITIALIZING.value
    started_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    finished_at: str | None = None
    stage_history: list[str] = field(default_factory=list)
    initialization_success: bool = False
    startup_fast_path_used: bool = False
    initial_z_before: float | None = None
    initial_roll_before: float | None = None
    initial_pitch_before: float | None = None
    orientation_zeroed: bool = False
    scan_z_reached: bool = False
    conveyor_complete: bool = False
    structured_light_success: bool = False
    structured_light_run_directory: str | None = None
    pose_json: str | None = None
    selected_roll: float | None = None
    selected_pitch: float | None = None
    manual_led_confirmed: bool = False
    automatic_z_success: bool = False
    best_z: float | None = None
    automatic_z_stop_reason: str | None = None
    projector_final_state: str | None = None
    projector_state_after_close: str | None = None
    interrupted: bool = False
    error_type: str | None = None
    error_message: str | None = None
    cleanup_errors: list[str] = field(default_factory=list)
    conveyor_out_executed: bool = False
    anomaly_executed: bool = False
    lighting_connected: bool = False
    inspection_led_initial_off: bool = False
    inspection_led_on: bool = False
    inspection_led_off_at_end: bool = False
    lighting_error: str | None = None
    search_mode: str | None = None
    inspection_planes: list[dict[str, Any]] = field(default_factory=list)
    pose_planning_reached: bool = False
    detected_planes: int = 0
    planes_total: int = 0
    planes_completed: int = 0
    planes_failed: int = 0
    projector_cover_opened: bool = False
    projector_cover_closed: bool = False
    projector_cover_cleanup_attempted: bool = False
    automatic_z_led_mode: str = "OFF_DEPTH_GEOMETRY"
    overall_status: str = "FAILED"
    execution_status: str = "FAILED"
    quality_judgement: str = "RECHECK"
    planned_planes: int = 0
    completed_planes: int = 0
    failed_planes: int = 0
    final_rgb_warmup_frames: int = FINAL_RGB_WARMUP_FRAMES
    conveyor_out_direction: str | None = None
    conveyor_out_steps: int | None = None
    final_platform_roll_deg: float | None = None
    final_platform_pitch_deg: float | None = None
    final_platform_z_cm: float | None = None
    platform_parked: bool = False
    cover_open: bool = False
    conveyor_out: str = "NOT_RUN"
    cycle_complete: bool = False
    inspection_status: str = "FAILED"
    final_judgement: str = "RECHECK"
    cycle_transport_complete: bool = False

    def save(self, path: str | Path) -> None:
        target = Path(path).expanduser()
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(
            json.dumps(asdict(self), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )


def create_timestamped_run_directory(base: str | Path) -> Path:
    root = Path(base).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    stem = datetime.now().strftime("run_%Y%m%d_%H%M%S")
    candidate = root / stem
    suffix = 1
    while candidate.exists():
        candidate = root / f"{stem}_{suffix:02d}"
        suffix += 1
    candidate.mkdir(parents=False)
    return candidate


def _inside(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


def resolve_current_pose_json(run_info: StructuredLightRunInfo) -> Path:
    """Resolve only the pose JSON belonging to this structured-light run."""
    run_root = Path(run_info.result_directory).expanduser().resolve()
    configured = run_info.pose_json_path
    if configured is None and run_info.manifest_path is not None:
        manifest_path = Path(run_info.manifest_path).expanduser().resolve()
        with manifest_path.open("r", encoding="utf-8") as handle:
            manifest = json.load(handle)
        value = manifest.get("artifacts", {}).get("pose_json")
        if value:
            configured = Path(value)
    if configured is None:
        candidates = sorted(run_root.rglob("FINAL_DC_MASK_PHASE*_pose.json"))
        if len(candidates) != 1:
            raise IntegratedCycleError(
                "current structured-light run must contain exactly one "
                f"FINAL_DC_MASK_PHASE*_pose.json; found {len(candidates)}"
            )
        configured = candidates[0]
    pose_json = Path(configured).expanduser().resolve()
    if not pose_json.is_file():
        raise IntegratedCycleError(f"pose JSON does not exist: {pose_json}")
    if not _inside(pose_json, run_root):
        raise IntegratedCycleError("pose JSON is outside the current structured-light run")
    return pose_json


class IntegratedInspectionCycle:
    """Independent, explicitly gated partial hardware diagnostic cycle."""

    def __init__(
        self, *, conveyor: Any, structured_light_runner: Any, pose_planner: Any,
        projector: Any, platform: Any, motion_diagnostic: Any, camera: Any,
        automatic_z_search: Any, scan_z: float, safe_z: float,
        run_directory: str | Path,
        led_checkpoint: Callable[[], bool] | None = None,
        lighting: Any | None = None,
        metric_pose_postprocessor: Callable[..., dict[str, Any]] = postprocess_metric_pose_json,
        anomaly_detector: Any | None = None,
        final_capture_acquirer: Callable[..., Any] = acquire_geometry_ready_final_frame,
        final_capture_inspection_config: InspectionConfig | None = None,
        final_geometry_saver: Callable[..., Any] = save_final_geometry_capture,
        final_rgb_acquirer: Callable[..., Any] = acquire_warmed_final_rgb_frame,
        final_rgb_saver: Callable[..., Any] = save_final_rgb_capture,
        final_rgb_warmup_frames: int = FINAL_RGB_WARMUP_FRAMES,
        conveyor_out_enabled: bool = False,
        aruco_detector_factory: Callable[[], Any] = create_aruco_detector,
        hybrid_roi_builder: Callable[..., Any] = build_hybrid_inspection_roi,
        depth_contour_roi_builder: Callable[..., Any] = build_depth_external_contour_roi,
        depth_contour_roi_saver: Callable[..., Any] = save_depth_external_contour_roi_artifacts,
    ) -> None:
        scan_z = float(scan_z)
        safe_z = float(safe_z)
        if not math.isfinite(scan_z):
            raise ValueError("scan_z must be finite")
        if not math.isfinite(safe_z):
            raise ValueError("safe_z must be finite")
        if not math.isclose(scan_z, 0.0, rel_tol=0.0, abs_tol=1e-9):
            raise ValueError("production structured-light scan_z must be 0")
        if safe_z < ORIENTATION_SAFE_Z_MIN_CM:
            raise ValueError(
                "production orientation safe_z must be at least "
                f"{ORIENTATION_SAFE_Z_MIN_CM:g} cm"
            )
        if (isinstance(final_rgb_warmup_frames, bool)
                or not isinstance(final_rgb_warmup_frames, int)
                or final_rgb_warmup_frames < 0):
            raise ValueError("final_rgb_warmup_frames must be a non-negative integer")
        self.conveyor = conveyor
        self.structured_light_runner = structured_light_runner
        self.pose_planner = pose_planner
        self.projector = projector
        self.platform = platform
        self.motion_diagnostic = motion_diagnostic
        self.camera = camera
        self.automatic_z_search = automatic_z_search
        self.scan_z = scan_z
        self.safe_z = safe_z
        self.paths = IntegratedCyclePaths.create(run_directory)
        self.led_checkpoint = led_checkpoint
        self.lighting = lighting
        self.metric_pose_postprocessor = metric_pose_postprocessor
        self.anomaly_detector = anomaly_detector
        self.final_capture_acquirer = final_capture_acquirer
        self.final_geometry_saver = final_geometry_saver
        self.final_rgb_acquirer = final_rgb_acquirer
        self.final_rgb_saver = final_rgb_saver
        self.final_rgb_warmup_frames = final_rgb_warmup_frames
        self.conveyor_out_enabled = bool(conveyor_out_enabled)
        self.aruco_detector_factory = aruco_detector_factory
        self.hybrid_roi_builder = hybrid_roi_builder
        self.depth_contour_roi_builder = depth_contour_roi_builder
        self.depth_contour_roi_saver = depth_contour_roi_saver
        self._active_final_candidate_metadata: dict[str, Any] | None = None
        self._active_final_candidate_bundle: dict[str, Any] | None = None
        automatic_z_evaluator = getattr(automatic_z_search, "evaluator", None)
        self.final_capture_inspection_config = (
            final_capture_inspection_config
            or getattr(automatic_z_evaluator, "config", None)
            or getattr(anomaly_detector, "inspection_config", None)
            or InspectionConfig.default()
        )
        self._active_pose_progress: tuple[int, int] | None = None
        self._active_pose_reported = False

    @staticmethod
    def _require_orientation_safe_height(telemetry: Any) -> None:
        z_cm = float(telemetry.z_cm)
        if not math.isfinite(z_cm) or z_cm < ORIENTATION_SAFE_Z_MIN_CM:
            raise IntegratedCycleError(
                "orientation blocked: fresh platform Z telemetry "
                f"{z_cm:g} cm is below the production minimum "
                f"{ORIENTATION_SAFE_Z_MIN_CM:g} cm"
            )

    @staticmethod
    def _projector_state(projector: Any) -> str | None:
        state = getattr(projector, "state", None)
        return state.value if hasattr(state, "value") else (str(state) if state is not None else None)

    def _show_and_require_black(self) -> None:
        self.projector.show_black()
        if getattr(self.projector, "state", None) is not ProjectorState.BLACK:
            raise IntegratedCycleError("projector did not enter BLACK state")

    def _lighting_off(self, result: IntegratedCycleResult) -> None:
        if self.lighting is None:
            return
        try:
            self.lighting.inspection_off()
        except Exception as exc:
            result.lighting_error = f"{type(exc).__name__}: {exc}"
            raise
        result.inspection_led_initial_off = True
        print("[LIGHTING] LED=OFF")
        self._stage(result, IntegratedCycleStage.LIGHTING_OFF)

    def _lighting_on(self, result: IntegratedCycleResult) -> None:
        if self.lighting is None:
            return
        try:
            self.lighting.inspection_on()
        except Exception as exc:
            result.lighting_error = f"{type(exc).__name__}: {exc}"
            raise
        result.inspection_led_on = True
        print("[LIGHTING] LED=NEUTRAL_WHITE")
        self._stage(result, IntegratedCycleStage.LIGHTING_ON)
        self._stage(result, IntegratedCycleStage.LED_ON)

    def _cover_open(self, result: IntegratedCycleResult) -> None:
        if self.lighting is not None and hasattr(self.lighting, "projector_cover_open"):
            self.lighting.projector_cover_open()
            result.projector_cover_opened = True

    def _cover_close(self, result: IntegratedCycleResult) -> None:
        if self.lighting is not None and hasattr(self.lighting, "projector_cover_close"):
            self.lighting.projector_cover_close()
            result.projector_cover_closed = True
        self._stage(result, IntegratedCycleStage.PROJECTOR_COVER_CLOSE)

    @staticmethod
    def _stage(result: IntegratedCycleResult, stage: IntegratedCycleStage) -> None:
        result.stage = stage.value
        result.stage_history.append(stage.value)

    @staticmethod
    def _pose_clamped(pose: Any) -> bool:
        if "clamped" in pose.metadata:
            return bool(pose.metadata["clamped"])
        raw_roll = float(pose.metadata.get("raw_roll", pose.roll_deg))
        raw_pitch = float(pose.metadata.get("raw_pitch", pose.pitch_deg))
        return not (
            math.isclose(raw_roll, float(pose.roll_deg), rel_tol=0.0, abs_tol=1e-9)
            and math.isclose(raw_pitch, float(pose.pitch_deg), rel_tol=0.0, abs_tol=1e-9)
        )

    @classmethod
    def _print_pose_plan_summary(cls, plan: Any) -> None:
        detected = int(plan.metadata.get("detected_plane_count", plan.metadata.get("parsed_plane_count", 0)))
        total = len(plan.poses)
        mode = str(plan.metadata.get("selection_policy", "unknown"))
        print("\nSTRUCTURED-LIGHT INSPECTION PLAN")
        print(f"pose_plan_mode : {mode}")
        print(f"Detected planes : {detected}")
        print(f"Metric-valid planes : {int(plan.metadata.get('metric_valid_plane_count', 0))}")
        print(f"Planned poses   : {total}")
        print(f"Reachable poses : {int(plan.metadata.get('reachable_pose_count', total))}")
        for order, pose in enumerate(plan.poses, 1):
            metadata = pose.metadata
            source_index = int(metadata.get("source_plane_index", order - 1))
            dominant = bool(metadata.get("dominant", metadata.get("plane_role") == "dominant"))
            raw_roll = float(metadata.get("raw_roll", pose.roll_deg))
            raw_pitch = float(metadata.get("raw_pitch", pose.pitch_deg))
            label = "Selected plane" if mode == "dominant_only" else f"Pose {order}/{total}"
            print(f"\n{label}")
            print(f"  inspection_order = {order}")
            print(f"  original_plane_index = {source_index}")
            print(f"  original_plane_name = {pose.pose_id}")
            print(f"  dominant = {dominant}")
            print(f"  points_count = {metadata.get('point_count')}")
            print(f"  legacy_phase_raw_roll = {raw_roll:.6f}")
            print(f"  legacy_phase_raw_pitch = {raw_pitch:.6f}")
            print(f"  requested_roll = {float(pose.roll_deg):.6f}")
            print(f"  requested_pitch = {float(pose.pitch_deg):.6f}")
            print(f"  applied_roll = {float(pose.roll_deg):.6f}")
            print(f"  applied_pitch = {float(pose.pitch_deg):.6f}")
            print(f"  clamped = {cls._pose_clamped(pose)}")
            metric = metadata.get("metric_pose", {})
            print(f"  depth_metric_points = {int(metric.get('depth_points_count', 0))}")
            print(f"  depth_coverage = {float(metric.get('depth_coverage', 0.0)):.6f}")
            print(f"  alignment_mode = {metric.get('alignment_mode', 'FULL')}")
            print(
                "  desired_roll/pitch = "
                f"{metric.get('desired_target_roll_deg')} / {metric.get('desired_target_pitch_deg')}"
            )
            print(
                "  commanded_roll/pitch = "
                f"{metric.get('commanded_target_roll_deg', pose.roll_deg)} / "
                f"{metric.get('commanded_target_pitch_deg', pose.pitch_deg)}"
            )
            print(
                "  predicted_residual_angle_deg = "
                f"{metric.get('predicted_residual_angle_deg', 0.0)}"
            )
            print("  reachable = YES")
        for rejected in plan.metadata.get("rejected_planes", []):
            print(f"\nRejected plane : {rejected.get('plane_name', 'unknown')}")
            print(f"  SL points = {int(rejected.get('sl_points', 0))}")
            print(f"  Depth metric points = {int(rejected.get('depth_metric_points', 0))}")
            print(f"  Depth coverage = {float(rejected.get('depth_coverage', 0.0)):.6f}")
            print(f"  Metric Roll/Pitch = {rejected.get('metric_roll')} / {rejected.get('metric_pitch')}")
            print("  Reachable = NO")
            print(f"  Reject reason = {rejected.get('reject_reason', 'unknown')}")
        print(f"\nTOTAL INSPECTION POSES = {total}\n")

    def _print_pose_start(self, index: int, total: int) -> None:
        self._active_pose_progress = (index, total)
        self._active_pose_reported = False
        print(f"\nINSPECTION POSE {index} / {total}")

    def _print_pose_end(self, index: int, total: int, status: str, best_z: float | None) -> None:
        print(f"POSE {index} / {total}")
        print(f"status = {status}")
        print(f"best_z = {best_z}")
        self._active_pose_reported = True
        self._active_pose_progress = None

    @staticmethod
    def _print_final_pose_summary(result: IntegratedCycleResult) -> None:
        print("\nINSPECTION POSE SUMMARY")
        if result.pose_planning_reached:
            print(f"Detected planes = {result.detected_planes}")
            print(f"Planned poses = {result.planes_total}")
        else:
            print("Detected planes = NOT REACHED")
            print("Planned poses = NOT REACHED")
        print(f"Completed poses = {result.planes_completed}")
        print(f"Failed poses = {result.planes_failed}")
        print(f"Execution status = {result.execution_status}")
        print(f"Quality judgement = {result.quality_judgement}")

    def _archive_structured_light(self, run_info: StructuredLightRunInfo, pose_json: Path | None = None) -> None:
        source_run = Path(run_info.result_directory).expanduser().resolve()
        source_link = self.paths.structured_light / "current_run"
        link_error = None
        if source_run != source_link.resolve():
            try:
                if not source_link.exists() and not source_link.is_symlink():
                    link_target = os.path.relpath(source_run, start=source_link.parent)
                    source_link.symlink_to(link_target, target_is_directory=True)
            except OSError as exc:
                link_error = f"{type(exc).__name__}: {exc}"
        payload = {
            "run_id": run_info.run_id,
            "result_directory": str(source_run),
            "return_code": run_info.return_code,
            "manifest_path": str(run_info.manifest_path) if run_info.manifest_path else None,
            "pose_json_path": str(pose_json or run_info.pose_json_path) if (pose_json or run_info.pose_json_path) else None,
            "current_run_link": str(source_link) if source_link.is_symlink() else None,
            "current_run_link_error": link_error,
        }
        (self.paths.structured_light / "run_info.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8",
        )
        if run_info.manifest_path is not None and Path(run_info.manifest_path).is_file():
            shutil.copy2(run_info.manifest_path, self.paths.structured_light / "structured_light_manifest.json")
        if pose_json is not None:
            shutil.copy2(pose_json, self.paths.structured_light / pose_json.name)
        (self.paths.logs / "structured_light_stdout.log").write_text(run_info.stdout, encoding="utf-8")
        (self.paths.logs / "structured_light_stderr.log").write_text(run_info.stderr, encoding="utf-8")

    def _save_automatic_z(self, result: Any, root: Path | None = None) -> None:
        target = self.paths.automatic_z if root is None else root
        target.mkdir(parents=True, exist_ok=True)
        result.save(target / "result.json")
        result.save(target / "candidates.csv")

    @staticmethod
    def _automatic_z_diagnostics(z_result: Any) -> dict[str, Any]:
        metrics = getattr(z_result, "best_metrics", None)
        if metrics is None and getattr(z_result, "candidates", ()):
            metrics = asdict(z_result.candidates[-1])
        metrics = metrics or {}
        return {
            "depth_valid_ratio": metrics.get("depth_valid_ratio"),
            "plane_inlier_ratio": metrics.get("plane_inlier_ratio"),
            "plane_residual": metrics.get("plane_residual"),
            "object_area_px": metrics.get("object_area_px"),
            "surface_area_px": metrics.get("surface_area_px"),
            "surface_ratio": metrics.get("surface_ratio"),
            "usable_patch_count": metrics.get(
                "usable_patch_count", metrics.get("surface_patch_count"),
            ),
            "automatic_z_candidates": [
                {
                    "z_cm": item.z_command,
                    "readiness_pass": item.readiness_pass,
                    "quality_score": item.quality_score,
                    "depth_valid_ratio": item.depth_valid_ratio,
                    "object_area_px": item.object_area_px,
                    "surface_area_px": item.surface_area_px,
                    "surface_ratio": item.surface_ratio,
                    "diagnostic_dir": item.diagnostic_dir,
                    "depth_p05_mm": item.depth_p05_mm,
                    "depth_median_mm": item.depth_median,
                    "depth_p95_mm": item.depth_p95_mm,
                    "board_roi_depth_valid_ratio": item.board_roi_depth_valid_ratio,
                    "requested_roll_deg": item.requested_roll_deg,
                    "requested_pitch_deg": item.requested_pitch_deg,
                    "applied_roll_deg": item.applied_roll_deg,
                    "applied_pitch_deg": item.applied_pitch_deg,
                    "combined_tilt_deg": item.combined_tilt_deg,
                    "max_combined_tilt_deg": item.max_combined_tilt_deg,
                    "tilt_scale": item.tilt_scale,
                }
                for item in getattr(z_result, "candidates", ())
            ],
            "z_selection_policy": getattr(z_result, "selection_policy", None),
            "selected_best_z": getattr(z_result, "best_z", None),
            "selected_best_z_quality_score": getattr(
                z_result, "selected_best_z_quality_score", metrics.get("quality_score"),
            ),
        }

    @staticmethod
    def _new_plane_result(pose: Any, plane_index: int, automatic_root: Path) -> dict[str, Any]:
        metric = pose.metadata.get("metric_pose", {})
        roll, pitch = float(pose.roll_deg), float(pose.pitch_deg)
        return {
            "inspection_order": plane_index + 1,
            "plane_index": int(pose.metadata.get("source_plane_index", plane_index)),
            "plane_name": pose.pose_id,
            "alignment_mode": pose.metadata.get(
                "alignment_mode", metric.get("alignment_mode", "FULL"),
            ),
            "desired_roll_deg": metric.get("desired_target_roll_deg", roll),
            "desired_pitch_deg": metric.get("desired_target_pitch_deg", pitch),
            "commanded_roll_deg": roll,
            "commanded_pitch_deg": pitch,
            # Backward-compatible names retained for existing consumers.
            "commanded_roll": roll,
            "commanded_pitch": pitch,
            "actual_platform_roll_deg": None,
            "actual_platform_pitch_deg": None,
            "actual_platform_z_cm": None,
            "status": "PENDING",
            "inspection_judgement": "RECHECK",
            "quality_judgement": "RECHECK",
            "failure_stage": None,
            "failure_reason": None,
            "automatic_z_success": False,
            "best_z": None,
            "automatic_z_stop_reason": None,
            "selected_candidate_rank": None,
            "selected_z": None,
            "selected_roll": None,
            "selected_pitch": None,
            "selected_quality_score": None,
            "final_candidate_attempts": [],
            "automatic_z_result_path": str(automatic_root / "result.json"),
            "anomaly_success": False,
            "anomaly_executed": False,
            "anomaly_result": None,
            "anomaly_result_path": None,
            "classification": None,
            "candidate_frames_reused_for_anomaly": False,
            "final_capture_attempts": 0,
            "final_capture_accepted_attempt": None,
            "final_capture_depth_valid_ratio": None,
            "final_capture_plane_inlier_ratio": None,
            "final_capture_plane_residual": None,
            "final_capture_object_area_px": 0,
            "final_capture_surface_area_px": 0,
            "final_capture_attempt_diagnostics": [],
            "geometry_capture_attempts": 0,
            "geometry_accepted_attempt": None,
            "geometry_capture_attempt_diagnostics": [],
            "final_rgb_warmup_frames": 0,
            "geometry_rgb_coordinate_contract": "Orbbec aligned depth to COLOR_STREAM",
            "final_depth_path": None,
            "final_rgb_path": None,
            "final_ir_path": None,
            "object_mask_path": None,
            "surface_mask_path": None,
            "surface_geometry_overlay_path": None,
            "inspection_mask_path": None,
            "inspection_mask_overlay_path": None,
            "surface_patch_overlay_path": None,
            "commanded_rp": {"roll_deg": roll, "pitch_deg": pitch},
            "actual_rp": {"roll_deg": None, "pitch_deg": None},
            "depth_valid_ratio": None,
            "plane_inlier_ratio": None,
            "plane_residual": None,
            "object_area_px": None,
            "surface_area_px": None,
            "surface_ratio": None,
            "inspection_area_px": None,
            "inspection_to_surface_ratio": None,
            "inspection_to_object_ratio": None,
            "anomaly_roi_type": None,
            "workspace_source": None,
            "workspace_area_px": None,
            "depth_candidate_area_px": None,
            "depth_main_component_area_px": None,
            "filled_object_area_px": None,
            "fill_gain_px": None,
            "fill_gain_ratio": None,
            "depth_contour_roi_artifacts": {},
            "selected_inspection_patch_count": None,
            "selected_surface_patch_count": None,
            "aruco_roi_status": "NOT_AVAILABLE",
            "aruco_marker_ids": [],
            "aruco_marker_count": 0,
            "aruco_fallback_reason": None,
            "board_roi_area_px": None,
            "board_plane_valid": False,
            "board_plane_inlier_ratio": None,
            "board_plane_residual_mm": None,
            "depth_object_area_px": None,
            "depth_unknown_area_px": None,
            "hybrid_inspection_area_px": None,
            "hybrid_to_depth_object_ratio": None,
            "depth_p05_mm": None,
            "depth_median_mm": None,
            "depth_p95_mm": None,
            "board_roi_depth_valid_ratio": None,
            "roi_diagnostic_artifacts": {},
            "usable_patch_count": None,
            "pose_metadata": pose.metadata,
        }

    @staticmethod
    def _save_plane_result(plane_root: Path, plane_result: dict[str, Any]) -> None:
        plane_root.mkdir(parents=True, exist_ok=True)
        (plane_root / "result.json").write_text(
            json.dumps(plane_result, ensure_ascii=False, indent=2), encoding="utf-8",
        )

    def _recover_after_plane(
        self, result: IntegratedCycleResult, plane_result: dict[str, Any], plane_root: Path,
    ) -> None:
        """Restore the safe inter-pose state; any failure here is fatal."""
        if self.lighting is not None:
            self.lighting.inspection_off()
            print("[LIGHTING] LED=OFF")
        self._show_and_require_black()
        self._stage(result, IntegratedCycleStage.MOVE_SAFE_Z)
        adaptive_motion = getattr(self.automatic_z_search.config, "search_mode", "explicit") == "adaptive"
        if adaptive_motion:
            entry_z = float(self.final_capture_inspection_config.quality.tilt_entry_z_cm)
            recovered = self.motion_diagnostic.execute_z(entry_z)
            recovered = self.motion_diagnostic.execute_orientation(
                roll_deg=0.0, pitch_deg=0.0, before=recovered, ack_safe_height=True,
            )
            if not math.isclose(self.safe_z, entry_z, rel_tol=0.0, abs_tol=1e-9):
                recovered = self.motion_diagnostic.execute_z(self.safe_z)
        else:
            recovered = self.motion_diagnostic.execute_z(self.safe_z)
        self._require_orientation_safe_height(recovered)
        plane_result.update(
            recovery_led_off=True,
            recovery_projector_black=True,
            recovery_safe_z=True,
            recovery_platform_roll_deg=float(recovered.roll_deg),
            recovery_platform_pitch_deg=float(recovered.pitch_deg),
            recovery_platform_z_cm=float(recovered.z_cm),
        )
        self._save_plane_result(plane_root, plane_result)

    def _adaptive_pose_for_z(self, pose: Any, z_cm: float) -> AdaptivePose:
        quality = self.final_capture_inspection_config.quality
        return adaptive_pose_for_z(
            z_cm, float(pose.roll_deg), float(pose.pitch_deg),
            roll_limit_deg=quality.inspection_roll_limit_deg,
            pitch_limit_deg=quality.inspection_pitch_limit_deg,
            envelope=quality.tilt_envelope,
        )

    def _apply_adaptive_pose(self, target: AdaptivePose, previous: AdaptivePose | None) -> None:
        apply_adaptive_pose_transition(self.motion_diagnostic, target, previous)

    def _move_to_candidate(self, target: AdaptivePose, current: AdaptivePose | None) -> None:
        """Move between final candidates with the safe order for each Z direction."""
        before = self.motion_diagnostic.read_before()
        if current is not None and target.z_cm < current.z_cm:
            self.motion_diagnostic.execute_orientation(
                roll_deg=target.applied_roll_deg, pitch_deg=target.applied_pitch_deg,
                before=before, ack_safe_height=True,
            )
            self.motion_diagnostic.execute_z(target.z_cm)
        elif current is not None and target.z_cm > current.z_cm:
            raised = self.motion_diagnostic.execute_z(target.z_cm)
            self.motion_diagnostic.execute_orientation(
                roll_deg=target.applied_roll_deg, pitch_deg=target.applied_pitch_deg,
                before=raised, ack_safe_height=True,
            )
        else:
            self.motion_diagnostic.execute_orientation(
                roll_deg=target.applied_roll_deg, pitch_deg=target.applied_pitch_deg,
                before=before, ack_safe_height=True,
            )

    def _candidate_pose(self, candidate: Any, pose: Any) -> AdaptivePose:
        quality = self.final_capture_inspection_config.quality
        return adaptive_pose_for_z(
            float(candidate.z_command), float(pose.roll_deg), float(pose.pitch_deg),
            roll_limit_deg=quality.inspection_roll_limit_deg,
            pitch_limit_deg=quality.inspection_pitch_limit_deg,
            envelope=quality.tilt_envelope,
        )

    def _select_usable_final_candidate(
        self, z_result: Any, pose: Any, aruco_context: dict[str, Any],
        result: IntegratedCycleResult, final_root: Path,
    ) -> Any:
        """Probe ranked PASS candidates; return the first usable final-ROI candidate."""
        candidates = [item for item in getattr(z_result, "candidates", ()) if item.accepted]
        candidates.sort(key=lambda item: (
            -(float(item.quality_score) if item.quality_score is not None else float("-inf")),
        ))
        quality = self.final_capture_inspection_config.quality
        limit = max(1, int(quality.max_final_candidate_attempts))
        if not candidates:
            raise RecoverablePlaneInspectionError(
                IntegratedCycleStage.AUTOMATIC_Z.value, "no passing candidates available",
            )
        best_candidate = next(
            (item for item in candidates if math.isclose(
                float(item.z_command), float(z_result.best_z), abs_tol=1e-9,
            )),
            candidates[0],
        )
        current = self._candidate_pose(best_candidate, pose)
        selected = None
        attempt_records: list[dict[str, Any]] = []
        for rank, candidate in enumerate(candidates[:limit], start=1):
            print(f"[FINAL CANDIDATE {rank}/{min(limit, len(candidates))}]")
            target = self._candidate_pose(candidate, pose)
            self._move_to_candidate(target, current)
            current = target
            record: dict[str, Any] = {
                "rank": rank, "z": candidate.z_command,
                "quality_score": candidate.quality_score,
                "requested_roll": float(pose.roll_deg),
                "requested_pitch": float(pose.pitch_deg),
                "applied_roll": target.applied_roll_deg,
                "applied_pitch": target.applied_pitch_deg,
                "combined_tilt": target.combined_tilt_deg,
                "max_allowed_tilt": target.max_combined_tilt_deg,
                "safety_scale": target.tilt_scale,
                "geometry_metrics": {
                    key: getattr(candidate, key, None) for key in (
                        "depth_valid_ratio", "plane_inlier_ratio",
                        "plane_inlier_residual_mm", "usable_area_ratio",
                        "usable_width_ratio", "usable_height_ratio",
                    )
                },
            }
            try:
                if self.lighting is not None and result.lighting_connected:
                    self._lighting_off(result)
                print("[FINAL GEOMETRY] LED=OFF")
                geometry_capture = self.final_capture_acquirer(
                    self.camera, self.final_capture_inspection_config,
                    max_attempts=FINAL_GEOMETRY_MAX_ATTEMPTS,
                )
                record["final_geometry"] = "PASS"
                depth = np.asarray(geometry_capture.frame.depth_mm)
                height, width = depth.shape
                intrinsics = self.camera.color_intrinsics(width, height)
                if self.lighting is not None and result.lighting_connected:
                    self._lighting_on(result)
                roi_config = self.final_capture_inspection_config.surface_roi
                for index in range(self.final_rgb_warmup_frames):
                    self.camera.capture()
                    print(
                        f"[FINAL ROI] warmup={index + 1}/"
                        f"{self.final_rgb_warmup_frames} discard"
                    )
                roi_frames = [np.asarray(self.camera.capture().depth_mm)
                              for _ in range(roi_config.roi_depth_frames)]
                contour_kwargs = dict(
                    marker_map=aruco_context["marker_map"], intrinsics=intrinsics,
                    depth_frames=roi_frames, min_votes=roi_config.roi_min_votes,
                    current_platform_roll_deg=float(target.applied_roll_deg),
                    current_platform_pitch_deg=float(target.applied_pitch_deg),
                    commanded_platform_roll_deg=float(target.applied_roll_deg),
                    commanded_platform_pitch_deg=float(target.applied_pitch_deg),
                )
                try:
                    contour = self.depth_contour_roi_builder(
                        roi_frames[0], geometry_capture.frame.color_bgr.shape,
                        self.final_capture_inspection_config,
                        board_quad=(aruco_context["board_quad"]
                                    if aruco_context["available"] else None),
                        **contour_kwargs,
                    )
                except Exception:
                    if not aruco_context["available"]:
                        raise
                    contour = self.depth_contour_roi_builder(
                        roi_frames[0], geometry_capture.frame.color_bgr.shape,
                        self.final_capture_inspection_config,
                        board_quad=None, **contour_kwargs,
                    )
                patches, _, patchable = measure_patchability(
                    contour.inspection_mask,
                    self.final_capture_inspection_config.patch.patch_size,
                    self.final_capture_inspection_config.patch.patch_stride,
                    self.final_capture_inspection_config.patch.patch_mask_coverage,
                )
                accepted_mask = contour.inspection_mask
                accepted_filled_mask = contour.depth_object_contour_filled
                min_patchable_ratio = roi_config.min_patchable_ratio
                print(
                    "[INSPECTION ROI] "
                    f"type=depth_external_contour_fill "
                    f"inspection_px={int(np.count_nonzero(accepted_mask))}"
                )
                print(
                    "[PATCHABILITY] "
                    f"ratio={patchable:.6f} selected_patches={len(patches)}"
                )
                record.update(
                    final_geometry="PASS",
                    roi="PASS" if patchable >= min_patchable_ratio else "FAIL",
                    patchable_ratio=patchable, selected_patches=len(patches),
                )
                if patchable < min_patchable_ratio:
                    board_reference = (
                        contour.plane_hypothesis_masks[
                            contour.selected_board_plane_hypothesis_index
                        ]
                        if contour.selected_board_plane_hypothesis_index is not None
                        and contour.selected_board_plane_hypothesis_index < len(
                            contour.plane_hypothesis_masks
                        ) else contour.board_plane_fit_mask
                    )
                    rgb_frame = self.camera.capture()
                    rgb_fallback = (
                        build_rgb_seeded_roi(
                            np.asarray(rgb_frame.color_bgr), contour.workspace_mask,
                            board_reference, contour.depth_main_component_mask,
                            self.final_capture_inspection_config.surface_roi,
                        ) if board_reference is not None else None
                    )
                    if rgb_fallback is not None:
                        fallback_mask = erode_surface_mask(
                            rgb_fallback.object_mask,
                            self.final_capture_inspection_config.surface_roi.boundary_margin_px,
                        )
                        fallback_mask = np.where(
                            (fallback_mask > 0) & (contour.workspace_mask > 0), 255, 0,
                        ).astype(np.uint8)
                        fallback_patches, _, fallback_patchable = measure_patchability(
                            fallback_mask,
                            self.final_capture_inspection_config.patch.patch_size,
                            self.final_capture_inspection_config.patch.patch_stride,
                            self.final_capture_inspection_config.patch.patch_mask_coverage,
                        )
                        if (
                            fallback_patchable >= min_patchable_ratio
                            and not mask_touches_frame_edge(
                                fallback_mask,
                                self.final_capture_inspection_config.surface_roi.fov_edge_margin_px,
                            )
                            and np.count_nonzero(rgb_fallback.object_mask) / max(
                                1, np.count_nonzero(contour.workspace_mask)
                            ) <= roi_config.max_object_area_ratio
                        ):
                            patchable = fallback_patchable
                            accepted_mask = fallback_mask
                            accepted_filled_mask = rgb_fallback.object_mask
                            record.update(
                                roi="PASS", patchable_ratio=patchable,
                                selected_patches=len(fallback_patches),
                                rgb_fallback_used=True,
                            )
                if patchable >= min_patchable_ratio:
                    selected = candidate
                    record["result"] = "ACCEPT"
                    self._active_final_candidate_bundle = {
                        "geometry_capture": geometry_capture,
                        "depth_contour_roi": contour,
                        "selected_mask": accepted_mask,
                        "filled_object_mask": accepted_filled_mask,
                        "rgb_fallback_used": bool(record.get("rgb_fallback_used", False)),
                    }
                else:
                    record.update(
                        result="REJECT",
                        reject_reason=(
                            f"patchable_ratio below {min_patchable_ratio}"
                        ),
                    )
            except Exception as exc:
                record.update(
                    roi="FAIL", result="REJECT",
                    reject_reason=f"{type(exc).__name__}: {exc}",
                )
                record.setdefault("final_geometry", "FAIL")
            finally:
                if (selected is None and self.lighting is not None
                        and result.lighting_connected):
                    self._lighting_off(result)
            attempt_records.append(record)
            for key in ("z", "applied_roll", "applied_pitch", "quality_score",
                        "final_geometry", "roi", "patchable_ratio", "result"):
                if key in record:
                    print(f"{key}={record[key]}")
            if selected is not None:
                break
        final_root.mkdir(parents=True, exist_ok=True)
        (final_root / "final_candidate_attempts.json").write_text(
            json.dumps(attempt_records, ensure_ascii=False, indent=2), encoding="utf-8",
        )
        result_record = {
            "selected_candidate_rank": None,
            "selected_z": None,
            "selected_roll": None,
            "selected_pitch": None,
            "selected_quality_score": None,
            "final_candidate_attempts": attempt_records,
        }
        if selected is None:
            self._active_final_candidate_bundle = None
            self._active_final_candidate_metadata = result_record
            raise RecoverablePlaneInspectionError(
                IntegratedCycleStage.FINAL_GEOMETRY_CAPTURE.value,
                "all ranked final candidates were unusable",
            )
        selected_record = next(item for item in attempt_records if item["result"] == "ACCEPT")
        result_record.update(
            selected_candidate_rank=selected_record["rank"],
            selected_z=selected_record["z"],
            selected_roll=selected_record["applied_roll"],
            selected_pitch=selected_record["applied_pitch"],
            selected_quality_score=selected_record["quality_score"],
        )
        self._active_final_candidate_metadata = result_record
        return selected

    @staticmethod
    def _quality_judgement(result: IntegratedCycleResult) -> str:
        judgements = [
            str(item.get("inspection_judgement", "RECHECK"))
            for item in result.inspection_planes
        ]
        if "NG" in judgements:
            return "NG"
        if (not judgements or "RECHECK" in judgements
                or len(judgements) != result.planes_total):
            return "RECHECK"
        return "OK" if all(value == "OK" for value in judgements) else "RECHECK"

    def _capture_aruco_board_reference(
        self,
        result: IntegratedCycleResult,
        diagnostics_root: Path,
    ) -> dict[str, Any]:
        """Optionally capture ArUco RGB without making the working path fatal."""
        context: dict[str, Any] = {
            "available": False,
            "aruco_roi_status": "NOT_AVAILABLE",
            "aruco_marker_ids": [],
            "aruco_marker_count": 0,
            "aruco_fallback_reason": None,
            "aruco_frame": None,
            "marker_map": {},
            "board_quad": None,
            "roi_diagnostic_artifacts": {},
        }
        cfg = self.final_capture_inspection_config.hybrid_roi
        if not cfg.enabled:
            context["aruco_fallback_reason"] = "hybrid ROI is disabled by configuration"
            return context
        if self.lighting is None or not result.lighting_connected:
            context["aruco_fallback_reason"] = "controlled Neutral White lighting is unavailable"
            return context
        if not hasattr(self.camera, "color_intrinsics"):
            context["aruco_fallback_reason"] = "camera color intrinsics API is unavailable"
            return context

        frame = None
        self._lighting_on(result)
        try:
            for _ in range(cfg.aruco_rgb_warmup_frames):
                self.camera.capture()
            frame = self.camera.capture()
            color = np.asarray(frame.color_bgr)
            detector = self.aruco_detector_factory()
            marker_map = detect_markers(color, detector)
            marker_ids = sorted(marker_map)
            board_quad = get_board_outer_quad(marker_map)
            context.update(
                aruco_frame=frame,
                marker_map=marker_map,
                board_quad=board_quad,
                aruco_marker_ids=marker_ids,
                aruco_marker_count=len(marker_ids),
            )
            context["roi_diagnostic_artifacts"] = save_hybrid_roi_artifacts(
                diagnostics_root,
                aruco_rgb=color,
                marker_map=marker_map,
                board_quad=board_quad,
            )
            if board_quad is None:
                missing = sorted(set((0, 1, 2, 3)) - set(marker_ids))
                raise HybridROIError(f"required ArUco marker IDs are missing: {missing}")
            context.update(
                available=True,
                aruco_roi_status="FALLBACK",
                aruco_fallback_reason="hybrid Depth processing has not completed",
            )
        except Exception as exc:
            context.update(
                available=False,
                aruco_roi_status="FALLBACK",
                aruco_fallback_reason=f"{type(exc).__name__}: {exc}",
            )
        finally:
            # An OFF failure remains fatal because subsequent geometry must not
            # consume LED-illuminated Depth.
            self._lighting_off(result)

        # Flush aligned pairs after illumination changes. Capture failures are
        # recorded as hybrid fallback; the legacy final geometry retry path is
        # still allowed to recover independently.
        try:
            for _ in range(cfg.depth_flush_frames_after_led_off):
                self.camera.capture()
        except Exception as exc:
            context.update(
                available=False,
                aruco_roi_status="FALLBACK",
                aruco_fallback_reason=(
                    "post-ArUco LED-OFF depth flush failed: "
                    f"{type(exc).__name__}: {exc}"
                ),
            )
        return context

    @staticmethod
    def _sync_result_schema(result: IntegratedCycleResult) -> None:
        result.execution_status = result.overall_status
        result.quality_judgement = IntegratedInspectionCycle._quality_judgement(result)
        result.inspection_status = result.overall_status
        result.final_judgement = result.quality_judgement
        result.planned_planes = result.planes_total
        result.completed_planes = result.planes_completed
        result.failed_planes = result.planes_failed

    def _park_and_run_optional_conveyor_out(self, result: IntegratedCycleResult) -> None:
        """Park level at Z=0, open the cover, then optionally run conveyor OUT."""
        if self.lighting is not None:
            self._lighting_off(result)
        self._show_and_require_black()
        self._stage(result, IntegratedCycleStage.MOVE_SAFE_Z)
        before = self.motion_diagnostic.read_before()
        self._require_orientation_safe_height(before)
        level = self.motion_diagnostic.execute_orientation(
            roll_deg=0.0, pitch_deg=0.0, before=before, ack_safe_height=True,
        )
        parked = self.motion_diagnostic.execute_z(0.0)
        if not (
            math.isclose(float(parked.roll_deg), 0.0, abs_tol=0.1)
            and math.isclose(float(parked.pitch_deg), 0.0, abs_tol=0.1)
            and math.isclose(float(parked.z_cm), 0.0, abs_tol=Z_TARGET_REACHED_TOLERANCE_CM)
        ):
            raise IntegratedCycleError("platform park telemetry mismatch")
        result.final_platform_roll_deg = float(parked.roll_deg)
        result.final_platform_pitch_deg = float(parked.pitch_deg)
        result.final_platform_z_cm = float(parked.z_cm)
        result.platform_parked = True
        self._cover_open(result)
        result.cover_open = True
        if not self.conveyor_out_enabled:
            return
        self._stage(result, IntegratedCycleStage.CONVEYOR_OUT)
        try:
            self.conveyor.move_out()
            self.conveyor.wait_until_stopped()
        except Exception:
            result.conveyor_out = "FAILED"
            raise
        result.conveyor_out_executed = True
        result.conveyor_out = "COMPLETE"
        result.cycle_transport_complete = True
        config = getattr(self.conveyor, "config", None)
        result.conveyor_out_direction = getattr(config, "exit_direction", None)
        result.conveyor_out_steps = getattr(config, "exit_steps", None)

    def _finish_multi_plane_result(self, result: IntegratedCycleResult) -> None:
        result.automatic_z_success = bool(result.inspection_planes) and all(
            bool(item.get("automatic_z_success")) for item in result.inspection_planes
        )
        completed_with_z = [
            item for item in result.inspection_planes if item.get("best_z") is not None
        ]
        if completed_with_z:
            result.best_z = completed_with_z[-1]["best_z"]
            result.automatic_z_stop_reason = completed_with_z[-1].get(
                "automatic_z_stop_reason"
            )
        if result.planes_completed == result.planes_total:
            result.overall_status = "COMPLETE"
            result.success = True
            result.cycle_complete = True
            self._stage(result, IntegratedCycleStage.COMPLETE)
        elif result.planes_completed > 0:
            result.overall_status = "PARTIAL_COMPLETE"
            result.success = False
            self._stage(result, IntegratedCycleStage.PARTIAL_COMPLETE)
        else:
            result.overall_status = "FAILED"
            result.success = False
            self._stage(result, IntegratedCycleStage.FAILED)
        self._sync_result_schema(result)

    def _run_multi_plane_inspection(self, result: IntegratedCycleResult, plan: Any) -> None:
        result.planes_total = len(plan.poses)
        result.search_mode = "adaptive" if getattr(self.automatic_z_search.config, "search_mode", "explicit") == "adaptive" else "explicit"
        self._show_and_require_black()
        if self.lighting is not None:
            self._lighting_on(result)
        elif self.led_checkpoint is not None:
            if self.led_checkpoint() is not True:
                raise ManualLEDConfirmationError("LED_ON was not confirmed; Automatic Z was not started")
            result.manual_led_confirmed = True
        for plane_index, pose in enumerate(plan.poses):
            self._print_pose_start(plane_index + 1, result.planes_total)
            if pose.roll_deg is None or pose.pitch_deg is None:
                raise IntegratedCycleError(f"plane {plane_index} requires finite roll and pitch")
            plane_root = self.paths.root / f"plane_{plane_index:02d}"
            automatic_root = plane_root / "automatic_z"
            plane_result = self._new_plane_result(pose, plane_index, automatic_root)
            plane_result.update(
                points_count=pose.metadata.get("point_count"),
                requested_roll=pose.metadata.get("requested_roll", pose.roll_deg),
                requested_pitch=pose.metadata.get("requested_pitch", pose.pitch_deg),
                applied_roll=pose.metadata.get("applied_roll", pose.roll_deg),
                applied_pitch=pose.metadata.get("applied_pitch", pose.pitch_deg),
                ready_for_anomaly=False,
            )
            result.inspection_planes.append(plane_result)
            self._show_and_require_black()
            self._stage(result, IntegratedCycleStage.MOVE_SAFE_Z)
            after_z = self.motion_diagnostic.execute_z(self.safe_z)
            self._require_orientation_safe_height(after_z)
            self._show_and_require_black()
            self._stage(result, IntegratedCycleStage.MOVE_ORIENTATION)
            self.motion_diagnostic.execute_orientation(
                roll_deg=float(pose.roll_deg), pitch_deg=float(pose.pitch_deg),
                before=after_z, ack_safe_height=True,
            )
            self._show_and_require_black()
            self._stage(result, IntegratedCycleStage.AUTOMATIC_Z)
            if hasattr(self.automatic_z_search, "artifact_store"):
                from src.inspection.hardware_z_search import CandidateArtifactStore
                self.automatic_z_search.artifact_store = CandidateArtifactStore(
                    automatic_root,
                )
            z_result = self.automatic_z_search.run(
                pose_id=pose.pose_id, roll=float(pose.roll_deg), pitch=float(pose.pitch_deg),
            )
            self._save_automatic_z(z_result, automatic_root)
            plane_result.update(
                automatic_z_success=bool(z_result.success), best_z=z_result.best_z,
                automatic_z_stop_reason=z_result.stop_reason,
                ready_for_anomaly=bool(z_result.success),
                **self._automatic_z_diagnostics(z_result),
            )
            if not z_result.success:
                plane_result.update(
                    status="FAILED", failure_stage=IntegratedCycleStage.AUTOMATIC_Z.value,
                    failure_reason=z_result.failure_reason or "NoValidInspectionZ",
                )
                result.planes_failed += 1
            else:
                self._stage(result, IntegratedCycleStage.READY_FOR_ANOMALY)
                plane_result["status"] = "COMPLETE"
                result.planes_completed += 1
            self._recover_after_plane(result, plane_result, plane_root)
            self._print_pose_end(
                plane_index + 1, result.planes_total,
                (IntegratedCycleStage.READY_FOR_ANOMALY.value
                 if z_result.success else "FAILED"), z_result.best_z,
            )
        self._finish_multi_plane_result(result)

    def _run_end_to_end_inspection(self, result: IntegratedCycleResult, plan: Any) -> None:
        """Production loop: Z readiness frames are never reused for anomaly inference."""
        result.planes_total = len(plan.poses)
        result.search_mode = (
            "adaptive" if getattr(self.automatic_z_search.config, "search_mode", "explicit")
            == "adaptive" else "explicit"
        )
        for plane_index, pose in enumerate(plan.poses):
            self._print_pose_start(plane_index + 1, result.planes_total)
            roll, pitch = float(pose.roll_deg), float(pose.pitch_deg)
            plane_root = self.paths.root / f"plane_{plane_index:02d}"
            automatic_root = plane_root / "automatic_z"
            final_root = plane_root / "final_capture"
            anomaly_root = plane_root / "anomaly"
            plane_result = self._new_plane_result(pose, plane_index, automatic_root)
            plane_result.update(
                requested_roll=pose.metadata.get("requested_roll", pose.roll_deg),
                requested_pitch=pose.metadata.get("requested_pitch", pose.pitch_deg),
                applied_roll=pose.metadata.get("applied_roll", pose.roll_deg),
                applied_pitch=pose.metadata.get("applied_pitch", pose.pitch_deg),
            )
            result.inspection_planes.append(plane_result)
            z_result = None
            try:
                self._show_and_require_black()
                self._stage(result, IntegratedCycleStage.MOVE_SAFE_Z)
                adaptive_motion = getattr(self.automatic_z_search.config, "search_mode", "explicit") == "adaptive"
                self._show_and_require_black()
                self._stage(result, IntegratedCycleStage.MOVE_ORIENTATION)
                if adaptive_motion:
                    entry_z = float(self.final_capture_inspection_config.quality.tilt_entry_z_cm)
                    if plane_index == 0:
                        before_pose = self.motion_diagnostic.read_before()
                        if float(before_pose.z_cm) < ORIENTATION_SAFE_Z_MIN_CM:
                            before_pose = self.motion_diagnostic.execute_z(self.safe_z)
                        self._require_orientation_safe_height(before_pose)
                        levelled = self.motion_diagnostic.execute_orientation(
                            roll_deg=0.0, pitch_deg=0.0, before=before_pose,
                            ack_safe_height=True,
                        )
                        raised = self.motion_diagnostic.execute_z(entry_z)
                    else:
                        before_pose = self.motion_diagnostic.read_before()
                        self._require_orientation_safe_height(before_pose)
                        transition_pose = self._adaptive_pose_for_z(
                            pose, float(before_pose.z_cm),
                        )
                        self._apply_adaptive_pose(transition_pose, None)
                        entry_pose = self._adaptive_pose_for_z(pose, entry_z)
                        self._apply_adaptive_pose(
                            entry_pose, transition_pose,
                        )
                        raised = self.motion_diagnostic.read_before()
                else:
                    after_z = self.motion_diagnostic.execute_z(self.safe_z)
                    if not math.isclose(
                        float(after_z.z_cm), self.safe_z, rel_tol=0.0,
                        abs_tol=Z_TARGET_REACHED_TOLERANCE_CM,
                    ):
                        raise IntegratedCycleError(
                            "safe-Z telemetry mismatch before orientation"
                        )
                    self._require_orientation_safe_height(after_z)
                    raised = self.motion_diagnostic.execute_orientation(
                        roll_deg=roll, pitch_deg=pitch, before=after_z, ack_safe_height=True,
                    )
                plane_result.update(
                    actual_platform_roll_deg=float(raised.roll_deg),
                    actual_platform_pitch_deg=float(raised.pitch_deg),
                    actual_platform_z_cm=float(raised.z_cm),
                    actual_rp={
                        "roll_deg": float(raised.roll_deg),
                        "pitch_deg": float(raised.pitch_deg),
                    },
                )
                self._show_and_require_black()
                self._stage(result, IntegratedCycleStage.AUTOMATIC_Z)
                if hasattr(self.automatic_z_search, "artifact_store"):
                    from src.inspection.hardware_z_search import CandidateArtifactStore
                    self.automatic_z_search.artifact_store = CandidateArtifactStore(
                        automatic_root,
                    )
                if adaptive_motion:
                    self.automatic_z_search.pose_for_z = (
                        lambda z_cm, inspection_pose=pose: self._adaptive_pose_for_z(
                            inspection_pose, z_cm,
                        )
                    )
                    self.automatic_z_search.before_z = self._apply_adaptive_pose
                z_result = self.automatic_z_search.run(
                    pose_id=pose.pose_id, roll=roll, pitch=pitch,
                )
                self._save_automatic_z(z_result, automatic_root)
                plane_result.update(
                    automatic_z_success=bool(z_result.success),
                    best_z=z_result.best_z,
                    automatic_z_stop_reason=z_result.stop_reason,
                    **self._automatic_z_diagnostics(z_result),
                )
                if not z_result.success or z_result.best_z is None:
                    raise RecoverablePlaneInspectionError(
                        IntegratedCycleStage.AUTOMATIC_Z.value,
                        z_result.failure_reason or "NoValidInspectionZ",
                    )

                # Explicitly settle at best Z even when the search implementation
                # already returned there. This keeps the final capture independent.
                settled = self.motion_diagnostic.execute_z(float(z_result.best_z))
                self._stage(result, IntegratedCycleStage.BEST_Z_SETTLED)
                plane_result.update(
                    actual_platform_roll_deg=float(settled.roll_deg),
                    actual_platform_pitch_deg=float(settled.pitch_deg),
                    actual_platform_z_cm=float(settled.z_cm),
                    actual_rp={
                        "roll_deg": float(settled.roll_deg),
                        "pitch_deg": float(settled.pitch_deg),
                    },
                )
                self._show_and_require_black()

                # Optional board observation. This is bracketed by LED ON/OFF,
                # followed by aligned-pair flushing, with no platform motion.
                roi_diagnostics_root = final_root / "roi_diagnostics"
                aruco_context = self._capture_aruco_board_reference(
                    result, roi_diagnostics_root,
                )
                plane_result.update({
                    key: value for key, value in aruco_context.items()
                    if key not in {"available", "aruco_frame", "marker_map", "board_quad"}
                })

                if getattr(self.automatic_z_search.config, "search_mode", "explicit") == "adaptive":
                    self._active_final_candidate_metadata = None
                    self._active_final_candidate_bundle = None
                    selected_candidate = self._select_usable_final_candidate(
                        z_result, pose, aruco_context, result, final_root,
                    )
                    selected_pose = self._candidate_pose(selected_candidate, pose)
                    settled = self.motion_diagnostic.read_before()
                    plane_result.update(
                        best_z=float(selected_candidate.z_command),
                        actual_platform_roll_deg=float(settled.roll_deg),
                        actual_platform_pitch_deg=float(settled.pitch_deg),
                        actual_platform_z_cm=float(settled.z_cm),
                        actual_rp={
                            "roll_deg": float(settled.roll_deg),
                            "pitch_deg": float(settled.pitch_deg),
                        },
                        commanded_roll_deg=float(selected_pose.applied_roll_deg),
                        commanded_pitch_deg=float(selected_pose.applied_pitch_deg),
                        commanded_roll=float(selected_pose.applied_roll_deg),
                        commanded_pitch=float(selected_pose.applied_pitch_deg),
                        **(self._active_final_candidate_metadata or {}),
                    )
                    roll = float(selected_pose.applied_roll_deg)
                    pitch = float(selected_pose.applied_pitch_deg)

                # Freeze inspection geometry from LED-OFF aligned Depth before
                # changing illumination. Automatic-Z candidate frames remain
                # independent and are never reused here.
                candidate_bundle = self._active_final_candidate_bundle
                if (candidate_bundle is None and self.lighting is not None
                        and result.lighting_connected):
                    self._lighting_off(result)
                self._stage(result, IntegratedCycleStage.FINAL_GEOMETRY_CAPTURE)
                if candidate_bundle is not None:
                    geometry_capture = candidate_bundle["geometry_capture"]
                else:
                    try:
                        geometry_capture = self.final_capture_acquirer(
                            self.camera,
                            self.final_capture_inspection_config,
                            max_attempts=FINAL_GEOMETRY_MAX_ATTEMPTS,
                        )
                    except FinalCaptureQualityError as exc:
                        plane_result.update(exc.metadata)
                        raise
                geometry_metadata = geometry_capture.metadata()
                plane_result.update(geometry_metadata)
                try:
                    inspection_roi = build_inspection_mask(
                        geometry_capture.geometry.object_mask,
                        geometry_capture.geometry.surface_mask,
                        self.final_capture_inspection_config,
                    )
                except ValueError as exc:
                    raise FinalCaptureQualityError(
                        f"invalid inspection mask: {exc}", metadata=geometry_metadata,
                    ) from exc
                selected_roi = inspection_roi
                anomaly_roi_type = "legacy_inspection_mask"
                depth_contour_roi = None
                depth = np.asarray(geometry_capture.frame.depth_mm)
                height, width = depth.shape
                try:
                    contour_intrinsics = self.camera.color_intrinsics(width, height)
                except Exception as exc:
                    raise RecoverablePlaneInspectionError(
                        IntegratedCycleStage.FINAL_GEOMETRY_CAPTURE.value,
                        f"metric board-plane intrinsics unavailable: {exc}",
                    ) from exc
                roi_depth_frames = [depth]
                roi_config = self.final_capture_inspection_config.surface_roi
                roi_frame_count = max(1, int(roi_config.roi_depth_frames))
                if (candidate_bundle is None and self.lighting is not None
                        and result.lighting_connected):
                    self._lighting_on(result)
                elif candidate_bundle is None and self.led_checkpoint is not None:
                    self._stage(result, IntegratedCycleStage.MANUAL_LED_CHECKPOINT)
                    if self.led_checkpoint() is not True:
                        raise ManualLEDConfirmationError(
                            "LED_ON was not confirmed before final ROI capture"
                        )
                    result.manual_led_confirmed = True
                    self._stage(result, IntegratedCycleStage.LED_ON)
                if candidate_bundle is None:
                    for index in range(self.final_rgb_warmup_frames):
                        self.camera.capture()
                        print(
                            f"[FINAL ROI] warmup={index + 1}/{self.final_rgb_warmup_frames} discard"
                        )
                    roi_depth_frames = [
                        np.asarray(self.camera.capture().depth_mm)
                        for _ in range(roi_frame_count)
                    ]
                depth = roi_depth_frames[0]
                plane_result.update(
                    roi_depth_lighting="NEUTRAL_WHITE",
                    roi_depth_frame_count=roi_frame_count,
                    roi_min_votes=roi_config.roi_min_votes,
                    rgb_fallback_used=False,
                )
                try:
                    if candidate_bundle is not None:
                        depth_contour_roi = candidate_bundle["depth_contour_roi"]
                    elif aruco_context["available"]:
                        try:
                            depth_contour_roi = self.depth_contour_roi_builder(
                                depth, geometry_capture.frame.color_bgr.shape,
                                self.final_capture_inspection_config,
                                board_quad=aruco_context["board_quad"],
                                marker_map=aruco_context["marker_map"],
                                intrinsics=contour_intrinsics,
                                depth_frames=roi_depth_frames,
                                min_votes=roi_config.roi_min_votes,
                                current_platform_roll_deg=float(settled.roll_deg),
                                current_platform_pitch_deg=float(settled.pitch_deg),
                                commanded_platform_roll_deg=roll,
                                commanded_platform_pitch_deg=pitch,
                            )
                        except Exception as aruco_workspace_error:
                            plane_result.update(
                                aruco_roi_status="FALLBACK",
                                aruco_fallback_reason=(
                                    "ArUco workspace contour ROI failed; using fallback "
                                    f"workspace: {type(aruco_workspace_error).__name__}: "
                                    f"{aruco_workspace_error}"
                                ),
                            )
                    if depth_contour_roi is None:
                        depth_contour_roi = self.depth_contour_roi_builder(
                            depth, geometry_capture.frame.color_bgr.shape,
                            self.final_capture_inspection_config,
                            board_quad=None,
                            marker_map=aruco_context["marker_map"],
                            intrinsics=contour_intrinsics,
                            depth_frames=roi_depth_frames,
                            min_votes=roi_config.roi_min_votes,
                            current_platform_roll_deg=float(settled.roll_deg),
                            current_platform_pitch_deg=float(settled.pitch_deg),
                            commanded_platform_roll_deg=roll,
                            commanded_platform_pitch_deg=pitch,
                        )
                    selected_mask = (
                        candidate_bundle["selected_mask"] if candidate_bundle is not None
                        else depth_contour_roi.inspection_mask
                    )
                    selected_area = int(np.count_nonzero(selected_mask))
                    selected_roi = InspectionMaskResult(
                        mask=selected_mask,
                        inspection_area_px=selected_area,
                        inspection_to_surface_ratio=(
                            selected_area
                            / geometry_capture.geometry.surface_area_px
                            if geometry_capture.geometry.surface_area_px > 0 else None
                        ),
                        inspection_to_object_ratio=(
                            selected_area
                            / geometry_capture.geometry.object_area_px
                            if geometry_capture.geometry.object_area_px > 0 else None
                        ),
                    )
                    anomaly_roi_type = "depth_external_contour_fill"
                    if candidate_bundle is not None and candidate_bundle["rgb_fallback_used"]:
                        anomaly_roi_type = "depth_rgb_seeded_fallback"
                    plane_result.update(
                        anomaly_roi_type=anomaly_roi_type,
                        **depth_contour_roi.metadata(),
                    )
                    if depth_contour_roi.workspace_source == "aruco":
                        plane_result.update(
                            aruco_roi_status="SUCCESS",
                            aruco_fallback_reason=None,
                        )
                except Exception as exc:
                    raise RecoverablePlaneInspectionError(
                        IntegratedCycleStage.FINAL_GEOMETRY_CAPTURE.value,
                        f"trusted board-plane ROI unavailable: {type(exc).__name__}: {exc}",
                    ) from exc

                # Retain the newer RGB-assisted hybrid as an explicit debug
                # option; it is not the production-default anomaly ROI.
                if (
                    self.final_capture_inspection_config.hybrid_roi.use_for_anomaly
                    and aruco_context["available"]
                ):
                    try:
                        aruco_frame = aruco_context["aruco_frame"]
                        height, width = depth.shape
                        intrinsics = self.camera.color_intrinsics(width, height)
                        hybrid = self.hybrid_roi_builder(
                            np.asarray(aruco_frame.color_bgr), depth, intrinsics,
                            aruco_context["board_quad"], aruco_context["marker_map"],
                            self.final_capture_inspection_config,
                        )
                        hybrid_paths = save_hybrid_roi_artifacts(
                            roi_diagnostics_root,
                            aruco_rgb=np.asarray(aruco_frame.color_bgr),
                            marker_map=aruco_context["marker_map"],
                            board_quad=aruco_context["board_quad"],
                            result=hybrid,
                        )
                        selected_roi = InspectionMaskResult(
                            mask=hybrid.inspection_mask,
                            inspection_area_px=hybrid.hybrid_inspection_area_px,
                            inspection_to_surface_ratio=(
                                hybrid.hybrid_inspection_area_px
                                / geometry_capture.geometry.surface_area_px
                                if geometry_capture.geometry.surface_area_px > 0 else None
                            ),
                            inspection_to_object_ratio=(
                                hybrid.hybrid_inspection_area_px
                                / geometry_capture.geometry.object_area_px
                                if geometry_capture.geometry.object_area_px > 0 else None
                            ),
                        )
                        plane_result.update(
                            aruco_roi_status="SUCCESS",
                            aruco_fallback_reason=None,
                            anomaly_roi_type="aruco_depth_rgb_hybrid",
                            roi_diagnostic_artifacts=hybrid_paths,
                            **hybrid.metadata(),
                        )
                        anomaly_roi_type = "aruco_depth_rgb_hybrid"
                    except Exception as exc:
                        # Optional hybrid processing must never make a plane that
                        # passed the contour-fill production path become RECHECK.
                        plane_result.update(
                            aruco_roi_status="FALLBACK",
                            aruco_fallback_reason=f"{type(exc).__name__}: {exc}",
                            anomaly_roi_type=anomaly_roi_type,
                        )
                else:
                    plane_result["anomaly_roi_type"] = anomaly_roi_type
                print(
                    "[INSPECTION ROI] "
                    f"type={anomaly_roi_type} "
                    f"aruco={plane_result['aruco_roi_status']} "
                    f"surface_px={geometry_capture.geometry.surface_area_px} "
                    f"inspection_px={selected_roi.inspection_area_px}"
                )
                geometry_artifacts = self.final_geometry_saver(
                    geometry_capture, final_root, inspection_mask=selected_roi,
                )
                if depth_contour_roi is not None:
                    plane_result["depth_contour_roi_artifacts"] = (
                        self.depth_contour_roi_saver(
                            final_root, depth_contour_roi,
                            color_bgr=geometry_capture.frame.color_bgr,
                        )
                    )
                    patch_config = self.final_capture_inspection_config.patch
                    _, _, patchable_ratio = measure_patchability(
                        selected_roi.mask, patch_config.patch_size, patch_config.patch_stride,
                        patch_config.patch_mask_coverage,
                    )
                    plane_result["patchable_ratio"] = patchable_ratio
                    plane_result["patch_union_area_px"] = int(
                        patchable_ratio * selected_roi.inspection_area_px
                    )
                    filled_object_mask_for_anomaly = (
                        candidate_bundle["filled_object_mask"] if candidate_bundle is not None
                        else depth_contour_roi.depth_object_contour_filled
                    )
                    if patchable_ratio < roi_config.min_patchable_ratio:
                        board_reference = (
                            depth_contour_roi.plane_hypothesis_masks[
                                depth_contour_roi.selected_board_plane_hypothesis_index
                            ]
                            if depth_contour_roi.selected_board_plane_hypothesis_index is not None
                            and depth_contour_roi.selected_board_plane_hypothesis_index < len(
                                depth_contour_roi.plane_hypothesis_masks
                            ) else depth_contour_roi.board_plane_fit_mask
                        )
                        fallback_frame = self.camera.capture()
                        rgb_fallback = build_rgb_seeded_roi(
                            np.asarray(fallback_frame.color_bgr),
                            depth_contour_roi.workspace_mask,
                            board_reference,
                            depth_contour_roi.depth_main_component_mask,
                            roi_config,
                        ) if board_reference is not None else None
                        if rgb_fallback is None:
                            raise RecoverablePlaneInspectionError(
                                IntegratedCycleStage.FINAL_GEOMETRY_CAPTURE.value,
                                "Depth ROI patchable ratio is low and RGB seeded fallback failed",
                            )
                        fallback_inspection = erode_surface_mask(
                            rgb_fallback.object_mask, self.final_capture_inspection_config.surface_roi.boundary_margin_px,
                        )
                        fallback_inspection = np.where(
                            (fallback_inspection > 0)
                            & (rgb_fallback.object_mask > 0)
                            & (depth_contour_roi.workspace_mask > 0), 255, 0,
                        ).astype(np.uint8)
                        fallback_area = int(np.count_nonzero(fallback_inspection))
                        _, _, fallback_ratio = measure_patchability(
                            fallback_inspection, patch_config.patch_size,
                            patch_config.patch_stride, patch_config.patch_mask_coverage,
                        )
                        if (
                            fallback_area == 0
                            or fallback_ratio < roi_config.min_patchable_ratio
                            or mask_touches_frame_edge(
                                fallback_inspection,
                                self.final_capture_inspection_config.surface_roi.fov_edge_margin_px,
                            )
                            or np.count_nonzero(rgb_fallback.object_mask) / max(
                                1, np.count_nonzero(depth_contour_roi.workspace_mask)
                            ) > self.final_capture_inspection_config.surface_roi.max_object_area_ratio
                        ):
                            raise RecoverablePlaneInspectionError(
                                IntegratedCycleStage.FINAL_GEOMETRY_CAPTURE.value,
                                "RGB seeded fallback did not produce a safe patchable ROI",
                            )
                        selected_roi = InspectionMaskResult(
                            mask=fallback_inspection,
                            inspection_area_px=fallback_area,
                            inspection_to_surface_ratio=(
                                fallback_area / geometry_capture.geometry.surface_area_px
                                if geometry_capture.geometry.surface_area_px > 0 else None
                            ),
                            inspection_to_object_ratio=(
                                fallback_area / geometry_capture.geometry.object_area_px
                                if geometry_capture.geometry.object_area_px > 0 else None
                            ),
                        )
                        filled_object_mask_for_anomaly = rgb_fallback.object_mask
                        _, _, patchable_ratio = measure_patchability(
                            selected_roi.mask, patch_config.patch_size,
                            patch_config.patch_stride, patch_config.patch_mask_coverage,
                        )
                        plane_result.update(
                            anomaly_roi_type="depth_rgb_seeded_fallback",
                            rgb_fallback_used=True,
                            rgb_fallback_seed_overlap_px=rgb_fallback.seed_overlap_px,
                            rgb_fallback_component_area_px=rgb_fallback.selected_component_area_px,
                            patchable_ratio=patchable_ratio,
                        )
                        import cv2
                        final_root.mkdir(parents=True, exist_ok=True)
                        cv2.imwrite(
                            str(final_root / "rgb_seeded_foreground.png"),
                            rgb_fallback.foreground_mask,
                        )
                        cv2.imwrite(
                            str(final_root / "rgb_seeded_component.png"),
                            rgb_fallback.selected_component_mask,
                        )
                plane_result.update(
                    final_depth_path=geometry_artifacts.depth_path,
                    object_mask_path=geometry_artifacts.object_mask_path,
                    surface_mask_path=geometry_artifacts.surface_mask_path,
                    surface_geometry_overlay_path=(
                        geometry_artifacts.surface_geometry_overlay_path
                    ),
                    inspection_mask_path=geometry_artifacts.inspection_mask_path,
                    inspection_mask_overlay_path=(
                        geometry_artifacts.inspection_mask_overlay_path
                    ),
                    inspection_area_px=selected_roi.inspection_area_px,
                    inspection_to_surface_ratio=(
                        selected_roi.inspection_to_surface_ratio
                    ),
                    inspection_to_object_ratio=(
                        selected_roi.inspection_to_object_ratio
                    ),
                    anomaly_roi_type=anomaly_roi_type,
                )

                self._stage(result, IntegratedCycleStage.FINAL_RGB_CAPTURE)
                surface_mask = geometry_capture.geometry.surface_mask
                if surface_mask is None:
                    raise FinalCaptureQualityError("accepted geometry has no surface mask")
                fresh_frame = self.final_rgb_acquirer(
                    self.camera,
                    warmup_frames=self.final_rgb_warmup_frames,
                    expected_shape=tuple(surface_mask.shape),
                )
                rgb_artifacts = self.final_rgb_saver(fresh_frame, final_root)
                plane_result.update(
                    final_rgb_warmup_frames=self.final_rgb_warmup_frames,
                    final_rgb_path=rgb_artifacts.rgb_path,
                    final_ir_path=rgb_artifacts.ir_path,
                )
                fresh_telemetry = self.motion_diagnostic.read_before()
                plane_result.update(
                    actual_platform_roll_deg=float(fresh_telemetry.roll_deg),
                    actual_platform_pitch_deg=float(fresh_telemetry.pitch_deg),
                    actual_platform_z_cm=float(fresh_telemetry.z_cm),
                    actual_rp={
                        "roll_deg": float(fresh_telemetry.roll_deg),
                        "pitch_deg": float(fresh_telemetry.pitch_deg),
                    },
                )
                self._stage(result, IntegratedCycleStage.ANOMALY_INFERENCE)
                anomaly = self.anomaly_detector.inspect_frame(
                    fresh_frame, pose_id=pose.pose_id, output_directory=anomaly_root,
                    rgb_path=rgb_artifacts.rgb_path,
                    depth_path=geometry_artifacts.depth_path,
                    ir_path=rgb_artifacts.ir_path,
                    platform_telemetry=fresh_telemetry,
                    surface_geometry=geometry_capture.geometry,
                    geometry_capture_metadata=geometry_metadata,
                    inspection_mask=selected_roi.mask,
                    filled_object_mask=filled_object_mask_for_anomaly,
                    final_capture_metadata={
                        "anomaly_roi_type": anomaly_roi_type,
                        "inspection_area_px": selected_roi.inspection_area_px,
                        "inspection_to_surface_ratio": (
                            selected_roi.inspection_to_surface_ratio
                        ),
                        "inspection_to_object_ratio": (
                            selected_roi.inspection_to_object_ratio
                        ),
                        "workspace_source": plane_result.get("workspace_source"),
                        "depth_candidate_area_px": plane_result.get(
                            "depth_candidate_area_px"
                        ),
                        "depth_main_component_area_px": plane_result.get(
                            "depth_main_component_area_px"
                        ),
                        "filled_object_area_px": plane_result.get(
                            "filled_object_area_px"
                        ),
                        "fill_gain_px": plane_result.get("fill_gain_px"),
                        "fill_gain_ratio": plane_result.get("fill_gain_ratio"),
                        "patchable_ratio": plane_result.get("patchable_ratio"),
                    },
                )
                if anomaly.classification not in ("NORMAL", "DEFECT"):
                    raise AnomalyInputDataError(
                        f"unsupported anomaly classification: {anomaly.classification!r}"
                    )
                judgement = "OK" if anomaly.classification == "NORMAL" else "NG"
                print(f"[ANOMALY] classification={anomaly.classification}")
                print(f"[ANOMALY] judgement={judgement}")
                anomaly_payload = asdict(anomaly)
                anomaly_payload.update({
                    "pose_id": pose.pose_id,
                    "final_rgb_path": rgb_artifacts.rgb_path,
                    "final_depth_path": geometry_artifacts.depth_path,
                    "final_ir_path": rgb_artifacts.ir_path,
                    "surface_mask_path": geometry_artifacts.surface_mask_path,
                    "inspection_mask_path": geometry_artifacts.inspection_mask_path,
                    "object_mask_path": geometry_artifacts.object_mask_path,
                    "anomaly_roi_type": anomaly_roi_type,
                    "aruco_roi_status": plane_result["aruco_roi_status"],
                    "inspection_judgement": judgement,
                    "quality_judgement": judgement,
                    "actual_platform_roll_deg": float(fresh_telemetry.roll_deg),
                    "actual_platform_pitch_deg": float(fresh_telemetry.pitch_deg),
                    "actual_platform_z_cm": float(fresh_telemetry.z_cm),
                })
                anomaly_root.mkdir(parents=True, exist_ok=True)
                anomaly_result_path = anomaly_root / "result.json"
                anomaly_result_path.write_text(
                    json.dumps(anomaly_payload, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
                plane_result.update(
                    status="COMPLETE", anomaly_success=True, anomaly_executed=True,
                    inspection_judgement=judgement, quality_judgement=judgement,
                    anomaly_result=anomaly_payload,
                    anomaly_result_path=str(anomaly_result_path),
                    anomaly_score=anomaly.score, anomaly_threshold=anomaly.threshold,
                    anomaly_classification=anomaly.classification,
                    classification=anomaly.classification,
                    anomaly_heatmap_path=anomaly.heatmap_path,
                    surface_patch_overlay_path=anomaly.metadata.get(
                        "surface_patch_overlay_path"
                    ),
                    selected_inspection_patch_count=anomaly.metadata.get(
                        "selected_inspection_patch_count"
                    ),
                    selected_surface_patch_count=anomaly.metadata.get(
                        "selected_surface_patch_count"
                    ),
                )
                result.anomaly_executed = True
                result.planes_completed += 1
            except RecoverablePlaneInspectionError as exc:
                plane_result.update(
                    status="FAILED", failure_stage=exc.stage, failure_reason=exc.reason,
                    inspection_judgement="RECHECK", quality_judgement="RECHECK",
                )
                if self._active_final_candidate_metadata is not None:
                    plane_result.update(self._active_final_candidate_metadata)
                result.planes_failed += 1
            is_last_pose = plane_index + 1 == result.planes_total
            if plane_result["status"] == "COMPLETE" and not is_last_pose:
                if self.lighting is not None:
                    self._lighting_off(result)
                self._show_and_require_black()
                self._save_plane_result(plane_root, plane_result)
            elif not is_last_pose:
                self._recover_after_plane(result, plane_result, plane_root)
            elif plane_result["status"] != "COMPLETE":
                self._recover_after_plane(result, plane_result, plane_root)
            else:
                self._save_plane_result(plane_root, plane_result)
            self._print_pose_end(
                plane_index + 1, result.planes_total, plane_result["status"],
                plane_result["best_z"],
            )
        self._park_and_run_optional_conveyor_out(result)
        self._finish_multi_plane_result(result)

    def _record_error(self, result: IntegratedCycleResult, exc: BaseException) -> None:
        result.success = False
        result.overall_status = "FAILED"
        if result.inspection_planes:
            plane_result = result.inspection_planes[-1]
            if plane_result.get("status") == "PENDING":
                plane_result.update(
                    status="FAILED", failure_stage=result.stage,
                    failure_reason=f"{type(exc).__name__}: {exc}",
                    inspection_judgement="RECHECK", quality_judgement="RECHECK",
                )
                result.planes_failed += 1
                try:
                    plane_index = int(plane_result["inspection_order"]) - 1
                    self._save_plane_result(
                        self.paths.root / f"plane_{plane_index:02d}", plane_result,
                    )
                except Exception:
                    # The primary fatal error must not be masked by best-effort
                    # failure-artifact persistence.
                    pass
        elif (self._active_pose_progress is not None and result.planes_total
              and result.planes_completed + result.planes_failed < result.planes_total):
            # Backward-compatible single-pose path does not create a plane
            # result until Automatic Z has returned.
            result.planes_failed += 1
        if self._active_pose_progress is not None and not self._active_pose_reported:
            index, total = self._active_pose_progress
            self._print_pose_end(index, total, "FAILED", None)
        result.error_type = type(exc).__name__
        result.error_message = str(exc)
        result.interrupted = isinstance(exc, KeyboardInterrupt)
        self._sync_result_schema(result)

    def _record_cleanup_error(self, result: IntegratedCycleResult, label: str, exc: BaseException) -> None:
        message = f"{label}: {type(exc).__name__}: {exc}"
        result.cleanup_errors.append(message)
        if result.error_type is None:
            result.success = False
            result.overall_status = "FAILED"
            result.stage = IntegratedCycleStage.CLEANUP.value
            result.stage_history.append(IntegratedCycleStage.CLEANUP.value)
            result.error_type = type(exc).__name__
            result.error_message = message

    def _initialize_platform(self, result: IntegratedCycleResult) -> None:
        """Return to the user-calibrated structured-light reference pose."""
        if self.lighting is None:
            self.platform.connect()
        initial = self.motion_diagnostic.read_before()
        result.initial_z_before = float(initial.z_cm)
        result.initial_roll_before = float(initial.roll_deg)
        result.initial_pitch_before = float(initial.pitch_deg)
        if initial.homing:
            raise PlatformInitializationError(
                "platform homing is active; initialization stopped before motion"
            )
        wait_config = getattr(self.motion_diagnostic, "wait_config", None)
        z_tolerance = float(getattr(
            wait_config, "z_target_tolerance_cm", Z_TARGET_REACHED_TOLERANCE_CM,
        ))
        orientation_tolerance = float(getattr(
            wait_config, "orientation_target_tolerance_deg",
            ORIENTATION_TARGET_REACHED_TOLERANCE_DEG,
        ))
        already_at_scan_pose = (
            bool(initial.stable)
            and math.isclose(
                float(initial.z_cm), self.scan_z, rel_tol=0.0,
                abs_tol=z_tolerance,
            )
            and math.isclose(
                float(initial.roll_deg), 0.0, rel_tol=0.0,
                abs_tol=orientation_tolerance,
            )
            and math.isclose(
                float(initial.pitch_deg), 0.0, rel_tol=0.0,
                abs_tol=orientation_tolerance,
            )
        )
        if already_at_scan_pose:
            print(
                "[STARTUP] already at scan pose; "
                "skipping Z15/RP0/Z0 initialization"
            )
            result.startup_fast_path_used = True
            result.orientation_zeroed = True
            result.scan_z_reached = True
            result.initialization_success = True
            return

        if (
            not initial.stable
            or float(initial.z_cm) < ORIENTATION_SAFE_Z_MIN_CM
            or not math.isclose(
                float(initial.z_cm), self.safe_z, rel_tol=0.0,
                abs_tol=Z_TARGET_REACHED_TOLERANCE_CM,
            )
        ):
            self._show_and_require_black()
            initial = self.motion_diagnostic.execute_z(self.safe_z)

        self._require_orientation_safe_height(initial)

        # Orientation-only absolute command: the controller packet intentionally
        # omits Z, so the current firmware Z target is not changed first.
        self._show_and_require_black()
        zeroed = self.motion_diagnostic.execute_orientation(
            roll_deg=0.0, pitch_deg=0.0, before=initial,
            ack_safe_height=True,
        )
        if zeroed.homing:
            raise PlatformInitializationError(
                "platform entered homing during orientation initialization"
            )
        result.orientation_zeroed = True

        self._show_and_require_black()
        scan_pose = self.motion_diagnostic.execute_z(self.scan_z)
        if scan_pose.homing:
            raise PlatformInitializationError(
                "platform entered homing during scan-Z initialization"
            )
        result.scan_z_reached = True
        result.initialization_success = True

    def run(self) -> IntegratedCycleResult:
        result = IntegratedCycleResult(
            str(self.paths.root), self.safe_z, self.scan_z,
            final_rgb_warmup_frames=self.final_rgb_warmup_frames,
        )
        projector_open_attempted = False
        camera_started = False
        run_info: StructuredLightRunInfo | None = None
        try:
            self._stage(result, IntegratedCycleStage.INITIALIZING)
            self._stage(result, IntegratedCycleStage.CONNECT)
            if self.lighting is not None:
                self.platform.connect()
            if self.lighting is not None:
                try:
                    self.lighting.connect()
                    result.lighting_connected = True
                    self._lighting_off(result)
                    self._cover_open(result)
                except Exception as exc:
                    result.lighting_error = f"{type(exc).__name__}: {exc}"
                    raise
            projector_open_attempted = True
            self.projector.open()
            self._show_and_require_black()
            self._stage(result, IntegratedCycleStage.PROJECTOR_BLACK)

            self._stage(result, IntegratedCycleStage.PLATFORM_INITIALIZE)
            self._initialize_platform(result)
            self.conveyor.connect()
            self._stage(result, IntegratedCycleStage.CONVEYOR_TO_INSPECTION)
            self.conveyor.move_to_inspection()
            self.conveyor.wait_until_stopped()
            result.conveyor_complete = True
            self._stage(result, IntegratedCycleStage.CONVEYOR_POSITION)

            if self.lighting is not None:
                self.lighting.inspection_off()
            self._cover_open(result)
            self._stage(result, IntegratedCycleStage.STRUCTURED_LIGHT_SCAN)
            run_info = self.structured_light_runner.run_scan()
            result.structured_light_success = True
            result.structured_light_run_directory = str(Path(run_info.result_directory).resolve())
            self._show_and_require_black()
            self._stage(result, IntegratedCycleStage.STRUCTURED_LIGHT)
            self._cover_close(result)

            self._stage(result, IntegratedCycleStage.PLAN_DOMINANT_POSE)
            self._stage(result, IntegratedCycleStage.METRIC_POSE)
            print("[STAGE] PLAN_INSPECTION_POSES")
            pose_json = resolve_current_pose_json(run_info)
            result.pose_json = str(pose_json)
            self.metric_pose_postprocessor(
                pose_json,
                fresh_telemetry_reader=self.motion_diagnostic.read_before,
            )
            plan = self.pose_planner.plan(pose_json)
            result.pose_planning_reached = True
            if not _inside(Path(plan.source_ply), Path(run_info.result_directory)):
                raise IntegratedCycleError("pose JSON input_ply is outside the current structured-light run")
            self._archive_structured_light(run_info, pose_json)
            result.detected_planes = int(
                plan.metadata.get("detected_plane_count", plan.metadata.get("parsed_plane_count", 0))
            )
            result.planes_total = len(plan.poses)
            result.planned_planes = result.planes_total
            self._print_pose_plan_summary(plan)
            if not plan.metadata.get("platform_motion_allowed") or not plan.poses:
                raise IntegratedCycleError(
                    "no reachable metric inspection pose; legacy phase-space fallback is forbidden"
                )
            plan.validate()
            if self.anomaly_detector is not None:
                self.camera.start()
                camera_started = True
                self._run_end_to_end_inspection(result, plan)
                return result
            if plan.metadata.get("selection_policy") == "all_valid_planes":
                self.camera.start()
                camera_started = True
                self._run_multi_plane_inspection(result, plan)
                return result
            if len(plan.poses) != 1:
                raise IntegratedCycleError("partial cycle requires exactly one dominant pose")
            pose = plan.poses[0]
            if pose.roll_deg is None or pose.pitch_deg is None:
                raise IntegratedCycleError("dominant pose requires roll and pitch")
            roll, pitch = float(pose.roll_deg), float(pose.pitch_deg)
            if not math.isfinite(roll) or not math.isfinite(pitch):
                raise IntegratedCycleError("dominant roll/pitch must be finite")
            result.selected_roll, result.selected_pitch = roll, pitch
            self._print_pose_start(1, result.planes_total)
            self._show_and_require_black()
            self._stage(result, IntegratedCycleStage.MOVE_SAFE_Z)
            after_z = self.motion_diagnostic.execute_z(self.safe_z)
            self._require_orientation_safe_height(after_z)

            self._show_and_require_black()
            self._stage(result, IntegratedCycleStage.MOVE_ORIENTATION)
            self.motion_diagnostic.execute_orientation(
                roll_deg=roll, pitch_deg=pitch, before=after_z,
                ack_safe_height=True,
            )

            self._show_and_require_black()
            if self.lighting is not None and result.lighting_connected:
                self._lighting_on(result)
            elif self.led_checkpoint is not None:
                self._stage(result, IntegratedCycleStage.MANUAL_LED_CHECKPOINT)
                if self.led_checkpoint() is not True:
                    raise ManualLEDConfirmationError("LED_ON was not confirmed; Automatic Z was not started")
                result.manual_led_confirmed = True

            self._show_and_require_black()
            self._stage(result, IntegratedCycleStage.AUTOMATIC_Z)
            self.camera.start()
            camera_started = True
            z_result = self.automatic_z_search.run(
                pose_id=pose.pose_id, roll=roll, pitch=pitch,
            )
            self._save_automatic_z(z_result)
            result.automatic_z_success = bool(z_result.success)
            result.best_z = z_result.best_z
            result.automatic_z_stop_reason = z_result.stop_reason
            result.search_mode = getattr(z_result, "search_mode", None)
            if not z_result.success:
                result.planes_failed = 1
                self._print_pose_end(1, result.planes_total, "FAILED", z_result.best_z)
                raise IntegratedCycleError(z_result.failure_reason or "NoValidInspectionZ")

            self._show_and_require_black()
            result.planes_completed = 1
            self._print_pose_end(
                1, result.planes_total, IntegratedCycleStage.READY_FOR_ANOMALY.value,
                z_result.best_z,
            )
            self._stage(result, IntegratedCycleStage.COMPLETE)
            result.success = True
            result.overall_status = "COMPLETE"
        except KeyboardInterrupt as exc:
            self._record_error(result, exc)
        except Exception as exc:
            self._record_error(result, exc)
        finally:
            if run_info is not None and not (self.paths.structured_light / "run_info.json").exists():
                try:
                    self._archive_structured_light(run_info)
                except Exception as exc:
                    self._record_cleanup_error(result, "structured-light log", exc)
            if projector_open_attempted:
                try:
                    self._show_and_require_black()
                    result.projector_final_state = ProjectorState.BLACK.value
                except Exception as exc:
                    result.projector_final_state = "BLACK_RESTORE_FAILED"
                    self._record_cleanup_error(result, "projector BLACK restore", exc)
            if self.lighting is not None:
                try:
                    self.lighting.inspection_off()
                    result.inspection_led_off_at_end = True
                except Exception as exc:
                    result.lighting_error = f"{type(exc).__name__}: {exc}"
                    self._record_cleanup_error(result, "inspection LED OFF", exc)
                if hasattr(self.lighting, "projector_cover_cleanup"):
                    result.projector_cover_cleanup_attempted = True
                    try:
                        self.lighting.projector_cover_cleanup()
                    except Exception as exc:
                        self._record_cleanup_error(result, "projector cover cleanup", exc)
            try:
                if getattr(self.motion_diagnostic, "log", None) is not None:
                    self.motion_diagnostic.log.save(self.paths.telemetry / "platform.json")
                    self.motion_diagnostic.log.save(self.paths.telemetry / "platform.csv")
            except Exception as exc:
                self._record_cleanup_error(result, "telemetry log", exc)
            for label, resource, method in (
                ("camera close", self.camera, "close"),
                ("platform close", self.platform, "close"),
                ("conveyor close", self.conveyor, "close"),
                ("lighting close", self.lighting, "close"),
                ("projector close", self.projector, "close"),
            ):
                if resource is None:
                    continue
                if label == "camera close" and not camera_started:
                    continue
                try:
                    getattr(resource, method)()
                except Exception as exc:
                    self._record_cleanup_error(result, label, exc)
            result.projector_state_after_close = self._projector_state(self.projector)
            result.finished_at = datetime.now(timezone.utc).isoformat()
            self._sync_result_schema(result)
            (self.paths.logs / "stages.log").write_text(
                "\n".join(result.stage_history) + "\n", encoding="utf-8",
            )
            result.save(self.paths.root / "cycle_result.json")
            self._print_final_pose_summary(result)
        return result
