"""Dry-run-first production-equivalent multiview normal capture tool."""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass
from datetime import datetime
import json
import math
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import cv2
import numpy as np

from src.camera.orbbec_controller import OrbbecCameraController
from src.config import InspectionConfig
from src.core.inspection_mask import InspectionMaskResult
from src.infer_anomaly import select_patch_positions
from src.inspection.adaptive_pose import (
    AdaptivePose, adaptive_pose_for_z, apply_adaptive_pose_transition,
)
from src.inspection.hardware_z_search import (
    CandidateArtifactStore, HardwareAutomaticZSearch, HardwareZSearchConfig,
    SensorQualityConfig, SurfaceReadinessEvaluator,
)
from src.integration.final_capture import (
    acquire_warmed_final_rgb_frame, save_final_geometry_capture,
    save_final_rgb_capture,
)
from src.integration.integrated_inspection_cycle import (
    IntegratedCycleResult, IntegratedInspectionCycle,
)
from src.integration.projector_controller import ProjectorState
from src.lighting.serial_controller import SerialLightingConfig, SerialLightingController
from src.platform.motion_diagnostic import DiagnosticZMover, PlatformMotionDiagnostic
from src.platform.serial_controller import SerialPlatformConfig, SerialPlatformController
from src.tools.production_motion_options import (
    add_production_motion_arguments, build_production_motion_wait_config,
)


ROOT = Path(__file__).resolve().parents[2]
FIELDS = ("path", "split", "label", "session", "mask", "source_run",
          "plane", "material", "view", "notes")


@dataclass(frozen=True)
class CapturePose:
    name: str
    roll_deg: float
    pitch_deg: float
    split: str


STANDARD_10 = (
    CapturePose("01", 0, 0, "train"), CapturePose("02", 10, -1, "train"),
    CapturePose("03", 20, -3, "train"), CapturePose("04", 25, -4, "train"),
    CapturePose("05", -10, 1, "train"), CapturePose("06", -20, 1, "train"),
    CapturePose("07", -25, 2, "train"), CapturePose("08", 0, 3, "train"),
    CapturePose("09", 15, -2, "val"), CapturePose("10", -15, 2, "val"),
)


class BlackProjector:
    state = ProjectorState.BLACK

    def show_black(self) -> None:
        self.state = ProjectorState.BLACK


def parse_custom_poses(raw: str) -> tuple[CapturePose, ...]:
    poses = []
    for index, item in enumerate(raw.split(";"), 1):
        parts = [part.strip() for part in item.split(",")]
        if len(parts) not in (2, 3):
            raise argparse.ArgumentTypeError("custom poses require R,P[,train|val]")
        roll, pitch = float(parts[0]), float(parts[1])
        split = parts[2] if len(parts) == 3 else "train"
        if not math.isfinite(roll) or not math.isfinite(pitch) or split not in ("train", "val"):
            raise argparse.ArgumentTypeError("custom pose values are invalid")
        poses.append(CapturePose(f"{index:02d}", roll, pitch, split))
    if not poses:
        raise argparse.ArgumentTypeError("at least one custom pose is required")
    return tuple(poses)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session", required=True)
    parser.add_argument("--material", required=True, choices=("gray", "blue"))
    parser.add_argument("--preset", choices=("standard_10",), default="standard_10")
    parser.add_argument("--poses", type=parse_custom_poses)
    parser.add_argument("--platform-port", required=True)
    parser.add_argument("--lighting-port", required=True)
    parser.add_argument("--cover-open-angle", type=int, choices=(0, 90), required=True)
    parser.add_argument("--cover-close-angle", type=int, choices=(0, 90), required=True)
    parser.add_argument("--z-start", type=float, default=25.0)
    parser.add_argument("--z-search-min", type=float, default=17.0)
    parser.add_argument("--z-max", type=float, default=25.0)
    parser.add_argument("--z-coarse-step", type=float, default=1.0)
    parser.add_argument("--z-fine-step", type=float, default=1.0)
    parser.add_argument("--quality-config", required=True, type=Path)
    parser.add_argument("--final-rgb-warmup-frames", type=int, default=3)
    parser.add_argument("--output-root", type=Path, default=ROOT / "results/multiview_normal")
    parser.add_argument("--manifest-output-dir", type=Path)
    parser.add_argument("--execute", action="store_true")
    add_production_motion_arguments(parser)
    return parser


def validate_args(args: argparse.Namespace) -> tuple[CapturePose, ...]:
    poses = args.poses or STANDARD_10
    if (args.cover_open_angle, args.cover_close_angle) != (90, 0):
        raise ValueError("production cover mapping must be OPEN=90 CLOSE=0")
    if args.z_start > args.z_max or args.z_search_min > args.z_start:
        raise ValueError("multiview adaptive Z bounds are invalid")
    if args.z_coarse_step <= 0 or args.z_fine_step <= 0:
        raise ValueError("multiview adaptive Z steps must be positive")
    if args.final_rgb_warmup_frames < 0:
        raise ValueError("final RGB warmup must be non-negative")
    for pose in poses:
        if abs(pose.roll_deg) > 25 or abs(pose.pitch_deg) > 25:
            raise ValueError(f"pose {pose.name} exceeds operational R/P limits")
    return poses


def build_multiview_z_config(args: argparse.Namespace) -> HardwareZSearchConfig:
    config = HardwareZSearchConfig(
        candidates=(), z_max=args.z_max,
        stable_timeout_s=args.platform_motion_timeout,
        selection_policy="best_surface_coverage", search_mode="adaptive",
        z_start=args.z_start, coarse_step=args.z_coarse_step,
        fine_step=args.z_fine_step, surface_area_weight=.6,
        depth_valid_weight=.4, stop_after_first_post_pass_failure=False,
        search_min_z_cm=args.z_search_min,
    )
    config.validate()
    return config


def build_multiview_motion(platform: Any, args: argparse.Namespace) -> PlatformMotionDiagnostic:
    return PlatformMotionDiagnostic(
        platform, timeout_s=args.platform_motion_timeout,
        wait_config=build_production_motion_wait_config(args), confirm=lambda _: True,
    )


def _write_manifests(rows: list[dict[str, str]], output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    for split in ("train", "val"):
        with (output / f"{split}.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=FIELDS)
            writer.writeheader(); writer.writerows(row for row in rows if row["split"] == split)


def _pose_for(config: InspectionConfig, pose: CapturePose, z: float) -> AdaptivePose:
    return adaptive_pose_for_z(
        z, pose.roll_deg, pose.pitch_deg,
        roll_limit_deg=config.quality.inspection_roll_limit_deg,
        pitch_limit_deg=config.quality.inspection_pitch_limit_deg,
        envelope=config.quality.tilt_envelope,
    )


def _save_overlay(path: Path, image: np.ndarray, mask: np.ndarray,
                  positions: list[tuple[int, int]], patch_size: int) -> None:
    overlay = image.copy()
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlay, contours, -1, (0, 255, 0), 2)
    for x, y in positions:
        cv2.rectangle(overlay, (x, y), (x + patch_size - 1, y + patch_size - 1),
                      (255, 255, 0), 1)
    if not cv2.imwrite(str(path), overlay):
        raise RuntimeError(f"failed to save overlay: {path}")


def _park_platform_and_open_cover(
    motion: PlatformMotionDiagnostic, lighting: SerialLightingController,
) -> None:
    """Park only after fresh telemetry confirms the production-safe motion path."""
    before = motion.read_before()
    motion.execute_orientation(
        roll_deg=0, pitch_deg=0, before=before, ack_safe_height=True,
    )
    parked = motion.execute_z(0)
    if not all(math.isclose(value, 0.0, abs_tol=.1) for value in
               (parked.roll_deg, parked.pitch_deg, parked.z_cm)):
        raise RuntimeError("final R0/P0/Z0 park telemetry mismatch")
    lighting.projector_cover_open()


def execute_capture(args: argparse.Namespace, poses: tuple[CapturePose, ...]) -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_root = args.output_root.expanduser().resolve() / args.session / stamp
    run_root.mkdir(parents=True)
    config = InspectionConfig.default()
    platform = SerialPlatformController(SerialPlatformConfig(args.platform_port))
    lighting = SerialLightingController(SerialLightingConfig(
        args.lighting_port, projector_cover_open_angle_deg=90,
        projector_cover_close_angle_deg=0, projector_cover_cleanup_state="OPEN",
    ))
    camera = OrbbecCameraController(config)
    projector = BlackProjector()
    motion = build_multiview_motion(platform, args)
    quality = SensorQualityConfig.from_json(args.quality_config)
    quality.require_execution_ready("best_surface_coverage")
    search = HardwareAutomaticZSearch(
        platform=DiagnosticZMover(motion), camera=camera, projector=projector,
        evaluator=SurfaceReadinessEvaluator(quality, config),
        config=build_multiview_z_config(args),
    )
    validator = IntegratedInspectionCycle(
        conveyor=None, structured_light_runner=None, pose_planner=None,
        projector=projector, platform=platform, motion_diagnostic=motion,
        camera=camera, automatic_z_search=search, scan_z=0, safe_z=15,
        run_directory=run_root, lighting=lighting,
        final_capture_inspection_config=config,
        final_rgb_warmup_frames=args.final_rgb_warmup_frames,
    )
    result = IntegratedCycleResult(str(run_root), 15, 0, lighting_connected=True)
    records: list[dict[str, Any]] = []
    rows: list[dict[str, str]] = []
    previous: AdaptivePose | None = None
    parked_and_open = False
    platform_connected = False
    lighting_connected = False
    try:
        platform.connect()
        platform_connected = True
        lighting.connect()
        lighting_connected = True
        validator._lighting_off(result)
        lighting.projector_cover_close()
        camera.start()
        initial = motion.read_before()
        if not all(math.isclose(value, 0.0, abs_tol=.1) for value in
                   (initial.roll_deg, initial.pitch_deg, initial.z_cm)):
            raise RuntimeError("multiview capture must start at R0/P0/Z0")
        motion.execute_z(15.0); motion.execute_orientation(
            roll_deg=0, pitch_deg=0, before=motion.read_before(), ack_safe_height=True,
        ); motion.execute_z(args.z_start)
        for index, pose in enumerate(poses):
            validator._lighting_off(result)
            target = _pose_for(config, pose, args.z_start)
            if previous is None:
                apply_adaptive_pose_transition(motion, target, None)
            else:
                at_current = _pose_for(config, pose, previous.z_cm)
                apply_adaptive_pose_transition(motion, at_current, None)
                apply_adaptive_pose_transition(motion, target, at_current)
            search.pose_for_z = lambda z, selected=pose: _pose_for(config, selected, z)
            search.before_z = lambda target_pose, old: apply_adaptive_pose_transition(
                motion, target_pose, old,
            )
            pose_root = run_root / f"pose_{index:02d}"
            pose_root.mkdir()
            search.artifact_store = CandidateArtifactStore(pose_root / "automatic_z")
            record: dict[str, Any] = {
                "session": args.session, "material": args.material,
                "split": pose.split, "requested_roll": pose.roll_deg,
                "requested_pitch": pose.pitch_deg, "status": "REJECT",
            }
            try:
                z_result = search.run(pose_id=pose.name, roll=pose.roll_deg,
                                      pitch=pose.pitch_deg)
                if not z_result.success:
                    raise RuntimeError(z_result.failure_reason or "Auto-Z failed")
                selected = validator._select_usable_final_candidate(
                    z_result, pose, {"available": False, "board_quad": None,
                                     "marker_map": {}}, result, pose_root,
                )
                bundle = validator._active_final_candidate_bundle
                assert bundle is not None
                mask = np.asarray(bundle["selected_mask"], dtype=np.uint8)
                patch_cfg = config.patch
                positions = select_patch_positions(
                    bundle["geometry_capture"].frame.color_bgr.shape,
                    patch_cfg.patch_size, patch_cfg.patch_stride,
                    surface_mask=mask, min_surface_coverage=1.0,
                )
                if not positions:
                    raise ValueError("production inspection mask selected zero patches")
                inspection = InspectionMaskResult(
                    mask, int(np.count_nonzero(mask)), None, None,
                )
                save_final_geometry_capture(bundle["geometry_capture"], pose_root,
                                            inspection_mask=inspection)
                validator.depth_contour_roi_saver(
                    pose_root, bundle["depth_contour_roi"],
                    color_bgr=bundle["geometry_capture"].frame.color_bgr,
                )
                rgb_frame = acquire_warmed_final_rgb_frame(
                    camera, warmup_frames=args.final_rgb_warmup_frames,
                    expected_shape=mask.shape,
                )
                save_final_rgb_capture(rgb_frame, pose_root)
                _save_overlay(pose_root / "surface_patch_overlay.png",
                              np.asarray(rgb_frame.color_bgr), mask, positions,
                              patch_cfg.patch_size)
                selected_pose = _pose_for(config, pose, selected.z_command)
                record.update(
                    status="ACCEPT", applied_roll=selected_pose.applied_roll_deg,
                    applied_pitch=selected_pose.applied_pitch_deg,
                    best_z=selected.z_command, quality_score=selected.quality_score,
                    depth_valid_ratio=selected.depth_valid_ratio,
                    roi_type=("depth_rgb_seeded_fallback" if bundle["rgb_fallback_used"]
                              else "depth_external_contour_fill"),
                    inspection_mask_area=int(np.count_nonzero(mask)),
                    selected_patch_count=len(positions),
                    surface_patch_coverage=1.0,
                )
                rows.append({
                    "path": str(pose_root / "final_rgb.png"), "split": pose.split,
                    "label": "normal", "session": args.session,
                    "mask": str(pose_root / "inspection_mask.png"),
                    "source_run": run_root.name, "plane": pose_root.name,
                    "material": args.material, "view": "production_pose",
                    "notes": f"patches={len(positions)};coverage=1.0",
                })
                previous = selected_pose
            except Exception as exc:
                record["reject_reason"] = f"{type(exc).__name__}: {exc}"
                previous = _pose_for(config, pose, float(motion.read_before().z_cm))
            finally:
                validator._lighting_off(result)
                (pose_root / "metadata.json").write_text(
                    json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8",
                )
                records.append(record)
        validator._lighting_off(result)
        _park_platform_and_open_cover(motion, lighting)
        parked_and_open = True
    finally:
        try:
            if lighting_connected:
                try:
                    validator._lighting_off(result)
                except Exception as exc:
                    print(f"[CLEANUP] LED OFF failed: {type(exc).__name__}: {exc}")
        finally:
            if platform_connected and lighting_connected and not parked_and_open:
                try:
                    _park_platform_and_open_cover(motion, lighting)
                except Exception as exc:
                    print(f"[CLEANUP] safe park/cover open skipped: {type(exc).__name__}: {exc}")
            camera.close(); platform.close(); lighting.close()
    summary = {"session": args.session, "material": args.material,
               "poses": records, "accepted": len(rows), "total": len(poses),
               "conveyor_used": False, "manifest_rows": rows}
    (run_root / "capture_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8",
    )
    (run_root / "manifest_ready.json").write_text(
        json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8",
    )
    if args.manifest_output_dir:
        _write_manifests(rows, args.manifest_output_dir)
    return run_root


def run(args: argparse.Namespace, *, confirmation_input=input) -> int:
    poses = validate_args(args)
    print(json.dumps([asdict(pose) for pose in poses], ensure_ascii=False, indent=2))
    if not args.execute:
        print("DRY RUN - no serial device or camera was opened")
        return 0
    if confirmation_input("Type EXECUTE to start multiview capture: ").strip() != "EXECUTE":
        raise ValueError("execution cancelled; no hardware was opened")
    execute_capture(args, poses)
    return 0


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
