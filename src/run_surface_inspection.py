from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from src.config import InspectionConfig
from src.core.depth_processing import (
    depth_object_candidate,
    fit_inverse_depth_plane_ransac,
    select_final_object_mask,
)
from src.core.inspection_quality import evaluate_inspection_readiness
from src.core.patch_extractor import generate_surface_patches
from src.core.preprocessing import preprocess_surface_image
from src.core.surface_roi import erode_surface_mask
from src.test_surface_only_pose_inspection import (
    create_aruco_detector,
    detect_markers,
    draw_mask_contour,
    draw_status,
    fallback_workspace_mask,
    find_color_profile,
    find_depth_profile,
    get_board_outer_quad,
    make_border_ring,
    polygon_mask,
    render_height_map,
    temporal_median_depth,
    wait_for_aligned_pair,
    configure_camera,
)
from pyorbbecsdk import AlignFilter, Config, OBFrameAggregateOutputMode, OBStreamType, Pipeline

PROJECT_ROOT = Path(__file__).resolve().parents[1]


@dataclass
class PoseTarget:
    pose_id: str = "pose_01"
    pitch: float | None = None
    roll: float | None = None


@dataclass
class InspectionFrameData:
    session: str
    pose: PoseTarget
    timestamp: str
    frame: np.ndarray
    depth_mm: np.ndarray
    workspace_source: str
    marker_ids: list[int]
    object_mask: np.ndarray | None
    surface_mask: np.ndarray | None
    valid_patches: list[dict[str, int | float]]
    quality_metrics: dict[str, Any]
    inspection_ready: bool
    reasons: list[str]


def parse_args() -> argparse.Namespace:
    cfg = InspectionConfig.default()
    parser = argparse.ArgumentParser(
        description="Final surface-only inspection runtime for manual Z inspection." 
    )
    parser.add_argument("--session", type=str, default="inspection_session")
    parser.add_argument("--pose-id", type=str, default="pose_01")
    parser.add_argument("--pose-name", type=str, default="pose_01")
    parser.add_argument("--pitch", type=float, default=0.0)
    parser.add_argument("--roll", type=float, default=0.0)
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "results" / "inspection")
    parser.add_argument("--warmup-frames", type=int, default=30)
    parser.add_argument("--depth-median-frames", type=int, default=cfg.depth.median_frames)
    parser.add_argument("--boundary-margin", type=int, default=cfg.surface_roi.boundary_margin_px)
    parser.add_argument("--patch-size", type=int, default=cfg.patch.patch_size)
    parser.add_argument("--patch-stride", type=int, default=cfg.patch.patch_stride)
    parser.add_argument("--patch-mask-coverage", type=float, default=cfg.patch.patch_mask_coverage)
    parser.add_argument("--min-depth-valid-ratio", type=float, default=cfg.quality.min_depth_valid_ratio)
    parser.add_argument("--min-plane-inlier-ratio", type=float, default=cfg.quality.min_plane_inlier_ratio)
    parser.add_argument("--max-plane-inlier-residual-mm", type=float, default=cfg.quality.max_plane_inlier_residual_mm)
    parser.add_argument("--min-valid-patches", type=int, default=cfg.patch.min_valid_patches)
    parser.add_argument("--ready-streak-frames", type=int, default=cfg.quality.ready_streak_frames)
    parser.add_argument("--fov-edge-margin-px", type=int, default=cfg.surface_roi.fov_edge_margin_px)
    parser.add_argument("--display-width", type=int, default=960)
    parser.add_argument("--display-height", type=int, default=600)
    parser.add_argument("--window-name", type=str, default="Surface-only Inspection")
    parser.add_argument("--save-patches", action="store_true")
    return parser.parse_args()


def build_pose_target(args: argparse.Namespace) -> PoseTarget:
    return PoseTarget(
        pose_id=args.pose_id or args.pose_name,
        pitch=float(args.pitch),
        roll=float(args.roll),
    )


def build_inspection_frame(
    session: str,
    pose: PoseTarget,
    frame: np.ndarray,
    depth_mm: np.ndarray,
    workspace_source: str,
    marker_ids: list[int],
    object_mask: np.ndarray | None,
    surface_mask: np.ndarray | None,
    valid_patches: list[dict[str, int | float]],
    quality_metrics: dict[str, Any],
    inspection_ready: bool,
    reasons: list[str],
) -> InspectionFrameData:
    return InspectionFrameData(
        session=session,
        pose=pose,
        timestamp=datetime.now().strftime("%Y%m%d_%H%M%S_%f"),
        frame=frame.copy(),
        depth_mm=depth_mm.copy(),
        workspace_source=workspace_source,
        marker_ids=marker_ids,
        object_mask=None if object_mask is None else object_mask.copy(),
        surface_mask=None if surface_mask is None else surface_mask.copy(),
        valid_patches=list(valid_patches),
        quality_metrics=dict(quality_metrics),
        inspection_ready=inspection_ready,
        reasons=list(reasons),
    )


def save_runtime_metadata(output_dir: Path, inspection: InspectionFrameData, patch_overlay: np.ndarray) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    image_dir = output_dir / "frames"
    image_dir.mkdir(parents=True, exist_ok=True)

    if inspection.frame is not None:
        cv2.imwrite(str(image_dir / "color.png"), inspection.frame)
    if inspection.depth_mm is not None:
        depth_vis = np.clip((inspection.depth_mm / 2000.0) * 255.0, 0, 255).astype(np.uint8)
        depth_vis = cv2.applyColorMap(depth_vis, cv2.COLORMAP_TURBO)
        cv2.imwrite(str(image_dir / "depth_visualization.png"), depth_vis)
    if inspection.object_mask is not None:
        cv2.imwrite(str(image_dir / "object_mask.png"), inspection.object_mask)
    if inspection.surface_mask is not None:
        cv2.imwrite(str(image_dir / "surface_mask.png"), inspection.surface_mask)
    if patch_overlay is not None:
        cv2.imwrite(str(image_dir / "valid_patch_overlay.png"), patch_overlay)

    payload = {
        "session": inspection.session,
        "pose_id": inspection.pose.pose_id,
        "pitch": inspection.pose.pitch,
        "roll": inspection.pose.roll,
        "timestamp": inspection.timestamp,
        "inspection_ready": inspection.inspection_ready,
        "workspace_source": inspection.workspace_source,
        "marker_ids": inspection.marker_ids,
        "quality_metrics": inspection.quality_metrics,
        "valid_patch_count": len(inspection.valid_patches),
        "patches": inspection.valid_patches,
        "reasons": inspection.reasons,
    }
    metadata_path = output_dir / "inspection_metadata.json"
    metadata_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return metadata_path


def save_patch_images(frame: np.ndarray, valid_patches: list[dict[str, int | float]], out_dir: Path) -> None:
    patch_dir = out_dir / "patches"
    patch_dir.mkdir(parents=True, exist_ok=True)
    for index, patch in enumerate(valid_patches):
        x = int(patch["x"])
        y = int(patch["y"])
        size = int(patch["w"])
        crop = frame[y:y + size, x:x + size]
        cv2.imwrite(str(patch_dir / f"patch_{index:04d}_x{x}_y{y}.png"), crop)


def main() -> None:
    args = parse_args()
    config = InspectionConfig.default()
    pose = build_pose_target(args)
    output_dir = args.output_dir / args.session / args.pose_id
    output_dir.mkdir(parents=True, exist_ok=True)

    pipeline = Pipeline()
    camera_config = Config()
    align_filter = AlignFilter(align_to_stream=OBStreamType.COLOR_STREAM)
    detector = create_aruco_detector()

    depth_history: list[np.ndarray] = []
    ready_streak = 0
    last_reasons: list[str] = []
    started = False

    try:
        color_profile = find_color_profile(pipeline)
        depth_profile = find_depth_profile(pipeline)
        camera_config.enable_stream(color_profile)
        camera_config.enable_stream(depth_profile)
        try:
            camera_config.set_frame_aggregate_output_mode(OBFrameAggregateOutputMode.FULL_FRAME_REQUIRE)
        except Exception:  # pragma: no cover - optional SDK support
            pass

        pipeline.start(camera_config)
        started = True

        device = pipeline.get_device()
        configure_camera(device, type("Args", (), {
            "brightness": int(config.camera.brightness),
            "exposure": int(config.camera.exposure),
            "gain": int(config.camera.gain),
            "white_balance": int(config.camera.white_balance),
            "depth_exposure": int(config.depth.exposure),
            "depth_gain": int(config.depth.gain),
            "depth_median_frames": int(config.depth.median_frames),
        })())

        print("=" * 88)
        print("SURFACE-ONLY FINAL INSPECTION RUNTIME")
        print("Mode: manual Z inspection ready")
        print("Press SPACE to store current frame when READY")
        print("Press Q/ESC to quit")
        print("=" * 88)

        for _ in range(max(1, int(args.warmup_frames))):
            wait_for_aligned_pair(pipeline, align_filter)

        while True:
            frame, depth_mm = wait_for_aligned_pair(pipeline, align_filter)
            depth_history.append(depth_mm)
            max_frames = max(1, int(args.depth_median_frames))
            if len(depth_history) > max_frames:
                depth_history = depth_history[-max_frames:]
            depth_median = temporal_median_depth(depth_history)

            marker_map = detect_markers(frame, detector)
            marker_ids = sorted(marker_map.keys())
            board_quad = get_board_outer_quad(marker_map)
            overlay = frame.copy()

            for marker_id, corners in marker_map.items():
                pts = corners.astype(np.int32)
                cv2.polylines(overlay, [pts], True, (0, 255, 255), 2)
                center = np.mean(corners, axis=0).astype(np.int32)
                cv2.putText(overlay, f"ID {marker_id}", (int(center[0]), int(center[1])), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2, cv2.LINE_AA)

            if board_quad is not None:
                workspace_mask = polygon_mask(frame.shape, board_quad)
                plane_ring = make_border_ring(workspace_mask, 0.18, is_fraction=True)
                workspace_source = "ARUCO"
                cv2.polylines(overlay, [board_quad.astype(np.int32)], True, (0, 255, 0), 3)
            else:
                workspace_mask = fallback_workspace_mask(frame.shape, 80)
                plane_ring = make_border_ring(workspace_mask, 120, is_fraction=False)
                workspace_source = "FALLBACK"

            valid_depth = (
                (workspace_mask > 0)
                & (depth_median >= config.depth.min_mm)
                & (depth_median <= config.depth.max_mm)
            )
            depth_valid_ratio = float(np.count_nonzero(valid_depth) / max(1, np.count_nonzero(workspace_mask > 0)))

            plane_depth, plane_inlier_ratio, plane_residual_mm = fit_inverse_depth_plane_ransac(
                depth_median,
                plane_ring,
                type("Cfg", (), {
                    "depth_min_mm": config.depth.min_mm,
                    "depth_max_mm": config.depth.max_mm,
                    "plane_ransac_mm": config.depth.plane_ransac_mm,
                    "plane_ransac_iters": config.depth.plane_ransac_iters,
                    "plane_min_points": config.depth.plane_min_points,
                    "plane_max_points": config.depth.plane_max_points,
                })(),
            )

            object_mask = None
            surface_mask = None
            candidate_mask = np.zeros(frame.shape[:2], dtype=np.uint8)
            height = np.zeros(frame.shape[:2], dtype=np.float32)
            patches: list[dict[str, int | float]] = []

            plane_good = (
                plane_depth is not None
                and depth_valid_ratio >= args.min_depth_valid_ratio
                and plane_inlier_ratio >= args.min_plane_inlier_ratio
                and plane_residual_mm <= args.max_plane_inlier_residual_mm
            )

            if plane_good:
                candidate_mask, height, _ = depth_object_candidate(
                    depth_median,
                    plane_depth,
                    workspace_mask,
                    type("Cfg", (), {
                        "depth_min_mm": config.depth.min_mm,
                        "depth_max_mm": config.depth.max_mm,
                        "height_threshold_mm": config.depth.height_threshold_mm,
                        "max_object_height_mm": config.depth.max_object_height_mm,
                        "object_open_size": config.depth.object_open_size,
                        "object_close_size": config.depth.object_close_size,
                        "object_close_iterations": config.depth.object_close_iterations,
                    })(),
                )
                object_mask = select_final_object_mask(
                    candidate_mask,
                    workspace_mask,
                    type("Cfg", (), {
                        "min_object_area": config.surface_roi.min_object_area,
                        "max_object_area_ratio": config.surface_roi.max_object_area_ratio,
                    })(),
                )

                if object_mask is not None:
                    surface_mask = erode_surface_mask(object_mask, args.boundary_margin)
                    patches = generate_surface_patches(
                        surface_mask,
                        patch_size=args.patch_size,
                        stride=args.patch_stride,
                        min_coverage=args.patch_mask_coverage,
                    )

            quality_result = evaluate_inspection_readiness(
                object_mask=object_mask,
                surface_mask=surface_mask,
                patches=patches,
                depth_valid_ratio=depth_valid_ratio,
                plane_inlier_ratio=plane_inlier_ratio,
                min_depth_valid_ratio=args.min_depth_valid_ratio,
                min_plane_inlier_ratio=args.min_plane_inlier_ratio,
                max_plane_inlier_residual_mm=args.max_plane_inlier_residual_mm,
                min_valid_patches=args.min_valid_patches,
                fov_edge_margin_px=args.fov_edge_margin_px,
                plane_residual_mm=plane_residual_mm,
            )
            current_ready = quality_result.ready
            ready_reasons = quality_result.reasons

            if current_ready:
                ready_streak += 1
            else:
                ready_streak = 0
            good_for_inspection = current_ready and ready_streak >= args.ready_streak_frames
            last_reasons = ready_reasons

            if object_mask is not None:
                draw_mask_contour(overlay, object_mask, (255, 0, 255), 4)
            if surface_mask is not None:
                draw_mask_contour(overlay, surface_mask, (0, 255, 255), 2)
            for patch in patches:
                x = int(patch["x"])
                y = int(patch["y"])
                w = int(patch["w"])
                h = int(patch["h"])
                cv2.rectangle(overlay, (x, y), (x + w - 1, y + h - 1), (255, 255, 0), 1)

            object_area = int(np.count_nonzero(object_mask)) if object_mask is not None else 0
            surface_area = int(np.count_nonzero(surface_mask)) if surface_mask is not None else 0
            if good_for_inspection:
                state_text = "STATE: INSPECTION READY - PRESS SPACE"
            elif current_ready:
                state_text = f"STATE: HOLD Z / STABILIZING {ready_streak}/{args.ready_streak_frames}"
            else:
                state_text = "STATE: ADJUST Z"

            reason_text = " | ".join(ready_reasons[:2]) if ready_reasons else "All quality gates passed"
            draw_status(
                overlay,
                [
                    state_text,
                    f"Workspace={workspace_source} | IDs={marker_ids}",
                    f"Depth valid={depth_valid_ratio * 100.0:.1f}% | Plane inlier={plane_inlier_ratio * 100.0:.1f}% | residual={plane_residual_mm:.2f} mm",
                    f"Object area={object_area}px | Surface area={surface_area}px",
                    f"Patch={args.patch_size}x{args.patch_size} | stride={args.patch_stride} | valid patches={len(patches)} (need>={args.min_valid_patches})",
                    f"Boundary margin={args.boundary_margin}px | coverage>={args.patch_mask_coverage:.2f}",
                    f"Pose={args.pose_id} | pitch={args.pitch} | roll={args.roll}",
                    f"Z guidance: {reason_text}",
                    "PURPLE=object | YELLOW=surface-only | CYAN=AE patches",
                ],
                good=good_for_inspection,
            )

            height_vis = render_height_map(height, max_mm=max(40.0, config.depth.height_threshold_mm * 10.0))
            preview = cv2.resize(overlay, (args.display_width, args.display_height), interpolation=cv2.INTER_AREA)
            cv2.imshow(args.window_name, preview)
            cv2.imshow("Height From Board Plane", cv2.resize(height_vis, (640, 400), interpolation=cv2.INTER_AREA))
            if surface_mask is not None:
                cv2.imshow("Surface Mask", cv2.resize(surface_mask, (640, 400), interpolation=cv2.INTER_NEAREST))

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), ord("Q"), 27):
                break
            if key == ord("r") or key == ord("R"):
                ready_streak = 0
                last_reasons = []
            if key == 32:
                if not good_for_inspection:
                    print("[SPACE] INSPECTION READY not satisfied")
                    if last_reasons:
                        print("  - " + "\n  - ".join(last_reasons))
                    continue

                inspection = build_inspection_frame(
                    session=args.session,
                    pose=pose,
                    frame=frame,
                    depth_mm=depth_median,
                    workspace_source=workspace_source,
                    marker_ids=marker_ids,
                    object_mask=object_mask,
                    surface_mask=surface_mask,
                    valid_patches=patches,
                    quality_metrics={
                        "depth_valid_ratio": depth_valid_ratio,
                        "plane_inlier_ratio": plane_inlier_ratio,
                        "plane_residual_mm": plane_residual_mm,
                        "object_area_px": object_area,
                        "surface_area_px": surface_area,
                        "valid_patch_count": len(patches),
                        "inspection_ready": good_for_inspection,
                        "ready_streak_frames": ready_streak,
                        "pose_id": args.pose_id,
                    },
                    inspection_ready=good_for_inspection,
                    reasons=ready_reasons,
                )
                save_dir = output_dir / inspection.timestamp
                save_runtime_metadata(save_dir, inspection, overlay)
                if args.save_patches:
                    save_patch_images(inspection.frame, inspection.valid_patches, save_dir)
                print(f"[SAVE] {save_dir}")

    finally:
        if started:
            pipeline.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
