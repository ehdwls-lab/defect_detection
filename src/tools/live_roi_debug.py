"""Live, motion-free viewer for the production final anomaly ROI."""

from __future__ import annotations

import argparse
from collections import deque
from datetime import datetime
import json
import math
from pathlib import Path
import sys
from typing import Any

import cv2
import numpy as np

if __package__ in (None, ""):
    repository_root = Path(__file__).resolve().parents[2]
    if str(repository_root) not in sys.path:
        sys.path.insert(0, str(repository_root))

from src.camera.orbbec_controller import OrbbecCameraController
from src.config import InspectionConfig
from src.core.aruco_board import create_aruco_detector, detect_markers, get_board_outer_quad
from src.core.depth_contour_roi import (
    DepthExternalContourROIResult,
    build_depth_external_contour_roi,
    save_depth_external_contour_roi_artifacts,
)
from src.core.patch_extractor import generate_surface_patches
from src.core.surface_geometry import extract_surface_geometry
from src.integration.platform_pose_calibration import predicted_board_normal_from_platform_pose
from src.lighting.serial_controller import SerialLightingConfig, SerialLightingController


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
STAGE_NAMES = {
    ord("1"): "aligned_depth", ord("2"): "board_plane_inliers",
    ord("3"): "signed_height", ord("4"): "per_frame_candidate",
    ord("5"): "vote_fusion", ord("6"): "main_component",
    ord("7"): "closed_component", ord("8"): "contour_filled",
    ord("9"): "inspection_mask", ord("0"): "patch_overlay",
}


def finite_number(raw: str) -> float:
    value = float(raw)
    if not math.isfinite(value):
        raise argparse.ArgumentTypeError("pose angles must be finite")
    return value


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Motion-free live viewer for the production final anomaly ROI",
    )
    parser.add_argument("--roll", required=True, type=finite_number)
    parser.add_argument("--pitch", required=True, type=finite_number)
    parser.add_argument(
        "--output-root", type=Path,
        default=REPOSITORY_ROOT / "results" / "live_roi_debug",
    )
    parser.add_argument("--camera-warmup-frames", type=int, default=30)
    parser.add_argument("--compare-legacy", action="store_true")
    parser.add_argument("--lighting-port")
    parser.add_argument(
        "--execute-camera", action="store_true",
        help="explicitly allow opening the camera; no motion subsystem is imported or opened",
    )
    return parser


def _gray_bgr(mask: np.ndarray | None) -> np.ndarray:
    if mask is None:
        return np.zeros((64, 64, 3), dtype=np.uint8)
    value = np.asarray(mask)
    if value.dtype != np.uint8:
        finite = np.isfinite(value)
        visual = np.zeros(value.shape, dtype=np.uint8)
        if np.any(finite):
            low, high = np.percentile(value[finite], (2, 98))
            if high > low:
                visual[finite] = np.clip((value[finite] - low) * 255 / (high - low), 0, 255)
        value = visual
    return cv2.cvtColor(value, cv2.COLOR_GRAY2BGR)


def depth_visualization(depth_mm: np.ndarray) -> np.ndarray:
    depth = np.asarray(depth_mm, dtype=np.float32)
    valid = np.isfinite(depth) & (depth > 0)
    gray = np.zeros(depth.shape, dtype=np.uint8)
    if np.any(valid):
        low, high = np.percentile(depth[valid], (2, 98))
        if high > low:
            gray[valid] = np.clip((high - depth[valid]) * 255 / (high - low), 0, 255)
    return cv2.applyColorMap(gray, cv2.COLORMAP_TURBO)


def patch_overlay(
    color_bgr: np.ndarray, inspection_mask: np.ndarray, config: InspectionConfig,
) -> tuple[np.ndarray, list[dict[str, int | float]]]:
    overlay = np.asarray(color_bgr).copy()
    contours, _ = cv2.findContours(
        np.asarray(inspection_mask, dtype=np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE,
    )
    cv2.drawContours(overlay, contours, -1, (0, 255, 0), 2)
    patches = generate_surface_patches(
        inspection_mask, config.patch.patch_size, config.patch.patch_stride,
        config.patch.patch_mask_coverage,
    )
    for patch in patches:
        x, y, w, h = (int(patch[key]) for key in ("x", "y", "w", "h"))
        cv2.rectangle(overlay, (x, y), (x + w - 1, y + h - 1), (255, 255, 0), 1)
    return overlay, patches


def stage_images(
    depth_mm: np.ndarray, color_bgr: np.ndarray,
    result: DepthExternalContourROIResult,
    patch_image: np.ndarray,
) -> dict[str, np.ndarray]:
    selected = result.selected_board_plane_hypothesis_index
    inliers = (
        result.plane_hypothesis_masks[selected]
        if selected is not None and selected < len(result.plane_hypothesis_masks)
        else result.board_plane_fit_mask
    )
    board_overlay = np.asarray(color_bgr).copy()
    if inliers is not None:
        board_overlay[np.asarray(inliers) > 0] = (255, 255, 0)
    return {
        "aligned_depth": depth_visualization(depth_mm),
        "board_plane_inliers": board_overlay,
        "signed_height": _gray_bgr(result.depth_signed_height),
        "per_frame_candidate": _gray_bgr(
            result.depth_candidate_frames[-1] if result.depth_candidate_frames else None
        ),
        "vote_fusion": _gray_bgr(result.depth_vote_count),
        "main_component": _gray_bgr(result.depth_main_component_mask),
        "closed_component": _gray_bgr(result.depth_closed_component_mask),
        "contour_filled": _gray_bgr(result.depth_object_contour_filled),
        "inspection_mask": _gray_bgr(result.inspection_mask),
        "patch_overlay": patch_image,
    }


def annotate(
    image: np.ndarray, result: DepthExternalContourROIResult | None,
    *, status: str, patch_count: int, roll: float, pitch: float,
) -> np.ndarray:
    output = np.asarray(image).copy()
    lines = [f"ROI={status}  R={roll:+.2f} P={pitch:+.2f}"]
    if result is not None:
        angle = result.normal_angle_error_deg
        lines.extend((
            f"source={result.board_plane_source} hypothesis={result.selected_board_plane_hypothesis_index}",
            f"normal_error={'N/A' if angle is None else f'{angle:.2f} deg'} object={result.depth_candidate_area_px}",
            f"filled={result.filled_object_area_px} inspection={result.inspection_area_px} patches={patch_count}",
        ))
    for index, line in enumerate(lines):
        cv2.putText(output, line, (10, 24 + index * 24), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(output, line, (10, 24 + index * 24), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, (255, 255, 255), 1, cv2.LINE_AA)
    return output


def save_snapshot(
    root: Path, *, color: np.ndarray, depth: np.ndarray,
    result: DepthExternalContourROIResult, patches: list[dict[str, Any]],
    patch_image: np.ndarray, legacy_mask: np.ndarray | None,
) -> Path:
    target = root / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    target.mkdir(parents=True, exist_ok=False)
    cv2.imwrite(str(target / "rgb.png"), color)
    cv2.imwrite(str(target / "depth_visualization.png"), depth_visualization(depth))
    cv2.imwrite(str(target / "object_candidate.png"), result.depth_object_candidate_mask)
    cv2.imwrite(str(target / "vote_count.png"), result.depth_vote_count)
    cv2.imwrite(str(target / "main_component.png"), result.depth_main_component_mask)
    cv2.imwrite(str(target / "closed_component.png"), result.depth_closed_component_mask)
    cv2.imwrite(str(target / "contour_filled.png"), result.depth_object_contour_filled)
    cv2.imwrite(str(target / "surface_patch_overlay.png"), patch_image)
    if legacy_mask is not None:
        cv2.imwrite(str(target / "legacy_inspection_mask.png"), legacy_mask)
    save_depth_external_contour_roi_artifacts(target, result, color_bgr=color)
    payload = result.metadata()
    payload["patches"] = patches
    payload["snapshot_directory"] = str(target)
    (target / "diagnostics.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8",
    )
    return target


def cleanup_lighting(lighting: Any | None, connected: bool) -> None:
    if lighting is None or not connected:
        return
    try:
        lighting.inspection_off()
    finally:
        lighting.close()


def run_live(args: argparse.Namespace) -> None:
    config = InspectionConfig.default()
    camera = OrbbecCameraController(config, warmup_frames=args.camera_warmup_frames)
    lighting = (
        SerialLightingController(SerialLightingConfig(args.lighting_port))
        if args.lighting_port else None
    )
    detector = create_aruco_detector()
    rolling: deque[np.ndarray] = deque(maxlen=config.surface_roi.roi_depth_frames)
    selected_stage = "patch_overlay"
    lighting_on = False
    lighting_connected = False
    camera.start()
    try:
        if lighting is not None:
            lighting.connect()
            lighting_connected = True
            lighting.inspection_off()
        while True:
            frame = camera.capture()
            color = np.asarray(frame.color_bgr)
            depth = np.asarray(frame.depth_mm, dtype=np.float32)
            rolling.append(depth.copy())
            display = color.copy()
            result = None
            patches: list[dict[str, Any]] = []
            legacy_mask = None
            status = f"BUFFER {len(rolling)}/{rolling.maxlen}"
            images = {"patch_overlay": display, "aligned_depth": depth_visualization(depth)}
            if len(rolling) == rolling.maxlen:
                try:
                    marker_map = detect_markers(color, detector)
                    board_quad = get_board_outer_quad(marker_map)
                    intrinsics = camera.color_intrinsics(depth.shape[1], depth.shape[0])
                    result = build_depth_external_contour_roi(
                        depth, color.shape, config, board_quad=board_quad,
                        marker_map=marker_map, intrinsics=intrinsics,
                        # The production helper treats depth_frames[0] as the
                        # reference grid/diagnostic frame, so keep "now" first.
                        depth_frames=list(reversed(rolling)),
                        min_votes=config.surface_roi.roi_min_votes,
                        current_platform_roll_deg=args.roll,
                        current_platform_pitch_deg=args.pitch,
                        commanded_platform_roll_deg=args.roll,
                        commanded_platform_pitch_deg=args.pitch,
                    )
                    display, patches = patch_overlay(color, result.inspection_mask, config)
                    images = stage_images(depth, color, result, display)
                    status = "READY"
                    if args.compare_legacy:
                        legacy = extract_surface_geometry(depth, color.shape, config)
                        legacy_mask = legacy.surface_mask
                except Exception as exc:
                    status = f"RECHECK {type(exc).__name__}: {exc}"
            annotated = annotate(
                display, result, status=status, patch_count=len(patches),
                roll=args.roll, pitch=args.pitch,
            )
            cv2.imshow("LIVE RGB + INSPECTION + CYAN PATCH", annotated)
            cv2.imshow("ROI INTERMEDIATE", images.get(selected_stage, annotated))
            if args.compare_legacy and legacy_mask is not None:
                legacy_overlay, _ = patch_overlay(color, legacy_mask, config)
                cv2.imshow("LEGACY LEFT | PRODUCTION RIGHT", np.hstack((legacy_overlay, display)))
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q")):
                break
            if key in STAGE_NAMES:
                selected_stage = STAGE_NAMES[key]
            elif key == ord("r"):
                rolling.clear()
            elif key == ord("s") and result is not None:
                saved = save_snapshot(
                    args.output_root, color=color, depth=depth, result=result,
                    patches=patches, patch_image=display, legacy_mask=legacy_mask,
                )
                print(f"[LIVE ROI] snapshot={saved}")
            elif key == ord("l") and lighting is not None:
                lighting_on = not lighting_on
                lighting.inspection_on() if lighting_on else lighting.inspection_off()
    finally:
        cleanup_lighting(lighting, lighting_connected)
        camera.close()
        cv2.destroyAllWindows()


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    normal = predicted_board_normal_from_platform_pose(
        args.roll, args.pitch, args.roll, args.pitch,
    )
    print("LIVE ROI DEBUG (motion subsystems are not opened)")
    print(f"R/P={args.roll:+.3f}/{args.pitch:+.3f}")
    print(f"predicted_board_normal={normal.tolist()}")
    print(f"rolling_frames={InspectionConfig.default().surface_roi.roi_depth_frames}")
    print(f"min_votes={InspectionConfig.default().surface_roi.roi_min_votes}")
    if not args.execute_camera:
        print("DRY RUN - camera and lighting were not opened; pass --execute-camera to start")
        return 0
    run_live(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
