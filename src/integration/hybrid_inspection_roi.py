"""ArUco + current aligned Depth + conservative RGB UNKNOWN recovery."""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from src.config import InspectionConfig
from src.core.aruco_board import draw_aruco_overlay, polygon_mask
from src.core.workspace import make_border_ring
from src.integration.metric_pose import (
    CameraIntrinsics,
    MetricFitConfig,
    MetricPoseError,
    backproject_depth_pixels,
    fit_metric_plane_ransac,
)


class HybridROIError(ValueError):
    """Raised when the optional hybrid ROI is unsafe; callers must fall back."""


@dataclass(frozen=True)
class HybridInspectionROIResult:
    inspection_mask: np.ndarray
    board_roi_mask: np.ndarray
    board_background_mask: np.ndarray
    depth_object_mask: np.ndarray
    depth_unknown_mask: np.ndarray
    rgb_recovered_unknown_mask: np.ndarray
    board_plane_normal: np.ndarray
    board_plane_center_mm: np.ndarray
    board_plane_inlier_ratio: float
    board_plane_residual_mm: float
    board_roi_area_px: int
    depth_object_area_px: int
    depth_unknown_area_px: int
    hybrid_inspection_area_px: int
    hybrid_to_depth_object_ratio: float
    board_roi_depth_valid_ratio: float
    depth_p05_mm: float
    depth_median_mm: float
    depth_p95_mm: float
    intrinsics_source: str

    def metadata(self) -> dict[str, Any]:
        payload = asdict(self)
        for key in (
            "inspection_mask", "board_roi_mask", "board_background_mask",
            "depth_object_mask", "depth_unknown_mask", "rgb_recovered_unknown_mask",
        ):
            payload.pop(key)
        payload["board_plane_normal"] = self.board_plane_normal.tolist()
        payload["board_plane_center_mm"] = self.board_plane_center_mm.tolist()
        payload["board_plane_valid"] = True
        return payload


def _validate_config(config: InspectionConfig) -> None:
    cfg = config.hybrid_roi
    numeric = (
        cfg.board_plane_border_fraction, cfg.board_plane_tolerance_mm,
        cfg.lab_similarity_threshold, cfg.max_board_frame_area_ratio,
        cfg.max_hybrid_frame_area_ratio,
    )
    if not all(math.isfinite(float(value)) for value in numeric):
        raise HybridROIError("hybrid ROI configuration must be finite")
    if not 0 < cfg.board_plane_border_fraction < 0.5:
        raise HybridROIError("board_plane_border_fraction must be in (0, 0.5)")
    if cfg.board_plane_tolerance_mm <= 0 or cfg.lab_similarity_threshold <= 0:
        raise HybridROIError("hybrid metric/color thresholds must be positive")
    if cfg.marker_ignore_px < 0 or cfg.unknown_recovery_radius_px < 0:
        raise HybridROIError("hybrid pixel margins must be non-negative")
    if cfg.close_size_px < 1 or cfg.close_size_px % 2 == 0:
        raise HybridROIError("hybrid close_size_px must be a positive odd integer")
    if cfg.close_iterations < 0:
        raise HybridROIError("hybrid close_iterations must be non-negative")
    if not 0 < cfg.max_hybrid_frame_area_ratio < 1:
        raise HybridROIError("max_hybrid_frame_area_ratio must be in (0, 1)")
    if not 0 < cfg.max_board_frame_area_ratio <= 1:
        raise HybridROIError("max_board_frame_area_ratio must be in (0, 1]")


def _largest_component(mask: np.ndarray) -> np.ndarray:
    import cv2

    binary = np.where(mask > 0, 255, 0).astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    if count <= 1:
        return np.zeros_like(binary)
    selected = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return np.where(labels == selected, 255, 0).astype(np.uint8)


def _enclosed_unknown(main_object: np.ndarray, unknown: np.ndarray) -> np.ndarray:
    """Return UNKNOWN pixels enclosed by the main depth-confirmed object."""
    import cv2

    inverse = np.where(main_object == 0, 255, 0).astype(np.uint8)
    count, labels, _, _ = cv2.connectedComponentsWithStats(inverse, connectivity=8)
    exterior = set(np.unique(np.concatenate((
        labels[0], labels[-1], labels[:, 0], labels[:, -1],
    ))).tolist())
    enclosed = np.zeros_like(main_object)
    for label in range(1, count):
        if label not in exterior:
            enclosed[(labels == label) & (unknown > 0)] = 255
    return enclosed


def _marker_exclusion_mask(
    shape: tuple[int, int], marker_map: dict[int, np.ndarray], margin_px: int,
) -> np.ndarray:
    import cv2

    mask = np.zeros(shape, dtype=np.uint8)
    for corners in marker_map.values():
        cv2.fillConvexPoly(mask, np.rint(corners).astype(np.int32), 255)
    if margin_px > 0 and np.any(mask):
        size = margin_px * 2 + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
        mask = cv2.dilate(mask, kernel, iterations=1)
    return mask


def _depth_percentiles(depth: np.ndarray, valid: np.ndarray) -> tuple[float, float, float]:
    values = np.asarray(depth, dtype=np.float32)[valid]
    if not values.size:
        raise HybridROIError("board ROI has no valid aligned Depth")
    p05, median, p95 = np.percentile(values, (5, 50, 95))
    return float(p05), float(median), float(p95)


def build_hybrid_inspection_roi(
    color_bgr: np.ndarray,
    depth_mm: np.ndarray,
    intrinsics: CameraIntrinsics,
    board_quad: np.ndarray,
    marker_map: dict[int, np.ndarray],
    config: InspectionConfig,
) -> HybridInspectionROIResult:
    """Build a current-pose ROI. UNKNOWN remains distinct until explicit recovery."""
    import cv2

    _validate_config(config)
    color = np.asarray(color_bgr)
    depth = np.asarray(depth_mm, dtype=np.float32)
    if color.ndim != 3 or color.shape[2] != 3 or depth.ndim != 2:
        raise HybridROIError("hybrid ROI requires BGR and 2D aligned Depth")
    if color.shape[:2] != depth.shape:
        raise HybridROIError("hybrid RGB/Depth shape mismatch")
    intrinsics.validate()
    if depth.shape != (intrinsics.height, intrinsics.width):
        raise HybridROIError("hybrid Depth/intrinsics grid mismatch")

    board = polygon_mask(color.shape, board_quad)
    board_area = int(np.count_nonzero(board))
    if board_area == 0:
        raise HybridROIError("ArUco board polygon is empty")
    if board_area / board.size >= config.hybrid_roi.max_board_frame_area_ratio:
        raise HybridROIError("ArUco board polygon is nearly full-frame")

    valid_depth = (
        (board > 0) & np.isfinite(depth)
        & (depth >= config.depth.min_mm) & (depth <= config.depth.max_mm)
    )
    board_valid_ratio = float(np.count_nonzero(valid_depth) / board_area)
    p05, median, p95 = _depth_percentiles(depth, valid_depth)

    ring = make_border_ring(
        board, config.hybrid_roi.board_plane_border_fraction, is_fraction=True,
    )
    marker_exclusion = _marker_exclusion_mask(
        depth.shape, marker_map, config.hybrid_roi.marker_ignore_px,
    )
    ring[(marker_exclusion > 0) | ~valid_depth] = 0
    vv, uu = np.nonzero(ring)
    ring_pixels = np.column_stack((uu, vv))
    fit_config = MetricFitConfig(
        min_depth_mm=config.depth.min_mm,
        max_depth_mm=config.depth.max_mm,
        ransac_threshold_mm=config.depth.plane_ransac_mm,
        ransac_iterations=config.depth.plane_ransac_iters,
        min_points=config.depth.plane_min_points,
        max_points=config.depth.plane_max_points,
    )
    try:
        ring_xyz, _ = backproject_depth_pixels(depth, ring_pixels, intrinsics, fit_config)
        normal, center, residual, inlier_ratio = fit_metric_plane_ransac(
            ring_xyz, fit_config,
        )
    except MetricPoseError as exc:
        raise HybridROIError(f"current board-plane fit failed: {exc}") from exc
    if (
        inlier_ratio < config.quality.min_plane_inlier_ratio
        or not math.isfinite(residual)
        or residual > config.quality.max_plane_inlier_residual_mm
    ):
        raise HybridROIError(
            "current board-plane quality rejected: "
            f"inlier={inlier_ratio:.6f}, residual={residual:.6f} mm"
        )

    vv, uu = np.nonzero(valid_depth)
    board_pixels = np.column_stack((uu, vv))
    board_xyz, _ = backproject_depth_pixels(depth, board_pixels, intrinsics, fit_config)
    # fit_metric_plane_ransac normal is camera-facing (Orbbec +Z is away from
    # camera), therefore positive signed distance is height above the board.
    signed_distance = (board_xyz - center) @ normal
    object_selected = (
        (signed_distance >= config.depth.height_threshold_mm)
        & (signed_distance <= config.depth.max_object_height_mm)
    )
    board_selected = np.abs(signed_distance) <= config.hybrid_roi.board_plane_tolerance_mm
    depth_object = np.zeros(depth.shape, dtype=np.uint8)
    board_background = np.zeros(depth.shape, dtype=np.uint8)
    depth_object[vv[object_selected], uu[object_selected]] = 255
    board_background[vv[board_selected], uu[board_selected]] = 255
    unknown = np.where((board > 0) & ~valid_depth, 255, 0).astype(np.uint8)

    main_object = _largest_component(depth_object)
    object_area = int(np.count_nonzero(main_object))
    if object_area == 0:
        raise HybridROIError("hybrid ROI has no metric Depth object evidence")
    enclosed = _enclosed_unknown(main_object, unknown)

    radius = config.hybrid_roi.unknown_recovery_radius_px
    if radius > 0:
        size = radius * 2 + 1
        neighborhood = cv2.dilate(
            main_object,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size)),
            iterations=1,
        )
    else:
        neighborhood = main_object.copy()
    rgb_candidates = (unknown > 0) & (neighborhood > 0) & (enclosed == 0)
    lab = cv2.cvtColor(color, cv2.COLOR_BGR2LAB).astype(np.float32)
    object_lab = lab[main_object > 0]
    median_lab = np.median(object_lab, axis=0)
    lab_distance = np.linalg.norm(lab - median_lab, axis=2)
    rgb_recovered = np.where(
        rgb_candidates
        & (lab_distance <= config.hybrid_roi.lab_similarity_threshold),
        255, 0,
    ).astype(np.uint8)

    hybrid = cv2.bitwise_or(main_object, enclosed)
    hybrid = cv2.bitwise_or(hybrid, rgb_recovered)
    if config.hybrid_roi.close_iterations > 0:
        close_size = config.hybrid_roi.close_size_px
        hybrid = cv2.morphologyEx(
            hybrid, cv2.MORPH_CLOSE,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_size, close_size)),
            iterations=config.hybrid_roi.close_iterations,
        )
    # Never convert metric board/background into object through morphology.
    allowed = ((depth_object > 0) | (unknown > 0)) & (board > 0)
    hybrid = np.where((hybrid > 0) & allowed, 255, 0).astype(np.uint8)
    hybrid = _largest_component(hybrid)
    margin = config.surface_roi.boundary_margin_px
    if margin > 0:
        size = margin * 2 + 1
        hybrid = cv2.erode(
            hybrid, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size)),
            iterations=1,
        )
    hybrid = cv2.bitwise_and(hybrid, board)
    hybrid_area = int(np.count_nonzero(hybrid))
    if hybrid_area == 0:
        raise HybridROIError("hybrid inspection mask is empty")
    if np.any((hybrid > 0) & (board == 0)):
        raise HybridROIError("hybrid inspection mask escaped the board ROI")
    if hybrid_area / hybrid.size >= config.hybrid_roi.max_hybrid_frame_area_ratio:
        raise HybridROIError("hybrid inspection mask is nearly full-frame")
    if hybrid_area > board_area:
        raise HybridROIError("hybrid inspection mask exceeds board ROI area")

    return HybridInspectionROIResult(
        inspection_mask=hybrid,
        board_roi_mask=board,
        board_background_mask=board_background,
        depth_object_mask=main_object,
        depth_unknown_mask=unknown,
        rgb_recovered_unknown_mask=rgb_recovered,
        board_plane_normal=normal,
        board_plane_center_mm=center,
        board_plane_inlier_ratio=float(inlier_ratio),
        board_plane_residual_mm=float(residual),
        board_roi_area_px=board_area,
        depth_object_area_px=object_area,
        depth_unknown_area_px=int(np.count_nonzero(unknown)),
        hybrid_inspection_area_px=hybrid_area,
        hybrid_to_depth_object_ratio=float(hybrid_area / object_area),
        board_roi_depth_valid_ratio=board_valid_ratio,
        depth_p05_mm=p05,
        depth_median_mm=median,
        depth_p95_mm=p95,
        intrinsics_source=intrinsics.source,
    )


def save_hybrid_roi_artifacts(
    output_directory: str | Path,
    *,
    aruco_rgb: np.ndarray,
    marker_map: dict[int, np.ndarray],
    board_quad: np.ndarray | None,
    result: HybridInspectionROIResult | None = None,
) -> dict[str, str]:
    """Save available diagnostics for both successful and fallback attempts."""
    import cv2

    root = Path(output_directory).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    paths: dict[str, str] = {}

    def save(name: str, image: np.ndarray) -> None:
        path = root / name
        if not cv2.imwrite(str(path), np.asarray(image)):
            raise RuntimeError(f"failed to save hybrid ROI artifact: {path}")
        paths[name] = str(path)

    save("aruco_rgb.png", aruco_rgb)
    save("aruco_overlay.png", draw_aruco_overlay(aruco_rgb, marker_map, board_quad))
    if marker_map and board_quad is None:
        save("aruco_partial_overlay.png", draw_aruco_overlay(aruco_rgb, marker_map, None))
    if board_quad is not None:
        save("board_roi_mask.png", polygon_mask(np.asarray(aruco_rgb).shape, board_quad))
    if result is None:
        return paths
    save("board_roi_mask.png", result.board_roi_mask)
    save("board_background_mask.png", result.board_background_mask)
    save("depth_object_mask.png", result.depth_object_mask)
    save("depth_unknown_mask.png", result.depth_unknown_mask)
    save("hybrid_inspection_mask.png", result.inspection_mask)
    board_overlay = np.asarray(aruco_rgb).copy()
    board_overlay[result.board_background_mask > 0] = (
        0.65 * board_overlay[result.board_background_mask > 0]
        + 0.35 * np.array([255, 0, 0])
    ).astype(np.uint8)
    board_overlay[result.depth_object_mask > 0] = (
        0.65 * board_overlay[result.depth_object_mask > 0]
        + 0.35 * np.array([0, 0, 255])
    ).astype(np.uint8)
    board_overlay[result.depth_unknown_mask > 0] = (
        0.65 * board_overlay[result.depth_unknown_mask > 0]
        + 0.35 * np.array([0, 255, 255])
    ).astype(np.uint8)
    save("board_plane_overlay.png", board_overlay)
    hybrid_overlay = np.asarray(aruco_rgb).copy()
    contours, _ = cv2.findContours(
        result.inspection_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE,
    )
    cv2.drawContours(hybrid_overlay, contours, -1, (0, 255, 0), 2)
    save("hybrid_inspection_overlay.png", hybrid_overlay)
    metadata_path = root / "hybrid_roi_diagnostics.json"
    metadata_path.write_text(
        json.dumps(result.metadata(), ensure_ascii=False, indent=2), encoding="utf-8",
    )
    paths[metadata_path.name] = str(metadata_path)
    return paths
