"""Verified surface-only external-contour-fill ROI for final RGB inspection."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from src.config import InspectionConfig
from src.core.aruco_board import marker_local_board_sample_mask, polygon_mask
from src.core.depth_processing import (
    depth_object_candidate,
    fill_external_object_contour,
    normal_angle_error_deg,
    select_main_object_component,
)
from src.integration.platform_pose_calibration import (
    PlatformPoseCalibrationError,
    predicted_board_normal_from_platform_pose,
)
from src.integration.metric_pose import (
    CameraIntrinsics, MetricFitConfig, MetricPoseError,
    backproject_depth_pixels, fit_metric_plane_ransac,
)
from src.core.surface_roi import erode_surface_mask
from src.core.workspace import fallback_workspace_mask, make_border_ring


class DepthExternalContourROIError(ValueError):
    """Raised when the current Depth cannot produce a guarded object contour."""


@dataclass(frozen=True)
class DepthExternalContourROIResult:
    workspace_mask: np.ndarray
    depth_object_candidate_mask: np.ndarray
    depth_main_component_mask: np.ndarray
    depth_object_contour_filled: np.ndarray
    inspection_mask: np.ndarray
    workspace_source: str
    workspace_area_px: int
    depth_candidate_area_px: int
    depth_main_component_area_px: int
    filled_object_area_px: int
    inspection_area_px: int
    fill_gain_px: int
    fill_gain_ratio: float
    depth_valid_ratio: float
    board_plane_inlier_ratio: float
    board_plane_residual_mm: float
    board_plane_source: str = "current_pose_outer_ring"
    board_plane_point_count: int = 0
    candidate_signed_height_median_mm: float | None = None
    candidate_signed_height_p05_mm: float | None = None
    candidate_signed_height_p95_mm: float | None = None
    roi_depth_frame_count: int = 1
    roi_min_votes: int = 1
    fused_candidate_area_px: int = 0
    closed_component_area_px: int = 0
    closing_gain_px: int = 0
    silhouette_completion: str = "none"
    hull_area_px: int = 0
    hull_expansion_ratio: float = 1.0
    hull_used: bool = False
    board_plane_fit_mask: np.ndarray | None = field(default=None, repr=False)
    board_plane_overlay: np.ndarray | None = field(default=None, repr=False)
    depth_candidate_frames: tuple[np.ndarray, ...] = field(default_factory=tuple, repr=False)
    depth_vote_count: np.ndarray | None = field(default=None, repr=False)
    depth_signed_height: np.ndarray | None = field(default=None, repr=False)
    depth_closed_component_mask: np.ndarray | None = field(default=None, repr=False)
    predicted_board_normal: tuple[float, float, float] | None = None
    plane_normal: tuple[float, float, float] | None = None
    normal_angle_error_deg: float | None = None
    plane_hypotheses: tuple[dict[str, Any], ...] = ()
    selected_board_plane_hypothesis_index: int | None = None
    aruco_detected_ids: tuple[int, ...] = ()
    aruco_detected_count: int = 0
    partial_aruco_sample_px: int = 0
    plane_hypothesis_masks: tuple[np.ndarray, ...] = field(default_factory=tuple, repr=False)

    def metadata(self) -> dict[str, Any]:
        payload = asdict(self)
        for name in (
            "workspace_mask", "depth_object_candidate_mask",
            "depth_main_component_mask", "depth_object_contour_filled",
            "inspection_mask",
            "board_plane_fit_mask", "board_plane_overlay", "depth_candidate_frames",
            "depth_vote_count", "depth_signed_height", "depth_closed_component_mask",
            "plane_hypothesis_masks",
        ):
            payload.pop(name)
        payload["candidate_area_px"] = payload["depth_candidate_area_px"]
        return payload


def build_depth_external_contour_roi(
    depth_mm: np.ndarray,
    image_shape: tuple[int, ...],
    config: InspectionConfig,
    *,
    board_quad: np.ndarray | None = None,
    marker_map: dict[int, np.ndarray] | None = None,
    intrinsics: CameraIntrinsics | None = None,
    depth_frames: tuple[np.ndarray, ...] | list[np.ndarray] | None = None,
    min_votes: int = 2,
    current_platform_roll_deg: float | None = None,
    current_platform_pitch_deg: float | None = None,
    commanded_platform_roll_deg: float | None = None,
    commanded_platform_pitch_deg: float | None = None,
) -> DepthExternalContourROIResult:
    """Fuse current-pose Depth candidates, then fill the selected silhouette."""
    frames = [np.asarray(item, dtype=np.float32) for item in (depth_frames or (depth_mm,))]
    if not frames:
        raise DepthExternalContourROIError("at least one final ROI Depth frame is required")
    depth = frames[0]
    if depth.ndim != 2 or len(image_shape) < 2 or tuple(image_shape[:2]) != depth.shape:
        raise DepthExternalContourROIError(
            "aligned final Depth and image shape are required"
        )
    if intrinsics is None:
        raise DepthExternalContourROIError("metric board-plane fitting requires camera intrinsics")
    intrinsics.validate()
    marker_map = marker_map or {}
    detected_ids = tuple(sorted(int(value) for value in marker_map))
    fallback_workspace = fallback_workspace_mask(
        image_shape, config.surface_roi.fallback_workspace_margin_px,
    )
    if board_quad is not None:
        workspace = polygon_mask(image_shape, board_quad)
        plane_ring = make_border_ring(
            workspace, config.hybrid_roi.board_plane_border_fraction,
            is_fraction=True,
        )
        workspace_source = "aruco"
        board_plane_source = "aruco_full_board"
    else:
        workspace = fallback_workspace
        workspace_source = "fallback"
        if marker_map:
            plane_ring = marker_local_board_sample_mask(
                image_shape, marker_map,
                outer_margin_px=config.surface_roi.partial_aruco_outer_margin_px,
                exclusion_margin_px=config.surface_roi.partial_aruco_exclusion_margin_px,
            )
            plane_ring = np.where(
                (plane_ring > 0) & (workspace > 0), 255, 0,
            ).astype(np.uint8)
            board_plane_source = "aruco_partial_local_depth"
        else:
            plane_ring = workspace.copy()
            board_plane_source = "depth_spatial_multi_plane"

    workspace_area = int(np.count_nonzero(workspace))
    if workspace_area == 0:
        raise DepthExternalContourROIError("inspection workspace is empty")
    if workspace_area == workspace.size:
        raise DepthExternalContourROIError("full-frame inspection workspace is forbidden")
    valid = (
        (workspace > 0)
        & np.isfinite(depth)
        & (depth >= config.depth.min_mm)
        & (depth <= config.depth.max_mm)
    )
    depth_valid_ratio = float(np.count_nonzero(valid) / workspace_area)
    candidates: list[np.ndarray] = []
    heights: list[np.ndarray] = []
    plane_depth = None
    inlier_ratio = 0.0
    residual = float("inf")
    plane_point_count = 0
    predicted_normal = None
    pose_values = (
        current_platform_roll_deg, current_platform_pitch_deg,
        commanded_platform_roll_deg, commanded_platform_pitch_deg,
    )
    if all(value is not None for value in pose_values):
        try:
            predicted_normal = predicted_board_normal_from_platform_pose(*pose_values)
        except PlatformPoseCalibrationError as exc:
            raise DepthExternalContourROIError(f"board normal prior unavailable: {exc}") from exc
    elif any(value is not None for value in pose_values):
        raise DepthExternalContourROIError("board normal prior requires all platform R/P values")
    hypotheses: list[dict[str, Any]] = []
    hypothesis_masks: list[np.ndarray] = []
    selected_normal = None
    selected_angle = None
    selected_index = None
    fit_config = MetricFitConfig(
        min_depth_mm=config.depth.min_mm, max_depth_mm=config.depth.max_mm,
        ransac_threshold_mm=config.depth.plane_ransac_mm,
        ransac_iterations=config.depth.plane_ransac_iters,
        min_points=config.depth.plane_min_points,
        max_points=config.depth.plane_max_points,
    )

    def render_metric_plane(normal: np.ndarray, center: np.ndarray) -> np.ndarray:
        yy, xx = np.mgrid[0:depth.shape[0], 0:depth.shape[1]]
        rx = (xx.astype(np.float64) - intrinsics.cx) / intrinsics.fx
        ry = (yy.astype(np.float64) - intrinsics.cy) / intrinsics.fy
        denominator = normal[0] * rx + normal[1] * ry + normal[2]
        numerator = float(np.dot(normal, center))
        rendered = np.zeros(depth.shape, dtype=np.float32)
        okay = np.abs(denominator) > 1e-12
        z = np.zeros(depth.shape, dtype=np.float64)
        z[okay] = numerator / denominator[okay]
        okay &= np.isfinite(z) & (z > 0)
        rendered[okay] = z[okay].astype(np.float32)
        return rendered

    def find_planes(frame: np.ndarray, sample: np.ndarray, maximum: int):
        sample_valid = (
            (sample > 0) & np.isfinite(frame)
            & (frame >= config.depth.min_mm) & (frame <= config.depth.max_mm)
        )
        vv, uu = np.nonzero(sample_valid)
        pixels = np.column_stack((uu, vv))
        xyz, _ = backproject_depth_pixels(frame, pixels, intrinsics, fit_config)
        remaining_pixels, remaining_xyz = pixels, xyz
        found = []
        for _ in range(maximum):
            if len(remaining_xyz) < fit_config.min_points:
                break
            try:
                normal, center, fit_residual, _ = fit_metric_plane_ransac(
                    remaining_xyz, fit_config,
                )
            except MetricPoseError:
                break
            distances = np.abs((remaining_xyz - center) @ normal)
            inliers = distances <= fit_config.ransac_threshold_mm
            support = int(np.count_nonzero(inliers))
            if support < fit_config.min_points:
                break
            support_mask = np.zeros(frame.shape, dtype=np.uint8)
            support_pixels = remaining_pixels[inliers]
            support_mask[support_pixels[:, 1], support_pixels[:, 0]] = 255
            found.append((
                normal, center, float(np.median(distances[inliers])),
                support / max(1, len(pixels)), support_mask,
            ))
            remaining_pixels = remaining_pixels[~inliers]
            remaining_xyz = remaining_xyz[~inliers]
        return found

    effective_source = board_plane_source
    for frame_index, frame in enumerate(frames):
        if frame.shape != depth.shape:
            raise DepthExternalContourROIError("final ROI Depth frames are not aligned")
        frame_valid = (
            (workspace > 0) & np.isfinite(frame)
            & (frame >= config.depth.min_mm) & (frame <= config.depth.max_mm)
        )
        if float(np.count_nonzero(frame_valid) / workspace_area) < config.quality.min_depth_valid_ratio:
            continue
        maximum = config.surface_roi.max_spatial_plane_hypotheses if effective_source == "depth_spatial_multi_plane" else 1
        found = find_planes(frame, plane_ring, maximum)
        if not found and effective_source == "aruco_partial_local_depth":
            effective_source = "depth_spatial_multi_plane"
            found = find_planes(frame, workspace, config.surface_roi.max_spatial_plane_hypotheses)
        acceptable = []
        edge_band = make_border_ring(workspace, 10)
        for normal, center, frame_residual, frame_inlier, support_mask in found:
            frame_plane = render_metric_plane(normal, center)
            frame_angle = None if predicted_normal is None else normal_angle_error_deg(normal, predicted_normal)
            candidate_mask, candidate_height, _ = depth_object_candidate(frame, frame_plane, workspace, config.depth)
            area = int(np.count_nonzero(candidate_mask))
            area_ratio = area / workspace_area
            main_guard = select_main_object_component(candidate_mask, workspace, config.surface_roi)
            rejected = None
            if frame_residual > config.quality.max_plane_inlier_residual_mm:
                rejected = "plane residual exceeds quality limit"
            elif predicted_normal is not None and frame_angle > config.surface_roi.board_normal_prior_max_error_deg:
                rejected = "metric normal exceeds board-normal guard"
            elif area_ratio > config.surface_roi.max_object_area_ratio:
                rejected = "resulting object occupies too much workspace"
            elif main_guard is None:
                rejected = "no plausible positive-height object"
            elif np.any((main_guard > 0) & (edge_band > 0)):
                rejected = "resulting object is peripheral/background"
            positive = candidate_height[candidate_mask > 0]
            ys_obj, xs_obj = np.where(candidate_mask > 0)
            hypothesis_index = len(hypotheses)
            metadata = {
                "frame_index": frame_index, "metric_normal": normal.tolist(),
                "normal_angle_error_deg": frame_angle,
                "support_px": int(np.count_nonzero(support_mask)),
                "support_ratio": float(frame_inlier),
                "median_depth_mm": float(np.median(frame[support_mask > 0])),
                "plane_offset_mm": float(np.dot(normal, center)),
                "inlier_ratio": float(frame_inlier), "residual_mm": frame_residual,
                "resulting_object_area_px": area,
                "resulting_object_area_ratio": float(area_ratio),
                "object_centroid": None if not len(xs_obj) else [float(np.mean(xs_obj)), float(np.mean(ys_obj))],
                "object_touches_workspace_edge": rejected == "resulting object is peripheral/background",
                "object_positive_height_median_mm": None if not positive.size else float(np.median(positive)),
                "accepted": rejected is None, "rejected_reason": rejected,
            }
            hypotheses.append(metadata)
            hypothesis_masks.append(support_mask)
            if rejected is None:
                acceptable.append(((float("inf") if frame_angle is None else round(frame_angle, 3),
                                    -frame_inlier, -metadata["median_depth_mm"]),
                                   hypothesis_index, frame_plane, normal,
                                   frame_residual, frame_inlier, candidate_mask,
                                   candidate_height, support_mask))
        if not acceptable:
            continue
        acceptable.sort(key=lambda item: item[0])
        _, chosen, frame_plane, frame_normal, frame_residual, frame_inlier, candidate, height, support_mask = acceptable[0]
        candidates.append(candidate)
        heights.append(height)
        if plane_depth is None:
            plane_depth, inlier_ratio, residual = frame_plane, frame_inlier, frame_residual
            plane_point_count = int(np.count_nonzero(support_mask))
            selected_normal = frame_normal
            selected_angle = hypotheses[chosen]["normal_angle_error_deg"]
            selected_index = chosen
            board_plane_source = effective_source
    if not candidates:
        raise DepthExternalContourROIError(
            "current board plane is not inspection-ready"
        )
    min_votes = max(1, min(int(min_votes), len(candidates)))
    vote_count = np.sum(np.stack([candidate > 0 for candidate in candidates]), axis=0)
    candidate = np.where((vote_count >= min_votes) & (workspace > 0), 255, 0).astype(np.uint8)
    selected_heights = np.concatenate([
        height[candidate_frame > 0] for height, candidate_frame in zip(heights, candidates)
    ])
    if not selected_heights.size or float(np.median(selected_heights)) <= 0:
        raise DepthExternalContourROIError("candidate signed-height polarity is invalid")
    main = select_main_object_component(
        candidate, workspace, config.surface_roi,
    )
    if main is None:
        raise DepthExternalContourROIError("main Depth object component was not found")
    from src.core.depth_processing import close_object_component, guarded_convex_hull
    closed = close_object_component(main, workspace, config.surface_roi)
    filled = fill_external_object_contour(closed, workspace)
    if filled is None:
        raise DepthExternalContourROIError("Depth object external contour was not found")

    closed_area = int(np.count_nonzero(closed))
    hull, hull_area, hull_expansion, hull_used = guarded_convex_hull(
        closed, workspace, config.surface_roi,
    )
    if (
        hull_used and hull is not None
        and hull_expansion > 1.05
        and int(np.count_nonzero(filled)) <= int(closed_area * 1.02)
    ):
        filled = fill_external_object_contour(hull, workspace)
    else:
        hull_used = False
        hull_area = 0
        hull_expansion = 1.0

    candidate_area = int(np.count_nonzero(candidate))
    main_area = int(np.count_nonzero(main))
    filled_area = int(np.count_nonzero(filled))
    if filled_area == 0:
        raise DepthExternalContourROIError("filled Depth object contour is empty")
    if filled_area / workspace_area > config.surface_roi.max_object_area_ratio:
        raise DepthExternalContourROIError(
            "filled Depth object contour exceeds the workspace area guard"
        )
    if np.any((filled > 0) & (workspace == 0)):
        raise DepthExternalContourROIError("filled object escaped the workspace")

    inspection = erode_surface_mask(
        filled, config.surface_roi.boundary_margin_px,
    )
    inspection = np.where(
        (inspection > 0) & (filled > 0) & (workspace > 0), 255, 0,
    ).astype(np.uint8)
    inspection_area = int(np.count_nonzero(inspection))
    if inspection_area == 0:
        raise DepthExternalContourROIError(
            "inspection mask became empty after boundary erosion"
        )
    gain = filled_area - main_area
    selected_height_values = selected_heights[np.isfinite(selected_heights)]
    close_gain = closed_area - main_area
    completion = "guarded_hull" if hull_used else ("morph_close" if close_gain > 0 else "none")
    return DepthExternalContourROIResult(
        workspace_mask=workspace,
        depth_object_candidate_mask=candidate,
        depth_main_component_mask=main,
        depth_object_contour_filled=filled,
        inspection_mask=inspection,
        workspace_source=workspace_source,
        workspace_area_px=workspace_area,
        depth_candidate_area_px=candidate_area,
        depth_main_component_area_px=main_area,
        filled_object_area_px=filled_area,
        inspection_area_px=inspection_area,
        fill_gain_px=gain,
        fill_gain_ratio=float(gain / main_area),
        depth_valid_ratio=depth_valid_ratio,
        board_plane_inlier_ratio=float(inlier_ratio),
        board_plane_residual_mm=float(residual),
        board_plane_source=board_plane_source,
        board_plane_point_count=plane_point_count,
        candidate_signed_height_median_mm=float(np.median(selected_height_values)),
        candidate_signed_height_p05_mm=float(np.percentile(selected_height_values, 5)),
        candidate_signed_height_p95_mm=float(np.percentile(selected_height_values, 95)),
        roi_depth_frame_count=len(frames),
        roi_min_votes=min_votes,
        fused_candidate_area_px=candidate_area,
        closed_component_area_px=closed_area,
        closing_gain_px=close_gain,
        silhouette_completion=completion,
        hull_area_px=hull_area,
        hull_expansion_ratio=hull_expansion,
        hull_used=hull_used,
        board_plane_fit_mask=plane_ring,
        depth_candidate_frames=tuple(candidates),
        depth_vote_count=vote_count.astype(np.uint8),
        depth_signed_height=heights[0],
        depth_closed_component_mask=closed,
        predicted_board_normal=(
            None if predicted_normal is None else tuple(float(value) for value in predicted_normal)
        ),
        plane_normal=(
            None if selected_normal is None else tuple(float(value) for value in selected_normal)
        ),
        normal_angle_error_deg=None if selected_angle is None else float(selected_angle),
        plane_hypotheses=tuple(hypotheses),
        selected_board_plane_hypothesis_index=selected_index,
        aruco_detected_ids=detected_ids,
        aruco_detected_count=len(detected_ids),
        partial_aruco_sample_px=(
            int(np.count_nonzero(plane_ring))
            if detected_ids and board_quad is None else 0
        ),
        plane_hypothesis_masks=tuple(hypothesis_masks),
    )


def save_depth_external_contour_roi_artifacts(
    output_directory: str | Path,
    result: DepthExternalContourROIResult,
    *,
    color_bgr: np.ndarray | None = None,
) -> dict[str, str]:
    """Persist the evidence, selected component, filled contour and final ROI."""
    import cv2

    root = Path(output_directory).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    paths: dict[str, str] = {}

    def save(name: str, image: np.ndarray) -> None:
        path = root / name
        if not cv2.imwrite(str(path), np.asarray(image)):
            raise RuntimeError(f"failed to save depth contour ROI artifact: {path}")
        paths[name] = str(path)

    save("depth_object_candidate_mask.png", result.depth_object_candidate_mask)
    for index, frame_mask in enumerate(result.depth_candidate_frames):
        save(f"depth_object_candidate_frame_{index:02d}.png", frame_mask)
    if result.depth_vote_count is not None:
        save("depth_object_vote_count.png", result.depth_vote_count)
    if result.depth_signed_height is not None:
        height = np.asarray(result.depth_signed_height, dtype=np.float32)
        finite = np.isfinite(height)
        visual = np.zeros(height.shape, dtype=np.uint8)
        if np.any(finite):
            visual[finite] = np.clip(128.0 + height[finite] * 2.0, 0, 255).astype(np.uint8)
        save("depth_signed_height.png", visual)
    if result.board_plane_fit_mask is not None:
        save("board_plane_sample_mask.png", result.board_plane_fit_mask)
        save("board_plane_fit_mask.png", result.board_plane_fit_mask)
        if color_bgr is not None:
            color = np.asarray(color_bgr)
            if color.ndim == 3 and color.shape[:2] == result.board_plane_fit_mask.shape:
                board_overlay = color.copy()
                selected = result.selected_board_plane_hypothesis_index
                selected_mask = (
                    result.plane_hypothesis_masks[selected]
                    if selected is not None and selected < len(result.plane_hypothesis_masks)
                    else result.board_plane_fit_mask
                )
                board_overlay[selected_mask > 0] = (255, 255, 0)
                save("board_plane_overlay.png", board_overlay)
    for index, hypothesis_mask in enumerate(result.plane_hypothesis_masks[:3]):
        save(f"depth_plane_hypothesis_{index:02d}.png", hypothesis_mask)
    if result.depth_closed_component_mask is not None:
        save("depth_closed_component_mask.png", result.depth_closed_component_mask)
    save("depth_main_component_mask.png", result.depth_main_component_mask)
    save("depth_object_contour_filled.png", result.depth_object_contour_filled)
    save("inspection_mask.png", result.inspection_mask)
    if color_bgr is not None:
        color = np.asarray(color_bgr)
        if color.ndim == 3 and color.shape[:2] == result.inspection_mask.shape:
            overlay = color.copy()
            magenta = np.zeros_like(overlay)
            magenta[..., 0] = 255
            magenta[..., 2] = 255
            yellow = np.zeros_like(overlay)
            yellow[..., 1] = 255
            yellow[..., 2] = 255
            overlay = np.where(
                (result.depth_object_contour_filled > 0)[..., None],
                cv2.addWeighted(overlay, 0.65, magenta, 0.35, 0), overlay,
            )
            overlay = np.where(
                (result.inspection_mask > 0)[..., None],
                cv2.addWeighted(overlay, 0.60, yellow, 0.40, 0), overlay,
            )
            save("inspection_mask_overlay.png", overlay)
    return paths
