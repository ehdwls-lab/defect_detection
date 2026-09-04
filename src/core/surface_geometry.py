"""Shared depth geometry used by Automatic Z and final anomaly inference."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from src.config import InspectionConfig
from src.core.depth_processing import (
    depth_object_candidate,
    fit_inverse_depth_plane_ransac,
    select_final_object_mask,
)
from src.core.inspection_quality import InspectionQualityResult, evaluate_inspection_readiness
from src.core.patch_extractor import generate_surface_patches
from src.core.surface_roi import erode_surface_mask, mask_touches_frame_edge
from src.core.workspace import fallback_workspace_mask, make_border_ring


@dataclass(frozen=True)
class SurfaceGeometryResult:
    object_mask: np.ndarray | None
    surface_mask: np.ndarray | None
    patches: tuple[dict[str, int | float], ...]
    object_area_px: int
    surface_area_px: int
    surface_ratio: float | None
    depth_valid_ratio: float
    plane_inlier_ratio: float
    plane_residual: float
    fov_edge_contact: bool


def evaluate_surface_geometry_readiness(
    geometry: SurfaceGeometryResult,
    config: InspectionConfig,
) -> InspectionQualityResult:
    """Apply the shared production geometry gates without a patch-count gate."""
    return evaluate_inspection_readiness(
        geometry.object_mask,
        geometry.surface_mask,
        list(geometry.patches),
        geometry.depth_valid_ratio,
        geometry.plane_inlier_ratio,
        geometry.plane_residual,
        min_depth_valid_ratio=config.quality.min_depth_valid_ratio,
        min_plane_inlier_ratio=config.quality.min_plane_inlier_ratio,
        max_plane_inlier_residual_mm=config.quality.max_plane_inlier_residual_mm,
        min_valid_patches=None,
        fov_edge_margin_px=config.surface_roi.fov_edge_margin_px,
    )


def extract_surface_geometry(
    depth_mm: np.ndarray,
    image_shape: tuple[int, ...],
    config: InspectionConfig,
    *,
    patch_min_coverage: float | None = None,
) -> SurfaceGeometryResult:
    """Extract a fresh object/surface mask using the production depth primitives."""
    depth = np.asarray(depth_mm, dtype=np.float32)
    if depth.ndim != 2:
        raise ValueError("depth_mm must be a 2D aligned depth image")
    if len(image_shape) < 2 or tuple(image_shape[:2]) != depth.shape:
        raise ValueError(
            f"aligned RGB/depth shape mismatch: RGB={tuple(image_shape[:2])}, "
            f"Depth={depth.shape}"
        )

    workspace = fallback_workspace_mask(
        image_shape, config.surface_roi.fallback_workspace_margin_px,
    )
    plane_ring = make_border_ring(
        workspace, config.surface_roi.fallback_plane_ring_px, is_fraction=False,
    )
    workspace_pixels = workspace > 0
    valid_depth = (
        workspace_pixels
        & (depth >= config.depth.min_mm)
        & (depth <= config.depth.max_mm)
    )
    depth_valid_ratio = float(
        np.count_nonzero(valid_depth) / max(1, np.count_nonzero(workspace_pixels))
    )
    plane_depth, plane_inlier_ratio, plane_residual = fit_inverse_depth_plane_ransac(
        depth, plane_ring, config.depth,
    )

    object_mask = surface_mask = None
    patches: list[dict[str, Any]] = []
    plane_good = (
        plane_depth is not None
        and depth_valid_ratio >= config.quality.min_depth_valid_ratio
        and plane_inlier_ratio >= config.quality.min_plane_inlier_ratio
        and plane_residual <= config.quality.max_plane_inlier_residual_mm
    )
    if plane_good:
        candidate_mask, _, _ = depth_object_candidate(
            depth, plane_depth, workspace, config.depth,
        )
        object_mask = select_final_object_mask(
            candidate_mask, workspace, config.surface_roi,
        )
        if object_mask is not None:
            surface_mask = erode_surface_mask(
                object_mask, config.surface_roi.boundary_margin_px,
            )
            coverage = (
                config.patch.patch_mask_coverage
                if patch_min_coverage is None else float(patch_min_coverage)
            )
            patches = generate_surface_patches(
                surface_mask,
                config.patch.patch_size,
                config.patch.patch_stride,
                coverage,
            )

    object_area_px = 0 if object_mask is None else int(np.count_nonzero(object_mask))
    surface_area_px = 0 if surface_mask is None else int(np.count_nonzero(surface_mask))
    surface_ratio = (
        float(surface_area_px / object_area_px) if object_area_px > 0 else None
    )
    return SurfaceGeometryResult(
        object_mask=object_mask,
        surface_mask=surface_mask,
        patches=tuple(patches),
        object_area_px=object_area_px,
        surface_area_px=surface_area_px,
        surface_ratio=surface_ratio,
        depth_valid_ratio=depth_valid_ratio,
        plane_inlier_ratio=float(plane_inlier_ratio),
        plane_residual=float(plane_residual),
        fov_edge_contact=mask_touches_frame_edge(
            object_mask, config.surface_roi.fov_edge_margin_px,
        ),
    )
