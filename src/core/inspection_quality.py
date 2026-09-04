from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from src.core.surface_roi import mask_touches_frame_edge


@dataclass
class InspectionQualityResult:
    ready: bool
    reasons: list[str] = field(default_factory=list)
    metrics: dict[str, Any] = field(default_factory=dict)


def evaluate_inspection_readiness(
    object_mask: np.ndarray | None,
    surface_mask: np.ndarray | None,
    patches: list[dict[str, float | int]],
    depth_valid_ratio: float,
    plane_inlier_ratio: float,
    plane_residual_mm: float,
    min_depth_valid_ratio: float = 0.25,
    min_plane_inlier_ratio: float = 0.25,
    max_plane_inlier_residual_mm: float = 2.0,
    min_valid_patches: int | None = 20,
    fov_edge_margin_px: int = 18,
) -> InspectionQualityResult:
    """Check whether the current frame is ready for surface-only inspection."""
    reasons: list[str] = []

    if depth_valid_ratio < min_depth_valid_ratio:
        reasons.append(
            f"Depth valid low ({depth_valid_ratio * 100.0:.1f}% < {min_depth_valid_ratio * 100.0:.0f}%)"
        )

    if plane_inlier_ratio < min_plane_inlier_ratio:
        reasons.append(
            f"Plane inlier low ({plane_inlier_ratio * 100.0:.1f}% < {min_plane_inlier_ratio * 100.0:.0f}%)"
        )

    if not np.isfinite(plane_residual_mm) or plane_residual_mm > max_plane_inlier_residual_mm:
        residual_text = f"{plane_residual_mm:.2f}" if np.isfinite(plane_residual_mm) else "inf"
        reasons.append(
            f"Plane residual high ({residual_text} mm > {max_plane_inlier_residual_mm:.1f} mm)"
        )

    object_area_px = 0 if object_mask is None else int(np.count_nonzero(object_mask))
    surface_area_px = 0 if surface_mask is None else int(np.count_nonzero(surface_mask))
    surface_ratio = (
        float(surface_area_px / object_area_px) if object_area_px > 0 else None
    )

    if object_area_px == 0:
        reasons.append("Object surface not found")
    if surface_area_px == 0:
        reasons.append("Surface-only mask not found")
    if min_valid_patches is not None and len(patches) < min_valid_patches:
        reasons.append(f"Too few valid patches ({len(patches)} < {min_valid_patches})")
    if mask_touches_frame_edge(object_mask, fov_edge_margin_px):
        reasons.append("Object too close to image edge / FOV")

    result = InspectionQualityResult(
        ready=(len(reasons) == 0),
        reasons=reasons,
        metrics={
            "depth_valid_ratio": float(depth_valid_ratio),
            "plane_inlier_ratio": float(plane_inlier_ratio),
            "plane_residual_mm": float(plane_residual_mm),
            "valid_patch_count": int(len(patches)),
            "usable_patch_count": int(len(patches)),
            "object_area_px": object_area_px,
            "surface_area_px": surface_area_px,
            "surface_ratio": surface_ratio,
        },
    )
    return result
