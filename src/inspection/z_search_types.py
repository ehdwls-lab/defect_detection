from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class InspectionQualitySample:
    z_cm: float
    depth_valid_ratio: float
    plane_inlier_ratio: float
    plane_residual_mm: float
    object_area_px: int
    surface_area_px: int
    valid_patch_count: int
    touches_fov_edge: bool
    rgb_mean_brightness: float | None = None
    rgb_saturated_ratio: float | None = None
    rgb_sharpness: float | None = None
    gate_passed: bool = False
    quality_score: float | None = None
    reasons: tuple[str, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class BestZResult:
    success: bool
    best_z_cm: float | None
    best_quality: InspectionQualitySample | None
    samples: tuple[InspectionQualitySample, ...]
    failure_reason: str | None
    pose_id: str
