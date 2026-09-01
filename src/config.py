from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class CameraConfig:
    width: int = 1280
    height: int = 800
    fps: int = 10
    brightness: int = 48
    exposure: int = 1100
    gain: int = 64
    white_balance: int = 4600
    auto_exposure: bool = False
    auto_white_balance: bool = False


@dataclass(frozen=True)
class DepthConfig:
    exposure: int = 3000
    gain: int = 16
    auto_exposure: bool = False
    median_frames: int = 5
    min_mm: float = 80.0
    max_mm: float = 2000.0
    height_threshold_mm: float = 4.0
    max_object_height_mm: float = 250.0
    plane_ransac_mm: float = 2.5
    plane_ransac_iters: int = 160
    plane_min_points: int = 500
    plane_max_points: int = 7000
    object_open_size: int = 5
    object_close_size: int = 21
    object_close_iterations: int = 2


@dataclass(frozen=True)
class SurfaceROIConfig:
    boundary_margin_px: int = 10
    min_object_area: int = 10000
    max_object_area_ratio: float = 0.75
    fov_edge_margin_px: int = 18


@dataclass(frozen=True)
class PatchConfig:
    patch_size: int = 64
    patch_stride: int = 32
    patch_mask_coverage: float = 1.0
    min_valid_patches: int = 20


@dataclass(frozen=True)
class InspectionQualityConfig:
    min_depth_valid_ratio: float = 0.25
    min_plane_inlier_ratio: float = 0.25
    max_plane_inlier_residual_mm: float = 2.0
    ready_streak_frames: int = 8


@dataclass(frozen=True)
class PreprocessingConfig:
    gamma: float = 0.82
    clahe_clip: float = 1.5
    tile_grid_size: tuple[int, int] = (8, 8)
    sigma: float = 1.0
    unsharp_alpha: float = 1.30
    unsharp_beta: float = -0.30
    unsharp_amount: float = 0.30


@dataclass(frozen=True)
class InspectionConfig:
    camera: CameraConfig = field(default_factory=CameraConfig)
    depth: DepthConfig = field(default_factory=DepthConfig)
    surface_roi: SurfaceROIConfig = field(default_factory=SurfaceROIConfig)
    patch: PatchConfig = field(default_factory=PatchConfig)
    quality: InspectionQualityConfig = field(default_factory=InspectionQualityConfig)
    preprocessing: PreprocessingConfig = field(default_factory=PreprocessingConfig)

    @classmethod
    def default(cls) -> "InspectionConfig":
        return cls()


DEFAULT_INSPECTION_CONFIG = InspectionConfig.default()
