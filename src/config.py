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
    fallback_workspace_margin_px: int = 80
    fallback_plane_ring_px: int = 120
    inspection_close_size_px: int = 9
    inspection_close_iterations: int = 1
    roi_depth_frames: int = 5
    roi_min_votes: int = 2
    hull_max_expansion_ratio: float = 1.5
    hull_max_frame_area_ratio: float = 0.75
    board_normal_prior_max_error_deg: float = 35.0
    partial_aruco_outer_margin_px: int = 28
    partial_aruco_exclusion_margin_px: int = 5
    max_spatial_plane_hypotheses: int = 3
    min_patchable_ratio: float = 0.5
    rgb_fallback_lab_distance: float = 35.0


@dataclass(frozen=True)
class HybridROIConfig:
    enabled: bool = True
    use_for_anomaly: bool = False
    aruco_rgb_warmup_frames: int = 3
    depth_flush_frames_after_led_off: int = 3
    board_plane_border_fraction: float = 0.18
    marker_ignore_px: int = 8
    board_plane_tolerance_mm: float = 3.0
    lab_similarity_threshold: float = 35.0
    unknown_recovery_radius_px: int = 40
    close_size_px: int = 9
    close_iterations: int = 1
    max_board_frame_area_ratio: float = 0.95
    max_hybrid_frame_area_ratio: float = 0.90


@dataclass(frozen=True)
class PatchConfig:
    patch_size: int = 64
    patch_stride: int = 32
    patch_mask_coverage: float = 1.0
    min_valid_patches: int = 6


@dataclass(frozen=True)
class InspectionQualityConfig:
    min_depth_valid_ratio: float = 0.25
    min_plane_inlier_ratio: float = 0.25
    max_plane_inlier_residual_mm: float = 2.0
    ready_streak_frames: int = 4
    inspection_roll_limit_deg: float = 25.0
    inspection_pitch_limit_deg: float = 25.0
    tilt_entry_z_cm: float = 25.0
    search_min_z_cm: float = 17.0
    search_max_z_cm: float = 25.0
    search_step_cm: float = 1.0
    tilt_envelope: tuple[tuple[float, float], ...] = (
        (17.0, 20.0), (18.0, 21.0), (19.0, 22.0),
        (20.0, 23.0), (21.0, 25.0),
    )
    max_final_candidate_attempts: int = 3


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
    hybrid_roi: HybridROIConfig = field(default_factory=HybridROIConfig)
    patch: PatchConfig = field(default_factory=PatchConfig)
    quality: InspectionQualityConfig = field(default_factory=InspectionQualityConfig)
    preprocessing: PreprocessingConfig = field(default_factory=PreprocessingConfig)

    @classmethod
    def default(cls) -> "InspectionConfig":
        return cls()


DEFAULT_INSPECTION_CONFIG = InspectionConfig.default()
