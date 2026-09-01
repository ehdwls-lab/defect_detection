"""Common core modules for the final surface-only inspection pipeline."""

from .depth_processing import (
    depth_object_candidate,
    fit_inverse_depth_plane_ransac,
    select_final_object_mask,
)
from .inspection_quality import evaluate_inspection_readiness
from .patch_extractor import extract_valid_surface_patches, generate_surface_patches
from .preprocessing import preprocess_surface_image
from .surface_roi import erode_surface_mask, mask_touches_frame_edge

__all__ = [
    "depth_object_candidate",
    "erode_surface_mask",
    "evaluate_inspection_readiness",
    "extract_valid_surface_patches",
    "fit_inverse_depth_plane_ransac",
    "generate_surface_patches",
    "mask_touches_frame_edge",
    "preprocess_surface_image",
    "select_final_object_mask",
]
