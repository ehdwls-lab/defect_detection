from __future__ import annotations

import numpy as np

from src.config import InspectionConfig
from src.core.depth_processing import fit_inverse_depth_plane_ransac
from src.core.inspection_quality import evaluate_inspection_readiness
from src.core.patch_extractor import generate_surface_patches
from src.core.preprocessing import preprocess_surface_image
from src.core.surface_roi import erode_surface_mask, mask_touches_frame_edge
from src.core.workspace import fallback_workspace_mask, make_border_ring
from src.preprocessing import preprocess_anomaly
from src.test_surface_only_pose_inspection import (
    erode_surface_mask as ref_erode_surface_mask,
    evaluate_inspection_readiness as ref_evaluate_inspection_readiness,
    generate_surface_patches as ref_generate_surface_patches,
    mask_touches_frame_edge as ref_mask_touches_frame_edge,
    fallback_workspace_mask as ref_fallback_workspace_mask,
    make_border_ring as ref_make_border_ring,
)


def test_preprocessing_matches_shared_reference():
    rng = np.random.default_rng(13)
    image = rng.integers(0, 256, size=(120, 160, 3), dtype=np.uint8)
    config = InspectionConfig.default()

    result = preprocess_surface_image(image, config)
    expected = preprocess_anomaly(
        image,
        gamma=config.preprocessing.gamma,
        clahe_clip=config.preprocessing.clahe_clip,
        unsharp_amount=config.preprocessing.unsharp_amount,
    )

    assert result.shape == expected.shape
    assert np.array_equal(result, expected)


def test_surface_patch_generation_matches_reference():
    mask = np.zeros((128, 128), dtype=np.uint8)
    mask[10:110, 20:100] = 255

    result = generate_surface_patches(mask, patch_size=64, stride=32, min_coverage=1.0)
    expected = ref_generate_surface_patches(mask, 64, 32, 1.0)

    assert result == expected


def test_surface_mask_erosion_matches_reference():
    mask = np.zeros((64, 64), dtype=np.uint8)
    mask[10:55, 12:52] = 255

    result = erode_surface_mask(mask, 10)
    expected = ref_erode_surface_mask(mask, 10)

    assert np.array_equal(result, expected)


def test_ready_gates_match_reference_behavior():
    object_mask = np.zeros((128, 128), dtype=np.uint8)
    object_mask[20:100, 30:90] = 255
    surface_mask = object_mask.copy()
    patches = [{"x": 0, "y": 0, "w": 64, "h": 64, "coverage": 1.0}]

    cfg = InspectionConfig.default()
    result = evaluate_inspection_readiness(
        object_mask=object_mask,
        surface_mask=surface_mask,
        patches=patches,
        depth_valid_ratio=0.6,
        plane_inlier_ratio=0.8,
        plane_residual_mm=1.0,
        min_depth_valid_ratio=cfg.quality.min_depth_valid_ratio,
        min_plane_inlier_ratio=cfg.quality.min_plane_inlier_ratio,
        max_plane_inlier_residual_mm=cfg.quality.max_plane_inlier_residual_mm,
        min_valid_patches=cfg.patch.min_valid_patches,
        fov_edge_margin_px=cfg.surface_roi.fov_edge_margin_px,
    )

    expected = ref_evaluate_inspection_readiness(
        object_mask,
        surface_mask,
        patches,
        0.6,
        0.8,
        1.0,
        type("Args", (), {
            "min_depth_valid_ratio": cfg.quality.min_depth_valid_ratio,
            "min_plane_inlier_ratio": cfg.quality.min_plane_inlier_ratio,
            "max_plane_inlier_residual_mm": cfg.quality.max_plane_inlier_residual_mm,
            "min_valid_patches": cfg.patch.min_valid_patches,
            "fov_edge_margin_px": cfg.surface_roi.fov_edge_margin_px,
        })(),
    )

    assert result.ready == expected[0]
    assert result.reasons == expected[1]


def test_mask_touches_frame_edge_matches_reference():
    mask = np.zeros((64, 64), dtype=np.uint8)
    mask[20:40, 25:45] = 255

    assert mask_touches_frame_edge(mask, 10) == ref_mask_touches_frame_edge(mask, 10)


def test_workspace_helpers_match_reference():
    shape = (800, 1280, 3)
    result = fallback_workspace_mask(shape, 80)
    expected = ref_fallback_workspace_mask(shape, 80)
    assert np.array_equal(result, expected)
    assert np.array_equal(
        make_border_ring(result, 120, is_fraction=False),
        ref_make_border_ring(expected, 120, is_fraction=False),
    )
