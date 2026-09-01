from __future__ import annotations

import cv2
import numpy as np


def generate_surface_patches(
    surface_mask: np.ndarray,
    patch_size: int = 64,
    stride: int = 32,
    min_coverage: float = 1.0,
) -> list[dict[str, int | float]]:
    """Create valid 64x64 surface patches where the entire patch lies inside the surface mask."""
    h, w = surface_mask.shape
    patch_size = int(patch_size)
    stride = int(stride)
    min_coverage = float(np.clip(min_coverage, 0.0, 1.0))

    patches: list[dict[str, int | float]] = []
    for y in range(0, h - patch_size + 1, stride):
        for x in range(0, w - patch_size + 1, stride):
            roi = surface_mask[y:y + patch_size, x:x + patch_size]
            coverage = float(np.mean(roi > 0))
            if coverage >= min_coverage:
                patches.append(
                    {
                        "x": int(x),
                        "y": int(y),
                        "w": int(patch_size),
                        "h": int(patch_size),
                        "coverage": coverage,
                    }
                )
    return patches


def extract_valid_surface_patches(
    image: np.ndarray,
    surface_mask: np.ndarray,
    patch_size: int = 64,
    stride: int = 32,
    min_coverage: float = 1.0,
) -> tuple[list[dict[str, int | float]], list[np.ndarray]]:
    """Return patch metadata and cropped patch images for valid surface-only patches."""
    patches = generate_surface_patches(surface_mask, patch_size, stride, min_coverage)
    patch_images: list[np.ndarray] = []
    for patch in patches:
        x = int(patch["x"])
        y = int(patch["y"])
        patch_images.append(image[y:y + patch_size, x:x + patch_size].copy())
    return patches, patch_images
