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


def patch_union_mask(
    image_shape: tuple[int, ...], patches: list[dict[str, int | float]],
) -> np.ndarray:
    """Return the union of accepted patch rectangles on the image grid."""
    union = np.zeros(tuple(image_shape[:2]), dtype=np.uint8)
    for patch in patches:
        x, y = int(patch["x"]), int(patch["y"])
        width, height = int(patch["w"]), int(patch["h"])
        union[y:y + height, x:x + width] = 255
    return union


def measure_patchability(
    inspection_mask: np.ndarray, patch_size: int, stride: int, coverage: float,
) -> tuple[list[dict[str, int | float]], np.ndarray, float]:
    patches = generate_surface_patches(inspection_mask, patch_size, stride, coverage)
    union = patch_union_mask(inspection_mask.shape, patches)
    area = int(np.count_nonzero(inspection_mask))
    return patches, union, float(np.count_nonzero(union) / area) if area else 0.0
