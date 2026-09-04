"""Build the RGB anomaly ROI from frozen LED-OFF depth geometry."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.config import InspectionConfig


@dataclass(frozen=True)
class InspectionMaskResult:
    mask: np.ndarray
    inspection_area_px: int
    inspection_to_surface_ratio: float | None
    inspection_to_object_ratio: float | None


def _binary(mask: np.ndarray | None, *, name: str) -> np.ndarray:
    if mask is None:
        raise ValueError(f"{name} is missing")
    value = np.asarray(mask)
    if value.ndim != 2 or value.size == 0:
        raise ValueError(f"{name} must be a non-empty 2D mask")
    return np.where(value > 0, 255, 0).astype(np.uint8)


def _largest_component(mask: np.ndarray) -> np.ndarray:
    import cv2

    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if count <= 1:
        return np.zeros_like(mask)
    label = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return np.where(labels == label, 255, 0).astype(np.uint8)


def _fill_enclosed_holes(mask: np.ndarray) -> np.ndarray:
    """Fill only background components that cannot reach the image boundary."""
    import cv2

    background = np.where(mask == 0, 255, 0).astype(np.uint8)
    count, labels, _, _ = cv2.connectedComponentsWithStats(background, connectivity=8)
    exterior_labels = set(np.unique(np.concatenate((
        labels[0, :], labels[-1, :], labels[:, 0], labels[:, -1],
    ))).tolist())
    holes = np.zeros_like(mask)
    for label in range(1, count):
        if label not in exterior_labels:
            holes[labels == label] = 255
    return np.maximum(mask, holes)


def build_inspection_mask(
    object_mask: np.ndarray | None,
    surface_mask: np.ndarray | None,
    config: InspectionConfig,
) -> InspectionMaskResult:
    """Create a conservative continuous anomaly ROI without filling background."""
    import cv2

    object_binary = _binary(object_mask, name="object_mask")
    surface_binary = _binary(surface_mask, name="surface_mask")
    if object_binary.shape != surface_binary.shape:
        raise ValueError(
            "object_mask and surface_mask shapes differ: "
            f"{object_binary.shape} != {surface_binary.shape}"
        )
    object_area = int(np.count_nonzero(object_binary))
    surface_area = int(np.count_nonzero(surface_binary))
    if object_area == 0 or surface_area == 0:
        raise ValueError("object_mask and surface_mask must both be non-empty")
    if object_area == object_binary.size:
        raise ValueError("full-frame object mask is not a valid inspection ROI")

    main = _largest_component(np.maximum(object_binary, surface_binary))
    if not np.any(main):
        raise ValueError("main object component was not found")
    filled = _fill_enclosed_holes(main)
    close_size = int(config.surface_roi.inspection_close_size_px)
    close_iterations = int(config.surface_roi.inspection_close_iterations)
    if close_size < 1 or close_size % 2 == 0:
        raise ValueError("inspection_close_size_px must be a positive odd integer")
    if close_iterations < 0:
        raise ValueError("inspection_close_iterations must be non-negative")
    if close_size > 1 and close_iterations > 0:
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (close_size, close_size),
        )
        filled = cv2.morphologyEx(
            filled, cv2.MORPH_CLOSE, kernel, iterations=close_iterations,
        )
    filled = _largest_component(filled)
    margin = int(config.surface_roi.boundary_margin_px)
    if margin < 0:
        raise ValueError("boundary_margin_px must be non-negative")
    if margin > 0:
        size = margin * 2 + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
        filled = cv2.erode(filled, kernel, iterations=1)
    area = int(np.count_nonzero(filled))
    if area == 0:
        raise ValueError("inspection mask became empty after boundary erosion")
    if area == filled.size:
        raise ValueError("full-frame inspection mask is forbidden")
    return InspectionMaskResult(
        mask=filled,
        inspection_area_px=area,
        inspection_to_surface_ratio=(float(area / surface_area) if surface_area else None),
        inspection_to_object_ratio=(float(area / object_area) if object_area else None),
    )
