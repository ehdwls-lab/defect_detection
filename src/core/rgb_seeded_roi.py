from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from src.core.depth_processing import fill_external_object_contour


@dataclass(frozen=True)
class RGBSeededROIResult:
    object_mask: np.ndarray
    board_reference_mask: np.ndarray
    foreground_mask: np.ndarray
    selected_component_mask: np.ndarray
    board_median_lab: tuple[float, float, float]
    selected_component_area_px: int
    seed_overlap_px: int


def build_rgb_seeded_roi(
    color_bgr: np.ndarray,
    workspace_mask: np.ndarray,
    board_reference_mask: np.ndarray,
    depth_seed_mask: np.ndarray,
    config,
) -> RGBSeededROIResult | None:
    """Recover a silhouette from RGB appearance, constrained by a Depth seed."""
    color = np.asarray(color_bgr)
    workspace = np.where(np.asarray(workspace_mask) > 0, 255, 0).astype(np.uint8)
    board = np.where(np.asarray(board_reference_mask) > 0, 255, 0).astype(np.uint8)
    seed = np.where(np.asarray(depth_seed_mask) > 0, 255, 0).astype(np.uint8)
    if color.ndim != 3 or color.shape[2] != 3:
        raise ValueError("RGB fallback requires a BGR image")
    if not (workspace.shape == board.shape == seed.shape == color.shape[:2]):
        raise ValueError("RGB fallback masks and image must be aligned")
    board_pixels = board > 0
    if not np.any(board_pixels) or not np.any(seed):
        return None
    lab = cv2.cvtColor(color, cv2.COLOR_BGR2LAB).astype(np.float32)
    reference = np.median(lab[board_pixels], axis=0)
    distance = np.linalg.norm(lab - reference, axis=2)
    foreground = np.where(
        (workspace > 0) & (distance >= float(getattr(config, "rgb_fallback_lab_distance", 35.0))),
        255, 0,
    ).astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(foreground, 8)
    if count <= 1:
        return None
    workspace_area = max(1, int(np.count_nonzero(workspace)))
    seed_pixels = seed > 0
    candidates: list[tuple[int, int, int]] = []
    for label in range(1, count):
        component = labels == label
        area = int(stats[label, cv2.CC_STAT_AREA])
        overlap = int(np.count_nonzero(component & seed_pixels))
        if overlap == 0 or area / workspace_area > float(getattr(config, "max_object_area_ratio", 0.75)):
            continue
        candidates.append((overlap, area, label))
    if not candidates:
        return None
    overlap, area, label = max(candidates, key=lambda item: (item[0], item[1]))
    selected = np.where(labels == label, 255, 0).astype(np.uint8)
    selected = cv2.bitwise_and(selected, workspace)
    object_mask = cv2.bitwise_or(seed, selected)
    filled = fill_external_object_contour(object_mask, workspace)
    if filled is None or not np.any(filled):
        return None
    return RGBSeededROIResult(
        object_mask=filled,
        board_reference_mask=board,
        foreground_mask=foreground,
        selected_component_mask=selected,
        board_median_lab=tuple(float(value) for value in reference),
        selected_component_area_px=area,
        seed_overlap_px=overlap,
    )