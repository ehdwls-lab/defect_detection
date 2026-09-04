from __future__ import annotations

import cv2
import numpy as np


def fallback_workspace_mask(shape: tuple[int, ...], margin_px: int) -> np.ndarray:
    h, w = shape[:2]
    margin = max(0, int(margin_px))
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.rectangle(mask, (margin, margin), (w - 1 - margin, h - 1 - margin), 255, -1)
    return mask


def make_border_ring(workspace_mask: np.ndarray, ring_value: float, *, is_fraction: bool = False) -> np.ndarray:
    ys, xs = np.where(workspace_mask > 0)
    if xs.size == 0:
        return np.zeros_like(workspace_mask)
    if is_fraction:
        minimum_dimension = min(int(xs.max() - xs.min() + 1), int(ys.max() - ys.min() + 1))
        width = int(round(float(ring_value) * minimum_dimension))
    else:
        width = int(ring_value)
    width = max(10, width)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2 * width + 1, 2 * width + 1))
    eroded = cv2.erode(workspace_mask, kernel, iterations=1)
    return cv2.bitwise_and(workspace_mask, cv2.bitwise_not(eroded))
