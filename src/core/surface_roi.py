from __future__ import annotations

import cv2
import numpy as np


def erode_surface_mask(object_mask: np.ndarray, margin_px: int) -> np.ndarray:
    """Shrink the object mask by a fixed boundary margin to exclude the object edge."""
    margin_px = max(0, int(margin_px))
    if margin_px == 0:
        return object_mask.copy()

    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (2 * margin_px + 1, 2 * margin_px + 1),
    )
    return cv2.erode(object_mask, kernel, iterations=1)


def mask_touches_frame_edge(mask: np.ndarray | None, margin_px: int) -> bool:
    if mask is None:
        return True

    ys, xs = np.where(mask > 0)
    if xs.size == 0:
        return True

    h, w = mask.shape
    m = max(0, int(margin_px))
    return (
        int(xs.min()) <= m
        or int(ys.min()) <= m
        or int(xs.max()) >= w - 1 - m
        or int(ys.max()) >= h - 1 - m
    )
