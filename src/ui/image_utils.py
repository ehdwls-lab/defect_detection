"""OpenCV/NumPy to Qt image conversion without importing hardware modules."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np


def load_bgr(path: str | Path | None) -> np.ndarray | None:
    if path is None or not Path(path).is_file():
        return None
    return cv2.imread(str(path), cv2.IMREAD_COLOR)


def depth_preview(path: str | Path | None) -> np.ndarray | None:
    if path is None or not Path(path).is_file():
        return None
    depth = np.asarray(np.load(path), dtype=np.float32)
    valid = np.isfinite(depth) & (depth > 0)
    if not np.any(valid):
        return np.zeros((*depth.shape[:2], 3), dtype=np.uint8)
    lo, hi = np.percentile(depth[valid], (2, 98))
    scaled = np.zeros(depth.shape[:2], dtype=np.uint8)
    scaled[valid] = np.clip((depth[valid] - lo) * 255 / max(hi - lo, 1e-6), 0, 255)
    colored = cv2.applyColorMap(255 - scaled, cv2.COLORMAP_TURBO)
    colored[~valid] = 0
    return colored


def roi_contour_overlay(rgb: np.ndarray | None, mask_path: str | Path | None) -> np.ndarray | None:
    if rgb is None:
        return None
    output = rgb.copy()
    if mask_path is None or not Path(mask_path).is_file():
        return output
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None or mask.shape != output.shape[:2]:
        return output
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(output, contours, -1, (0, 255, 0), 2)
    return output


def threshold_relative_heatmap(
    heatmap: np.ndarray | None, score: float | None, threshold: float | None,
) -> np.ndarray | None:
    """Visualization only: scale saved spatial intensity by score/threshold."""
    if heatmap is None:
        return None
    gray = cv2.cvtColor(heatmap, cv2.COLOR_BGR2GRAY) if heatmap.ndim == 3 else heatmap
    relative = 0.0 if score is None or not threshold or threshold <= 0 else score / threshold
    spatial = gray.astype(np.float32) / max(float(gray.max()), 1.0)
    normalized = np.clip(spatial * relative / 1.5, 0, 1)
    display = cv2.applyColorMap((normalized * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
    cv2.putText(display, "0.0     0.5     TH 1.0     HIGH 1.5+", (12, 25),
                cv2.FONT_HERSHEY_SIMPLEX, .55, (255, 255, 255), 1, cv2.LINE_AA)
    return display


def anomaly_localization_overlay(
    rgb: np.ndarray | None, heatmap: np.ndarray | None,
    score: float | None, threshold: float | None,
) -> np.ndarray | None:
    """Blend threshold-relative anomaly evidence onto RGB for HMI display only."""
    if rgb is None:
        return threshold_relative_heatmap(heatmap, score, threshold)
    colored = threshold_relative_heatmap(heatmap, score, threshold)
    if colored is None:
        return rgb.copy()
    colored = cv2.resize(colored, (rgb.shape[1], rgb.shape[0]), interpolation=cv2.INTER_LINEAR)
    gray = cv2.cvtColor(colored, cv2.COLOR_BGR2GRAY)
    alpha = (gray.astype(np.float32) / 255 * .58)[..., None]
    output = np.clip(rgb * (1 - alpha) + colored * alpha, 0, 255).astype(np.uint8)
    relative = 0.0 if score is None or not threshold or threshold <= 0 else score / threshold
    if relative >= 1.0:
        hot = np.uint8(gray >= np.percentile(gray, 92)) * 255
        contours, _ = cv2.findContours(hot, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            if cv2.contourArea(contour) >= 16:
                x, y, w, h = cv2.boundingRect(contour)
                cv2.rectangle(output, (x, y), (x + w, y + h), (0, 0, 255), 3)
    return output


def ndarray_to_qimage(image: np.ndarray) -> Any:
    from PySide6.QtGui import QImage

    array = np.ascontiguousarray(image)
    if array.ndim == 2:
        return QImage(array.data, array.shape[1], array.shape[0], array.strides[0],
                      QImage.Format_Grayscale8).copy()
    if array.ndim != 3 or array.shape[2] not in (3, 4):
        raise ValueError("expected HxW, HxWx3, or HxWx4 image")
    fmt = QImage.Format_BGR888 if array.shape[2] == 3 else QImage.Format_RGBA8888
    return QImage(array.data, array.shape[1], array.shape[0], array.strides[0], fmt).copy()
