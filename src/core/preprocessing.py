from __future__ import annotations

from typing import Optional

import cv2
import numpy as np

from src.config import InspectionConfig


def preprocess_surface_image(
    image_bgr: np.ndarray,
    config: Optional[InspectionConfig] = None,
) -> np.ndarray:
    """Apply the current shared surface-only preprocessing pipeline.

    Input and output are BGR uint8 arrays. This deliberately matches the
    verified behavior used by the surface-only prototype and training pipeline.
    """
    if image_bgr.ndim != 3 or image_bgr.shape[2] != 3:
        raise ValueError("image_bgr must have shape (height, width, 3)")

    cfg = config if config is not None else InspectionConfig.default()
    gamma = float(cfg.preprocessing.gamma)
    clahe_clip = float(cfg.preprocessing.clahe_clip)
    tile_grid_size = tuple(cfg.preprocessing.tile_grid_size)
    sigma = float(cfg.preprocessing.sigma)
    unsharp_alpha = float(cfg.preprocessing.unsharp_alpha)
    unsharp_beta = float(cfg.preprocessing.unsharp_beta)

    image_uint8 = np.asarray(image_bgr, dtype=np.uint8)
    lookup = np.array(
        [((index / 255.0) ** gamma) * 255.0 for index in range(256)],
        dtype=np.uint8,
    )
    gamma_corrected = cv2.LUT(image_uint8, lookup)

    lab = cv2.cvtColor(gamma_corrected, cv2.COLOR_BGR2LAB)
    clahe = cv2.createCLAHE(clipLimit=clahe_clip, tileGridSize=tile_grid_size)
    lab[:, :, 0] = clahe.apply(lab[:, :, 0])
    enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    blur = cv2.GaussianBlur(enhanced, (0, 0), sigmaX=sigma)
    return cv2.addWeighted(
        enhanced,
        unsharp_alpha,
        blur,
        unsharp_beta,
        0,
    )


preprocess_anomaly = preprocess_surface_image
