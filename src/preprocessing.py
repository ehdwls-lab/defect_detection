from __future__ import annotations

import cv2
import numpy as np


def preprocess_anomaly(
    image: np.ndarray,
    gamma: float = 0.82,
    clahe_clip: float = 1.5,
    unsharp_amount: float = 0.30,
) -> np.ndarray:
    """Apply the shared BGR preprocessing used by training and inference."""
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("image must have shape (height, width, 3)")
    if gamma <= 0 or clahe_clip <= 0 or unsharp_amount < 0:
        raise ValueError("gamma and clahe_clip must be positive; unsharp_amount cannot be negative")

    image_uint8 = np.asarray(image, dtype=np.uint8)
    lookup = np.array(
        [((index / 255.0) ** gamma) * 255.0 for index in range(256)],
        dtype=np.uint8,
    )
    gamma_corrected = cv2.LUT(image_uint8, lookup)

    lab = cv2.cvtColor(gamma_corrected, cv2.COLOR_BGR2LAB)
    clahe = cv2.createCLAHE(clipLimit=clahe_clip, tileGridSize=(8, 8))
    lab[:, :, 0] = clahe.apply(lab[:, :, 0])
    enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    blur = cv2.GaussianBlur(enhanced, (0, 0), sigmaX=1.0)
    return cv2.addWeighted(
        enhanced,
        1.0 + unsharp_amount,
        blur,
        -unsharp_amount,
        0,
    )