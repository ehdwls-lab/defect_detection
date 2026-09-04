"""Shared ArUco board detection contract used by production ROI code."""

from __future__ import annotations

from typing import Any

import numpy as np


REQUIRED_BOARD_MARKER_IDS = (0, 1, 2, 3)


def create_aruco_detector() -> Any:
    """Create the OpenCV 4x4/50 detector used by the existing ROI tools."""
    import cv2

    if not hasattr(cv2, "aruco"):
        raise RuntimeError("cv2.aruco is unavailable; opencv-contrib-python is required")
    aruco = cv2.aruco
    dictionary = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
    params = (
        aruco.DetectorParameters()
        if hasattr(aruco, "DetectorParameters")
        else aruco.DetectorParameters_create()
    )
    if hasattr(params, "cornerRefinementMethod"):
        params.cornerRefinementMethod = aruco.CORNER_REFINE_SUBPIX
    if hasattr(params, "adaptiveThreshWinSizeMin"):
        params.adaptiveThreshWinSizeMin = 3
    if hasattr(params, "adaptiveThreshWinSizeMax"):
        params.adaptiveThreshWinSizeMax = 35
    if hasattr(params, "adaptiveThreshWinSizeStep"):
        params.adaptiveThreshWinSizeStep = 4
    if hasattr(aruco, "ArucoDetector"):
        return aruco.ArucoDetector(dictionary, params)
    return aruco, dictionary, params


def detect_markers(image_bgr: np.ndarray, detector: Any) -> dict[int, np.ndarray]:
    """Return marker corners in OpenCV TL,TR,BR,BL order, keyed by ID."""
    import cv2

    image = np.asarray(image_bgr)
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("ArUco input must be a non-empty BGR image")
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if hasattr(detector, "detectMarkers"):
        corners, ids, _ = detector.detectMarkers(gray)
    else:
        aruco, dictionary, params = detector
        corners, ids, _ = aruco.detectMarkers(gray, dictionary, parameters=params)
    marker_map: dict[int, np.ndarray] = {}
    if ids is not None:
        for marker_id, corners_4 in zip(ids.flatten(), corners):
            marker_map[int(marker_id)] = (
                np.asarray(corners_4).reshape(4, 2).astype(np.float32)
            )
    return marker_map


def get_board_outer_quad(
    marker_map: dict[int, np.ndarray],
    required_ids: tuple[int, int, int, int] = REQUIRED_BOARD_MARKER_IDS,
) -> np.ndarray | None:
    """Build TL,TR,BR,BL board polygon from the four outer marker corners."""
    if not all(marker_id in marker_map for marker_id in required_ids):
        return None
    top_left, top_right, bottom_left, bottom_right = required_ids
    return np.asarray(
        [
            marker_map[top_left][0],
            marker_map[top_right][1],
            marker_map[bottom_right][2],
            marker_map[bottom_left][3],
        ],
        dtype=np.float32,
    )


def polygon_mask(shape: tuple[int, ...], polygon: np.ndarray) -> np.ndarray:
    import cv2

    if len(shape) < 2 or shape[0] <= 0 or shape[1] <= 0:
        raise ValueError("polygon mask requires a valid image shape")
    points = np.asarray(polygon, dtype=np.float32)
    if points.shape != (4, 2) or not np.all(np.isfinite(points)):
        raise ValueError("board polygon must be finite 4x2 corners")
    mask = np.zeros(shape[:2], dtype=np.uint8)
    cv2.fillConvexPoly(mask, np.rint(points).astype(np.int32), 255)
    return mask


def marker_local_board_sample_mask(
    shape: tuple[int, ...], marker_map: dict[int, np.ndarray],
    *, outer_margin_px: int, exclusion_margin_px: int,
) -> np.ndarray:
    """Return the union of local bands outside detected marker polygons."""
    import cv2

    sample = np.zeros(shape[:2], dtype=np.uint8)
    outer_margin = max(1, int(outer_margin_px))
    exclusion_margin = max(0, int(exclusion_margin_px))
    for corners in marker_map.values():
        marker = polygon_mask(shape, np.asarray(corners, dtype=np.float32))
        outer = cv2.dilate(
            marker,
            cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE, (2 * outer_margin + 1, 2 * outer_margin + 1),
            ),
        )
        excluded = cv2.dilate(
            marker,
            cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE,
                (2 * exclusion_margin + 1, 2 * exclusion_margin + 1),
            ),
        ) if exclusion_margin else marker
        sample = cv2.bitwise_or(sample, cv2.bitwise_and(outer, cv2.bitwise_not(excluded)))
    return sample


def draw_aruco_overlay(
    image_bgr: np.ndarray,
    marker_map: dict[int, np.ndarray],
    board_quad: np.ndarray | None,
) -> np.ndarray:
    import cv2

    overlay = np.asarray(image_bgr).copy()
    for marker_id, corners in marker_map.items():
        points = np.rint(corners).astype(np.int32)
        cv2.polylines(overlay, [points], True, (0, 255, 255), 2)
        center = np.rint(np.mean(corners, axis=0)).astype(np.int32)
        cv2.putText(
            overlay, f"ID {marker_id}", (int(center[0]), int(center[1])),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2, cv2.LINE_AA,
        )
    if board_quad is not None:
        cv2.polylines(
            overlay, [np.rint(board_quad).astype(np.int32)], True, (0, 255, 0), 3,
        )
    return overlay
