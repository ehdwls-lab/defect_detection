from __future__ import annotations

import cv2
import numpy as np


def odd(value: int) -> int:
    value = max(1, int(value))
    return value if value % 2 == 1 else value + 1


def fit_inverse_depth_plane_ransac(depth_mm: np.ndarray, sample_mask: np.ndarray, config) -> tuple[np.ndarray | None, float, float]:
    """Fit a board plane using the same inverse-depth RANSAC logic as the surface-only prototype."""
    depth_min_mm = float(getattr(config, "depth_min_mm", getattr(config, "min_mm", 80.0)))
    depth_max_mm = float(getattr(config, "depth_max_mm", getattr(config, "max_mm", 2000.0)))
    plane_ransac_mm = float(getattr(config, "plane_ransac_mm", 2.5))
    plane_ransac_iters = int(getattr(config, "plane_ransac_iters", 160))
    plane_min_points = int(getattr(config, "plane_min_points", 500))
    plane_max_points = int(getattr(config, "plane_max_points", 7000))

    valid = (
        (sample_mask > 0)
        & (depth_mm >= depth_min_mm)
        & (depth_mm <= depth_max_mm)
    )

    ys, xs = np.where(valid)
    if len(xs) < plane_min_points:
        return None, 0.0, float("inf")

    rng = np.random.default_rng(42)
    if len(xs) > plane_max_points:
        idx = rng.choice(len(xs), size=plane_max_points, replace=False)
        xs = xs[idx]
        ys = ys[idx]

    z = depth_mm[ys, xs].astype(np.float64)
    h, w = depth_mm.shape
    xn = (xs.astype(np.float64) / max(1.0, w - 1.0) * 2.0 - 1.0)
    yn = (ys.astype(np.float64) / max(1.0, h - 1.0) * 2.0 - 1.0)
    A = np.column_stack((xn, yn, np.ones_like(xn)))
    invz = 1.0 / np.maximum(z, 1e-6)

    best_inliers = None
    best_count = 0
    for _ in range(max(20, int(plane_ransac_iters))):
        sample_idx = rng.choice(len(z), size=3, replace=False)
        try:
            coeff = np.linalg.solve(A[sample_idx], invz[sample_idx])
        except np.linalg.LinAlgError:
            continue

        pred_inv = A @ coeff
        pred_z = np.where(pred_inv > 1e-9, 1.0 / pred_inv, np.inf)
        residual = np.abs(pred_z - z)
        inliers = residual <= plane_ransac_mm
        count = int(np.count_nonzero(inliers))
        if count > best_count:
            best_count = count
            best_inliers = inliers

    if best_inliers is None or best_count < plane_min_points:
        return None, 0.0, float("inf")

    coeff, *_ = np.linalg.lstsq(A[best_inliers], invz[best_inliers], rcond=None)

    yy, xx = np.mgrid[0:h, 0:w]
    x_full = (xx.astype(np.float64) / max(1.0, w - 1.0) * 2.0 - 1.0)
    y_full = (yy.astype(np.float64) / max(1.0, h - 1.0) * 2.0 - 1.0)
    pred_inv_full = coeff[0] * x_full + coeff[1] * y_full + coeff[2]
    plane_depth = np.zeros((h, w), dtype=np.float32)
    okay = pred_inv_full > 1e-9
    plane_depth[okay] = (1.0 / pred_inv_full[okay]).astype(np.float32)

    sample_pred_inv = A @ coeff
    sample_pred_z = np.where(sample_pred_inv > 1e-9, 1.0 / sample_pred_inv, np.inf)
    residual = np.abs(sample_pred_z - z)
    final_inliers = np.isfinite(residual) & (residual <= plane_ransac_mm)
    inlier_ratio = float(np.mean(final_inliers)) if final_inliers.size else 0.0
    inlier_residual = float(np.median(residual[final_inliers])) if np.any(final_inliers) else float("inf")
    return plane_depth, inlier_ratio, inlier_residual


def depth_object_candidate(depth_mm: np.ndarray, plane_depth_mm: np.ndarray, workspace_mask: np.ndarray, config) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate the geometry-based candidate object mask from plane-relative height."""
    depth_min_mm = float(getattr(config, "depth_min_mm", getattr(config, "min_mm", 80.0)))
    depth_max_mm = float(getattr(config, "depth_max_mm", getattr(config, "max_mm", 2000.0)))
    height_threshold_mm = float(getattr(config, "height_threshold_mm", 4.0))
    max_object_height_mm = float(getattr(config, "max_object_height_mm", 250.0))
    object_open_size = int(getattr(config, "object_open_size", 5))
    object_close_size = int(getattr(config, "object_close_size", 21))
    object_close_iterations = int(getattr(config, "object_close_iterations", 2))

    valid = (
        (workspace_mask > 0)
        & (depth_mm >= depth_min_mm)
        & (depth_mm <= depth_max_mm)
        & (plane_depth_mm > 0)
    )

    height = np.zeros_like(depth_mm, dtype=np.float32)
    height[valid] = plane_depth_mm[valid] - depth_mm[valid]
    candidate = valid & (height >= height_threshold_mm) & (height <= max_object_height_mm)

    mask = np.where(candidate, 255, 0).astype(np.uint8)
    k_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (odd(object_open_size), odd(object_open_size)))
    k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (odd(object_close_size), odd(object_close_size)))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k_open, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k_close, iterations=max(1, object_close_iterations))
    return mask, height, valid


def select_final_object_mask(candidate: np.ndarray, workspace_mask: np.ndarray, config) -> np.ndarray | None:
    """Keep the largest central object component and discard background/board-like structures."""
    min_object_area = int(getattr(config, "min_object_area", 10000))
    max_object_area_ratio = float(getattr(config, "max_object_area_ratio", 0.75))

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(candidate, 8)
    if num_labels <= 1:
        return None

    ys, xs = np.where(workspace_mask > 0)
    if xs.size == 0:
        return None

    center = np.asarray([float(np.mean(xs)), float(np.mean(ys))], dtype=np.float32)
    workspace_area = max(1, int(np.count_nonzero(workspace_mask)))

    candidates: list[tuple[float, int]] = []
    for label in range(1, num_labels):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area < min_object_area:
            continue
        if area / workspace_area > max_object_area_ratio:
            continue

        c = centroids[label]
        dist = float(np.linalg.norm(c - center))
        score = float(area) - 20.0 * dist
        candidates.append((score, label))

    if not candidates:
        return None

    candidates.sort(reverse=True)
    label = candidates[0][1]
    component = np.where(labels == label, 255, 0).astype(np.uint8)
    contours, _ = cv2.findContours(component, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    contour = max(contours, key=cv2.contourArea)
    final_mask = np.zeros_like(component)
    cv2.drawContours(final_mask, [contour], -1, 255, cv2.FILLED)
    return final_mask
