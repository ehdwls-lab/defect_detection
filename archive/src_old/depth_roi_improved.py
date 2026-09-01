from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class MaskResult:
    mask: np.ndarray
    floor_depth_mm: float
    mask_area: int
    bbox: tuple[int, int, int, int] | None
    invalid_ratio_in_bbox: float


def estimate_floor_depth(depth_mm: np.ndarray, border_ratio: float = 0.10) -> float:
    if depth_mm.ndim != 2:
        raise ValueError("depth_mm은 2차원 배열이어야 합니다.")
    if not 0.0 < border_ratio <= 0.5:
        raise ValueError("border_ratio는 0 초과 0.5 이하여야 합니다.")

    h, w = depth_mm.shape
    by = max(1, int(round(h * border_ratio)))
    bx = max(1, int(round(w * border_ratio)))

    border = np.zeros((h, w), dtype=bool)
    border[:by, :] = True
    border[-by:, :] = True
    border[:, :bx] = True
    border[:, -bx:] = True

    valid = np.isfinite(depth_mm) & (depth_mm > 0)
    values = depth_mm[border & valid]
    if values.size == 0:
        raise RuntimeError("바깥 영역에서 유효한 바닥 Depth를 찾지 못했습니다.")

    return float(np.median(values))


def _odd(value: int) -> int:
    if value <= 0:
        return 0
    return value if value % 2 else value + 1


def _fill_enclosed_holes(mask: np.ndarray) -> np.ndarray:
    inverse = cv2.bitwise_not(mask)
    count, labels = cv2.connectedComponents(inverse, connectivity=8)
    if count <= 1:
        return mask

    border_labels = set(np.unique(np.concatenate([
        labels[0, :], labels[-1, :], labels[:, 0], labels[:, -1]
    ])).tolist())

    holes = np.zeros_like(mask)
    for label in range(1, count):
        if label not in border_labels:
            holes[labels == label] = 255

    return cv2.bitwise_or(mask, holes)


def _keep_valid_components(mask: np.ndarray, min_object_area: int, largest_component: bool) -> np.ndarray:
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    valid = [
        i for i in range(1, count)
        if stats[i, cv2.CC_STAT_AREA] >= min_object_area
    ]

    if largest_component and valid:
        valid = [max(valid, key=lambda i: stats[i, cv2.CC_STAT_AREA])]

    out = np.zeros_like(mask)
    for label in valid:
        out[labels == label] = 255
    return out


def _fill_external_contours(mask: np.ndarray) -> np.ndarray:
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out = np.zeros_like(mask)
    if contours:
        cv2.drawContours(out, contours, -1, 255, thickness=cv2.FILLED)
    return out


def create_candidate_mask(
    depth_mm: np.ndarray,
    min_height_mm: float = 10.0,
    border_ratio: float = 0.10,
) -> tuple[np.ndarray, np.ndarray, float]:
    if min_height_mm <= 0:
        raise ValueError("min_height_mm는 0보다 커야 합니다.")

    floor_depth = estimate_floor_depth(depth_mm, border_ratio)
    valid = np.isfinite(depth_mm) & (depth_mm > 0)
    candidate = valid & (depth_mm < floor_depth - min_height_mm)

    return candidate.astype(np.uint8) * 255, valid, floor_depth


def clean_object_mask(
    candidate_mask: np.ndarray,
    min_object_area: int = 10000,
    largest_component: bool = True,
    morphology_size: int = 7,
    bridge_size: int = 21,
    mask_erode: int = 15,
    contour_fill: bool = True,
) -> np.ndarray:
    """반사로 생긴 Depth hole을 메우고 경계는 안쪽으로 줄인 AE용 mask."""
    if min_object_area < 1:
        raise ValueError("min_object_area는 1 이상이어야 합니다.")

    mask = (candidate_mask > 0).astype(np.uint8) * 255

    morph = _odd(morphology_size)
    if morph > 0:
        kernel = np.ones((morph, morph), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    bridge = _odd(bridge_size)
    if bridge > 0:
        kernel = np.ones((bridge, bridge), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    mask = _keep_valid_components(mask, min_object_area, largest_component)

    if contour_fill and np.any(mask):
        mask = _fill_external_contours(mask)

    mask = _fill_enclosed_holes(mask)
    mask = _keep_valid_components(mask, min_object_area, largest_component)

    erode = _odd(mask_erode)
    if erode > 0 and np.any(mask):
        kernel = np.ones((erode, erode), np.uint8)
        mask = cv2.erode(mask, kernel, iterations=1)

    return mask


def get_mask_bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    ys, xs = np.nonzero(mask)
    if xs.size == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def create_object_mask(
    depth_mm: np.ndarray,
    min_height_mm: float = 10.0,
    border_ratio: float = 0.10,
    min_object_area: int = 10000,
    largest_component: bool = True,
    morphology_size: int = 7,
    bridge_size: int = 21,
    mask_erode: int = 15,
    contour_fill: bool = True,
) -> MaskResult:
    candidate, valid, floor_depth = create_candidate_mask(
        depth_mm, min_height_mm, border_ratio
    )

    mask = clean_object_mask(
        candidate,
        min_object_area=min_object_area,
        largest_component=largest_component,
        morphology_size=morphology_size,
        bridge_size=bridge_size,
        mask_erode=mask_erode,
        contour_fill=contour_fill,
    )

    bbox = get_mask_bbox(mask)
    invalid_ratio = 0.0
    if bbox is not None:
        x1, y1, x2, y2 = bbox
        invalid_ratio = float(np.mean(~valid[y1:y2, x1:x2]))

    return MaskResult(
        mask=mask,
        floor_depth_mm=floor_depth,
        mask_area=int(np.count_nonzero(mask)),
        bbox=bbox,
        invalid_ratio_in_bbox=invalid_ratio,
    )


def create_object_mask_debug(
    depth_mm: np.ndarray,
    min_height_mm: float = 10.0,
    border_ratio: float = 0.10,
    min_object_area: int = 10000,
    largest_component: bool = True,
    morphology_size: int = 7,
    bridge_size: int = 21,
    mask_erode: int = 15,
    contour_fill: bool = True,
) -> tuple[MaskResult, dict[str, np.ndarray]]:
    candidate, valid, floor_depth = create_candidate_mask(
        depth_mm, min_height_mm, border_ratio
    )

    initial = candidate.copy()

    after_morph = initial.copy()
    morph = _odd(morphology_size)
    if morph > 0:
        kernel = np.ones((morph, morph), np.uint8)
        after_morph = cv2.morphologyEx(after_morph, cv2.MORPH_OPEN, kernel)
        after_morph = cv2.morphologyEx(after_morph, cv2.MORPH_CLOSE, kernel)

    after_bridge = after_morph.copy()
    bridge = _odd(bridge_size)
    if bridge > 0:
        kernel = np.ones((bridge, bridge), np.uint8)
        after_bridge = cv2.morphologyEx(after_bridge, cv2.MORPH_CLOSE, kernel)

    component = _keep_valid_components(
        after_bridge, min_object_area, largest_component
    )

    support = component.copy()
    if contour_fill and np.any(support):
        support = _fill_external_contours(support)
    support = _fill_enclosed_holes(support)

    final_mask = support.copy()
    erode = _odd(mask_erode)
    if erode > 0 and np.any(final_mask):
        kernel = np.ones((erode, erode), np.uint8)
        final_mask = cv2.erode(final_mask, kernel, iterations=1)

    bbox = get_mask_bbox(final_mask)
    invalid_ratio = 0.0
    if bbox is not None:
        x1, y1, x2, y2 = bbox
        invalid_ratio = float(np.mean(~valid[y1:y2, x1:x2]))

    result = MaskResult(
        mask=final_mask,
        floor_depth_mm=floor_depth,
        mask_area=int(np.count_nonzero(final_mask)),
        bbox=bbox,
        invalid_ratio_in_bbox=invalid_ratio,
    )

    debug = {
        "valid_depth_mask": valid.astype(np.uint8) * 255,
        "candidate_before_cleanup": initial,
        "mask_after_morphology": after_morph,
        "mask_after_bridge": after_bridge,
        "support_before_erode": support,
        "final_object_mask": final_mask,
    }
    return result, debug


def calculate_patch_coverage(mask: np.ndarray, positions: list[tuple[int, int]], patch_size: int) -> np.ndarray:
    binary = mask > 0
    return np.asarray([
        float(np.mean(binary[y:y + patch_size, x:x + patch_size]))
        for x, y in positions
    ], dtype=np.float32)


def make_depth_visualization(depth_mm: np.ndarray) -> np.ndarray:
    valid = np.isfinite(depth_mm) & (depth_mm > 0)
    output = np.zeros((*depth_mm.shape, 3), dtype=np.uint8)
    if not np.any(valid):
        return output

    low, high = np.percentile(depth_mm[valid], [2.0, 98.0])
    if high <= low:
        high = low + 1.0

    normalized = np.zeros(depth_mm.shape, dtype=np.uint8)
    normalized[valid] = (
        np.clip((depth_mm[valid] - low) / (high - low), 0, 1) * 255
    ).astype(np.uint8)

    colored = cv2.applyColorMap(normalized, cv2.COLORMAP_TURBO)
    colored[~valid] = 0
    return colored


def make_mask_views(
    color_bgr: np.ndarray,
    mask: np.ndarray,
    bbox: tuple[int, int, int, int] | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if mask.shape[:2] != color_bgr.shape[:2]:
        mask = cv2.resize(
            mask,
            (color_bgr.shape[1], color_bgr.shape[0]),
            interpolation=cv2.INTER_NEAREST,
        )

    overlay = color_bgr.copy()
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlay, contours, -1, (0, 255, 0), 2)

    bbox_preview = overlay.copy()
    if bbox is not None:
        x1, y1, x2, y2 = bbox
        cv2.rectangle(bbox_preview, (x1, y1), (x2 - 1, y2 - 1), (0, 255, 255), 2)

    masked = np.zeros_like(color_bgr)
    masked[mask > 0] = color_bgr[mask > 0]
    return overlay, bbox_preview, masked
