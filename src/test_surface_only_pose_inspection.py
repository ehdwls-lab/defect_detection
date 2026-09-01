from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from pyorbbecsdk import (
    AlignFilter,
    Config,
    OBFormat,
    OBFrameAggregateOutputMode,
    OBPropertyID,
    OBSensorType,
    OBStreamType,
    Pipeline,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]

WIDTH = 1280
HEIGHT = 800

COLOR_FPS = 10
COLOR_FORMAT = OBFormat.MJPG

DEPTH_PREFERRED_FPS = (10, 15)
DEPTH_FORMAT = OBFormat.Y16

FRAME_TIMEOUT_MS = 3000
REQUIRED_IDS = (0, 1, 2, 3)


# =============================================================================
# Arguments
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Surface-only inspection prototype. "
            "현재 자세의 RGB+Depth에서 물체 외곽을 구하고, "
            "경계/바닥을 제외한 내부 표면의 64x64 patch만 검사 대상으로 생성한다."
        )
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "results" / "surface_only_pose_test",
    )

    # RGB camera
    parser.add_argument("--brightness", type=int, default=48)
    parser.add_argument("--exposure", type=int, default=1100)
    parser.add_argument("--gain", type=int, default=64)
    parser.add_argument("--white-balance", type=int, default=4600)
    parser.add_argument("--warmup-frames", type=int, default=40)

    # Depth
    parser.add_argument("--depth-exposure", type=int, default=3000)
    parser.add_argument("--depth-gain", type=int, default=16)
    parser.add_argument("--depth-median-frames", type=int, default=5)
    parser.add_argument("--depth-min-mm", type=float, default=80.0)
    parser.add_argument("--depth-max-mm", type=float, default=2000.0)

    parser.add_argument(
        "--depth-height-threshold-mm",
        type=float,
        default=4.0,
        help="판 평면보다 카메라 쪽으로 이 값 이상 떠 있으면 물체 후보",
    )
    parser.add_argument(
        "--depth-max-object-height-mm",
        type=float,
        default=250.0,
    )

    # Plane RANSAC
    parser.add_argument("--plane-ransac-mm", type=float, default=2.5)
    parser.add_argument("--plane-ransac-iters", type=int, default=160)
    parser.add_argument("--plane-max-points", type=int, default=7000)
    parser.add_argument("--plane-min-points", type=int, default=500)

    # Board plane sampling
    parser.add_argument(
        "--board-ring-fraction",
        type=float,
        default=0.18,
        help="ArUco board polygon 외곽 ring에서 plane sample을 취할 폭 비율",
    )
    parser.add_argument(
        "--fallback-margin",
        type=int,
        default=80,
        help="ArUco 4개가 안 보일 때 사용할 화면 중앙 workspace margin",
    )
    parser.add_argument(
        "--fallback-ring-px",
        type=int,
        default=120,
        help="fallback workspace의 바깥쪽 ring 폭",
    )

    # Object mask cleanup
    parser.add_argument("--object-open-size", type=int, default=5)
    parser.add_argument("--object-close-size", type=int, default=21)
    parser.add_argument("--object-close-iterations", type=int, default=2)
    parser.add_argument("--min-object-area", type=int, default=10000)
    parser.add_argument("--max-object-area-ratio", type=float, default=0.75)

    # Surface-only inspection
    parser.add_argument("--patch-size", type=int, default=64)
    parser.add_argument("--patch-stride", type=int, default=32)
    parser.add_argument(
        "--boundary-margin-px",
        type=int,
        default=10,
        help=(
            "물체 외곽에서 이만큼 안쪽으로 먼저 줄인 뒤 patch를 생성한다. "
            "바닥/경계 patch를 학습·검사에서 제거하는 핵심 값."
        ),
    )
    parser.add_argument(
        "--patch-mask-coverage",
        type=float,
        default=1.0,
        help="patch 내부에서 surface mask가 차지해야 하는 최소 비율. 기본 1.0=100%%",
    )

    # Confidence gates
    parser.add_argument(
        "--min-depth-valid-ratio",
        type=float,
        default=0.25,
        help="workspace 내 Depth valid 비율이 이 값 미만이면 검사 중지",
    )
    parser.add_argument(
        "--min-plane-inlier-ratio",
        type=float,
        default=0.25,
        help="plane RANSAC inlier 비율이 이 값 미만이면 검사 중지",
    )
    parser.add_argument(
        "--max-plane-inlier-residual-mm",
        type=float,
        default=2.0,
        help="RANSAC inlier residual median이 이 값보다 크면 검사 중지",
    )

    parser.add_argument(
        "--save-patches",
        action="store_true",
        help="SPACE 저장 시 유효한 RGB 64x64 patch 이미지도 저장",
    )
    parser.add_argument(
        "--pose-name",
        type=str,
        default="pose",
        help="저장 폴더에 들어갈 현재 검사 자세 이름",
    )

    # ------------------------------------------------------------------
    # Manual-Z inspection readiness
    # 3D/PLY가 Pitch/Roll을 정한 뒤, 사용자는 Z만 조절한다.
    # ------------------------------------------------------------------
    parser.add_argument(
        "--min-valid-patches",
        type=int,
        default=20,
        help="정밀검사를 시작하기 위한 최소 surface-only 64x64 patch 수",
    )
    parser.add_argument(
        "--ready-streak-frames",
        type=int,
        default=8,
        help="READY 조건이 연속으로 유지되어야 하는 프레임 수",
    )
    parser.add_argument(
        "--fov-edge-margin-px",
        type=int,
        default=18,
        help="물체 mask가 영상 가장자리에 이 거리 이내로 접근하면 READY 금지",
    )

    return parser.parse_args()


# =============================================================================
# Basic utilities
# =============================================================================

def odd(value: int) -> int:
    value = max(1, int(value))
    return value if value % 2 == 1 else value + 1


def draw_status(
    image: np.ndarray,
    lines: list[str],
    good: bool,
):
    line_h = 29
    box_h = 12 + line_h * len(lines)

    overlay = image.copy()

    cv2.rectangle(
        overlay,
        (0, 0),
        (
            min(
                image.shape[1] - 1,
                1090,
            ),
            box_h,
        ),
        (0, 0, 0),
        -1,
    )

    cv2.addWeighted(
        overlay,
        0.86,
        image,
        0.14,
        0,
        image,
    )

    state_color = (
        (0, 255, 0)
        if good
        else (0, 0, 255)
    )

    for i, line in enumerate(lines):
        color = (
            state_color
            if i == 0
            else (255, 255, 255)
        )

        cv2.putText(
            image,
            line,
            (
                12,
                25 + i * line_h,
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.67,
            color,
            2,
            cv2.LINE_AA,
        )


# =============================================================================
# Camera
# =============================================================================

def find_color_profile(pipeline: Pipeline):
    profiles = pipeline.get_stream_profile_list(
        OBSensorType.COLOR_SENSOR
    )

    fallback = None

    for i in range(profiles.get_count()):
        p = profiles.get_stream_profile_by_index(i)

        if p.get_format() != COLOR_FORMAT:
            continue

        if fallback is None:
            fallback = p

        if (
            p.get_width() == WIDTH
            and p.get_height() == HEIGHT
            and p.get_fps() == COLOR_FPS
        ):
            return p

    if fallback is not None:
        print(
            "[WARN] exact Color profile 미검출 -> "
            f"{fallback.get_width()}x{fallback.get_height()} @"
            f"{fallback.get_fps()} 사용"
        )
        return fallback

    raise RuntimeError("MJPG Color profile을 찾지 못했습니다.")


def find_depth_profile(pipeline: Pipeline):
    profiles = pipeline.get_stream_profile_list(
        OBSensorType.DEPTH_SENSOR
    )

    fallback = None

    for preferred_fps in DEPTH_PREFERRED_FPS:
        for i in range(profiles.get_count()):
            p = profiles.get_stream_profile_by_index(i)

            if p.get_format() != DEPTH_FORMAT:
                continue

            if fallback is None:
                fallback = p

            if (
                p.get_width() == WIDTH
                and p.get_height() == HEIGHT
                and p.get_fps() == preferred_fps
            ):
                return p

    if fallback is not None:
        print(
            "[WARN] exact Depth profile 미검출 -> "
            f"{fallback.get_width()}x{fallback.get_height()} @"
            f"{fallback.get_fps()} 사용"
        )
        return fallback

    raise RuntimeError("Y16 Depth profile을 찾지 못했습니다.")


def frame_to_bgr(color_frame):
    raw = np.frombuffer(
        color_frame.get_data(),
        dtype=np.uint8,
    )

    image = cv2.imdecode(
        raw,
        cv2.IMREAD_COLOR,
    )

    if image is None:
        raise RuntimeError("MJPG decode 실패")

    return image


def depth_frame_to_mm(depth_frame):
    h = depth_frame.get_height()
    w = depth_frame.get_width()

    raw = np.frombuffer(
        depth_frame.get_data(),
        dtype=np.uint16,
    ).reshape(
        h,
        w,
    )

    scale = float(
        depth_frame.get_depth_scale()
    )

    return (
        raw.astype(np.float32)
        * scale
    )


def wait_for_aligned_pair(
    pipeline: Pipeline,
    align_filter: AlignFilter,
):
    while True:
        frames = pipeline.wait_for_frames(
            FRAME_TIMEOUT_MS
        )

        if frames is None:
            continue

        aligned = align_filter.process(
            frames
        )

        if aligned is None:
            continue

        if hasattr(
            aligned,
            "as_frame_set",
        ):
            aligned = aligned.as_frame_set()

        color_frame = aligned.get_color_frame()
        depth_frame = aligned.get_depth_frame()

        if (
            color_frame is None
            or depth_frame is None
        ):
            continue

        color = frame_to_bgr(
            color_frame
        )

        depth = depth_frame_to_mm(
            depth_frame
        )

        if depth.shape[:2] != color.shape[:2]:
            depth = cv2.resize(
                depth,
                (
                    color.shape[1],
                    color.shape[0],
                ),
                interpolation=cv2.INTER_NEAREST,
            )

        return color, depth


def set_int_property(
    device,
    prop,
    value,
    label,
):
    try:
        device.set_int_property(
            prop,
            int(value),
        )
        print(
            f"{label}: {value}"
        )
    except Exception as exc:
        print(
            f"[WARN] {label} 설정 실패: {exc}"
        )


def set_bool_property(
    device,
    prop,
    value,
    label,
):
    try:
        device.set_bool_property(
            prop,
            bool(value),
        )
        print(
            f"{label}: {value}"
        )
    except Exception as exc:
        print(
            f"[WARN] {label} 설정 실패: {exc}"
        )


def configure_camera(
    device,
    args,
):
    print("=" * 72)
    print("Manual camera settings")

    set_bool_property(
        device,
        OBPropertyID.OB_PROP_COLOR_AUTO_EXPOSURE_BOOL,
        False,
        "Color AE",
    )
    set_bool_property(
        device,
        OBPropertyID.OB_PROP_COLOR_AUTO_WHITE_BALANCE_BOOL,
        False,
        "Color AWB",
    )

    set_int_property(
        device,
        OBPropertyID.OB_PROP_COLOR_BRIGHTNESS_INT,
        args.brightness,
        "Brightness",
    )
    set_int_property(
        device,
        OBPropertyID.OB_PROP_COLOR_EXPOSURE_INT,
        args.exposure,
        "Color Exposure",
    )
    set_int_property(
        device,
        OBPropertyID.OB_PROP_COLOR_GAIN_INT,
        args.gain,
        "Color Gain",
    )
    set_int_property(
        device,
        OBPropertyID.OB_PROP_COLOR_WHITE_BALANCE_INT,
        args.white_balance,
        "White Balance",
    )

    set_bool_property(
        device,
        OBPropertyID.OB_PROP_DEPTH_AUTO_EXPOSURE_BOOL,
        False,
        "Depth AE",
    )
    set_int_property(
        device,
        OBPropertyID.OB_PROP_DEPTH_EXPOSURE_INT,
        args.depth_exposure,
        "Depth Exposure",
    )
    set_int_property(
        device,
        OBPropertyID.OB_PROP_DEPTH_GAIN_INT,
        args.depth_gain,
        "Depth Gain",
    )

    print("=" * 72)


def temporal_median_depth(
    frames: list[np.ndarray],
):
    if not frames:
        return None

    stack = np.stack(
        frames,
        axis=0,
    ).astype(np.float32)

    safe = np.where(
        stack > 0,
        stack,
        np.nan,
    )

    with np.errstate(
        invalid="ignore",
    ):
        med = np.nanmedian(
            safe,
            axis=0,
        )

    return np.where(
        np.isfinite(med),
        med,
        0.0,
    ).astype(np.float32)


# =============================================================================
# ArUco / workspace
# =============================================================================

def create_aruco_detector():
    if not hasattr(cv2, "aruco"):
        raise RuntimeError(
            "cv2.aruco가 없습니다. opencv-contrib-python 필요."
        )

    aruco = cv2.aruco

    dictionary = aruco.getPredefinedDictionary(
        aruco.DICT_4X4_50
    )

    if hasattr(
        aruco,
        "DetectorParameters",
    ):
        params = aruco.DetectorParameters()
    else:
        params = aruco.DetectorParameters_create()

    if hasattr(
        params,
        "cornerRefinementMethod",
    ):
        params.cornerRefinementMethod = (
            aruco.CORNER_REFINE_SUBPIX
        )

    if hasattr(
        aruco,
        "ArucoDetector",
    ):
        return aruco.ArucoDetector(
            dictionary,
            params,
        )

    return (
        aruco,
        dictionary,
        params,
    )


def detect_markers(
    image_bgr,
    detector,
):
    gray = cv2.cvtColor(
        image_bgr,
        cv2.COLOR_BGR2GRAY,
    )

    if hasattr(
        detector,
        "detectMarkers",
    ):
        corners, ids, _ = detector.detectMarkers(
            gray
        )
    else:
        aruco, dictionary, params = detector

        corners, ids, _ = aruco.detectMarkers(
            gray,
            dictionary,
            parameters=params,
        )

    marker_map = {}

    if ids is not None:
        for marker_id, corners_4 in zip(
            ids.flatten(),
            corners,
        ):
            marker_map[
                int(marker_id)
            ] = (
                corners_4
                .reshape(4, 2)
                .astype(np.float32)
            )

    return marker_map


def get_board_outer_quad(
    marker_map,
):
    if not all(
        marker_id in marker_map
        for marker_id in REQUIRED_IDS
    ):
        return None

    # marker corner: TL,TR,BR,BL
    tl = marker_map[0][0]
    tr = marker_map[1][1]
    bl = marker_map[2][3]
    br = marker_map[3][2]

    return np.asarray(
        [
            tl,
            tr,
            br,
            bl,
        ],
        dtype=np.float32,
    )


def polygon_mask(
    shape,
    polygon,
):
    mask = np.zeros(
        shape[:2],
        dtype=np.uint8,
    )

    cv2.fillPoly(
        mask,
        [
            polygon.astype(
                np.int32
            )
        ],
        255,
    )

    return mask


def fallback_workspace_mask(
    shape,
    margin: int,
):
    h, w = shape[:2]

    m = max(
        0,
        int(margin),
    )

    mask = np.zeros(
        (
            h,
            w,
        ),
        dtype=np.uint8,
    )

    cv2.rectangle(
        mask,
        (
            m,
            m,
        ),
        (
            w - 1 - m,
            h - 1 - m,
        ),
        255,
        -1,
    )

    return mask


def make_border_ring(
    workspace_mask,
    ring_value,
    is_fraction=True,
):
    ys, xs = np.where(
        workspace_mask > 0
    )

    if xs.size == 0:
        return np.zeros_like(
            workspace_mask
        )

    x1 = int(
        xs.min()
    )
    x2 = int(
        xs.max()
    )
    y1 = int(
        ys.min()
    )
    y2 = int(
        ys.max()
    )

    min_dim = max(
        1,
        min(
            x2 - x1 + 1,
            y2 - y1 + 1,
        ),
    )

    if is_fraction:
        ring_px = int(
            round(
                float(ring_value)
                * min_dim
            )
        )
    else:
        ring_px = int(
            ring_value
        )

    ring_px = max(
        10,
        ring_px,
    )

    inner = cv2.erode(
        workspace_mask,
        cv2.getStructuringElement(
            cv2.MORPH_RECT,
            (
                2 * ring_px + 1,
                2 * ring_px + 1,
            ),
        ),
        iterations=1,
    )

    return cv2.bitwise_and(
        workspace_mask,
        cv2.bitwise_not(
            inner
        ),
    )


# =============================================================================
# Depth plane / object mask
# =============================================================================

def fit_inverse_depth_plane_ransac(
    depth_mm,
    sample_mask,
    args,
):
    valid = (
        (sample_mask > 0)
        & (
            depth_mm
            >= args.depth_min_mm
        )
        & (
            depth_mm
            <= args.depth_max_mm
        )
    )

    ys, xs = np.where(
        valid
    )

    if len(xs) < args.plane_min_points:
        return (
            None,
            0.0,
            float("inf"),
        )

    rng = np.random.default_rng(
        42
    )

    if len(xs) > args.plane_max_points:
        idx = rng.choice(
            len(xs),
            size=args.plane_max_points,
            replace=False,
        )

        xs = xs[
            idx
        ]
        ys = ys[
            idx
        ]

    z = depth_mm[
        ys,
        xs,
    ].astype(np.float64)

    h, w = depth_mm.shape

    xn = (
        xs.astype(np.float64)
        / max(
            1.0,
            w - 1.0,
        )
        * 2.0
        - 1.0
    )

    yn = (
        ys.astype(np.float64)
        / max(
            1.0,
            h - 1.0,
        )
        * 2.0
        - 1.0
    )

    A = np.column_stack(
        (
            xn,
            yn,
            np.ones_like(
                xn
            ),
        )
    )

    invz = 1.0 / np.maximum(
        z,
        1e-6,
    )

    best_inliers = None
    best_count = 0

    for _ in range(
        max(
            20,
            int(
                args.plane_ransac_iters
            ),
        )
    ):
        sample_idx = rng.choice(
            len(z),
            size=3,
            replace=False,
        )

        try:
            coeff = np.linalg.solve(
                A[
                    sample_idx
                ],
                invz[
                    sample_idx
                ],
            )
        except np.linalg.LinAlgError:
            continue

        pred_inv = (
            A
            @ coeff
        )

        pred_z = np.where(
            pred_inv > 1e-9,
            1.0 / pred_inv,
            np.inf,
        )

        residual = np.abs(
            pred_z
            - z
        )

        inliers = (
            residual
            <= args.plane_ransac_mm
        )

        count = int(
            np.count_nonzero(
                inliers
            )
        )

        if count > best_count:
            best_count = count
            best_inliers = inliers

    if (
        best_inliers is None
        or best_count
        < args.plane_min_points
    ):
        return (
            None,
            0.0,
            float("inf"),
        )

    coeff, *_ = np.linalg.lstsq(
        A[
            best_inliers
        ],
        invz[
            best_inliers
        ],
        rcond=None,
    )

    yy, xx = np.mgrid[
        0:h,
        0:w,
    ]

    x_full = (
        xx.astype(np.float64)
        / max(
            1.0,
            w - 1.0,
        )
        * 2.0
        - 1.0
    )

    y_full = (
        yy.astype(np.float64)
        / max(
            1.0,
            h - 1.0,
        )
        * 2.0
        - 1.0
    )

    pred_inv_full = (
        coeff[0]
        * x_full
        + coeff[1]
        * y_full
        + coeff[2]
    )

    plane_depth = np.zeros(
        (
            h,
            w,
        ),
        dtype=np.float32,
    )

    okay = (
        pred_inv_full
        > 1e-9
    )

    plane_depth[
        okay
    ] = (
        1.0
        / pred_inv_full[
            okay
        ]
    ).astype(
        np.float32
    )

    sample_pred_inv = (
        A
        @ coeff
    )

    sample_pred_z = np.where(
        sample_pred_inv > 1e-9,
        1.0 / sample_pred_inv,
        np.inf,
    )

    residual = np.abs(
        sample_pred_z
        - z
    )

    final_inliers = (
        np.isfinite(
            residual
        )
        & (
            residual
            <= args.plane_ransac_mm
        )
    )

    inlier_ratio = float(
        np.mean(
            final_inliers
        )
    )

    if np.any(
        final_inliers
    ):
        inlier_residual = float(
            np.median(
                residual[
                    final_inliers
                ]
            )
        )
    else:
        inlier_residual = float(
            "inf"
        )

    return (
        plane_depth,
        inlier_ratio,
        inlier_residual,
    )


def depth_object_candidate(
    depth_mm,
    plane_depth_mm,
    workspace_mask,
    args,
):
    valid = (
        (workspace_mask > 0)
        & (
            depth_mm
            >= args.depth_min_mm
        )
        & (
            depth_mm
            <= args.depth_max_mm
        )
        & (
            plane_depth_mm
            > 0
        )
    )

    height = np.zeros_like(
        depth_mm,
        dtype=np.float32,
    )

    height[
        valid
    ] = (
        plane_depth_mm[
            valid
        ]
        - depth_mm[
            valid
        ]
    )

    candidate = (
        valid
        & (
            height
            >= args.depth_height_threshold_mm
        )
        & (
            height
            <= args.depth_max_object_height_mm
        )
    )

    mask = np.where(
        candidate,
        255,
        0,
    ).astype(
        np.uint8
    )

    k_open = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (
            odd(
                args.object_open_size
            ),
            odd(
                args.object_open_size
            ),
        ),
    )

    k_close = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (
            odd(
                args.object_close_size
            ),
            odd(
                args.object_close_size
            ),
        ),
    )

    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_OPEN,
        k_open,
        iterations=1,
    )

    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_CLOSE,
        k_close,
        iterations=max(
            1,
            int(
                args.object_close_iterations
            ),
        ),
    )

    return (
        mask,
        height,
        valid,
    )


def select_final_object_mask(
    candidate,
    workspace_mask,
    args,
):
    num_labels, labels, stats, centroids = (
        cv2.connectedComponentsWithStats(
            candidate,
            8,
        )
    )

    if num_labels <= 1:
        return None

    ys, xs = np.where(
        workspace_mask > 0
    )

    if xs.size == 0:
        return None

    center = np.asarray(
        [
            float(
                np.mean(
                    xs
                )
            ),
            float(
                np.mean(
                    ys
                )
            ),
        ],
        dtype=np.float32,
    )

    workspace_area = max(
        1,
        int(
            np.count_nonzero(
                workspace_mask
            )
        ),
    )

    candidates = []

    for label in range(
        1,
        num_labels,
    ):
        area = int(
            stats[
                label,
                cv2.CC_STAT_AREA,
            ]
        )

        if area < args.min_object_area:
            continue

        if (
            area
            / workspace_area
            > args.max_object_area_ratio
        ):
            continue

        c = centroids[
            label
        ]

        dist = float(
            np.linalg.norm(
                c
                - center
            )
        )

        # 큰 component를 우선하되 중앙에서 터무니없이 먼 구조물은 감점
        score = (
            float(area)
            - 20.0
            * dist
        )

        candidates.append(
            (
                score,
                label,
            )
        )

    if not candidates:
        return None

    candidates.sort(
        reverse=True
    )

    label = candidates[
        0
    ][1]

    component = np.where(
        labels == label,
        255,
        0,
    ).astype(
        np.uint8
    )

    contours, _ = cv2.findContours(
        component,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    if not contours:
        return None

    contour = max(
        contours,
        key=cv2.contourArea,
    )

    final_mask = np.zeros_like(
        component
    )

    # 내부 depth hole은 결함검사 ROI와 무관하므로 외곽 내부를 전부 채운다.
    cv2.drawContours(
        final_mask,
        [
            contour
        ],
        -1,
        255,
        cv2.FILLED,
    )

    return final_mask


# =============================================================================
# Surface-only / patch generation
# =============================================================================

def erode_surface_mask(
    object_mask,
    margin_px,
):
    margin_px = max(
        0,
        int(
            margin_px
        ),
    )

    if margin_px == 0:
        return object_mask.copy()

    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (
            2 * margin_px + 1,
            2 * margin_px + 1,
        ),
    )

    return cv2.erode(
        object_mask,
        kernel,
        iterations=1,
    )


def generate_surface_patches(
    surface_mask,
    patch_size,
    stride,
    min_coverage,
):
    h, w = surface_mask.shape

    patch_size = int(
        patch_size
    )
    stride = int(
        stride
    )

    min_coverage = float(
        np.clip(
            min_coverage,
            0.0,
            1.0,
        )
    )

    patches = []

    for y in range(
        0,
        h - patch_size + 1,
        stride,
    ):
        for x in range(
            0,
            w - patch_size + 1,
            stride,
        ):
            roi = surface_mask[
                y:y + patch_size,
                x:x + patch_size,
            ]

            coverage = float(
                np.mean(
                    roi > 0
                )
            )

            if coverage >= min_coverage:
                patches.append(
                    {
                        "x": int(
                            x
                        ),
                        "y": int(
                            y
                        ),
                        "w": int(
                            patch_size
                        ),
                        "h": int(
                            patch_size
                        ),
                        "coverage": coverage,
                    }
                )

    return patches


def draw_mask_contour(
    image,
    mask,
    color,
    thickness,
):
    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    cv2.drawContours(
        image,
        contours,
        -1,
        color,
        thickness,
    )


def render_height_map(
    height,
    max_mm=60.0,
):
    clipped = np.clip(
        height,
        0.0,
        float(
            max_mm
        ),
    )

    gray = (
        clipped
        / max(
            1e-6,
            float(
                max_mm
            ),
        )
        * 255.0
    ).astype(
        np.uint8
    )

    return cv2.applyColorMap(
        gray,
        cv2.COLORMAP_TURBO,
    )


# =============================================================================
# Manual-Z inspection readiness
# =============================================================================

def mask_touches_frame_edge(
    mask: Optional[np.ndarray],
    margin_px: int,
) -> bool:
    if mask is None:
        return True

    ys, xs = np.where(
        mask > 0
    )

    if xs.size == 0:
        return True

    h, w = mask.shape
    m = max(
        0,
        int(margin_px),
    )

    return (
        int(xs.min()) <= m
        or int(ys.min()) <= m
        or int(xs.max()) >= w - 1 - m
        or int(ys.max()) >= h - 1 - m
    )


def evaluate_inspection_readiness(
    object_mask: Optional[np.ndarray],
    surface_mask: Optional[np.ndarray],
    patches: list[dict],
    depth_valid_ratio: float,
    plane_inlier_ratio: float,
    plane_residual_mm: float,
    args: argparse.Namespace,
):
    """
    자동 Z 위치를 계산하지 않는다.
    사용자가 Z를 올리고/내리면서 현재 센서 상태가 검사 가능한지만 판단한다.
    """
    reasons = []

    if depth_valid_ratio < args.min_depth_valid_ratio:
        reasons.append(
            f"Depth valid low ({depth_valid_ratio*100:.1f}% < "
            f"{args.min_depth_valid_ratio*100:.0f}%)"
        )

    if plane_inlier_ratio < args.min_plane_inlier_ratio:
        reasons.append(
            f"Plane inlier low ({plane_inlier_ratio*100:.1f}% < "
            f"{args.min_plane_inlier_ratio*100:.0f}%)"
        )

    if (
        not np.isfinite(
            plane_residual_mm
        )
        or plane_residual_mm
        > args.max_plane_inlier_residual_mm
    ):
        residual_text = (
            f"{plane_residual_mm:.2f}"
            if np.isfinite(plane_residual_mm)
            else "inf"
        )

        reasons.append(
            f"Plane residual high ({residual_text} mm > "
            f"{args.max_plane_inlier_residual_mm:.1f} mm)"
        )

    if object_mask is None:
        reasons.append(
            "Object surface not found"
        )

    if surface_mask is None:
        reasons.append(
            "Surface-only mask not found"
        )

    if len(patches) < args.min_valid_patches:
        reasons.append(
            f"Too few valid patches ({len(patches)} < "
            f"{args.min_valid_patches})"
        )

    if mask_touches_frame_edge(
        object_mask,
        args.fov_edge_margin_px,
    ):
        reasons.append(
            "Object too close to image edge / FOV"
        )

    ready = (
        len(
            reasons
        )
        == 0
    )

    return (
        ready,
        reasons,
    )


# =============================================================================
# Save
# =============================================================================

def save_pose_result(
    args,
    frame,
    overlay,
    object_mask,
    surface_mask,
    candidate_mask,
    height_map,
    patches,
    metrics,
):
    timestamp = datetime.now().strftime(
        "%Y%m%d_%H%M%S"
    )

    pose_name = (
        args.pose_name
        .strip()
        .replace(
            " ",
            "_",
        )
        or "pose"
    )

    out = (
        args.output_dir
        / f"{pose_name}_{timestamp}"
    )

    out.mkdir(
        parents=True,
        exist_ok=True,
    )

    cv2.imwrite(
        str(
            out
            / "color.png"
        ),
        frame,
    )

    cv2.imwrite(
        str(
            out
            / "object_mask.png"
        ),
        object_mask,
    )

    cv2.imwrite(
        str(
            out
            / "surface_mask.png"
        ),
        surface_mask,
    )

    cv2.imwrite(
        str(
            out
            / "depth_object_candidate.png"
        ),
        candidate_mask,
    )

    cv2.imwrite(
        str(
            out
            / "height_from_board.png"
        ),
        height_map,
    )

    cv2.imwrite(
        str(
            out
            / "surface_patch_overlay.png"
        ),
        overlay,
    )

    payload = {
        "pose_name": pose_name,
        "patch_size": int(
            args.patch_size
        ),
        "patch_stride": int(
            args.patch_stride
        ),
        "boundary_margin_px": int(
            args.boundary_margin_px
        ),
        "patch_mask_coverage": float(
            args.patch_mask_coverage
        ),
        "patch_count": len(
            patches
        ),
        "patches": patches,
        "metrics": metrics,
    }

    (
        out
        / "surface_patches.json"
    ).write_text(
        json.dumps(
            payload,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    if args.save_patches:
        patch_dir = (
            out
            / "patches"
        )

        patch_dir.mkdir(
            exist_ok=True
        )

        for index, patch in enumerate(
            patches
        ):
            x = patch[
                "x"
            ]
            y = patch[
                "y"
            ]
            size = patch[
                "w"
            ]

            crop = frame[
                y:y + size,
                x:x + size,
            ]

            cv2.imwrite(
                str(
                    patch_dir
                    / f"patch_{index:04d}_x{x}_y{y}.png"
                ),
                crop,
            )

    print(
        "[SAVE]",
        out.resolve(),
    )


# =============================================================================
# Main
# =============================================================================

def main():
    args = parse_args()

    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    pipeline = Pipeline()
    config = Config()
    align_filter = AlignFilter(
        align_to_stream=OBStreamType.COLOR_STREAM
    )

    detector = create_aruco_detector()

    depth_history = []

    ready_streak = 0
    last_ready = False
    last_reasons = []

    started = False

    try:
        color_profile = find_color_profile(
            pipeline
        )

        depth_profile = find_depth_profile(
            pipeline
        )

        config.enable_stream(
            color_profile
        )
        config.enable_stream(
            depth_profile
        )

        try:
            config.set_frame_aggregate_output_mode(
                OBFrameAggregateOutputMode.FULL_FRAME_REQUIRE
            )
        except Exception as exc:
            print(
                f"[WARN] aggregate mode 설정 실패: {exc}"
            )

        pipeline.start(
            config
        )
        started = True

        device = pipeline.get_device()

        configure_camera(
            device,
            args,
        )

        print("=" * 88)
        print("SURFACE-ONLY MANUAL-Z INSPECTION PROTOTYPE")
        print("")
        print("목적:")
        print("  - 바닥/경계를 AE 학습·검사에서 제거")
        print("  - PLY/3D 형상이 Pitch/Roll 검사 자세를 결정")
        print("  - 사용자는 그 자세에서 Z축 높이만 수동 조절")
        print("  - 각 프레임의 RGB+Depth 품질로 INSPECTION READY 여부 판단")
        print("  - 바닥/경계는 학습·검사에서 제외하고 내부 surface patch만 생성")
        print("  - Z 이동량/ROI tracking/빈 판 background model 사용 안 함")
        print("")
        print("표시:")
        print("  PURPLE = 현재 물체 외곽")
        print("  YELLOW = boundary margin을 제거한 surface mask")
        print("  CYAN   = 실제 AE에 넣을 64x64 valid patch")
        print("")
        print("SPACE = READY 상태에서 현재 자세 결과 저장")
        print("Q/ESC = 종료")
        print("=" * 88)

        for _ in range(
            args.warmup_frames
        ):
            wait_for_aligned_pair(
                pipeline,
                align_filter,
            )

        while True:
            frame, depth_mm = wait_for_aligned_pair(
                pipeline,
                align_filter,
            )

            depth_history.append(
                depth_mm
            )

            max_frames = max(
                1,
                int(
                    args.depth_median_frames
                ),
            )

            if len(
                depth_history
            ) > max_frames:
                depth_history = depth_history[
                    -max_frames:
                ]

            depth_median = temporal_median_depth(
                depth_history
            )

            marker_map = detect_markers(
                frame,
                detector,
            )

            marker_ids = sorted(
                marker_map.keys()
            )

            board_quad = get_board_outer_quad(
                marker_map
            )

            overlay = frame.copy()

            # marker visualization
            for marker_id, corners in marker_map.items():
                pts = corners.astype(
                    np.int32
                )

                cv2.polylines(
                    overlay,
                    [
                        pts
                    ],
                    True,
                    (0, 255, 255),
                    2,
                )

                center = np.mean(
                    corners,
                    axis=0,
                ).astype(
                    np.int32
                )

                cv2.putText(
                    overlay,
                    f"ID {marker_id}",
                    (
                        int(
                            center[0]
                        ),
                        int(
                            center[1]
                        ),
                    ),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 255),
                    2,
                    cv2.LINE_AA,
                )

            if board_quad is not None:
                workspace_mask = polygon_mask(
                    frame.shape,
                    board_quad,
                )

                plane_ring = make_border_ring(
                    workspace_mask,
                    args.board_ring_fraction,
                    is_fraction=True,
                )

                workspace_source = "ARUCO"

                cv2.polylines(
                    overlay,
                    [
                        board_quad.astype(
                            np.int32
                        )
                    ],
                    True,
                    (0, 255, 0),
                    3,
                )

            else:
                workspace_mask = fallback_workspace_mask(
                    frame.shape,
                    args.fallback_margin,
                )

                plane_ring = make_border_ring(
                    workspace_mask,
                    args.fallback_ring_px,
                    is_fraction=False,
                )

                workspace_source = "FALLBACK"

            workspace_pixels = (
                workspace_mask > 0
            )

            valid_depth = (
                workspace_pixels
                & (
                    depth_median
                    >= args.depth_min_mm
                )
                & (
                    depth_median
                    <= args.depth_max_mm
                )
            )

            depth_valid_ratio = float(
                np.count_nonzero(
                    valid_depth
                )
                / max(
                    1,
                    np.count_nonzero(
                        workspace_pixels
                    ),
                )
            )

            (
                plane_depth,
                plane_inlier_ratio,
                plane_residual_mm,
            ) = fit_inverse_depth_plane_ransac(
                depth_median,
                plane_ring,
                args,
            )

            object_mask = None
            surface_mask = None
            candidate_mask = np.zeros(
                frame.shape[:2],
                dtype=np.uint8,
            )
            height = np.zeros(
                frame.shape[:2],
                dtype=np.float32,
            )
            patches = []

            plane_good = (
                plane_depth is not None
                and depth_valid_ratio
                >= args.min_depth_valid_ratio
                and plane_inlier_ratio
                >= args.min_plane_inlier_ratio
                and plane_residual_mm
                <= args.max_plane_inlier_residual_mm
            )

            if plane_good:
                (
                    candidate_mask,
                    height,
                    _,
                ) = depth_object_candidate(
                    depth_median,
                    plane_depth,
                    workspace_mask,
                    args,
                )

                object_mask = select_final_object_mask(
                    candidate_mask,
                    workspace_mask,
                    args,
                )

                if object_mask is not None:
                    surface_mask = erode_surface_mask(
                        object_mask,
                        args.boundary_margin_px,
                    )

                    patches = generate_surface_patches(
                        surface_mask,
                        args.patch_size,
                        args.patch_stride,
                        args.patch_mask_coverage,
                    )

            (
                current_ready,
                ready_reasons,
            ) = evaluate_inspection_readiness(
                object_mask=object_mask,
                surface_mask=surface_mask,
                patches=patches,
                depth_valid_ratio=depth_valid_ratio,
                plane_inlier_ratio=plane_inlier_ratio,
                plane_residual_mm=plane_residual_mm,
                args=args,
            )

            if current_ready:
                ready_streak += 1
            else:
                ready_streak = 0

            good_for_inspection = (
                current_ready
                and ready_streak
                >= int(
                    args.ready_streak_frames
                )
            )

            last_ready = good_for_inspection
            last_reasons = ready_reasons

            # visualization
            if object_mask is not None:
                draw_mask_contour(
                    overlay,
                    object_mask,
                    (255, 0, 255),
                    4,
                )

            if surface_mask is not None:
                draw_mask_contour(
                    overlay,
                    surface_mask,
                    (0, 255, 255),
                    2,
                )

            for patch in patches:
                x = patch[
                    "x"
                ]
                y = patch[
                    "y"
                ]
                w = patch[
                    "w"
                ]
                h = patch[
                    "h"
                ]

                cv2.rectangle(
                    overlay,
                    (
                        x,
                        y,
                    ),
                    (
                        x + w - 1,
                        y + h - 1,
                    ),
                    (255, 255, 0),
                    1,
                )

            object_area = (
                int(
                    np.count_nonzero(
                        object_mask
                    )
                )
                if object_mask is not None
                else 0
            )

            surface_area = (
                int(
                    np.count_nonzero(
                        surface_mask
                    )
                )
                if surface_mask is not None
                else 0
            )

            if good_for_inspection:
                state_text = "STATE: INSPECTION READY - PRESS SPACE"
            elif current_ready:
                state_text = (
                    "STATE: HOLD Z / STABILIZING "
                    f"{ready_streak}/{args.ready_streak_frames}"
                )
            else:
                state_text = "STATE: ADJUST Z"

            reason_text = (
                " | ".join(
                    ready_reasons[:2]
                )
                if ready_reasons
                else "All quality gates passed"
            )

            draw_status(
                overlay,
                [
                    state_text,
                    (
                        f"Workspace={workspace_source} "
                        f"| IDs={marker_ids}"
                    ),
                    (
                        f"Depth valid={depth_valid_ratio*100:.1f}% "
                        f"| Plane inlier={plane_inlier_ratio*100:.1f}% "
                        f"| inlier residual={plane_residual_mm:.2f} mm"
                    ),
                    (
                        f"Object area={object_area}px "
                        f"| Surface area={surface_area}px"
                    ),
                    (
                        f"Patch={args.patch_size}x{args.patch_size} "
                        f"| stride={args.patch_stride} "
                        f"| valid patches={len(patches)} "
                        f"(need>={args.min_valid_patches})"
                    ),
                    (
                        f"Boundary margin={args.boundary_margin_px}px "
                        f"| patch coverage>={args.patch_mask_coverage:.2f}"
                    ),
                    (
                        f"Z guidance: {reason_text}"
                    ),
                    "PURPLE=object | YELLOW=surface-only | CYAN=AE patches",
                ],
                good=good_for_inspection,
            )

            preview = cv2.resize(
                overlay,
                (
                    960,
                    600,
                ),
                interpolation=cv2.INTER_AREA,
            )

            cv2.imshow(
                "Surface-only Pose Inspection",
                preview,
            )

            height_vis = render_height_map(
                height,
                max_mm=max(
                    40.0,
                    args.depth_height_threshold_mm
                    * 10.0,
                ),
            )

            cv2.imshow(
                "Height From Board Plane",
                cv2.resize(
                    height_vis,
                    (
                        640,
                        400,
                    ),
                    interpolation=cv2.INTER_AREA,
                ),
            )

            if surface_mask is not None:
                cv2.imshow(
                    "Surface Mask",
                    cv2.resize(
                        surface_mask,
                        (
                            640,
                            400,
                        ),
                        interpolation=cv2.INTER_NEAREST,
                    ),
                )

            key = (
                cv2.waitKey(
                    1
                )
                & 0xFF
            )

            if key in (
                ord("q"),
                ord("Q"),
                27,
            ):
                break

            if key == 32:
                if not good_for_inspection:
                    print(
                        "[SPACE] 아직 INSPECTION READY가 아닙니다."
                    )

                    if last_reasons:
                        print(
                            "  - "
                            + "\n  - ".join(
                                last_reasons
                            )
                        )
                    continue

                metrics = {
                    "workspace_source": workspace_source,
                    "marker_ids": marker_ids,
                    "depth_valid_ratio": depth_valid_ratio,
                    "plane_inlier_ratio": plane_inlier_ratio,
                    "plane_inlier_residual_mm": plane_residual_mm,
                    "object_area_px": object_area,
                    "surface_area_px": surface_area,
                    "valid_patch_count": len(
                        patches
                    ),
                    "inspection_ready": good_for_inspection,
                    "ready_streak_frames": ready_streak,
                    "ready_reasons": ready_reasons,
                }

                save_pose_result(
                    args,
                    frame,
                    overlay,
                    object_mask,
                    surface_mask,
                    candidate_mask,
                    height_vis,
                    patches,
                    metrics,
                )

    finally:
        if started:
            pipeline.stop()

        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

