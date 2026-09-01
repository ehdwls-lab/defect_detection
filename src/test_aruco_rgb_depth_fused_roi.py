from __future__ import annotations

import argparse
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

# Depth는 RGB에 software D2C 정렬해서 사용.
# 1280x800@10 Y16을 우선 찾고, 없으면 1280x800@15 Y16,
# 그것도 없으면 첫 Y16 profile을 fallback으로 사용한다.
DEPTH_PREFERRED_FPS = (10, 15)
DEPTH_FORMAT = OBFormat.Y16

FRAME_TIMEOUT_MS = 3000

# ID0=좌상 / ID1=우상 / ID2=좌하 / ID3=우하
REQUIRED_IDS = (0, 1, 2, 3)


# =============================================================================
# Arguments
# =============================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "ArUco 판을 canonical view로 정규화하고, "
            "빈 판 30프레임의 pixel-wise Median/MAD background model과 비교하여 "
            "LOW-Z에서 물체 ROI만 안정적으로 추출하는 테스트"
        )
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "results" / "background_model_roi_test",
    )

    # Camera: 프로그램 시작부터 끝까지 동일한 수동 설정
    parser.add_argument("--brightness", type=int, default=48)
    parser.add_argument("--exposure", type=int, default=1100)
    parser.add_argument("--gain", type=int, default=64)
    parser.add_argument("--white-balance", type=int, default=4600)
    parser.add_argument("--warmup-frames", type=int, default=40)

    # Background model
    parser.add_argument("--background-frames", type=int, default=30)
    parser.add_argument("--background-max-attempts", type=int, default=100)
    parser.add_argument("--canonical-size", type=int, default=800)

    # Pixel-wise deviation thresholds
    parser.add_argument(
        "--abs-luma-threshold",
        type=float,
        default=7.0,
        help="LAB L 절대 변화량 최소 threshold",
    )
    parser.add_argument(
        "--abs-chroma-threshold",
        type=float,
        default=5.0,
        help="LAB chroma 변화량 최소 threshold",
    )
    parser.add_argument(
        "--noise-sigma-mult",
        type=float,
        default=5.0,
        help="빈 판 pixel별 sigma 대비 변화량 배수",
    )
    parser.add_argument(
        "--noise-floor",
        type=float,
        default=1.5,
        help="MAD가 거의 0인 pixel의 최소 sigma floor",
    )

    # Candidate cleanup
    parser.add_argument("--open-size", type=int, default=5)
    parser.add_argument("--close-size", type=int, default=19)
    parser.add_argument("--close-iterations", type=int, default=2)
    parser.add_argument("--min-object-area", type=int, default=12000)
    parser.add_argument(
        "--max-hole-area",
        type=int,
        default=5000,
        help="이 면적 이하의 내부 hole은 물체 내부로 채움",
    )
    parser.add_argument(
        "--border-ignore-px",
        type=int,
        default=8,
        help="canonical board 가장자리 무시 폭",
    )
    parser.add_argument(
        "--marker-ignore-px",
        type=int,
        default=10,
        help="ArUco marker 주변을 추가로 제외할 여백",
    )

    # Seed-based GrabCut refinement
    parser.add_argument(
        "--strong-seed-mult",
        type=float,
        default=1.6,
        help=(
            "Background threshold의 몇 배 이상 차이나는 pixel을 "
            "확실한 foreground seed로 볼지"
        ),
    )
    parser.add_argument(
        "--density-window",
        type=int,
        default=51,
        help=(
            "fragmented background-change pixel을 물체 envelope로 묶기 위한 "
            "local density window 크기"
        ),
    )
    parser.add_argument(
        "--density-threshold",
        type=float,
        default=0.10,
        help=(
            "density window 내부에서 changed pixel 비율이 이 값 이상이면 "
            "물체 support 후보"
        ),
    )
    parser.add_argument(
        "--support-close-size",
        type=int,
        default=27,
    )
    parser.add_argument(
        "--support-dilate-px",
        type=int,
        default=18,
        help="GrabCut이 실제 경계를 찾을 수 있도록 support 바깥에 줄 여유 영역",
    )
    parser.add_argument(
        "--grabcut-iterations",
        type=int,
        default=3,
    )
    parser.add_argument(
        "--min-strong-seed-area",
        type=int,
        default=300,
    )
    parser.add_argument(
        "--max-support-area-ratio",
        type=float,
        default=0.65,
        help="판 전체의 너무 큰 영역을 물체 support로 잘못 잡는 것 방지",
    )

    # ------------------------------------------------------------------
    # Depth fusion
    # ------------------------------------------------------------------
    parser.add_argument(
        "--depth-exposure",
        type=int,
        default=3000,
        help="Depth 수동 exposure. sweep 결과 기본값=3000",
    )
    parser.add_argument(
        "--depth-gain",
        type=int,
        default=16,
        help="Depth 수동 gain. sweep 결과 기본값=16",
    )
    parser.add_argument(
        "--depth-median-frames",
        type=int,
        default=5,
        help="aligned Depth temporal median에 사용할 최근 프레임 수",
    )
    parser.add_argument(
        "--depth-min-mm",
        type=float,
        default=80.0,
    )
    parser.add_argument(
        "--depth-max-mm",
        type=float,
        default=2000.0,
    )
    parser.add_argument(
        "--depth-height-threshold-mm",
        type=float,
        default=4.0,
        help="추정된 판 평면보다 카메라 쪽으로 이 값 이상 떠 있으면 물체 seed",
    )
    parser.add_argument(
        "--depth-max-object-height-mm",
        type=float,
        default=250.0,
        help="비정상적으로 큰 plane residual 제거용 상한",
    )
    parser.add_argument(
        "--depth-plane-ring-fraction",
        type=float,
        default=0.22,
        help="canonical 판 외곽에서 plane fitting에 사용할 ring 비율",
    )
    parser.add_argument(
        "--depth-plane-ransac-mm",
        type=float,
        default=2.5,
        help="inverse-depth plane RANSAC의 depth residual threshold(mm)",
    )
    parser.add_argument(
        "--depth-plane-ransac-iters",
        type=int,
        default=120,
    )
    parser.add_argument(
        "--depth-plane-max-points",
        type=int,
        default=5000,
    )
    parser.add_argument(
        "--depth-min-plane-points",
        type=int,
        default=400,
    )
    parser.add_argument(
        "--depth-open-size",
        type=int,
        default=5,
    )
    parser.add_argument(
        "--depth-close-size",
        type=int,
        default=17,
    )
    parser.add_argument(
        "--depth-seed-dilate-px",
        type=int,
        default=3,
        help="Depth object seed를 약간 확장해 GrabCut strong FG로 전달",
    )

    # ------------------------------------------------------------------
    # LOW-Z final ROI post-processing
    # ------------------------------------------------------------------
    parser.add_argument(
        "--final-close-size",
        type=int,
        default=9,
        help="최종 외곽의 작은 끊김/요철을 연결하는 closing kernel",
    )
    parser.add_argument(
        "--final-smooth-size",
        type=int,
        default=7,
        help="최종 외곽을 약하게 부드럽게 할 Gaussian kernel",
    )
    parser.add_argument(
        "--contour-epsilon-ratio",
        type=float,
        default=0.0015,
        help="외곽 contour approxPolyDP epsilon/perimeter 비율",
    )
    parser.add_argument(
        "--inspection-erode-px",
        type=int,
        default=10,
        help="실제 결함검사용 mask를 물체 외곽에서 안쪽으로 줄일 거리(px)",
    )

    # ------------------------------------------------------------------
    # Current-frame dynamic ROI tracking
    # Z 이동량을 사용하지 않는다.
    # 이전 ROI 주변에서 현재 RGB + 현재 Depth로 매 프레임 ROI를 다시 계산한다.
    # ------------------------------------------------------------------
    parser.add_argument(
        "--track-search-scale",
        type=float,
        default=0.07,
        help="이전 ROI sqrt(area)에 곱해 local search margin을 정하는 비율",
    )
    parser.add_argument(
        "--track-search-min-px",
        type=int,
        default=16,
    )
    parser.add_argument(
        "--track-search-max-px",
        type=int,
        default=42,
    )
    parser.add_argument(
        "--track-plane-ring-px",
        type=int,
        default=70,
        help="search area 바깥에서 판 plane fitting에 사용할 ring 폭",
    )
    parser.add_argument(
        "--track-prev-core-erode-px",
        type=int,
        default=8,
        help="이전 ROI 내부 core를 GrabCut definite foreground로 사용",
    )
    parser.add_argument(
        "--track-prev-prob-dilate-px",
        type=int,
        default=12,
        help="이전 ROI 주변을 GrabCut probable foreground로 확장",
    )
    parser.add_argument(
        "--track-min-area-ratio",
        type=float,
        default=0.90,
        help="현재 ROI / 이전 ROI 면적 비율 최소값",
    )
    parser.add_argument(
        "--track-max-area-ratio",
        type=float,
        default=1.12,
        help="Depth 신뢰도가 낮을 때 적용할 보수적인 최대 면적 증가율",
    )
    parser.add_argument(
        "--track-medium-max-area-ratio",
        type=float,
        default=1.30,
        help="Depth/plane 신뢰도가 중간일 때 허용할 최대 면적 증가율",
    )
    parser.add_argument(
        "--track-high-max-area-ratio",
        type=float,
        default=1.36,
        help="Depth/plane 신뢰도가 높을 때 허용할 최대 면적 증가율",
    )
    parser.add_argument(
        "--track-medium-depth-valid",
        type=float,
        default=0.25,
    )
    parser.add_argument(
        "--track-high-depth-valid",
        type=float,
        default=0.60,
    )
    parser.add_argument(
        "--track-medium-plane-inlier",
        type=float,
        default=0.35,
    )
    parser.add_argument(
        "--track-high-plane-inlier",
        type=float,
        default=0.60,
    )
    parser.add_argument(
        "--track-medium-plane-residual-mm",
        type=float,
        default=4.0,
    )
    parser.add_argument(
        "--track-high-plane-residual-mm",
        type=float,
        default=1.5,
    )
    parser.add_argument(
        "--track-min-overlap-ratio",
        type=float,
        default=0.82,
        help="이전 ROI 중 현재 ROI와 겹쳐야 하는 최소 비율",
    )
    parser.add_argument(
        "--track-min-depth-valid-ratio",
        type=float,
        default=0.15,
        help="tracking search area에서 Depth valid가 이 값 미만이면 ROI 갱신 대신 HOLD",
    )

    return parser.parse_args()


# =============================================================================
# Camera
# =============================================================================

def find_color_profile(pipeline: Pipeline):
    profiles = pipeline.get_stream_profile_list(
        OBSensorType.COLOR_SENSOR
    )

    fallback = None

    for i in range(profiles.get_count()):
        profile = profiles.get_stream_profile_by_index(i)

        if profile.get_format() != COLOR_FORMAT:
            continue

        if fallback is None:
            fallback = profile

        if (
            profile.get_width() == WIDTH
            and profile.get_height() == HEIGHT
            and profile.get_fps() == COLOR_FPS
        ):
            return profile

    if fallback is not None:
        print(
            "[WARN] exact Color profile 미검출 -> "
            f"{fallback.get_width()}x{fallback.get_height()} @"
            f"{fallback.get_fps()} {fallback.get_format()} 사용"
        )
        return fallback

    raise RuntimeError("MJPG Color profile을 찾지 못했습니다.")


def find_depth_profile(pipeline: Pipeline):
    profiles = pipeline.get_stream_profile_list(
        OBSensorType.DEPTH_SENSOR
    )

    fallback = None

    # 같은 해상도 + 선호 fps 순서
    for preferred_fps in DEPTH_PREFERRED_FPS:
        for i in range(profiles.get_count()):
            profile = profiles.get_stream_profile_by_index(i)

            if profile.get_format() != DEPTH_FORMAT:
                continue

            if fallback is None:
                fallback = profile

            if (
                profile.get_width() == WIDTH
                and profile.get_height() == HEIGHT
                and profile.get_fps() == preferred_fps
            ):
                return profile

    if fallback is not None:
        print(
            "[WARN] exact Depth profile 미검출 -> "
            f"{fallback.get_width()}x{fallback.get_height()} @"
            f"{fallback.get_fps()} {fallback.get_format()} 사용"
        )
        return fallback

    raise RuntimeError("Y16 Depth profile을 찾지 못했습니다.")


def frame_to_bgr(color_frame) -> np.ndarray:
    raw = np.frombuffer(
        color_frame.get_data(),
        dtype=np.uint8,
    )

    image = cv2.imdecode(
        raw,
        cv2.IMREAD_COLOR,
    )

    if image is None:
        raise RuntimeError("MJPG Color frame decode 실패")

    return image


def depth_frame_to_mm(depth_frame) -> np.ndarray:
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
    """
    Software D2C alignment:
      raw frameset
        -> AlignFilter(COLOR_STREAM)
        -> Color + Depth aligned to color view
    """
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

        # pyorbbecsdk 버전에 따라 Frame 또는 FrameSet일 수 있어 방어적으로 처리
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

        color_bgr = frame_to_bgr(
            color_frame
        )

        depth_mm = depth_frame_to_mm(
            depth_frame
        )

        # 정렬 후에도 shape가 다르면 마지막 안전장치로 nearest resize
        if (
            depth_mm.shape[:2]
            != color_bgr.shape[:2]
        ):
            depth_mm = cv2.resize(
                depth_mm,
                (
                    color_bgr.shape[1],
                    color_bgr.shape[0],
                ),
                interpolation=cv2.INTER_NEAREST,
            )

        return (
            color_bgr,
            depth_mm,
        )


def configure_depth_manual(
    device,
    args: argparse.Namespace,
) -> None:
    print("=" * 72)
    print("Depth 수동 설정")

    try:
        device.set_bool_property(
            OBPropertyID.OB_PROP_DEPTH_AUTO_EXPOSURE_BOOL,
            False,
        )
        print("Depth Auto Exposure: OFF")
    except Exception as exc:
        print(
            f"[WARN] Depth Auto Exposure OFF 실패: {exc}"
        )

    try:
        device.set_int_property(
            OBPropertyID.OB_PROP_DEPTH_EXPOSURE_INT,
            int(args.depth_exposure),
        )
        print(
            f"Depth Exposure: {args.depth_exposure}"
        )
    except Exception as exc:
        print(
            f"[WARN] Depth Exposure 설정 실패: {exc}"
        )

    try:
        device.set_int_property(
            OBPropertyID.OB_PROP_DEPTH_GAIN_INT,
            int(args.depth_gain),
        )
        print(
            f"Depth Gain: {args.depth_gain}"
        )
    except Exception as exc:
        print(
            f"[WARN] Depth Gain 설정 실패: {exc}"
        )

    print("=" * 72)


def set_manual_int_property(
    device,
    property_id,
    requested: int,
    label: str,
):
    value = int(requested)

    try:
        rng = device.get_int_property_range(property_id)
        low = int(rng.min)
        high = int(rng.max)

        value = int(np.clip(value, low, high))

        device.set_int_property(
            property_id,
            value,
        )

        print(
            f"{label}: {value}  (range {low}~{high})"
        )

    except Exception as exc:
        try:
            device.set_int_property(
                property_id,
                value,
            )
            print(
                f"{label}: {value}  (range query failed: {exc})"
            )
        except Exception as exc2:
            print(
                f"[WARN] {label} 설정 실패: {exc2}"
            )


def configure_manual_camera(
    device,
    args: argparse.Namespace,
) -> None:
    print("=" * 72)
    print("카메라 수동 고정")

    try:
        device.set_bool_property(
            OBPropertyID.OB_PROP_COLOR_AUTO_EXPOSURE_BOOL,
            False,
        )
        print("Auto Exposure: OFF")
    except Exception as exc:
        print(f"[WARN] Auto Exposure OFF 실패: {exc}")

    try:
        device.set_bool_property(
            OBPropertyID.OB_PROP_COLOR_AUTO_WHITE_BALANCE_BOOL,
            False,
        )
        print("Auto White Balance: OFF")
    except Exception as exc:
        print(
            f"[WARN] Auto White Balance OFF 실패: {exc}"
        )

    set_manual_int_property(
        device,
        OBPropertyID.OB_PROP_COLOR_EXPOSURE_INT,
        args.exposure,
        "Exposure",
    )

    set_manual_int_property(
        device,
        OBPropertyID.OB_PROP_COLOR_GAIN_INT,
        args.gain,
        "Gain",
    )

    set_manual_int_property(
        device,
        OBPropertyID.OB_PROP_COLOR_WHITE_BALANCE_INT,
        args.white_balance,
        "White Balance",
    )

    set_manual_int_property(
        device,
        OBPropertyID.OB_PROP_COLOR_BRIGHTNESS_INT,
        args.brightness,
        "Brightness",
    )

    print("=" * 72)


# =============================================================================
# ArUco
# =============================================================================

def create_aruco_detector():
    if not hasattr(cv2, "aruco"):
        raise RuntimeError(
            "cv2.aruco가 없습니다. opencv-contrib-python이 필요합니다."
        )

    aruco = cv2.aruco

    dictionary = aruco.getPredefinedDictionary(
        aruco.DICT_4X4_50
    )

    if hasattr(aruco, "DetectorParameters"):
        params = aruco.DetectorParameters()
    else:
        params = aruco.DetectorParameters_create()

    if hasattr(params, "cornerRefinementMethod"):
        params.cornerRefinementMethod = (
            aruco.CORNER_REFINE_SUBPIX
        )

    if hasattr(params, "adaptiveThreshWinSizeMin"):
        params.adaptiveThreshWinSizeMin = 3

    if hasattr(params, "adaptiveThreshWinSizeMax"):
        params.adaptiveThreshWinSizeMax = 35

    if hasattr(params, "adaptiveThreshWinSizeStep"):
        params.adaptiveThreshWinSizeStep = 4

    if hasattr(aruco, "ArucoDetector"):
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
    image_bgr: np.ndarray,
    detector,
) -> dict[int, np.ndarray]:
    gray = cv2.cvtColor(
        image_bgr,
        cv2.COLOR_BGR2GRAY,
    )

    if hasattr(detector, "detectMarkers"):
        corners, ids, _ = detector.detectMarkers(gray)
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
            marker_map[int(marker_id)] = (
                corners_4
                .reshape(4, 2)
                .astype(np.float32)
            )

    return marker_map


def get_board_outer_quad(
    marker_map: dict[int, np.ndarray]
) -> Optional[np.ndarray]:
    """
    OpenCV marker corner:
      0 TL, 1 TR, 2 BR, 3 BL

    판 기준은 마커 4개의 가장 바깥쪽 corner.
    반환: TL, TR, BR, BL
    """
    if not all(
        marker_id in marker_map
        for marker_id in REQUIRED_IDS
    ):
        return None

    tl = marker_map[0][0]
    tr = marker_map[1][1]
    bl = marker_map[2][3]
    br = marker_map[3][2]

    return np.asarray(
        [tl, tr, br, bl],
        dtype=np.float32,
    )


# =============================================================================
# Canonical board
# =============================================================================

def canonical_corners(size: int) -> np.ndarray:
    s = float(size - 1)

    return np.asarray(
        [
            [0.0, 0.0],
            [s, 0.0],
            [s, s],
            [0.0, s],
        ],
        dtype=np.float32,
    )


def warp_board_to_canonical(
    image_bgr: np.ndarray,
    board_quad: np.ndarray,
    size: int,
):
    dst = canonical_corners(size)

    H = cv2.getPerspectiveTransform(
        board_quad.astype(np.float32),
        dst,
    )

    warped = cv2.warpPerspective(
        image_bgr,
        H,
        (size, size),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )

    return warped, H


def inverse_warp_mask_to_frame(
    canonical_mask: np.ndarray,
    board_quad: np.ndarray,
    frame_shape,
):
    dst = canonical_corners(
        canonical_mask.shape[0]
    )

    H_inv = cv2.getPerspectiveTransform(
        dst,
        board_quad.astype(np.float32),
    )

    return cv2.warpPerspective(
        canonical_mask,
        H_inv,
        (
            frame_shape[1],
            frame_shape[0],
        ),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )


def make_canonical_valid_mask(
    marker_map: dict[int, np.ndarray],
    H: np.ndarray,
    size: int,
    args: argparse.Namespace,
):
    valid = np.full(
        (size, size),
        255,
        dtype=np.uint8,
    )

    # board 가장자리 제거
    b = int(max(0, args.border_ignore_px))

    if b > 0:
        valid[:b, :] = 0
        valid[-b:, :] = 0
        valid[:, :b] = 0
        valid[:, -b:] = 0

    # ArUco marker polygon을 canonical 좌표로 옮겨 제외
    for marker_id in REQUIRED_IDS:
        if marker_id not in marker_map:
            continue

        pts = marker_map[marker_id].reshape(-1, 1, 2)

        canonical_pts = cv2.perspectiveTransform(
            pts.astype(np.float32),
            H,
        ).reshape(-1, 2)

        marker_mask = np.zeros_like(valid)

        cv2.fillConvexPoly(
            marker_mask,
            canonical_pts.astype(np.int32),
            255,
        )

        if args.marker_ignore_px > 0:
            k = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE,
                (
                    2 * args.marker_ignore_px + 1,
                    2 * args.marker_ignore_px + 1,
                ),
            )

            marker_mask = cv2.dilate(
                marker_mask,
                k,
                iterations=1,
            )

        valid[
            marker_mask > 0
        ] = 0

    return valid


# =============================================================================
# Background Model
# =============================================================================

def build_background_model(
    pipeline: Pipeline,
    align_filter: AlignFilter,
    detector,
    args: argparse.Namespace,
):
    """
    빈 판을 canonical 좌표로 30장 모아
      median LAB
      MAD -> sigma
    를 pixel-wise로 생성한다.
    """
    canonical_frames = []
    marker_maps = []
    quads = []

    attempts = 0

    print("=" * 80)
    print(
        f"[BACKGROUND MODEL] 빈 판 {args.background_frames}장 수집 시작"
    )
    print("수집 중에는 판/카메라/Z축을 움직이지 마세요.")

    while (
        len(canonical_frames) < args.background_frames
        and attempts < args.background_max_attempts
    ):
        attempts += 1

        frame, _ = wait_for_aligned_pair(
            pipeline,
            align_filter,
        )

        marker_map = detect_markers(
            frame,
            detector,
        )

        quad = get_board_outer_quad(
            marker_map
        )

        if quad is None:
            continue

        canonical, _ = warp_board_to_canonical(
            frame,
            quad,
            args.canonical_size,
        )

        canonical_frames.append(
            canonical
        )

        marker_maps.append(
            marker_map
        )

        quads.append(
            quad.copy()
        )

        print(
            f"\r  valid {len(canonical_frames):02d}/{args.background_frames} "
            f"(attempt {attempts})",
            end="",
            flush=True,
        )

    print()

    if len(canonical_frames) < args.background_frames:
        raise RuntimeError(
            "Background model 수집 실패: "
            f"{len(canonical_frames)}/{args.background_frames}장만 확보됨. "
            "ArUco 검출 상태를 확인하세요."
        )

    stack_bgr = np.stack(
        canonical_frames,
        axis=0,
    ).astype(np.uint8)

    # Canonical median reference
    median_bgr = np.median(
        stack_bgr,
        axis=0,
    ).astype(np.uint8)

    # LAB stack
    lab_frames = [
        cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2LAB,
        )
        for frame in canonical_frames
    ]

    stack_lab = np.stack(
        lab_frames,
        axis=0,
    ).astype(np.uint8)

    median_lab = np.median(
        stack_lab,
        axis=0,
    ).astype(np.float32)

    sigma_lab = np.zeros_like(
        median_lab,
        dtype=np.float32,
    )

    # robust sigma ≈ 1.4826 * MAD
    for channel in range(3):
        values = stack_lab[
            :,
            :,
            :,
            channel
        ].astype(np.float32)

        med = median_lab[
            :,
            :,
            channel
        ]

        mad = np.median(
            np.abs(
                values - med[None, :, :]
            ),
            axis=0,
        )

        sigma_lab[
            :,
            :,
            channel
        ] = np.maximum(
            1.4826 * mad,
            args.noise_floor,
        )

    # 기준 quad는 수집된 quad의 median
    reference_quad = np.median(
        np.stack(quads, axis=0),
        axis=0,
    ).astype(np.float32)

    # marker ignore mask는 첫 정상 frame의 marker map 기준
    _, H_ref = warp_board_to_canonical(
        canonical_frames[0],  # H만 필요하지 않지만 API 통일용 아님
        canonical_corners(args.canonical_size),
        args.canonical_size,
    )
    # 위 H_ref는 identity가 되어버리므로 실제 frame->canonical H를 다시 계산
    actual_H_ref = cv2.getPerspectiveTransform(
        quads[0].astype(np.float32),
        canonical_corners(args.canonical_size),
    )

    valid_mask = make_canonical_valid_mask(
        marker_maps[0],
        actual_H_ref,
        args.canonical_size,
        args,
    )

    print("[BACKGROUND MODEL] 생성 완료")
    print("=" * 80)

    return {
        "median_bgr": median_bgr,
        "median_lab": median_lab,
        "sigma_lab": sigma_lab,
        "reference_quad": reference_quad,
        "valid_mask": valid_mask,
    }


def compute_background_candidate(
    canonical_bgr: np.ndarray,
    model: dict,
    args: argparse.Namespace,
):
    current_lab = cv2.cvtColor(
        canonical_bgr,
        cv2.COLOR_BGR2LAB,
    ).astype(np.float32)

    median_lab = model[
        "median_lab"
    ]

    sigma_lab = model[
        "sigma_lab"
    ]

    dL = np.abs(
        current_lab[:, :, 0]
        - median_lab[:, :, 0]
    )

    dA = np.abs(
        current_lab[:, :, 1]
        - median_lab[:, :, 1]
    )

    dB = np.abs(
        current_lab[:, :, 2]
        - median_lab[:, :, 2]
    )

    sigma_L = sigma_lab[:, :, 0]

    sigma_C = np.sqrt(
        sigma_lab[:, :, 1] ** 2
        + sigma_lab[:, :, 2] ** 2
    )

    dC = np.sqrt(
        dA ** 2
        + dB ** 2
    )

    threshold_L = np.maximum(
        args.abs_luma_threshold,
        args.noise_sigma_mult
        * sigma_L,
    )

    threshold_C = np.maximum(
        args.abs_chroma_threshold,
        args.noise_sigma_mult
        * sigma_C,
    )

    changed = (
        (dL >= threshold_L)
        | (dC >= threshold_C)
    )

    candidate = np.zeros(
        canonical_bgr.shape[:2],
        dtype=np.uint8,
    )

    candidate[
        changed
    ] = 255

    candidate = cv2.bitwise_and(
        candidate,
        model["valid_mask"],
    )

    return (
        candidate,
        dL,
        dC,
        threshold_L,
        threshold_C,
    )


# =============================================================================
# Depth plane / object-height fusion
# =============================================================================

def temporal_median_depth(
    depth_frames: list[np.ndarray],
) -> Optional[np.ndarray]:
    if not depth_frames:
        return None

    stack = np.stack(
        depth_frames,
        axis=0,
    ).astype(np.float32)

    valid = stack > 0

    safe = np.where(
        valid,
        stack,
        np.nan,
    )

    with np.errstate(
        invalid="ignore",
    ):
        median = np.nanmedian(
            safe,
            axis=0,
        )

    median = np.where(
        np.isfinite(median),
        median,
        0.0,
    ).astype(np.float32)

    return median


def warp_depth_to_canonical(
    depth_mm: np.ndarray,
    H: np.ndarray,
    size: int,
) -> np.ndarray:
    return cv2.warpPerspective(
        depth_mm.astype(np.float32),
        H,
        (
            size,
            size,
        ),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )


def make_plane_ring_mask(
    valid_mask: np.ndarray,
    fraction: float,
) -> np.ndarray:
    h, w = valid_mask.shape

    fraction = float(
        np.clip(
            fraction,
            0.05,
            0.45,
        )
    )

    px = int(
        round(
            min(
                h,
                w,
            )
            * fraction
        )
    )

    ring = np.zeros_like(
        valid_mask
    )

    ring[
        :px,
        :
    ] = 255

    ring[
        -px:,
        :
    ] = 255

    ring[
        :,
        :px
    ] = 255

    ring[
        :,
        -px:
    ] = 255

    return cv2.bitwise_and(
        ring,
        valid_mask,
    )


def fit_inverse_depth_plane_ransac(
    depth_mm: np.ndarray,
    sample_mask: np.ndarray,
    args: argparse.Namespace,
):
    """
    실제 3D 평면은 pinhole 영상에서 1/Z가 image x,y에 대해 선형이다.

      1/Z = a*x + b*y + c

    따라서 단순 Z=ax+by+c보다 플랫폼 Pitch/Roll이 큰 경우에 더 적합하다.

    반환:
      predicted_depth_mm
      inlier_ratio
      residual_median_mm
    """
    valid = (
        (sample_mask > 0)
        & (depth_mm >= args.depth_min_mm)
        & (depth_mm <= args.depth_max_mm)
    )

    ys, xs = np.where(
        valid
    )

    if len(xs) < args.depth_min_plane_points:
        return None, 0.0, float("inf")

    rng = np.random.default_rng(
        42
    )

    if len(xs) > args.depth_plane_max_points:
        indices = rng.choice(
            len(xs),
            size=args.depth_plane_max_points,
            replace=False,
        )

        xs = xs[indices]
        ys = ys[indices]

    z = depth_mm[
        ys,
        xs,
    ].astype(np.float64)

    h, w = depth_mm.shape

    xn = (
        xs.astype(np.float64)
        / max(
            1.0,
            float(w - 1),
        )
        * 2.0
        - 1.0
    )

    yn = (
        ys.astype(np.float64)
        / max(
            1.0,
            float(h - 1),
        )
        * 2.0
        - 1.0
    )

    A = np.column_stack(
        [
            xn,
            yn,
            np.ones_like(
                xn
            ),
        ]
    )

    invz = (
        1.0
        / np.maximum(
            z,
            1e-6,
        )
    )

    best_inliers = None
    best_count = 0

    n = len(z)

    for _ in range(
        max(
            20,
            int(args.depth_plane_ransac_iters),
        )
    ):
        sample_idx = rng.choice(
            n,
            size=3,
            replace=False,
        )

        try:
            coeff = np.linalg.solve(
                A[sample_idx],
                invz[sample_idx],
            )
        except np.linalg.LinAlgError:
            continue

        pred_inv = A @ coeff

        valid_pred = pred_inv > 1e-9

        pred_z = np.full_like(
            z,
            np.inf,
            dtype=np.float64,
        )

        pred_z[
            valid_pred
        ] = (
            1.0
            / pred_inv[
                valid_pred
            ]
        )

        residual = np.abs(
            pred_z
            - z
        )

        inliers = (
            residual
            <= args.depth_plane_ransac_mm
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
        or best_count < args.depth_min_plane_points
    ):
        return None, 0.0, float("inf")

    try:
        coeff, *_ = np.linalg.lstsq(
            A[
                best_inliers
            ],
            invz[
                best_inliers
            ],
            rcond=None,
        )
    except np.linalg.LinAlgError:
        return None, 0.0, float("inf")

    # full canonical plane
    yy, xx = np.mgrid[
        0:h,
        0:w,
    ]

    x_full = (
        xx.astype(np.float64)
        / max(
            1.0,
            float(w - 1),
        )
        * 2.0
        - 1.0
    )

    y_full = (
        yy.astype(np.float64)
        / max(
            1.0,
            float(h - 1),
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

    okay = pred_inv_full > 1e-9

    plane_depth[
        okay
    ] = (
        1.0
        / pred_inv_full[
            okay
        ]
    ).astype(np.float32)

    sample_pred = (
        A
        @ coeff
    )

    sample_pred_z = np.where(
        sample_pred > 1e-9,
        1.0 / sample_pred,
        np.inf,
    )

    sample_residual = np.abs(
        sample_pred_z
        - z
    )

    final_inliers = (
        np.isfinite(sample_residual)
        & (
            sample_residual
            <= args.depth_plane_ransac_mm
        )
    )

    inlier_ratio = float(
        np.mean(
            final_inliers
        )
    )

    # 중요:
    # plane 품질은 물체/벽 같은 outlier까지 포함한 전체 residual median이 아니라
    # RANSAC이 실제 "판 평면"으로 인정한 inlier들의 residual로 평가해야 한다.
    if np.any(
        final_inliers
    ):
        residual_median = float(
            np.median(
                sample_residual[
                    final_inliers
                ]
            )
        )
    else:
        residual_median = float("inf")

    return (
        plane_depth,
        inlier_ratio,
        residual_median,
    )


def make_depth_object_mask(
    canonical_depth_mm: np.ndarray,
    plane_depth_mm: np.ndarray,
    valid_mask: np.ndarray,
    args: argparse.Namespace,
):
    valid_depth = (
        (canonical_depth_mm >= args.depth_min_mm)
        & (canonical_depth_mm <= args.depth_max_mm)
        & (plane_depth_mm > 0)
        & (valid_mask > 0)
    )

    height_mm = np.zeros_like(
        canonical_depth_mm,
        dtype=np.float32,
    )

    height_mm[
        valid_depth
    ] = (
        plane_depth_mm[
            valid_depth
        ]
        - canonical_depth_mm[
            valid_depth
        ]
    )

    object_pixels = (
        valid_depth
        & (
            height_mm
            >= args.depth_height_threshold_mm
        )
        & (
            height_mm
            <= args.depth_max_object_height_mm
        )
    )

    mask = np.zeros_like(
        valid_mask
    )

    mask[
        object_pixels
    ] = 255

    k_open = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (
            odd(args.depth_open_size),
            odd(args.depth_open_size),
        ),
    )

    k_close = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (
            odd(args.depth_close_size),
            odd(args.depth_close_size),
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
        iterations=1,
    )

    mask = cv2.bitwise_and(
        mask,
        valid_mask,
    )

    if args.depth_seed_dilate_px > 0:
        k = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (
                2
                * args.depth_seed_dilate_px
                + 1,
                2
                * args.depth_seed_dilate_px
                + 1,
            ),
        )

        strong_seed = cv2.dilate(
            mask,
            k,
            iterations=1,
        )

        strong_seed = cv2.bitwise_and(
            strong_seed,
            valid_mask,
        )
    else:
        strong_seed = mask.copy()

    return (
        mask,
        strong_seed,
        height_mm,
        valid_depth.astype(
            np.uint8
        )
        * 255,
    )


def render_depth_debug(
    depth_mm: Optional[np.ndarray],
) -> Optional[np.ndarray]:
    if depth_mm is None:
        return None

    valid = depth_mm > 0

    if not np.any(
        valid
    ):
        return np.zeros(
            (
                depth_mm.shape[0],
                depth_mm.shape[1],
                3,
            ),
            dtype=np.uint8,
        )

    vals = depth_mm[
        valid
    ]

    lo = float(
        np.percentile(
            vals,
            2,
        )
    )

    hi = float(
        np.percentile(
            vals,
            98,
        )
    )

    if hi <= lo:
        hi = lo + 1.0

    norm = np.zeros_like(
        depth_mm,
        dtype=np.float32,
    )

    norm[
        valid
    ] = (
        depth_mm[
            valid
        ]
        - lo
    ) / (
        hi
        - lo
    )

    norm = np.clip(
        norm,
        0.0,
        1.0,
    )

    gray = (
        norm
        * 255.0
    ).astype(np.uint8)

    color = cv2.applyColorMap(
        gray,
        cv2.COLORMAP_JET,
    )

    color[
        ~valid
    ] = 0

    return color


def render_height_debug(
    height_mm: Optional[np.ndarray],
    max_height_mm: float = 40.0,
) -> Optional[np.ndarray]:
    if height_mm is None:
        return None

    positive = np.clip(
        height_mm,
        0.0,
        max_height_mm,
    )

    gray = (
        positive
        / max(
            1e-6,
            max_height_mm,
        )
        * 255.0
    ).astype(np.uint8)

    return cv2.applyColorMap(
        gray,
        cv2.COLORMAP_TURBO,
    )


# =============================================================================
# Seed-based GrabCut refinement
# =============================================================================

def choose_central_component(
    binary_mask: np.ndarray,
    min_area: int,
    max_area_ratio: float,
) -> Optional[np.ndarray]:
    """
    큰 component이면서 canonical board 중앙에 가까운 component를 선택한다.
    """
    contours, _ = cv2.findContours(
        binary_mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    if not contours:
        return None

    h, w = binary_mask.shape

    center = np.asarray(
        [w / 2.0, h / 2.0],
        dtype=np.float32,
    )

    diag = max(
        1.0,
        float(np.hypot(w, h)),
    )

    max_area = (
        float(h * w)
        * float(max_area_ratio)
    )

    scored = []

    for contour in contours:
        area = float(
            cv2.contourArea(contour)
        )

        if area < min_area:
            continue

        if area > max_area:
            continue

        x, y, bw, bh = cv2.boundingRect(
            contour
        )

        component_center = np.asarray(
            [
                x + bw / 2.0,
                y + bh / 2.0,
            ],
            dtype=np.float32,
        )

        distance = float(
            np.linalg.norm(
                component_center - center
            )
        ) / diag

        score = (
            area
            / (
                1.0
                + 1.6 * distance
            )
        )

        scored.append(
            (
                score,
                contour,
            )
        )

    if not scored:
        return None

    scored.sort(
        key=lambda item: item[0],
        reverse=True,
    )

    mask = np.zeros_like(
        binary_mask
    )

    cv2.drawContours(
        mask,
        [scored[0][1]],
        -1,
        255,
        thickness=cv2.FILLED,
    )

    return mask


def build_density_support(
    candidate: np.ndarray,
    valid_mask: np.ndarray,
    args: argparse.Namespace,
):
    """
    Background candidate는 물체 안에서도 반사/색 차이에 따라 조각조각 끊길 수 있다.

    물체 영역에서는 changed pixel이 '밀집'되어 있고,
    판의 노이즈는 대체로 sparse하다는 점을 이용한다.

    local foreground density -> rough object envelope
    """
    binary = (
        candidate > 0
    ).astype(np.float32)

    window = odd(
        args.density_window
    )

    density = cv2.boxFilter(
        binary,
        ddepth=-1,
        ksize=(
            window,
            window,
        ),
        normalize=True,
        borderType=cv2.BORDER_REFLECT,
    )

    density_mask = np.zeros_like(
        candidate
    )

    density_mask[
        density >= args.density_threshold
    ] = 255

    density_mask = cv2.bitwise_and(
        density_mask,
        valid_mask,
    )

    k_close = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (
            odd(args.support_close_size),
            odd(args.support_close_size),
        ),
    )

    density_mask = cv2.morphologyEx(
        density_mask,
        cv2.MORPH_CLOSE,
        k_close,
        iterations=1,
    )

    # 너무 작은 speck 제거
    k_open = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (7, 7),
    )

    density_mask = cv2.morphologyEx(
        density_mask,
        cv2.MORPH_OPEN,
        k_open,
        iterations=1,
    )

    support = choose_central_component(
        density_mask,
        min_area=max(
            1500,
            int(args.min_object_area * 0.35),
        ),
        max_area_ratio=args.max_support_area_ratio,
    )

    return support, density


def select_component_by_seed_overlap(
    mask: np.ndarray,
    seed_mask: np.ndarray,
) -> Optional[np.ndarray]:
    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    if not contours:
        return None

    best = None
    best_score = -1.0

    for contour in contours:
        component = np.zeros_like(
            mask
        )

        cv2.drawContours(
            component,
            [contour],
            -1,
            255,
            thickness=cv2.FILLED,
        )

        area = np.count_nonzero(
            component > 0
        )

        if area == 0:
            continue

        overlap = np.count_nonzero(
            (component > 0)
            & (seed_mask > 0)
        )

        # seed overlap를 가장 중요하게,
        # 동률이면 큰 component가 이기도록 약한 area term 추가
        score = (
            float(overlap)
            + 0.03 * float(area)
        )

        if score > best_score:
            best_score = score
            best = component

    return best


def seed_grabcut_refine(
    canonical_bgr: np.ndarray,
    candidate: np.ndarray,
    dL: np.ndarray,
    dC: np.ndarray,
    threshold_L: np.ndarray,
    threshold_C: np.ndarray,
    valid_mask: np.ndarray,
    args: argparse.Namespace,
    extra_candidate: Optional[np.ndarray] = None,
    extra_strong_seed: Optional[np.ndarray] = None,
):
    """
    단계:
      1) fragmented Background Difference -> local density support
      2) threshold를 강하게 넘은 pixel -> definite foreground seed
      3) support 내부 -> probable foreground
      4) support 바깥 dilation ring -> probable background
      5) 나머지 -> definite background
      6) GrabCut으로 실제 영상 경계를 따라 object mask 복원
    """

    fused_candidate = candidate.copy()

    if extra_candidate is not None:
        fused_candidate = cv2.bitwise_or(
            fused_candidate,
            extra_candidate,
        )

    support, density = build_density_support(
        fused_candidate,
        valid_mask,
        args,
    )

    if support is None:
        return (
            None,
            None,
            None,
            None,
            density,
        )

    # 강한 foreground seed
    strong_changed = (
        (
            dL
            >= (
                threshold_L
                * args.strong_seed_mult
            )
        )
        | (
            dC
            >= (
                threshold_C
                * args.strong_seed_mult
            )
        )
    )

    strong_seed = np.zeros_like(
        candidate
    )

    strong_seed[
        strong_changed
    ] = 255

    strong_seed = cv2.bitwise_and(
        strong_seed,
        support,
    )

    strong_seed = cv2.bitwise_and(
        strong_seed,
        valid_mask,
    )

    # 유효 Depth에서 판보다 위에 있다고 확인된 pixel은 매우 강한 FG seed.
    # Depth invalid는 절대 background로 쓰지 않는다.
    if extra_strong_seed is not None:
        strong_seed = cv2.bitwise_or(
            strong_seed,
            extra_strong_seed,
        )

        strong_seed = cv2.bitwise_and(
            strong_seed,
            valid_mask,
        )

    # strong seed가 너무 적으면 일반 candidate 중 support 내부 pixel로 완화
    if (
        np.count_nonzero(
            strong_seed > 0
        )
        < args.min_strong_seed_area
    ):
        strong_seed = cv2.bitwise_and(
            fused_candidate,
            support,
        )

    # seed speck 정리
    seed_open = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (3, 3),
    )

    strong_seed = cv2.morphologyEx(
        strong_seed,
        cv2.MORPH_OPEN,
        seed_open,
        iterations=1,
    )

    if (
        np.count_nonzero(
            strong_seed > 0
        )
        < 50
    ):
        return (
            None,
            strong_seed,
            support,
            None,
            density,
        )

    # GrabCut이 support 경계 밖으로 실제 object edge까지 약간 움직일 여유
    dilate_px = max(
        1,
        int(args.support_dilate_px),
    )

    dilate_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (
            2 * dilate_px + 1,
            2 * dilate_px + 1,
        ),
    )

    support_outer = cv2.dilate(
        support,
        dilate_kernel,
        iterations=1,
    )

    support_outer = cv2.bitwise_and(
        support_outer,
        valid_mask,
    )

    # GrabCut label map
    gc_mask = np.full(
        candidate.shape,
        cv2.GC_BGD,
        dtype=np.uint8,
    )

    # outer band 안쪽은 우선 probable background
    gc_mask[
        support_outer > 0
    ] = cv2.GC_PR_BGD

    # density support 안쪽은 probable foreground
    gc_mask[
        support > 0
    ] = cv2.GC_PR_FGD

    # strong background-difference pixel은 definite foreground
    gc_mask[
        strong_seed > 0
    ] = cv2.GC_FGD

    # board valid 밖은 절대 background
    gc_mask[
        valid_mask == 0
    ] = cv2.GC_BGD

    bg_model = np.zeros(
        (1, 65),
        dtype=np.float64,
    )

    fg_model = np.zeros(
        (1, 65),
        dtype=np.float64,
    )

    try:
        cv2.grabCut(
            canonical_bgr,
            gc_mask,
            None,
            bg_model,
            fg_model,
            max(
                1,
                int(args.grabcut_iterations),
            ),
            cv2.GC_INIT_WITH_MASK,
        )
    except cv2.error:
        return (
            None,
            strong_seed,
            support,
            gc_mask,
            density,
        )

    grabcut_binary = np.where(
        (
            gc_mask == cv2.GC_FGD
        )
        | (
            gc_mask == cv2.GC_PR_FGD
        ),
        255,
        0,
    ).astype(np.uint8)

    grabcut_binary = cv2.bitwise_and(
        grabcut_binary,
        valid_mask,
    )

    # seed와 실제로 겹치는 object component만 사용
    object_mask = select_component_by_seed_overlap(
        grabcut_binary,
        strong_seed,
    )

    if object_mask is None:
        return (
            None,
            strong_seed,
            support,
            gc_mask,
            density,
        )

    object_mask = fill_small_holes(
        object_mask,
        args.max_hole_area,
    )

    return (
        object_mask,
        strong_seed,
        support,
        gc_mask,
        density,
    )


def visualize_grabcut_labels(
    gc_mask: Optional[np.ndarray],
) -> Optional[np.ndarray]:
    """
    디버그용 grayscale:
      BGD    0
      PR_BGD 85
      PR_FGD 170
      FGD    255
    """
    if gc_mask is None:
        return None

    vis = np.zeros_like(
        gc_mask,
        dtype=np.uint8,
    )

    vis[
        gc_mask == cv2.GC_PR_BGD
    ] = 85

    vis[
        gc_mask == cv2.GC_PR_FGD
    ] = 170

    vis[
        gc_mask == cv2.GC_FGD
    ] = 255

    return vis


# =============================================================================
# Mask cleanup / object selection
# =============================================================================

def odd(v: int) -> int:
    v = max(1, int(v))
    return v if v % 2 == 1 else v + 1


def fill_small_holes(
    mask: np.ndarray,
    max_hole_area: int,
) -> np.ndarray:
    """
    외부 배경과 연결되지 않은 내부 hole 중 작은 것만 채운다.
    큰 실제 관통구멍/오목부는 보존한다.
    """
    binary = (
        mask > 0
    ).astype(np.uint8)

    inv = (
        1 - binary
    ).astype(np.uint8)

    # 외부 배경 flood fill
    flood = inv.copy()

    h, w = flood.shape

    flood_mask = np.zeros(
        (h + 2, w + 2),
        dtype=np.uint8,
    )

    cv2.floodFill(
        flood,
        flood_mask,
        (0, 0),
        2,
    )

    # 값 1로 남은 부분 = 내부 hole
    holes = np.where(
        flood == 1,
        255,
        0,
    ).astype(np.uint8)

    num_labels, labels, stats, _ = (
        cv2.connectedComponentsWithStats(
            holes,
            connectivity=8,
        )
    )

    result = mask.copy()

    for label in range(
        1,
        num_labels,
    ):
        area = int(
            stats[
                label,
                cv2.CC_STAT_AREA
            ]
        )

        if area <= max_hole_area:
            result[
                labels == label
            ] = 255

    return result


def select_object_component(
    candidate: np.ndarray,
    args: argparse.Namespace,
):
    k_open = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (
            odd(args.open_size),
            odd(args.open_size),
        ),
    )

    k_close = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (
            odd(args.close_size),
            odd(args.close_size),
        ),
    )

    clean = cv2.morphologyEx(
        candidate,
        cv2.MORPH_OPEN,
        k_open,
        iterations=1,
    )

    clean = cv2.morphologyEx(
        clean,
        cv2.MORPH_CLOSE,
        k_close,
        iterations=args.close_iterations,
    )

    contours, _ = cv2.findContours(
        clean,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    valid = []

    h, w = clean.shape

    center = np.asarray(
        [
            w / 2.0,
            h / 2.0,
        ],
        dtype=np.float32,
    )

    diag = float(
        np.hypot(
            w,
            h,
        )
    )

    for contour in contours:
        area = float(
            cv2.contourArea(contour)
        )

        if area < args.min_object_area:
            continue

        x, y, bw, bh = cv2.boundingRect(
            contour
        )

        c = np.asarray(
            [
                x + bw / 2.0,
                y + bh / 2.0,
            ],
            dtype=np.float32,
        )

        distance = float(
            np.linalg.norm(
                c - center
            )
        ) / max(
            diag,
            1.0,
        )

        # 큰 component + board 중앙에 가까운 component 우선
        score = (
            area
            / (
                1.0
                + 1.4 * distance
            )
        )

        valid.append(
            (
                score,
                contour,
            )
        )

    if not valid:
        return None, clean

    valid.sort(
        key=lambda item: item[0],
        reverse=True,
    )

    contour = valid[0][1]

    object_mask = np.zeros_like(
        clean
    )

    cv2.drawContours(
        object_mask,
        [contour],
        -1,
        255,
        thickness=cv2.FILLED,
    )

    object_mask = fill_small_holes(
        object_mask,
        args.max_hole_area,
    )

    return object_mask, clean


def bbox_from_mask(mask: np.ndarray):
    ys, xs = np.where(
        mask > 0
    )

    if len(xs) == 0:
        return None

    x1 = int(xs.min())
    y1 = int(ys.min())
    x2 = int(xs.max())
    y2 = int(ys.max())

    return (
        x1,
        y1,
        x2 - x1 + 1,
        y2 - y1 + 1,
    )


# =============================================================================
# Current-frame dynamic ROI tracking
# =============================================================================

def dilate_mask_px(
    mask: np.ndarray,
    px: int,
) -> np.ndarray:
    px = max(
        0,
        int(px),
    )

    if px == 0:
        return mask.copy()

    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (
            2 * px + 1,
            2 * px + 1,
        ),
    )

    return cv2.dilate(
        mask,
        kernel,
        iterations=1,
    )


def erode_mask_px(
    mask: np.ndarray,
    px: int,
) -> np.ndarray:
    px = max(
        0,
        int(px),
    )

    if px == 0:
        return mask.copy()

    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (
            2 * px + 1,
            2 * px + 1,
        ),
    )

    return cv2.erode(
        mask,
        kernel,
        iterations=1,
    )


def build_tracking_regions(
    previous_mask: np.ndarray,
    args: argparse.Namespace,
):
    """
    Z값/모터 이동량을 사용하지 않는다.

    이전 ROI의 현재 영상상 크기만 보고:
      - search area
      - 판 plane fitting ring
    을 만든다.

    플랫폼이 올라가 물체가 점점 커져도,
    매 프레임 새 ROI를 이전 ROI로 갱신하므로 search area도 함께 커진다.
    """
    previous_area = int(
        np.count_nonzero(
            previous_mask > 0
        )
    )

    object_scale = float(
        np.sqrt(
            max(
                1,
                previous_area,
            )
        )
    )

    search_margin = int(
        np.clip(
            object_scale
            * float(args.track_search_scale),
            int(args.track_search_min_px),
            int(args.track_search_max_px),
        )
    )

    search_mask = dilate_mask_px(
        previous_mask,
        search_margin,
    )

    plane_outer = dilate_mask_px(
        search_mask,
        int(args.track_plane_ring_px),
    )

    # 물체 후보 search area 바로 바깥만 판 fitting에 사용.
    # 물체가 조금 search 밖으로 나가도 RANSAC이 일부 outlier를 제거한다.
    plane_ring = cv2.bitwise_and(
        plane_outer,
        cv2.bitwise_not(
            search_mask
        ),
    )

    return (
        search_mask,
        plane_ring,
        search_margin,
    )


def track_grabcut_current_frame(
    frame: np.ndarray,
    previous_mask: np.ndarray,
    search_mask: np.ndarray,
    depth_candidate: Optional[np.ndarray],
    depth_strong_seed: Optional[np.ndarray],
    args: argparse.Namespace,
):
    """
    LOW-Z background model을 HIGH-Z에서 억지로 재사용하지 않는다.

    현재 프레임의:
      - 이전 ROI core
      - 현재 Depth object seed
      - 현재 RGB 경계
    만 이용해 GrabCut을 다시 수행한다.
    """
    gc_mask = np.full(
        frame.shape[:2],
        cv2.GC_BGD,
        dtype=np.uint8,
    )

    # search 내부는 우선 probable background
    gc_mask[
        search_mask > 0
    ] = cv2.GC_PR_BGD

    prev_prob = dilate_mask_px(
        previous_mask,
        int(args.track_prev_prob_dilate_px),
    )

    prev_prob = cv2.bitwise_and(
        prev_prob,
        search_mask,
    )

    # 이전 ROI 주변은 probable foreground
    gc_mask[
        prev_prob > 0
    ] = cv2.GC_PR_FGD

    # 현재 Depth에서 물체라고 확인된 영역도 probable foreground
    if depth_candidate is not None:
        gc_mask[
            depth_candidate > 0
        ] = cv2.GC_PR_FGD

    previous_core = erode_mask_px(
        previous_mask,
        int(args.track_prev_core_erode_px),
    )

    # 이전 ROI 내부 core는 현재 프레임에서도 물체일 가능성이 높으므로 definite FG
    gc_mask[
        previous_core > 0
    ] = cv2.GC_FGD

    # 현재 Depth strong seed는 definite FG
    if depth_strong_seed is not None:
        gc_mask[
            depth_strong_seed > 0
        ] = cv2.GC_FGD

    # search 밖은 확실한 background
    gc_mask[
        search_mask == 0
    ] = cv2.GC_BGD

    strong_reference = previous_core.copy()

    if depth_strong_seed is not None:
        strong_reference = cv2.bitwise_or(
            strong_reference,
            depth_strong_seed,
        )

    if np.count_nonzero(
        strong_reference > 0
    ) < 50:
        return None, gc_mask

    bg_model = np.zeros(
        (1, 65),
        dtype=np.float64,
    )

    fg_model = np.zeros(
        (1, 65),
        dtype=np.float64,
    )

    try:
        cv2.grabCut(
            frame,
            gc_mask,
            None,
            bg_model,
            fg_model,
            max(
                1,
                int(args.grabcut_iterations),
            ),
            cv2.GC_INIT_WITH_MASK,
        )
    except cv2.error:
        return None, gc_mask

    binary = np.where(
        (
            gc_mask == cv2.GC_FGD
        )
        | (
            gc_mask == cv2.GC_PR_FGD
        ),
        255,
        0,
    ).astype(np.uint8)

    binary = cv2.bitwise_and(
        binary,
        search_mask,
    )

    object_mask = select_component_by_seed_overlap(
        binary,
        strong_reference,
    )

    return (
        object_mask,
        gc_mask,
    )


def tracking_confidence_and_max_growth(
    raw_depth_valid_ratio: float,
    plane_inlier_ratio: float,
    plane_residual_mm: float,
    args: argparse.Namespace,
):
    """
    Z 이동량은 사용하지 않는다.

    현재 센서 신뢰도만 보고 한 프레임에서 ROI가 얼마나 커질 수 있는지 결정한다.
    """
    residual_ok_high = (
        np.isfinite(plane_residual_mm)
        and plane_residual_mm
        <= float(args.track_high_plane_residual_mm)
    )

    residual_ok_medium = (
        np.isfinite(plane_residual_mm)
        and plane_residual_mm
        <= float(args.track_medium_plane_residual_mm)
    )

    if (
        raw_depth_valid_ratio
        >= float(args.track_high_depth_valid)
        and plane_inlier_ratio
        >= float(args.track_high_plane_inlier)
        and residual_ok_high
    ):
        return (
            "HIGH",
            float(args.track_high_max_area_ratio),
            True,
        )

    if (
        raw_depth_valid_ratio
        >= float(args.track_medium_depth_valid)
        and plane_inlier_ratio
        >= float(args.track_medium_plane_inlier)
        and residual_ok_medium
    ):
        return (
            "MEDIUM",
            float(args.track_medium_max_area_ratio),
            True,
        )

    return (
        "LOW",
        float(args.track_max_area_ratio),
        False,
    )


def validate_tracking_update(
    previous_mask: np.ndarray,
    current_mask: np.ndarray,
    raw_depth_valid_ratio: float,
    plane_inlier_ratio: float,
    plane_residual_mm: float,
    args: argparse.Namespace,
):
    prev_area = float(
        np.count_nonzero(
            previous_mask > 0
        )
    )

    current_area = float(
        np.count_nonzero(
            current_mask > 0
        )
    )

    if (
        prev_area <= 0
        or current_area <= 0
    ):
        return (
            False,
            0.0,
            0.0,
            "LOW",
            float(args.track_max_area_ratio),
        )

    area_ratio = (
        current_area
        / prev_area
    )

    intersection = float(
        np.count_nonzero(
            (previous_mask > 0)
            & (current_mask > 0)
        )
    )

    overlap_ratio = (
        intersection
        / prev_area
    )

    (
        confidence,
        allowed_max_area_ratio,
        sensor_ok,
    ) = tracking_confidence_and_max_growth(
        raw_depth_valid_ratio,
        plane_inlier_ratio,
        plane_residual_mm,
        args,
    )

    good = (
        sensor_ok
        and float(args.track_min_area_ratio)
        <= area_ratio
        <= allowed_max_area_ratio
        and overlap_ratio
        >= float(args.track_min_overlap_ratio)
    )

    return (
        good,
        area_ratio,
        overlap_ratio,
        confidence,
        allowed_max_area_ratio,
    )



# =============================================================================
# LOW-Z final silhouette / inspection mask
# =============================================================================

def finalize_object_silhouette(
    raw_mask: Optional[np.ndarray],
    args: argparse.Namespace,
):
    """
    목적:
      - GrabCut 내부 hole / depth invalid를 무시
      - 물체의 '가장 큰 외곽 contour'만 사용
      - 작은 edge 요철만 완화
      - 내부는 전부 채운 최종 Object Mask 생성
      - 실제 AE 검사에는 외곽에서 일정 거리 안쪽의 Inspection Mask 사용

    ConvexHull은 사용하지 않는다.
    따라서 실제 오목 형상을 무조건 볼록하게 메우지 않는다.
    """
    if raw_mask is None:
        return None, None, None

    binary = np.where(
        raw_mask > 0,
        255,
        0,
    ).astype(np.uint8)

    # 1) 작은 끊김 / 작은 움푹 패임 완화
    close_size = odd(
        args.final_close_size
    )

    if close_size > 1:
        k_close = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (
                close_size,
                close_size,
            ),
        )

        binary = cv2.morphologyEx(
            binary,
            cv2.MORPH_CLOSE,
            k_close,
            iterations=1,
        )

    # 2) 약한 blur -> threshold로 jagged edge 완화
    smooth_size = odd(
        args.final_smooth_size
    )

    if smooth_size > 1:
        blurred = cv2.GaussianBlur(
            binary,
            (
                smooth_size,
                smooth_size,
            ),
            0,
        )

        _, binary = cv2.threshold(
            blurred,
            127,
            255,
            cv2.THRESH_BINARY,
        )

    # 3) 가장 큰 외곽 contour만 사용
    contours, _ = cv2.findContours(
        binary,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_NONE,
    )

    if not contours:
        return None, None, None

    contour = max(
        contours,
        key=cv2.contourArea,
    )

    if cv2.contourArea(contour) <= 0:
        return None, None, None

    # 4) 외곽선 약한 단순화
    perimeter = cv2.arcLength(
        contour,
        True,
    )

    epsilon = max(
        0.0,
        float(args.contour_epsilon_ratio)
        * perimeter,
    )

    if epsilon > 0:
        contour = cv2.approxPolyDP(
            contour,
            epsilon,
            True,
        )

    # 5) 외곽 내부를 완전히 채운 최종 Object Mask
    final_mask = np.zeros_like(
        binary
    )

    cv2.drawContours(
        final_mask,
        [contour],
        -1,
        255,
        thickness=cv2.FILLED,
    )

    # 6) 실제 검사 영역은 경계에서 안쪽으로 erosion
    inspection_mask = final_mask.copy()

    erode_px = max(
        0,
        int(args.inspection_erode_px),
    )

    if erode_px > 0:
        k_erode = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (
                2 * erode_px + 1,
                2 * erode_px + 1,
            ),
        )

        inspection_mask = cv2.erode(
            inspection_mask,
            k_erode,
            iterations=1,
        )

    return (
        final_mask,
        inspection_mask,
        contour,
    )


# =============================================================================
# UI / Save
# =============================================================================

def draw_status(
    image: np.ndarray,
    lines,
    good=True,
):
    box_h = (
        29 * len(lines)
        + 12
    )

    cv2.rectangle(
        image,
        (0, 0),
        (1080, box_h),
        (0, 0, 0),
        -1,
    )

    for i, line in enumerate(lines):
        cv2.putText(
            image,
            line,
            (
                12,
                27 + i * 29,
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (
                (0, 255, 0)
                if (
                    i == 0
                    and good
                )
                else (
                    (0, 0, 255)
                    if i == 0
                    else (255, 255, 255)
                )
            ),
            2,
            cv2.LINE_AA,
        )


def draw_mask_contour(
    image: np.ndarray,
    mask: np.ndarray,
    color,
    thickness=4,
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


def save_results(
    output_root: Path,
    original: np.ndarray,
    overlay: np.ndarray,
    canonical: Optional[np.ndarray],
    candidate: Optional[np.ndarray],
    clean_candidate: Optional[np.ndarray],
    strong_seed: Optional[np.ndarray],
    density_support: Optional[np.ndarray],
    grabcut_labels: Optional[np.ndarray],
    depth_aligned_vis: Optional[np.ndarray],
    canonical_depth_vis: Optional[np.ndarray],
    depth_height_vis: Optional[np.ndarray],
    depth_object_mask: Optional[np.ndarray],
    depth_valid_mask: Optional[np.ndarray],
    fused_candidate: Optional[np.ndarray],
    raw_object_mask_canonical: Optional[np.ndarray],
    final_object_mask_canonical: Optional[np.ndarray],
    inspection_mask_canonical: Optional[np.ndarray],
    object_mask_frame: Optional[np.ndarray],
    inspection_mask_frame: Optional[np.ndarray],
    model: Optional[dict],
    board_quad: Optional[np.ndarray],
    depth_metrics: Optional[dict],
):
    timestamp = datetime.now().strftime(
        "%Y%m%d_%H%M%S_%f"
    )

    out = (
        output_root
        / timestamp
    )

    out.mkdir(
        parents=True,
        exist_ok=True,
    )

    cv2.imwrite(
        str(out / "original.png"),
        original,
    )

    cv2.imwrite(
        str(out / "overlay.png"),
        overlay,
    )

    if canonical is not None:
        cv2.imwrite(
            str(out / "canonical_current.png"),
            canonical,
        )

    if candidate is not None:
        cv2.imwrite(
            str(out / "background_candidate.png"),
            candidate,
        )

    if clean_candidate is not None:
        cv2.imwrite(
            str(out / "clean_candidate.png"),
            clean_candidate,
        )

    if strong_seed is not None:
        cv2.imwrite(
            str(out / "strong_foreground_seed.png"),
            strong_seed,
        )

    if density_support is not None:
        cv2.imwrite(
            str(out / "density_support.png"),
            density_support,
        )

    if grabcut_labels is not None:
        cv2.imwrite(
            str(out / "grabcut_labels.png"),
            grabcut_labels,
        )

    if depth_aligned_vis is not None:
        cv2.imwrite(
            str(out / "depth_aligned.png"),
            depth_aligned_vis,
        )

    if canonical_depth_vis is not None:
        cv2.imwrite(
            str(out / "depth_canonical.png"),
            canonical_depth_vis,
        )

    if depth_height_vis is not None:
        cv2.imwrite(
            str(out / "depth_height_map.png"),
            depth_height_vis,
        )

    if depth_object_mask is not None:
        cv2.imwrite(
            str(out / "depth_object_mask.png"),
            depth_object_mask,
        )

    if depth_valid_mask is not None:
        cv2.imwrite(
            str(out / "depth_valid_mask.png"),
            depth_valid_mask,
        )

    if fused_candidate is not None:
        cv2.imwrite(
            str(out / "rgb_depth_fused_candidate.png"),
            fused_candidate,
        )

    if raw_object_mask_canonical is not None:
        cv2.imwrite(
            str(out / "raw_grabcut_mask_canonical.png"),
            raw_object_mask_canonical,
        )

    if final_object_mask_canonical is not None:
        cv2.imwrite(
            str(out / "final_object_mask_canonical.png"),
            final_object_mask_canonical,
        )

    if inspection_mask_canonical is not None:
        cv2.imwrite(
            str(out / "inspection_mask_canonical.png"),
            inspection_mask_canonical,
        )

    if object_mask_frame is not None:
        cv2.imwrite(
            str(out / "final_object_mask_frame.png"),
            object_mask_frame,
        )

    if inspection_mask_frame is not None:
        cv2.imwrite(
            str(out / "inspection_mask_frame.png"),
            inspection_mask_frame,
        )

    if model is not None:
        cv2.imwrite(
            str(out / "background_median.png"),
            model["median_bgr"],
        )

        cv2.imwrite(
            str(out / "canonical_valid_mask.png"),
            model["valid_mask"],
        )

        np.savez_compressed(
            str(out / "background_model.npz"),
            median_lab=model["median_lab"],
            sigma_lab=model["sigma_lab"],
            reference_quad=model["reference_quad"],
            valid_mask=model["valid_mask"],
        )

    summary = [
        f"board_quad={board_quad}",
    ]

    if depth_metrics is not None:
        for key, value in depth_metrics.items():
            summary.append(
                f"{key}={value}"
            )

    (
        out
        / "summary.txt"
    ).write_text(
        "\n".join(summary)
        + "\n",
        encoding="utf-8",
    )

    print(
        f"[SAVE] {out.resolve()}"
    )


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    args = parse_args()

    pipeline = Pipeline()
    config = Config()
    started = False

    detector = create_aruco_detector()

    align_filter = AlignFilter(
        align_to_stream=OBStreamType.COLOR_STREAM
    )

    state = "WAIT_BACKGROUND"

    model = None

    latest_canonical = None
    latest_candidate = None
    latest_clean = None
    latest_strong_seed = None
    latest_density_support = None
    latest_grabcut_labels = None
    latest_raw_object_canonical = None
    latest_object_canonical = None
    latest_inspection_canonical = None
    latest_object_frame = None
    latest_inspection_frame = None
    latest_board_quad = None

    latest_depth_aligned_vis = None
    latest_canonical_depth_vis = None
    latest_depth_height_vis = None
    latest_depth_object_mask = None
    latest_depth_valid_mask = None
    latest_fused_candidate = None
    latest_depth_metrics = None

    # TRACK_OBJECT 상태에서 프레임 간 유지되는 mask
    tracked_object_frame = None
    tracked_inspection_frame = None

    depth_history = []

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
                f"[WARN] FULL_FRAME_REQUIRE 설정 실패: {exc}"
            )

        pipeline.start(
            config
        )

        started = True

        device = pipeline.get_device()

        configure_manual_camera(
            device,
            args,
        )

        configure_depth_manual(
            device,
            args,
        )

        print("=" * 88)
        print("RGB + Depth 현재프레임 기반 Dynamic ROI 테스트")
        print("")
        print("순서:")
        print("  1) LOW Z + 빈 판 + ArUco ID0~3")
        print(f"  2) B -> 빈 판 {args.background_frames}프레임 수집")
        print("     -> canonical board 좌표에서 pixel-wise Median/MAD model 생성")
        print("  3) 같은 LOW Z에서 검사체를 올림")
        print("  4) 빈 판의 정상 pixel 변화량보다 크게 달라진 영역 = foreground seed")
        print("  5) changed-pixel local density로 rough object support 생성")
        print("  6) Depth Exposure/Gain=3000/16 고정 + software D2C alignment")
        print("  7) 판 외곽 Depth로 inverse-depth plane RANSAC")
        print("  8) plane보다 위에 있는 유효 Depth pixel을 strong FG seed로 추가")
        print("  9) RGB seed + Depth seed를 GrabCut에 넣어 실제 영상 경계로 ROI 복원")
        print("")
        print("이번 코드는 LOW-Z에서 최종 외곽 ROI + 검사 안전영역까지 확정하는 테스트입니다.")
        print("PURPLE=물체 최종 외곽 / YELLOW=결함검사용 내부 안전영역")
        print("LOW-Z ROI가 안정되면 T를 눌러 현재프레임 기반 Dynamic Tracking을 시작합니다.")
        print("Tracking 중에는 Z 이동량/모터 위치를 사용하지 않습니다.")
        print("V4 PLANE-FIX: RANSAC plane 품질을 inlier residual로 평가 + adaptive tracking")
        print("")
        print("키:")
        print("  B      : 빈 판 background model 30프레임 생성")
        print("  T      : 현재 LOW-Z ROI를 기준으로 Dynamic Tracking 시작")
        print("  R      : Tracking 중지 -> LOW-Z ROI 검출 상태로 복귀")
        print("  C      : background model/Tracking 전체 초기화")
        print("  SPACE  : 현재 결과 저장")
        print("  Q/ESC  : 종료")
        print("=" * 88)

        for _ in range(
            args.warmup_frames
        ):
            wait_for_aligned_pair(
                pipeline,
                align_filter,
            )

        print(
            "워밍업 완료. LOW Z + 빈 판 + ID0~3 상태에서 B를 누르세요."
        )

        while True:
            frame, aligned_depth_mm = wait_for_aligned_pair(
                pipeline,
                align_filter,
            )

            depth_history.append(
                aligned_depth_mm
            )

            max_depth_frames = max(
                1,
                int(args.depth_median_frames),
            )

            if len(depth_history) > max_depth_frames:
                depth_history = depth_history[
                    -max_depth_frames:
                ]

            depth_median_mm = temporal_median_depth(
                depth_history
            )

            latest_depth_aligned_vis = render_depth_debug(
                depth_median_mm
            )

            marker_map = detect_markers(
                frame,
                detector,
            )

            marker_ids = sorted(
                marker_map.keys()
            )

            live_quad = get_board_outer_quad(
                marker_map
            )

            overlay = frame.copy()

            for marker_id, corners in marker_map.items():
                pts = corners.astype(
                    np.int32
                )

                cv2.polylines(
                    overlay,
                    [pts],
                    True,
                    (0, 255, 255),
                    2,
                )

                center = np.mean(
                    corners,
                    axis=0,
                ).astype(np.int32)

                cv2.putText(
                    overlay,
                    f"ID {marker_id}",
                    (
                        int(center[0]) + 5,
                        int(center[1]) - 5,
                    ),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (0, 255, 255),
                    2,
                    cv2.LINE_AA,
                )

            latest_candidate = None
            latest_clean = None
            latest_strong_seed = None
            latest_density_support = None
            latest_grabcut_labels = None
            latest_raw_object_canonical = None
            latest_object_canonical = None
            latest_inspection_canonical = None
            latest_object_frame = None
            latest_inspection_frame = None

            latest_canonical_depth_vis = None
            latest_depth_height_vis = None
            latest_depth_object_mask = None
            latest_depth_valid_mask = None
            latest_fused_candidate = None
            latest_depth_metrics = None

            if state == "WAIT_BACKGROUND":
                if live_quad is not None:
                    latest_board_quad = live_quad.copy()

                    cv2.polylines(
                        overlay,
                        [
                            live_quad.astype(
                                np.int32
                            )
                        ],
                        True,
                        (0, 255, 0),
                        3,
                    )

                draw_status(
                    overlay,
                    [
                        "STATE: WAIT_BACKGROUND",
                        f"Detected IDs: {marker_ids}",
                        (
                            f"LOW Z + EMPTY BOARD -> press B "
                            f"({args.background_frames} frames)"
                        ),
                    ],
                    good=(
                        live_quad is not None
                    ),
                )

            elif state == "WAIT_OBJECT":
                # LOW Z에서는 live ArUco가 보이면 live quad,
                # 잠깐 안 보이면 background model의 저장 quad 사용.
                board_quad = (
                    live_quad.copy()
                    if live_quad is not None
                    else model["reference_quad"].copy()
                )

                latest_board_quad = board_quad.copy()

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

                canonical, H = warp_board_to_canonical(
                    frame,
                    board_quad,
                    args.canonical_size,
                )

                latest_canonical = canonical

                (
                    candidate,
                    dL,
                    dC,
                    threshold_L,
                    threshold_C,
                ) = compute_background_candidate(
                    canonical,
                    model,
                    args,
                )

                latest_candidate = candidate

                # -----------------------------------------------------
                # Depth plane -> object height seed
                # -----------------------------------------------------
                depth_candidate = None
                depth_strong_seed = None
                canonical_depth_mm = None
                height_mm = None
                depth_valid_mask = None

                plane_inlier_ratio = 0.0
                plane_residual_mm = float("inf")

                if depth_median_mm is not None:
                    canonical_depth_mm = warp_depth_to_canonical(
                        depth_median_mm,
                        H,
                        args.canonical_size,
                    )

                    plane_ring_mask = make_plane_ring_mask(
                        model["valid_mask"],
                        args.depth_plane_ring_fraction,
                    )

                    (
                        plane_depth_mm,
                        plane_inlier_ratio,
                        plane_residual_mm,
                    ) = fit_inverse_depth_plane_ransac(
                        canonical_depth_mm,
                        plane_ring_mask,
                        args,
                    )

                    if plane_depth_mm is not None:
                        (
                            depth_candidate,
                            depth_strong_seed,
                            height_mm,
                            depth_valid_mask,
                        ) = make_depth_object_mask(
                            canonical_depth_mm,
                            plane_depth_mm,
                            model["valid_mask"],
                            args,
                        )

                        latest_depth_object_mask = depth_candidate
                        latest_depth_valid_mask = depth_valid_mask
                        latest_canonical_depth_vis = render_depth_debug(
                            canonical_depth_mm
                        )
                        latest_depth_height_vis = render_height_debug(
                            height_mm,
                            max_height_mm=max(
                                20.0,
                                args.depth_height_threshold_mm
                                * 8.0,
                            ),
                        )

                fused_candidate = candidate.copy()

                if depth_candidate is not None:
                    fused_candidate = cv2.bitwise_or(
                        fused_candidate,
                        depth_candidate,
                    )

                latest_fused_candidate = fused_candidate

                (
                    object_mask,
                    strong_seed,
                    density_support,
                    gc_mask,
                    density,
                ) = seed_grabcut_refine(
                    canonical_bgr=canonical,
                    candidate=candidate,
                    dL=dL,
                    dC=dC,
                    threshold_L=threshold_L,
                    threshold_C=threshold_C,
                    valid_mask=model["valid_mask"],
                    args=args,
                    extra_candidate=depth_candidate,
                    extra_strong_seed=depth_strong_seed,
                )

                depth_valid_ratio = (
                    float(
                        np.mean(
                            depth_valid_mask > 0
                        )
                    )
                    if depth_valid_mask is not None
                    else 0.0
                )

                depth_object_ratio = (
                    float(
                        np.mean(
                            depth_candidate > 0
                        )
                    )
                    if depth_candidate is not None
                    else 0.0
                )

                latest_depth_metrics = {
                    "depth_exposure": args.depth_exposure,
                    "depth_gain": args.depth_gain,
                    "depth_valid_ratio": f"{depth_valid_ratio:.6f}",
                    "depth_object_ratio": f"{depth_object_ratio:.6f}",
                    "plane_inlier_ratio": f"{plane_inlier_ratio:.6f}",
                    "plane_residual_median_mm": (
                        f"{plane_residual_mm:.3f}"
                        if np.isfinite(plane_residual_mm)
                        else "inf"
                    ),
                    "depth_height_threshold_mm": args.depth_height_threshold_mm,
                }

                latest_strong_seed = strong_seed
                latest_density_support = density_support
                latest_grabcut_labels = visualize_grabcut_labels(
                    gc_mask
                )

                latest_raw_object_canonical = object_mask

                if object_mask is not None:
                    (
                        final_object_mask,
                        inspection_mask,
                        final_contour,
                    ) = finalize_object_silhouette(
                        object_mask,
                        args,
                    )

                    object_mask = final_object_mask
                else:
                    inspection_mask = None
                    final_contour = None

                # 기존 clean_candidate 출력 자리는 density support를 넣어
                # 이전 코드와 저장 구조 호환성을 유지
                latest_clean = (
                    density_support
                    if density_support is not None
                    else candidate
                )

                changed_ratio = (
                    100.0
                    * np.count_nonzero(
                        candidate > 0
                    )
                    / float(
                        candidate.size
                    )
                )

                if object_mask is None:
                    draw_status(
                        overlay,
                        [
                            "STATE: WAIT_OBJECT / NO STABLE OBJECT",
                            f"Detected IDs: {marker_ids}",
                            (
                                f"Changed pixels={changed_ratio:.2f}% "
                                f"| min area={args.min_object_area}"
                            ),
                            (
                                "No stable seed/support. "
                                "Try --density-threshold 0.08 or lower BG thresholds."
                            ),
                        ],
                        good=False,
                    )

                else:
                    latest_object_canonical = object_mask
                    latest_inspection_canonical = inspection_mask

                    frame_mask = inverse_warp_mask_to_frame(
                        object_mask,
                        board_quad,
                        frame.shape,
                    )

                    latest_object_frame = frame_mask

                    if inspection_mask is not None:
                        inspection_frame_mask = inverse_warp_mask_to_frame(
                            inspection_mask,
                            board_quad,
                            frame.shape,
                        )
                    else:
                        inspection_frame_mask = None

                    latest_inspection_frame = inspection_frame_mask

                    # PURPLE: 최종 물체 외곽
                    draw_mask_contour(
                        overlay,
                        frame_mask,
                        (255, 0, 255),
                        4,
                    )

                    # YELLOW: 실제 결함 검사 안전영역
                    if inspection_frame_mask is not None:
                        draw_mask_contour(
                            overlay,
                            inspection_frame_mask,
                            (0, 255, 255),
                            2,
                        )

                    bbox = bbox_from_mask(
                        frame_mask
                    )

                    if bbox is not None:
                        x, y, w, h = bbox

                        cv2.rectangle(
                            overlay,
                            (x, y),
                            (x + w, y + h),
                            (255, 255, 0),
                            2,
                        )

                    object_area = np.count_nonzero(
                        object_mask > 0
                    )

                    inspection_area = (
                        np.count_nonzero(
                            inspection_mask > 0
                        )
                        if inspection_mask is not None
                        else 0
                    )

                    seed_area = (
                        np.count_nonzero(
                            strong_seed > 0
                        )
                        if strong_seed is not None
                        else 0
                    )

                    support_area = (
                        np.count_nonzero(
                            density_support > 0
                        )
                        if density_support is not None
                        else 0
                    )

                    draw_status(
                        overlay,
                        [
                            "STATE: SEED-GRABCUT OBJECT ROI",
                            f"Detected IDs: {marker_ids}",
                            (
                                f"Changed={changed_ratio:.2f}% "
                                f"| seed={seed_area}px "
                                f"| support={support_area}px"
                            ),
                            (
                                f"Final object={object_area}px "
                                f"| inspection={inspection_area}px "
                                f"| erode={args.inspection_erode_px}px"
                            ),
                            (
                                f"Depth valid={depth_valid_ratio*100:.1f}% "
                                f"| Depth object={depth_object_ratio*100:.1f}%"
                            ),
                            (
                                f"Plane inlier={plane_inlier_ratio*100:.1f}% "
                                f"| residual={plane_residual_mm:.2f} mm "
                                f"| H>{args.depth_height_threshold_mm:.1f}mm"
                            ),
                            "PURPLE=object outer ROI | YELLOW=inspection ROI",
                        ],
                        good=True,
                    )

            elif state == "TRACK_OBJECT":
                # -----------------------------------------------------
                # 현재 프레임 기반 Dynamic ROI
                # - ArUco 필요 없음
                # - LOW-Z background model 재사용 안 함
                # - Z 이동량/모터 위치 사용 안 함
                # -----------------------------------------------------
                if tracked_object_frame is None:
                    state = "WAIT_OBJECT"

                    draw_status(
                        overlay,
                        [
                            "STATE: TRACK_OBJECT / NO PREVIOUS MASK",
                            "Tracking mask missing -> return WAIT_OBJECT",
                        ],
                        good=False,
                    )

                else:
                    (
                        search_mask,
                        plane_ring_mask,
                        search_margin,
                    ) = build_tracking_regions(
                        tracked_object_frame,
                        args,
                    )

                    plane_depth_mm = None
                    plane_inlier_ratio = 0.0
                    plane_residual_mm = float("inf")

                    raw_depth_valid_ratio = 0.0
                    if depth_median_mm is not None:
                        inside_search = search_mask > 0
                        search_count = int(np.count_nonzero(inside_search))
                        if search_count > 0:
                            raw_depth_valid_ratio = float(
                                np.count_nonzero(
                                    inside_search
                                    & (depth_median_mm >= args.depth_min_mm)
                                    & (depth_median_mm <= args.depth_max_mm)
                                ) / search_count
                            )

                    if depth_median_mm is not None and raw_depth_valid_ratio >= args.track_min_depth_valid_ratio:
                        (
                            plane_depth_mm,
                            plane_inlier_ratio,
                            plane_residual_mm,
                        ) = fit_inverse_depth_plane_ransac(
                            depth_median_mm,
                            plane_ring_mask,
                            args,
                        )

                    depth_candidate = None
                    depth_strong_seed = None
                    depth_valid_mask = None
                    height_mm = None

                    if plane_depth_mm is not None:
                        (
                            depth_candidate,
                            depth_strong_seed,
                            height_mm,
                            depth_valid_mask,
                        ) = make_depth_object_mask(
                            depth_median_mm,
                            plane_depth_mm,
                            search_mask,
                            args,
                        )

                        latest_depth_object_mask = depth_candidate
                        latest_depth_valid_mask = depth_valid_mask
                        latest_depth_height_vis = render_height_debug(
                            height_mm,
                            max_height_mm=max(
                                20.0,
                                args.depth_height_threshold_mm
                                * 8.0,
                            ),
                        )

                    (
                        raw_track_mask,
                        track_gc_mask,
                    ) = track_grabcut_current_frame(
                        frame=frame,
                        previous_mask=tracked_object_frame,
                        search_mask=search_mask,
                        depth_candidate=depth_candidate,
                        depth_strong_seed=depth_strong_seed,
                        args=args,
                    )

                    accepted = False
                    area_ratio = 0.0
                    overlap_ratio = 0.0
                    tracking_confidence = "LOW"
                    allowed_max_area_ratio = float(
                        args.track_max_area_ratio
                    )

                    if raw_track_mask is not None:
                        (
                            new_object_mask,
                            new_inspection_mask,
                            _,
                        ) = finalize_object_silhouette(
                            raw_track_mask,
                            args,
                        )

                        if new_object_mask is not None:
                            (
                                accepted,
                                area_ratio,
                                overlap_ratio,
                                tracking_confidence,
                                allowed_max_area_ratio,
                            ) = validate_tracking_update(
                                tracked_object_frame,
                                new_object_mask,
                                raw_depth_valid_ratio,
                                plane_inlier_ratio,
                                plane_residual_mm,
                                args,
                            )
                        else:
                            new_inspection_mask = None
                    else:
                        new_object_mask = None
                        new_inspection_mask = None

                    if (
                        accepted
                        and new_object_mask is not None
                    ):
                        tracked_object_frame = new_object_mask.copy()

                        tracked_inspection_frame = (
                            new_inspection_mask.copy()
                            if new_inspection_mask is not None
                            else None
                        )

                        track_state_text = "TRACK UPDATE"
                        track_good = True

                    else:
                        # 실패 프레임에서는 이전 ROI를 그대로 유지.
                        # 잘못된 ROI로 순간 점프하는 것보다 안전하다.
                        track_state_text = "TRACK HOLD"
                        track_good = False

                    latest_object_frame = tracked_object_frame.copy()

                    latest_inspection_frame = (
                        tracked_inspection_frame.copy()
                        if tracked_inspection_frame is not None
                        else None
                    )

                    latest_grabcut_labels = visualize_grabcut_labels(
                        track_gc_mask
                    )

                    # tracking search area를 debug candidate 창에 표시
                    latest_fused_candidate = search_mask.copy()

                    # PURPLE: 동적으로 갱신되는 현재 object ROI
                    draw_mask_contour(
                        overlay,
                        tracked_object_frame,
                        (255, 0, 255),
                        4,
                    )

                    # YELLOW: 실제 검사 영역
                    if tracked_inspection_frame is not None:
                        draw_mask_contour(
                            overlay,
                            tracked_inspection_frame,
                            (0, 255, 255),
                            2,
                        )

                    # CYAN: 현재 프레임에서 허용하는 search area
                    draw_mask_contour(
                        overlay,
                        search_mask,
                        (255, 255, 0),
                        2,
                    )

                    current_area = int(
                        np.count_nonzero(
                            tracked_object_frame > 0
                        )
                    )

                    depth_valid_ratio = (
                        float(
                            np.mean(
                                depth_valid_mask > 0
                            )
                        )
                        if depth_valid_mask is not None
                        else 0.0
                    )

                    depth_object_ratio = (
                        float(
                            np.count_nonzero(
                                depth_candidate > 0
                            )
                            / max(
                                1,
                                np.count_nonzero(
                                    search_mask > 0
                                ),
                            )
                        )
                        if depth_candidate is not None
                        else 0.0
                    )

                    latest_depth_metrics = {
                        "tracking_state": track_state_text,
                        "search_margin_px": search_margin,
                        "area_ratio": f"{area_ratio:.4f}",
                        "overlap_ratio": f"{overlap_ratio:.4f}",
                        "tracking_confidence": tracking_confidence,
                        "allowed_max_area_ratio": f"{allowed_max_area_ratio:.4f}",
                        "depth_exposure": args.depth_exposure,
                        "depth_gain": args.depth_gain,
                        "depth_valid_ratio": f"{depth_valid_ratio:.6f}",
                        "depth_object_ratio_in_search": f"{depth_object_ratio:.6f}",
                        "plane_inlier_ratio": f"{plane_inlier_ratio:.6f}",
                        "plane_residual_median_mm": (
                            f"{plane_residual_mm:.3f}"
                            if np.isfinite(plane_residual_mm)
                            else "inf"
                        ),
                    }

                    draw_status(
                        overlay,
                        [
                            f"STATE: {track_state_text}",
                            (
                                f"Current ROI={current_area}px "
                                f"| search margin={search_margin}px"
                            ),
                            (
                                f"area ratio={area_ratio:.3f} "
                                f"| overlap={overlap_ratio:.3f}"
                            ),
                            (
                                f"Depth valid={raw_depth_valid_ratio*100:.1f}% "
                                f"| object in search={depth_object_ratio*100:.1f}%"
                            ),
                            (
                                f"Plane inlier={plane_inlier_ratio*100:.1f}% "
                                f"| inlier residual={plane_residual_mm:.2f} mm"
                            ),
                            "PURPLE=current ROI | YELLOW=inspection | CYAN=search",
                            "No Z position / no scale hardcoding",
                        ],
                        good=track_good,
                    )

            # 화면 표시
            preview = cv2.resize(
                overlay,
                (
                    960,
                    600,
                ),
                interpolation=cv2.INTER_AREA,
            )

            cv2.imshow(
                "Background Model Object ROI",
                preview,
            )

            if latest_candidate is not None:
                mask_preview = cv2.resize(
                    latest_candidate,
                    (
                        600,
                        600,
                    ),
                    interpolation=cv2.INTER_NEAREST,
                )

                cv2.imshow(
                    "Background Candidate (Canonical)",
                    mask_preview,
                )

            if latest_density_support is not None:
                support_preview = cv2.resize(
                    latest_density_support,
                    (
                        600,
                        600,
                    ),
                    interpolation=cv2.INTER_NEAREST,
                )

                cv2.imshow(
                    "Density Support (Canonical)",
                    support_preview,
                )

            if latest_object_canonical is not None:
                final_preview = cv2.resize(
                    latest_object_canonical,
                    (
                        600,
                        600,
                    ),
                    interpolation=cv2.INTER_NEAREST,
                )

                cv2.imshow(
                    "LOW-Z Final Object Mask",
                    final_preview,
                )

            if latest_inspection_canonical is not None:
                inspection_preview = cv2.resize(
                    latest_inspection_canonical,
                    (
                        600,
                        600,
                    ),
                    interpolation=cv2.INTER_NEAREST,
                )

                cv2.imshow(
                    "LOW-Z Inspection Mask",
                    inspection_preview,
                )

            if latest_depth_object_mask is not None:
                depth_mask_preview = cv2.resize(
                    latest_depth_object_mask,
                    (
                        600,
                        600,
                    ),
                    interpolation=cv2.INTER_NEAREST,
                )

                cv2.imshow(
                    "Depth Plane Object Mask",
                    depth_mask_preview,
                )

            if latest_depth_height_vis is not None:
                depth_height_preview = cv2.resize(
                    latest_depth_height_vis,
                    (
                        600,
                        600,
                    ),
                    interpolation=cv2.INTER_AREA,
                )

                cv2.imshow(
                    "Depth Height From Board Plane",
                    depth_height_preview,
                )

            if latest_fused_candidate is not None:
                fused_preview = cv2.resize(
                    latest_fused_candidate,
                    (
                        600,
                        600,
                    ),
                    interpolation=cv2.INTER_NEAREST,
                )

                cv2.imshow(
                    "Candidate / Tracking Search Area",
                    fused_preview,
                )

            key = (
                cv2.waitKey(1)
                & 0xFF
            )

            if key in (
                ord("q"),
                ord("Q"),
                27,
            ):
                break

            if key in (
                ord("t"),
                ord("T"),
            ):
                if state == "WAIT_OBJECT":
                    if latest_object_frame is None:
                        print(
                            "[T] 먼저 LOW-Z에서 안정적인 보라색 ROI가 나와야 합니다."
                        )
                    else:
                        tracked_object_frame = latest_object_frame.copy()

                        tracked_inspection_frame = (
                            latest_inspection_frame.copy()
                            if latest_inspection_frame is not None
                            else None
                        )

                        state = "TRACK_OBJECT"

                        print("=" * 80)
                        print("[DYNAMIC TRACKING START]")
                        print(
                            "이제 Z축을 천천히/단계적으로 올려보세요."
                        )
                        print(
                            "Z 이동량은 사용하지 않고 현재 RGB + 현재 Depth로 ROI를 갱신합니다."
                        )
                        print("=" * 80)

                elif state == "TRACK_OBJECT":
                    print(
                        "[T] 이미 Dynamic Tracking 중입니다."
                    )
                else:
                    print(
                        "[T] 먼저 B로 background model을 만든 뒤 물체 ROI를 검출하세요."
                    )

                continue

            if key in (
                ord("r"),
                ord("R"),
            ):
                if model is not None:
                    state = "WAIT_OBJECT"
                    tracked_object_frame = None
                    tracked_inspection_frame = None

                    print(
                        "[R] Tracking 중지 -> LOW-Z/current ArUco ROI 검출 상태로 복귀"
                    )
                else:
                    print(
                        "[R] background model이 없습니다."
                    )

                continue

            if key in (
                ord("b"),
                ord("B"),
            ):
                if state != "WAIT_BACKGROUND":
                    print(
                        "[B] 이미 background model이 있습니다. "
                        "다시 만들려면 C를 누르세요."
                    )
                    continue

                if live_quad is None:
                    print(
                        "[B] ID0,1,2,3이 모두 보여야 합니다."
                    )
                    continue

                model = build_background_model(
                    pipeline,
                    align_filter,
                    detector,
                    args,
                )

                # background model을 즉시 저장
                ref_dir = (
                    args.output_dir
                    / "background_model"
                )

                ref_dir.mkdir(
                    parents=True,
                    exist_ok=True,
                )

                cv2.imwrite(
                    str(
                        ref_dir
                        / "background_median.png"
                    ),
                    model["median_bgr"],
                )

                cv2.imwrite(
                    str(
                        ref_dir
                        / "canonical_valid_mask.png"
                    ),
                    model["valid_mask"],
                )

                np.savez_compressed(
                    str(
                        ref_dir
                        / "background_model.npz"
                    ),
                    median_lab=model["median_lab"],
                    sigma_lab=model["sigma_lab"],
                    reference_quad=model["reference_quad"],
                    valid_mask=model["valid_mask"],
                )

                state = "WAIT_OBJECT"

                print("=" * 80)
                print("[BACKGROUND MODEL READY]")
                print(
                    (
                        ref_dir
                        / "background_model.npz"
                    ).resolve()
                )
                print(
                    "같은 LOW Z를 유지하고 검사체를 올리세요."
                )
                print("=" * 80)

                continue

            if key in (
                ord("c"),
                ord("C"),
            ):
                model = None
                state = "WAIT_BACKGROUND"

                tracked_object_frame = None
                tracked_inspection_frame = None

                latest_canonical = None
                latest_candidate = None
                latest_clean = None
                latest_strong_seed = None
                latest_density_support = None
                latest_grabcut_labels = None
                latest_raw_object_canonical = None
                latest_object_canonical = None
                latest_inspection_canonical = None
                latest_object_frame = None
                latest_inspection_frame = None

                latest_depth_aligned_vis = None
                latest_canonical_depth_vis = None
                latest_depth_height_vis = None
                latest_depth_object_mask = None
                latest_depth_valid_mask = None
                latest_fused_candidate = None
                latest_depth_metrics = None

                depth_history.clear()

                print(
                    "[CLEAR] background model 삭제. "
                    "LOW Z + 빈 판에서 B를 다시 누르세요."
                )

                continue

            if key == 32:
                save_results(
                    output_root=args.output_dir,
                    original=frame,
                    overlay=overlay,
                    canonical=latest_canonical,
                    candidate=latest_candidate,
                    clean_candidate=latest_clean,
                    strong_seed=latest_strong_seed,
                    density_support=latest_density_support,
                    grabcut_labels=latest_grabcut_labels,
                    depth_aligned_vis=latest_depth_aligned_vis,
                    canonical_depth_vis=latest_canonical_depth_vis,
                    depth_height_vis=latest_depth_height_vis,
                    depth_object_mask=latest_depth_object_mask,
                    depth_valid_mask=latest_depth_valid_mask,
                    fused_candidate=latest_fused_candidate,
                    raw_object_mask_canonical=latest_raw_object_canonical,
                    final_object_mask_canonical=latest_object_canonical,
                    inspection_mask_canonical=latest_inspection_canonical,
                    object_mask_frame=latest_object_frame,
                    inspection_mask_frame=latest_inspection_frame,
                    model=model,
                    board_quad=latest_board_quad,
                    depth_metrics=latest_depth_metrics,
                )

    finally:
        if started:
            pipeline.stop()

        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

