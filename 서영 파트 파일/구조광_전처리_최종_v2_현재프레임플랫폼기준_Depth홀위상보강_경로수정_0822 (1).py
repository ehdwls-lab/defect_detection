#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# =============================================================================
# 2026-08-22 v2 추가 변경
# - Depth 실제 물체 마스크를 최종 물체 경계로 그대로 믿지 않음
# - Depth가 놓친 '물체 내부 구멍'을 같은 프레임 구조광 phase로 검증
# - 외곽 contour 내부의 Depth hole 중 플랫폼 phase와 확실히 다른 영역만 물체로 복구
# - 실제 관통구멍처럼 플랫폼 phase와 같은 영역은 플랫폼으로 유지
# - 새 물체의 모양 자체를 하드코딩하지 않음
#
# 2026-08-22 최종 변경
# - Depth: E1999/G64 + 현재배치 동일조건 빈 플랫폼 기준
# - 구조광 direction 기본: horizontal / period 80
# - G/E 촬영/픽셀단위 융합 앞단은 기존 13,287줄 원본 유지
# - 외부 빈 플랫폼 Reference phase 차분을 최종 형상에서 제거
# - 현재 물체 촬영 프레임 안의 플랫폼으로 phase field를 직접 모델링
# - 플랫폼/물체 분리 후 물체에서만 local quality-guided unwrap
# - 최종 PLY 플랫폼은 상대높이 0 한 색, 물체는 상대위상 높이 컬러
# - 실제 mm가 아니라 기존과 동일한 상대 위상 기반 Z
# =============================================================================


import argparse
import csv
import json
import heapq
import re
import subprocess
import time
import warnings
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from pyorbbecsdk import (
    AlignFilter,
    Config,
    OBFormat,
    OBPropertyID,
    OBSensorType,
    OBStreamType,
    Pipeline,
)

try:
    from pyorbbecsdk import OBFrameAggregateOutputMode
except ImportError:
    OBFrameAggregateOutputMode = None


기본_저장_루트 = Path(
    "/home/seoyeong/졸업작품/전처리와구조광_통합/플랫폼 바닥 따기/구조광_전처리"
)
기본_카메라값_파일 = Path(
    "/home/seoyeong/졸업작품/전처리와구조광_통합/구조광_전처리_촬영/공통_카메라_고정값.json"
)

위상_목록 = [
    (0.0, "000"),
    (np.pi / 2.0, "090"),
    (np.pi, "180"),
    (3.0 * np.pi / 2.0, "270"),
]

색상_표시명 = {
    "white": "흰색",
    "green": "초록색",
    "red": "빨간색",
    "blue": "파란색",
}


# ============================================================
# Depth 기반 자동 물체 영역 검출 설정
# - 방금 확정한 기준 적용
#   Depth E1999/G64
#   10 mm seed + 5 mm 연결 확장
#   Opening 3x3 / Closing 11x11
#   수동 지정 프로젝터 X/Y 최대범위 안에서만 검출
#   최종 전처리는 빨간 사각형 내부 전체 사용
# ============================================================

프로젝트_폴더 = Path("/home/seoyeong/졸업작품/전처리와구조광_통합")
플랫폼_바닥_폴더 = 프로젝트_폴더 / "플랫폼 바닥 따기"

선택용_프로젝터_RGB = 255

# 현재 배치에서 최종 선택한 빈 플랫폼 Depth: E1999/G64
기준_Depth_경로 = Path(
    "/home/seoyeong/졸업작품/전처리와구조광_통합/플랫폼 바닥 따기/"
    "현재배치_기준데이터/기준촬영_20260818_191520/"
    "E1999_G64/플랫폼_바닥_depth.npy"
)

# 방금 X/Y 2점으로 다시 저장한 범위 JSON
# 파일명은 기존 호환 때문에 그대로 "프로젝터_세로범위.json"
프로젝터_XY범위_JSON = (
    프로젝트_폴더
    / "프로젝터 수동 범위 확인"
    / "프로젝터_세로범위.json"
)

# Depth 검출 중간 결과
결과_루트 = (
    플랫폼_바닥_폴더
    / "구조광_전처리"
)

# 빈 플랫폼과 물체 Depth를 같은 조건으로 고정
Depth_노출 = 1999
Depth_게인 = 64

워밍업_프레임수 = 20
Depth_프레임수 = 15
프레임_대기시간_ms = 1000

# 확실한 물체 중심 / 연결 확장
확실한_물체_높이차_mm = 10.0
확장_물체_높이차_mm = 5.0

# X/Y 수동 범위를 검출 시작부터 적용하므로 별도 화면 가장자리 제거는 0
가로_가장자리_제외비율 = 0.00
세로_가장자리_제외비율 = 0.00

최소_물체_면적_px = 1500

# 검출된 물체를 감싸는 사각형에 5% 여유
사각영역_여유비율 = 0.05

오프닝_커널 = (3, 3)
클로징_커널 = (11, 11)

빔_XY제한_사용 = True

def 구분선():
    print("=" * 70)


def 안전한_이름(text):
    return re.sub(r'[\\/:*?"<>|]+', "_", str(text).strip()) or "이름없음"


def 재귀_값_찾기(data, 후보_키):
    if isinstance(data, dict):
        for key, value in data.items():
            if str(key) in 후보_키:
                return value
        for value in data.values():
            found = 재귀_값_찾기(value, 후보_키)
            if found is not None:
                return found
    elif isinstance(data, list):
        for value in data:
            found = 재귀_값_찾기(value, 후보_키)
            if found is not None:
                return found
    return None


def 공통_카메라값_불러오기(path, 기본_게인, 기본_흰색보정):
    values = {
        "gain": int(기본_게인),
        "white_balance_temperature": int(기본_흰색보정),
    }

    path = Path(path)
    if not path.exists():
        print(f"공통 카메라값 파일이 없어 기본값을 사용합니다: {path}")
        return values

    try:
        with path.open("r", encoding="utf-8") as file:
            data = json.load(file)

        gain = 재귀_값_찾기(data, {"gain", "게인"})
        wb = 재귀_값_찾기(
            data,
            {
                "white_balance_temperature",
                "white_balance",
                "흰색보정",
                "화이트밸런스",
            },
        )

        if gain is not None:
            values["gain"] = int(float(gain))
        if wb is not None:
            values["white_balance_temperature"] = int(float(wb))

        print(f"공통 카메라값을 불러왔습니다: {path}")
    except Exception as exc:
        print(f"공통 카메라값을 읽지 못해 기본값을 사용합니다: {exc}")

    return values


def xrandr_모니터_목록():
    try:
        result = subprocess.run(
            ["xrandr", "--query"],
            capture_output=True,
            text=True,
            check=True,
        )
    except Exception:
        return []

    pattern = re.compile(
        r"^(?P<name>\S+)\s+connected(?:\s+primary)?\s+"
        r"(?P<w>\d+)x(?P<h>\d+)\+(?P<x>-?\d+)\+(?P<y>-?\d+)"
    )

    monitors = []
    for line in result.stdout.splitlines():
        match = pattern.search(line)
        if match:
            monitors.append(
                {
                    "name": match.group("name"),
                    "w": int(match.group("w")),
                    "h": int(match.group("h")),
                    "x": int(match.group("x")),
                    "y": int(match.group("y")),
                    "primary": " primary " in f" {line} ",
                }
            )
    return monitors


def 프로젝터_화면_선택(args):
    monitors = xrandr_모니터_목록()

    구분선()
    print("xrandr 모니터 목록")
    if not monitors:
        print("모니터 목록을 읽지 못했습니다.")
    else:
        for index, monitor in enumerate(monitors):
            role = "주 화면" if monitor["primary"] else "보조 화면"
            print(
                f"[{index}] {monitor['name']} | "
                f"{monitor['w']}×{monitor['h']} | "
                f"위치 ({monitor['x']}, {monitor['y']}) | {role}"
            )
    구분선()

    selected = None

    if args.monitor != "auto":
        selected = next(
            (monitor for monitor in monitors if monitor["name"] == args.monitor),
            None,
        )
    else:
        selected = next(
            (
                monitor
                for monitor in monitors
                if "HDMI" in monitor["name"].upper()
            ),
            None,
        )

        if selected is None:
            selected = next(
                (monitor for monitor in monitors if not monitor["primary"]),
                None,
            )

    if selected is None:
        selected = {
            "name": "수동 기본값",
            "w": args.w,
            "h": args.h,
            "x": args.x,
            "y": args.y,
            "primary": False,
        }

    print("선택된 프로젝터 화면")
    print(f"이름: {selected['name']}")
    print(f"크기: {selected['w']} × {selected['h']}")
    print(f"위치: x={selected['x']}, y={selected['y']}")
    구분선()

    return selected


def Depth프레임_mm로_변환(depth_frame):
    width = depth_frame.get_width()
    height = depth_frame.get_height()

    raw = np.frombuffer(
        depth_frame.get_data(),
        dtype=np.uint16,
    ).reshape(
        (height, width)
    )

    scale = float(
        depth_frame.get_depth_scale()
    )

    return (
        raw.astype(np.float32)
        * scale
    )

def 여러Depth_중앙값(depth_frames):
    stack = np.stack(
        depth_frames,
        axis=0,
    ).astype(np.float32)

    stack[
        stack <= 0
    ] = np.nan

    with warnings.catch_warnings():
        warnings.simplefilter(
            "ignore",
            category=RuntimeWarning,
        )

        median = np.nanmedian(
            stack,
            axis=0,
        )

    return np.nan_to_num(
        median,
        nan=0.0,
    ).astype(np.float32)

def 높이차_시각화(height_diff, valid):
    vis = np.zeros(
        height_diff.shape,
        dtype=np.uint8,
    )

    positive = (
        valid
        & (height_diff > 0)
    )

    if np.any(positive):
        high = max(
            20.0,
            float(
                np.percentile(
                    height_diff[positive],
                    98,
                )
            ),
        )

        normalized = np.clip(
            height_diff,
            0.0,
            high,
        ) / high * 255.0

        vis[positive] = normalized[
            positive
        ].astype(np.uint8)

    return cv2.applyColorMap(
        vis,
        cv2.COLORMAP_JET,
    )

def 사각영역_여유추가(x, y, w, h, image_w, image_h):
    margin_x = int(
        round(
            w * 사각영역_여유비율
        )
    )

    margin_y = int(
        round(
            h * 사각영역_여유비율
        )
    )

    x1 = max(
        0,
        x - margin_x,
    )

    y1 = max(
        0,
        y - margin_y,
    )

    x2 = min(
        image_w,
        x + w + margin_x,
    )

    y2 = min(
        image_h,
        y + h + margin_y,
    )

    return (
        x1,
        y1,
        x2,
        y2,
    )

def 사각영역_180도회전(rect, image_w, image_h):
    x1, y1, x2, y2 = rect

    return (
        image_w - x2,
        image_h - y2,
        image_w - x1,
        image_h - y1,
    )

# ============================================================
# Depth 노출 / 게인 E81/G16 고정
# ============================================================

def Depth_수동설정_적용(device):
    device.set_bool_property(
        OBPropertyID.OB_PROP_DEPTH_AUTO_EXPOSURE_BOOL,
        False,
    )

    device.set_int_property(
        OBPropertyID.OB_PROP_DEPTH_EXPOSURE_INT,
        int(Depth_노출),
    )

    device.set_int_property(
        OBPropertyID.OB_PROP_DEPTH_GAIN_INT,
        int(Depth_게인),
    )

    try:
        실제_노출 = int(
            device.get_int_property(
                OBPropertyID.OB_PROP_DEPTH_EXPOSURE_INT
            )
        )
    except Exception:
        실제_노출 = "확인불가"

    try:
        실제_게인 = int(
            device.get_int_property(
                OBPropertyID.OB_PROP_DEPTH_GAIN_INT
            )
        )
    except Exception:
        실제_게인 = "확인불가"

    print("")
    print("Depth 수동 설정")
    print(
        f"요청 E{Depth_노출}/G{Depth_게인} | "
        f"실제 E{실제_노출}/G{실제_게인}"
    )

    return 실제_노출, 실제_게인



def 프로젝터_화면_자동선택():
    monitors = xrandr_모니터_목록()

    if not monitors:
        raise RuntimeError(
            "xrandr에서 모니터 목록을 읽지 못했습니다."
        )

    selected = next(
        (
            monitor
            for monitor in monitors
            if "HDMI" in monitor["name"].upper()
        ),
        None,
    )

    if selected is None:
        selected = next(
            (
                monitor
                for monitor in monitors
                if not monitor["primary"]
            ),
            None,
        )

    if selected is None:
        raise RuntimeError(
            "프로젝터로 사용할 보조 화면을 찾지 못했습니다."
        )

    print("")
    print("선택된 프로젝터 화면")
    print(
        f"이름: {selected['name']} | "
        f"{selected['w']}×{selected['h']} | "
        f"위치 ({selected['x']}, {selected['y']})"
    )

    return selected

def 프로젝터_균일광_켜기(monitor, rgb_value):
    window_name = "Depth 자동검출용 ROI 선택 균일광"

    cv2.namedWindow(
        window_name,
        cv2.WINDOW_NORMAL,
    )

    image = np.full(
        (
            monitor["h"],
            monitor["w"],
            3,
        ),
        int(rgb_value),
        dtype=np.uint8,
    )

    cv2.imshow(
        window_name,
        image,
    )
    cv2.waitKey(300)

    cv2.moveWindow(
        window_name,
        monitor["x"],
        monitor["y"],
    )
    cv2.waitKey(300)

    cv2.setWindowProperty(
        window_name,
        cv2.WND_PROP_FULLSCREEN,
        cv2.WINDOW_FULLSCREEN,
    )

    cv2.imshow(
        window_name,
        image,
    )
    cv2.waitKey(1000)

    print(
        f"프로젝터 ROI 선택용 균일광 투사 중: "
        f"RGB={rgb_value}"
    )

    return window_name

def 프로젝터_검정화면(window_name, monitor):
    if window_name is None:
        return

    black = np.zeros(
        (
            monitor["h"],
            monitor["w"],
            3,
        ),
        dtype=np.uint8,
    )

    cv2.imshow(
        window_name,
        black,
    )
    cv2.waitKey(300)

# ============================================================
# 프로젝터 X/Y 사용범위
# ============================================================

def 저장된_프로젝터_XY범위_읽기():
    """
    수동 2점 코드에서 180도 회전된 Color 영상 기준으로 저장한
    X/Y 사용범위를 읽는다.

    필수 값:
      안전_왼쪽
      안전_오른쪽
      안전_위
      안전_아래
    """

    if not 프로젝터_XY범위_JSON.exists():
        raise FileNotFoundError(
            "새로 만든 프로젝터 X/Y 범위 JSON이 없습니다.\n"
            f"필요 파일: {프로젝터_XY범위_JSON}\n"
            "먼저 프로젝터 X/Y 범위 수동 2점 설정 코드를 실행하세요."
        )

    try:
        data = json.loads(
            프로젝터_XY범위_JSON.read_text(
                encoding="utf-8"
            )
        )
    except Exception as exc:
        raise RuntimeError(
            "프로젝터 X/Y 범위 JSON을 읽지 못했습니다.\n"
            f"{exc}"
        )

    required = (
        "안전_왼쪽",
        "안전_오른쪽",
        "안전_위",
        "안전_아래",
    )

    missing = [
        key
        for key in required
        if key not in data
    ]

    if missing:
        raise RuntimeError(
            "현재 JSON은 예전 Y 전용 형식이거나 값이 부족합니다.\n"
            f"없는 항목: {missing}\n"
            "프로젝터 X/Y 범위 수동 2점 설정 코드를 다시 실행하세요."
        )

    safe_left = int(data["안전_왼쪽"])
    safe_right = int(data["안전_오른쪽"])
    safe_top = int(data["안전_위"])
    safe_bottom = int(data["안전_아래"])

    if safe_right <= safe_left:
        raise RuntimeError(
            "프로젝터 X 범위가 잘못되었습니다. "
            f"왼쪽={safe_left}, 오른쪽={safe_right}"
        )

    if safe_bottom <= safe_top:
        raise RuntimeError(
            "프로젝터 Y 범위가 잘못되었습니다. "
            f"위={safe_top}, 아래={safe_bottom}"
        )

    return data


def 회전후_XY범위_원본좌표로_변환(
    beam_info,
    image_w,
    image_h,
):
    """
    JSON의 X/Y 범위는 180도 회전된 Color 영상 기준이다.

    중요:
    XY 수동 설정 영상 해상도와 현재 D2C Depth 해상도가
    다를 수 있으므로 먼저 해상도 비율만큼 좌표를 스케일링한 뒤
    180도 회전 전 원본 좌표로 변환한다.
    """

    safe_left = float(
        beam_info["안전_왼쪽"]
    )
    safe_right = float(
        beam_info["안전_오른쪽"]
    )
    safe_top = float(
        beam_info["안전_위"]
    )
    safe_bottom = float(
        beam_info["안전_아래"]
    )

    # ------------------------------------------------------------
    # XY 수동 설정 당시 영상 해상도
    # 예: [640, 480]
    # ------------------------------------------------------------
    saved_resolution = beam_info.get(
        "카메라해상도",
        [image_w, image_h],
    )

    if (
        not isinstance(saved_resolution, (list, tuple))
        or len(saved_resolution) < 2
    ):
        raise RuntimeError(
            "프로젝터 XY JSON의 카메라해상도 값이 잘못되었습니다."
        )

    saved_w = float(saved_resolution[0])
    saved_h = float(saved_resolution[1])

    if saved_w <= 0 or saved_h <= 0:
        raise RuntimeError(
            "프로젝터 XY JSON의 카메라해상도가 0 이하입니다."
        )

    # ------------------------------------------------------------
    # 저장 영상 → 현재 D2C Depth 해상도 좌표 스케일링
    # ------------------------------------------------------------
    scale_x = float(image_w) / saved_w
    scale_y = float(image_h) / saved_h

    scaled_left = safe_left * scale_x
    scaled_right = safe_right * scale_x
    scaled_top = safe_top * scale_y
    scaled_bottom = safe_bottom * scale_y

    print("")
    print(
        "프로젝터 XY 좌표 해상도 변환: "
        f"{int(saved_w)}x{int(saved_h)}"
        " -> "
        f"{image_w}x{image_h}"
    )
    print(
        f"XY scale: "
        f"x={scale_x:.4f}, "
        f"y={scale_y:.4f}"
    )
    print(
        "스케일 적용 후 회전영상 좌표: "
        f"x={scaled_left:.1f}~{scaled_right:.1f}, "
        f"y={scaled_top:.1f}~{scaled_bottom:.1f}"
    )

    # ------------------------------------------------------------
    # 180도 회전 영상 좌표 → 원본 D2C 좌표
    # ------------------------------------------------------------
    raw_x1 = int(
        round(image_w - scaled_right)
    )
    raw_x2 = int(
        round(image_w - scaled_left)
    )

    raw_y1 = int(
        round(image_h - scaled_bottom)
    )
    raw_y2 = int(
        round(image_h - scaled_top)
    )

    raw_x1 = int(
        np.clip(raw_x1, 0, image_w)
    )
    raw_x2 = int(
        np.clip(raw_x2, 0, image_w)
    )
    raw_y1 = int(
        np.clip(raw_y1, 0, image_h)
    )
    raw_y2 = int(
        np.clip(raw_y2, 0, image_h)
    )

    if (
        raw_x2 <= raw_x1
        or raw_y2 <= raw_y1
    ):
        raise RuntimeError(
            "프로젝터 X/Y 범위를 원본 Depth 좌표로 "
            "변환한 결과가 잘못되었습니다."
        )

    print(
        "최종 Depth 원본좌표: "
        f"x={raw_x1}~{raw_x2}, "
        f"y={raw_y1}~{raw_y2}"
    )

    return (
        raw_x1,
        raw_y1,
        raw_x2,
        raw_y2,
    )


def 물체영역_빔XY범위로_제한(
    object_rect,
    beam_info,
    image_w,
    image_h,
):
    """
    180도 회전 후의 물체 사각영역과
    프로젝터 X/Y 사용영역의 교집합을 계산한다.

    JSON 좌표는 XY 설정 당시 해상도 기준이므로
    현재 구조광/Depth 영상 해상도로 먼저 스케일링한다.
    """

    x1, y1, x2, y2 = object_rect

    # ------------------------------------------------------------
    # XY 수동 설정 당시 영상 해상도
    # 예: 640 x 480
    # ------------------------------------------------------------
    saved_resolution = beam_info.get(
        "카메라해상도",
        [image_w, image_h],
    )

    if (
        not isinstance(saved_resolution, (list, tuple))
        or len(saved_resolution) < 2
    ):
        raise RuntimeError(
            "프로젝터 XY JSON의 카메라해상도 값이 잘못되었습니다."
        )

    saved_w = float(saved_resolution[0])
    saved_h = float(saved_resolution[1])

    if saved_w <= 0 or saved_h <= 0:
        raise RuntimeError(
            "프로젝터 XY JSON의 카메라해상도가 잘못되었습니다."
        )

    scale_x = float(image_w) / saved_w
    scale_y = float(image_h) / saved_h

    # ------------------------------------------------------------
    # JSON의 180도 회전 영상 좌표를
    # 현재 1280x800 회전 영상 좌표로 확대
    # ------------------------------------------------------------
    safe_left = int(round(
        float(beam_info["안전_왼쪽"])
        * scale_x
    ))

    safe_right = int(round(
        float(beam_info["안전_오른쪽"])
        * scale_x
    ))

    safe_top = int(round(
        float(beam_info["안전_위"])
        * scale_y
    ))

    safe_bottom = int(round(
        float(beam_info["안전_아래"])
        * scale_y
    ))

    safe_left = int(
        np.clip(safe_left, 0, image_w)
    )
    safe_right = int(
        np.clip(safe_right, 0, image_w)
    )
    safe_top = int(
        np.clip(safe_top, 0, image_h)
    )
    safe_bottom = int(
        np.clip(safe_bottom, 0, image_h)
    )

    print("")
    print(
        "회전후 물체 bbox: "
        f"x={int(x1)}~{int(x2)}, "
        f"y={int(y1)}~{int(y2)}"
    )

    print(
        "회전후 빔 XY 범위(해상도 보정): "
        f"x={safe_left}~{safe_right}, "
        f"y={safe_top}~{safe_bottom}"
    )

    final_x1 = max(
        int(x1),
        safe_left,
    )

    final_x2 = min(
        int(x2),
        safe_right,
    )

    final_y1 = max(
        int(y1),
        safe_top,
    )

    final_y2 = min(
        int(y2),
        safe_bottom,
    )

    if (
        final_x2 <= final_x1
        or final_y2 <= final_y1
    ):
        raise RuntimeError(
            "물체 영역과 프로젝터 X/Y 사용영역이 겹치지 않습니다.\n"
            f"물체 bbox = "
            f"x={int(x1)}~{int(x2)}, "
            f"y={int(y1)}~{int(y2)}\n"
            f"빔 bbox = "
            f"x={safe_left}~{safe_right}, "
            f"y={safe_top}~{safe_bottom}"
        )

    print(
        "최종 물체/빔 교집합: "
        f"x={final_x1}~{final_x2}, "
        f"y={final_y1}~{final_y2}"
    )

    return (
        final_x1,
        final_y1,
        final_x2,
        final_y2,
    )


# ============================================================
# 물체 마스크 핵심
# ============================================================

def 마스크_형태정리(mask_bool, open_kernel, close_kernel):
    mask_u8 = mask_bool.astype(np.uint8) * 255

    open_k = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        open_kernel,
    )

    close_k = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        close_kernel,
    )

    mask_u8 = cv2.morphologyEx(
        mask_u8,
        cv2.MORPH_OPEN,
        open_k,
    )

    mask_u8 = cv2.morphologyEx(
        mask_u8,
        cv2.MORPH_CLOSE,
        close_k,
    )

    return mask_u8 > 0


def 가장큰_seed_찾기(seed_mask):
    seed_u8 = seed_mask.astype(np.uint8) * 255

    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        seed_u8,
        connectivity=8,
    )

    candidates = []

    for label in range(1, count):
        area = int(
            stats[
                label,
                cv2.CC_STAT_AREA,
            ]
        )

        if area >= 최소_물체_면적_px:
            candidates.append((label, area))

    if not candidates:
        raise RuntimeError(
            "10 mm 이상의 확실한 물체 시작영역을 찾지 못했습니다. "
            "물체가 매우 낮다면 높이차 기준을 다시 조정해야 합니다."
        )

    candidates.sort(
        key=lambda item: item[1],
        reverse=True,
    )

    largest_label, largest_area = candidates[0]
    largest_seed = labels == largest_label

    return largest_seed, largest_area


def seed에서_후보영역까지_연결확장(seed_mask, allowed_mask):
    """
    10 mm 이상인 확실한 물체에서 시작해,
    5 mm 이상 후보 중 실제로 연결된 픽셀만 반복적으로 확장한다.

    따라서 화면 다른 곳의 5 mm 노이즈/플랫폼 변화는 연결되지 않으면 들어오지 않는다.
    """

    marker = seed_mask.astype(np.uint8)
    allowed = allowed_mask.astype(np.uint8)

    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (3, 3),
    )

    while True:
        grown = cv2.dilate(
            marker,
            kernel,
            iterations=1,
        )

        grown = (
            (grown > 0)
            & (allowed > 0)
        ).astype(np.uint8)

        if np.array_equal(grown, marker):
            break

        marker = grown

    return marker > 0


def 최종_물체마스크_계산(height_diff, valid, search_mask):
    # 10 mm 이상: 확실한 물체
    seed_raw = (
        valid
        & search_mask
        & (height_diff >= 확실한_물체_높이차_mm)
    )

    # 5 mm 이상: 확장 가능한 주변부
    grow_raw = (
        valid
        & search_mask
        & (height_diff >= 확장_물체_높이차_mm)
    )

    seed_clean = 마스크_형태정리(
        seed_raw,
        오프닝_커널,
        클로징_커널,
    )

    grow_clean = 마스크_형태정리(
        grow_raw,
        오프닝_커널,
        클로징_커널,
    )

    # ============================================================
    # DEBUG: Depth 물체 seed 검출 상태
    # ============================================================
    debug_region = valid & search_mask
    debug_values = height_diff[debug_region]

    print("")
    print("=" * 70)
    print("[DEBUG] Depth 물체 검출 진단")
    print(f"valid 픽셀       : {np.count_nonzero(valid)}")
    print(f"search_mask 픽셀 : {np.count_nonzero(search_mask)}")
    print(f"valid & search   : {np.count_nonzero(debug_region)}")

    if debug_values.size > 0:
        print(
            "height_diff [mm]  : "
            f"min={np.min(debug_values):.2f}, "
            f"median={np.median(debug_values):.2f}, "
            f"P90={np.percentile(debug_values, 90):.2f}, "
            f"P95={np.percentile(debug_values, 95):.2f}, "
            f"P99={np.percentile(debug_values, 99):.2f}, "
            f"max={np.max(debug_values):.2f}"
        )
    else:
        print("height_diff      : 검사 가능한 픽셀 없음")

    print(
        f">= {확실한_물체_높이차_mm:.1f} mm raw seed : "
        f"{np.count_nonzero(seed_raw)} px"
    )
    print(
        f">= {확장_물체_높이차_mm:.1f} mm grow     : "
        f"{np.count_nonzero(grow_raw)} px"
    )
    print(
        f"형태정리 후 seed : "
        f"{np.count_nonzero(seed_clean)} px"
    )

    debug_u8 = seed_clean.astype(np.uint8) * 255
    debug_count, _, debug_stats, _ = cv2.connectedComponentsWithStats(
        debug_u8,
        connectivity=8,
    )

    debug_areas = []

    for debug_label in range(1, debug_count):
        debug_areas.append(
            int(debug_stats[debug_label, cv2.CC_STAT_AREA])
        )

    debug_areas.sort(reverse=True)

    print(
        "seed 연결영역 큰 순서: "
        f"{debug_areas[:10]}"
    )
    print(
        f"최소 필요 면적   : {최소_물체_면적_px} px"
    )
    print("=" * 70)
    print("")

    largest_seed, seed_area = 가장큰_seed_찾기(
        seed_clean
    )

    final_mask = seed에서_후보영역까지_연결확장(
        largest_seed,
        grow_clean,
    )

    # 마지막으로 작은 끊김만 한 번 더 연결
    final_close_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        클로징_커널,
    )

    final_mask = cv2.morphologyEx(
        final_mask.astype(np.uint8) * 255,
        cv2.MORPH_CLOSE,
        final_close_kernel,
    ) > 0

    # 화면 가장자리 제한을 다시 한 번 강제 적용
    final_mask &= search_mask

    return {
        "seed_raw": seed_raw,
        "grow_raw": grow_raw,
        "seed_clean": seed_clean,
        "grow_clean": grow_clean,
        "largest_seed": largest_seed,
        "seed_area": int(seed_area),
        "final_mask": final_mask,
    }



# ============================================================
# pyorbbecsdk Color 캡처 어댑터
# - 기존 OpenCV 코드의 cap.read()/cap.release() 호출 형태는 유지
# - 카메라 획득과 Color 속성 제어만 Orbbec SDK로 수행
# ============================================================


def Orbbec컬러프레임_BGR로_변환(color_frame):
    width = color_frame.get_width()
    height = color_frame.get_height()
    color_format = color_frame.get_format()

    data = np.frombuffer(
        color_frame.get_data(),
        dtype=np.uint8,
    )

    if color_format == OBFormat.MJPG:
        image = cv2.imdecode(
            data,
            cv2.IMREAD_COLOR,
        )
        if image is None:
            raise RuntimeError("MJPG Color 프레임 디코딩에 실패했습니다.")
        return image

    if color_format == OBFormat.RGB:
        image = data.reshape(
            (height, width, 3)
        )
        return cv2.cvtColor(
            image,
            cv2.COLOR_RGB2BGR,
        )

    if color_format == OBFormat.BGR:
        return data.reshape(
            (height, width, 3)
        ).copy()

    if color_format == OBFormat.YUYV:
        image = data.reshape(
            (height, width, 2)
        )
        return cv2.cvtColor(
            image,
            cv2.COLOR_YUV2BGR_YUY2,
        )

    raise RuntimeError(
        f"지원하지 않는 Orbbec Color 포맷입니다: {color_format}"
    )


class Orbbec컬러캡처:
    def __init__(
        self,
        pipeline,
        device,
        width,
        height,
        stream_fps,
    ):
        self.pipeline = pipeline
        self.device = device
        self.width = int(width)
        self.height = int(height)
        self.stream_fps = int(stream_fps)
        self._released = False

    def read(self):
        if self._released:
            return False, None

        frames = self.pipeline.wait_for_frames(
            1000
        )

        if not frames:
            return False, None

        color_frame = frames.get_color_frame()

        if color_frame is None:
            return False, None

        try:
            image = Orbbec컬러프레임_BGR로_변환(
                color_frame
            )
        except Exception as exc:
            print(
                f"Orbbec Color 프레임 변환 실패: {exc}"
            )
            return False, None

        return True, image

    def release(self):
        if self._released:
            return

        self._released = True

        try:
            self.pipeline.stop()
        except Exception:
            pass


def SDK_정수속성_범위_읽기(cap, property_id):
    try:
        value_range = cap.device.get_int_property_range(
            property_id
        )
    except Exception as exc:
        print(
            f"SDK 속성 허용범위를 읽지 못했습니다: {exc}"
        )
        return None

    default_value = None
    for name in (
        "def",
        "def_",
        "default",
    ):
        if hasattr(value_range, name):
            default_value = getattr(
                value_range,
                name,
            )
            break

    return {
        "min": int(value_range.min),
        "max": int(value_range.max),
        "step": int(value_range.step),
        "default": (
            int(default_value)
            if default_value is not None
            else None
        ),
    }


def SDK_정수속성_읽기(cap, property_id):
    try:
        return int(
            cap.device.get_int_property(
                property_id
            )
        )
    except Exception:
        return "확인불가"


def 노출_허용범위_읽기(device):
    return SDK_정수속성_범위_읽기(
        device,
        OBPropertyID.OB_PROP_COLOR_EXPOSURE_INT,
    )


def 공통_카메라값_적용(cap, cam_index, settings):
    # OpenCV/V4L2의 auto_exposure=1은 수동 노출 모드였음.
    # Orbbec SDK에서는 Color AE=False가 같은 의미임.
    cap.device.set_bool_property(
        OBPropertyID.OB_PROP_COLOR_AUTO_EXPOSURE_BOOL,
        False,
    )

    cap.device.set_bool_property(
        OBPropertyID.OB_PROP_COLOR_AUTO_WHITE_BALANCE_BOOL,
        False,
    )

    cap.device.set_int_property(
        OBPropertyID.OB_PROP_COLOR_GAIN_INT,
        int(settings["gain"]),
    )

    cap.device.set_int_property(
        OBPropertyID.OB_PROP_COLOR_WHITE_BALANCE_INT,
        int(settings["white_balance_temperature"]),
    )

    time.sleep(0.7)

    actual_gain = SDK_정수속성_읽기(
        cap,
        OBPropertyID.OB_PROP_COLOR_GAIN_INT,
    )
    actual_wb = SDK_정수속성_읽기(
        cap,
        OBPropertyID.OB_PROP_COLOR_WHITE_BALANCE_INT,
    )

    print(
        f"SDK 공통 카메라값 | "
        f"Gain={actual_gain}, "
        f"WhiteBalance={actual_wb}, "
        f"Color AE=OFF, AWB=OFF"
    )


def 노출값_적용(cap, cam_index, exposure, settle_seconds):
    cap.device.set_bool_property(
        OBPropertyID.OB_PROP_COLOR_AUTO_EXPOSURE_BOOL,
        False,
    )

    cap.device.set_int_property(
        OBPropertyID.OB_PROP_COLOR_EXPOSURE_INT,
        int(exposure),
    )

    time.sleep(max(0.0, settle_seconds))

    actual = SDK_정수속성_읽기(
        cap,
        OBPropertyID.OB_PROP_COLOR_EXPOSURE_INT,
    )

    print(f"요청 노출: {exposure} / 실제 노출: {actual}")
    return actual


def SDK_컬러_프로파일_선택(
    pipeline,
    width,
    height,
    requested_fps,
):
    profiles = pipeline.get_stream_profile_list(
        OBSensorType.COLOR_SENSOR
    )

    # 원본은 8 fps를 요청했지만 Gemini 336L의 1280x800 MJPG는
    # 8 fps 프로파일이 없으므로 먼저 8을 그대로 시도하고,
    # 실패하면 가장 가까운 지원값 10 fps를 사용한다.
    fps_candidates = []

    for value in (
        int(requested_fps),
        10,
        5,
        15,
        30,
        60,
    ):
        if value not in fps_candidates:
            fps_candidates.append(value)

    errors = []

    for fps in fps_candidates:
        try:
            profile = profiles.get_video_stream_profile(
                int(width),
                int(height),
                OBFormat.MJPG,
                int(fps),
            )
            return profile, int(fps)
        except Exception as exc:
            errors.append(
                f"{fps}fps: {exc}"
            )

    raise RuntimeError(
        "1280x800 MJPG Color 프로파일을 열 수 없습니다. "
        + " | ".join(errors)
    )


def 카메라_열기(args, settings):
    pipeline = Pipeline()
    config = Config()

    color_profile, stream_fps = SDK_컬러_프로파일_선택(
        pipeline,
        args.cam_w,
        args.cam_h,
        args.fps,
    )

    depth_profiles = pipeline.get_stream_profile_list(
        OBSensorType.DEPTH_SENSOR
    )
    depth_profile = depth_profiles.get_default_video_stream_profile()

    config.enable_stream(
        color_profile
    )
    config.enable_stream(
        depth_profile
    )

    if OBFrameAggregateOutputMode is not None:
        try:
            config.set_frame_aggregate_output_mode(
                OBFrameAggregateOutputMode.FULL_FRAME_REQUIRE
            )
        except Exception:
            pass

    try:
        pipeline.enable_frame_sync()
    except Exception:
        pass

    pipeline.start(
        config
    )

    device = pipeline.get_device()

    cap = Orbbec컬러캡처(
        pipeline,
        device,
        args.cam_w,
        args.cam_h,
        stream_fps,
    )

    args.sdk_actual_fps = int(stream_fps)
    args.sdk_color_format = "MJPG"

    try:
        args.sdk_depth_width = int(depth_profile.get_width())
        args.sdk_depth_height = int(depth_profile.get_height())
        args.sdk_depth_fps = int(depth_profile.get_fps())
        args.sdk_depth_format = str(depth_profile.get_format())
    except Exception:
        args.sdk_depth_width = None
        args.sdk_depth_height = None
        args.sdk_depth_fps = None
        args.sdk_depth_format = "확인불가"

    print("")
    print("pyorbbecsdk Color + Depth 단일 Pipeline 시작")
    print(
        f"Color: {args.cam_w}x{args.cam_h} | "
        f"포맷: MJPG | "
        f"원본 요청 FPS: {args.fps} | "
        f"SDK 실제 프로파일 FPS: {stream_fps}"
    )
    if args.sdk_depth_width is not None:
        print(
            f"Depth: {args.sdk_depth_width}x{args.sdk_depth_height} | "
            f"FPS: {args.sdk_depth_fps} | "
            f"포맷: {args.sdk_depth_format}"
        )

    time.sleep(0.7)
    공통_카메라값_적용(
        cap,
        args.cam,
        settings,
    )

    return cap


def 패턴_좌표_생성(width, height, direction):
    if direction == "vertical":
        return np.tile(
            np.arange(width, dtype=np.float32),
            (height, 1),
        )

    return np.tile(
        np.arange(height, dtype=np.float32).reshape(height, 1),
        (1, width),
    )


def 사인파_밝기_생성(coord, period, base, amplitude, phase):
    pattern = base + amplitude * np.cos(
        2.0 * np.pi * coord / float(period) + phase
    )
    return np.clip(pattern, 0, 255).astype(np.uint8)


def 색상패턴_생성(gray_pattern, color_name):
    zeros = np.zeros_like(gray_pattern)

    if color_name == "white":
        return cv2.merge([gray_pattern, gray_pattern, gray_pattern])

    if color_name == "green":
        return cv2.merge([zeros, gray_pattern, zeros])

    if color_name == "red":
        return cv2.merge([zeros, zeros, gray_pattern])

    if color_name == "blue":
        return cv2.merge([gray_pattern, zeros, zeros])

    raise ValueError(f"지원하지 않는 색상입니다: {color_name}")


def 프로젝터_창_준비(monitor):
    name = "검은색 도장면 RGB 구조광 패턴 비교"

    cv2.namedWindow(name, cv2.WINDOW_NORMAL)
    black = np.zeros((monitor["h"], monitor["w"], 3), dtype=np.uint8)

    cv2.imshow(name, black)
    cv2.waitKey(500)
    cv2.moveWindow(name, monitor["x"], monitor["y"])
    cv2.waitKey(500)
    cv2.setWindowProperty(
        name,
        cv2.WND_PROP_FULLSCREEN,
        cv2.WINDOW_FULLSCREEN,
    )
    cv2.waitKey(1000)

    return name


def 패턴_안정화_대기(seconds):
    start = time.time()
    while time.time() - start < seconds:
        if (cv2.waitKey(50) & 0xFF) == ord("q"):
            raise KeyboardInterrupt


def 카메라_버퍼_비우기(cap, count):
    for _ in range(max(0, int(count))):
        cap.read()
        cv2.waitKey(5)



def 시간기준_버퍼_비우기(
    cap,
    seconds,
    extra_frames,
):
    """
    이전 위상 프레임이 첫 저장 프레임에 섞이지 않도록
    시간 기준으로 계속 읽어서 버림.
    """
    count = 0
    end = (
        time.monotonic()
        + max(
            0.0,
            float(seconds),
        )
    )

    while time.monotonic() < end:
        ret, _ = cap.read()

        if ret:
            count += 1

        key = cv2.waitKey(1) & 0xFF

        if key in (
            ord("q"),
            27,
        ):
            raise KeyboardInterrupt

    for _ in range(
        max(
            0,
            int(extra_frames),
        )
    ):
        ret, _ = cap.read()

        if ret:
            count += 1

        cv2.waitKey(1)

    return count


def 컬러_평균촬영(cap, count):
    frames = []

    for _ in range(max(1, int(count))):
        ret, frame = cap.read()
        if ret and frame is not None:
            frames.append(frame.astype(np.float32))
        cv2.waitKey(30)

    if not frames:
        raise RuntimeError("카메라 프레임 촬영에 실패했습니다.")

    average = np.mean(frames, axis=0)
    return np.clip(average, 0, 255).astype(np.uint8)


def 화면_투사후_촬영(cap, window_name, image, args):
    cv2.imshow(window_name, image)
    cv2.waitKey(200)

    # 패턴 변경 직후부터 계속 프레임을 소진해 stale frame 방지
    end = (
        time.monotonic()
        + max(
            0.0,
            float(args.delay),
        )
    )

    while time.monotonic() < end:
        cap.read()

        key = cv2.waitKey(1) & 0xFF

        if key in (
            ord("q"),
            27,
        ):
            raise KeyboardInterrupt

    시간기준_버퍼_비우기(
        cap,
        args.phase_flush_seconds,
        args.phase_extra_discard,
    )

    return 컬러_평균촬영(
        cap,
        args.avg_frames,
    )


def 물체영역_선택(frame_bgr):
    try:
        import tkinter as tk
        from PIL import Image, ImageTk
    except Exception as exc:
        raise RuntimeError(
            "물체 영역 선택에는 tkinter와 Pillow가 필요합니다. "
            f"현재 오류: {exc}"
        ) from exc

    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    original_h, original_w = rgb.shape[:2]

    scale = min(
        1.0,
        1500.0 / float(original_w),
        900.0 / float(original_h),
    )

    display_w = max(1, int(round(original_w * scale)))
    display_h = max(1, int(round(original_h * scale)))

    resized = cv2.resize(
        rgb,
        (display_w, display_h),
        interpolation=cv2.INTER_AREA,
    )

    root = tk.Tk()
    root.title("물체 영역 선택")
    root.resizable(False, False)

    tk.Label(
        root,
        text=(
            "도장면 물체만 드래그하세요. "
            "Enter 또는 Space로 확정, Esc로 취소"
        ),
        font=("Sans", 12),
        pady=8,
    ).pack()

    canvas = tk.Canvas(
        root,
        width=display_w,
        height=display_h,
        cursor="cross",
    )
    canvas.pack()

    photo = ImageTk.PhotoImage(Image.fromarray(resized))
    canvas.create_image(0, 0, anchor=tk.NW, image=photo)

    state = {
        "start_x": None,
        "start_y": None,
        "end_x": None,
        "end_y": None,
        "rect": None,
        "confirmed": False,
    }

    def on_press(event):
        state["start_x"] = max(0, min(display_w - 1, event.x))
        state["start_y"] = max(0, min(display_h - 1, event.y))
        state["end_x"] = state["start_x"]
        state["end_y"] = state["start_y"]

        if state["rect"] is not None:
            canvas.delete(state["rect"])

        state["rect"] = canvas.create_rectangle(
            state["start_x"],
            state["start_y"],
            state["end_x"],
            state["end_y"],
            outline="red",
            width=3,
        )

    def on_drag(event):
        if state["start_x"] is None:
            return

        state["end_x"] = max(0, min(display_w - 1, event.x))
        state["end_y"] = max(0, min(display_h - 1, event.y))

        canvas.coords(
            state["rect"],
            state["start_x"],
            state["start_y"],
            state["end_x"],
            state["end_y"],
        )

    def confirm(_event=None):
        if state["start_x"] is None:
            return

        if (
            abs(state["end_x"] - state["start_x"]) < 10
            or abs(state["end_y"] - state["start_y"]) < 10
        ):
            return

        state["confirmed"] = True
        root.destroy()

    def cancel(_event=None):
        root.destroy()

    canvas.bind("<ButtonPress-1>", on_press)
    canvas.bind("<B1-Motion>", on_drag)
    root.bind("<Return>", confirm)
    root.bind("<space>", confirm)
    root.bind("<Escape>", cancel)

    root.mainloop()

    if not state["confirmed"]:
        raise RuntimeError("물체 영역 선택이 취소되었습니다.")

    x1, x2 = sorted([state["start_x"], state["end_x"]])
    y1, y2 = sorted([state["start_y"], state["end_y"]])

    x1 = int(round(x1 / scale))
    x2 = int(round(x2 / scale))
    y1 = int(round(y1 / scale))
    y2 = int(round(y2 / scale))

    return (
        max(0, x1),
        max(0, y1),
        min(original_w, x2),
        min(original_h, y2),
    )


def 색상조건_촬영(
    cap,
    window_name,
    coord,
    monitor,
    args,
    folder,
    color_name,
):
    folder.mkdir(parents=True, exist_ok=True)

    gray_frames = {}
    color_frames = {}

    print(
        f"{색상_표시명[color_name]} 패턴 촬영 시작 | "
        f"노출={args.exposure}, 주기={args.period}, "
        f"base={args.base}, 진폭={args.amplitude}"
    )

    for phase, phase_name in 위상_목록:
        gray_pattern = 사인파_밝기_생성(
            coord,
            args.period,
            args.base,
            args.amplitude,
            phase,
        )
        color_pattern = 색상패턴_생성(gray_pattern, color_name)

        cv2.imwrite(
            str(folder / f"투사패턴_{phase_name}.png"),
            color_pattern,
        )

        color = 화면_투사후_촬영(
            cap,
            window_name,
            color_pattern,
            args,
        )
        gray = cv2.cvtColor(color, cv2.COLOR_BGR2GRAY)

        color_frames[phase_name] = color
        gray_frames[phase_name] = gray

        cv2.imwrite(
            str(folder / f"phase_{phase_name}_color.png"),
            color,
        )
        cv2.imwrite(
            str(folder / f"phase_{phase_name}.png"),
            gray,
        )

        print(
            f"{색상_표시명[color_name]} / "
            f"{phase_name}도 촬영 완료"
        )

    return {
        "색상": color_name,
        "컬러": color_frames,
        "회색": gray_frames,
    }


def 분석영역_마스크_생성(shape, rect, args):
    """
    구조광 품질 계산에 사용할 분석 영역을 만든다.

    - 수동 선택 코드와의 호환을 위해 기본값은 rect 사각형 전체.
    - Depth 자동 검출 통합에서는 args.자동_물체_분석마스크에
      '최종 빨간 사각형 내부 전체' 마스크를 넣어 동일 영역을 사용한다.
    """
    image_h, image_w = shape[:2]
    x1, y1, x2, y2 = rect

    rect_area = np.zeros(
        (image_h, image_w),
        dtype=bool,
    )
    rect_area[
        max(0, int(y1)):min(image_h, int(y2)),
        max(0, int(x1)):min(image_w, int(x2)),
    ] = True

    object_mask = getattr(
        args,
        "자동_물체_분석마스크",
        None,
    )

    if object_mask is None:
        return rect_area

    object_mask = np.asarray(
        object_mask,
        dtype=bool,
    )

    if object_mask.shape != rect_area.shape:
        print(
            "주의: 자동 물체 분석 마스크 크기가 영상과 달라 "
            "기존 사각형 영역으로 계산합니다. "
            f"마스크={object_mask.shape}, 영상={rect_area.shape}"
        )
        return rect_area

    area = object_mask & rect_area

    # 혹시 Depth 마스크가 비정상적으로 비어도 실험 전체가 중단되지 않도록
    # 마지막 안전장치로 기존 사각형 영역을 사용한다.
    if np.count_nonzero(area) < 100:
        print(
            "주의: 자동 전처리 사각 마스크의 유효 픽셀이 너무 적어 "
            "기존 사각형 영역으로 계산합니다."
        )
        return rect_area

    return area


def 품질_계산(frames, rect, args):
    i0 = frames["000"].astype(np.float32)
    i90 = frames["090"].astype(np.float32)
    i180 = frames["180"].astype(np.float32)
    i270 = frames["270"].astype(np.float32)

    modulation = 0.5 * np.sqrt(
        (i0 - i180) ** 2 + (i270 - i90) ** 2
    )

    max_phase = np.maximum.reduce([i0, i90, i180, i270])
    min_phase = np.minimum.reduce([i0, i90, i180, i270])

    saturation = max_phase >= float(args.saturation_threshold)
    dark = max_phase <= float(args.dark_threshold)

    valid = (
        (modulation >= float(args.modulation_threshold))
        & (~saturation)
        & (~dark)
    )

    wrapped = np.arctan2(
        i270 - i90,
        i0 - i180,
    )

    area = 분석영역_마스크_생성(
        i0.shape,
        rect,
        args,
    )

    count = int(np.count_nonzero(area))
    if count <= 0:
        raise RuntimeError("선택한 물체 영역이 비어 있습니다.")

    return {
        "평균_변조도": float(np.mean(modulation[area])),
        "중앙값_변조도": float(np.median(modulation[area])),
        "유효_위상_비율": float(
            np.count_nonzero(valid & area) / count * 100.0
        ),
        "포화_비율": float(
            np.count_nonzero(saturation & area) / count * 100.0
        ),
        "암부_비율": float(
            np.count_nonzero(dark & area) / count * 100.0
        ),
        "평균_최대밝기": float(np.mean(max_phase[area])),
        "평균_최소밝기": float(np.mean(min_phase[area])),
        "변조도_지도": modulation,
        "유효_마스크": valid,
        "래핑_위상맵": wrapped,
    }


def 변조도_시각화(modulation):
    scale = max(1.0, float(np.percentile(modulation, 99)))
    return np.clip(
        modulation / scale * 255.0,
        0,
        255,
    ).astype(np.uint8)


def 위상_시각화(wrapped):
    return np.clip(
        (wrapped + np.pi) / (2.0 * np.pi) * 255.0,
        0,
        255,
    ).astype(np.uint8)


def 품질영상_저장(folder, quality, rect, preview):
    x1, y1, x2, y2 = rect

    overlay = preview.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 0, 255), 3)
    cv2.imwrite(str(folder / "선택한_물체영역.png"), overlay)

    cv2.imwrite(
        str(folder / "변조도_지도.png"),
        변조도_시각화(quality["변조도_지도"]),
    )

    valid_vis = np.zeros(
        (*quality["유효_마스크"].shape, 3),
        dtype=np.uint8,
    )
    valid_vis[quality["유효_마스크"]] = (255, 255, 255)
    cv2.rectangle(valid_vis, (x1, y1), (x2, y2), (0, 0, 255), 2)
    cv2.imwrite(str(folder / "유효위상_마스크.png"), valid_vis)

    cv2.imwrite(
        str(folder / "래핑_위상맵.png"),
        위상_시각화(quality["래핑_위상맵"]),
    )



기본_다중조건_저장_루트 = Path(
    "/home/seoyeong/졸업작품/전처리와구조광_통합/플랫폼 바닥 따기/구조광_전처리"
)


def 인자_읽기():
    parser = argparse.ArgumentParser(
        description=(
            "검은색 도장면 다중 게인·다중 노출 자동 전처리"
        )
    )

    # 카메라
    parser.add_argument("--cam", type=int, default=8)
    parser.add_argument("--cam_w", type=int, default=1280)
    parser.add_argument("--cam_h", type=int, default=800)
    parser.add_argument("--fps", type=int, default=8)

    # 프로젝터
    parser.add_argument("--monitor", type=str, default="auto")
    parser.add_argument("--w", type=int, default=1920)
    parser.add_argument("--h", type=int, default=1080)
    parser.add_argument("--x", type=int, default=2560)
    parser.add_argument("--y", type=int, default=0)

    # 흰색 사인파 패턴
    parser.add_argument("--period", type=int, default=80)
    parser.add_argument(
        "--direction",
        type=str,
        default="horizontal",
        choices=["vertical", "horizontal"],
    )
    parser.add_argument("--base", type=float, default=128.0)
    parser.add_argument("--amplitude", type=float, default=127.0)

    # 다중 촬영 조건: 게인:노출
    parser.add_argument(
        "--conditions",
        nargs="+",
        default=[
            "48:1999",
            "64:1999",
            "48:1700",
            "48:1400",
            "48:1100",
            "32:1999",
            "32:1600",
            "24:1999",
            "24:1600",
            "16:1600",
            "16:1200",
            "16:800",
        ],
        help=(
            "촬영할 게인:노출 쌍. "
            "예: --conditions 48:1999 64:1999 32:1999 16:1200"
        ),
    )
    parser.add_argument(
        "--primary",
        type=str,
        default="48:1999",
        help="기본으로 사용할 게인:노출 조건",
    )

    # 촬영 안정화
    parser.add_argument("--delay", type=float, default=2.0)
    parser.add_argument("--exposure_settle", type=float, default=1.0)
    parser.add_argument("--gain_settle", type=float, default=1.0)
    parser.add_argument("--buffer_clear", type=int, default=20)
    parser.add_argument("--avg_frames", type=int, default=3)
    parser.add_argument(
        "--phase_flush_seconds",
        type=float,
        default=2.5,
        help=(
            "위상 변경 후 stale frame 제거를 위해 "
            "계속 프레임을 읽어 버리는 시간"
        ),
    )
    parser.add_argument(
        "--phase_extra_discard",
        type=int,
        default=5,
        help=(
            "시간 기준 flush 후 추가로 버릴 프레임 수"
        ),
    )
    parser.add_argument(
        "--fast",
        action="store_true",
        help=(
            "조건을 확정한 뒤 사용하는 고속 촬영 모드. "
            "패턴 대기 0.8초, 버퍼 제거 8프레임, "
            "위상당 1프레임 촬영으로 변경함."
        ),
    )

    # 품질 기준
    parser.add_argument("--modulation_threshold", type=float, default=15.0)
    parser.add_argument("--saturation_threshold", type=int, default=250)
    parser.add_argument("--dark_threshold", type=int, default=10)

    # 카메라 고정값
    parser.add_argument(
        "--camera_settings",
        type=str,
        default=str(기본_카메라값_파일),
    )
    parser.add_argument("--initial_gain", type=int, default=48)
    parser.add_argument("--white_balance", type=int, default=4600)

    # 저장
    parser.add_argument("--sample", type=str, default="샘플")
    parser.add_argument(
        "--condition",
        type=str,
        default="M15_프로젝터기본광설정후_다중조건융합",
    )
    parser.add_argument(
        "--out_root",
        type=str,
        default=str(기본_다중조건_저장_루트),
    )

    # ========================================================
    # 팀원 방식 Reference 위상차 + Relative Point Cloud
    # ========================================================
    parser.add_argument(
        "--reference_dir",
        type=str,
        default="/home/seoyeong/졸업작품/전처리와구조광_통합/플랫폼 바닥 따기/현재배치_기준데이터/기준촬영_20260818_191520/E480_G16/Reference_4위상",
        help=(
            "빈 플랫폼 Reference 4-step 폴더. "
            "phase_000.png, phase_090.png, phase_180.png, phase_270.png 필요."
        ),
    )
    parser.add_argument(
        "--reference_rotate_180",
        action="store_true",
        help=(
            "Reference 4장을 180도 회전해서 현재 전처리 영상 방향과 맞춤. "
            "현재 Reference 촬영본은 이 옵션을 사용하는 것을 권장."
        ),
    )
    parser.add_argument(
        "--relative_z_scale",
        type=float,
        default=40.0,
        help="Relative Point Cloud의 Z 시각화 배율(실제 mm 아님)",
    )
    parser.add_argument(
        "--relative_z_sign",
        type=float,
        default=-1.0,
        help="Relative Point Cloud Z 방향 부호",
    )
    parser.add_argument(
        "--point_skip",
        type=int,
        default=2,
        help="PLY 점 간격. 2이면 x,y 모두 2픽셀 간격",
    )
    parser.add_argument(
        "--phase_p_low",
        type=float,
        default=10.0,
        help="Phase Difference 하위 percentile clip",
    )
    parser.add_argument(
        "--phase_p_high",
        type=float,
        default=90.0,
        help="Phase Difference 상위 percentile clip",
    )
    parser.add_argument(
        "--relative_median_ksize",
        type=int,
        default=5,
        help="Relative surface Median kernel",
    )
    parser.add_argument(
        "--relative_gaussian_ksize",
        type=int,
        default=7,
        help="Relative surface Gaussian kernel",
    )
    parser.add_argument(
        "--no_relative_gaussian",
        action="store_true",
        help="Relative Point Cloud 생성 시 Gaussian smoothing 비활성화",
    )
    parser.add_argument(
        "--no_relative_plane_remove",
        action="store_true",
        help="Relative Point Cloud 생성 시 best-fit plane 제거 비활성화",
    )

    parser.add_argument(
        "--self_test",
        action="store_true",
        help="카메라/프로젝터 없이 통합 glue 로직 자체검증만 실행",
    )

    return parser.parse_args()


def 제어값_허용범위_읽기(device, control_name):
    if control_name == "gain":
        return SDK_정수속성_범위_읽기(
            device,
            OBPropertyID.OB_PROP_COLOR_GAIN_INT,
        )

    if control_name in (
        "exposure",
        "exposure_time_absolute",
    ):
        return SDK_정수속성_범위_읽기(
            device,
            OBPropertyID.OB_PROP_COLOR_EXPOSURE_INT,
        )

    return None


def 게인값_적용(cap, cam_index, gain, settle_seconds):
    cap.device.set_bool_property(
        OBPropertyID.OB_PROP_COLOR_AUTO_EXPOSURE_BOOL,
        False,
    )

    cap.device.set_int_property(
        OBPropertyID.OB_PROP_COLOR_GAIN_INT,
        int(gain),
    )

    time.sleep(max(0.0, settle_seconds))

    actual = SDK_정수속성_읽기(
        cap,
        OBPropertyID.OB_PROP_COLOR_GAIN_INT,
    )

    print(f"요청 게인: {gain} / 실제 게인: {actual}")
    return actual


def 조건문자열_해석(text):
    cleaned = str(text).strip().replace(",", ":")
    parts = cleaned.split(":")

    if len(parts) != 2:
        raise ValueError(
            f"조건 형식이 잘못되었습니다: {text} "
            "(올바른 형식 예: 48:1999)"
        )

    gain = int(parts[0])
    exposure = int(parts[1])

    return {
        "gain": gain,
        "exposure": exposure,
        "key": f"{gain}:{exposure}",
        "label": f"게인{gain}_노출{exposure}",
    }


def 조건목록_준비(args, gain_range, exposure_range):
    parsed = []
    seen = set()

    for text in args.conditions:
        item = 조건문자열_해석(text)

        if item["key"] in seen:
            continue

        if gain_range is not None:
            if not (
                gain_range["min"]
                <= item["gain"]
                <= gain_range["max"]
            ):
                print(
                    f"{item['key']} 제외: 게인 허용범위 "
                    f"{gain_range['min']}~{gain_range['max']} 밖"
                )
                continue

        if exposure_range is not None:
            if not (
                exposure_range["min"]
                <= item["exposure"]
                <= exposure_range["max"]
            ):
                print(
                    f"{item['key']} 제외: 노출 허용범위 "
                    f"{exposure_range['min']}~{exposure_range['max']} 밖"
                )
                continue

        seen.add(item["key"])
        parsed.append(item)

    primary = 조건문자열_해석(args.primary)

    if primary["key"] not in seen:
        if gain_range is not None and not (
            gain_range["min"] <= primary["gain"] <= gain_range["max"]
        ):
            raise RuntimeError("기본 조건의 게인이 허용범위 밖입니다.")

        if exposure_range is not None and not (
            exposure_range["min"]
            <= primary["exposure"]
            <= exposure_range["max"]
        ):
            raise RuntimeError("기본 조건의 노출이 허용범위 밖입니다.")

        parsed.insert(0, primary)
        seen.add(primary["key"])

    if not parsed:
        raise RuntimeError("사용 가능한 촬영 조건이 없습니다.")

    primary_index = next(
        index
        for index, item in enumerate(parsed)
        if item["key"] == primary["key"]
    )

    return parsed, primary_index


def 예상촬영시간_계산(args, condition_count):
    phase_seconds = (
        float(args.delay)
        + float(args.buffer_clear) / max(1.0, float(args.fps))
        + float(args.avg_frames) / max(1.0, float(args.fps))
        + 0.35
    )
    setting_seconds = (
        float(args.exposure_settle)
        + float(args.gain_settle)
    )
    total = condition_count * (
        4.0 * phase_seconds + setting_seconds
    )
    return total


def 흰색조건_촬영(
    cap,
    window_name,
    coord,
    monitor,
    args,
    folder,
    gain,
    exposure,
):
    folder.mkdir(parents=True, exist_ok=True)

    color_frames = {}
    gray_frames = {}

    print(
        f"4위상 촬영 시작 | 게인={gain}, 노출={exposure}, "
        f"주기={args.period}"
    )

    for phase, phase_name in 위상_목록:
        gray_pattern = 사인파_밝기_생성(
            coord,
            args.period,
            args.base,
            args.amplitude,
            phase,
        )
        color_pattern = 색상패턴_생성(
            gray_pattern,
            "white",
        )

        if not (folder / f"투사패턴_{phase_name}.png").exists():
            cv2.imwrite(
                str(folder / f"투사패턴_{phase_name}.png"),
                color_pattern,
            )

        color = 화면_투사후_촬영(
            cap,
            window_name,
            color_pattern,
            args,
        )
        gray = cv2.cvtColor(color, cv2.COLOR_BGR2GRAY)

        color_frames[phase_name] = color
        gray_frames[phase_name] = gray

        cv2.imwrite(
            str(folder / f"phase_{phase_name}_color.png"),
            color,
        )
        cv2.imwrite(
            str(folder / f"phase_{phase_name}.png"),
            gray,
        )

        print(f"  {phase_name}도 촬영 완료")

    return {
        "gain": int(gain),
        "exposure": int(exposure),
        "컬러": color_frames,
        "회색": gray_frames,
    }


def 조건품질_계산(capture, rect, args):
    gray = capture["회색"]
    color = capture["컬러"]

    i0 = gray["000"].astype(np.float32)
    i90 = gray["090"].astype(np.float32)
    i180 = gray["180"].astype(np.float32)
    i270 = gray["270"].astype(np.float32)

    modulation = 0.5 * np.sqrt(
        (i0 - i180) ** 2 + (i270 - i90) ** 2
    )

    gray_max = np.maximum.reduce([i0, i90, i180, i270])
    gray_min = np.minimum.reduce([i0, i90, i180, i270])

    channel_max_per_phase = []
    for phase_name in ["000", "090", "180", "270"]:
        frame = color[phase_name].astype(np.float32)
        channel_max_per_phase.append(np.max(frame, axis=2))

    color_max = np.maximum.reduce(channel_max_per_phase)

    saturation = color_max >= float(args.saturation_threshold)
    dark = gray_max <= float(args.dark_threshold)
    low_modulation = modulation < float(args.modulation_threshold)

    valid = (
        (~saturation)
        & (~dark)
        & (~low_modulation)
    )

    wrapped = np.arctan2(
        i270 - i90,
        i0 - i180,
    )

    area = 분석영역_마스크_생성(
        i0.shape,
        rect,
        args,
    )
    count = int(np.count_nonzero(area))

    return {
        "변조도_지도": modulation,
        "포화_마스크": saturation,
        "암부_마스크": dark,
        "저변조도_마스크": low_modulation,
        "유효_마스크": valid,
        "래핑_위상맵": wrapped,
        "물체영역_마스크": area,
        "평균_변조도": float(np.mean(modulation[area])),
        "유효_위상_비율": float(
            np.count_nonzero(valid & area) / max(1, count) * 100.0
        ),
        "포화_비율": float(
            np.count_nonzero(saturation & area) / max(1, count) * 100.0
        ),
        "암부_비율": float(
            np.count_nonzero(dark & area) / max(1, count) * 100.0
        ),
        "저변조도_비율": float(
            np.count_nonzero(
                low_modulation & (~saturation) & (~dark) & area
            )
            / max(1, count)
            * 100.0
        ),
    }


def 조건품질영상_저장(folder, quality, rect, preview):
    품질영상_저장(folder, quality, rect, preview)

    x1, y1, x2, y2 = rect

    for filename, mask in [
        ("포화_마스크.png", quality["포화_마스크"]),
        ("암부_마스크.png", quality["암부_마스크"]),
        ("저변조도_마스크.png", quality["저변조도_마스크"]),
    ]:
        vis = np.zeros((*mask.shape, 3), dtype=np.uint8)
        vis[mask] = (255, 255, 255)
        cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.imwrite(str(folder / filename), vis)


def 후보선택_순서(conditions, primary_index):
    # 기본 조건이 무효인 픽셀에서:
    # 1) 노출이 높은 조건 우선: 실제 광자를 더 많이 확보
    # 2) 같은 노출에서는 게인이 낮은 조건 우선: 잡음 증폭 억제
    candidates = [
        index
        for index in range(len(conditions))
        if index != primary_index
    ]
    return sorted(
        candidates,
        key=lambda index: (
            -conditions[index]["exposure"],
            conditions[index]["gain"],
        ),
    )


def 픽셀단위_조건선택(
    captures,
    qualities,
    conditions,
    primary_index,
):
    valid_stack = np.stack(
        [quality["유효_마스크"] for quality in qualities],
        axis=0,
    )
    saturation_stack = np.stack(
        [quality["포화_마스크"] for quality in qualities],
        axis=0,
    )
    dark_stack = np.stack(
        [quality["암부_마스크"] for quality in qualities],
        axis=0,
    )
    modulation_stack = np.stack(
        [quality["변조도_지도"] for quality in qualities],
        axis=0,
    )

    shape = valid_stack.shape[1:]
    selected_index = np.full(
        shape,
        primary_index,
        dtype=np.int16,
    )
    selected_valid = valid_stack[primary_index].copy()

    # 기본 조건이 유효한 곳은 무조건 보존한다.
    # 기본 조건이 무효인 픽셀만 다른 조건으로 보완한다.
    for candidate_index in 후보선택_순서(
        conditions,
        primary_index,
    ):
        take = (
            (~selected_valid)
            & valid_stack[candidate_index]
        )
        selected_index[take] = candidate_index
        selected_valid[take] = True

    # 어느 조건에서도 유효하지 않은 곳은 최종 무효로 남긴다.
    # 단, 육안 확인용 위상 영상에는 포화·암부가 아닌 조건 중
    # 변조도가 가장 높은 값을 넣는다.
    unresolved = ~selected_valid
    usable_stack = (~saturation_stack) & (~dark_stack)
    diagnostic_score = np.where(
        usable_stack,
        modulation_stack,
        -1.0,
    )
    diagnostic_index = np.argmax(
        diagnostic_score,
        axis=0,
    ).astype(np.int16)

    selected_index[unresolved] = diagnostic_index[unresolved]

    fused_color = {}
    fused_gray = {}

    row_index, col_index = np.indices(shape)

    for phase_name in ["000", "090", "180", "270"]:
        color_stack = np.stack(
            [
                capture["컬러"][phase_name]
                for capture in captures
            ],
            axis=0,
        )
        gray_stack = np.stack(
            [
                capture["회색"][phase_name]
                for capture in captures
            ],
            axis=0,
        )

        fused_color[phase_name] = color_stack[
            selected_index,
            row_index,
            col_index,
            :,
        ]
        fused_gray[phase_name] = gray_stack[
            selected_index,
            row_index,
            col_index,
        ]

    return {
        "선택조건번호": selected_index,
        "최종유효_마스크": selected_valid,
        "융합컬러": fused_color,
        "융합회색": fused_gray,
        "유효스택": valid_stack,
        "포화스택": saturation_stack,
        "암부스택": dark_stack,
        "변조도스택": modulation_stack,
    }


def 선택조건지도_만들기(selected_index, condition_count, rect):
    denominator = max(1, condition_count - 1)

    normalized = np.clip(
        selected_index.astype(np.float32)
        / float(denominator)
        * 255.0,
        0,
        255,
    ).astype(np.uint8)

    color = cv2.applyColorMap(
        normalized,
        cv2.COLORMAP_TURBO,
    )

    x1, y1, x2, y2 = rect
    cv2.rectangle(color, (x1, y1), (x2, y2), (255, 255, 255), 2)
    return color


def 마스크_저장(path, mask, rect):
    vis = np.zeros((*mask.shape, 3), dtype=np.uint8)
    vis[mask] = (255, 255, 255)

    x1, y1, x2, y2 = rect
    cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 0, 255), 2)
    cv2.imwrite(str(path), vis)


def 융합결과_저장(
    run_dir,
    fusion,
    fusion_quality,
    qualities,
    conditions,
    primary_index,
    rect,
    args,
    actual_settings,
    monitor,
):
    result_dir = run_dir / "전처리_결과"
    result_dir.mkdir(parents=True, exist_ok=True)

    selected_index = fusion["선택조건번호"]
    area = fusion_quality["물체영역_마스크"]
    final_valid = fusion["최종유효_마스크"] & area

    primary_quality = qualities[primary_index]
    primary_valid = primary_quality["유효_마스크"] & area
    primary_saturation = primary_quality["포화_마스크"] & area
    primary_dark = primary_quality["암부_마스크"] & area
    primary_low_modulation = (
        primary_quality["저변조도_마스크"]
        & (~primary_quality["포화_마스크"])
        & (~primary_quality["암부_마스크"])
        & area
    )

    recovered = final_valid & (~primary_valid)
    recovered_saturation = recovered & primary_saturation
    recovered_dark = recovered & primary_dark
    recovered_low_modulation = recovered & primary_low_modulation

    area_count = max(1, int(np.count_nonzero(area)))

    for phase_name in ["000", "090", "180", "270"]:
        color = fusion["융합컬러"][phase_name]
        gray = fusion["융합회색"][phase_name]

        cv2.imwrite(
            str(result_dir / f"phase_{phase_name}_color.png"),
            color,
        )
        cv2.imwrite(
            str(result_dir / f"phase_{phase_name}.png"),
            gray,
        )

    마스크_저장(
        result_dir / "유효위상_마스크.png",
        fusion["최종유효_마스크"],
        rect,
    )
    마스크_저장(
        result_dir / "기본조건_유효위상_마스크.png",
        primary_quality["유효_마스크"],
        rect,
    )
    마스크_저장(
        result_dir / "기본조건_포화_마스크.png",
        primary_quality["포화_마스크"],
        rect,
    )
    마스크_저장(
        result_dir / "포화영역_복구_마스크.png",
        recovered_saturation,
        rect,
    )
    마스크_저장(
        result_dir / "저변조도영역_복구_마스크.png",
        recovered_low_modulation,
        rect,
    )

    cv2.imwrite(
        str(result_dir / "변조도_지도.png"),
        변조도_시각화(fusion_quality["변조도_지도"]),
    )
    cv2.imwrite(
        str(result_dir / "래핑_위상맵.png"),
        위상_시각화(fusion_quality["래핑_위상맵"]),
    )

    cv2.imwrite(
        str(result_dir / "선택조건_지도.png"),
        선택조건지도_만들기(
            selected_index,
            len(conditions),
            rect,
        ),
    )
    cv2.imwrite(
        str(result_dir / "선택조건번호_16비트.png"),
        selected_index.astype(np.uint16),
    )

    condition_rows = []
    for index, condition in enumerate(conditions):
        selected_area = (
            (selected_index == index)
            & area
        )
        selected_valid_area = (
            selected_area
            & fusion["최종유효_마스크"]
        )

        condition_rows.append(
            {
                "조건번호": index,
                "게인": condition["gain"],
                "노출": condition["exposure"],
                "기본조건": (
                    "예" if index == primary_index else "아니오"
                ),
                "전체물체영역_선택비율": (
                    np.count_nonzero(selected_area)
                    / area_count
                    * 100.0
                ),
                "유효영역_선택비율": (
                    np.count_nonzero(selected_valid_area)
                    / area_count
                    * 100.0
                ),
                "개별조건_유효비율": qualities[index][
                    "유효_위상_비율"
                ],
                "개별조건_포화비율": qualities[index][
                    "포화_비율"
                ],
                "개별조건_평균변조도": qualities[index][
                    "평균_변조도"
                ],
            }
        )

    with (result_dir / "선택조건표.csv").open(
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(condition_rows[0].keys()),
        )
        writer.writeheader()
        writer.writerows(condition_rows)

    summary = {
        "기본조건": conditions[primary_index],
        "기본조건_유효비율": (
            np.count_nonzero(primary_valid)
            / area_count
            * 100.0
        ),
        "기본조건_포화비율": (
            np.count_nonzero(primary_saturation)
            / area_count
            * 100.0
        ),
        "기본조건_암부비율": (
            np.count_nonzero(primary_dark)
            / area_count
            * 100.0
        ),
        "기본조건_저변조도비율": (
            np.count_nonzero(primary_low_modulation)
            / area_count
            * 100.0
        ),
        "융합_유효비율": (
            np.count_nonzero(final_valid)
            / area_count
            * 100.0
        ),
        "융합_평균변조도": float(
            np.mean(
                fusion_quality["변조도_지도"][area]
            )
        ),
        "전체_복구비율": (
            np.count_nonzero(recovered)
            / area_count
            * 100.0
        ),
        "포화영역_복구비율_전체기준": (
            np.count_nonzero(recovered_saturation)
            / area_count
            * 100.0
        ),
        "포화영역_복구율": (
            np.count_nonzero(recovered_saturation)
            / max(1, np.count_nonzero(primary_saturation))
            * 100.0
        ),
        "저변조도영역_복구비율_전체기준": (
            np.count_nonzero(recovered_low_modulation)
            / area_count
            * 100.0
        ),
        "저변조도영역_복구율": (
            np.count_nonzero(recovered_low_modulation)
            / max(1, np.count_nonzero(primary_low_modulation))
            * 100.0
        ),
        "암부영역_복구비율_전체기준": (
            np.count_nonzero(recovered_dark)
            / area_count
            * 100.0
        ),
    }

    report_lines = [
        "검은색 도장면 다중 게인·다중 노출 4위상 융합 결과",
        "=" * 70,
        (
            f"기본 조건: 게인 {conditions[primary_index]['gain']}, "
            f"노출 {conditions[primary_index]['exposure']}"
        ),
        f"기본 조건 유효 비율: {summary['기본조건_유효비율']:.2f}%",
        f"기본 조건 포화 비율: {summary['기본조건_포화비율']:.2f}%",
        (
            f"기본 조건 저변조도 비율: "
            f"{summary['기본조건_저변조도비율']:.2f}%"
        ),
        "",
        f"융합 유효 비율: {summary['융합_유효비율']:.2f}%",
        f"융합 평균 변조도: {summary['융합_평균변조도']:.2f}",
        f"전체 복구 비율: {summary['전체_복구비율']:.2f}%p",
        (
            f"기본 조건 포화 픽셀 복구율: "
            f"{summary['포화영역_복구율']:.2f}%"
        ),
        (
            f"기본 조건 저변조도 픽셀 복구율: "
            f"{summary['저변조도영역_복구율']:.2f}%"
        ),
        "",
        "선택 원칙",
        "1. 기본 조건이 유효하면 기본 조건의 네 위상 값을 그대로 사용함.",
        (
            "2. 기본 조건이 무효일 때만 다른 조건을 사용하며, "
            "노출이 높은 조건을 우선하고 같은 노출에서는 낮은 게인을 우선함."
        ),
        (
            "3. 한 픽셀의 0·90·180·270도 값은 항상 동일한 "
            "게인·노출 조건에서 가져옴."
        ),
        (
            "4. 모든 조건에서 무효인 픽셀은 최종 유효 마스크에서 "
            "무효로 유지함."
        ),
        "",
        f"최종 전처리 결과 폴더: {result_dir}",
    ]

    (result_dir / "전처리_결과.txt").write_text(
        "\n".join(report_lines),
        encoding="utf-8",
    )

    metadata = {
        "실행시각": datetime.now().isoformat(timespec="seconds"),
        "샘플": args.sample,
        "실험조건명": args.condition,
        "프로젝터": monitor,
        "패턴": {
            "색상": "흰색",
            "주기": args.period,
            "방향": args.direction,
            "base": args.base,
            "진폭": args.amplitude,
        },
        "카메라": {
            "장치": "pyorbbecsdk / Gemini Color",
            "해상도": [args.cam_w, args.cam_h],
            "촬영속도_원본요청값": args.fps,
            "촬영속도_SDK프로파일": getattr(args, "sdk_actual_fps", None),
            "컬러포맷": getattr(args, "sdk_color_format", "MJPG"),
            "실제설정": actual_settings,
            "화이트밸런스": args.white_balance,
        },
        "품질기준": {
            "변조도_임시기준": args.modulation_threshold,
            "포화기준_채널최댓값": args.saturation_threshold,
            "암부기준": args.dark_threshold,
        },
        "물체영역": {
            "x1": rect[0],
            "y1": rect[1],
            "x2": rect[2],
            "y2": rect[3],
        },
        "조건목록": conditions,
        "기본조건번호": primary_index,
        "결과요약": summary,
        "조건별선택결과": condition_rows,
    }

    (result_dir / "실험정보.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    return summary, condition_rows, result_dir



카메라_180도_회전 = True

표면_목록 = [
    ("흰색", "흰색_도장면"),
    ("회색", "회색_도장면"),
    ("빨간색", "빨간색_도장면"),
    ("검은색", "검은색_도장면"),
]

기준_게인 = 16
기준_노출 = 156

낮은_노출 = 78
높은_노출목록 = [250, 400, 600]

광응답_프로브값 = 64
기본_유효목표 = 90.0

# 임시 광응답 구간
# 흰색 보정 광응답 중앙값을 100으로 정규화
고반응_경계 = 70.0
저반응_경계 = 30.0


def 필요시_회전(frame):
    if 카메라_180도_회전:
        return cv2.rotate(
            frame,
            cv2.ROTATE_180,
        )
    return frame


def 화면투사후_촬영_회전(
    cap,
    window_name,
    image,
    args,
):
    frame = 화면_투사후_촬영(
        cap,
        window_name,
        image,
        args,
    )
    return 필요시_회전(frame)


def 흰색조건_촬영_회전(
    cap,
    window_name,
    coord,
    monitor,
    args,
    folder,
    gain,
    exposure,
):
    folder.mkdir(
        parents=True,
        exist_ok=True,
    )

    color_frames = {}
    gray_frames = {}

    print(
        f"4위상 촬영 시작 | 게인={gain}, "
        f"노출={exposure}, 주기={args.period}"
    )

    for phase, phase_name in 위상_목록:
        gray_pattern = 사인파_밝기_생성(
            coord,
            args.period,
            args.base,
            args.amplitude,
            phase,
        )

        color_pattern = 색상패턴_생성(
            gray_pattern,
            "white",
        )

        cv2.imwrite(
            str(
                folder
                / f"투사패턴_{phase_name}.png"
            ),
            color_pattern,
        )

        color = 화면투사후_촬영_회전(
            cap,
            window_name,
            color_pattern,
            args,
        )

        gray = cv2.cvtColor(
            color,
            cv2.COLOR_BGR2GRAY,
        )

        color_frames[
            phase_name
        ] = color

        gray_frames[
            phase_name
        ] = gray

        cv2.imwrite(
            str(
                folder
                / f"phase_{phase_name}_color.png"
            ),
            color,
        )

        cv2.imwrite(
            str(
                folder
                / f"phase_{phase_name}.png"
            ),
            gray,
        )

        print(
            f"  {phase_name}도 촬영 완료"
        )

    return {
        "gain": int(gain),
        "exposure": int(exposure),
        "컬러": color_frames,
        "회색": gray_frames,
    }


def 광응답_계산(
    black_color,
    probe_color,
    rect,
):
    black_gray = cv2.cvtColor(
        black_color,
        cv2.COLOR_BGR2GRAY,
    ).astype(np.float32)

    probe_gray = cv2.cvtColor(
        probe_color,
        cv2.COLOR_BGR2GRAY,
    ).astype(np.float32)

    x1, y1, x2, y2 = rect

    area = np.zeros(
        black_gray.shape,
        dtype=bool,
    )

    area[
        y1:y2,
        x1:x2,
    ] = True

    corrected = np.clip(
        probe_gray - black_gray,
        0.0,
        None,
    )

    values = corrected[area]

    return {
        "기본광평균": float(
            np.mean(
                black_gray[area]
            )
        ),
        "프로브평균": float(
            np.mean(
                probe_gray[area]
            )
        ),
        "보정광응답평균": float(
            np.mean(values)
        ),
        "보정광응답중앙값": float(
            np.median(values)
        ),
        "보정광응답P90": float(
            np.percentile(
                values,
                90,
            )
        ),
    }


def 노출추가_결정(
    quality,
):
    valid = quality[
        "유효_위상_비율"
    ]
    saturation = quality[
        "포화_비율"
    ]
    low_mod = quality[
        "저변조도_비율"
    ]

    if valid >= 기본_유효목표:
        return []

    extra = []

    # 포화가 눈에 띄면 기준보다 낮은 노출을 추가
    if saturation >= 2.0:
        extra.append(
            낮은_노출
        )

    # 저변조도가 남으면 게인은 올리지 않고
    # 노출만 단계적으로 늘림
    if low_mod >= 2.0:
        extra.extend(
            높은_노출목록
        )

    if not extra:
        # 유효율이 90% 미만인데 원인이 뚜렷하지 않으면
        # 노출만 넓게 확인
        extra = [
            낮은_노출,
            250,
            400,
            600,
        ]

    result = []

    for value in extra:
        if (
            value != 기준_노출
            and value not in result
        ):
            result.append(value)

    return result


def 광응답그룹(index):
    if index >= 고반응_경계:
        return "고반응"
    if index >= 저반응_경계:
        return "중반응"
    return "저반응"



# ============================================================
# 9단계: 단일 도장면 유효영역 95% 자동 탐색
# - 시작: 팀원 기준 Gain 16 / Exposure 156
# - 먼저 Exposure만 조절/융합
# - 그래도 95% 미만이면 고Gain 조건 추가
# - 목표를 달성하면 즉시 종료
# - 후보를 모두 써도 95% 미만이면 무한 반복하지 않고 종료
# ============================================================

목표_유효율 = 95.0
시작_게인 = 16
시작_노출 = 156

# 밝은 표면에서 포화가 주원인일 때 먼저 시도
낮은_노출_후보 = [
    100,
    78,
    60,
    50,
    40,
    30,
    20,
]

# 어두운 표면에서 저변조도가 주원인일 때 먼저 시도
높은_노출_후보 = [
    250,
    400,
    600,
    800,
    1200,
    1600,
    1999,
]

# Exposure-only로 95%가 안 될 때만 사용
# 회색/중간 반응 표면에서는 E1999 고정이 아니라
# Exposure-only 결과 중 "포화가 낮고 유효율이 높은 노출"을 기준으로 Gain을 올림.
# 기본 E156에서 포화가 높았던 표면은,
# 먼저 "높은 노출 + 낮은 Gain" 조합도 확인한다.
낮은_게인_후보 = [
    12,
    8,
    4,
    0,
]

고게인_후보 = [
    24,
    32,
    48,
    64,
    80,
    96,
    112,
]

저게인_시도_포화기준 = 5.0

# 저Gain에서 이 정도보다 적게 좋아지는 상태가 연속되면
# 더 낮은 Gain을 계속 시험하지 않고 즉시 고Gain 단계로 넘어감.
저게인_최소개선량 = 0.10
저게인_정체연속횟수 = 2

# 고Gain 시작 Exposure를 고를 때 허용할 포화율
고게인_기준_최대포화 = 5.0

# 고Gain 조건 하나를 찍었을 때 포화가 너무 커지면
# 같은 Gain에서 한 단계 낮은 Exposure도 추가로 시험.
고게인_포화보정기준 = 8.0

# 전체 탐색에서 개선이 거의 없는 상태가 계속되면 종료.
전체_최소개선량 = 0.05
전체_정체연속횟수 = 3


def 사용가능_노출목록_만들기(
    device,
    values,
):
    info = 노출_허용범위_읽기(
        device
    )

    if info is None:
        return list(values)

    result = []

    for value in values:
        if (
            info["min"]
            <= value
            <= info["max"]
        ):
            result.append(value)

    return result


def 사용가능_게인목록_만들기(
    device,
    values,
):
    info = 제어값_허용범위_읽기(
        device,
        "gain",
    )

    if info is None:
        return list(values)

    result = []

    for value in values:
        if (
            info["min"]
            <= value
            <= info["max"]
        ):
            result.append(value)

    return result


def 현재융합_계산(
    captures,
    qualities,
    conditions,
    rect,
    args,
):
    fusion = 픽셀단위_조건선택(
        captures,
        qualities,
        conditions,
        0,
    )

    fused_capture = {
        "gain": -1,
        "exposure": -1,
        "컬러": fusion["융합컬러"],
        "회색": fusion["융합회색"],
    }

    fusion_quality = 조건품질_계산(
        fused_capture,
        rect,
        args,
    )

    area = fusion_quality[
        "물체영역_마스크"
    ]

    final_valid = (
        fusion["최종유효_마스크"]
        & area
    )

    area_count = max(
        1,
        int(
            np.count_nonzero(
                area
            )
        ),
    )

    valid_ratio = (
        np.count_nonzero(
            final_valid
        )
        / area_count
        * 100.0
    )

    return (
        fusion,
        fusion_quality,
        float(valid_ratio),
    )


def 조건_추가촬영(
    cap,
    window_name,
    coord,
    monitor,
    args,
    run_dir,
    rect,
    preview,
    captures,
    qualities,
    conditions,
    gain,
    exposure,
):
    key = f"{gain}:{exposure}"

    if any(
        item["key"] == key
        for item in conditions
    ):
        return None

    print("")
    print("=" * 78)
    print(
        f"추가 조건 촬영 | "
        f"Gain={gain}, "
        f"Exposure={exposure}"
    )
    print("=" * 78)

    actual_gain = 게인값_적용(
        cap,
        args.cam,
        gain,
        args.gain_settle,
    )

    actual_exposure = 노출값_적용(
        cap,
        args.cam,
        exposure,
        args.exposure_settle,
    )

    folder = (
        run_dir
        / f"G{gain}_E{exposure}"
    )

    capture = 흰색조건_촬영_회전(
        cap,
        window_name,
        coord,
        monitor,
        args,
        folder,
        gain,
        exposure,
    )

    quality = 조건품질_계산(
        capture,
        rect,
        args,
    )

    조건품질영상_저장(
        folder,
        quality,
        rect,
        preview,
    )

    condition = {
        "gain": int(gain),
        "exposure": int(exposure),
        "key": key,
        "label": (
            f"게인{gain}_노출{exposure}"
        ),
    }

    captures.append(
        capture
    )
    qualities.append(
        quality
    )
    conditions.append(
        condition
    )

    print(
        f"개별 조건 결과 | "
        f"유효 "
        f"{quality['유효_위상_비율']:.2f}% | "
        f"포화 "
        f"{quality['포화_비율']:.2f}% | "
        f"저변조도 "
        f"{quality['저변조도_비율']:.2f}% | "
        f"암부 "
        f"{quality['암부_비율']:.2f}% | "
        f"평균 M "
        f"{quality['평균_변조도']:.2f}"
    )

    return {
        "actual_gain": actual_gain,
        "actual_exposure": actual_exposure,
    }



def 게인기준노출_선택(
    qualities,
    conditions,
):
    """
    Gain=16으로 촬영한 Exposure-only 결과 중
    포화가 낮으면서 개별 유효율이 높은 조건을 선택한다.

    우선순위
    1) 포화 <= 고게인_기준_최대포화 조건
    2) 개별 유효율 최대
    3) 저변조도 최소
    4) 평균 M 최대

    포화 <= 기준 조건이 하나도 없으면
    전체 중 개별 유효율이 가장 높은 조건을 사용한다.
    """
    candidates = []

    for index, (quality, condition) in enumerate(
        zip(
            qualities,
            conditions,
        )
    ):
        if int(condition["gain"]) != 시작_게인:
            continue

        candidates.append(
            {
                "index": index,
                "gain": int(
                    condition["gain"]
                ),
                "exposure": int(
                    condition["exposure"]
                ),
                "유효": float(
                    quality[
                        "유효_위상_비율"
                    ]
                ),
                "포화": float(
                    quality[
                        "포화_비율"
                    ]
                ),
                "저변조도": float(
                    quality[
                        "저변조도_비율"
                    ]
                ),
                "평균M": float(
                    quality[
                        "평균_변조도"
                    ]
                ),
            }
        )

    if not candidates:
        return 시작_노출

    safe = [
        item
        for item in candidates
        if (
            item["포화"]
            <= 고게인_기준_최대포화
        )
    ]

    pool = safe if safe else candidates

    selected = sorted(
        pool,
        key=lambda item: (
            -item["유효"],
            item["저변조도"],
            -item["평균M"],
            item["포화"],
        ),
    )[0]

    print("")
    print(
        "고Gain 기준 Exposure 자동 선택"
    )
    print(
        f"→ E{selected['exposure']} | "
        f"개별 유효 {selected['유효']:.2f}% | "
        f"포화 {selected['포화']:.2f}% | "
        f"저변조도 {selected['저변조도']:.2f}% | "
        f"M {selected['평균M']:.2f}"
    )

    return int(
        selected["exposure"]
    )


def 융합개선량(
    이전,
    현재,
):
    return float(현재 - 이전)



# ============================================================
# 13단계: RGB 초기광응답 + 양방향/2축 적응형 고속 탐색
#
# 핵심:
# 1) RGB0 + 균일 RGB64를 먼저 촬영하여 초기 방향을 정함.
# 2) 기본 4위상 G16/E156 결과로 초기 판단을 보정함.
# 3) Exposure/Gain을 "한 방향으로 고정"하지 않음.
# 4) 현재 방향이 정체되면 즉시 반대 방향/다른 축으로 전환.
# 5) 큰 간격의 후보만 사용하여 정밀 최적화보다 빠른 탐색을 우선.
# 6) 각 픽셀은 반드시 한 촬영조건의 4위상 세트를 통째로 사용.
# 7) 최종 융합 유효율 95% 이상이면 즉시 종료.
#
# 주의:
# 광학적 가림, 극단적 정반사, 카메라/프로젝터 사각 등으로
# 물리적으로 유효 신호가 없는 픽셀은 Gain/Exposure만으로
# 항상 95% 이상을 보장할 수 없음.
# ============================================================

목표_유효율_13 = 95.0
시작_게인_13 = 16
시작_노출_13 = 156
광응답_프로브값_13 = 64

# 정밀화 단계가 아니므로 간격을 크게 둠.
노출_앵커_13 = [
    20,
    60,
    156,
    400,
    900,
    1400,
    1999,
]

게인_앵커_13 = [
    16,
    32,
    64,
    96,
    128,
]

최대_촬영조건수_13 = 16

# 융합 유효율 증가량 기준
좋은_개선량_13 = 0.50
정체_개선량_13 = 0.10
정체_연속횟수_13 = 2

# 품질 원인 판정
포화_높음_13 = 8.0
저변조도_높음_13 = 10.0


def 범위내_앵커_13(
    values,
    control_range,
):
    if control_range is None:
        return sorted(
            set(
                int(v)
                for v in values
            )
        )

    result = [
        int(v)
        for v in values
        if (
            control_range["min"]
            <= int(v)
            <= control_range["max"]
        )
    ]

    return sorted(
        set(result)
    )


def 가장가까운_인덱스_13(
    values,
    target,
):
    return min(
        range(len(values)),
        key=lambda i: abs(
            values[i] - target
        ),
    )


def 균일광_초기측정_13(
    cap,
    window_name,
    monitor,
    args,
    rect,
    save_dir,
):
    save_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    black_pattern = np.zeros(
        (
            monitor["h"],
            monitor["w"],
            3,
        ),
        dtype=np.uint8,
    )

    probe_pattern = np.full(
        (
            monitor["h"],
            monitor["w"],
            3,
        ),
        광응답_프로브값_13,
        dtype=np.uint8,
    )

    print("")
    print(
        "초기 광응답 측정: "
        "RGB0 → RGB64"
    )

    black = 화면투사후_촬영_회전(
        cap,
        window_name,
        black_pattern,
        args,
    )

    probe = 화면투사후_촬영_회전(
        cap,
        window_name,
        probe_pattern,
        args,
    )

    cv2.imwrite(
        str(
            save_dir
            / "RGB000_기본광.png"
        ),
        black,
    )

    cv2.imwrite(
        str(
            save_dir
            / "RGB064_균일광.png"
        ),
        probe,
    )

    # 자동 Depth 통합에서는 사각형 전체가 아니라 실제 물체 픽셀만 사용.
    # 수동 영역 코드에서는 자동 마스크가 없으므로 기존 rect 전체와 동일하다.
    area = 분석영역_마스크_생성(
        black.shape[:2],
        rect,
        args,
    )

    black_pixels = black[area].astype(
        np.float32
    )
    probe_pixels = probe[area].astype(
        np.float32
    )

    if black_pixels.size == 0 or probe_pixels.size == 0:
        raise RuntimeError(
            "초기 광응답 계산용 물체 영역이 비어 있습니다."
        )

    corrected = np.clip(
        probe_pixels - black_pixels,
        0.0,
        None,
    )

    # BGR 기준
    b = corrected[:, 0]
    g = corrected[:, 1]
    r = corrected[:, 2]

    corrected_gray = (
        0.114 * b
        + 0.587 * g
        + 0.299 * r
    )

    black_gray_full = cv2.cvtColor(
        black,
        cv2.COLOR_BGR2GRAY,
    ).astype(np.float32)

    probe_gray_full = cv2.cvtColor(
        probe,
        cv2.COLOR_BGR2GRAY,
    ).astype(np.float32)

    black_gray = black_gray_full[area]
    probe_gray = probe_gray_full[area]

    raw_max = np.max(
        probe_pixels,
        axis=1,
    )

    count = max(
        1,
        probe_gray.size,
    )

    channel_medians = {
        "B": float(np.median(b)),
        "G": float(np.median(g)),
        "R": float(np.median(r)),
    }

    min_channel = max(
        1.0,
        min(channel_medians.values()),
    )

    color_imbalance = (
        max(channel_medians.values())
        / min_channel
    )

    stats = {
        "RGB0_회색중앙값": float(
            np.median(black_gray)
        ),
        "RGB64_회색중앙값": float(
            np.median(probe_gray)
        ),
        "RGB64_회색P95": float(
            np.percentile(
                probe_gray,
                95,
            )
        ),
        "RGB64_포화비율": float(
            np.count_nonzero(
                raw_max >= 250
            )
            / count
            * 100.0
        ),
        "보정광응답_중앙값": float(
            np.median(corrected_gray)
        ),
        "보정광응답_P90": float(
            np.percentile(
                corrected_gray,
                90,
            )
        ),
        "보정_B_중앙값": channel_medians["B"],
        "보정_G_중앙값": channel_medians["G"],
        "보정_R_중앙값": channel_medians["R"],
        "채널불균형비": float(
            color_imbalance
        ),
    }

    print(
        "초기 균일광 결과 | "
        f"RGB64 중앙값 "
        f"{stats['RGB64_회색중앙값']:.2f} | "
        f"P95 "
        f"{stats['RGB64_회색P95']:.2f} | "
        f"포화 "
        f"{stats['RGB64_포화비율']:.2f}%"
    )

    print(
        "보정 광응답 | "
        f"중앙값 "
        f"{stats['보정광응답_중앙값']:.2f} | "
        f"P90 "
        f"{stats['보정광응답_P90']:.2f}"
    )

    print(
        "보정 채널 중앙값 | "
        f"R={stats['보정_R_중앙값']:.2f} | "
        f"G={stats['보정_G_중앙값']:.2f} | "
        f"B={stats['보정_B_중앙값']:.2f} | "
        f"불균형비="
        f"{stats['채널불균형비']:.2f}"
    )

    return stats




def 초기방향_판정_13(
    probe_stats,
    base_quality,
):
    """
    RGB 균일광을 1차 판단으로 사용하고,
    기본 4위상 품질을 2차 확인으로 사용.

    반환:
    - 밝음: 노출/게인을 낮추는 후보 우선
    - 어두움: 노출/게인을 높이는 후보 우선
    - 혼합: 양방향과 2축을 동시에 빠르게 탐색
    """

    probe_sat = probe_stats[
        "RGB64_포화비율"
    ]

    probe_p95 = probe_stats[
        "RGB64_회색P95"
    ]

    response = probe_stats[
        "보정광응답_중앙값"
    ]

    imbalance = probe_stats[
        "채널불균형비"
    ]

    base_sat = base_quality[
        "포화_비율"
    ]

    base_low = base_quality[
        "저변조도_비율"
    ]

    # 먼저 균일광으로 초기 방향.
    if (
        probe_sat >= 1.0
        or probe_p95 >= 220.0
    ):
        direction = "밝음"

    elif (
        response <= 18.0
        and probe_p95 <= 100.0
    ):
        direction = "어두움"

    else:
        direction = "혼합"

    # 색 선택적 반사가 강하면 한 방향 단정 대신 양방향.
    if imbalance >= 1.8:
        direction = "혼합"

    # 기본 4위상에서 원인이 아주 명확할 때만 보정.
    if (
        base_sat >= 포화_높음_13
        and base_sat
        >= base_low * 1.5
    ):
        direction = "밝음"

    elif (
        base_low >= 저변조도_높음_13
        and base_low
        >= base_sat * 1.5
    ):
        direction = "어두움"

    return direction


def 후보키_13(
    gain,
    exposure,
):
    return (
        int(gain),
        int(exposure),
    )


def 후보추가_13(
    queue,
    queued,
    visited,
    gain,
    exposure,
    gain_values,
    exposure_values,
    priority,
    reason,
):
    gain = int(gain)
    exposure = int(exposure)

    if gain not in gain_values:
        return

    if exposure not in exposure_values:
        return

    key = 후보키_13(
        gain,
        exposure,
    )

    if key in visited:
        return

    if key in queued:
        return

    queued.add(
        key
    )

    queue.append(
        {
            "gain": gain,
            "exposure": exposure,
            "priority": float(
                priority
            ),
            "reason": str(
                reason
            ),
        }
    )


def 큐꺼내기_13(
    queue,
    queued,
):
    queue.sort(
        key=lambda item: (
            item["priority"],
            abs(
                item["gain"]
                - 시작_게인_13
            ),
            abs(
                item["exposure"]
                - 시작_노출_13
            ),
        )
    )

    item = queue.pop(
        0
    )

    queued.discard(
        후보키_13(
            item["gain"],
            item["exposure"],
        )
    )

    return item


def 시작후보_구성_13(
    direction,
    queue,
    queued,
    visited,
    gains,
    exposures,
):
    """
    초기 판정에 따라 탐색 방향을 고정한다.

    밝음:
      - G16을 기본으로 유지하면서 E60을 먼저 사용한다.
      - E60으로 95%가 안 될 때만 E400을 융합 보완용으로 1회 사용한다.
      - 그 뒤 E20, 마지막 G16/E60까지만 확인한다.
      - 높은 Gain으로 올라가는 후보는 만들지 않는다.

    어두움:
      - Exposure를 900→1400→1999 방향으로 먼저 올리고 Gain을 32→64→96→128 방향으로만 올린다.
      - 낮은 Gain으로 되돌아가는 후보는 만들지 않는다.
      - 각 높은 Gain에서는 E1400/E1999를 확인한다.

    혼합:
      - 기존 2D 탐색의 시작 후보를 그대로 사용한다.
    """

    def add(g, e, p, reason):
        후보추가_13(
            queue,
            queued,
            visited,
            int(g),
            int(e),
            gains,
            exposures,
            float(p),
            reason,
        )

    def nearest_gain(target):
        return gains[
            가장가까운_인덱스_13(
                gains,
                target,
            )
        ]

    def nearest_exposure(target):
        return exposures[
            가장가까운_인덱스_13(
                exposures,
                target,
            )
        ]

    base_g = nearest_gain(시작_게인_13)
    base_e = nearest_exposure(시작_노출_13)

    if direction == "밝음":
        bright_plan = [
            (
                base_g,
                nearest_exposure(60),
                "밝음 고정 → Exposure 60으로 낮춤",
            ),
            (
                base_g,
                nearest_exposure(400),
                "밝음 융합 보완 1회 → Exposure 400",
            ),
            (
                base_g,
                nearest_exposure(20),
                "밝음 고정 → Exposure 20으로 더 낮춤",
            ),
            (
                nearest_gain(16),
                nearest_exposure(60),
                "밝음 마지막 보완 → 낮은 Gain + Exposure 60",
            ),
        ]

        for priority, (gain, exposure, reason) in enumerate(bright_plan):
            add(gain, exposure, priority, reason)

        return

    if direction == "어두움":
        dark_plan = [
            (
                base_g,
                nearest_exposure(900),
                "어두움 고정 → Exposure 900으로 높임",
            ),
            (
                base_g,
                nearest_exposure(1400),
                "어두움 고정 → Exposure 1400으로 높임",
            ),
            (
                base_g,
                nearest_exposure(1999),
                "어두움 고정 → Exposure 1999로 높임",
            ),
            (
                nearest_gain(32),
                nearest_exposure(1400),
                "어두움 고정 → Gain 32로 높이고 Exposure 1400",
            ),
            (
                nearest_gain(32),
                nearest_exposure(1999),
                "어두움 고정 → Gain 32에서 Exposure 1999",
            ),
            (
                nearest_gain(64),
                nearest_exposure(1400),
                "어두움 고정 → Gain 64로 높이고 Exposure 1400",
            ),
            (
                nearest_gain(64),
                nearest_exposure(1999),
                "어두움 고정 → Gain 64에서 Exposure 1999",
            ),
            (
                nearest_gain(96),
                nearest_exposure(1400),
                "어두움 고정 → Gain 96으로 높이고 Exposure 1400",
            ),
            (
                nearest_gain(96),
                nearest_exposure(1999),
                "어두움 고정 → Gain 96에서 Exposure 1999",
            ),
            (
                nearest_gain(128),
                nearest_exposure(1400),
                "어두움 고정 → Gain 128로 높이고 Exposure 1400",
            ),
            (
                nearest_gain(128),
                nearest_exposure(1999),
                "어두움 고정 → Gain 128에서 Exposure 1999",
            ),
        ]

        for priority, (gain, exposure, reason) in enumerate(dark_plan):
            add(gain, exposure, priority, reason)

        return

    # 혼합 판정만 기존 시작 후보 사용
    low_e = nearest_exposure(60)
    high_e = nearest_exposure(900)
    low_g = nearest_gain(16)
    high_g = nearest_gain(64)

    add(base_g, low_e, 0, "혼합 → 저노출 방향 확인")
    add(base_g, high_e, 0, "혼합 → 고노출 방향 확인")
    add(low_g, base_e, 1, "혼합 → 저Gain 방향 확인")
    add(high_g, base_e, 1, "혼합 → 고Gain 방향 확인")
    add(high_g, low_e, 2, "혼합 → 저노출+고Gain 조합 확인")

def 주변후보_생성_13(
    condition,
    quality,
    improvement,
    queue,
    queued,
    visited,
    gains,
    exposures,
):
    g = int(
        condition["gain"]
    )
    e = int(
        condition["exposure"]
    )

    gi = 가장가까운_인덱스_13(
        gains,
        g,
    )

    ei = 가장가까운_인덱스_13(
        exposures,
        e,
    )

    sat = float(
        quality["포화_비율"]
    )

    low = float(
        quality["저변조도_비율"]
    )

    # 개선이 좋았던 조건의 주변은 우선순위를 높임.
    if improvement >= 좋은_개선량_13:
        base_priority = 0.0
    elif improvement >= 정체_개선량_13:
        base_priority = 1.0
    else:
        base_priority = 2.0

    def add_idx(
        g_index,
        e_index,
        offset,
        reason,
    ):
        if not (
            0
            <= g_index
            < len(gains)
        ):
            return

        if not (
            0
            <= e_index
            < len(exposures)
        ):
            return

        후보추가_13(
            queue,
            queued,
            visited,
            gains[g_index],
            exposures[e_index],
            gains,
            exposures,
            base_priority
            + float(offset),
            reason,
        )

    if (
        sat >= 포화_높음_13
        and sat > low
    ):
        # 너무 밝음:
        # 노출↓, Gain↓를 먼저.
        add_idx(
            gi,
            ei - 1,
            0,
            "포화 우세 → 노출 낮춤",
        )
        add_idx(
            gi,
            ei - 2,
            0.2,
            "포화 강함 → 노출 크게 낮춤",
        )
        add_idx(
            gi - 1,
            ei,
            0.4,
            "포화 우세 → Gain 낮춤",
        )

        # 노출을 낮추면 저변조도가 생길 수 있으므로
        # 낮은 노출 + 높은 Gain 조합도 미리 준비.
        add_idx(
            gi + 1,
            ei - 1,
            0.6,
            "포화 억제 + 저변조도 보완",
        )

    elif (
        low >= 저변조도_높음_13
        and low > sat
    ):
        # 너무 어두움/변조도 부족:
        # 노출↑, Gain↑를 먼저.
        add_idx(
            gi,
            ei + 1,
            0,
            "저변조도 우세 → 노출 높임",
        )
        add_idx(
            gi,
            ei + 2,
            0.2,
            "저변조도 강함 → 노출 크게 높임",
        )
        add_idx(
            gi + 1,
            ei,
            0.4,
            "저변조도 우세 → Gain 높임",
        )

        # 높은 노출에서 포화가 생길 가능성에 대비.
        add_idx(
            gi - 1,
            ei + 1,
            0.6,
            "광량 확보 + 포화 억제",
        )

    else:
        # 포화와 저변조도가 비슷하거나 둘 다 중간:
        # 4방향을 모두 열어 둠.
        add_idx(
            gi,
            ei - 1,
            0,
            "혼합 원인 → 노출 낮춤 확인",
        )
        add_idx(
            gi,
            ei + 1,
            0,
            "혼합 원인 → 노출 높임 확인",
        )
        add_idx(
            gi - 1,
            ei,
            0.3,
            "혼합 원인 → Gain 낮춤 확인",
        )
        add_idx(
            gi + 1,
            ei,
            0.3,
            "혼합 원인 → Gain 높임 확인",
        )

    # 현재 방향에서 거의 안 좋아졌다면
    # 반대 방향 + 다른 축을 즉시 후보에 올림.
    if improvement < 정체_개선량_13:
        add_idx(
            gi,
            ei - 1,
            -0.5,
            "정체 → 반대 노출 방향 확인",
        )
        add_idx(
            gi,
            ei + 1,
            -0.5,
            "정체 → 반대 노출 방향 확인",
        )
        add_idx(
            gi - 1,
            ei,
            -0.3,
            "정체 → 반대 Gain 방향 확인",
        )
        add_idx(
            gi + 1,
            ei,
            -0.3,
            "정체 → 반대 Gain 방향 확인",
        )

        # 대각선 조합도 바로 열어 둠.
        add_idx(
            gi + 1,
            ei - 1,
            0,
            "정체 → Gain↑ + Exposure↓",
        )
        add_idx(
            gi - 1,
            ei + 1,
            0,
            "정체 → Gain↓ + Exposure↑",
        )


def 탈출후보_추가_13(
    queue,
    queued,
    visited,
    gains,
    exposures,
    best_condition,
):
    """
    연속 정체가 발생했을 때 현재 방향을 버리고
    2차원 공간에서 멀리 떨어진 조건을 빠르게 찍는다.
    """
    g = int(
        best_condition["gain"]
    )
    e = int(
        best_condition["exposure"]
    )

    gi = 가장가까운_인덱스_13(
        gains,
        g,
    )

    ei = 가장가까운_인덱스_13(
        exposures,
        e,
    )

    candidates = [
        (
            max(
                0,
                gi - 2,
            ),
            max(
                0,
                ei - 2,
            ),
            "탈출 → Gain↓ Exposure↓",
        ),
        (
            min(
                len(gains) - 1,
                gi + 2,
            ),
            max(
                0,
                ei - 2,
            ),
            "탈출 → Gain↑ Exposure↓",
        ),
        (
            max(
                0,
                gi - 2,
            ),
            min(
                len(exposures) - 1,
                ei + 2,
            ),
            "탈출 → Gain↓ Exposure↑",
        ),
        (
            min(
                len(gains) - 1,
                gi + 2,
            ),
            min(
                len(exposures) - 1,
                ei + 2,
            ),
            "탈출 → Gain↑ Exposure↑",
        ),
    ]

    for g_idx, e_idx, reason in candidates:
        후보추가_13(
            queue,
            queued,
            visited,
            gains[g_idx],
            exposures[e_idx],
            gains,
            exposures,
            -1.0,
            reason,
        )


def 최고개별조건_13(
    qualities,
    conditions,
):
    index = max(
        range(len(qualities)),
        key=lambda i: (
            qualities[i][
                "유효_위상_비율"
            ],
            -qualities[i][
                "포화_비율"
            ],
            -qualities[i][
                "저변조도_비율"
            ],
        ),
    )

    return (
        conditions[index],
        qualities[index],
    )


def Depth기반_자동_물체영역_검출(cap, 저장_이름, 결과_폴더_직접=None):
    """
    현재 구조광 통합 Pipeline 안에서 Depth를 이용해 물체 주변 사각형을 찾는다.

    핵심 기준:
      1) Depth E81/G16 고정
      2) 빈 플랫폼 Depth와 현재 Depth 비교
      3) 10 mm 이상을 확실한 seed로 사용
      4) seed와 연결된 5 mm 이상 영역까지만 확장
      5) 프로젝터 X/Y 최대범위 밖은 검출 시작부터 제외
      6) 최종 구조광 전처리에는 Depth 픽셀 마스크가 아니라
         빨간 사각형 내부 전체를 사용
    """

    안전한_저장_이름 = re.sub(
        r'[\\/:*?"<>|]+',
        "_",
        저장_이름,
    )

    if 결과_폴더_직접 is None:
        결과_폴더 = (
            결과_루트
            / 안전한_저장_이름
        )
    else:
        결과_폴더 = Path(
            결과_폴더_직접
        )

    결과_폴더.mkdir(
        parents=True,
        exist_ok=True,
    )

    현재_Depth_경로 = 결과_폴더 / "현재_물체_depth.npy"
    현재_Depth_PNG_경로 = 결과_폴더 / "현재_물체_depth.png"
    높이차_시각화_경로 = 결과_폴더 / "플랫폼과_높이차.png"

    확실한_seed_경로 = (
        결과_폴더 / "01_확실한_물체_10mm_seed.png"
    )
    확장후보_경로 = (
        결과_폴더 / "02_확장후보_5mm.png"
    )

    # Depth가 실제로 물체라고 판단한 픽셀은 진단용
    Depth_물체픽셀_경로 = (
        결과_폴더 / "03_Depth_물체픽셀_확인용.png"
    )

    # 실제 구조광 전처리에 직접 사용할 사각형 전체 마스크
    구조광_분석마스크_경로 = (
        결과_폴더 / "04_전처리_직접사용_사각마스크.png"
    )

    물체_마스크_확인_경로 = (
        결과_폴더 / "물체_마스크_확인용.png"
    )

    자동영역_확인_경로 = (
        결과_폴더 / "자동_물체영역_확인.png"
    )

    자동영역_회전후_확인_경로 = (
        결과_폴더
        / "자동_물체영역_180도회전후_전처리영역.png"
    )

    Depth_픽셀_파란색_확인_경로 = (
        결과_폴더
        / "Depth_물체픽셀_파란색_확인용.png"
    )

    빔영역_마스크_경로 = (
        결과_폴더
        / "프로젝터_XY범위_마스크.png"
    )

    XY범위_적용확인_경로 = (
        결과_폴더
        / "프로젝터_XY범위_적용확인.png"
    )

    자동영역_텍스트_경로 = (
        결과_폴더
        / "자동_물체영역.txt"
    )

    if not 기준_Depth_경로.exists():
        raise FileNotFoundError(
            "플랫폼 바닥 기준 Depth가 없습니다.\n"
            f"필요 파일: {기준_Depth_경로}"
        )

    if 빔_XY제한_사용:
        저장된_빔정보 = 저장된_프로젝터_XY범위_읽기()
    else:
        저장된_빔정보 = None

    print("")
    print("=" * 72)
    print("Depth 기반 물체 영역 자동 검출")
    print("=" * 72)
    print(
        f"기준 Depth: {기준_Depth_경로}"
    )
    print(
        f"Depth 결과 저장: {결과_폴더}"
    )
    print(
        f"Depth 고정: E{Depth_노출}/G{Depth_게인}"
    )
    print(
        f"물체 검출: "
        f"{확실한_물체_높이차_mm:.1f} mm seed + "
        f"{확장_물체_높이차_mm:.1f} mm 연결 확장"
    )

    if 저장된_빔정보 is not None:
        print(
            "프로젝터 X/Y 최대범위: "
            f"x={저장된_빔정보['안전_왼쪽']} ~ "
            f"{저장된_빔정보['안전_오른쪽']} | "
            f"y={저장된_빔정보['안전_위']} ~ "
            f"{저장된_빔정보['안전_아래']}"
        )

    background_depth = np.load(
        기준_Depth_경로
    ).astype(np.float32)

    pipeline = cap.pipeline
    device = cap.device

    projector_monitor = None
    projector_window = None
    구조광_미리보기 = None

    try:
        # --------------------------------------------------------
        # ROI 확인용 RGB255
        # --------------------------------------------------------
        projector_monitor = (
            프로젝터_화면_자동선택()
        )

        projector_window = (
            프로젝터_균일광_켜기(
                projector_monitor,
                선택용_프로젝터_RGB,
            )
        )

        # --------------------------------------------------------
        # 빈 플랫폼 촬영 때와 동일한 Depth E81/G16 강제
        # --------------------------------------------------------
        실제_Depth_노출, 실제_Depth_게인 = (
            Depth_수동설정_적용(
                device
            )
        )

        align_filter = AlignFilter(
            align_to_stream=OBStreamType.COLOR_STREAM
        )

        print("")
        print("Depth → Color 정렬(D2C) 적용 중...")
        print(
            f"Depth 설정 안정화를 위해 "
            f"{워밍업_프레임수}프레임을 버립니다."
        )

        for _ in range(
            워밍업_프레임수
        ):
            pipeline.wait_for_frames(
                프레임_대기시간_ms
            )

        depth_frames = []
        last_color = None

        while (
            len(depth_frames)
            < Depth_프레임수
        ):
            frames = pipeline.wait_for_frames(
                프레임_대기시간_ms
            )

            if not frames:
                continue

            aligned = align_filter.process(
                frames
            )

            if not aligned:
                continue

            color_frame = (
                aligned.get_color_frame()
            )
            depth_frame = (
                aligned.get_depth_frame()
            )

            if (
                color_frame is None
                or depth_frame is None
            ):
                continue

            current_depth_frame = (
                Depth프레임_mm로_변환(
                    depth_frame
                )
            )

            depth_frames.append(
                current_depth_frame
            )

            last_color = (
                Orbbec컬러프레임_BGR로_변환(
                    color_frame
                )
            )

            print(
                f"현재 Depth 수집 "
                f"{len(depth_frames)}/"
                f"{Depth_프레임수}"
            )

        current_depth = (
            여러Depth_중앙값(
                depth_frames
            )
        )

        if (
            current_depth.shape
            != background_depth.shape
        ):
            raise RuntimeError(
                "현재 Depth와 기준 Depth 크기가 다릅니다.\n"
                f"기준={background_depth.shape}, "
                f"현재={current_depth.shape}"
            )

        image_h, image_w = (
            current_depth.shape
        )

        # --------------------------------------------------------
        # 빈 플랫폼보다 카메라 쪽으로 가까워진 양
        # --------------------------------------------------------
        height_diff = (
            background_depth
            - current_depth
        )

        valid = (
            (background_depth > 0)
            & (current_depth > 0)
        )

        # --------------------------------------------------------
        # 물체 후보를 검사할 범위
        #
        # JSON은 180도 회전 후 구조광 영상 좌표 기준이므로
        # 원본 D2C Depth 좌표로 되돌려서 검출 시작부터 제한.
        # 암실 벽/프로젝터 범위 밖은 seed 후보 자체에 못 들어온다.
        # --------------------------------------------------------
        x_margin = int(
            image_w
            * 가로_가장자리_제외비율
        )
        y_margin = int(
            image_h
            * 세로_가장자리_제외비율
        )

        search_mask = np.ones(
            (image_h, image_w),
            dtype=bool,
        )

        if (
            x_margin > 0
            or y_margin > 0
        ):
            edge_mask = np.zeros(
                (image_h, image_w),
                dtype=bool,
            )

            edge_mask[
                y_margin:image_h - y_margin,
                x_margin:image_w - x_margin,
            ] = True

            search_mask &= edge_mask

        beam_rect_raw = None

        if 저장된_빔정보 is not None:
            beam_rect_raw = (
                회전후_XY범위_원본좌표로_변환(
                    저장된_빔정보,
                    image_w,
                    image_h,
                )
            )

            brx1, bry1, brx2, bry2 = (
                beam_rect_raw
            )

            beam_search_mask = np.zeros(
                (image_h, image_w),
                dtype=bool,
            )

            beam_search_mask[
                bry1:bry2,
                brx1:brx2,
            ] = True

            search_mask &= beam_search_mask

            print(
                "Depth 검출 원본좌표 제한: "
                f"x={brx1}~{brx2}, "
                f"y={bry1}~{bry2}"
            )

        # --------------------------------------------------------
        # 10 mm seed + 연결된 5 mm 영역 확장
        # --------------------------------------------------------
        mask_result = (
            최종_물체마스크_계산(
                height_diff,
                valid,
                search_mask,
            )
        )

        largest_mask = (
            mask_result["final_mask"]
        )
        seed_area = int(
            mask_result["seed_area"]
        )

        largest_area = int(
            np.count_nonzero(
                largest_mask
            )
        )

        if (
            largest_area
            < 최소_물체_면적_px
        ):
            raise RuntimeError(
                "최종 물체 마스크가 너무 작습니다."
            )

        # --------------------------------------------------------
        # 검출된 물체 픽셀을 감싸는 사각형
        # --------------------------------------------------------
        ys, xs = np.where(
            largest_mask
        )

        x = int(xs.min())
        y = int(ys.min())
        x_end = int(xs.max()) + 1
        y_end = int(ys.max()) + 1

        w = x_end - x
        h = y_end - y

        rect = 사각영역_여유추가(
            x,
            y,
            w,
            h,
            image_w,
            image_h,
        )

        rect_rotated = (
            사각영역_180도회전(
                rect,
                image_w,
                image_h,
            )
        )

        # --------------------------------------------------------
        # 물체 주변 사각형을 사용자가 지정한 프로젝터 X/Y 최대범위로 자름
        # --------------------------------------------------------
        beam_info = None
        final_rect_rotated = (
            rect_rotated
        )

        if 빔_XY제한_사용:
            beam_info = (
                저장된_빔정보
            )

            final_rect_rotated = (
                물체영역_빔XY범위로_제한(
                    rect_rotated,
                    beam_info,
                    image_w,
                    image_h,
                )
            )

        frx1, fry1, frx2, fry2 = (
            final_rect_rotated
        )

        # --------------------------------------------------------
        # 진단용 Depth 실제 물체 픽셀
        # --------------------------------------------------------
        rotated_object_mask = (
            cv2.rotate(
                largest_mask.astype(
                    np.uint8
                )
                * 255,
                cv2.ROTATE_180,
            )
            > 0
        )

        final_object_mask = np.zeros(
            (image_h, image_w),
            dtype=bool,
        )

        final_object_mask[
            fry1:fry2,
            frx1:frx2,
        ] = rotated_object_mask[
            fry1:fry2,
            frx1:frx2,
        ]

        # --------------------------------------------------------
        # 실제 구조광 전처리에 직접 사용할 마스크
        #
        # 빨간 사각형 내부 전체 = True
        # Depth가 물체 일부를 놓쳐도 사각형 안은 모두 사용.
        # 단, 이 사각형은 프로젝터 X/Y 최대범위를 절대 넘지 않음.
        # --------------------------------------------------------
        final_roi_mask = np.zeros(
            (image_h, image_w),
            dtype=bool,
        )

        final_roi_mask[
            fry1:fry2,
            frx1:frx2,
        ] = True

        if (
            np.count_nonzero(
                final_roi_mask
            )
            < 100
        ):
            raise RuntimeError(
                "최종 전처리용 사각형 영역이 너무 작습니다."
            )

        object_height_values = (
            height_diff[
                largest_mask
                & valid
            ]
        )

        # --------------------------------------------------------
        # 저장
        # --------------------------------------------------------
        np.save(
            현재_Depth_경로,
            current_depth,
        )

        cv2.imwrite(
            str(
                현재_Depth_PNG_경로
            ),
            np.clip(
                np.rint(
                    current_depth
                ),
                0,
                65535,
            ).astype(
                np.uint16
            ),
        )

        cv2.imwrite(
            str(
                높이차_시각화_경로
            ),
            높이차_시각화(
                height_diff,
                valid,
            ),
        )

        cv2.imwrite(
            str(
                확실한_seed_경로
            ),
            mask_result[
                "largest_seed"
            ].astype(
                np.uint8
            )
            * 255,
        )

        cv2.imwrite(
            str(
                확장후보_경로
            ),
            mask_result[
                "grow_clean"
            ].astype(
                np.uint8
            )
            * 255,
        )

        cv2.imwrite(
            str(
                Depth_물체픽셀_경로
            ),
            final_object_mask.astype(
                np.uint8
            )
            * 255,
        )

        cv2.imwrite(
            str(
                구조광_분석마스크_경로
            ),
            final_roi_mask.astype(
                np.uint8
            )
            * 255,
        )

        # 원본 방향의 Depth 픽셀 / bbox 확인
        mask_vis = np.zeros(
            (
                image_h,
                image_w,
                3,
            ),
            dtype=np.uint8,
        )

        mask_vis[
            largest_mask
        ] = (
            255,
            255,
            255,
        )

        x1, y1, x2, y2 = rect

        cv2.rectangle(
            mask_vis,
            (x1, y1),
            (x2, y2),
            (0, 0, 255),
            2,
        )

        cv2.imwrite(
            str(
                물체_마스크_확인_경로
            ),
            mask_vis,
        )

        if last_color is not None:
            # ----------------------------------------------------
            # 원본 D2C 방향 확인
            # 초록 = 프로젝터 X/Y 범위를 원본좌표로 되돌린 범위
            # 빨강 = 검출 물체 주변 bbox
            # ----------------------------------------------------
            overlay = (
                last_color.copy()
            )

            if (
                beam_rect_raw
                is not None
            ):
                (
                    brx1,
                    bry1,
                    brx2,
                    bry2,
                ) = beam_rect_raw

                cv2.rectangle(
                    overlay,
                    (
                        brx1,
                        bry1,
                    ),
                    (
                        brx2,
                        bry2,
                    ),
                    (
                        0,
                        255,
                        0,
                    ),
                    2,
                )

            cv2.rectangle(
                overlay,
                (x1, y1),
                (x2, y2),
                (0, 0, 255),
                3,
            )

            cv2.putText(
                overlay,
                "Depth object bbox",
                (
                    x1,
                    max(
                        30,
                        y1 - 10,
                    ),
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (
                    0,
                    0,
                    255,
                ),
                2,
                cv2.LINE_AA,
            )

            cv2.imwrite(
                str(
                    자동영역_확인_경로
                ),
                overlay,
            )

            # ----------------------------------------------------
            # 구조광 방향(180도) 확인
            # 노랑 = Depth 물체 bbox + 여유
            # 초록 = 사용자가 정한 프로젝터 최대 X/Y 범위
            # 빨강 = 실제 구조광 전처리에 사용할 최종 사각형
            # ----------------------------------------------------
            구조광_미리보기 = (
                cv2.rotate(
                    last_color,
                    cv2.ROTATE_180,
                )
            )

            rotated = (
                구조광_미리보기.copy()
            )

            (
                orx1,
                ory1,
                orx2,
                ory2,
            ) = rect_rotated

            cv2.rectangle(
                rotated,
                (
                    orx1,
                    ory1,
                ),
                (
                    orx2,
                    ory2,
                ),
                (
                    0,
                    255,
                    255,
                ),
                2,
            )

            if beam_info is not None:
                safe_left = int(
                    beam_info[
                        "안전_왼쪽"
                    ]
                )
                safe_right = int(
                    beam_info[
                        "안전_오른쪽"
                    ]
                )
                safe_top = int(
                    beam_info[
                        "안전_위"
                    ]
                )
                safe_bottom = int(
                    beam_info[
                        "안전_아래"
                    ]
                )

                cv2.rectangle(
                    rotated,
                    (
                        safe_left,
                        safe_top,
                    ),
                    (
                        safe_right,
                        safe_bottom,
                    ),
                    (
                        0,
                        255,
                        0,
                    ),
                    2,
                )

                beam_vis = np.zeros(
                    (
                        image_h,
                        image_w,
                        3,
                    ),
                    dtype=np.uint8,
                )

                beam_vis[
                    safe_top:safe_bottom,
                    safe_left:safe_right,
                ] = (
                    255,
                    255,
                    255,
                )

                cv2.imwrite(
                    str(
                        빔영역_마스크_경로
                    ),
                    beam_vis,
                )

            rx1, ry1, rx2, ry2 = (
                final_rect_rotated
            )

            cv2.rectangle(
                rotated,
                (
                    rx1,
                    ry1,
                ),
                (
                    rx2,
                    ry2,
                ),
                (
                    0,
                    0,
                    255,
                ),
                4,
            )

            cv2.putText(
                rotated,
                "RED RECT = PREPROCESS ROI",
                (
                    rx1,
                    max(
                        30,
                        ry1 - 10,
                    ),
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.75,
                (
                    0,
                    0,
                    255,
                ),
                2,
                cv2.LINE_AA,
            )

            cv2.imwrite(
                str(
                    자동영역_회전후_확인_경로
                ),
                rotated,
            )

            cv2.imwrite(
                str(
                    XY범위_적용확인_경로
                ),
                rotated,
            )

            # 파란색은 Depth 픽셀 진단용으로만 별도 저장
            depth_pixel_vis = (
                rotated.copy()
            )
            blue_overlay = (
                depth_pixel_vis.copy()
            )

            blue_overlay[
                final_object_mask
            ] = (
                255,
                0,
                0,
            )

            depth_pixel_vis = (
                cv2.addWeighted(
                    depth_pixel_vis,
                    0.70,
                    blue_overlay,
                    0.30,
                    0,
                )
            )

            cv2.imwrite(
                str(
                    Depth_픽셀_파란색_확인_경로
                ),
                depth_pixel_vis,
            )

        object_ratio = (
            largest_area
            / float(
                image_h
                * image_w
            )
            * 100.0
        )

        seed_ratio = (
            seed_area
            / float(
                image_h
                * image_w
            )
            * 100.0
        )

        rect_pixel_count = max(
            1,
            (
                final_rect_rotated[2]
                - final_rect_rotated[0]
            )
            * (
                final_rect_rotated[3]
                - final_rect_rotated[1]
            ),
        )

        depth_object_count = int(
            np.count_nonzero(
                final_object_mask
            )
        )

        roi_count = int(
            np.count_nonzero(
                final_roi_mask
            )
        )

        text_lines = [
            "Depth 기반 자동 물체 영역",
            "=" * 60,
            (
                f"Depth 요청 설정: "
                f"E{Depth_노출}/G{Depth_게인}"
            ),
            (
                f"Depth 실제 설정: "
                f"E{실제_Depth_노출}/"
                f"G{실제_Depth_게인}"
            ),
            "",
            (
                f"확실한 seed 기준: "
                f"{확실한_물체_높이차_mm:.1f} mm"
            ),
            (
                f"연결 확장 기준: "
                f"{확장_물체_높이차_mm:.1f} mm"
            ),
            (
                f"seed 픽셀 수: "
                f"{seed_area} "
                f"({seed_ratio:.2f}%)"
            ),
            (
                f"연결 확장 후 물체 픽셀 수: "
                f"{largest_area} "
                f"({object_ratio:.2f}%)"
            ),
            (
                "물체 높이차 중앙값: "
                f"{np.median(object_height_values):.1f} mm"
            ),
            (
                "물체 높이차 P95: "
                f"{np.percentile(object_height_values, 95):.1f} mm"
            ),
            "",
            (
                f"원본 D2C 좌표 rect: "
                f"{rect}"
            ),
            (
                f"180도 회전 후 Depth rect: "
                f"{rect_rotated}"
            ),
            (
                f"최종 구조광용 rect: "
                f"{final_rect_rotated}"
            ),
            (
                "Depth 진단용 물체 픽셀: "
                f"{depth_object_count} / "
                f"사각형 {rect_pixel_count} "
                f"({depth_object_count / rect_pixel_count * 100.0:.2f}%)"
            ),
            (
                "실제 구조광 전처리 사용 픽셀: "
                f"{roi_count} / "
                f"사각형 {rect_pixel_count} "
                f"({roi_count / rect_pixel_count * 100.0:.2f}%)"
            ),
            (
                "전처리 직접 사용영역: "
                "빨간 사각형 내부 전체"
            ),
        ]

        if beam_info is not None:
            text_lines.extend(
                [
                    "",
                    "프로젝터 수동 지정 X/Y 최대범위",
                    (
                        f"사용 X 범위: "
                        f"{beam_info['안전_왼쪽']} ~ "
                        f"{beam_info['안전_오른쪽']}"
                    ),
                    (
                        f"사용 Y 범위: "
                        f"{beam_info['안전_위']} ~ "
                        f"{beam_info['안전_아래']}"
                    ),
                    (
                        f"회전 전 Depth 적용 범위: "
                        f"{beam_rect_raw}"
                    ),
                    (
                        "범위 출처: "
                        f"{beam_info.get('설정방식', '수동 2점 X/Y 선택')}"
                    ),
                ]
            )

        자동영역_텍스트_경로.write_text(
            "\n".join(
                text_lines
            ),
            encoding="utf-8",
        )

        print("")
        print("=" * 72)
        print("Depth 자동 물체 검출 완료")
        print("=" * 72)
        print(
            f"Depth 요청 E{Depth_노출}/G{Depth_게인} | "
            f"실제 E{실제_Depth_노출}/G{실제_Depth_게인}"
        )
        print(
            f"확실한 seed "
            f"{확실한_물체_높이차_mm:.1f} mm | "
            f"연결 확장 "
            f"{확장_물체_높이차_mm:.1f} mm"
        )
        print(
            f"최종 구조광 코드용 좌표: "
            f"{final_rect_rotated}"
        )

        if beam_info is not None:
            print(
                "수동 지정 프로젝터 최대범위: "
                f"x={beam_info['안전_왼쪽']} ~ "
                f"{beam_info['안전_오른쪽']} | "
                f"y={beam_info['안전_위']} ~ "
                f"{beam_info['안전_아래']}"
            )

        print(
            "실제 구조광 전처리 사용영역: "
            "빨간 사각형 내부 전체"
        )
        print(
            f"전처리 사각 마스크: "
            f"{구조광_분석마스크_경로}"
        )
        print(
            f"Color 확인: "
            f"{자동영역_회전후_확인_경로}"
        )
        print(
            "파란색 Depth 물체 픽셀은 진단용이며 "
            "구조광 품질 계산에는 직접 사용하지 않습니다."
        )
        print("=" * 72)

        if 구조광_미리보기 is None:
            raise RuntimeError(
                "Depth 자동검출용 Color 미리보기를 만들지 못했습니다."
            )

        # 중요:
        # 세 번째 반환값은 실제 Depth 픽셀 마스크가 아니라
        # '빨간 사각형 내부 전체' 마스크다.
        # 이후 구조광 품질/융합/위상 계산이 이 사각형 전체를 사용한다.
        return (
            final_rect_rotated,
            구조광_미리보기,
            final_roi_mask,
        )

    finally:
        try:
            if (
                projector_window is not None
                and projector_monitor is not None
            ):
                프로젝터_검정화면(
                    projector_window,
                    projector_monitor,
                )
        except Exception:
            pass

        cv2.destroyAllWindows()


# ============================================================
# 사용자 구조광 형상복원 통합부
# - 팀원 전처리의 최종 융합 Wrapped Phase + Valid Mask 사용
# - 별도 Reference 4-step 폴더와 차분
# - Relative Surface / PLY 생성
# - 팀원 원본 알고리즘은 변경하지 않고 후단에만 연결
# ============================================================


def 통합_회색이미지_읽기(path):
    image = cv2.imread(
        str(path),
        cv2.IMREAD_GRAYSCALE,
    )
    if image is None:
        raise FileNotFoundError(
            f"Reference 이미지를 읽을 수 없습니다: {path}"
        )
    return image.astype(np.float32)


def 통합_Reference_4위상_읽기(reference_dir, args):
    reference_dir = Path(reference_dir).expanduser().resolve()

    if not reference_dir.exists():
        raise FileNotFoundError(
            f"Reference 폴더가 없습니다: {reference_dir}"
        )

    gray = {}
    color = {}

    for phase_name in ["000", "090", "180", "270"]:
        gray_path = reference_dir / f"phase_{phase_name}.png"
        gray_frame = 통합_회색이미지_읽기(gray_path)
        if args.reference_rotate_180:
            gray_frame = cv2.rotate(
                gray_frame,
                cv2.ROTATE_180,
            )
        gray[phase_name] = gray_frame

        color_path = reference_dir / f"phase_{phase_name}_color.png"
        if color_path.exists():
            color_frame = cv2.imread(
                str(color_path),
                cv2.IMREAD_COLOR,
            )
            if color_frame is not None:
                if args.reference_rotate_180:
                    color_frame = cv2.rotate(
                        color_frame,
                        cv2.ROTATE_180,
                    )
                color[phase_name] = color_frame.astype(np.float32)

    shapes = {frame.shape for frame in gray.values()}
    if len(shapes) != 1:
        raise RuntimeError(
            f"Reference 4위상 이미지 크기가 서로 다릅니다: {shapes}"
        )

    i0 = gray["000"]
    i90 = gray["090"]
    i180 = gray["180"]
    i270 = gray["270"]

    modulation = 0.5 * np.sqrt(
        (i0 - i180) ** 2
        + (i270 - i90) ** 2
    )

    gray_max = np.maximum.reduce(
        [i0, i90, i180, i270]
    )

    if len(color) == 4:
        channel_max_per_phase = []
        for phase_name in ["000", "090", "180", "270"]:
            channel_max_per_phase.append(
                np.max(color[phase_name], axis=2)
            )
        intensity_max = np.maximum.reduce(channel_max_per_phase)
    else:
        intensity_max = gray_max

    saturation = (
        intensity_max
        >= float(args.saturation_threshold)
    )
    dark = (
        gray_max
        <= float(args.dark_threshold)
    )
    low_modulation = (
        modulation
        < float(args.modulation_threshold)
    )

    valid = (
        (~saturation)
        & (~dark)
        & (~low_modulation)
    )

    wrapped = np.arctan2(
        i270 - i90,
        i0 - i180,
    ).astype(np.float32)

    return {
        "폴더": reference_dir,
        "래핑_위상맵": wrapped,
        "변조도_지도": modulation.astype(np.float32),
        "유효_마스크": valid.astype(bool),
        "포화_마스크": saturation.astype(bool),
        "암부_마스크": dark.astype(bool),
        "저변조도_마스크": low_modulation.astype(bool),
    }


def 통합_wrap_to_pi(data):
    return (
        (data + np.pi)
        % (2.0 * np.pi)
        - np.pi
    ).astype(np.float32)


def 통합_컬러맵_저장(path, data, mask=None):
    image = np.asarray(data, dtype=np.float32)

    if mask is None:
        valid = np.isfinite(image)
    else:
        valid = np.asarray(mask, dtype=bool) & np.isfinite(image)

    gray = np.zeros(image.shape, dtype=np.uint8)

    if np.any(valid):
        values = image[valid]
        low = float(np.percentile(values, 1))
        high = float(np.percentile(values, 99))
        if high <= low:
            high = low + 1e-6

        clipped = np.clip(image, low, high)
        normalized = (
            (clipped - low)
            / (high - low)
        )
        gray[valid] = np.clip(
            normalized[valid] * 255.0,
            0,
            255,
        ).astype(np.uint8)

    color = cv2.applyColorMap(
        gray,
        cv2.COLORMAP_JET,
    )
    color[~valid] = (0, 0, 0)
    cv2.imwrite(str(path), color)


def 통합_마스크PNG_저장(path, mask):
    image = np.zeros(
        np.asarray(mask).shape,
        dtype=np.uint8,
    )
    image[np.asarray(mask, dtype=bool)] = 255
    cv2.imwrite(str(path), image)


def 통합_masked_gaussian(data, mask, ksize):
    if int(ksize) <= 0:
        result = np.asarray(data, dtype=np.float32).copy()
        result[~mask] = np.nan
        return result

    ksize = int(ksize)
    if ksize % 2 == 0:
        ksize += 1

    source = np.where(
        mask,
        data,
        0.0,
    ).astype(np.float32)
    weight = mask.astype(np.float32)

    numerator = cv2.GaussianBlur(
        source,
        (ksize, ksize),
        0,
    )
    denominator = cv2.GaussianBlur(
        weight,
        (ksize, ksize),
        0,
    )

    result = numerator / (denominator + 1e-9)
    result[denominator < 0.25] = np.nan
    result[~mask] = np.nan
    return result.astype(np.float32)


def 통합_best_fit_plane_제거(data, mask):
    h, w = data.shape
    yy, xx = np.mgrid[0:h, 0:w]

    valid = mask & np.isfinite(data)
    values = data[valid]

    if values.size < 100:
        result = data.copy()
        result[~valid] = np.nan
        return result.astype(np.float32), None

    x = xx[valid].astype(np.float32)
    y = yy[valid].astype(np.float32)
    z = values.astype(np.float32)

    low = np.percentile(z, 10)
    high = np.percentile(z, 90)
    fit = (z >= low) & (z <= high)

    if np.count_nonzero(fit) < 100:
        result = data.copy()
        result[~valid] = np.nan
        return result.astype(np.float32), None

    A = np.column_stack(
        [x[fit], y[fit], np.ones(np.count_nonzero(fit), dtype=np.float32)]
    )
    coeff, _, _, _ = np.linalg.lstsq(
        A,
        z[fit],
        rcond=None,
    )

    a, b, c = [float(v) for v in coeff]
    plane = a * xx + b * yy + c

    result = data - plane
    result[~valid] = np.nan

    return result.astype(np.float32), {
        "a": a,
        "b": b,
        "c": c,
    }


def 통합_PLY_ASCII_저장(path, points, colors):
    with Path(path).open("w", encoding="ascii") as file:
        file.write("ply\n")
        file.write("format ascii 1.0\n")
        file.write(f"element vertex {len(points)}\n")
        file.write("property float x\n")
        file.write("property float y\n")
        file.write("property float z\n")
        file.write("property uchar red\n")
        file.write("property uchar green\n")
        file.write("property uchar blue\n")
        file.write("end_header\n")

        for point, color in zip(points, colors):
            x, y, z = point
            r, g, b = color
            file.write(
                f"{x:.6f} {y:.6f} {z:.6f} "
                f"{int(r)} {int(g)} {int(b)}\n"
            )


def 통합_Relative_PLY_생성(
    phase_diff_masked,
    valid_mask,
    output_dir,
    args,
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    depth = np.asarray(
        phase_diff_masked,
        dtype=np.float32,
    ).copy()
    valid = (
        np.asarray(valid_mask, dtype=bool)
        & np.isfinite(depth)
    )

    values = depth[valid]
    if values.size == 0:
        raise RuntimeError(
            "Relative PLY 생성용 유효 Phase Difference가 없습니다."
        )

    p_low = float(args.phase_p_low)
    p_high = float(args.phase_p_high)

    if p_high <= p_low:
        raise ValueError(
            "--phase_p_high는 --phase_p_low보다 커야 합니다."
        )

    low = float(np.percentile(values, p_low))
    high = float(np.percentile(values, p_high))

    clipped = np.clip(
        depth,
        low,
        high,
    )
    clipped[~valid] = np.nan

    if args.no_relative_plane_remove:
        center = float(np.nanmedian(clipped[valid]))
        processed = clipped - center
        processed[~valid] = np.nan
        plane_info = None
    else:
        processed, plane_info = 통합_best_fit_plane_제거(
            clipped,
            valid,
        )

    values2 = processed[valid & np.isfinite(processed)]
    if values2.size == 0:
        raise RuntimeError(
            "Plane 처리 후 유효 데이터가 없습니다."
        )

    fill_value = float(np.median(values2))
    filled = np.where(
        valid,
        processed,
        fill_value,
    ).astype(np.float32)

    dmin = float(np.min(filled))
    dmax = float(np.max(filled))

    if abs(dmax - dmin) < 1e-9:
        raise RuntimeError(
            "Relative surface 값 범위가 너무 작습니다."
        )

    median_ksize = max(1, int(args.relative_median_ksize))
    if median_ksize % 2 == 0:
        median_ksize += 1

    normalized_8u = np.clip(
        (filled - dmin)
        / (dmax - dmin + 1e-9)
        * 255.0,
        0,
        255,
    ).astype(np.uint8)

    median_8u = cv2.medianBlur(
        normalized_8u,
        median_ksize,
    )
    median_f = (
        median_8u.astype(np.float32)
        / 255.0
        * (dmax - dmin)
        + dmin
    )
    median_f[~valid] = np.nan

    if args.no_relative_gaussian:
        smooth = median_f.copy()
    else:
        smooth = 통합_masked_gaussian(
            median_f,
            valid,
            int(args.relative_gaussian_ksize),
        )

    final_mask = valid & np.isfinite(smooth)
    final_values = smooth[final_mask]

    if final_values.size == 0:
        raise RuntimeError(
            "Relative Point Cloud 최종 유효 데이터가 없습니다."
        )

    zmin = float(np.min(final_values))
    zmax = float(np.max(final_values))

    np.save(
        output_dir / "relative_surface_smooth.npy",
        smooth,
    )
    np.save(
        output_dir / "relative_final_mask.npy",
        final_mask,
    )
    통합_마스크PNG_저장(
        output_dir / "relative_final_mask.png",
        final_mask,
    )
    통합_컬러맵_저장(
        output_dir / "relative_surface_color.png",
        smooth,
        final_mask,
    )

    h, w = smooth.shape
    skip = max(1, int(args.point_skip))

    points = []
    colors = []

    for y in range(0, h, skip):
        for x in range(0, w, skip):
            if not final_mask[y, x]:
                continue

            z = float(smooth[y, x])
            if not np.isfinite(z):
                continue

            X = float(x - w / 2.0)
            Y = float((h - y) - h / 2.0)
            Z = float(
                float(args.relative_z_sign)
                * z
                * float(args.relative_z_scale)
            )

            z_norm = int(
                np.clip(
                    255.0
                    * (z - zmin)
                    / (zmax - zmin + 1e-9),
                    0,
                    255,
                )
            )

            color_bgr = cv2.applyColorMap(
                np.array([[z_norm]], dtype=np.uint8),
                cv2.COLORMAP_JET,
            )[0, 0]
            color_rgb = color_bgr[::-1]

            points.append([X, Y, Z])
            colors.append(color_rgb)

    points = np.asarray(points, dtype=np.float32)
    colors = np.asarray(colors, dtype=np.uint8)

    ply_path = output_dir / "structured_light_relative_pointcloud.ply"
    통합_PLY_ASCII_저장(
        ply_path,
        points,
        colors,
    )

    return {
        "ply_path": str(ply_path),
        "point_count": int(len(points)),
        "clip_low": low,
        "clip_high": high,
        "final_min": zmin,
        "final_max": zmax,
        "plane": plane_info,
    }


def 사용자_구조광_형상복원_실행(
    result_dir,
    fusion,
    fusion_quality,
    args,
):
    if not args.reference_dir:
        print("")
        print("[통합 구조광] --reference_dir 미지정 → 팀원 전처리까지만 완료")
        return None

    output_dir = Path(result_dir) / "구조광_형상복원"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("")
    print("=" * 78)
    print("팀원 전처리 + 사용자 Phase Difference + Relative PLY 통합")
    print("=" * 78)
    print(f"Reference: {args.reference_dir}")
    print(f"Output: {output_dir}")

    reference = 통합_Reference_4위상_읽기(
        args.reference_dir,
        args,
    )

    object_phase = np.asarray(
        fusion_quality["래핑_위상맵"],
        dtype=np.float32,
    )
    object_area = np.asarray(
        fusion_quality["물체영역_마스크"],
        dtype=bool,
    )
    object_valid = (
        np.asarray(
            fusion["최종유효_마스크"],
            dtype=bool,
        )
        & object_area
    )

    reference_phase = reference["래핑_위상맵"]
    reference_valid = reference["유효_마스크"]

    if reference_phase.shape != object_phase.shape:
        raise RuntimeError(
            "Reference와 Object 위상맵 크기가 다릅니다: "
            f"Reference={reference_phase.shape}, Object={object_phase.shape}"
        )

    common_valid = (
        object_area
        & object_valid
        & reference_valid
        & np.isfinite(reference_phase)
        & np.isfinite(object_phase)
    )

    phase_diff = 통합_wrap_to_pi(
        object_phase - reference_phase
    )
    phase_diff_masked = phase_diff.copy()
    phase_diff_masked[~common_valid] = np.nan

    area_count = max(1, int(np.count_nonzero(object_area)))
    object_valid_count = int(np.count_nonzero(object_valid))
    reference_valid_in_object = int(
        np.count_nonzero(reference_valid & object_area)
    )
    common_valid_count = int(np.count_nonzero(common_valid))

    object_coverage = object_valid_count / area_count * 100.0
    reference_coverage = reference_valid_in_object / area_count * 100.0
    common_coverage = common_valid_count / area_count * 100.0

    np.save(
        output_dir / "reference_wrapped_phase.npy",
        reference_phase,
    )
    np.save(
        output_dir / "object_fused_wrapped_phase.npy",
        object_phase,
    )
    np.save(
        output_dir / "object_area_mask.npy",
        object_area,
    )
    np.save(
        output_dir / "object_final_valid_mask.npy",
        object_valid,
    )
    np.save(
        output_dir / "reference_valid_mask.npy",
        reference_valid,
    )
    np.save(
        output_dir / "common_valid_mask.npy",
        common_valid,
    )
    np.save(
        output_dir / "phase_difference.npy",
        phase_diff,
    )
    np.save(
        output_dir / "phase_difference_masked.npy",
        phase_diff_masked,
    )

    통합_마스크PNG_저장(
        output_dir / "01_object_area_mask.png",
        object_area,
    )
    통합_마스크PNG_저장(
        output_dir / "02_object_final_valid_mask.png",
        object_valid,
    )
    통합_마스크PNG_저장(
        output_dir / "03_reference_valid_mask.png",
        reference_valid & object_area,
    )
    통합_마스크PNG_저장(
        output_dir / "04_common_valid_mask.png",
        common_valid,
    )

    통합_컬러맵_저장(
        output_dir / "05_reference_wrapped_phase.png",
        reference_phase,
        object_area,
    )
    통합_컬러맵_저장(
        output_dir / "06_object_fused_wrapped_phase.png",
        object_phase,
        object_valid,
    )
    통합_컬러맵_저장(
        output_dir / "07_phase_difference.png",
        phase_diff,
        object_area,
    )
    통합_컬러맵_저장(
        output_dir / "08_phase_difference_masked.png",
        phase_diff_masked,
        common_valid,
    )

    ply_info = 통합_Relative_PLY_생성(
        phase_diff_masked,
        common_valid,
        output_dir,
        args,
    )

    phase_values = phase_diff_masked[common_valid]

    summary = {
        "reference_dir": str(reference["폴더"]),
        "object_area_pixels": int(area_count),
        "object_final_valid_pixels": int(object_valid_count),
        "reference_valid_pixels_in_object_area": int(reference_valid_in_object),
        "common_valid_pixels": int(common_valid_count),
        "object_valid_coverage_percent": float(object_coverage),
        "reference_valid_coverage_percent": float(reference_coverage),
        "phase_difference_common_coverage_percent": float(common_coverage),
        "phase_difference_min_rad": (
            float(np.min(phase_values))
            if phase_values.size else None
        ),
        "phase_difference_max_rad": (
            float(np.max(phase_values))
            if phase_values.size else None
        ),
        "phase_difference_mean_rad": (
            float(np.mean(phase_values))
            if phase_values.size else None
        ),
        "phase_difference_median_rad": (
            float(np.median(phase_values))
            if phase_values.size else None
        ),
        "relative_pointcloud": ply_info,
        "important_note": (
            "PLY의 X/Y는 픽셀 좌표, Z는 Phase Difference × relative_z_scale인 "
            "상대 단위이며 실제 mm 3D가 아님."
        ),
    }

    (output_dir / "00_구조광_형상복원_요약.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    report = [
        "팀원 전처리 + 사용자 구조광 형상복원 통합 결과",
        "=" * 78,
        f"Reference: {reference['폴더']}",
        f"물체 분석영역: {area_count} px",
        f"Object 최종 Valid: {object_coverage:.2f}%",
        f"Reference Valid(물체영역 기준): {reference_coverage:.2f}%",
        f"Phase Difference 공통 Valid: {common_coverage:.2f}%",
        f"Relative PLY points: {ply_info['point_count']}",
        f"Relative PLY: {ply_info['ply_path']}",
        "",
        "※ Object Phase는 팀원 전처리의 픽셀별 다중 Gain/Exposure 융합 결과를 사용함.",
        "※ Object/Reference가 둘 다 유효한 픽셀만 Phase Difference에 사용함.",
        "※ Relative PLY는 실제 mm가 아니라 위상차 기반 상대 형상임.",
    ]

    (output_dir / "00_구조광_형상복원_요약.txt").write_text(
        "\n".join(report),
        encoding="utf-8",
    )

    print(f"Object 최종 Valid: {object_coverage:.2f}%")
    print(f"Reference Valid: {reference_coverage:.2f}%")
    print(f"Phase Difference 공통 Valid: {common_coverage:.2f}%")
    print(f"Relative PLY points: {ply_info['point_count']}")
    print(f"PLY 저장: {ply_info['ply_path']}")
    print("=" * 78)

    return summary






원본_STAGE6_SOURCE = r'''
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
clean HDR phase_difference_masked.npy를 대상으로 하는
quality-guided branch-index(k) 전파 진단.

이전 connected-component 방식의 문제:
- branch edge를 "벽"으로만 만들면 branch 선이 닫힌 곡선이 아닐 때
  다른 길로 돌아가 같은 component로 다시 연결될 수 있다.
- 그래서 실제로는 k가 달라야 하는 두 픽셀이 같은 component에 들어가 버릴 수 있다.

이번 방식:
- 모든 유효 인접 픽셀쌍에 대해
      delta_k = round(-(phi_q - phi_p) / 2π)
  를 계산한다.
- circular 차이가 작은 edge부터 우선하는 quality-guided spanning forest로
  픽셀 단위 k를 전파한다.
- 행/열 방향 np.unwrap처럼 한 방향으로 누적하지 않는다.
- smoothing/interpolation 없음.
- 기존 파일 덮어쓰기 없음.
- PLY 생성 없음.
- 결과는 "후보 unwrapped phase"로 별도 저장만 한다.

핵심 수식:
    phi_candidate = phi_raw + 2π * k

중요:
- k의 절대값에는 global offset 자유도가 있다.
  즉 전체 k에 +1을 더해도 wrapped phase는 같다.
- 우리가 보는 것은 "공간적으로 어디서 k가 달라지는가"와
  "그 k를 적용했을 때 350~360°짜리 가짜 jump가 사라지는가"이다.
"""

from pathlib import Path
import heapq
import json
import csv
from collections import Counter, deque

import cv2
import numpy as np
import matplotlib.pyplot as plt


# =============================================================================
# 0. 경로
# =============================================================================

STRUCT_DIR = Path(
    "/home/seoyeong/졸업작품/전처리와구조광_통합/플랫폼 바닥 따기/전처리_실험용/"
    "남색/촬영_20260813_125737/전처리_결과/구조광_형상복원"
)

통합_출력_루트 = STRUCT_DIR / "3_6통합 버전"

CLEAN_HDR_DIR = (
    통합_출력_루트
    / "05_기존HDR원리_원래마스크유지_2pi언랩_최종"
)

PHASE_PATH = (
    CLEAN_HDR_DIR
    / "phase_difference_masked.npy"
)

VALID_PATH = (
    CLEAN_HDR_DIR
    / "common_valid_mask.npy"
)

OBJECT_MASK_CANDIDATES = [
    STRUCT_DIR / "object_area_mask.npy",
    STRUCT_DIR / "실제_본넷_마스크.npy",
    STRUCT_DIR / "실제본넷_위상차_재계산" / "실제_본넷_마스크.npy",
]

BRIGHTNESS_ROOT = Path(
    "/home/seoyeong/졸업작품/전처리와구조광_통합/플랫폼 바닥 따기/전처리_실험용/"
    "남색_디퍼런트페이즈가 아쉬움/촬영_20260811_191720/"
    "전처리_결과/구조광_형상복원/밝기 2_30 2씩 증가"
)

P20_DIR = BRIGHTNESS_ROOT / "P020_G16_E156"

OUTPUT_DIR = (
    통합_출력_루트
    / "06_quality_guided_kmap_진단"
)

PHASE_NAMES = ["000", "090", "180", "270"]


# =============================================================================
# 1. 진단 기준
# =============================================================================

# 우리가 앞에서 검증한 branch 정의
RAW_BRANCH_MIN_DEG = 300.0
BRANCH_CIRCULAR_MAX_DEG = 30.0

# quality-guided 전파에서 너무 큰 circular edge를 우선적으로 피한다.
# 다만 모든 픽셀을 버리지 않기 위해 1차/2차로 나눠 처리한다.
PRIMARY_EDGE_MAX_CIRCULAR_DEG = 45.0
SECONDARY_EDGE_MAX_CIRCULAR_DEG = 90.0

# 이 값보다 큰 k 범위가 나오면 강한 경고
K_ABS_WARNING = 4


# =============================================================================
# 2. 기본 함수
# =============================================================================

def require(path, label):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"{label} 없음:\n{path}"
        )
    return path


def wrap_to_pi(x):
    x = np.asarray(
        x,
        dtype=np.float64,
    )
    return (
        (
            x + np.pi
        )
        % (
            2.0 * np.pi
        )
        - np.pi
    )


def find_first_mask(
    candidates,
    shape,
    label,
):
    for path in candidates:
        path = Path(path)

        if not path.exists():
            continue

        mask = np.load(
            path
        ).astype(bool)

        if mask.shape != shape:
            print(
                f"[건너뜀] {label} 크기 불일치: "
                f"{path} | {mask.shape} != {shape}"
            )
            continue

        print(
            f"{label} 사용: {path}"
        )

        return (
            mask,
            path,
        )

    raise FileNotFoundError(
        f"사용 가능한 {label}를 찾지 못했습니다."
    )


def save_mask(
    path,
    mask,
):
    image = np.zeros(
        mask.shape,
        dtype=np.uint8,
    )

    image[
        mask
    ] = 255

    cv2.imwrite(
        str(path),
        image,
    )


def find_phase_file(
    folder,
    name,
):
    folder = Path(folder)

    candidates = [
        folder / f"phase_{name}.png",
        folder / f"phase_{name}.jpg",
        folder / f"phase_{name}_color.png",
        folder / f"phase_{name}_color.jpg",
    ]

    for path in candidates:
        if path.exists():
            return path

    return None


def load_preview(
    shape,
):
    images = []

    for name in PHASE_NAMES:
        path = find_phase_file(
            P20_DIR,
            name,
        )

        if path is None:
            return None

        image = cv2.imread(
            str(path),
            cv2.IMREAD_COLOR,
        )

        if image is None:
            return None

        if image.shape[:2] != shape:
            return None

        images.append(
            image.astype(
                np.float32
            )
        )

    preview = np.mean(
        np.stack(
            images,
            axis=0,
        ),
        axis=0,
    )

    return np.clip(
        preview,
        0,
        255,
    ).astype(
        np.uint8
    )


# =============================================================================
# 3. edge 관계
# =============================================================================

def neighbor_relation(
    phase_p,
    phase_q,
):
    """
    p -> q

    raw = phi_q - phi_p

    연속 위상을 위해:
        (phi_q + 2π k_q) - (phi_p + 2π k_p)
    가 작은 값이 되어야 한다.

    따라서:
        k_q - k_p = round(-raw / 2π)
    """
    raw = float(
        phase_q - phase_p
    )

    circular = float(
        wrap_to_pi(
            raw
        )
    )

    delta_k = int(
        np.rint(
            -raw
            / (
                2.0 * np.pi
            )
        )
    )

    return (
        raw,
        circular,
        delta_k,
    )


def iter_neighbors(
    y,
    x,
    h,
    w,
):
    if x > 0:
        yield y, x - 1

    if x < w - 1:
        yield y, x + 1

    if y > 0:
        yield y - 1, x

    if y < h - 1:
        yield y + 1, x


# =============================================================================
# 4. domain의 일반 connected group
# =============================================================================

def label_domain_groups(
    domain,
):
    """
    branch를 막지 않고 단순 valid-domain의 연결 그룹만 구한다.
    서로 끊긴 섬은 global k를 서로 비교할 수 없으므로 별도 root를 둔다.
    """
    domain = np.asarray(
        domain,
        dtype=bool,
    )

    h, w = domain.shape

    labels = np.full(
        (h, w),
        -1,
        dtype=np.int32,
    )

    sizes = []

    label_id = 0

    ys, xs = np.where(
        domain
    )

    for sy, sx in zip(
        ys,
        xs,
    ):
        if labels[
            sy,
            sx
        ] >= 0:
            continue

        q = deque([
            (
                int(sy),
                int(sx),
            )
        ])

        labels[
            sy,
            sx
        ] = label_id

        size = 0

        while q:
            y, x = q.popleft()
            size += 1

            for ny, nx in iter_neighbors(
                y,
                x,
                h,
                w,
            ):
                if not domain[
                    ny,
                    nx
                ]:
                    continue

                if labels[
                    ny,
                    nx
                ] >= 0:
                    continue

                labels[
                    ny,
                    nx
                ] = label_id

                q.append(
                    (
                        ny,
                        nx,
                    )
                )

        sizes.append(
            size
        )

        label_id += 1

    return (
        labels,
        np.asarray(
            sizes,
            dtype=np.int64,
        ),
    )


# =============================================================================
# 5. quality-guided k propagation
# =============================================================================

def choose_seed_near_centroid(
    group_mask,
):
    ys, xs = np.where(
        group_mask
    )

    if ys.size == 0:
        raise RuntimeError(
            "빈 group입니다."
        )

    cy = float(
        np.mean(
            ys
        )
    )

    cx = float(
        np.mean(
            xs
        )
    )

    distance2 = (
        (
            ys - cy
        ) ** 2
        + (
            xs - cx
        ) ** 2
    )

    idx = int(
        np.argmin(
            distance2
        )
    )

    return (
        int(
            ys[
                idx
            ]
        ),
        int(
            xs[
                idx
            ]
        ),
    )


def quality_guided_group(
    phase,
    domain,
    group_mask,
    k_map,
    visited,
):
    """
    Prim-like minimum-cost spanning tree.

    우선순위:
        |circular neighbor difference|가 작은 edge부터.

    장점:
    - 행/열 한 방향 누적이 아님
    - 2π branch edge도 circular 차이가 작으면
      높은 신뢰 edge로 자연스럽게 사용됨
    - branch crossing에서는 delta_k = ±1이 자동으로 붙음

    단계:
    1차: circular <=45°
    2차: circular <=90°
    3차: 남은 픽셀은 제한 없이 연결하되 가장 낮은 cost edge부터
    """
    h, w = phase.shape

    sy, sx = choose_seed_near_centroid(
        group_mask
    )

    k_map[
        sy,
        sx
    ] = 0

    visited[
        sy,
        sx
    ] = True

    heap = []

    def push_from(
        y,
        x,
        max_circ_deg,
        stage,
    ):
        for ny, nx in iter_neighbors(
            y,
            x,
            h,
            w,
        ):
            if not group_mask[
                ny,
                nx
            ]:
                continue

            if visited[
                ny,
                nx
            ]:
                continue

            raw, circ, delta_k = neighbor_relation(
                phase[
                    y,
                    x
                ],
                phase[
                    ny,
                    nx
                ],
            )

            circ_abs_deg = abs(
                np.degrees(
                    circ
                )
            )

            if (
                max_circ_deg is not None
                and circ_abs_deg > max_circ_deg
            ):
                continue

            heapq.heappush(
                heap,
                (
                    circ_abs_deg,
                    stage,
                    y,
                    x,
                    ny,
                    nx,
                    delta_k,
                )
            )

    # -------------------------------------------------------------
    # 1차
    # -------------------------------------------------------------
    push_from(
        sy,
        sx,
        PRIMARY_EDGE_MAX_CIRCULAR_DEG,
        1,
    )

    visited_count = 1
    group_total = int(
        np.count_nonzero(
            group_mask
        )
    )

    while heap:
        (
            cost,
            stage,
            y,
            x,
            ny,
            nx,
            delta_k,
        ) = heapq.heappop(
            heap
        )

        if visited[
            ny,
            nx
        ]:
            continue

        if not visited[
            y,
            x
        ]:
            continue

        k_map[
            ny,
            nx
        ] = (
            k_map[
                y,
                x
            ]
            + delta_k
        )

        visited[
            ny,
            nx
        ] = True

        visited_count += 1

        push_from(
            ny,
            nx,
            PRIMARY_EDGE_MAX_CIRCULAR_DEG,
            1,
        )

    # -------------------------------------------------------------
    # 아직 남은 픽셀이 있으면,
    # 현재 visited 영역에서 2차 edge를 밀어 넣는다.
    # -------------------------------------------------------------
    if visited_count < group_total:
        ys, xs = np.where(
            group_mask
            & visited
        )

        for y, x in zip(
            ys,
            xs,
        ):
            push_from(
                int(y),
                int(x),
                SECONDARY_EDGE_MAX_CIRCULAR_DEG,
                2,
            )

        while heap:
            (
                cost,
                stage,
                y,
                x,
                ny,
                nx,
                delta_k,
            ) = heapq.heappop(
                heap
            )

            if visited[
                ny,
                nx
            ]:
                continue

            if not visited[
                y,
                x
            ]:
                continue

            k_map[
                ny,
                nx
            ] = (
                k_map[
                    y,
                    x
                ]
                + delta_k
            )

            visited[
                ny,
                nx
            ] = True

            visited_count += 1

            push_from(
                ny,
                nx,
                SECONDARY_EDGE_MAX_CIRCULAR_DEG,
                2,
            )

    # -------------------------------------------------------------
    # 그래도 남으면 제한 없이 마지막 연결
    # -------------------------------------------------------------
    if visited_count < group_total:
        ys, xs = np.where(
            group_mask
            & visited
        )

        for y, x in zip(
            ys,
            xs,
        ):
            push_from(
                int(y),
                int(x),
                None,
                3,
            )

        while heap:
            (
                cost,
                stage,
                y,
                x,
                ny,
                nx,
                delta_k,
            ) = heapq.heappop(
                heap
            )

            if visited[
                ny,
                nx
            ]:
                continue

            if not visited[
                y,
                x
            ]:
                continue

            k_map[
                ny,
                nx
            ] = (
                k_map[
                    y,
                    x
                ]
                + delta_k
            )

            visited[
                ny,
                nx
            ] = True

            visited_count += 1

            push_from(
                ny,
                nx,
                None,
                3,
            )

    return {
        "seed_y": int(
            sy
        ),
        "seed_x": int(
            sx
        ),
        "visited_count": int(
            visited_count
        ),
        "group_total": int(
            group_total
        ),
    }


def infer_k_map(
    phase,
    domain,
):
    group_labels, group_sizes = label_domain_groups(
        domain
    )

    k_map = np.zeros(
        phase.shape,
        dtype=np.int32,
    )

    visited = np.zeros(
        phase.shape,
        dtype=bool,
    )

    group_summaries = []

    order = np.argsort(
        -group_sizes
    )

    for group_id in order:
        group_id = int(
            group_id
        )

        group_mask = (
            group_labels
            == group_id
        )

        result = quality_guided_group(
            phase,
            domain,
            group_mask,
            k_map,
            visited,
        )

        result[
            "group_id"
        ] = group_id

        result[
            "pixel_count"
        ] = int(
            group_sizes[
                group_id
            ]
        )

        group_summaries.append(
            result
        )

    if not np.array_equal(
        visited,
        domain,
    ):
        missing = int(
            np.count_nonzero(
                domain
                & (~visited)
            )
        )

        raise RuntimeError(
            f"k 전파 누락 픽셀: {missing}"
        )

    return (
        k_map,
        group_labels,
        group_sizes,
        group_summaries,
    )


# =============================================================================
# 6. 전체 edge 검증
# =============================================================================

def collect_edge_metrics(
    phase,
    domain,
    k_map=None,
):
    h, w = phase.shape

    raw_before = []
    circular = []
    raw_after = []
    desired_delta_k = []
    actual_delta_k = []

    branch_flags = []

    # horizontal
    ys, xs = np.where(
        domain[
            :,
            :-1
        ]
        & domain[
            :,
            1:
        ]
    )

    for y, x in zip(
        ys,
        xs,
    ):
        raw, circ, dk = neighbor_relation(
            phase[
                y,
                x
            ],
            phase[
                y,
                x + 1
            ],
        )

        raw_deg = abs(
            np.degrees(
                raw
            )
        )

        circ_deg = abs(
            np.degrees(
                circ
            )
        )

        is_branch = (
            raw_deg
            >= RAW_BRANCH_MIN_DEG
            and circ_deg
            <= BRANCH_CIRCULAR_MAX_DEG
        )

        raw_before.append(
            raw_deg
        )

        circular.append(
            circ_deg
        )

        desired_delta_k.append(
            dk
        )

        branch_flags.append(
            is_branch
        )

        if k_map is not None:
            actual_dk = int(
                k_map[
                    y,
                    x + 1
                ]
                - k_map[
                    y,
                    x
                ]
            )

            corrected = (
                raw
                + 2.0 * np.pi * actual_dk
            )

            raw_after.append(
                abs(
                    np.degrees(
                        corrected
                    )
                )
            )

            actual_delta_k.append(
                actual_dk
            )

    # vertical
    ys, xs = np.where(
        domain[
            :-1,
            :
        ]
        & domain[
            1:,
            :
        ]
    )

    for y, x in zip(
        ys,
        xs,
    ):
        raw, circ, dk = neighbor_relation(
            phase[
                y,
                x
            ],
            phase[
                y + 1,
                x
            ],
        )

        raw_deg = abs(
            np.degrees(
                raw
            )
        )

        circ_deg = abs(
            np.degrees(
                circ
            )
        )

        is_branch = (
            raw_deg
            >= RAW_BRANCH_MIN_DEG
            and circ_deg
            <= BRANCH_CIRCULAR_MAX_DEG
        )

        raw_before.append(
            raw_deg
        )

        circular.append(
            circ_deg
        )

        desired_delta_k.append(
            dk
        )

        branch_flags.append(
            is_branch
        )

        if k_map is not None:
            actual_dk = int(
                k_map[
                    y + 1,
                    x
                ]
                - k_map[
                    y,
                    x
                ]
            )

            corrected = (
                raw
                + 2.0 * np.pi * actual_dk
            )

            raw_after.append(
                abs(
                    np.degrees(
                        corrected
                    )
                )
            )

            actual_delta_k.append(
                actual_dk
            )

    raw_before = np.asarray(
        raw_before,
        dtype=np.float64,
    )

    circular = np.asarray(
        circular,
        dtype=np.float64,
    )

    desired_delta_k = np.asarray(
        desired_delta_k,
        dtype=np.int32,
    )

    branch_flags = np.asarray(
        branch_flags,
        dtype=bool,
    )

    result = {
        "edge_count": int(
            raw_before.size
        ),

        "before_over180_percent": float(
            np.mean(
                raw_before
                > 180.0
            )
            * 100.0
        ),

        "before_over300_percent": float(
            np.mean(
                raw_before
                > 300.0
            )
            * 100.0
        ),

        "before_median_deg": float(
            np.median(
                raw_before
            )
        ),

        "before_p90_deg": float(
            np.percentile(
                raw_before,
                90,
            )
        ),

        "circular_median_deg": float(
            np.median(
                circular
            )
        ),

        "circular_p90_deg": float(
            np.percentile(
                circular,
                90,
            )
        ),

        "branch_edge_count": int(
            np.count_nonzero(
                branch_flags
            )
        ),
    }

    if k_map is None:
        return result

    raw_after = np.asarray(
        raw_after,
        dtype=np.float64,
    )

    actual_delta_k = np.asarray(
        actual_delta_k,
        dtype=np.int32,
    )

    constraint_match = (
        actual_delta_k
        == desired_delta_k
    )

    branch_constraint_match = (
        constraint_match[
            branch_flags
        ]
    )

    normal_constraint_match = (
        constraint_match[
            ~branch_flags
        ]
    )

    result.update({
        "after_over180_percent": float(
            np.mean(
                raw_after
                > 180.0
            )
            * 100.0
        ),

        "after_over300_percent": float(
            np.mean(
                raw_after
                > 300.0
            )
            * 100.0
        ),

        "after_median_deg": float(
            np.median(
                raw_after
            )
        ),

        "after_p90_deg": float(
            np.percentile(
                raw_after,
                90,
            )
        ),

        "all_edge_constraint_match_percent": float(
            np.mean(
                constraint_match
            )
            * 100.0
        ),

        "branch_edge_constraint_match_percent": (
            float(
                np.mean(
                    branch_constraint_match
                )
                * 100.0
            )
            if branch_constraint_match.size
            else np.nan
        ),

        "normal_edge_constraint_match_percent": (
            float(
                np.mean(
                    normal_constraint_match
                )
                * 100.0
            )
            if normal_constraint_match.size
            else np.nan
        ),

        "desired_delta_k_histogram": {
            str(int(k)): int(v)
            for k, v in sorted(
                Counter(
                    desired_delta_k.tolist()
                ).items()
            )
        },

        "actual_delta_k_histogram": {
            str(int(k)): int(v)
            for k, v in sorted(
                Counter(
                    actual_delta_k.tolist()
                ).items()
            )
        },
    })

    return result


# =============================================================================
# 7. 시각화
# =============================================================================

def save_k_map_figure(
    path,
    k_map,
    domain,
):
    values = k_map[
        domain
    ]

    k_min = int(
        values.min()
    )

    k_max = int(
        values.max()
    )

    shown = np.where(
        domain,
        k_map.astype(
            np.float32
        ),
        np.nan,
    )

    plt.figure(
        figsize=(12, 7)
    )

    im = plt.imshow(
        shown,
        cmap="tab20",
        interpolation="nearest",
    )

    cbar = plt.colorbar(
        im
    )

    cbar.set_label(
        "branch index k"
    )

    plt.title(
        f"Quality-guided k map | range {k_min} ... {k_max}"
    )

    plt.axis("off")
    plt.tight_layout()

    plt.savefig(
        path,
        dpi=180,
        bbox_inches="tight",
    )

    plt.close()


def save_k_overlay(
    path,
    preview,
    k_map,
    domain,
):
    if preview is None:
        return

    rgb = cv2.cvtColor(
        preview,
        cv2.COLOR_BGR2RGB,
    )

    masked = np.ma.masked_where(
        ~domain,
        k_map,
    )

    plt.figure(
        figsize=(12, 7)
    )

    plt.imshow(
        rgb
    )

    plt.imshow(
        masked,
        cmap="tab20",
        alpha=0.50,
        interpolation="nearest",
    )

    plt.title(
        "P20 preview + quality-guided k map"
    )

    plt.axis("off")
    plt.tight_layout()

    plt.savefig(
        path,
        dpi=180,
        bbox_inches="tight",
    )

    plt.close()


def save_phase_image(
    path,
    phase,
    domain,
    title,
):
    shown = np.where(
        domain,
        phase,
        np.nan,
    )

    plt.figure(
        figsize=(12, 7)
    )

    im = plt.imshow(
        shown,
        cmap="turbo",
        interpolation="nearest",
    )

    plt.colorbar(
        im,
        label="phase (rad)",
    )

    plt.title(
        title
    )

    plt.axis("off")
    plt.tight_layout()

    plt.savefig(
        path,
        dpi=180,
        bbox_inches="tight",
    )

    plt.close()


def save_jump_map(
    path,
    phase,
    domain,
    title,
):
    h, w = phase.shape

    jump = np.full(
        phase.shape,
        np.nan,
        dtype=np.float32,
    )

    # right/down 중 큰 linear jump
    right = np.full(
        phase.shape,
        np.nan,
        dtype=np.float32,
    )

    down = np.full(
        phase.shape,
        np.nan,
        dtype=np.float32,
    )

    valid_x = (
        domain[
            :,
            :-1
        ]
        & domain[
            :,
            1:
        ]
    )

    valid_y = (
        domain[
            :-1,
            :
        ]
        & domain[
            1:,
            :
        ]
    )

    dx = np.abs(
        np.degrees(
            phase[
                :,
                1:
            ]
            - phase[
                :,
                :-1
            ]
        )
    )

    dy = np.abs(
        np.degrees(
            phase[
                1:,
                :
            ]
            - phase[
                :-1,
                :
            ]
        )
    )

    tmp = right[
        :,
        :-1
    ]

    tmp[
        valid_x
    ] = dx[
        valid_x
    ]

    tmp = down[
        :-1,
        :
    ]

    tmp[
        valid_y
    ] = dy[
        valid_y
    ]

    both = np.stack(
        [
            np.nan_to_num(
                right,
                nan=-1.0,
            ),
            np.nan_to_num(
                down,
                nan=-1.0,
            ),
        ],
        axis=0,
    )

    jump = np.max(
        both,
        axis=0,
    )

    jump[
        jump < 0
    ] = np.nan

    shown = np.where(
        domain,
        np.clip(
            jump,
            0,
            360,
        ),
        np.nan,
    )

    plt.figure(
        figsize=(12, 7)
    )

    im = plt.imshow(
        shown,
        cmap="turbo",
        vmin=0,
        vmax=360,
        interpolation="nearest",
    )

    plt.colorbar(
        im,
        label="max linear neighbor jump (deg)",
    )

    plt.title(
        title
    )

    plt.axis("off")
    plt.tight_layout()

    plt.savefig(
        path,
        dpi=180,
        bbox_inches="tight",
    )

    plt.close()


# =============================================================================
# 8. MAIN
# =============================================================================

def main():
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    require(
        PHASE_PATH,
        "phase_difference_masked.npy",
    )

    require(
        VALID_PATH,
        "common_valid_mask.npy",
    )

    phase = np.load(
        PHASE_PATH
    ).astype(
        np.float64
    )

    common_valid = np.load(
        VALID_PATH
    ).astype(
        bool
    )

    if phase.shape != common_valid.shape:
        raise ValueError(
            f"phase/valid 크기 불일치: "
            f"{phase.shape} != {common_valid.shape}"
        )

    shape = phase.shape

    object_mask, object_mask_path = find_first_mask(
        OBJECT_MASK_CANDIDATES,
        shape,
        "본넷 마스크",
    )

    domain = (
        object_mask
        & common_valid
        & np.isfinite(
            phase
        )
    )

    preview = load_preview(
        shape
    )

    print("")
    print("=" * 112)
    print(
        "Quality-guided pixel-level branch-index(k) 진단"
    )
    print("=" * 112)

    print(
        f"유효 본넷 픽셀: "
        f"{np.count_nonzero(domain)}"
    )

    before_stats = collect_edge_metrics(
        phase,
        domain,
        k_map=None,
    )

    print("")
    print(
        "[적용 전]"
    )

    print(
        f"전체 인접 edge: "
        f"{before_stats['edge_count']}"
    )

    print(
        f">180°: "
        f"{before_stats['before_over180_percent']:.4f}%"
    )

    print(
        f">300°: "
        f"{before_stats['before_over300_percent']:.4f}%"
    )

    print(
        f"linear jump 중앙값/P90: "
        f"{before_stats['before_median_deg']:.2f}° / "
        f"{before_stats['before_p90_deg']:.2f}°"
    )

    print(
        f"circular jump 중앙값/P90: "
        f"{before_stats['circular_median_deg']:.2f}° / "
        f"{before_stats['circular_p90_deg']:.2f}°"
    )

    print(
        f"high-confidence branch edge: "
        f"{before_stats['branch_edge_count']}"
    )

    print("")
    print(
        "k-map 전파 중..."
    )

    (
        k_map,
        group_labels,
        group_sizes,
        group_summaries,
    ) = infer_k_map(
        phase,
        domain,
    )

    candidate = (
        phase
        + 2.0
        * np.pi
        * k_map
    )

    after_stats = collect_edge_metrics(
        phase,
        domain,
        k_map=k_map,
    )

    k_values, k_pixel_counts = np.unique(
        k_map[
            domain
        ],
        return_counts=True,
    )

    print("")
    print(
        "[k 분포]"
    )

    for k_value, count in zip(
        k_values,
        k_pixel_counts,
    ):
        print(
            f"k={int(k_value):+d}: "
            f"{int(count)} px"
        )

    print(
        f"k range: "
        f"{int(k_values.min())} ~ {int(k_values.max())}"
    )

    print(
        f"valid-domain 독립 group: "
        f"{len(group_sizes)}"
    )

    print("")
    print(
        "[k 적용 가정 후]"
    )

    print(
        f">180°: "
        f"{after_stats['after_over180_percent']:.4f}%"
    )

    print(
        f">300°: "
        f"{after_stats['after_over300_percent']:.4f}%"
    )

    print(
        f"linear jump 중앙값/P90: "
        f"{after_stats['after_median_deg']:.2f}° / "
        f"{after_stats['after_p90_deg']:.2f}°"
    )

    print(
        f"전체 edge k-constraint 일치: "
        f"{after_stats['all_edge_constraint_match_percent']:.2f}%"
    )

    print(
        f"branch edge k-constraint 일치: "
        f"{after_stats['branch_edge_constraint_match_percent']:.2f}%"
    )

    print(
        f"normal edge k-constraint 일치: "
        f"{after_stats['normal_edge_constraint_match_percent']:.2f}%"
    )

    if (
        np.max(
            np.abs(
                k_values
            )
        )
        >= K_ABS_WARNING
    ):
        print("")
        print(
            "⚠ 경고: |k|가 크게 나온 영역이 있습니다."
        )

        print(
            "k-map 이미지를 반드시 확인한 뒤 "
            "실제 형상 복원에는 아직 사용하지 마세요."
        )

    # -------------------------------------------------------------------------
    # 저장
    # -------------------------------------------------------------------------
    np.save(
        OUTPUT_DIR
        / "branch_index_k_map.npy",
        k_map,
    )

    np.save(
        OUTPUT_DIR
        / "candidate_unwrapped_phase.npy",
        candidate.astype(
            np.float32
        ),
    )

    np.save(
        OUTPUT_DIR
        / "domain_group_label.npy",
        group_labels,
    )

    save_mask(
        OUTPUT_DIR
        / "00_유효본넷마스크.png",
        domain,
    )

    save_k_map_figure(
        OUTPUT_DIR
        / "01_quality_guided_k_map.png",
        k_map,
        domain,
    )

    save_k_overlay(
        OUTPUT_DIR
        / "02_P20preview_kmap_overlay.png",
        preview,
        k_map,
        domain,
    )

    save_phase_image(
        OUTPUT_DIR
        / "03_적용전_phase_difference.png",
        phase,
        domain,
        "Before: raw phase_difference_masked",
    )

    save_phase_image(
        OUTPUT_DIR
        / "04_k적용후_candidate_unwrapped_phase.png",
        candidate,
        domain,
        "After candidate: phase + 2*pi*k",
    )

    save_jump_map(
        OUTPUT_DIR
        / "05_적용전_linear_neighbor_jump.png",
        phase,
        domain,
        "Before: max linear neighbor jump",
    )

    save_jump_map(
        OUTPUT_DIR
        / "06_k적용후_linear_neighbor_jump.png",
        candidate,
        domain,
        "After candidate: max linear neighbor jump",
    )

    # k 분포 CSV
    k_csv = (
        OUTPUT_DIR
        / "10_k_pixel_distribution.csv"
    )

    with k_csv.open(
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as f:
        writer = csv.writer(
            f
        )

        writer.writerow(
            [
                "k",
                "pixel_count",
            ]
        )

        for k_value, count in zip(
            k_values,
            k_pixel_counts,
        ):
            writer.writerow(
                [
                    int(k_value),
                    int(count),
                ]
            )

    summary = {
        "input": {
            "phase_path": str(
                PHASE_PATH
            ),
            "valid_path": str(
                VALID_PATH
            ),
            "object_mask_path": str(
                object_mask_path
            ),
        },

        "parameters": {
            "raw_branch_min_deg": RAW_BRANCH_MIN_DEG,
            "branch_circular_max_deg": BRANCH_CIRCULAR_MAX_DEG,
            "primary_edge_max_circular_deg": PRIMARY_EDGE_MAX_CIRCULAR_DEG,
            "secondary_edge_max_circular_deg": SECONDARY_EDGE_MAX_CIRCULAR_DEG,
        },

        "before": before_stats,

        "after": after_stats,

        "k": {
            "min": int(
                k_values.min()
            ),
            "max": int(
                k_values.max()
            ),

            "pixel_distribution": {
                str(
                    int(k_value)
                ): int(
                    count
                )
                for k_value, count
                in zip(
                    k_values,
                    k_pixel_counts,
                )
            },
        },

        "domain_groups": {
            "count": int(
                len(
                    group_sizes
                )
            ),

            "sizes_top20": [
                int(v)
                for v
                in np.sort(
                    group_sizes
                )[::-1][
                    :20
                ]
            ],

            "group_summaries": group_summaries,
        },

        "important": (
            "candidate_unwrapped_phase.npy는 진단 후보 파일일 뿐 원본을 덮어쓰지 않는다. "
            "k-map이 공간적으로 자연스럽고, branch-edge constraint 일치율이 높으며, "
            "k 적용 후 >300도 및 >180도 linear jump가 크게 감소하는지 확인한 뒤에만 "
            "다음 단계에서 relative surface/PLY 입력으로 사용할 것."
        ),
    }

    summary_path = (
        OUTPUT_DIR
        / "11_quality_guided_kmap_진단요약.json"
    )

    summary_path.write_text(
        json.dumps(
            summary,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print("")
    print("=" * 112)
    print(
        "완료"
    )
    print("=" * 112)

    print("")
    print(
        "[가장 먼저 확인할 파일]"
    )

    for filename in [
        "01_quality_guided_k_map.png",
        "02_P20preview_kmap_overlay.png",
        "03_적용전_phase_difference.png",
        "04_k적용후_candidate_unwrapped_phase.png",
        "05_적용전_linear_neighbor_jump.png",
        "06_k적용후_linear_neighbor_jump.png",
        "11_quality_guided_kmap_진단요약.json",
    ]:
        print(
            OUTPUT_DIR
            / filename
        )

    print("")
    print(
        "candidate_unwrapped_phase.npy는 별도 폴더에만 저장됨"
    )

    print(
        "원본 phase_difference_masked.npy 변경 없음"
    )

    print(
        "PLY 생성 없음 / smoothing 없음 / interpolation 없음"
    )

    print("=" * 112)


if __name__ == "__main__":
    main()
'''

원본_PLATFORM_STAGE_SOURCE = r'''
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
남색 본넷 기존 quality-guided 결과를 그대로 사용하여
'실제 플랫폼 영역만' 기준으로 평면을 계산한 뒤 PLY를 다시 생성한다.

핵심 목적
1) 기존 candidate_unwrapped_phase.npy는 다시 계산하지 않음
2) 기존 Graph-Cut 결과도 사용하지 않음 (기본은 quality_guided_kmap_진단 결과)
3) Depth에서 검출해 저장했던 실제 본넷 픽셀 마스크를 이용해 본넷을 제외
4) 남은 플랫폼 픽셀만으로 robust plane fitting
5) 그 플랫폼 plane을 전체 유효 영역에서 빼서 플랫폼 기울기 제거
6) CloudCompare 확인용 RAW / 기존 시각화 방식 PLY 두 개 생성

새 촬영 없음 / 카메라 없음 / 프로젝터 없음
"""

from pathlib import Path
import json

import cv2
import numpy as np


# =============================================================================
# 0. 현재 남색 실험 경로
# =============================================================================
STRUCT_DIR = Path(
    "/home/seoyeong/졸업작품/전처리와구조광_통합/플랫폼 바닥 따기/전처리_실험용/"
    "남색/촬영_20260813_125737/전처리_결과/구조광_형상복원"
)

통합_출력_루트 = STRUCT_DIR / "3_6통합 버전"

CLEAN_HDR_DIR = (
    통합_출력_루트
    / "05_기존HDR원리_원래마스크유지_2pi언랩_최종"
)

# Graph-Cut 보정은 이번에 사실상 차이가 거의 없었으므로
# 기존 quality-guided 결과를 기본 입력으로 사용한다.
PHASE_CANDIDATES = [
    통합_출력_루트
    / "06_quality_guided_kmap_진단"
    / "candidate_unwrapped_phase.npy",
]

COMMON_VALID_PATH = CLEAN_HDR_DIR / "common_valid_mask.npy"
ANALYSIS_AREA_PATH = STRUCT_DIR / "object_area_mask.npy"

# 통합 코드가 Depth 자동 검출 때 저장했던 폴더
DEPTH_RESULT_DIR = Path(
    "/home/seoyeong/졸업작품/전처리와구조광_통합/플랫폼 바닥 따기/전처리_실험용/남색"
)

# 이 PNG는 통합 코드에서 실제 Depth 물체 픽셀을 180도 회전한 뒤 저장한 것.
DEPTH_OBJECT_MASK_PNG = (
    DEPTH_RESULT_DIR / "03_Depth_물체픽셀_확인용.png"
)

# PNG가 없을 때만 재구성용
CURRENT_DEPTH_PATH = DEPTH_RESULT_DIR / "현재_물체_depth.npy"
BACKGROUND_DEPTH_PATH = Path(
    "/home/seoyeong/졸업작품/전처리와구조광_통합/플랫폼 바닥 따기/플랫폼_바닥_depth.npy"
)

OUTPUT_DIR = 통합_출력_루트 / "07_플랫폼기준면_PLY_재생성"


# =============================================================================
# 1. 파라미터
# =============================================================================
# 본넷 가장자리까지 플랫폼 fitting에 섞이지 않도록 여유 있게 제외
OBJECT_EXCLUDE_DILATE_PX = 15

# robust plane fitting
ROBUST_ITERATIONS = 6
ROBUST_MAD_SIGMA = 3.5
ROBUST_MIN_ABS_THRESHOLD_RAD = 0.03
MIN_PLATFORM_POINTS = 2000
MAX_FIT_POINTS = 150000

# CloudCompare 시각화 조건: 어제 사용하던 상대 PLY 조건 유지
POINT_SKIP = 2
RELATIVE_Z_SIGN = -1.0
RELATIVE_Z_SCALE = 40.0

# 비교용 '기존 Relative 방식' 시각화
PHASE_P_LOW = 10.0
PHASE_P_HIGH = 90.0
MEDIAN_KSIZE = 5
GAUSSIAN_KSIZE = 7


# =============================================================================
# 2. 기본 함수
# =============================================================================
def require(path, label):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"{label} 없음:\n{path}")
    return path


def load_first_existing(paths, label):
    for path in paths:
        path = Path(path)
        if path.exists():
            print(f"{label} 사용: {path}")
            return np.load(path), path

    joined = "\n".join(str(Path(p)) for p in paths)
    raise FileNotFoundError(
        f"{label} 후보를 찾지 못했습니다.\n{joined}"
    )


def save_mask(path, mask):
    image = np.zeros(mask.shape, dtype=np.uint8)
    image[np.asarray(mask, dtype=bool)] = 255
    cv2.imwrite(str(path), image)


def save_phase_color(path, data, mask, p_low=1.0, p_high=99.0):
    data = np.asarray(data, dtype=np.float32)
    valid = np.asarray(mask, dtype=bool) & np.isfinite(data)

    gray = np.zeros(data.shape, dtype=np.uint8)

    if np.any(valid):
        values = data[valid]
        low = float(np.percentile(values, p_low))
        high = float(np.percentile(values, p_high))

        if high <= low:
            high = low + 1e-6

        normalized = np.clip(
            (data - low) / (high - low),
            0.0,
            1.0,
        )
        gray[valid] = np.rint(
            normalized[valid] * 255.0
        ).astype(np.uint8)

    color = cv2.applyColorMap(gray, cv2.COLORMAP_TURBO)
    color[~valid] = (0, 0, 0)
    cv2.imwrite(str(path), color)


def largest_connected(mask):
    mask_u8 = np.asarray(mask, dtype=np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask_u8,
        connectivity=8,
    )

    if count <= 1:
        return np.zeros(mask_u8.shape, dtype=bool)

    areas = stats[1:, cv2.CC_STAT_AREA]
    label = int(np.argmax(areas)) + 1
    return labels == label


def grow_from_seed(seed_mask, allowed_mask):
    marker = seed_mask.astype(np.uint8)
    allowed = allowed_mask.astype(np.uint8)
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (3, 3),
    )

    while True:
        grown = cv2.dilate(marker, kernel, iterations=1)
        grown = (
            (grown > 0)
            & (allowed > 0)
        ).astype(np.uint8)

        if np.array_equal(grown, marker):
            break

        marker = grown

    return marker > 0


def reconstruct_depth_object_mask(shape):
    """
    03_Depth_물체픽셀_확인용.png가 없을 때만 사용하는 fallback.
    통합 코드의 10 mm seed + 5 mm 연결 확장 논리를 최대한 동일하게 재현한다.
    """
    require(CURRENT_DEPTH_PATH, "현재 물체 Depth")
    require(BACKGROUND_DEPTH_PATH, "빈 플랫폼 Depth")

    current = np.load(CURRENT_DEPTH_PATH).astype(np.float32)
    background = np.load(BACKGROUND_DEPTH_PATH).astype(np.float32)

    if current.shape != background.shape:
        raise ValueError(
            f"현재/배경 Depth 크기 불일치: {current.shape} != {background.shape}"
        )

    height_diff = background - current
    valid = (background > 0) & (current > 0)

    seed = valid & (height_diff >= 10.0)
    grow = valid & (height_diff >= 5.0)

    open_k = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (3, 3),
    )
    close_k = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (11, 11),
    )

    seed_clean = cv2.morphologyEx(
        seed.astype(np.uint8) * 255,
        cv2.MORPH_OPEN,
        open_k,
    )
    seed_clean = cv2.morphologyEx(
        seed_clean,
        cv2.MORPH_CLOSE,
        close_k,
    ) > 0

    grow_clean = cv2.morphologyEx(
        grow.astype(np.uint8) * 255,
        cv2.MORPH_OPEN,
        open_k,
    )
    grow_clean = cv2.morphologyEx(
        grow_clean,
        cv2.MORPH_CLOSE,
        close_k,
    ) > 0

    largest_seed = largest_connected(seed_clean)
    if not np.any(largest_seed):
        raise RuntimeError("Depth fallback에서 물체 seed를 찾지 못했습니다.")

    object_mask = grow_from_seed(largest_seed, grow_clean)
    object_mask = cv2.morphologyEx(
        object_mask.astype(np.uint8) * 255,
        cv2.MORPH_CLOSE,
        close_k,
    ) > 0

    # 구조광 저장 영상은 180도 회전되어 있으므로 맞춰 준다.
    object_mask = cv2.rotate(
        object_mask.astype(np.uint8) * 255,
        cv2.ROTATE_180,
    ) > 0

    if object_mask.shape != tuple(shape):
        raise ValueError(
            f"재구성 Depth 물체마스크 크기 불일치: "
            f"{object_mask.shape} != {shape}"
        )

    return object_mask


def load_depth_object_mask(shape):
    if DEPTH_OBJECT_MASK_PNG.exists():
        image = cv2.imread(
            str(DEPTH_OBJECT_MASK_PNG),
            cv2.IMREAD_GRAYSCALE,
        )
        if image is None:
            raise RuntimeError(
                f"Depth 물체 마스크 PNG 읽기 실패: {DEPTH_OBJECT_MASK_PNG}"
            )

        mask = image > 127

        if mask.shape != tuple(shape):
            raise ValueError(
                f"Depth 물체 마스크 크기 불일치: "
                f"{mask.shape} != {shape}"
            )

        print(f"Depth 실제 본넷 마스크 사용: {DEPTH_OBJECT_MASK_PNG}")
        return mask, str(DEPTH_OBJECT_MASK_PNG)

    print("Depth 실제 본넷 PNG가 없어 현재/빈 플랫폼 Depth로 재구성합니다.")
    mask = reconstruct_depth_object_mask(shape)
    return mask, "Depth NPY 재구성"


def dilate_mask(mask, radius_px):
    radius_px = max(0, int(radius_px))
    if radius_px <= 0:
        return np.asarray(mask, dtype=bool).copy()

    k = radius_px * 2 + 1
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (k, k),
    )

    return cv2.dilate(
        np.asarray(mask, dtype=np.uint8),
        kernel,
        iterations=1,
    ) > 0


# =============================================================================
# 3. 플랫폼 전용 robust plane fitting
# =============================================================================
def fit_plane_lstsq(x, y, z):
    A = np.column_stack([
        x,
        y,
        np.ones(x.size, dtype=np.float64),
    ])

    coeff, _, _, _ = np.linalg.lstsq(A, z, rcond=None)
    return [float(v) for v in coeff]


def robust_platform_plane(data, platform_mask):
    data = np.asarray(data, dtype=np.float64)
    platform_mask = (
        np.asarray(platform_mask, dtype=bool)
        & np.isfinite(data)
    )

    yy, xx = np.mgrid[0:data.shape[0], 0:data.shape[1]]

    ys, xs = np.where(platform_mask)
    zs = data[platform_mask]

    if zs.size < MIN_PLATFORM_POINTS:
        raise RuntimeError(
            f"플랫폼 평면 fitting 픽셀이 너무 적습니다: {zs.size} px"
        )

    # 계산량만 제한. 공간적으로 고르게 뽑는다.
    if zs.size > MAX_FIT_POINTS:
        indices = np.linspace(
            0,
            zs.size - 1,
            MAX_FIT_POINTS,
            dtype=np.int64,
        )
        xs_fit = xs[indices].astype(np.float64)
        ys_fit = ys[indices].astype(np.float64)
        zs_fit = zs[indices].astype(np.float64)
    else:
        xs_fit = xs.astype(np.float64)
        ys_fit = ys.astype(np.float64)
        zs_fit = zs.astype(np.float64)

    inlier = np.ones(zs_fit.shape, dtype=bool)
    threshold = np.nan

    for _ in range(ROBUST_ITERATIONS):
        if np.count_nonzero(inlier) < MIN_PLATFORM_POINTS:
            break

        a, b, c = fit_plane_lstsq(
            xs_fit[inlier],
            ys_fit[inlier],
            zs_fit[inlier],
        )

        prediction = a * xs_fit + b * ys_fit + c
        residual = zs_fit - prediction

        center = float(np.median(residual[inlier]))
        abs_dev = np.abs(residual[inlier] - center)
        mad = float(np.median(abs_dev))
        sigma = 1.4826 * mad

        threshold = max(
            ROBUST_MIN_ABS_THRESHOLD_RAD,
            ROBUST_MAD_SIGMA * sigma,
        )

        new_inlier = np.abs(residual - center) <= threshold

        if np.array_equal(new_inlier, inlier):
            inlier = new_inlier
            break

        inlier = new_inlier

    if np.count_nonzero(inlier) < MIN_PLATFORM_POINTS:
        raise RuntimeError(
            "robust fitting 후 플랫폼 inlier가 너무 적습니다: "
            f"{np.count_nonzero(inlier)} px"
        )

    a, b, c = fit_plane_lstsq(
        xs_fit[inlier],
        ys_fit[inlier],
        zs_fit[inlier],
    )

    plane = a * xx + b * yy + c

    # 전체 플랫폼 후보에서 residual 통계
    all_residual = zs - (
        a * xs.astype(np.float64)
        + b * ys.astype(np.float64)
        + c
    )

    return plane.astype(np.float64), {
        "a_rad_per_pixel_x": float(a),
        "b_rad_per_pixel_y": float(b),
        "c_rad": float(c),
        "fit_candidate_points": int(zs.size),
        "fit_used_points": int(zs_fit.size),
        "fit_inliers": int(np.count_nonzero(inlier)),
        "fit_inlier_percent": float(
            np.count_nonzero(inlier) / max(1, inlier.size) * 100.0
        ),
        "robust_threshold_rad": float(threshold),
        "platform_residual_median_rad": float(np.median(all_residual)),
        "platform_residual_mad_rad": float(
            np.median(np.abs(all_residual - np.median(all_residual)))
        ),
        "platform_residual_p90_abs_rad": float(
            np.percentile(np.abs(all_residual), 90)
        ),
    }


def plane_slope_of_data(data, mask):
    valid = np.asarray(mask, dtype=bool) & np.isfinite(data)
    ys, xs = np.where(valid)
    z = np.asarray(data, dtype=np.float64)[valid]

    if z.size < 100:
        return None

    a, b, c = fit_plane_lstsq(
        xs.astype(np.float64),
        ys.astype(np.float64),
        z,
    )

    return {
        "a_rad_per_pixel_x": float(a),
        "b_rad_per_pixel_y": float(b),
        "c_rad": float(c),
    }


# =============================================================================
# 4. PLY
# =============================================================================
def save_ply_ascii(path, points, colors):
    with Path(path).open("w", encoding="ascii") as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {len(points)}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("property uchar red\n")
        f.write("property uchar green\n")
        f.write("property uchar blue\n")
        f.write("end_header\n")

        for p, color in zip(points, colors):
            f.write(
                f"{p[0]:.6f} {p[1]:.6f} {p[2]:.6f} "
                f"{int(color[0])} {int(color[1])} {int(color[2])}\n"
            )


def make_ply(data, valid_mask, path, skip=POINT_SKIP):
    data = np.asarray(data, dtype=np.float32)
    valid = np.asarray(valid_mask, dtype=bool) & np.isfinite(data)

    values = data[valid]
    if values.size == 0:
        raise RuntimeError("PLY 생성용 유효값이 없습니다.")

    color_low = float(np.percentile(values, 1))
    color_high = float(np.percentile(values, 99))
    if color_high <= color_low:
        color_high = color_low + 1e-6

    h, w = data.shape
    points = []
    colors = []

    skip = max(1, int(skip))

    for y in range(0, h, skip):
        for x in range(0, w, skip):
            if not valid[y, x]:
                continue

            z_phase = float(data[y, x])

            X = float(x - w / 2.0)
            Y = float((h - y) - h / 2.0)
            Z = float(
                RELATIVE_Z_SIGN
                * z_phase
                * RELATIVE_Z_SCALE
            )

            norm = int(np.clip(
                255.0
                * (z_phase - color_low)
                / (color_high - color_low + 1e-9),
                0,
                255,
            ))

            bgr = cv2.applyColorMap(
                np.array([[norm]], dtype=np.uint8),
                cv2.COLORMAP_TURBO,
            )[0, 0]

            rgb = bgr[::-1]
            points.append([X, Y, Z])
            colors.append(rgb.tolist())

    points = np.asarray(points, dtype=np.float32)
    colors = np.asarray(colors, dtype=np.uint8)

    save_ply_ascii(path, points, colors)
    return int(len(points))


def masked_gaussian(data, mask, ksize):
    ksize = max(1, int(ksize))
    if ksize % 2 == 0:
        ksize += 1

    if ksize <= 1:
        out = np.asarray(data, dtype=np.float32).copy()
        out[~mask] = np.nan
        return out

    source = np.where(mask, data, 0.0).astype(np.float32)
    weight = mask.astype(np.float32)

    numerator = cv2.GaussianBlur(source, (ksize, ksize), 0)
    denominator = cv2.GaussianBlur(weight, (ksize, ksize), 0)

    result = numerator / (denominator + 1e-9)
    result[denominator < 0.25] = np.nan
    result[~mask] = np.nan
    return result.astype(np.float32)


def existing_visualization_process(plane_corrected, valid):
    """
    어제의 '02_candidate_기존Relative방식'과 비교하기 쉽게
    10/90 clip + median5 + Gaussian7 흐름을 유지한다.
    플랫폼 plane은 이미 실제 플랫폼만으로 제거된 상태에서 적용한다.
    """
    data = np.asarray(plane_corrected, dtype=np.float32).copy()
    valid = np.asarray(valid, dtype=bool) & np.isfinite(data)

    values = data[valid]
    low = float(np.percentile(values, PHASE_P_LOW))
    high = float(np.percentile(values, PHASE_P_HIGH))

    clipped = np.clip(data, low, high)
    clipped[~valid] = np.nan

    fill_value = float(np.median(clipped[valid]))
    filled = np.where(valid, clipped, fill_value).astype(np.float32)

    dmin = float(np.min(filled))
    dmax = float(np.max(filled))

    normalized = np.clip(
        (filled - dmin) / (dmax - dmin + 1e-9) * 255.0,
        0,
        255,
    ).astype(np.uint8)

    k = max(1, int(MEDIAN_KSIZE))
    if k % 2 == 0:
        k += 1

    median_u8 = cv2.medianBlur(normalized, k)
    median_f = (
        median_u8.astype(np.float32)
        / 255.0
        * (dmax - dmin)
        + dmin
    )
    median_f[~valid] = np.nan

    smooth = masked_gaussian(
        median_f,
        valid,
        GAUSSIAN_KSIZE,
    )

    return smooth, {
        "clip_low_rad": low,
        "clip_high_rad": high,
        "median_ksize": int(k),
        "gaussian_ksize": int(GAUSSIAN_KSIZE),
    }


# =============================================================================
# 5. MAIN
# =============================================================================
def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    candidate, candidate_path = load_first_existing(
        PHASE_CANDIDATES,
        "candidate unwrapped phase",
    )
    candidate = candidate.astype(np.float64)

    require(COMMON_VALID_PATH, "common_valid_mask.npy")
    require(ANALYSIS_AREA_PATH, "object_area_mask.npy")

    common_valid = np.load(COMMON_VALID_PATH).astype(bool)
    analysis_area = np.load(ANALYSIS_AREA_PATH).astype(bool)

    if not (
        candidate.shape
        == common_valid.shape
        == analysis_area.shape
    ):
        raise ValueError(
            "입력 크기 불일치: "
            f"candidate={candidate.shape}, "
            f"common_valid={common_valid.shape}, "
            f"analysis_area={analysis_area.shape}"
        )

    shape = candidate.shape

    # 6번에서 실제로 사용된 전체 유효 사각 분석영역
    domain = (
        analysis_area
        & common_valid
        & np.isfinite(candidate)
    )

    depth_object_mask, depth_mask_source = load_depth_object_mask(shape)
    depth_object_mask &= analysis_area

    # 본넷 가장자리 주변을 조금 더 제외해서 플랫폼 plane fitting 오염 방지
    object_excluded = dilate_mask(
        depth_object_mask,
        OBJECT_EXCLUDE_DILATE_PX,
    )

    platform_fit_mask = (
        domain
        & (~object_excluded)
    )

    platform_count = int(np.count_nonzero(platform_fit_mask))
    object_count = int(np.count_nonzero(depth_object_mask & domain))
    domain_count = int(np.count_nonzero(domain))

    print("")
    print("=" * 100)
    print("플랫폼 전용 기준면으로 PLY 재생성")
    print("=" * 100)
    print(f"입력 candidate: {candidate_path}")
    print(f"전체 유효 domain: {domain_count} px")
    print(f"Depth 실제 본넷: {object_count} px")
    print(f"플랫폼 plane fitting 후보: {platform_count} px")
    print(f"본넷 제외 팽창: {OBJECT_EXCLUDE_DILATE_PX} px")

    if platform_count < MIN_PLATFORM_POINTS:
        raise RuntimeError(
            "플랫폼 fitting 영역이 너무 작습니다. "
            "00~02 마스크를 확인해야 합니다."
        )

    # 플랫폼 기준면 fitting
    plane, plane_info = robust_platform_plane(
        candidate,
        platform_fit_mask,
    )

    corrected = candidate - plane
    corrected[~domain] = np.nan

    slope_before = plane_slope_of_data(
        candidate,
        platform_fit_mask,
    )
    slope_after = plane_slope_of_data(
        corrected,
        platform_fit_mask,
    )

    # 저장: 마스크 / plane / 보정 phase
    save_mask(
        OUTPUT_DIR / "00_전체유효domain.png",
        domain,
    )
    save_mask(
        OUTPUT_DIR / "01_Depth실제본넷마스크.png",
        depth_object_mask,
    )
    save_mask(
        OUTPUT_DIR / "02_플랫폼평면피팅마스크.png",
        platform_fit_mask,
    )

    save_phase_color(
        OUTPUT_DIR / "03_보정전_candidate.png",
        candidate,
        domain,
    )
    save_phase_color(
        OUTPUT_DIR / "04_플랫폼평면보정후.png",
        corrected,
        domain,
    )

    np.save(
        OUTPUT_DIR / "candidate_platform_plane_corrected.npy",
        corrected.astype(np.float32),
    )
    np.save(
        OUTPUT_DIR / "platform_fit_mask.npy",
        platform_fit_mask.astype(bool),
    )
    np.save(
        OUTPUT_DIR / "depth_object_mask.npy",
        depth_object_mask.astype(bool),
    )
    np.save(
        OUTPUT_DIR / "platform_plane.npy",
        plane.astype(np.float32),
    )

    # RAW PLY: smoothing/clip 없음
    raw_ply = OUTPUT_DIR / "01_플랫폼평면보정_RAW.ply"
    raw_points = make_ply(
        corrected,
        domain,
        raw_ply,
        skip=POINT_SKIP,
    )

    # 기존 Relative 방식과 비교용 PLY
    smooth, visual_info = existing_visualization_process(
        corrected,
        domain,
    )

    np.save(
        OUTPUT_DIR / "relative_surface_platform_plane_corrected.npy",
        smooth.astype(np.float32),
    )
    save_phase_color(
        OUTPUT_DIR / "05_플랫폼평면보정_기존시각화.png",
        smooth,
        domain & np.isfinite(smooth),
    )

    visual_ply = OUTPUT_DIR / "02_플랫폼평면보정_기존Relative방식.ply"
    visual_points = make_ply(
        smooth,
        domain & np.isfinite(smooth),
        visual_ply,
        skip=POINT_SKIP,
    )

    summary = {
        "input_candidate": str(candidate_path),
        "depth_object_mask_source": depth_mask_source,
        "domain_pixels": domain_count,
        "depth_object_pixels_in_domain": object_count,
        "platform_fit_candidate_pixels": platform_count,
        "object_exclude_dilate_px": OBJECT_EXCLUDE_DILATE_PX,
        "plane": plane_info,
        "platform_slope_before": slope_before,
        "platform_slope_after": slope_after,
        "ply": {
            "point_skip": POINT_SKIP,
            "relative_z_sign": RELATIVE_Z_SIGN,
            "relative_z_scale": RELATIVE_Z_SCALE,
            "raw_ply": str(raw_ply),
            "raw_points": raw_points,
            "existing_relative_ply": str(visual_ply),
            "existing_relative_points": visual_points,
            "existing_relative_process": visual_info,
        },
        "important": (
            "이 결과는 실제 mm 3D가 아니라 기존과 동일한 상대 위상 기반 PLY이다. "
            "플랫폼 평면은 본넷을 Depth 마스크로 제외한 플랫폼 픽셀만으로 추정했다. "
            "CloudCompare에서 플랫폼이 여전히 위치에 따라 체계적으로 기울면 "
            "단순 기준면 문제가 아니라 카메라-프로젝터 기하/캘리브레이션 문제를 의심한다."
        ),
    }

    summary_path = OUTPUT_DIR / "06_플랫폼기준면_진단요약.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("")
    print("[플랫폼 plane]")
    print(
        f"a(x)={plane_info['a_rad_per_pixel_x']:+.9f} rad/px | "
        f"b(y)={plane_info['b_rad_per_pixel_y']:+.9f} rad/px"
    )
    print(
        f"inlier={plane_info['fit_inliers']} / "
        f"{plane_info['fit_used_points']} "
        f"({plane_info['fit_inlier_percent']:.2f}%)"
    )

    if slope_before is not None and slope_after is not None:
        print("")
        print("[플랫폼 잔여 기울기 확인]")
        print(
            "보정 전 | "
            f"x={slope_before['a_rad_per_pixel_x']:+.9f}, "
            f"y={slope_before['b_rad_per_pixel_y']:+.9f} rad/px"
        )
        print(
            "보정 후 | "
            f"x={slope_after['a_rad_per_pixel_x']:+.9f}, "
            f"y={slope_after['b_rad_per_pixel_y']:+.9f} rad/px"
        )

    print("")
    print("=" * 100)
    print("완료")
    print("=" * 100)
    print("먼저 확인:")
    print(OUTPUT_DIR / "01_Depth실제본넷마스크.png")
    print(OUTPUT_DIR / "02_플랫폼평면피팅마스크.png")
    print(OUTPUT_DIR / "04_플랫폼평면보정후.png")
    print("")
    print("CloudCompare:")
    print(raw_ply)
    print(visual_ply)
    print("")
    print("원본 candidate / 기존 결과는 덮어쓰지 않음")
    print("새 촬영 없음")
    print("=" * 100)


if __name__ == "__main__":
    main()

'''



# =============================================================================
# 최종 자동 통합 추가부
# - 기존 P100 G/E 전처리 동작은 위 함수들을 그대로 사용.
# - 색 이름은 제어 판단에 사용하지 않음.
# - 초기 RGB0/RGB64 보정광응답 P90을 흰색 기준 100으로 환산해
#   "밝음 / 중간 / 어두움" 기본광 제어 전략만 결정.
# =============================================================================

import shutil
import traceback


흰색_기준_보정광응답_P90 = 59.001999
밝음_상대광응답_경계 = 96.34
어두움_상대광응답_경계 = 18.40

# 현재 실험에서 확보한 상대광응답 기준점.
# 색 이름은 코드 분기에 사용하지 않고 숫자 기준점만 사용한다.
중간_광응답_기준표 = [
    {
        "기준_상대광응답": 92.68,
        "대표_프로젝터밝기": 14,
        "후보_프로젝터밝기": [12, 14, 16],
    },
    {
        "기준_상대광응답": 22.04,
        "대표_프로젝터밝기": 26,
        "후보_프로젝터밝기": [24, 26, 28],
    },
    {
        "기준_상대광응답": 45.46,
        "대표_프로젝터밝기": 20,
        "후보_프로젝터밝기": [18, 20, 22],
    },
]

어두움_GE_후보 = [
    {"projector_percent": 100, "gain": 64, "exposure": 1400, "role": "이전"},
    {"projector_percent": 100, "gain": 96, "exposure": 1400, "role": "기준"},
    {"projector_percent": 100, "gain": 128, "exposure": 1400, "role": "다음"},
]

# 기존 4번째 단계와 동일한 자동 문제영역 기준.
기본광_근포화_기준 = 245
기본광_자동문제_최소후보수 = 3
기본광_자동문제_후보합의집중도_최소 = 0.95
기본광_자동문제_기준대합의_차이_deg = 15.0
기본광_자동문제_오프닝 = 3
기본광_자동문제_클로징 = 9
기본광_자동문제_최소컴포넌트_px = 150
기본광_자동문제_팽창_px = 2


def 상대광응답_기본광전략_결정(probe_stats):
    p90 = float(
        probe_stats[
            "보정광응답_P90"
        ]
    )

    relative = (
        p90
        / float(
            흰색_기준_보정광응답_P90
        )
        * 100.0
    )

    # 결과 JSON/기존 저장 구조에도 상대값이 남도록 추가한다.
    probe_stats[
        "상대광응답_흰색100"
    ] = float(
        relative
    )

    if relative >= float(
        밝음_상대광응답_경계
    ):
        result = {
            "분류": "밝음",
            "상대광응답_흰색100": float(relative),
            "보정광응답_P90": float(p90),
            "흰색기준_P90": float(흰색_기준_보정광응답_P90),
            "기본광_밝기제어": False,
            "후보조건": [],
            "설명": (
                "흰색 기준에 가까운 고반응. "
                "별도 기본광 밝기 제어 없이 P100의 기존 G/E 전처리 결과를 사용."
            ),
        }
        print("")
        print("=" * 78)
        print(
            f"상대광응답 {relative:.2f} → 밝음"
        )
        print(
            "기본광 밝기 제어 생략, P100 + 기존 G/E 전처리 결과 사용"
        )
        print("=" * 78)
        return result

    if relative < float(
        어두움_상대광응답_경계
    ):
        conditions = [
            dict(item)
            for item in 어두움_GE_후보
        ]

        result = {
            "분류": "어두움",
            "상대광응답_흰색100": float(relative),
            "보정광응답_P90": float(p90),
            "흰색기준_P90": float(흰색_기준_보정광응답_P90),
            "기본광_밝기제어": False,
            "후보조건": conditions,
            "대표조건": {
                "projector_percent": 100,
                "gain": 96,
                "exposure": 1400,
            },
            "설명": (
                "저반응. 프로젝터 P100 고정, "
                "G64/E1400 → G96/E1400 → G128/E1400 세 조건으로 기본광 영역 보완."
            ),
        }
        print("")
        print("=" * 78)
        print(
            f"상대광응답 {relative:.2f} → 어두움"
        )
        print(
            "P100 고정 | G64/E1400, G96/E1400, G128/E1400"
        )
        print("=" * 78)
        return result

    nearest = min(
        중간_광응답_기준표,
        key=lambda row: abs(
            float(relative)
            - float(
                row[
                    "기준_상대광응답"
                ]
            )
        ),
    )

    brightnesses = list(
        nearest[
            "후보_프로젝터밝기"
        ]
    )

    conditions = []
    for index, brightness in enumerate(
        brightnesses
    ):
        conditions.append(
            {
                "projector_percent": int(brightness),
                "gain": 16,
                "exposure": 156,
                "role": (
                    "이전"
                    if index == 0
                    else (
                        "기준"
                        if index == 1
                        else "다음"
                    )
                ),
            }
        )

    result = {
        "분류": "중간",
        "상대광응답_흰색100": float(relative),
        "보정광응답_P90": float(p90),
        "흰색기준_P90": float(흰색_기준_보정광응답_P90),
        "기본광_밝기제어": True,
        "가장가까운_기준상대광응답": float(
            nearest[
                "기준_상대광응답"
            ]
        ),
        "대표_프로젝터밝기": int(
            nearest[
                "대표_프로젝터밝기"
            ]
        ),
        "후보조건": conditions,
        "설명": (
            f"상대광응답 {relative:.2f}가 기준 "
            f"{nearest['기준_상대광응답']:.2f}에 가장 가까움. "
            f"P{brightnesses[0]}/P{brightnesses[1]}/P{brightnesses[2]} "
            "세 조건을 실제 촬영해 기본광 문제영역을 보완."
        ),
    }

    print("")
    print("=" * 78)
    print(
        f"상대광응답 {relative:.2f} → 중간"
    )
    print(
        f"가장 가까운 기준점 "
        f"{nearest['기준_상대광응답']:.2f}"
    )
    print(
        "기본광 후보: "
        + ", ".join(
            f"P{x}"
            for x in brightnesses
        )
    )
    print("=" * 78)

    return result


def 기본광후보_4위상촬영(
    cap,
    window_name,
    coord,
    monitor,
    args,
    folder,
    projector_percent,
    gain,
    exposure,
):
    """
    기존 흰색 4위상 촬영과 동일한 촬영 순서/버퍼 처리/180도 회전을 사용한다.
    차이는 사인파 base/amplitude에 projector_percent/100 배율을 곱하는 것뿐이다.
    """
    folder = Path(folder)
    folder.mkdir(
        parents=True,
        exist_ok=True,
    )

    actual_gain = 게인값_적용(
        cap,
        args.cam,
        int(gain),
        args.gain_settle,
    )
    actual_exposure = 노출값_적용(
        cap,
        args.cam,
        int(exposure),
        args.exposure_settle,
    )

    scale = float(
        projector_percent
    ) / 100.0

    effective_base = float(
        args.base
    ) * scale

    effective_amplitude = float(
        args.amplitude
    ) * scale

    print("")
    print("=" * 78)
    print(
        f"기본광 후보 촬영 | "
        f"P{int(projector_percent)} / "
        f"G{int(gain)} / E{int(exposure)}"
    )
    print(
        f"사인파 base={effective_base:.2f}, "
        f"amplitude={effective_amplitude:.2f}"
    )
    print("=" * 78)

    color_frames = {}
    gray_frames = {}

    for phase, phase_name in 위상_목록:
        gray_pattern = 사인파_밝기_생성(
            coord,
            args.period,
            effective_base,
            effective_amplitude,
            phase,
        )

        color_pattern = 색상패턴_생성(
            gray_pattern,
            "white",
        )

        cv2.imwrite(
            str(
                folder
                / f"투사패턴_{phase_name}.png"
            ),
            color_pattern,
        )

        color = 화면투사후_촬영_회전(
            cap,
            window_name,
            color_pattern,
            args,
        )

        gray = cv2.cvtColor(
            color,
            cv2.COLOR_BGR2GRAY,
        )

        color_frames[
            phase_name
        ] = color
        gray_frames[
            phase_name
        ] = gray

        cv2.imwrite(
            str(
                folder
                / f"phase_{phase_name}_color.png"
            ),
            color,
        )

        cv2.imwrite(
            str(
                folder
                / f"phase_{phase_name}.png"
            ),
            gray,
        )

        print(
            f"  {phase_name}도 촬영 완료"
        )

    return {
        "gain": int(gain),
        "exposure": int(exposure),
        "projector_percent": int(projector_percent),
        "actual_gain": actual_gain,
        "actual_exposure": actual_exposure,
        "컬러": color_frames,
        "회색": gray_frames,
    }


def 기본광후보_준비(
    cap,
    window_name,
    coord,
    monitor,
    args,
    run_dir,
    rect,
    preview,
    strategy,
    baseline_conditions,
    baseline_captures,
    baseline_qualities,
):
    """
    중간: P-2/P/P+2를 새 촬영.
    어두움: P100 + G64/G96/G128 E1400.
             P100 baseline에서 이미 같은 G/E를 촬영했다면 그 사진을 그대로 재사용하고
             없는 조건만 추가 촬영한다.
    밝음: 빈 목록.
    """
    if strategy[
        "분류"
    ] == "밝음":
        return []

    candidate_root = (
        Path(run_dir)
        / "03_기본광후보"
    )
    candidate_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    prepared = []

    baseline_map = {}
    for index, condition in enumerate(
        baseline_conditions
    ):
        key = (
            100,
            int(
                condition[
                    "gain"
                ]
            ),
            int(
                condition[
                    "exposure"
                ]
            ),
        )
        baseline_map[
            key
        ] = index

    for order, specification in enumerate(
        strategy[
            "후보조건"
        ]
    ):
        projector_percent = int(
            specification[
                "projector_percent"
            ]
        )
        gain = int(
            specification[
                "gain"
            ]
        )
        exposure = int(
            specification[
                "exposure"
            ]
        )

        key = (
            projector_percent,
            gain,
            exposure,
        )

        # 어두움 P100 G/E 후보는 기존 P100 탐색에서 이미 촬영했다면 재사용.
        if key in baseline_map:
            index = baseline_map[
                key
            ]
            capture = baseline_captures[
                index
            ]
            quality = baseline_qualities[
                index
            ]
            folder = (
                Path(run_dir)
                / f"G{gain}_E{exposure}"
            )
            source = "P100 기존 G/E 촬영 재사용"
        else:
            folder = (
                candidate_root
                / (
                    f"P{projector_percent:03d}_"
                    f"G{gain}_E{exposure}"
                )
            )

            capture = 기본광후보_4위상촬영(
                cap,
                window_name,
                coord,
                monitor,
                args,
                folder,
                projector_percent,
                gain,
                exposure,
            )

            quality = 조건품질_계산(
                capture,
                rect,
                args,
            )

            조건품질영상_저장(
                folder,
                quality,
                rect,
                preview,
            )

            source = "신규 촬영"

        item = {
            "order": int(order),
            "role": str(
                specification.get(
                    "role",
                    order,
                )
            ),
            "projector_percent": int(
                projector_percent
            ),
            "gain": int(gain),
            "exposure": int(exposure),
            "folder": Path(folder),
            "capture": capture,
            "quality": quality,
            "source": source,
        }

        prepared.append(
            item
        )

        print(
            f"기본광 후보 준비 | "
            f"P{projector_percent} G{gain}/E{exposure} | "
            f"{source} | "
            f"유효 {quality['유효_위상_비율']:.2f}%"
        )

    return prepared


def 후처리_4위상품질(
    capture,
    args,
):
    gray = capture[
        "회색"
    ]
    color = capture[
        "컬러"
    ]

    i0 = gray["000"].astype(
        np.float32
    )
    i90 = gray["090"].astype(
        np.float32
    )
    i180 = gray["180"].astype(
        np.float32
    )
    i270 = gray["270"].astype(
        np.float32
    )

    modulation = 0.5 * np.sqrt(
        (i0 - i180) ** 2
        + (i270 - i90) ** 2
    )

    gray_max = np.maximum.reduce(
        [
            i0,
            i90,
            i180,
            i270,
        ]
    )

    color_phase_max = []

    for name in [
        "000",
        "090",
        "180",
        "270",
    ]:
        c = color[
            name
        ].astype(
            np.float32
        )
        color_phase_max.append(
            np.max(
                c,
                axis=2,
            )
        )

    color_max = np.maximum.reduce(
        color_phase_max
    )

    saturation = (
        color_max
        >= float(
            args.saturation_threshold
        )
    )

    near_saturation = (
        color_max
        >= float(
            기본광_근포화_기준
        )
    )

    dark = (
        gray_max
        <= float(
            args.dark_threshold
        )
    )

    low_modulation = (
        modulation
        < float(
            args.modulation_threshold
        )
    )

    valid = (
        (~saturation)
        & (~dark)
        & (~low_modulation)
    )

    safe_valid = (
        (~near_saturation)
        & (~dark)
        & (~low_modulation)
    )

    wrapped = np.arctan2(
        i270 - i90,
        i0 - i180,
    ).astype(
        np.float32
    )

    return {
        "modulation": modulation,
        "color_max": color_max,
        "saturation": saturation,
        "near_saturation": near_saturation,
        "dark": dark,
        "low_modulation": low_modulation,
        "valid": valid,
        "safe_valid": safe_valid,
        "wrapped": wrapped,
    }


def 기본광_작은컴포넌트_제거(
    mask,
    min_area,
):
    mask_u8 = (
        np.asarray(
            mask,
            dtype=np.uint8,
        )
        * 255
    )

    count, labels, stats, _ = (
        cv2.connectedComponentsWithStats(
            mask_u8,
            connectivity=8,
        )
    )

    output = np.zeros(
        mask_u8.shape,
        dtype=bool,
    )

    for label in range(
        1,
        count,
    ):
        area = int(
            stats[
                label,
                cv2.CC_STAT_AREA,
            ]
        )

        if area >= int(
            min_area
        ):
            output[
                labels == label
            ] = True

    return output


def 기본광_자동문제영역_오버레이_저장(
    path,
    base_image_path,
    allowed_mask,
    problem_mask,
):
    image = cv2.imread(
        str(
            base_image_path
        ),
        cv2.IMREAD_COLOR,
    )

    if image is None:
        image = np.zeros(
            (
                problem_mask.shape[0],
                problem_mask.shape[1],
                3,
            ),
            dtype=np.uint8,
        )

    result = image.copy()

    # 허용영역은 약하게 초록, 문제영역은 빨강.
    allowed_overlay = result.copy()
    allowed_overlay[
        np.asarray(
            allowed_mask,
            dtype=bool,
        )
    ] = (
        0,
        160,
        0,
    )

    result = cv2.addWeighted(
        result,
        0.85,
        allowed_overlay,
        0.15,
        0,
    )

    problem_overlay = result.copy()
    problem_overlay[
        np.asarray(
            problem_mask,
            dtype=bool,
        )
    ] = (
        0,
        0,
        255,
    )

    result = cv2.addWeighted(
        result,
        0.60,
        problem_overlay,
        0.40,
        0,
    )

    contours, _ = cv2.findContours(
        np.asarray(
            problem_mask,
            dtype=np.uint8,
        ),
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    cv2.drawContours(
        result,
        contours,
        -1,
        (
            255,
            255,
            255,
        ),
        2,
    )

    cv2.putText(
        result,
        "AUTO BASIC-LIGHT PROBLEM AREA",
        (
            30,
            40,
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (
            255,
            255,
            255,
        ),
        2,
        cv2.LINE_AA,
    )

    cv2.imwrite(
        str(path),
        result,
    )


def 기본광_문제영역_자동검출(
    output_dir,
    base_image_path,
    allowed_mask,
    reference_phase,
    reference_valid,
    baseline_diff,
    baseline_common_valid,
    baseline_capture,
    candidate_items,
    args,
):
    """
    기존 4번째 단계의 핵심 자동 검출 원리를 그대로 사용한다.
    차이점은 P2~P30 전체 폴더를 훑는 대신,
    초기 상대광응답으로 이미 정한 3개 후보조건만 입력받는다는 점이다.
    """
    output_dir = Path(
        output_dir
    )
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    save_npy = (
        output_dir
        / "문제영역_마스크.npy"
    )
    save_png = (
        output_dir
        / "문제영역_마스크.png"
    )
    save_overlay = (
        output_dir
        / "01_자동문제영역_색상오버레이.png"
    )
    save_diag = (
        output_dir
        / "02_자동문제영역_진단.json"
    )

    allowed = (
        np.asarray(
            allowed_mask,
            dtype=bool,
        )
        & np.asarray(
            baseline_common_valid,
            dtype=bool,
        )
        & np.asarray(
            reference_valid,
            dtype=bool,
        )
        & np.isfinite(
            baseline_diff
        )
        & np.isfinite(
            reference_phase
        )
    )

    if np.count_nonzero(
        allowed
    ) < 100:
        raise RuntimeError(
            "자동 문제영역 검출용 유효 픽셀이 너무 적습니다."
        )

    base_quality = 후처리_4위상품질(
        baseline_capture,
        args,
    )

    baseline_quality_risk = (
        allowed
        & (
            base_quality[
                "near_saturation"
            ]
            | base_quality[
                "dark"
            ]
            | base_quality[
                "low_modulation"
            ]
        )
    )

    sin_sum = np.zeros(
        baseline_diff.shape,
        dtype=np.float64,
    )
    cos_sum = np.zeros(
        baseline_diff.shape,
        dtype=np.float64,
    )
    valid_count = np.zeros(
        baseline_diff.shape,
        dtype=np.int16,
    )

    condition_logs = []

    for item in candidate_items:
        q = 후처리_4위상품질(
            item[
                "capture"
            ],
            args,
        )

        candidate_valid = (
            q[
                "safe_valid"
            ]
            & np.asarray(
                reference_valid,
                dtype=bool,
            )
            & np.asarray(
                allowed_mask,
                dtype=bool,
            )
        )

        candidate_diff = 통합_wrap_to_pi(
            q[
                "wrapped"
            ].astype(
                np.float64
            )
            - reference_phase
        )

        align_mask = (
            candidate_valid
            & allowed
            & np.isfinite(
                candidate_diff
            )
        )

        global_shift = 0.0

        align_pixels = int(
            np.count_nonzero(
                align_mask
            )
        )

        if align_pixels >= 100:
            residual = 통합_wrap_to_pi(
                candidate_diff[
                    align_mask
                ]
                - baseline_diff[
                    align_mask
                ]
            )

            global_shift = float(
                np.arctan2(
                    np.mean(
                        np.sin(
                            residual
                        )
                    ),
                    np.mean(
                        np.cos(
                            residual
                        )
                    ),
                )
            )

        aligned_diff = 통합_wrap_to_pi(
            candidate_diff
            - global_shift
        )

        usable = (
            candidate_valid
            & np.isfinite(
                aligned_diff
            )
            & np.asarray(
                allowed_mask,
                dtype=bool,
            )
        )

        sin_sum[
            usable
        ] += np.sin(
            aligned_diff[
                usable
            ]
        )

        cos_sum[
            usable
        ] += np.cos(
            aligned_diff[
                usable
            ]
        )

        valid_count[
            usable
        ] += 1

        condition_logs.append(
            {
                "projector_percent": int(
                    item[
                        "projector_percent"
                    ]
                ),
                "gain": int(
                    item[
                        "gain"
                    ]
                ),
                "exposure": int(
                    item[
                        "exposure"
                    ]
                ),
                "role": str(
                    item[
                        "role"
                    ]
                ),
                "folder": str(
                    item[
                        "folder"
                    ]
                ),
                "align_pixels": int(
                    align_pixels
                ),
                "global_phase_shift_deg": float(
                    np.degrees(
                        global_shift
                    )
                ),
                "safe_valid_pixels": int(
                    np.count_nonzero(
                        usable
                    )
                ),
            }
        )

    enough = (
        valid_count
        >= int(
            기본광_자동문제_최소후보수
        )
    )

    consensus = np.full(
        baseline_diff.shape,
        np.nan,
        dtype=np.float64,
    )

    concentration = np.zeros(
        baseline_diff.shape,
        dtype=np.float64,
    )

    consensus[
        enough
    ] = np.arctan2(
        sin_sum[
            enough
        ],
        cos_sum[
            enough
        ],
    )

    concentration[
        enough
    ] = (
        np.sqrt(
            sin_sum[
                enough
            ] ** 2
            + cos_sum[
                enough
            ] ** 2
        )
        / np.maximum(
            valid_count[
                enough
            ].astype(
                np.float64
            ),
            1.0,
        )
    )

    baseline_vs_consensus_deg = np.full(
        baseline_diff.shape,
        np.nan,
        dtype=np.float64,
    )

    compare_mask = (
        allowed
        & enough
        & np.isfinite(
            consensus
        )
    )

    baseline_vs_consensus_deg[
        compare_mask
    ] = np.abs(
        np.degrees(
            통합_wrap_to_pi(
                baseline_diff[
                    compare_mask
                ]
                - consensus[
                    compare_mask
                ]
            )
        )
    )

    candidate_agrees = (
        compare_mask
        & (
            concentration
            >= float(
                기본광_자동문제_후보합의집중도_최소
            )
        )
    )

    phase_disagreement = (
        candidate_agrees
        & (
            baseline_vs_consensus_deg
            >= float(
                기본광_자동문제_기준대합의_차이_deg
            )
        )
    )

    photometric_recoverable = (
        baseline_quality_risk
        & candidate_agrees
    )

    raw_problem = (
        phase_disagreement
        | photometric_recoverable
    )

    raw_problem &= np.asarray(
        allowed_mask,
        dtype=bool,
    )

    open_kernel = (
        cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (
                기본광_자동문제_오프닝,
                기본광_자동문제_오프닝,
            ),
        )
    )

    close_kernel = (
        cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (
                기본광_자동문제_클로징,
                기본광_자동문제_클로징,
            ),
        )
    )

    cleaned = cv2.morphologyEx(
        raw_problem.astype(
            np.uint8
        )
        * 255,
        cv2.MORPH_OPEN,
        open_kernel,
    )

    cleaned = cv2.morphologyEx(
        cleaned,
        cv2.MORPH_CLOSE,
        close_kernel,
    )

    problem_mask = (
        cleaned
        > 0
    )

    problem_mask = (
        기본광_작은컴포넌트_제거(
            problem_mask,
            기본광_자동문제_최소컴포넌트_px,
        )
    )

    if (
        기본광_자동문제_팽창_px
        > 0
        and np.any(
            problem_mask
        )
    ):
        size = (
            int(
                기본광_자동문제_팽창_px
            )
            * 2
            + 1
        )

        dilate_kernel = (
            cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE,
                (
                    size,
                    size,
                ),
            )
        )

        problem_mask = (
            cv2.dilate(
                problem_mask.astype(
                    np.uint8
                ),
                dilate_kernel,
                iterations=1,
            )
            > 0
        )

    problem_mask &= np.asarray(
        allowed_mask,
        dtype=bool,
    )

    np.save(
        save_npy,
        problem_mask.astype(
            bool
        ),
    )

    cv2.imwrite(
        str(
            save_png
        ),
        problem_mask.astype(
            np.uint8
        )
        * 255,
    )

    기본광_자동문제영역_오버레이_저장(
        save_overlay,
        base_image_path,
        allowed_mask,
        problem_mask,
    )

    valid_delta = (
        baseline_vs_consensus_deg[
            np.isfinite(
                baseline_vs_consensus_deg
            )
        ]
    )

    diagnostics = {
        "method": (
            "cross-condition circular consensus + baseline photometric risk; "
            "2pi branch is not used as an automatic problem criterion"
        ),
        "thresholds": {
            "min_candidate_count": int(
                기본광_자동문제_최소후보수
            ),
            "candidate_concentration_min": float(
                기본광_자동문제_후보합의집중도_최소
            ),
            "baseline_vs_consensus_deg_min": float(
                기본광_자동문제_기준대합의_차이_deg
            ),
            "min_component_px": int(
                기본광_자동문제_최소컴포넌트_px
            ),
            "dilate_px": int(
                기본광_자동문제_팽창_px
            ),
        },
        "pixels": {
            "allowed": int(
                np.count_nonzero(
                    allowed
                )
            ),
            "enough_candidate": int(
                np.count_nonzero(
                    enough
                    & allowed
                )
            ),
            "candidate_agrees": int(
                np.count_nonzero(
                    candidate_agrees
                )
            ),
            "baseline_quality_risk": int(
                np.count_nonzero(
                    baseline_quality_risk
                )
            ),
            "phase_disagreement_raw": int(
                np.count_nonzero(
                    phase_disagreement
                )
            ),
            "photometric_recoverable_raw": int(
                np.count_nonzero(
                    photometric_recoverable
                )
            ),
            "raw_problem": int(
                np.count_nonzero(
                    raw_problem
                )
            ),
            "final_problem": int(
                np.count_nonzero(
                    problem_mask
                )
            ),
        },
        "baseline_vs_consensus_deg": {
            "median": (
                float(
                    np.median(
                        valid_delta
                    )
                )
                if valid_delta.size
                else None
            ),
            "p90": (
                float(
                    np.percentile(
                        valid_delta,
                        90,
                    )
                )
                if valid_delta.size
                else None
            ),
            "p99": (
                float(
                    np.percentile(
                        valid_delta,
                        99,
                    )
                )
                if valid_delta.size
                else None
            ),
        },
        "conditions": condition_logs,
        "overlay": str(
            save_overlay
        ),
    }

    save_diag.write_text(
        json.dumps(
            diagnostics,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print("")
    print("=" * 90)
    print(
        "기본광 문제영역 자동 검출 완료"
    )
    print(
        "기준: 3개 후보조건 circular phase 합의 + 기존 P100 HDR 품질 위험"
    )
    print(
        "2π branch는 여기서 판단하지 않고 뒤 Quality-guided 단계가 담당"
    )
    print(
        f"자동 문제영역: "
        f"{np.count_nonzero(problem_mask)} px"
    )
    print("=" * 90)

    return problem_mask


def 기본광_원래HDR방식_문제영역교체(
    output_dir,
    baseline_capture,
    baseline_valid,
    problem_mask,
    candidate_items,
    reference_phase,
    reference_valid,
    object_area,
    args,
):
    """
    기존 5번째 단계의 최종 선택 원리:
    - 문제영역 밖 기존 HDR 보존
    - 문제영역 안에서 후보를 지정 순서대로 검사
    - 처음 valid인 후보를 채택
    - 같은 픽셀 네 위상은 한 조건에서 통째로 가져옴
    - 모든 후보가 무효이면 기존 HDR 데이터/기존 valid를 그대로 유지
    """
    output_dir = Path(
        output_dir
    )
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    candidate_qualities = [
        후처리_4위상품질(
            item[
                "capture"
            ],
            args,
        )
        for item in candidate_items
    ]

    shape = np.asarray(
        baseline_valid,
        dtype=bool,
    ).shape

    selected_index = np.full(
        shape,
        -1,
        dtype=np.int16,
    )

    final_valid = np.asarray(
        baseline_valid,
        dtype=bool,
    ).copy()

    unresolved = np.asarray(
        problem_mask,
        dtype=bool,
    ).copy()

    for index, quality in enumerate(
        candidate_qualities
    ):
        take = (
            unresolved
            & quality[
                "valid"
            ]
        )

        selected_index[
            take
        ] = int(
            index
        )

        final_valid[
            take
        ] = True

        unresolved[
            take
        ] = False

    fused_gray = {
        name: baseline_capture[
            "회색"
        ][
            name
        ].copy()
        for name in [
            "000",
            "090",
            "180",
            "270",
        ]
    }

    fused_color = {
        name: baseline_capture[
            "컬러"
        ][
            name
        ].copy()
        for name in [
            "000",
            "090",
            "180",
            "270",
        ]
    }

    replaced = (
        np.asarray(
            problem_mask,
            dtype=bool,
        )
        & (
            selected_index
            >= 0
        )
    )

    rr, cc = np.indices(
        shape
    )

    for name in [
        "000",
        "090",
        "180",
        "270",
    ]:
        gray_stack = np.stack(
            [
                item[
                    "capture"
                ][
                    "회색"
                ][
                    name
                ]
                for item in candidate_items
            ],
            axis=0,
        )

        color_stack = np.stack(
            [
                item[
                    "capture"
                ][
                    "컬러"
                ][
                    name
                ]
                for item in candidate_items
            ],
            axis=0,
        )

        if np.any(
            replaced
        ):
            index = selected_index[
                replaced
            ]

            y = rr[
                replaced
            ]
            x = cc[
                replaced
            ]

            fused_gray[
                name
            ][
                replaced
            ] = gray_stack[
                index,
                y,
                x,
            ]

            fused_color[
                name
            ][
                replaced
            ] = color_stack[
                index,
                y,
                x,
                :,
            ]

    fused_capture = {
        "회색": fused_gray,
        "컬러": fused_color,
    }

    fused_quality = 후처리_4위상품질(
        fused_capture,
        args,
    )

    fused_object_phase = (
        fused_quality[
            "wrapped"
        ].astype(
            np.float64
        )
    )

    fused_diff = 통합_wrap_to_pi(
        fused_object_phase
        - reference_phase
    )

    # 기존 common-valid를 기본으로 유지하고,
    # 실제 교체한 픽셀만 새 후보 valid + reference valid로 갱신.
    common_valid = (
        np.asarray(
            object_area,
            dtype=bool,
        )
        & np.asarray(
            final_valid,
            dtype=bool,
        )
        & np.asarray(
            reference_valid,
            dtype=bool,
        )
        & np.isfinite(
            fused_diff
        )
    )

    fused_diff[
        ~common_valid
    ] = np.nan

    for name in [
        "000",
        "090",
        "180",
        "270",
    ]:
        cv2.imwrite(
            str(
                output_dir
                / f"phase_{name}.png"
            ),
            fused_gray[
                name
            ],
        )

        cv2.imwrite(
            str(
                output_dir
                / f"phase_{name}_color.png"
            ),
            fused_color[
                name
            ],
        )

    np.save(
        output_dir
        / "phase_difference_masked.npy",
        fused_diff.astype(
            np.float32
        ),
    )

    np.save(
        output_dir
        / "common_valid_mask.npy",
        common_valid.astype(
            bool
        ),
    )

    np.save(
        output_dir
        / "선택조건번호.npy",
        selected_index,
    )

    cv2.imwrite(
        str(
            output_dir
            / "문제영역_마스크.png"
        ),
        np.asarray(
            problem_mask,
            dtype=np.uint8,
        )
        * 255,
    )

    # 선택조건 지도: 0,1,2를 서로 다른 명암으로 저장.
    selection_vis = np.zeros(
        shape,
        dtype=np.uint8,
    )

    for index in range(
        len(
            candidate_items
        )
    ):
        take = (
            np.asarray(
                problem_mask,
                dtype=bool,
            )
            & (
                selected_index
                == index
            )
        )

        selection_vis[
            take
        ] = int(
            round(
                (
                    index
                    + 1
                )
                / max(
                    1,
                    len(
                        candidate_items
                    )
                )
                * 255.0
            )
        )

    cv2.imwrite(
        str(
            output_dir
            / "선택조건_지도.png"
        ),
        cv2.applyColorMap(
            selection_vis,
            cv2.COLORMAP_TURBO,
        ),
    )

    selection_rows = []

    for index, item in enumerate(
        candidate_items
    ):
        count = int(
            np.count_nonzero(
                np.asarray(
                    problem_mask,
                    dtype=bool,
                )
                & (
                    selected_index
                    == index
                )
            )
        )

        selection_rows.append(
            {
                "순서": int(index),
                "role": str(
                    item[
                        "role"
                    ]
                ),
                "projector_percent": int(
                    item[
                        "projector_percent"
                    ]
                ),
                "gain": int(
                    item[
                        "gain"
                    ]
                ),
                "exposure": int(
                    item[
                        "exposure"
                    ]
                ),
                "pixels": int(
                    count
                ),
                "folder": str(
                    item[
                        "folder"
                    ]
                ),
            }
        )

    summary = {
        "problem_pixels": int(
            np.count_nonzero(
                problem_mask
            )
        ),
        "replaced_pixels": int(
            np.count_nonzero(
                replaced
            )
        ),
        "unresolved_pixels_keep_baseline": int(
            np.count_nonzero(
                unresolved
            )
        ),
        "selection": selection_rows,
        "rules": [
            "문제영역 밖 기존 HDR 보존",
            "문제영역 안 후보조건 지정 순서 검사",
            "M/포화/암부 valid 판정은 기존 기준",
            "처음 유효한 후보 선택",
            "같은 픽셀 000/090/180/270 동일 촬영조건 사용",
            "모든 후보 무효면 기존 HDR 데이터와 기존 valid 유지",
        ],
    }

    (
        output_dir
        / "00_결과요약.json"
    ).write_text(
        json.dumps(
            summary,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print("")
    print("=" * 90)
    print(
        "기본광 문제영역 교체 완료"
    )
    print(
        f"문제영역: "
        f"{summary['problem_pixels']} px"
    )
    print(
        f"교체: "
        f"{summary['replaced_pixels']} px"
    )
    print(
        f"후보 모두 무효 → 기존 HDR 유지: "
        f"{summary['unresolved_pixels_keep_baseline']} px"
    )
    print("=" * 90)

    return {
        "output_dir": output_dir,
        "phase_difference": fused_diff,
        "common_valid": common_valid,
        "selected_index": selected_index,
        "summary": summary,
    }


def P100_Object핵심결과_저장(
    baseline_result_dir,
    baseline_fusion,
    baseline_fusion_quality,
):
    struct_dir = (
        Path(
            baseline_result_dir
        )
        / "구조광_형상복원"
    )

    struct_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    object_phase = np.asarray(
        baseline_fusion_quality[
            "래핑_위상맵"
        ],
        dtype=np.float32,
    )

    object_area = np.asarray(
        baseline_fusion_quality[
            "물체영역_마스크"
        ],
        dtype=bool,
    )

    object_valid = (
        np.asarray(
            baseline_fusion[
                "최종유효_마스크"
            ],
            dtype=bool,
        )
        & object_area
    )

    np.save(
        struct_dir
        / "object_fused_wrapped_phase.npy",
        object_phase,
    )

    np.save(
        struct_dir
        / "object_area_mask.npy",
        object_area,
    )

    np.save(
        struct_dir
        / "object_final_valid_mask.npy",
        object_valid,
    )

    return (
        struct_dir,
        object_phase,
        object_area,
        object_valid,
    )


def Reference180_기준생성(
    struct_dir,
    post_root,
    baseline_result_dir,
    object_phase,
    object_area,
    object_valid,
    args,
):
    output_dir = (
        Path(post_root)
        / "04_Reference_180도"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # 확정된 방향을 자동 적용.
    args.reference_rotate_180 = True

    reference = 통합_Reference_4위상_읽기(
        args.reference_dir,
        args,
    )

    reference_phase = np.asarray(
        reference[
            "래핑_위상맵"
        ],
        dtype=np.float32,
    )

    reference_valid = np.asarray(
        reference[
            "유효_마스크"
        ],
        dtype=bool,
    )

    if (
        reference_phase.shape
        != object_phase.shape
    ):
        raise RuntimeError(
            "Reference와 Object 위상맵 크기가 다릅니다: "
            f"Reference={reference_phase.shape}, "
            f"Object={object_phase.shape}"
        )

    common_valid = (
        np.asarray(
            object_area,
            dtype=bool,
        )
        & np.asarray(
            object_valid,
            dtype=bool,
        )
        & reference_valid
        & np.isfinite(
            reference_phase
        )
        & np.isfinite(
            object_phase
        )
    )

    phase_diff = 통합_wrap_to_pi(
        object_phase
        - reference_phase
    )

    phase_diff_masked = (
        phase_diff.copy()
    )

    phase_diff_masked[
        ~common_valid
    ] = np.nan

    np.save(
        output_dir
        / "reference_wrapped_phase.npy",
        reference_phase,
    )

    np.save(
        output_dir
        / "phase_difference.npy",
        phase_diff,
    )

    np.save(
        output_dir
        / "phase_difference_masked.npy",
        phase_diff_masked,
    )

    np.save(
        output_dir
        / "common_valid_mask.npy",
        common_valid,
    )

    np.save(
        struct_dir
        / "reference_valid_mask.npy",
        reference_valid,
    )

    통합_마스크PNG_저장(
        output_dir
        / "04_common_valid_mask.png",
        common_valid,
    )

    통합_컬러맵_저장(
        output_dir
        / "05_reference_wrapped_phase.png",
        reference_phase,
        object_area,
    )

    통합_컬러맵_저장(
        output_dir
        / "06_object_fused_wrapped_phase.png",
        object_phase,
        object_valid,
    )

    통합_컬러맵_저장(
        output_dir
        / "07_phase_difference.png",
        phase_diff,
        object_area,
    )

    통합_컬러맵_저장(
        output_dir
        / "08_phase_difference_masked.png",
        phase_diff_masked,
        common_valid,
    )

    # 기존 문제영역 자동검출 오버레이가 사용할 동일 계열 relative 이미지 생성.
    통합_Relative_PLY_생성(
        phase_diff_masked,
        common_valid,
        output_dir,
        args,
    )

    return {
        "output_dir": output_dir,
        "reference_phase": reference_phase,
        "reference_valid": reference_valid,
        "common_valid": common_valid,
        "phase_difference_masked": phase_diff_masked,
        "relative_final_mask": np.load(
            output_dir
            / "relative_final_mask.npy"
        ).astype(
            bool
        ),
        "relative_surface_color": (
            output_dir
            / "relative_surface_color.png"
        ),
    }


def 깨끗한HDR_기준그대로저장(
    output_dir,
    baseline_result_dir,
    reference_info,
):
    """
    밝음 분기 또는 자동 문제영역이 0픽셀인 경우:
    별도 기본광 교체 없이 현재 P100 HDR + Reference180 결과를 clean 입력으로 사용.
    """
    output_dir = Path(
        output_dir
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    for name in [
        "000",
        "090",
        "180",
        "270",
    ]:
        shutil.copy2(
            Path(
                baseline_result_dir
            )
            / f"phase_{name}.png",
            output_dir
            / f"phase_{name}.png",
        )

        shutil.copy2(
            Path(
                baseline_result_dir
            )
            / f"phase_{name}_color.png",
            output_dir
            / f"phase_{name}_color.png",
        )

    shutil.copy2(
        reference_info[
            "output_dir"
        ]
        / "phase_difference_masked.npy",
        output_dir
        / "phase_difference_masked.npy",
    )

    shutil.copy2(
        reference_info[
            "output_dir"
        ]
        / "common_valid_mask.npy",
        output_dir
        / "common_valid_mask.npy",
    )

    empty = np.zeros(
        reference_info[
            "common_valid"
        ].shape,
        dtype=bool,
    )

    np.save(
        output_dir
        / "문제영역_마스크.npy",
        empty,
    )

    cv2.imwrite(
        str(
            output_dir
            / "문제영역_마스크.png"
        ),
        empty.astype(
            np.uint8
        )
        * 255,
    )

    (
        output_dir
        / "00_결과요약.json"
    ).write_text(
        json.dumps(
            {
                "basic_light_correction": False,
                "reason": (
                    "밝음 분기 또는 자동 문제영역 없음. "
                    "P100 기존 G/E HDR 결과를 그대로 보존."
                ),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    return output_dir


def _원본소스_경로섹션_교체(
    source,
    start_heading,
    next_heading,
    replacement_body,
):
    start_marker = (
        "# =============================================================================\n"
        f"# 0. {start_heading}\n"
        "# =============================================================================\n"
    )

    next_marker = (
        "# =============================================================================\n"
        f"# 1. {next_heading}\n"
        "# =============================================================================\n"
    )

    start = source.find(
        start_marker
    )

    if start < 0:
        raise RuntimeError(
            f"원본 소스 경로 시작 구간을 찾지 못했습니다: {start_heading}"
        )

    end = source.find(
        next_marker,
        start,
    )

    if end < 0:
        raise RuntimeError(
            f"원본 소스 다음 구간을 찾지 못했습니다: {next_heading}"
        )

    return (
        source[
            :start
        ]
        + start_marker
        + replacement_body.rstrip()
        + "\n\n\n"
        + source[
            end:
        ]
    )


def 원본_QualityGuided_실행(
    struct_dir,
    clean_dir,
    preview_folder,
    output_dir,
):
    replacement = f"""
STRUCT_DIR = Path({str(Path(struct_dir))!r})
통합_출력_루트 = Path({str(Path(output_dir).parent)!r})
CLEAN_HDR_DIR = Path({str(Path(clean_dir))!r})
PHASE_PATH = CLEAN_HDR_DIR / "phase_difference_masked.npy"
VALID_PATH = CLEAN_HDR_DIR / "common_valid_mask.npy"
OBJECT_MASK_CANDIDATES = [
    STRUCT_DIR / "object_area_mask.npy",
]
BRIGHTNESS_ROOT = Path({str(Path(preview_folder).parent)!r})
P20_DIR = Path({str(Path(preview_folder))!r})
OUTPUT_DIR = Path({str(Path(output_dir))!r})
PHASE_NAMES = ["000", "090", "180", "270"]
"""

    source = _원본소스_경로섹션_교체(
        원본_STAGE6_SOURCE,
        "경로",
        "진단 기준",
        replacement,
    )

    namespace = {
        "__name__": "__main__",
        "__file__": "<최종통합:QualityGuided>",
        "__package__": None,
    }

    code = compile(
        source,
        namespace[
            "__file__"
        ],
        "exec",
    )

    exec(
        code,
        namespace,
        namespace,
    )

    return Path(
        output_dir
    )


def 원본_플랫폼기준면_실행(
    struct_dir,
    clean_dir,
    quality_dir,
    depth_result_dir,
    output_dir,
):
    depth_mask_path = (
        Path(
            depth_result_dir
        )
        / "03_Depth_물체픽셀_확인용.png"
    )

    current_depth_path = (
        Path(
            depth_result_dir
        )
        / "현재_물체_depth.npy"
    )

    replacement = f"""
STRUCT_DIR = Path({str(Path(struct_dir))!r})
통합_출력_루트 = Path({str(Path(output_dir).parent)!r})
CLEAN_HDR_DIR = Path({str(Path(clean_dir))!r})
PHASE_CANDIDATES = [
    Path({str(Path(quality_dir) / "candidate_unwrapped_phase.npy")!r}),
]
COMMON_VALID_PATH = CLEAN_HDR_DIR / "common_valid_mask.npy"
ANALYSIS_AREA_PATH = STRUCT_DIR / "object_area_mask.npy"
DEPTH_RESULT_DIR = Path({str(Path(depth_result_dir))!r})
DEPTH_OBJECT_MASK_PNG = Path({str(depth_mask_path)!r})
CURRENT_DEPTH_PATH = Path({str(current_depth_path)!r})
BACKGROUND_DEPTH_PATH = Path(
    "/home/seoyeong/졸업작품/전처리와구조광_통합/플랫폼 바닥 따기/플랫폼_바닥_depth.npy"
)
OUTPUT_DIR = Path({str(Path(output_dir))!r})
"""

    source = _원본소스_경로섹션_교체(
        원본_PLATFORM_STAGE_SOURCE,
        "현재 남색 실험 경로",
        "파라미터",
        replacement,
    )

    namespace = {
        "__name__": "__main__",
        "__file__": "<최종통합:플랫폼기준면>",
        "__package__": None,
    }

    code = compile(
        source,
        namespace[
            "__file__"
        ],
        "exec",
    )

    exec(
        code,
        namespace,
        namespace,
    )

    return Path(
        output_dir
    )



# =============================================================================
# 현재 프레임 플랫폼 기준 복원
# =============================================================================
#
# 왜 이 경로를 쓰는가
# -------------------
# 실제 동일코드 반복 실험에서:
#   - 물체 상태 반복: 대체로 재현
#   - 빈 플랫폼 상태 반복: 매우 안정
#   - 물체 상태 ↔ 빈 플랫폼 Reference: 큰 공간 위상 차이
# 가 확인되었다.
#
# 따라서 최종 형상복원에서 "다른 시점의 빈 플랫폼 Reference phase"를 빼지 않는다.
# 현재 물체 촬영 프레임 안에 동시에 보이는 플랫폼 픽셀을 기준면으로 사용한다.
#
# 앞단 G/E 탐색 + 픽셀단위 4위상 융합은 그대로 유지한다.
# 외부 Reference에 의존하던 Reference 차분 / 기본광 문제영역 HDR / 전체-domain QG는
# 최종 형상 계산에서 사용하지 않는다.
# =============================================================================

현재프레임_2PI = 2.0 * np.pi
현재프레임_물체제외_팽창PX = 15
현재프레임_최소플랫폼픽셀 = 2000
현재프레임_최소물체컴포넌트 = 50
현재프레임_플랫폼_평면차수 = 2
현재프레임_평면반복 = 7
현재프레임_평면MAD배수 = 3.5
현재프레임_평면최소문턱_RAD = 0.05
현재프레임_스파이크문턱_RAD = 0.80
현재프레임_PLY_SKIP = 2
현재프레임_Z_SIGN = -1.0
현재프레임_Z_SCALE = 40.0

# Depth 내부 hole 보강.
# 플랫폼 residual 분포에 따라 자동 threshold를 만들고,
# Depth object 외곽 내부의 hole만 검사한다.
현재프레임_홀보강_최소면적PX = 20
현재프레임_홀보강_플랫폼P95배수 = 1.35
현재프레임_홀보강_최소위상문턱RAD = 0.65
현재프레임_홀보강_강한픽셀비율 = 0.60


def 현재프레임_wrap_to_pi(value):
    value = np.asarray(
        value,
        dtype=np.float64,
    )
    return np.arctan2(
        np.sin(value),
        np.cos(value),
    )


def 현재프레임_마스크저장(path, mask):
    image = np.zeros(
        np.asarray(mask).shape,
        dtype=np.uint8,
    )
    image[
        np.asarray(
            mask,
            dtype=bool,
        )
    ] = 255
    cv2.imwrite(
        str(path),
        image,
    )


def 현재프레임_가장큰연결영역(mask):
    mask_u8 = (
        np.asarray(
            mask,
            dtype=bool,
        ).astype(
            np.uint8
        )
    )

    count, labels, stats, _ = (
        cv2.connectedComponentsWithStats(
            mask_u8,
            connectivity=8,
        )
    )

    if count <= 1:
        return np.zeros(
            mask_u8.shape,
            dtype=bool,
        )

    areas = stats[
        1:,
        cv2.CC_STAT_AREA,
    ]

    target = (
        int(
            np.argmax(
                areas
            )
        )
        + 1
    )

    return labels == target


def 현재프레임_작은컴포넌트제거(
    mask,
    min_size,
):
    mask_u8 = np.asarray(
        mask,
        dtype=np.uint8,
    )

    count, labels, stats, _ = (
        cv2.connectedComponentsWithStats(
            mask_u8,
            connectivity=8,
        )
    )

    result = np.zeros(
        mask_u8.shape,
        dtype=bool,
    )

    for label_id in range(
        1,
        count,
    ):
        area = int(
            stats[
                label_id,
                cv2.CC_STAT_AREA,
            ]
        )

        if area >= int(
            min_size
        ):
            result[
                labels
                == label_id
            ] = True

    return result


def 현재프레임_마스크팽창(
    mask,
    radius_px,
):
    radius_px = max(
        0,
        int(
            radius_px
        ),
    )

    if radius_px <= 0:
        return np.asarray(
            mask,
            dtype=bool,
        ).copy()

    k = (
        radius_px
        * 2
        + 1
    )

    kernel = (
        cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (k, k),
        )
    )

    return (
        cv2.dilate(
            np.asarray(
                mask,
                dtype=np.uint8,
            ),
            kernel,
            iterations=1,
        )
        > 0
    )


def 현재프레임_Depth물체마스크_읽기(
    depth_result_dir,
    shape,
    object_area,
):
    path = (
        Path(
            depth_result_dir
        )
        / "03_Depth_물체픽셀_확인용.png"
    )

    if not path.exists():
        raise FileNotFoundError(
            "현재 프레임 기준 복원에는 "
            "이번 촬영에서 만든 Depth 실제 물체 마스크가 필요합니다.\n"
            f"필요 파일: {path}"
        )

    image = cv2.imread(
        str(path),
        cv2.IMREAD_GRAYSCALE,
    )

    if image is None:
        raise RuntimeError(
            f"Depth 물체 마스크를 읽지 못했습니다: {path}"
        )

    mask = image > 127

    if mask.shape != tuple(
        shape
    ):
        raise RuntimeError(
            "Depth 물체 마스크와 구조광 위상 크기가 다릅니다: "
            f"Depth={mask.shape}, phase={shape}"
        )

    mask &= np.asarray(
        object_area,
        dtype=bool,
    )

    mask = (
        현재프레임_작은컴포넌트제거(
            mask,
            현재프레임_최소물체컴포넌트,
        )
    )

    if np.count_nonzero(
        mask
    ) < 현재프레임_최소물체컴포넌트:
        raise RuntimeError(
            "Depth 실제 물체 마스크가 너무 작습니다."
        )

    return (
        mask,
        path,
    )


def 현재프레임_QG언랩(
    wrapped,
    mask,
    quality=None,
):
    """
    mask 내부 연결영역별 quality-guided local unwrap.

    중요:
    - 플랫폼과 물체를 섞어서 전파하지 않는다.
    - 각 연결영역의 global 2π offset은 이 함수 밖에서 별도로 결정한다.
    """
    wrapped = np.asarray(
        wrapped,
        dtype=np.float64,
    )

    mask = (
        np.asarray(
            mask,
            dtype=bool,
        )
        & np.isfinite(
            wrapped
        )
    )

    if quality is None:
        quality = np.ones(
            wrapped.shape,
            dtype=np.float64,
        )
    else:
        quality = np.asarray(
            quality,
            dtype=np.float64,
        )
        quality = np.nan_to_num(
            quality,
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )

    count, labels, stats, _ = (
        cv2.connectedComponentsWithStats(
            mask.astype(
                np.uint8
            ),
            connectivity=8,
        )
    )

    unwrapped = np.full(
        wrapped.shape,
        np.nan,
        dtype=np.float64,
    )

    k_map = np.zeros(
        wrapped.shape,
        dtype=np.int16,
    )

    component_infos = []

    h, w = wrapped.shape

    for label_id in range(
        1,
        count,
    ):
        component = (
            labels
            == label_id
        )

        pixels = int(
            np.count_nonzero(
                component
            )
        )

        if pixels < (
            현재프레임_최소물체컴포넌트
        ):
            continue

        ys, xs = np.where(
            component
        )

        q_values = quality[
            component
        ]

        seed_local_index = int(
            np.argmax(
                q_values
            )
        )

        sy = int(
            ys[
                seed_local_index
            ]
        )
        sx = int(
            xs[
                seed_local_index
            ]
        )

        visited = np.zeros(
            wrapped.shape,
            dtype=bool,
        )

        visited[
            sy,
            sx,
        ] = True

        unwrapped[
            sy,
            sx,
        ] = wrapped[
            sy,
            sx,
        ]

        heap = []

        def push_edges(
            py,
            px,
        ):
            for ny, nx in (
                (py - 1, px),
                (py + 1, px),
                (py, px - 1),
                (py, px + 1),
            ):
                if (
                    ny < 0
                    or ny >= h
                    or nx < 0
                    or nx >= w
                ):
                    continue

                if (
                    not component[
                        ny,
                        nx,
                    ]
                    or visited[
                        ny,
                        nx,
                    ]
                ):
                    continue

                circular_jump = abs(
                    float(
                        현재프레임_wrap_to_pi(
                            wrapped[
                                ny,
                                nx,
                            ]
                            - wrapped[
                                py,
                                px,
                            ]
                        )
                    )
                )

                # 같은 jump라면 modulation이 높은 쪽을 먼저.
                q = min(
                    float(
                        quality[
                            py,
                            px,
                        ]
                    ),
                    float(
                        quality[
                            ny,
                            nx,
                        ]
                    ),
                )

                heapq.heappush(
                    heap,
                    (
                        circular_jump,
                        -q,
                        py,
                        px,
                        ny,
                        nx,
                    ),
                )

        push_edges(
            sy,
            sx,
        )

        solved = 1

        while heap:
            (
                _,
                _,
                py,
                px,
                y,
                x,
            ) = heapq.heappop(
                heap
            )

            if (
                visited[
                    y,
                    x,
                ]
                or not visited[
                    py,
                    px,
                ]
            ):
                continue

            local_delta = float(
                현재프레임_wrap_to_pi(
                    wrapped[
                        y,
                        x,
                    ]
                    - wrapped[
                        py,
                        px,
                    ]
                )
            )

            value = (
                float(
                    unwrapped[
                        py,
                        px,
                    ]
                )
                + local_delta
            )

            unwrapped[
                y,
                x,
            ] = value

            k_value = int(
                np.rint(
                    (
                        value
                        - float(
                            wrapped[
                                y,
                                x,
                            ]
                        )
                    )
                    / 현재프레임_2PI
                )
            )

            k_map[
                y,
                x,
            ] = np.int16(
                np.clip(
                    k_value,
                    -32768,
                    32767,
                )
            )

            visited[
                y,
                x,
            ] = True

            solved += 1

            push_edges(
                y,
                x,
            )

        component_infos.append(
            {
                "label": int(
                    label_id
                ),
                "pixels": pixels,
                "solved": int(
                    solved
                ),
                "seed_y": sy,
                "seed_x": sx,
            }
        )

    return (
        unwrapped,
        k_map,
        labels,
        component_infos,
    )


def 현재프레임_다항식행렬(
    x_norm,
    y_norm,
):
    return np.column_stack(
        [
            np.ones_like(
                x_norm,
                dtype=np.float64,
            ),
            x_norm,
            y_norm,
            x_norm
            * x_norm,
            x_norm
            * y_norm,
            y_norm
            * y_norm,
        ]
    )


def 현재프레임_플랫폼위상면_적합(
    platform_unwrapped,
    platform_fit_mask,
):
    """
    현재 프레임에서 보이는 플랫폼의 unwrapped carrier phase를
    robust 2차식으로 모델링한다.

    외부 빈 플랫폼 Reference를 사용하지 않는다.
    """
    data = np.asarray(
        platform_unwrapped,
        dtype=np.float64,
    )

    mask = (
        np.asarray(
            platform_fit_mask,
            dtype=bool,
        )
        & np.isfinite(
            data
        )
    )

    ys, xs = np.where(
        mask
    )

    z = data[
        mask
    ]

    if z.size < (
        현재프레임_최소플랫폼픽셀
    ):
        raise RuntimeError(
            "현재 프레임 플랫폼 기준면 fitting 픽셀이 너무 적습니다: "
            f"{z.size}px"
        )

    h, w = data.shape

    cx = (
        w
        - 1
    ) / 2.0
    cy = (
        h
        - 1
    ) / 2.0

    sx = max(
        1.0,
        (
            w
            - 1
        )
        / 2.0,
    )
    sy = max(
        1.0,
        (
            h
            - 1
        )
        / 2.0,
    )

    xn = (
        xs.astype(
            np.float64
        )
        - cx
    ) / sx

    yn = (
        ys.astype(
            np.float64
        )
        - cy
    ) / sy

    A = (
        현재프레임_다항식행렬(
            xn,
            yn,
        )
    )

    inlier = np.ones(
        z.shape,
        dtype=bool,
    )

    threshold = (
        현재프레임_평면최소문턱_RAD
    )

    coeff = None

    for _ in range(
        현재프레임_평면반복
    ):
        if np.count_nonzero(
            inlier
        ) < (
            현재프레임_최소플랫폼픽셀
        ):
            break

        coeff, _, _, _ = (
            np.linalg.lstsq(
                A[
                    inlier
                ],
                z[
                    inlier
                ],
                rcond=None,
            )
        )

        prediction = (
            A
            @ coeff
        )

        residual = (
            z
            - prediction
        )

        center = float(
            np.median(
                residual[
                    inlier
                ]
            )
        )

        mad = float(
            np.median(
                np.abs(
                    residual[
                        inlier
                    ]
                    - center
                )
            )
        )

        sigma = (
            1.4826
            * mad
        )

        threshold = max(
            현재프레임_평면최소문턱_RAD,
            현재프레임_평면MAD배수
            * sigma,
        )

        new_inlier = (
            np.abs(
                residual
                - center
            )
            <= threshold
        )

        if np.array_equal(
            new_inlier,
            inlier,
        ):
            inlier = (
                new_inlier
            )
            break

        inlier = (
            new_inlier
        )

    if (
        coeff is None
        or np.count_nonzero(
            inlier
        ) < (
            현재프레임_최소플랫폼픽셀
        )
    ):
        raise RuntimeError(
            "현재 프레임 플랫폼 위상면 robust fitting 실패"
        )

    coeff, _, _, _ = (
        np.linalg.lstsq(
            A[
                inlier
            ],
            z[
                inlier
            ],
            rcond=None,
        )
    )

    yy, xx = np.mgrid[
        0:h,
        0:w,
    ]

    xxn = (
        xx.astype(
            np.float64
        )
        - cx
    ) / sx

    yyn = (
        yy.astype(
            np.float64
        )
        - cy
    ) / sy

    A_full = np.stack(
        [
            np.ones_like(
                xxn
            ),
            xxn,
            yyn,
            xxn
            * xxn,
            xxn
            * yyn,
            yyn
            * yyn,
        ],
        axis=-1,
    )

    plane = np.tensordot(
        A_full,
        coeff,
        axes=(
            [-1],
            [0],
        ),
    )

    all_prediction = (
        A
        @ coeff
    )

    all_residual = (
        z
        - all_prediction
    )

    return (
        plane.astype(
            np.float64
        ),
        {
            "model": (
                "robust quadratic "
                "c0+c1*x+c2*y+c3*x2+c4*xy+c5*y2"
            ),
            "coeff_normalized": [
                float(
                    value
                )
                for value in coeff
            ],
            "candidate_pixels": int(
                z.size
            ),
            "inlier_pixels": int(
                np.count_nonzero(
                    inlier
                )
            ),
            "inlier_percent": float(
                np.count_nonzero(
                    inlier
                )
                / max(
                    1,
                    inlier.size,
                )
                * 100.0
            ),
            "robust_threshold_rad": float(
                threshold
            ),
            "platform_residual_median_rad": float(
                np.median(
                    all_residual
                )
            ),
            "platform_residual_mad_rad": float(
                np.median(
                    np.abs(
                        all_residual
                        - np.median(
                            all_residual
                        )
                    )
                )
            ),
            "platform_residual_p90_abs_rad": float(
                np.percentile(
                    np.abs(
                        all_residual
                    ),
                    90,
                )
            ),
        },
    )


def 현재프레임_물체globalshift_선택(
    values,
):
    """
    카메라-프로젝터 calibration 전이므로 절대 mm branch는 알 수 없다.
    물체는 플랫폼 위에 놓여 있다는 현재 프로젝트 전제만 사용해서
    대부분의 상대 위상이 음수가 되지 않는 가장 작은 global 2π branch를 고른다.

    이 global 상수는 물체 내부의 높이 변화/곡률 자체에는 영향을 주지 않는다.
    """
    values = np.asarray(
        values,
        dtype=np.float64,
    )

    values = values[
        np.isfinite(
            values
        )
    ]

    if values.size == 0:
        return (
            0,
            {},
        )

    rows = []

    for shift in range(
        -4,
        7,
    ):
        shifted = (
            values
            + 현재프레임_2PI
            * shift
        )

        negative_ratio = float(
            np.mean(
                shifted
                < -0.15
            )
        )

        (
            p1,
            p5,
            p50,
            p95,
            p99,
        ) = [
            float(
                value
            )
            for value in np.percentile(
                shifted,
                [
                    1,
                    5,
                    50,
                    95,
                    99,
                ],
            )
        ]

        score = (
            max(
                0.0,
                negative_ratio
                - 0.02,
            )
            * 10000.0
            + max(
                0.0,
                -p1
                - 0.20,
            )
            * 30.0
            + abs(
                p5
            )
            + max(
                0.0,
                p99
                - 18.0,
            )
            * 10.0
        )

        rows.append(
            (
                score,
                shift,
                negative_ratio,
                p1,
                p5,
                p50,
                p95,
                p99,
            )
        )

    best = min(
        rows,
        key=lambda item: (
            item[
                0
            ],
            abs(
                item[
                    1
                ]
            ),
        ),
    )

    return (
        int(
            best[
                1
            ]
        ),
        {
            "shift": int(
                best[
                    1
                ]
            ),
            "negative_ratio": float(
                best[
                    2
                ]
            ),
            "p1": float(
                best[
                    3
                ]
            ),
            "p5": float(
                best[
                    4
                ]
            ),
            "p50": float(
                best[
                    5
                ]
            ),
            "p95": float(
                best[
                    6
                ]
            ),
            "p99": float(
                best[
                    7
                ]
            ),
        },
    )


def 현재프레임_물체스파이크만_정리(
    relative_raw,
    object_mask,
):
    """
    물체의 전체 곡률/기울기는 보존한다.
    3x3 local median에서 매우 크게 튄 점만 교정한다.
    """
    data = np.asarray(
        relative_raw,
        dtype=np.float32,
    ).copy()

    object_mask = (
        np.asarray(
            object_mask,
            dtype=bool,
        )
        & np.isfinite(
            data
        )
    )

    spike_mask = np.zeros(
        object_mask.shape,
        dtype=bool,
    )

    count, labels, stats, _ = (
        cv2.connectedComponentsWithStats(
            object_mask.astype(
                np.uint8
            ),
            connectivity=8,
        )
    )

    for label_id in range(
        1,
        count,
    ):
        component = (
            labels
            == label_id
        )

        if np.count_nonzero(
            component
        ) < (
            현재프레임_최소물체컴포넌트
        ):
            continue

        median_value = float(
            np.median(
                data[
                    component
                ]
            )
        )

        filled = np.full(
            data.shape,
            median_value,
            dtype=np.float32,
        )

        filled[
            component
        ] = data[
            component
        ]

        local_median = (
            cv2.medianBlur(
                filled,
                3,
            )
        )

        local_residual = np.abs(
            data
            - local_median
        )

        bad = (
            component
            & (
                local_residual
                > 현재프레임_스파이크문턱_RAD
            )
        )

        data[
            bad
        ] = local_median[
            bad
        ]

        spike_mask |= (
            bad
        )

    return (
        data,
        spike_mask,
    )


def 현재프레임_높이컬러(
    surface,
    final_valid,
    object_mask,
):
    surface = np.asarray(
        surface,
        dtype=np.float32,
    )

    final_valid = np.asarray(
        final_valid,
        dtype=bool,
    )

    object_mask = (
        np.asarray(
            object_mask,
            dtype=bool,
        )
        & final_valid
        & np.isfinite(
            surface
        )
    )

    object_values = (
        surface[
            object_mask
        ]
    )

    if object_values.size > 0:
        low = float(
            np.percentile(
                object_values,
                2,
            )
        )

        high = float(
            np.percentile(
                object_values,
                98,
            )
        )
    else:
        all_values = surface[
            final_valid
            & np.isfinite(
                surface
            )
        ]

        low = float(
            np.percentile(
                all_values,
                2,
            )
        )
        high = float(
            np.percentile(
                all_values,
                98,
            )
        )

    if high <= (
        low
        + 1e-6
    ):
        high = (
            low
            + 1.0
        )

    normalized = np.clip(
        (
            surface
            - low
        )
        / (
            high
            - low
        ),
        0.0,
        1.0,
    )

    u8 = (
        np.nan_to_num(
            normalized,
            nan=0.0,
        )
        * 255.0
    ).astype(
        np.uint8
    )

    colors_bgr = (
        cv2.applyColorMap(
            u8,
            cv2.COLORMAP_TURBO,
        )
    )

    colors_bgr[
        ~final_valid
    ] = (
        0,
        0,
        0,
    )

    # 플랫폼은 반드시 한 색으로 표시.
    platform = (
        final_valid
        & (
            ~object_mask
        )
    )

    colors_bgr[
        platform
    ] = (
        220,
        220,
        220,
    )

    return (
        colors_bgr,
        low,
        high,
    )


def 현재프레임_PLY저장(
    path,
    surface,
    valid_mask,
    colors_bgr,
):
    surface = np.asarray(
        surface,
        dtype=np.float32,
    )

    valid = (
        np.asarray(
            valid_mask,
            dtype=bool,
        )
        & np.isfinite(
            surface
        )
    )

    sample = np.zeros(
        valid.shape,
        dtype=bool,
    )

    sample[
        ::현재프레임_PLY_SKIP,
        ::현재프레임_PLY_SKIP,
    ] = True

    valid &= sample

    ys, xs = np.where(
        valid
    )

    count = int(
        xs.size
    )

    h, w = surface.shape

    with Path(
        path
    ).open(
        "w",
        encoding="ascii",
    ) as file:
        file.write(
            "ply\n"
        )
        file.write(
            "format ascii 1.0\n"
        )
        file.write(
            f"element vertex {count}\n"
        )
        file.write(
            "property float x\n"
        )
        file.write(
            "property float y\n"
        )
        file.write(
            "property float z\n"
        )
        file.write(
            "property uchar red\n"
        )
        file.write(
            "property uchar green\n"
        )
        file.write(
            "property uchar blue\n"
        )
        file.write(
            "end_header\n"
        )

        for y, x in zip(
            ys,
            xs,
        ):
            X = float(
                x
                - w
                / 2.0
            )

            Y = float(
                h
                / 2.0
                - y
            )

            Z = float(
                현재프레임_Z_SIGN
                * 현재프레임_Z_SCALE
                * surface[
                    y,
                    x,
                ]
            )

            b, g, r = [
                int(
                    value
                )
                for value in colors_bgr[
                    y,
                    x,
                ]
            ]

            file.write(
                f"{X:.6f} "
                f"{Y:.6f} "
                f"{Z:.6f} "
                f"{r} {g} {b}\n"
            )

    return count



def 현재프레임_플랫폼피팅마스크_생성(
    object_mask,
    domain,
    object_area,
):
    """
    물체 주변은 팽창해서 제외하고,
    실제로 연결된 가장 큰 플랫폼 영역만 fitting에 사용한다.
    """
    object_excluded = (
        현재프레임_마스크팽창(
            object_mask,
            현재프레임_물체제외_팽창PX,
        )
    )

    # 분석 사각형 가장자리 영향 제거.
    area_eroded = (
        cv2.erode(
            np.asarray(
                object_area,
                dtype=np.uint8,
            ),
            np.ones(
                (9, 9),
                dtype=np.uint8,
            ),
            iterations=1,
        )
        > 0
    )

    platform_fit = (
        np.asarray(
            domain,
            dtype=bool,
        )
        & (~object_excluded)
        & area_eroded
    )

    # 물체 내부 Depth hole이 임시로 플랫폼처럼 보여도,
    # 바깥 플랫폼과 분리돼 있으면 fitting에 들어오지 않도록 가장 큰 연결영역만 사용.
    return 현재프레임_가장큰연결영역(
        platform_fit
    )


def 현재프레임_Depth내부홀_위상보강(
    depth_object_seed,
    domain,
    relative_to_platform_wrapped,
    platform_fit,
):
    """
    Depth가 반사/결측 때문에 물체 내부를 놓친 경우만 복구한다.

    핵심:
    1) Depth object mask의 '외곽 contour 안쪽'만 hole 후보로 본다.
       → 물체 바깥 플랫폼을 확장해서 먹지 않는다.
    2) 후보 hole의 구조광 phase가 현재 플랫폼 phase와 비슷하면
       실제 구멍/플랫폼으로 보고 그대로 둔다.
    3) 후보 hole의 구조광 phase가 플랫폼과 확실히 다르면
       Depth dropout으로 보고 물체로 복구한다.

    따라서 단순 binary hole-fill보다 실제 관통구멍을 보존할 가능성이 높다.
    """
    seed = (
        np.asarray(
            depth_object_seed,
            dtype=bool,
        )
        & np.asarray(
            domain,
            dtype=bool,
        )
    )

    relative = np.asarray(
        relative_to_platform_wrapped,
        dtype=np.float64,
    )

    platform_fit = (
        np.asarray(
            platform_fit,
            dtype=bool,
        )
        & np.isfinite(
            relative
        )
    )

    platform_values = np.abs(
        relative[
            platform_fit
        ]
    )

    if platform_values.size < 500:
        raise RuntimeError(
            "Depth hole 위상보강용 플랫폼 residual 픽셀이 너무 적습니다."
        )

    platform_p50 = float(
        np.percentile(
            platform_values,
            50,
        )
    )
    platform_p90 = float(
        np.percentile(
            platform_values,
            90,
        )
    )
    platform_p95 = float(
        np.percentile(
            platform_values,
            95,
        )
    )

    phase_threshold = max(
        현재프레임_홀보강_최소위상문턱RAD,
        platform_p95
        * 현재프레임_홀보강_플랫폼P95배수,
    )

    # Depth 물체의 외곽 silhouette를 채운다.
    # RETR_EXTERNAL이므로 내부 hole은 envelope에 포함된다.
    seed_u8 = seed.astype(
        np.uint8
    )

    contours, _ = cv2.findContours(
        seed_u8,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    envelope = np.zeros(
        seed.shape,
        dtype=np.uint8,
    )

    used_contours = 0

    for contour in contours:
        area = float(
            cv2.contourArea(
                contour
            )
        )

        if area < (
            현재프레임_최소물체컴포넌트
        ):
            continue

        cv2.drawContours(
            envelope,
            [contour],
            -1,
            1,
            thickness=-1,
        )
        used_contours += 1

    envelope = (
        envelope
        > 0
    ) & np.asarray(
        domain,
        dtype=bool,
    )

    hole_candidate = (
        envelope
        & (~seed)
        & np.isfinite(
            relative
        )
    )

    count, labels, stats, _ = (
        cv2.connectedComponentsWithStats(
            hole_candidate.astype(
                np.uint8
            ),
            connectivity=8,
        )
    )

    fill_mask = np.zeros(
        seed.shape,
        dtype=bool,
    )

    hole_infos = []

    for label_id in range(
        1,
        count,
    ):
        component = (
            labels
            == label_id
        )

        area = int(
            stats[
                label_id,
                cv2.CC_STAT_AREA,
            ]
        )

        if area < (
            현재프레임_홀보강_최소면적PX
        ):
            continue

        values = np.abs(
            relative[
                component
            ]
        )

        values = values[
            np.isfinite(
                values
            )
        ]

        if values.size == 0:
            continue

        p25 = float(
            np.percentile(
                values,
                25,
            )
        )
        p50 = float(
            np.percentile(
                values,
                50,
            )
        )
        p90 = float(
            np.percentile(
                values,
                90,
            )
        )

        strong_ratio = float(
            np.mean(
                values
                >= phase_threshold
            )
        )

        # 전체 component의 대부분이 플랫폼 phase와 다를 때만
        # "Depth가 놓친 물체"로 복구한다.
        fill_this = bool(
            (
                p50
                >= phase_threshold
            )
            and (
                strong_ratio
                >= 현재프레임_홀보강_강한픽셀비율
            )
        )

        if fill_this:
            fill_mask |= (
                component
            )

        hole_infos.append(
            {
                "label": int(
                    label_id
                ),
                "pixels": area,
                "p25_abs_rad": p25,
                "median_abs_rad": p50,
                "p90_abs_rad": p90,
                "strong_ratio": strong_ratio,
                "filled_as_object": fill_this,
            }
        )

    refined_object = (
        seed
        | fill_mask
    )

    info = {
        "used_external_contours": int(
            used_contours
        ),
        "platform_residual_p50_abs_rad": (
            platform_p50
        ),
        "platform_residual_p90_abs_rad": (
            platform_p90
        ),
        "platform_residual_p95_abs_rad": (
            platform_p95
        ),
        "phase_threshold_rad": float(
            phase_threshold
        ),
        "hole_candidate_pixels": int(
            np.count_nonzero(
                hole_candidate
            )
        ),
        "filled_pixels": int(
            np.count_nonzero(
                fill_mask
            )
        ),
        "depth_seed_pixels": int(
            np.count_nonzero(
                seed
            )
        ),
        "refined_object_pixels": int(
            np.count_nonzero(
                refined_object
            )
        ),
        "hole_components": (
            hole_infos
        ),
    }

    return (
        refined_object,
        hole_candidate,
        fill_mask,
        envelope,
        info,
    )


def 현재프레임_복원_실행(
    struct_dir,
    post_root,
    depth_result_dir,
    object_phase,
    object_area,
    object_valid,
    modulation,
):
    output_dir = (
        Path(
            post_root
        )
        / "04_현재프레임플랫폼기준_최종_v2_Depth홀위상보강"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    object_phase = np.asarray(
        object_phase,
        dtype=np.float64,
    )

    object_area = np.asarray(
        object_area,
        dtype=bool,
    )

    object_valid = np.asarray(
        object_valid,
        dtype=bool,
    )

    modulation = np.asarray(
        modulation,
        dtype=np.float64,
    )

    domain = (
        object_area
        & object_valid
        & np.isfinite(
            object_phase
        )
    )

    (
        depth_object_mask,
        depth_mask_path,
    ) = 현재프레임_Depth물체마스크_읽기(
        depth_result_dir,
        object_phase.shape,
        object_area,
    )

    # Depth는 최종 물체 경계가 아니라 "물체 seed/silhouette"로만 사용.
    depth_object_seed = (
        depth_object_mask
        & domain
    )

    object_mask = (
        depth_object_seed.copy()
    )

    platform_all = (
        domain
        & (
            ~object_mask
        )
    )

    platform_fit = (
        현재프레임_플랫폼피팅마스크_생성(
            object_mask,
            domain,
            object_area,
        )
    )

    platform_count = int(
        np.count_nonzero(
            platform_fit
        )
    )

    depth_seed_count = int(
        np.count_nonzero(
            depth_object_seed
        )
    )

    object_count = (
        depth_seed_count
    )

    domain_count = int(
        np.count_nonzero(
            domain
        )
    )

    if platform_count < (
        현재프레임_최소플랫폼픽셀
    ):
        raise RuntimeError(
            "현재 프레임에서 기준으로 쓸 플랫폼 픽셀이 너무 적습니다: "
            f"{platform_count}px"
        )

    print("")
    print("=" * 100)
    print(
        "현재 프레임 플랫폼 기준 상대형상 복원"
    )
    print("=" * 100)
    print(
        "외부 빈 플랫폼 Reference phase: 사용 안 함"
    )
    print(
        f"전체 유효 domain: {domain_count} px"
    )
    print(
        f"Depth 물체 seed: {depth_seed_count} px"
    )
    print(
        f"현재 프레임 플랫폼 fitting: {platform_count} px"
    )
    print(
        f"Depth 물체 제외 여유: {현재프레임_물체제외_팽창PX}px"
    )

    # -----------------------------------------------------------------
    # 1) 같은 프레임의 플랫폼 raw wrapped phase만 local unwrap
    # -----------------------------------------------------------------
    (
        platform_unwrapped,
        platform_local_k,
        _,
        platform_components,
    ) = 현재프레임_QG언랩(
        object_phase,
        platform_fit,
        modulation,
    )

    # -----------------------------------------------------------------
    # 2) 같은 프레임 플랫폼으로 carrier/reference phase field 모델 생성
    # -----------------------------------------------------------------
    (
        platform_phase_surface,
        platform_model_info,
    ) = 현재프레임_플랫폼위상면_적합(
        platform_unwrapped,
        platform_fit,
    )

    platform_residual_wrapped = (
        현재프레임_wrap_to_pi(
            object_phase
            - platform_phase_surface
        )
    )

    platform_residual_values = (
        platform_residual_wrapped[
            platform_fit
        ]
    )

    platform_residual_p90 = float(
        np.percentile(
            np.abs(
                platform_residual_values
            ),
            90,
        )
    )

    # -----------------------------------------------------------------
    # 2-B) Depth 내부 hole을 같은 프레임 phase로 검증해서 물체로 복구
    # -----------------------------------------------------------------
    (
        refined_object_mask,
        depth_hole_candidate,
        depth_hole_filled,
        depth_object_envelope,
        depth_hole_info,
    ) = 현재프레임_Depth내부홀_위상보강(
        depth_object_seed,
        domain,
        platform_residual_wrapped,
        platform_fit,
    )

    object_mask = (
        refined_object_mask
        & domain
    )

    object_count = int(
        np.count_nonzero(
            object_mask
        )
    )

    # 물체 마스크가 바뀌었으므로 플랫폼 영역과 기준면을 한 번 최종 재계산.
    platform_all = (
        domain
        & (
            ~object_mask
        )
    )

    platform_fit = (
        현재프레임_플랫폼피팅마스크_생성(
            object_mask,
            domain,
            object_area,
        )
    )

    platform_count = int(
        np.count_nonzero(
            platform_fit
        )
    )

    if platform_count < (
        현재프레임_최소플랫폼픽셀
    ):
        raise RuntimeError(
            "Depth hole 보강 후 플랫폼 fitting 픽셀이 너무 적습니다: "
            f"{platform_count}px"
        )

    (
        platform_unwrapped,
        platform_local_k,
        _,
        platform_components,
    ) = 현재프레임_QG언랩(
        object_phase,
        platform_fit,
        modulation,
    )

    (
        platform_phase_surface,
        platform_model_info,
    ) = 현재프레임_플랫폼위상면_적합(
        platform_unwrapped,
        platform_fit,
    )

    platform_residual_wrapped = (
        현재프레임_wrap_to_pi(
            object_phase
            - platform_phase_surface
        )
    )

    platform_residual_values = (
        platform_residual_wrapped[
            platform_fit
        ]
    )

    platform_residual_p90 = float(
        np.percentile(
            np.abs(
                platform_residual_values
            ),
            90,
        )
    )

    print("")
    print(
        "[Depth 내부 hole 구조광 phase 보강]"
    )
    print(
        f"Depth seed: {depth_seed_count:,} px"
    )
    print(
        f"hole 후보: "
        f"{depth_hole_info['hole_candidate_pixels']:,} px"
    )
    print(
        f"물체로 복구: "
        f"{depth_hole_info['filled_pixels']:,} px"
    )
    print(
        f"최종 물체: {object_count:,} px"
    )
    print(
        f"phase 문턱: "
        f"{depth_hole_info['phase_threshold_rad']:.4f} rad "
        f"({np.degrees(depth_hole_info['phase_threshold_rad']):.2f}°)"
    )

    for hole in depth_hole_info[
        "hole_components"
    ]:
        print(
            "  hole "
            f"{hole['label']}: "
            f"{hole['pixels']:,}px | "
            f"median={hole['median_abs_rad']:.3f}rad | "
            f"strong={hole['strong_ratio']*100.0:.1f}% | "
            f"물체복구={hole['filled_as_object']}"
        )

    # -----------------------------------------------------------------
    # 3) 물체 상대위상: 외부 Reference가 아니라 같은 프레임 플랫폼 모델을 뺌
    # -----------------------------------------------------------------
    relative_wrapped = (
        현재프레임_wrap_to_pi(
            object_phase
            - platform_phase_surface
        )
    )

    relative_wrapped[
        ~domain
    ] = np.nan

    # 플랫폼은 기준면이므로 최종 상대높이 0.
    relative_raw = np.full(
        object_phase.shape,
        np.nan,
        dtype=np.float64,
    )

    relative_raw[
        platform_all
    ] = 0.0

    # -----------------------------------------------------------------
    # 4) 물체 내부만 local QG unwrap
    #    플랫폼과 물체를 같은 graph로 연결하지 않음.
    # -----------------------------------------------------------------
    (
        object_local_unwrapped,
        object_local_k,
        object_labels,
        object_components,
    ) = 현재프레임_QG언랩(
        relative_wrapped,
        object_mask,
        modulation,
    )

    final_k = np.zeros(
        object_phase.shape,
        dtype=np.int16,
    )

    shift_infos = []

    for info in object_components:
        label_id = int(
            info[
                "label"
            ]
        )

        component = (
            object_labels
            == label_id
        )

        component &= np.isfinite(
            object_local_unwrapped
        )

        if np.count_nonzero(
            component
        ) == 0:
            continue

        shift, shift_info = (
            현재프레임_물체globalshift_선택(
                object_local_unwrapped[
                    component
                ]
            )
        )

        relative_raw[
            component
        ] = (
            object_local_unwrapped[
                component
            ]
            + 현재프레임_2PI
            * shift
        )

        k_total = (
            object_local_k[
                component
            ].astype(
                np.int32
            )
            + int(
                shift
            )
        )

        final_k[
            component
        ] = np.clip(
            k_total,
            -32768,
            32767,
        ).astype(
            np.int16
        )

        shift_info[
            "component_label"
        ] = label_id

        shift_info[
            "pixels"
        ] = int(
            np.count_nonzero(
                component
            )
        )

        shift_infos.append(
            shift_info
        )

    object_solved = (
        object_mask
        & np.isfinite(
            relative_raw
        )
    )

    final_valid = (
        platform_all
        | object_solved
    )

    # -----------------------------------------------------------------
    # 5) 물체의 큰 점튀기만 정리. 곡률/높이 gradient는 그대로 보존.
    # -----------------------------------------------------------------
    (
        relative_final,
        spike_mask,
    ) = 현재프레임_물체스파이크만_정리(
        relative_raw,
        object_solved,
    )

    # 플랫폼은 표시/형상 모두 정확히 기준 0으로 고정.
    relative_final[
        platform_all
    ] = 0.0

    relative_final[
        ~final_valid
    ] = np.nan

    # -----------------------------------------------------------------
    # 6) 높이 색 + PLY
    # -----------------------------------------------------------------
    (
        colors_bgr,
        color_low,
        color_high,
    ) = 현재프레임_높이컬러(
        relative_final,
        final_valid,
        object_solved,
    )

    raw_colors_bgr, _, _ = (
        현재프레임_높이컬러(
            relative_raw,
            final_valid,
            object_solved,
        )
    )

    raw_ply = (
        output_dir
        / "01_v2_현재프레임기준_RAW_물체+플랫폼.ply"
    )

    final_ply = (
        output_dir
        / "02_v2_현재프레임기준_최종_물체+플랫폼.ply"
    )

    object_ply = (
        output_dir
        / "03_v2_현재프레임기준_최종_물체만.ply"
    )

    raw_points = 현재프레임_PLY저장(
        raw_ply,
        relative_raw,
        final_valid,
        raw_colors_bgr,
    )

    final_points = 현재프레임_PLY저장(
        final_ply,
        relative_final,
        final_valid,
        colors_bgr,
    )

    object_points = 현재프레임_PLY저장(
        object_ply,
        relative_final,
        object_solved,
        colors_bgr,
    )

    # -----------------------------------------------------------------
    # 7) 저장
    # -----------------------------------------------------------------
    np.save(
        output_dir
        / "01_domain.npy",
        domain.astype(
            bool
        ),
    )

    np.save(
        output_dir
        / "02_depth_object_seed_mask.npy",
        depth_object_seed.astype(
            bool
        ),
    )

    np.save(
        output_dir
        / "02A_depth_hole_candidate.npy",
        depth_hole_candidate.astype(
            bool
        ),
    )

    np.save(
        output_dir
        / "02B_depth_hole_filled_as_object.npy",
        depth_hole_filled.astype(
            bool
        ),
    )

    np.save(
        output_dir
        / "02C_final_object_mask.npy",
        object_mask.astype(
            bool
        ),
    )

    # 기존 후속 분석 호환용 이름도 최종 보강 object mask로 저장.
    np.save(
        output_dir
        / "02_depth_object_mask.npy",
        object_mask.astype(
            bool
        ),
    )

    np.save(
        output_dir
        / "03_platform_all_mask.npy",
        platform_all.astype(
            bool
        ),
    )

    np.save(
        output_dir
        / "04_platform_fit_mask.npy",
        platform_fit.astype(
            bool
        ),
    )

    np.save(
        output_dir
        / "05_platform_unwrapped_phase.npy",
        platform_unwrapped.astype(
            np.float32
        ),
    )

    np.save(
        output_dir
        / "06_current_platform_phase_surface.npy",
        platform_phase_surface.astype(
            np.float32
        ),
    )

    np.save(
        output_dir
        / "07_relative_wrapped.npy",
        relative_wrapped.astype(
            np.float32
        ),
    )

    np.save(
        output_dir
        / "08_object_local_unwrapped.npy",
        object_local_unwrapped.astype(
            np.float32
        ),
    )

    np.save(
        output_dir
        / "09_final_k_map.npy",
        final_k.astype(
            np.int16
        ),
    )

    np.save(
        output_dir
        / "10_relative_RAW.npy",
        relative_raw.astype(
            np.float32
        ),
    )

    np.save(
        output_dir
        / "11_relative_FINAL.npy",
        relative_final.astype(
            np.float32
        ),
    )

    np.save(
        output_dir
        / "12_spike_mask.npy",
        spike_mask.astype(
            bool
        ),
    )

    현재프레임_마스크저장(
        output_dir
        / "01_domain.png",
        domain,
    )

    현재프레임_마스크저장(
        output_dir
        / "02_Depth물체_seed_원본.png",
        depth_object_seed,
    )

    현재프레임_마스크저장(
        output_dir
        / "02A_Depth내부hole_후보.png",
        depth_hole_candidate,
    )

    현재프레임_마스크저장(
        output_dir
        / "02B_구조광phase로_물체복구된hole.png",
        depth_hole_filled,
    )

    현재프레임_마스크저장(
        output_dir
        / "02C_최종물체마스크.png",
        object_mask,
    )

    # 기존 이름 호환용.
    현재프레임_마스크저장(
        output_dir
        / "02_Depth물체.png",
        object_mask,
    )

    현재프레임_마스크저장(
        output_dir
        / "03_플랫폼전체.png",
        platform_all,
    )

    현재프레임_마스크저장(
        output_dir
        / "04_플랫폼기준면피팅.png",
        platform_fit,
    )

    현재프레임_마스크저장(
        output_dir
        / "05_물체스파이크교정.png",
        spike_mask,
    )

    cv2.imwrite(
        str(
            output_dir
            / "06_RAW_높이컬러.png"
        ),
        raw_colors_bgr,
    )

    cv2.imwrite(
        str(
            output_dir
            / "07_FINAL_높이컬러.png"
        ),
        colors_bgr,
    )

    # 플랫폼 residual 확인용 컬러맵
    residual_vis = np.zeros(
        (
            object_phase.shape[
                0
            ],
            object_phase.shape[
                1
            ],
            3,
        ),
        dtype=np.uint8,
    )

    residual_abs = np.abs(
        platform_residual_wrapped
    )

    residual_u8 = np.clip(
        residual_abs
        / np.pi
        * 255.0,
        0.0,
        255.0,
    ).astype(
        np.uint8
    )

    residual_color = (
        cv2.applyColorMap(
            residual_u8,
            cv2.COLORMAP_TURBO,
        )
    )

    residual_vis[
        platform_fit
    ] = residual_color[
        platform_fit
    ]

    cv2.imwrite(
        str(
            output_dir
            / "08_현재플랫폼모델_잔차.png"
        ),
        residual_vis,
    )

    object_values = relative_final[
        object_solved
        & np.isfinite(
            relative_final
        )
    ]

    if object_values.size > 0:
        object_stats = {
            "p1_rad": float(
                np.percentile(
                    object_values,
                    1,
                )
            ),
            "p5_rad": float(
                np.percentile(
                    object_values,
                    5,
                )
            ),
            "median_rad": float(
                np.median(
                    object_values
                )
            ),
            "p95_rad": float(
                np.percentile(
                    object_values,
                    95,
                )
            ),
            "p99_rad": float(
                np.percentile(
                    object_values,
                    99,
                )
            ),
            "range_p5_p95_rad": float(
                np.percentile(
                    object_values,
                    95,
                )
                - np.percentile(
                    object_values,
                    5,
                )
            ),
        }
    else:
        object_stats = {}

    summary = {
        "method": (
            "same-frame platform phase field; "
            "no external empty-platform Reference phase subtraction"
        ),
        "depth_object_mask_source": str(
            depth_mask_path
        ),
        "depth_mask_role": (
            "seed/silhouette only; internal holes are phase-validated"
        ),
        "depth_seed_pixels": int(
            depth_seed_count
        ),
        "depth_hole_phase_refinement": (
            depth_hole_info
        ),
        "domain_pixels": domain_count,
        "object_pixels_after_hole_refinement": object_count,
        "object_solved_pixels": int(
            np.count_nonzero(
                object_solved
            )
        ),
        "platform_pixels": int(
            np.count_nonzero(
                platform_all
            )
        ),
        "platform_fit_pixels": platform_count,
        "platform_exclude_dilate_px": int(
            현재프레임_물체제외_팽창PX
        ),
        "platform_qg_components": (
            platform_components
        ),
        "platform_model": (
            platform_model_info
        ),
        "platform_same_frame_residual_p90_abs_rad": (
            platform_residual_p90
        ),
        "object_qg_components": (
            object_components
        ),
        "object_global_shifts": (
            shift_infos
        ),
        "object_relative_stats": (
            object_stats
        ),
        "spike_pixels_corrected": int(
            np.count_nonzero(
                spike_mask
            )
        ),
        "height_color": {
            "low_rad": float(
                color_low
            ),
            "high_rad": float(
                color_high
            ),
            "platform_color": (
                "fixed light gray"
            ),
        },
        "ply": {
            "actual_mm": False,
            "z_formula": (
                f"{현재프레임_Z_SIGN} * relative_phase * "
                f"{현재프레임_Z_SCALE}"
            ),
            "raw": str(
                raw_ply
            ),
            "final": str(
                final_ply
            ),
            "object_only": str(
                object_ply
            ),
            "raw_points": int(
                raw_points
            ),
            "final_points": int(
                final_points
            ),
            "object_points": int(
                object_points
            ),
        },
        "important": (
            "외부 Reference phase는 최종 상대형상 계산에 사용하지 않았다. "
            "Depth 실제 물체 픽셀은 최종 경계가 아니라 seed/silhouette로만 사용했다. "
            "Depth 외곽 내부의 hole은 같은 프레임 플랫폼 phase와 비교해 "
            "플랫폼과 확실히 다른 hole만 물체로 복구했다. "
            "현재 물체 촬영 프레임에서 보이는 플랫폼만으로 carrier/reference phase field를 만들고, "
            "플랫폼과 물체를 분리한 상태로 물체에서만 local QG unwrap을 수행했다. "
            "Z는 기존 프로젝트와 같은 상대위상 시각화값이며 실제 mm가 아니다."
        ),
    }

    summary_path = (
        output_dir
        / "00_현재프레임기준_요약.json"
    )

    summary_path.write_text(
        json.dumps(
            summary,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print("")
    print(
        "[현재 프레임 플랫폼 모델]"
    )
    print(
        f"플랫폼 residual P90: "
        f"{platform_residual_p90:.6f} rad "
        f"({np.degrees(platform_residual_p90):.2f}°)"
    )
    print(
        f"플랫폼 model inlier: "
        f"{platform_model_info['inlier_pixels']:,} / "
        f"{platform_model_info['candidate_pixels']:,} "
        f"({platform_model_info['inlier_percent']:.2f}%)"
    )

    if object_stats:
        print("")
        print(
            "[물체 상대위상]"
        )
        print(
            f"P5/median/P95: "
            f"{object_stats['p5_rad']:.4f} / "
            f"{object_stats['median_rad']:.4f} / "
            f"{object_stats['p95_rad']:.4f} rad"
        )
        print(
            f"P5~P95 변화폭: "
            f"{object_stats['range_p5_p95_rad']:.4f} rad"
        )

    print("")
    print("=" * 100)
    print(
        "현재 프레임 플랫폼 기준 복원 완료"
    )
    print("=" * 100)
    print(
        f"FINAL PLY: {final_ply}"
    )
    print(
        f"물체만 PLY: {object_ply}"
    )
    print(
        f"높이 컬러 미리보기: "
        f"{output_dir / '07_FINAL_높이컬러.png'}"
    )
    print(
        f"요약: {summary_path}"
    )
    print("=" * 100)

    return {
        "output_dir": output_dir,
        "final_ply": final_ply,
        "object_ply": object_ply,
        "summary": summary_path,
    }


def 최종통합_후처리_실행(
    cap,
    window_name,
    coord,
    monitor,
    args,
    run_dir,
    depth_result_dir,
    rect,
    preview,
    probe_stats,
    기본광_전략,
    baseline_result_dir,
    baseline_fusion,
    baseline_fusion_quality,
    baseline_conditions,
    baseline_captures,
    baseline_qualities,
):
    """
    최종 실행 경로.

    유지:
    - 기존 Depth 자동 영역
    - 기존 Color G/E 자동 탐색
    - 기존 픽셀단위 4위상 G/E 융합
    - 기존 P100 object phase/valid 생성

    교체:
    - 외부 빈 플랫폼 Reference 차분 사용 안 함
    - Reference-dependent 기본광 문제영역/HDR 사용 안 함
    - 플랫폼+물체를 한 graph로 묶는 기존 QG 사용 안 함

    새 최종:
    - 현재 물체 프레임의 플랫폼만 local unwrap
    - 그 플랫폼으로 현재 phase field 모델 생성
    - 같은 프레임 기준 상대위상
    - 물체에서만 local QG unwrap
    - 플랫폼은 최종 상대높이 0
    """
    baseline_result_dir = Path(
        baseline_result_dir
    )

    (
        struct_dir,
        object_phase,
        object_area,
        object_valid,
    ) = P100_Object핵심결과_저장(
        baseline_result_dir,
        baseline_fusion,
        baseline_fusion_quality,
    )

    post_root = (
        struct_dir
        / "최종자동통합"
    )

    post_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    # 현재 프레임 기준 방식에서는 외부 Reference를 읽지 않는다.
    # 사용자가 실행 명령에 --reference_dir을 남겨도 최종 형상에는 영향이 없다.
    info_path = (
        post_root
        / "00_현재프레임기준_실행정보.json"
    )

    info_path.write_text(
        json.dumps(
            {
                "final_method": (
                    "same-frame platform reference + phase-validated Depth internal-hole refinement"
                ),
                "external_reference_used": False,
                "reference_dir_argument_ignored_for_final_shape": str(
                    getattr(
                        args,
                        "reference_dir",
                        "",
                    )
                ),
                "depth_exposure": int(
                    Depth_노출
                ),
                "depth_gain": int(
                    Depth_게인
                ),
                "background_depth": str(
                    기준_Depth_경로
                ),
                "direction": str(
                    args.direction
                ),
                "period": int(
                    args.period
                ),
                "base": float(
                    args.base
                ),
                "amplitude": float(
                    args.amplitude
                ),
                "note": (
                    "G/E fusion까지는 기존 코드 그대로. "
                    "외부 Reference/HDR/QG 후단만 현재 프레임 플랫폼 기준으로 교체."
                ),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    modulation = np.asarray(
        baseline_fusion_quality[
            "변조도_지도"
        ],
        dtype=np.float64,
    )

    current_result = (
        현재프레임_복원_실행(
            struct_dir=struct_dir,
            post_root=post_root,
            depth_result_dir=depth_result_dir,
            object_phase=object_phase,
            object_area=object_area,
            object_valid=object_valid,
            modulation=modulation,
        )
    )

    final_ply = Path(
        current_result[
            "final_ply"
        ]
    )

    if not final_ply.exists():
        raise FileNotFoundError(
            "현재 프레임 기준 최종 PLY가 생성되지 않았습니다: "
            f"{final_ply}"
        )

    print("")
    print("=" * 100)
    print(
        "최종 자동 통합 완료 - 현재 프레임 플랫폼 기준"
    )
    print(
        f"상대광응답: "
        f"{기본광_전략['상대광응답_흰색100']:.2f}"
    )
    print(
        f"G/E 기본광 분류: "
        f"{기본광_전략['분류']}"
    )
    print(
        "외부 Reference phase: 최종 복원에서 사용 안 함"
    )
    print(
        f"최종 PLY: {final_ply}"
    )
    print("=" * 100)

    return {
        "후처리_루트": str(
            post_root
        ),
        "clean_dir": (
            "외부Reference 기반 clean HDR 미사용"
        ),
        "quality_dir": (
            "현재프레임 물체전용 local QG"
        ),
        "platform_dir": str(
            current_result[
                "output_dir"
            ]
        ),
        "최종_PLY": str(
            final_ply
        ),
    }

def 자체검증_실행():
    """
    카메라/프로젝터 없이 통합 glue 로직의 핵심 불변조건을 검사한다.
    실제 장비 I/O는 검사하지 않는다.
    """
    # 1) 하드코딩 전략
    expected = [
        (59.001999, "밝음"),
        (54.681999, "중간"),
        (13.002000, "중간"),
        (26.825001, "중간"),
        (8.712700, "어두움"),
    ]

    for p90, label in expected:
        stats = {
            "보정광응답_P90": float(
                p90
            )
        }

        result = 상대광응답_기본광전략_결정(
            stats
        )

        assert (
            result[
                "분류"
            ]
            == label
        )

    # 2) 같은 픽셀 네 위상 동일 조건 보존 / unresolved baseline 보존
    h, w = 8, 9
    problem = np.zeros(
        (
            h,
            w,
        ),
        dtype=bool,
    )
    problem[
        2:6,
        3:7,
    ] = True

    baseline_valid = np.ones(
        (
            h,
            w,
        ),
        dtype=bool,
    )
    object_area = np.ones(
        (
            h,
            w,
        ),
        dtype=bool,
    )
    reference_phase = np.zeros(
        (
            h,
            w,
        ),
        dtype=np.float32,
    )
    reference_valid = np.ones(
        (
            h,
            w,
        ),
        dtype=bool,
    )

    def make_capture(offset):
        gray = {}
        color = {}

        # 정상적인 4위상 관계를 유지하면서 조건별 offset을 구분.
        values = {
            "000": 120 + offset,
            "090": 80 + offset,
            "180": 40 + offset,
            "270": 80 + offset,
        }

        for name, value in values.items():
            g = np.full(
                (
                    h,
                    w,
                ),
                np.clip(
                    value,
                    0,
                    240,
                ),
                dtype=np.uint8,
            )

            gray[
                name
            ] = g

            color[
                name
            ] = cv2.merge(
                [
                    g,
                    g,
                    g,
                ]
            )

        return {
            "회색": gray,
            "컬러": color,
        }

    baseline_capture = make_capture(
        0
    )

    candidates = []

    for index, offset in enumerate(
        [
            2,
            4,
            6,
        ]
    ):
        candidates.append(
            {
                "role": str(
                    index
                ),
                "projector_percent": int(
                    10
                    + index
                    * 2
                ),
                "gain": 16,
                "exposure": 156,
                "folder": Path(
                    f"/tmp/selftest_{index}"
                ),
                "capture": make_capture(
                    offset
                ),
            }
        )

    class _Args:
        saturation_threshold = 250
        dark_threshold = 10
        modulation_threshold = 15.0

    with __import__(
        "tempfile"
    ).TemporaryDirectory() as tmp:
        result = 기본광_원래HDR방식_문제영역교체(
            output_dir=Path(
                tmp
            ),
            baseline_capture=baseline_capture,
            baseline_valid=baseline_valid,
            problem_mask=problem,
            candidate_items=candidates,
            reference_phase=reference_phase,
            reference_valid=reference_valid,
            object_area=object_area,
            args=_Args(),
        )

        assert result[
            "common_valid"
        ].shape == (
            h,
            w,
        )

        # problem 밖은 baseline phase000 그대로.
        saved = cv2.imread(
            str(
                Path(
                    tmp
                )
                / "phase_000.png"
            ),
            cv2.IMREAD_GRAYSCALE,
        )

        assert np.all(
            saved[
                ~problem
            ]
            == baseline_capture[
                "회색"
            ][
                "000"
            ][
                ~problem
            ]
        )

    # 3) 원본 성공 후단 소스가 포함되어 있고 main이 존재하는지 컴파일.
    compile(
        원본_STAGE6_SOURCE,
        "<selftest:stage6>",
        "exec",
    )

    compile(
        원본_PLATFORM_STAGE_SOURCE,
        "<selftest:platform>",
        "exec",
    )

    print("")
    print("=" * 78)
    print(
        "SELF TEST 통과"
    )
    print(
        "- 상대광응답 분기"
    )
    print(
        "- 문제영역 교체의 baseline 보존"
    )
    print(
        "- 원본 Quality-guided / 플랫폼 후단 소스 문법"
    )
    print(
        "※ 실제 Orbbec/프로젝터 I/O는 장비가 연결된 PC에서만 검증 가능합니다."
    )
    print("=" * 78)


def main():
    import csv

    args = 인자_읽기()

    if getattr(args, "self_test", False):
        자체검증_실행()
        return

    args.modulation_threshold = 15.0

    sample_name = str(
        args.sample
    ).strip()

    if not sample_name:
        sample_name = "이름없음"

    safe_sample = 안전한_이름(
        sample_name
    )

    run_dir = (
        Path(args.out_root)
        / safe_sample
        / datetime.now().strftime(
            "촬영_%Y%m%d_%H%M%S"
        )
    )

    run_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    monitor = 프로젝터_화면_선택(
        args
    )

    settings = 공통_카메라값_불러오기(
        args.camera_settings,
        시작_게인_13,
        args.white_balance,
    )

    settings["gain"] = 시작_게인_13

    cap = None

    try:
        cap = 카메라_열기(
            args,
            settings,
        )

        depth_result_dir = (
            run_dir
            / "00_Depth자동영역"
        )

        rect, preview, object_analysis_mask = Depth기반_자동_물체영역_검출(
            cap,
            sample_name,
            depth_result_dir,
        )

        # 이후 초기 RGB 광응답, 개별 4위상 품질, 최종 융합 유효율 모두
        # 방금 Depth로 찾은 '빨간 사각형 내부 전체'를 기준으로 계산한다.
        # X/Y 수동 범위는 최대 한계일 뿐이며, 작은 물체는 작은 사각형으로 잡힌다.
        args.자동_물체_분석마스크 = object_analysis_mask

        window_name = 프로젝터_창_준비(
            monitor
        )

        coord = 패턴_좌표_생성(
            monitor["w"],
            monitor["h"],
            args.direction,
        )

        device = cap

        gain_range = 제어값_허용범위_읽기(
            device,
            "gain",
        )

        exposure_range = 노출_허용범위_읽기(
            device
        )

        gains = 범위내_앵커_13(
            게인_앵커_13,
            gain_range,
        )

        exposures = 범위내_앵커_13(
            노출_앵커_13,
            exposure_range,
        )

        # 시작값이 장치 범위 내이면 반드시 포함.
        if (
            gain_range is None
            or (
                gain_range["min"]
                <= 시작_게인_13
                <= gain_range["max"]
            )
        ):
            gains = sorted(
                set(
                    gains
                    + [
                        시작_게인_13
                    ]
                )
            )

        if (
            exposure_range is None
            or (
                exposure_range["min"]
                <= 시작_노출_13
                <= exposure_range["max"]
            )
        ):
            exposures = sorted(
                set(
                    exposures
                    + [
                        시작_노출_13
                    ]
                )
            )

        print("")
        print("=" * 78)
        print(
            "RGB 초기광응답 + 유동형 2축 고속 탐색"
        )
        print(
            f"목표 유효율: "
            f"{목표_유효율_13:.2f}%"
        )
        print(
            f"Exposure 후보: "
            f"{exposures}"
        )
        print(
            f"Gain 후보: "
            f"{gains}"
        )
        print(
            f"최대 촬영조건 수: "
            f"{최대_촬영조건수_13}"
        )
        print(
            "중요: 밝음/어두움 판정은 탐색 방향을 고정하고, "
            "혼합 판정에서만 기존 2D 탐색을 사용합니다."
        )
        print("=" * 78)

        # ----------------------------------------------------
        # 1) Depth 기반 자동 물체 영역 검출 완료
        #    자동 검출에는 위에서 시작한 동일 Pipeline을 그대로 사용함.
        #    구조광 측정 시작 전 기존 실험 기준 G16/E156으로 맞춤.
        # ----------------------------------------------------
        print("")
        print(
            "Depth 자동 물체 영역 검출 완료 → "
            "실험 기준 G16/E156으로 맞춥니다."
        )

        게인값_적용(
            cap,
            args.cam,
            시작_게인_13,
            args.gain_settle,
        )

        노출값_적용(
            cap,
            args.cam,
            시작_노출_13,
            args.exposure_settle,
        )

        # ----------------------------------------------------
        # 2) RGB0 + RGB64 초기광응답
        # ----------------------------------------------------
        probe_stats = 균일광_초기측정_13(
            cap,
            window_name,
            monitor,
            args,
            rect,
            run_dir
            / "00_초기광응답",
        )

        기본광_전략 = 상대광응답_기본광전략_결정(
            probe_stats
        )

        (
            run_dir
            / "00_초기광응답"
            / "상대광응답_기본광전략.json"
        ).write_text(
            json.dumps(
                기본광_전략,
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

        # ----------------------------------------------------
        # 3) 기본 4위상 G16/E156
        # ----------------------------------------------------
        captures = []
        qualities = []
        conditions = []
        actual_log = []
        progress_rows = []

        base_result = 조건_추가촬영(
            cap,
            window_name,
            coord,
            monitor,
            args,
            run_dir,
            rect,
            preview,
            captures,
            qualities,
            conditions,
            시작_게인_13,
            시작_노출_13,
        )

        if base_result is not None:
            actual_log.append(
                base_result
            )

        (
            fusion,
            fusion_quality,
            fused_valid,
        ) = 현재융합_계산(
            captures,
            qualities,
            conditions,
            rect,
            args,
        )

        base_quality = qualities[
            0
        ]

        initial_direction = 초기방향_판정_13(
            probe_stats,
            base_quality,
        )

        print("")
        print(
            f"초기 자동 판정: "
            f"{initial_direction}"
        )

        # ----------------------------------------------------
        # P100 G/E 탐색 방향 정책
        # - 상대광응답이 '어두움'으로 분류된 저반응 표면은
        #   기존처럼 높은 G/E까지 올라간 뒤 크게 회복되는 경우가 있으므로
        #   기존 고정 상승 경로를 그대로 유지한다.
        # - 그 외 표면은 초기 판정을 첫 방향 힌트로만 사용하고,
        #   이후에는 현재 촬영의 포화/저변조도/융합 개선량으로 방향을 바꾼다.
        # - 색 이름(--sample)은 판단에 사용하지 않는다.
        # ----------------------------------------------------
        저반응_GE고정상승_예외_13 = (
            기본광_전략.get("분류") == "어두움"
        )

        if 저반응_GE고정상승_예외_13:
            print(
                "→ 저반응 예외: 초기 개선량이 작아도 기존처럼 "
                "높은 G/E 방향 탐색을 유지합니다."
            )
            print(
                "→ 색 이름이 아니라 상대광응답 '어두움' 분기를 기준으로 합니다."
            )
        else:
            print(
                "→ 일반 적응형: 초기 판정은 첫 추가조건의 방향만 정합니다."
            )
            print(
                f"→ 이후 개선량이 {정체_개선량_13:.2f}%p 미만이면 "
                "기존 방향 큐를 버리고 반대 방향 후보를 우선 탐색합니다."
            )
            print(
                "→ 포화 우세면 G/E↓, 저변조도 우세면 G/E↑로 계속 재판단합니다."
            )

        progress_rows.append(
            {
                "순서": 1,
                "Gain": 시작_게인_13,
                "Exposure": 시작_노출_13,
                "선택이유": "기본 조건",
                "개별유효율": (
                    base_quality[
                        "유효_위상_비율"
                    ]
                ),
                "포화율": (
                    base_quality[
                        "포화_비율"
                    ]
                ),
                "저변조도율": (
                    base_quality[
                        "저변조도_비율"
                    ]
                ),
                "평균M": (
                    base_quality[
                        "평균_변조도"
                    ]
                ),
                "융합유효율": (
                    fused_valid
                ),
                "개선량": (
                    fused_valid
                ),
            }
        )

        print(
            f"기본 융합 유효율: "
            f"{fused_valid:.2f}%"
        )

        # ----------------------------------------------------
        # 4) 유동형 2D 탐색
        # ----------------------------------------------------
        visited = {
            후보키_13(
                시작_게인_13,
                시작_노출_13,
            )
        }

        queue = []
        queued = set()

        시작후보_구성_13(
            initial_direction,
            queue,
            queued,
            visited,
            gains,
            exposures,
        )

        # 저반응 예외가 아닌 밝음/어두움 판정은
        # 미리 만든 한 방향 전체 경로를 따라가지 않고 첫 후보 1개만 사용한다.
        # 첫 추가 촬영 뒤부터는 현재 품질을 보고 다음 방향을 다시 만든다.
        if (
            (not 저반응_GE고정상승_예외_13)
            and initial_direction in ("밝음", "어두움")
            and queue
        ):
            queue.sort(
                key=lambda item: (
                    item["priority"],
                    abs(item["gain"] - 시작_게인_13),
                    abs(item["exposure"] - 시작_노출_13),
                )
            )
            first_candidate = dict(queue[0])
            queue[:] = [first_candidate]
            queued.clear()
            queued.add(
                후보키_13(
                    first_candidate["gain"],
                    first_candidate["exposure"],
                )
            )

        low_improve_streak = 0
        sequence = 1

        while (
            fused_valid
            < 목표_유효율_13
            and len(conditions)
            < 최대_촬영조건수_13
            and queue
        ):
            candidate = 큐꺼내기_13(
                queue,
                queued,
            )

            key = 후보키_13(
                candidate["gain"],
                candidate["exposure"],
            )

            if key in visited:
                continue

            visited.add(
                key
            )

            previous_fused = (
                fused_valid
            )

            print("")
            print("=" * 78)
            print(
                f"자동 선택 후보 | "
                f"Gain={candidate['gain']} | "
                f"Exposure={candidate['exposure']}"
            )
            print(
                f"선택 이유: "
                f"{candidate['reason']}"
            )
            print("=" * 78)

            result = 조건_추가촬영(
                cap,
                window_name,
                coord,
                monitor,
                args,
                run_dir,
                rect,
                preview,
                captures,
                qualities,
                conditions,
                candidate["gain"],
                candidate["exposure"],
            )

            if result is None:
                continue

            actual_log.append(
                result
            )

            (
                fusion,
                fusion_quality,
                fused_valid,
            ) = 현재융합_계산(
                captures,
                qualities,
                conditions,
                rect,
                args,
            )

            q = qualities[
                -1
            ]

            improvement = float(
                fused_valid
                - previous_fused
            )

            sequence += 1

            progress_rows.append(
                {
                    "순서": sequence,
                    "Gain": (
                        candidate[
                            "gain"
                        ]
                    ),
                    "Exposure": (
                        candidate[
                            "exposure"
                        ]
                    ),
                    "선택이유": (
                        candidate[
                            "reason"
                        ]
                    ),
                    "개별유효율": (
                        q[
                            "유효_위상_비율"
                        ]
                    ),
                    "포화율": (
                        q[
                            "포화_비율"
                        ]
                    ),
                    "저변조도율": (
                        q[
                            "저변조도_비율"
                        ]
                    ),
                    "평균M": (
                        q[
                            "평균_변조도"
                        ]
                    ),
                    "융합유효율": (
                        fused_valid
                    ),
                    "개선량": (
                        improvement
                    ),
                }
            )

            print(
                f"→ 융합 유효율 "
                f"{fused_valid:.2f}% "
                f"(+{improvement:.2f}%p)"
            )

            if (
                fused_valid
                >= 목표_유효율_13
            ):
                print(
                    "목표 95% 달성 → 즉시 종료."
                )
                break

            # 저반응 예외는 기존 고정 상승 경로를 그대로 유지한다.
            # 그 외 표면은 매 촬영 결과로 다음 G/E 방향을 다시 결정한다.
            if not 저반응_GE고정상승_예외_13:
                if improvement < 정체_개선량_13:
                    # 이전 방향에서 실질적인 회복이 없으면
                    # 미리 남아 있던 한 방향 후보를 버리고 현재점에서 재탐색한다.
                    queue.clear()
                    queued.clear()
                    low_improve_streak += 1

                    print(
                        f"→ 개선량 {improvement:.2f}%p < "
                        f"{정체_개선량_13:.2f}%p: "
                        "기존 방향 큐 제거 후 반대/주변 방향 재탐색"
                    )
                else:
                    low_improve_streak = 0

                주변후보_생성_13(
                    conditions[-1],
                    q,
                    improvement,
                    queue,
                    queued,
                    visited,
                    gains,
                    exposures,
                )

                if low_improve_streak >= 정체_연속횟수_13:
                    print("")
                    print(
                        f"정체 감지: "
                        f"{정체_연속횟수_13}회 연속 "
                        f"{정체_개선량_13:.2f}%p 미만 개선"
                    )
                    print(
                        "→ 2단계 이상 건너뛰는 2D 탈출은 사용하지 않습니다."
                    )
                    print(
                        "→ 현재 조건에서 한 단계 떨어진 반대/주변 후보를 먼저 소진합니다."
                    )

                    # 주변후보_생성_13()이 이미 현재 조건 기준
                    # Gain/Exposure ±1단계 후보를 넣어 둔 상태다.
                    # 예: G32/E20에서 낮은 방향을 확인할 때
                    # 바로 한 단계 아래인 G16/E20을 먼저 확인한다.
                    low_improve_streak = 0

        # ----------------------------------------------------
        # 5) 큐가 비었는데 아직 95% 미만이면
        #    전체 범위의 극단 조합을 마지막으로 빠르게 확인
        # ----------------------------------------------------
        if (
            (not 저반응_GE고정상승_예외_13)
            and fused_valid
            < 목표_유효율_13
            and len(conditions)
            < 최대_촬영조건수_13
        ):
            print("")
            print("=" * 78)
            print(
                "일반 적응형 후보가 소진되었으나 "
                "95% 미만 → 마지막 극단 조합 확인"
            )
            print("=" * 78)

            last_resort = [
                (
                    gains[0],
                    exposures[0],
                    "최종 탈출: 최소Gain/최소Exposure",
                ),
                (
                    gains[-1],
                    exposures[0],
                    "최종 탈출: 최대Gain/최소Exposure",
                ),
                (
                    gains[0],
                    exposures[-1],
                    "최종 탈출: 최소Gain/최대Exposure",
                ),
                (
                    gains[-1],
                    exposures[-1],
                    "최종 탈출: 최대Gain/최대Exposure",
                ),
            ]

            for gain, exposure, reason in last_resort:
                if (
                    fused_valid
                    >= 목표_유효율_13
                ):
                    break

                if (
                    len(conditions)
                    >= 최대_촬영조건수_13
                ):
                    break

                key = 후보키_13(
                    gain,
                    exposure,
                )

                if key in visited:
                    continue

                visited.add(
                    key
                )

                previous_fused = (
                    fused_valid
                )

                result = 조건_추가촬영(
                    cap,
                    window_name,
                    coord,
                    monitor,
                    args,
                    run_dir,
                    rect,
                    preview,
                    captures,
                    qualities,
                    conditions,
                    gain,
                    exposure,
                )

                if result is None:
                    continue

                actual_log.append(
                    result
                )

                (
                    fusion,
                    fusion_quality,
                    fused_valid,
                ) = 현재융합_계산(
                    captures,
                    qualities,
                    conditions,
                    rect,
                    args,
                )

                q = qualities[
                    -1
                ]

                improvement = float(
                    fused_valid
                    - previous_fused
                )

                sequence += 1

                progress_rows.append(
                    {
                        "순서": sequence,
                        "Gain": gain,
                        "Exposure": exposure,
                        "선택이유": reason,
                        "개별유효율": (
                            q[
                                "유효_위상_비율"
                            ]
                        ),
                        "포화율": (
                            q[
                                "포화_비율"
                            ]
                        ),
                        "저변조도율": (
                            q[
                                "저변조도_비율"
                            ]
                        ),
                        "평균M": (
                            q[
                                "평균_변조도"
                            ]
                        ),
                        "융합유효율": (
                            fused_valid
                        ),
                        "개선량": (
                            improvement
                        ),
                    }
                )

                print(
                    f"최종 탈출 후보 "
                    f"G{gain}/E{exposure} "
                    f"→ 융합 "
                    f"{fused_valid:.2f}% "
                    f"(+{improvement:.2f}%p)"
                )

        if (
            initial_direction in ("밝음", "어두움")
            and fused_valid < 목표_유효율_13
            and not queue
        ):
            print("")
            print("=" * 78)
            print(
                f"{initial_direction} 고정 탐색 경로를 모두 확인했지만 "
                f"융합 유효율은 {fused_valid:.2f}%입니다."
            )
            print(
                "반대 방향 후보를 무한히 추가하지 않고 여기서 종료합니다."
            )
            print("=" * 78)

        # ----------------------------------------------------
        # 6) 최종 저장
        # ----------------------------------------------------
        actual_settings = {
            "초기광응답": (
                probe_stats
            ),
            "초기판정": (
                initial_direction
            ),
            "상대광응답_기본광전략": (
                기본광_전략
            ),
            "조건별_실제설정": (
                actual_log
            ),
            "목표유효율": (
                목표_유효율_13
            ),
            "최대촬영조건수": (
                최대_촬영조건수_13
            ),
            "노출앵커": (
                exposures
            ),
            "게인앵커": (
                gains
            ),
        }

        summary, condition_rows, result_dir = (
            융합결과_저장(
                run_dir,
                fusion,
                fusion_quality,
                qualities,
                conditions,
                0,
                rect,
                args,
                actual_settings,
                monitor,
            )
        )

        # ----------------------------------------------------
        # 7) 상대광응답 기반 기본광 제어 + 최종 후처리
        #    - 밝음: 기본광 밝기 제어 없이 P100 P100-HDR 결과 사용
        #    - 중간: 상대광응답 기준 P-2 / P / P+2 세 조건
        #    - 어두움: P100 고정, G64/G96/G128 + E1400 세 조건
        #    이후 Reference 180도 → 문제영역 → 기존 HDR 원리 교체
        #    → Quality-guided 2π → 플랫폼 기준면 PLY까지 자동 실행
        # ----------------------------------------------------
        최종통합_결과 = 최종통합_후처리_실행(
            cap=cap,
            window_name=window_name,
            coord=coord,
            monitor=monitor,
            args=args,
            run_dir=run_dir,
            depth_result_dir=depth_result_dir,
            rect=rect,
            preview=preview,
            probe_stats=probe_stats,
            기본광_전략=기본광_전략,
            baseline_result_dir=result_dir,
            baseline_fusion=fusion,
            baseline_fusion_quality=fusion_quality,
            baseline_conditions=conditions,
            baseline_captures=captures,
            baseline_qualities=qualities,
        )

        progress_csv = (
            run_dir
            / "유동형95_탐색진행표.csv"
        )

        with progress_csv.open(
            "w",
            newline="",
            encoding="utf-8-sig",
        ) as f:
            writer = csv.DictWriter(
                f,
                fieldnames=list(
                    progress_rows[0].keys()
                ),
            )
            writer.writeheader()
            writer.writerows(
                progress_rows
            )

        reached = (
            fused_valid
            >= 목표_유효율_13
        )

        report_lines = [
            (
                f"{sample_name} "
                "RGB 초기광응답 + 유동형 2축 탐색 결과"
            ),
            "=" * 78,
            "",
            (
                f"P100 G/E 초기판정: "
                f"{initial_direction}"
            ),
            (
                f"상대광응답(흰색=100): "
                f"{기본광_전략['상대광응답_흰색100']:.2f}"
            ),
            (
                f"기본광 제어 분기: "
                f"{기본광_전략['분류']}"
            ),
            (
                f"기본광 후보 설명: "
                f"{기본광_전략['설명']}"
            ),
            (
                f"목표 유효율: "
                f"{목표_유효율_13:.2f}%"
            ),
            (
                f"최종 융합 유효율: "
                f"{fused_valid:.2f}%"
            ),
            (
                f"목표 달성 여부: "
                f"{'달성' if reached else '미달성'}"
            ),
            (
                f"총 촬영 조건 수: "
                f"{len(conditions)}"
            ),
            "",
            "초기 광응답",
        ]

        for key, value in probe_stats.items():
            report_lines.append(
                f"{key}: {value}"
            )

        report_lines.extend(
            [
                "",
                "촬영 조건 순서",
            ]
        )

        for row in progress_rows:
            report_lines.append(
                (
                    f"{row['순서']:02d}. "
                    f"G{row['Gain']} / "
                    f"E{row['Exposure']} | "
                    f"개별 {row['개별유효율']:.2f}% | "
                    f"융합 {row['융합유효율']:.2f}% | "
                    f"개선 +{row['개선량']:.2f}%p | "
                    f"{row['선택이유']}"
                )
            )

        report_lines.extend(
            [
                "",
                (
                    "※ 이 알고리즘은 색 이름을 분류하지 않고, "
                    "RGB 균일광 반응 + 구조광 품질 피드백으로 "
                    "Exposure/Gain 방향을 계속 바꿔가며 탐색함."
                ),
                (
                    "※ 95%는 현재 M15 + 비포화 + 비암부 기준의 "
                    "유효 픽셀 비율이며 3D 정확도 95%를 의미하지 않음."
                ),
                (
                    "※ 후보 범위를 크게 움직이는 고속 탐색 단계이므로, "
                    "나중에 최종 장비 조건이 확정되면 주변 값을 더 촘촘히 "
                    "정밀화할 수 있음."
                ),
                (
                    "※ 모든 방향과 극단 조합에서도 95%가 안 되면 "
                    "Gain/Exposure 외의 광학적 원인 "
                    "(정반사, 가림, 투사 사각, 패턴 왜곡 등)을 의심해야 함."
                ),
                "",
                (
                    f"최종 전처리 결과: "
                    f"{result_dir}"
                ),
                (
                    f"최종 후처리 루트: "
                    f"{최종통합_결과['후처리_루트']}"
                ),
                (
                    f"최종 PLY: "
                    f"{최종통합_결과['최종_PLY']}"
                ),
            ]
        )

        report_path = (
            run_dir
            / "유동형95_최종요약.txt"
        )

        report_path.write_text(
            "\n".join(
                report_lines
            ),
            encoding="utf-8",
        )

        print("")
        print("=" * 78)
        print(
            "유동형 자동 탐색 완료"
        )
        print(
            f"초기 판정: "
            f"{initial_direction}"
        )
        print(
            f"최종 융합 유효율: "
            f"{fused_valid:.2f}%"
        )
        print(
            f"목표 달성: "
            f"{'예' if reached else '아니오'}"
        )
        print(
            f"총 촬영 조건: "
            f"{len(conditions)}개"
        )
        print(
            f"최종 4위상 결과: "
            f"{result_dir}"
        )
        print(
            f"진행표: "
            f"{progress_csv}"
        )
        print(
            f"요약: "
            f"{report_path}"
        )
        print(
            f"최종 후처리 루트: "
            f"{최종통합_결과['후처리_루트']}"
        )
        print(
            f"최종 CloudCompare PLY: "
            f"{최종통합_결과['최종_PLY']}"
        )
        print("=" * 78)

    except KeyboardInterrupt:
        print(
            "\n사용자 요청으로 종료했습니다."
        )

    finally:
        if cap is not None:
            cap.release()

        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()