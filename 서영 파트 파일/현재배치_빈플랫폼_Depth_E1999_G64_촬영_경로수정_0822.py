#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
현재배치 빈 플랫폼 기준 Depth E1999/G64 촬영

- Gemini 336L
- Depth E1999 / G64 고정
- 현재 구조광 통합 코드와 동일 계열 Color+Depth Pipeline
- 현재 구조광 Depth 영역검출과 동일한 RGB255 프로젝터 균일광
- D2C(Depth -> Color) 정렬
- 20프레임 워밍업 후 15프레임 중앙값

최종 저장 위치는 structured_light_paths.DEPTH_CALIBRATION_DIR 아래이다.

기존 파일이 있으면 덮어쓰기 전에 자동 백업함.
마우스 필요 없음. 물체를 치우고 Enter만 누르면 됨.
"""

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

from structured_light_paths import DEPTH_CALIBRATION_DIR, PLATFORM_ROOT, ROOT

try:
    from pyorbbecsdk import OBFrameAggregateOutputMode
except ImportError:
    OBFrameAggregateOutputMode = None


# ============================================================
# 현재 검은색 Depth 2D E/G 테스트에서 선택한 기준
# ============================================================

프로젝트_폴더 = ROOT
플랫폼_바닥_폴더 = PLATFORM_ROOT

현재배치_기준_폴더 = DEPTH_CALIBRATION_DIR
현재배치_기준_폴더.mkdir(parents=True, exist_ok=True)

저장_NPY = 현재배치_기준_폴더 / "플랫폼_바닥_depth.npy"
저장_RAW_PNG = 현재배치_기준_폴더 / "플랫폼_바닥_depth.png"
저장_VIS_PNG = 현재배치_기준_폴더 / "플랫폼_바닥_depth_시각화.png"
저장_TXT = 현재배치_기준_폴더 / "플랫폼_바닥_depth_정보.txt"

Depth_노출 = 1999
Depth_게인 = 64

워밍업_프레임수 = 20
Depth_프레임수 = 15
프레임_대기시간_ms = 1000

Color_너비 = 1280
Color_높이 = 800
Color_요청_FPS = 8

프로젝터_RGB = 255


# ============================================================
# 공통 함수
# ============================================================

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

    import re
    pattern = re.compile(
        r"^(?P<name>\S+)\s+connected(?:\s+primary)?\s+"
        r"(?P<w>\d+)x(?P<h>\d+)\+(?P<x>-?\d+)\+(?P<y>-?\d+)"
    )

    monitors = []
    for line in result.stdout.splitlines():
        match = pattern.search(line)
        if not match:
            continue
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


def 프로젝터_화면_자동선택():
    monitors = xrandr_모니터_목록()
    if not monitors:
        raise RuntimeError("xrandr에서 모니터 목록을 읽지 못했습니다.")

    selected = next(
        (m for m in monitors if "HDMI" in m["name"].upper()),
        None,
    )

    if selected is None:
        selected = next((m for m in monitors if not m["primary"]), None)

    if selected is None:
        raise RuntimeError("프로젝터로 사용할 보조 화면을 찾지 못했습니다.")

    print(
        f"프로젝터 화면: {selected['name']} | "
        f"{selected['w']}x{selected['h']} | "
        f"위치 ({selected['x']}, {selected['y']})"
    )
    return selected


def 프로젝터_균일광_켜기(monitor, rgb_value):
    window_name = "E480_G16_빈플랫폼_Depth_촬영용_RGB255"

    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    image = np.full(
        (monitor["h"], monitor["w"], 3),
        int(rgb_value),
        dtype=np.uint8,
    )

    cv2.imshow(window_name, image)
    cv2.waitKey(300)
    cv2.moveWindow(window_name, monitor["x"], monitor["y"])
    cv2.waitKey(300)

    cv2.setWindowProperty(
        window_name,
        cv2.WND_PROP_FULLSCREEN,
        cv2.WINDOW_FULLSCREEN,
    )

    cv2.imshow(window_name, image)
    cv2.waitKey(1000)

    print(f"프로젝터 균일광 투사: RGB={rgb_value}")
    return window_name


def 프로젝터_검정화면(window_name, monitor):
    if window_name is None:
        return

    black = np.zeros(
        (monitor["h"], monitor["w"], 3),
        dtype=np.uint8,
    )
    cv2.imshow(window_name, black)
    cv2.waitKey(300)


def Depth프레임_mm로_변환(depth_frame):
    width = depth_frame.get_width()
    height = depth_frame.get_height()

    raw = np.frombuffer(
        depth_frame.get_data(),
        dtype=np.uint16,
    ).reshape((height, width))

    scale = float(depth_frame.get_depth_scale())
    return raw.astype(np.float32) * scale


def 여러Depth_중앙값(depth_frames):
    stack = np.stack(depth_frames, axis=0).astype(np.float32)
    stack[stack <= 0] = np.nan

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        median = np.nanmedian(stack, axis=0)

    return np.nan_to_num(median, nan=0.0).astype(np.float32)


def Color_프로파일_선택(pipeline):
    profiles = pipeline.get_stream_profile_list(OBSensorType.COLOR_SENSOR)

    fps_candidates = []
    for fps in (Color_요청_FPS, 10, 5, 15, 30, 60):
        if fps not in fps_candidates:
            fps_candidates.append(fps)

    errors = []
    for fps in fps_candidates:
        try:
            profile = profiles.get_video_stream_profile(
                Color_너비,
                Color_높이,
                OBFormat.MJPG,
                int(fps),
            )
            return profile, int(fps)
        except Exception as exc:
            errors.append(f"{fps}fps={exc}")

    raise RuntimeError(
        "1280x800 MJPG Color 프로파일 선택 실패 | " + " | ".join(errors)
    )


def Pipeline_열기():
    pipeline = Pipeline()
    config = Config()

    color_profile, color_fps = Color_프로파일_선택(pipeline)

    depth_profiles = pipeline.get_stream_profile_list(OBSensorType.DEPTH_SENSOR)
    depth_profile = depth_profiles.get_default_video_stream_profile()

    config.enable_stream(color_profile)
    config.enable_stream(depth_profile)

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

    pipeline.start(config)
    device = pipeline.get_device()

    print("")
    print("=" * 72)
    print("후보6용 빈 플랫폼 Depth 촬영")
    print("=" * 72)
    print(f"Color: {Color_너비}x{Color_높이} MJPG | FPS={color_fps}")

    try:
        print(
            f"Depth: {depth_profile.get_width()}x{depth_profile.get_height()} | "
            f"FPS={depth_profile.get_fps()} | "
            f"Format={depth_profile.get_format()}"
        )
    except Exception:
        pass

    return pipeline, device


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

    time.sleep(0.3)

    try:
        actual_e = int(
            device.get_int_property(OBPropertyID.OB_PROP_DEPTH_EXPOSURE_INT)
        )
    except Exception:
        actual_e = None

    try:
        actual_g = int(
            device.get_int_property(OBPropertyID.OB_PROP_DEPTH_GAIN_INT)
        )
    except Exception:
        actual_g = None

    print(
        f"Depth 요청 E{Depth_노출}/G{Depth_게인} | "
        f"실제 E{actual_e}/G{actual_g}"
    )

    if actual_e is not None and actual_e != Depth_노출:
        print("[주의] 실제 Exposure가 요청값과 다릅니다.")
    if actual_g is not None and actual_g != Depth_게인:
        print("[주의] 실제 Gain이 요청값과 다릅니다.")

    return actual_e, actual_g


def Depth_시각화(depth):
    valid = depth > 0
    vis = np.zeros(depth.shape, dtype=np.uint8)

    if np.any(valid):
        values = depth[valid]
        lo = float(np.percentile(values, 2))
        hi = float(np.percentile(values, 98))
        if hi <= lo:
            hi = lo + 1.0

        normalized = np.clip((depth - lo) / (hi - lo), 0.0, 1.0)
        vis[valid] = np.rint(normalized[valid] * 255.0).astype(np.uint8)

    color = cv2.applyColorMap(vis, cv2.COLORMAP_TURBO)
    color[~valid] = 0
    return color


def 기존파일_백업(path):
    if not path.exists():
        return None

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup = path.with_name(f"{path.stem}_백업_{stamp}{path.suffix}")
    path.rename(backup)
    print(f"기존 파일 백업: {backup}")
    return backup


# ============================================================
# 메인
# ============================================================

def main():
    플랫폼_바닥_폴더.mkdir(parents=True, exist_ok=True)

    print("")
    print("=" * 72)
    print("중요")
    print("=" * 72)
    print("플랫폼 위의 물체를 전부 치워주세요.")
    print("플랫폼/카메라/프로젝터 위치는 실제 후보6 실행 상태 그대로 두세요.")
    print(f"저장 위치: {저장_NPY}")
    print("")
    input("빈 플랫폼만 남았으면 Enter를 누르세요...")

    pipeline = None
    projector_monitor = None
    projector_window = None

    try:
        pipeline, device = Pipeline_열기()

        projector_monitor = 프로젝터_화면_자동선택()
        projector_window = 프로젝터_균일광_켜기(
            projector_monitor,
            프로젝터_RGB,
        )

        actual_e, actual_g = Depth_수동설정_적용(device)

        align_filter = AlignFilter(
            align_to_stream=OBStreamType.COLOR_STREAM
        )

        print("")
        print("Depth -> Color 정렬(D2C) 적용")
        print(f"설정 안정화용 {워밍업_프레임수}프레임 버리는 중...")

        for _ in range(워밍업_프레임수):
            pipeline.wait_for_frames(프레임_대기시간_ms)

        depth_frames = []

        while len(depth_frames) < Depth_프레임수:
            frames = pipeline.wait_for_frames(프레임_대기시간_ms)
            if not frames:
                continue

            aligned = align_filter.process(frames)
            if not aligned:
                continue

            color_frame = aligned.get_color_frame()
            depth_frame = aligned.get_depth_frame()

            if color_frame is None or depth_frame is None:
                continue

            depth_frames.append(Depth프레임_mm로_변환(depth_frame))

            print(
                f"Depth 수집 {len(depth_frames):02d}/{Depth_프레임수}",
                end="\r",
                flush=True,
            )

        print("")

        background_depth = 여러Depth_중앙값(depth_frames)
        valid = background_depth > 0

        valid_count = int(np.count_nonzero(valid))
        total_count = int(background_depth.size)
        valid_ratio = valid_count / max(1, total_count) * 100.0

        valid_values = background_depth[valid]
        median_mm = float(np.median(valid_values)) if valid_values.size else float("nan")
        p05_mm = float(np.percentile(valid_values, 5)) if valid_values.size else float("nan")
        p95_mm = float(np.percentile(valid_values, 95)) if valid_values.size else float("nan")

        # 기존 canonical 파일만 백업. PNG/TXT는 그냥 최신값으로 갱신.
        기존파일_백업(저장_NPY)

        np.save(
            저장_NPY,
            background_depth.astype(np.float32),
        )

        cv2.imwrite(
            str(저장_RAW_PNG),
            np.clip(
                np.rint(background_depth),
                0,
                65535,
            ).astype(np.uint16),
        )

        cv2.imwrite(
            str(저장_VIS_PNG),
            Depth_시각화(background_depth),
        )

        info = [
            "후보6용 빈 플랫폼 기준 Depth",
            "=" * 60,
            f"저장 시간: {datetime.now().isoformat(timespec='seconds')}",
            f"NPY: {저장_NPY}",
            f"Depth 요청: E{Depth_노출}/G{Depth_게인}",
            f"Depth 실제: E{actual_e}/G{actual_g}",
            f"워밍업 프레임: {워밍업_프레임수}",
            f"중앙값 생성 프레임: {Depth_프레임수}",
            "정렬: Depth -> Color (D2C)",
            f"프로젝터 균일광: RGB{프로젝터_RGB}",
            f"Depth shape: {background_depth.shape}",
            f"유효 Depth: {valid_count}/{total_count} ({valid_ratio:.2f}%)",
            f"유효 Depth P05: {p05_mm:.2f} mm",
            f"유효 Depth 중앙값: {median_mm:.2f} mm",
            f"유효 Depth P95: {p95_mm:.2f} mm",
        ]

        저장_TXT.write_text("\n".join(info), encoding="utf-8")

        print("")
        print("=" * 72)
        print("저장 완료")
        print("=" * 72)
        print(f"통합 코드에 연결할 기준 Depth: {저장_NPY}")
        print(f"Raw PNG: {저장_RAW_PNG}")
        print(f"확인용 시각화: {저장_VIS_PNG}")
        print(f"정보: {저장_TXT}")
        print(f"유효 Depth: {valid_ratio:.2f}%")
        print("")
        print("이제 물체를 다시 올리고 후보6을 실행하면 됩니다.")

    finally:
        if projector_window is not None and projector_monitor is not None:
            try:
                프로젝터_검정화면(projector_window, projector_monitor)
            except Exception:
                pass

        if pipeline is not None:
            try:
                pipeline.stop()
            except Exception:
                pass

        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
