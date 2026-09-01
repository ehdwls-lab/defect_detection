#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
import json
import re
import subprocess

import cv2
import numpy as np

from pyorbbecsdk import (
    Config,
    OBFormat,
    OBSensorType,
    Pipeline,
)


# ============================================================
# 저장 위치
# ============================================================

프로젝트_폴더 = Path(
    "/home/seoyeong/졸업작품/전처리와구조광_통합/프로젝터 수동 범위 확인"
)

# 기존 Depth 코드와 경로 호환을 위해 파일 이름은 그대로 유지
프로젝터_범위_JSON = (
    프로젝트_폴더 / "프로젝터_세로범위.json"
)

선택용_촬영이미지_경로 = (
    프로젝트_폴더 / "프로젝터_XY범위_RGB255_수동선택용.png"
)

선택결과_확인이미지_경로 = (
    프로젝트_폴더 / "프로젝터_XY범위_수동2점_확인.png"
)


# ============================================================
# 설정
# ============================================================

선택용_프로젝터_RGB = 255

워밍업_프레임수 = 20
패턴_안정화용_버릴프레임수 = 15
촬영_평균프레임수 = 5
프레임_대기시간_ms = 1000

# 직접 안전한 안쪽을 클릭할 것이므로 기본 여유는 0
빔_경계_안쪽여유_px = 0


# ============================================================
# Orbbec Color
# ============================================================

def 컬러프레임_BGR로_변환(color_frame):
    width = color_frame.get_width()
    height = color_frame.get_height()

    data = np.frombuffer(
        color_frame.get_data(),
        dtype=np.uint8,
    )

    expected = width * height * 3

    if data.size != expected:
        raise RuntimeError(
            "Color 프레임이 RGB 3채널 형식이 아닙니다. "
            f"현재 데이터 크기={data.size}, 예상={expected}"
        )

    rgb = data.reshape(
        (height, width, 3)
    )

    return cv2.cvtColor(
        rgb,
        cv2.COLOR_RGB2BGR,
    )


def Orbbec_컬러파이프라인_열기():
    pipeline = Pipeline()
    config = Config()

    color_profiles = (
        pipeline.get_stream_profile_list(
            OBSensorType.COLOR_SENSOR
        )
    )

    color_profile = (
        color_profiles.get_video_stream_profile(
            0,
            0,
            OBFormat.RGB,
            0,
        )
    )

    config.enable_stream(
        color_profile
    )

    pipeline.start(
        config
    )

    for _ in range(
        워밍업_프레임수
    ):
        pipeline.wait_for_frames(
            프레임_대기시간_ms
        )

    return pipeline


def Color_평균촬영(pipeline):
    frames = []

    while len(frames) < 촬영_평균프레임수:
        frameset = pipeline.wait_for_frames(
            프레임_대기시간_ms
        )

        if not frameset:
            continue

        color_frame = frameset.get_color_frame()

        if color_frame is None:
            continue

        frame = 컬러프레임_BGR로_변환(
            color_frame
        )

        frames.append(
            frame.astype(np.float32)
        )

    averaged = np.mean(
        frames,
        axis=0,
    )

    return np.clip(
        averaged,
        0,
        255,
    ).astype(np.uint8)


# ============================================================
# 프로젝터
# ============================================================

def xrandr_모니터_목록():
    result = subprocess.run(
        ["xrandr", "--query"],
        capture_output=True,
        text=True,
        check=True,
    )

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


def 프로젝터_화면_자동선택():
    monitors = xrandr_모니터_목록()

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

    print(
        f"프로젝터: {selected['name']} | "
        f"{selected['w']}×{selected['h']} | "
        f"({selected['x']}, {selected['y']})"
    )

    return selected


def 프로젝터_RGB255_켜기(monitor):
    window_name = "PROJECTOR_RGB255"

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
        선택용_프로젝터_RGB,
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

    return window_name


def 프로젝터_검정화면(
    window_name,
    monitor,
):
    if (
        window_name is None
        or monitor is None
    ):
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
# X/Y 사용범위 수동 2점 선택
# ============================================================

def XY_2점_수동선택(image):
    """
    180도 회전된 Color 영상에서:

    1번째 클릭 = 사용영역의 왼쪽 위
    2번째 클릭 = 사용영역의 오른쪽 아래

    클릭 순서를 반대로 해도 X/Y를 자동 정렬함.

    S = 저장
    R = 다시 선택
    Q / ESC = 취소
    """

    window_name = "PROJECTOR_XY_RANGE_SELECT"

    points = []

    def mouse_callback(
        event,
        x,
        y,
        flags,
        param,
    ):
        nonlocal points

        if (
            event == cv2.EVENT_LBUTTONDOWN
            and len(points) < 2
        ):
            points.append(
                (int(x), int(y))
            )

            if len(points) == 1:
                print(
                    f"1번째 점 선택: X={x}, Y={y}"
                )
            else:
                print(
                    f"2번째 점 선택: X={x}, Y={y}"
                )

    cv2.namedWindow(
        window_name,
        cv2.WINDOW_NORMAL,
    )

    initial = image.copy()

    cv2.putText(
        initial,
        "1) TOP-LEFT   2) BOTTOM-RIGHT",
        (30, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.85,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    cv2.putText(
        initial,
        "S: save   R: reset   Q/ESC: quit",
        (30, 75),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.70,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    cv2.imshow(
        window_name,
        initial,
    )

    cv2.waitKey(300)

    cv2.moveWindow(
        window_name,
        100,
        80,
    )

    cv2.waitKey(200)

    cv2.setMouseCallback(
        window_name,
        mouse_callback,
    )

    while True:
        preview = image.copy()

        cv2.putText(
            preview,
            "1) TOP-LEFT   2) BOTTOM-RIGHT",
            (30, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.85,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        cv2.putText(
            preview,
            "S: save   R: reset   Q/ESC: quit",
            (30, 75),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.70,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        if len(points) >= 1:
            cv2.circle(
                preview,
                points[0],
                7,
                (0, 0, 255),
                -1,
            )

            cv2.putText(
                preview,
                f"P1 = ({points[0][0]}, {points[0][1]})",
                (
                    30,
                    110,
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.70,
                (0, 255, 255),
                2,
                cv2.LINE_AA,
            )

        if len(points) == 2:
            cv2.circle(
                preview,
                points[1],
                7,
                (0, 0, 255),
                -1,
            )

            x1_raw, y1_raw = points[0]
            x2_raw, y2_raw = points[1]

            safe_left = min(
                x1_raw,
                x2_raw,
            ) + int(
                빔_경계_안쪽여유_px
            )

            safe_right = max(
                x1_raw,
                x2_raw,
            ) - int(
                빔_경계_안쪽여유_px
            )

            safe_top = min(
                y1_raw,
                y2_raw,
            ) + int(
                빔_경계_안쪽여유_px
            )

            safe_bottom = max(
                y1_raw,
                y2_raw,
            ) - int(
                빔_경계_안쪽여유_px
            )

            cv2.rectangle(
                preview,
                (
                    safe_left,
                    safe_top,
                ),
                (
                    safe_right,
                    safe_bottom,
                ),
                (0, 255, 0),
                3,
            )

            cv2.putText(
                preview,
                (
                    f"SAFE X={safe_left}~{safe_right}  "
                    f"Y={safe_top}~{safe_bottom}"
                ),
                (
                    30,
                    145,
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.70,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )

        cv2.imshow(
            window_name,
            preview,
        )

        key = cv2.waitKey(20) & 0xFF

        if key in (
            ord("r"),
            ord("R"),
        ):
            points = []

            print("")
            print(
                "선택 초기화. "
                "왼쪽 위 → 오른쪽 아래 순서로 다시 클릭하세요."
            )

        elif key in (
            ord("q"),
            ord("Q"),
            27,
        ):
            cv2.destroyWindow(
                window_name
            )

            raise KeyboardInterrupt(
                "사용자가 선택을 취소했습니다."
            )

        elif (
            key in (
                ord("s"),
                ord("S"),
            )
            and len(points) == 2
        ):
            x1_raw, y1_raw = points[0]
            x2_raw, y2_raw = points[1]

            safe_left = min(
                x1_raw,
                x2_raw,
            ) + int(
                빔_경계_안쪽여유_px
            )

            safe_right = max(
                x1_raw,
                x2_raw,
            ) - int(
                빔_경계_안쪽여유_px
            )

            safe_top = min(
                y1_raw,
                y2_raw,
            ) + int(
                빔_경계_안쪽여유_px
            )

            safe_bottom = max(
                y1_raw,
                y2_raw,
            ) - int(
                빔_경계_안쪽여유_px
            )

            if safe_right <= safe_left:
                print(
                    "좌/우 범위가 잘못되었습니다. "
                    "R을 눌러 다시 선택하세요."
                )
                continue

            if safe_bottom <= safe_top:
                print(
                    "상/하 범위가 잘못되었습니다. "
                    "R을 눌러 다시 선택하세요."
                )
                continue

            final_preview = image.copy()

            cv2.rectangle(
                final_preview,
                (
                    safe_left,
                    safe_top,
                ),
                (
                    safe_right,
                    safe_bottom,
                ),
                (0, 255, 0),
                3,
            )

            cv2.putText(
                final_preview,
                (
                    f"SAFE X={safe_left}~{safe_right}  "
                    f"Y={safe_top}~{safe_bottom}"
                ),
                (30, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.80,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )

            cv2.imwrite(
                str(
                    선택결과_확인이미지_경로
                ),
                final_preview,
            )

            cv2.destroyWindow(
                window_name
            )

            return {
                "첫번째_클릭점": [
                    int(points[0][0]),
                    int(points[0][1]),
                ],
                "두번째_클릭점": [
                    int(points[1][0]),
                    int(points[1][1]),
                ],

                # 새 X/Y 제한
                "안전_왼쪽": int(
                    safe_left
                ),
                "안전_오른쪽": int(
                    safe_right
                ),
                "안전_위": int(
                    safe_top
                ),
                "안전_아래": int(
                    safe_bottom
                ),

                # 기존 Y 전용 코드와의 호환을 위해 유지
                "왼쪽_위": int(
                    safe_top
                ),
                "오른쪽_위": int(
                    safe_top
                ),
                "왼쪽_아래": int(
                    safe_bottom
                ),
                "오른쪽_아래": int(
                    safe_bottom
                ),
            }


# ============================================================
# main
# ============================================================

def main():
    프로젝트_폴더.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 72)
    print("프로젝터 X/Y 사용범위 수동 2점 지정")
    print("=" * 72)
    print("")
    print(
        "카메라/프로젝터 위치를 바꿀 때만 "
        "다시 실행하면 됩니다."
    )
    print("")
    print(
        "RGB255를 투사한 카메라 영상을 "
        "180도 회전해서 보여줍니다."
    )
    print("")
    print(
        "프로젝터 패턴을 실제로 사용해도 안전한 "
        "사각 범위를 직접 지정합니다."
    )
    print("")
    print("마우스로 딱 2번 클릭하세요.")
    print("")
    print("1. 사용할 영역의 왼쪽 위")
    print("2. 사용할 영역의 오른쪽 아래")
    print("")
    print(
        "암실 벽이나 프로젝터 빛이 약한 부분이 "
        "사각형 밖에 있도록 조금 안쪽으로 클릭하세요."
    )
    print("")
    print("S = 저장")
    print("R = 다시 선택")
    print("Q 또는 ESC = 취소")
    print("")

    input(
        "준비되면 Enter를 누르세요..."
    )

    projector_monitor = None
    projector_window = None
    pipeline = None

    try:
        projector_monitor = (
            프로젝터_화면_자동선택()
        )

        projector_window = (
            프로젝터_RGB255_켜기(
                projector_monitor
            )
        )

        pipeline = (
            Orbbec_컬러파이프라인_열기()
        )

        for _ in range(
            패턴_안정화용_버릴프레임수
        ):
            pipeline.wait_for_frames(
                프레임_대기시간_ms
            )

        color = Color_평균촬영(
            pipeline
        )

        # 현재 구조광 코드와 동일하게 180도 회전
        rotated = cv2.rotate(
            color,
            cv2.ROTATE_180,
        )

        cv2.imwrite(
            str(
                선택용_촬영이미지_경로
            ),
            rotated,
        )

        print("")
        print("선택 창을 띄웁니다.")
        print(
            "왼쪽 위 → 오른쪽 아래 순서로 "
            "클릭하세요."
        )
        print("")

        info = XY_2점_수동선택(
            rotated
        )

        data = {
            "좌표기준": (
                "pyorbbecsdk Color / 180도 회전 후"
            ),
            "설정방식": (
                "수동 2점 X/Y 사각범위 선택"
            ),
            "프로젝터_RGB": int(
                선택용_프로젝터_RGB
            ),
            "경계_안쪽여유_px": int(
                빔_경계_안쪽여유_px
            ),
            "카메라해상도": [
                int(
                    rotated.shape[1]
                ),
                int(
                    rotated.shape[0]
                ),
            ],
            **info,
        }

        프로젝터_범위_JSON.write_text(
            json.dumps(
                data,
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

        print("")
        print("=" * 72)
        print("프로젝터 X/Y 사용범위 저장 완료")
        print("=" * 72)

        print(
            f"X 범위: "
            f"{info['안전_왼쪽']} ~ "
            f"{info['안전_오른쪽']}"
        )

        print(
            f"Y 범위: "
            f"{info['안전_위']} ~ "
            f"{info['안전_아래']}"
        )

        print("")
        print(
            f"JSON: "
            f"{프로젝터_범위_JSON}"
        )

        print(
            f"확인 이미지: "
            f"{선택결과_확인이미지_경로}"
        )

        print("=" * 72)

    finally:
        if pipeline is not None:
            try:
                pipeline.stop()
            except Exception:
                pass

        try:
            프로젝터_검정화면(
                projector_window,
                projector_monitor,
            )
        except Exception:
            pass

        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
