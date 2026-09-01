from __future__ import annotations

import argparse
import csv
import re
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from pyorbbecsdk import (
    Config,
    OBFormat,
    OBPropertyID,
    OBSensorType,
    Pipeline,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]

WIDTH = 1280
HEIGHT = 800
FPS = 10
PIXEL_FORMAT = OBFormat.MJPG

FRAME_TIMEOUT_MS = 3000
DEFAULT_WARMUP_FRAMES = 60
DEFAULT_AVERAGE_FRAMES = 4


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Gemini 336L을 이용한 "
            "1280x800 MJPG 평균 영상 데이터 수집"
        )
    )

    parser.add_argument(
        "--category",
        choices=("normal", "defect"),
        required=True,
        help="수집할 데이터 종류",
    )

    parser.add_argument(
        "--session",
        type=str,
        required=True,
        help="촬영 세션 이름",
    )

    parser.add_argument(
        "--target-count",
        type=int,
        default=10,
        help="이번 실행에서 촬영할 이미지 수",
    )

    parser.add_argument(
        "--warmup-frames",
        type=int,
        default=DEFAULT_WARMUP_FRAMES,
        help="카메라 시작 후 버릴 워밍업 프레임 수",
    )

    parser.add_argument(
        "--average-frames",
        type=int,
        default=DEFAULT_AVERAGE_FRAMES,
        help="원본 한 장을 만들 때 평균할 프레임 수",
    )

    parser.add_argument(
        "--brightness",
        type=str,
        default="max",
        help=(
            "Color Brightness 설정. "
            "'max'를 주면 장치 최대값 사용, "
            "또는 정수값 직접 지정 (기본: max)"
        ),
    )

    return parser.parse_args()


def find_color_profile(
    pipeline: Pipeline,
):
    """1280x800 @10fps MJPG 프로파일을 찾는다."""

    profiles = pipeline.get_stream_profile_list(
        OBSensorType.COLOR_SENSOR
    )

    for index in range(profiles.get_count()):
        profile = (
            profiles.get_stream_profile_by_index(
                index
            )
        )

        if (
            profile.get_width() == WIDTH
            and profile.get_height() == HEIGHT
            and profile.get_fps() == FPS
            and profile.get_format()
            == PIXEL_FORMAT
        ):
            return profile

    raise RuntimeError(
        "요청한 Color 프로파일을 찾지 못했습니다: "
        f"{WIDTH}x{HEIGHT} @{FPS}fps MJPG"
    )


def frame_to_bgr(
    color_frame,
) -> np.ndarray:
    """Orbbec MJPG 프레임을 BGR 영상으로 변환한다."""

    raw_data = np.frombuffer(
        color_frame.get_data(),
        dtype=np.uint8,
    )

    image = cv2.imdecode(
        raw_data,
        cv2.IMREAD_COLOR,
    )

    if image is None:
        raise RuntimeError(
            "MJPG 프레임 디코딩에 실패했습니다."
        )

    if image.shape[:2] != (
        HEIGHT,
        WIDTH,
    ):
        raise RuntimeError(
            "수신 해상도가 요청값과 다릅니다. "
            f"수신={image.shape[1]}x"
            f"{image.shape[0]}, "
            f"요청={WIDTH}x{HEIGHT}"
        )

    return image


def wait_for_color_frame(
    pipeline: Pipeline,
):
    """정상 Color 프레임을 받을 때까지 기다린다."""

    while True:
        frames = pipeline.wait_for_frames(
            FRAME_TIMEOUT_MS
        )

        if frames is None:
            continue

        color_frame = frames.get_color_frame()

        if color_frame is not None:
            return color_frame


def calculate_metrics(
    averaged_image: np.ndarray,
    temporal_noise: float,
) -> dict[str, float]:
    """촬영 영상의 기본 품질 지표를 계산한다."""

    gray = cv2.cvtColor(
        averaged_image,
        cv2.COLOR_BGR2GRAY,
    )

    brightness = float(
        np.mean(gray)
    )

    overexposed_percent = float(
        np.mean(gray >= 250) * 100.0
    )

    sharpness = float(
        cv2.Laplacian(
            gray,
            cv2.CV_64F,
        ).var()
    )

    return {
        "brightness": brightness,
        "overexposed_percent": (
            overexposed_percent
        ),
        "sharpness": sharpness,
        "temporal_noise": temporal_noise,
    }


def capture_average_image(
    pipeline: Pipeline,
    average_frame_count: int,
) -> tuple[np.ndarray, dict[str, float]]:
    """연속 프레임을 평균해 원본 이미지 한 장을 만든다."""

    accumulated = np.zeros(
        (HEIGHT, WIDTH, 3),
        dtype=np.float64,
    )

    gray_sum = np.zeros(
        (HEIGHT, WIDTH),
        dtype=np.float64,
    )

    gray_square_sum = np.zeros(
        (HEIGHT, WIDTH),
        dtype=np.float64,
    )

    for frame_index in range(
        average_frame_count
    ):
        color_frame = wait_for_color_frame(
            pipeline
        )

        image = frame_to_bgr(
            color_frame
        )

        image_float = image.astype(
            np.float64
        )

        accumulated += image_float

        gray = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2GRAY,
        ).astype(np.float64)

        gray_sum += gray
        gray_square_sum += gray * gray

        print(
            f"\r평균 프레임 촬영: "
            f"{frame_index + 1}/"
            f"{average_frame_count}",
            end="",
            flush=True,
        )

    print()

    averaged_float = (
        accumulated
        / average_frame_count
    )

    averaged_image = np.clip(
        averaged_float,
        0,
        255,
    ).astype(np.uint8)

    gray_mean = (
        gray_sum
        / average_frame_count
    )

    gray_variance = (
        gray_square_sum
        / average_frame_count
        - gray_mean * gray_mean
    )

    gray_variance = np.maximum(
        gray_variance,
        0,
    )

    temporal_noise = float(
        np.mean(
            np.sqrt(gray_variance)
        )
    )

    metrics = calculate_metrics(
        averaged_image=averaged_image,
        temporal_noise=temporal_noise,
    )

    return averaged_image, metrics


def find_next_index(
    output_dir: Path,
    category: str,
) -> int:
    """기존 파일 다음 번호를 찾는다."""

    pattern = re.compile(
        rf"^{re.escape(category)}_(\d{{4}})_"
    )

    largest_index = 0

    for path in output_dir.glob(
        f"{category}_*.png"
    ):
        match = pattern.match(
            path.name
        )

        if match is None:
            continue

        largest_index = max(
            largest_index,
            int(match.group(1)),
        )

    return largest_index + 1


def append_metadata(
    csv_path: Path,
    row: dict,
) -> None:
    """촬영 정보를 CSV에 추가한다."""

    fieldnames = [
        "filename",
        "category",
        "session",
        "captured_at",
        "width",
        "height",
        "requested_fps",
        "format",
        "average_frames",
        "brightness",
        "overexposed_percent",
        "sharpness",
        "temporal_noise",
    ]

    write_header = (
        not csv_path.exists()
        or csv_path.stat().st_size == 0
    )

    with csv_path.open(
        "a",
        newline="",
        encoding="utf-8-sig",
    ) as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=fieldnames,
        )

        if write_header:
            writer.writeheader()

        writer.writerow(row)


def make_preview(
    image: np.ndarray,
    category: str,
    session: str,
    saved_count: int,
    target_count: int,
) -> np.ndarray:
    preview = cv2.resize(
        image,
        (960, 600),
        interpolation=cv2.INTER_AREA,
    )

    cv2.rectangle(
        preview,
        (0, 0),
        (960, 82),
        (0, 0, 0),
        -1,
    )

    cv2.putText(
        preview,
        (
            f"{category.upper()} | "
            f"{session} | "
            f"{saved_count}/{target_count}"
        ),
        (15, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    cv2.putText(
        preview,
        "SPACE: capture averaged image | Q: quit",
        (15, 62),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (0, 255, 0),
        2,
        cv2.LINE_AA,
    )

    return preview



def configure_color_controls(
    device,
    brightness_option: str,
) -> dict[str, int | bool]:
    """
    데이터 수집용 Color 설정.

    - Exposure/Gain은 Auto Exposure에 맡긴다.
    - White Balance도 자동으로 사용한다.
    - Color Brightness는 장치가 허용하는 최대값으로 설정한다.
    - 실제 적용값과 범위를 반환해 터미널에서 확인할 수 있게 한다.
    """

    device.set_bool_property(
        OBPropertyID.OB_PROP_COLOR_AUTO_EXPOSURE_BOOL,
        True,
    )

    device.set_bool_property(
        OBPropertyID.OB_PROP_COLOR_AUTO_WHITE_BALANCE_BOOL,
        True,
    )

    device.set_int_property(
        OBPropertyID.OB_PROP_COLOR_AUTO_EXPOSURE_PRIORITY_INT,
        0,
    )

    device.set_int_property(
        OBPropertyID.OB_PROP_COLOR_POWER_LINE_FREQUENCY_INT,
        2,
    )

    device.set_int_property(
        OBPropertyID.OB_PROP_COLOR_AE_MAX_EXPOSURE_INT,
        1249,
    )

    # 하드코딩하지 않고 실제 장치가 지원하는 Brightness 범위를 조회한다.
    brightness_range = device.get_int_property_range(
        OBPropertyID.OB_PROP_COLOR_BRIGHTNESS_INT
    )

    brightness_min = int(brightness_range.min)
    brightness_max = int(brightness_range.max)
    brightness_step = max(
        1,
        int(brightness_range.step),
    )

    if brightness_option.lower() == "max":
        brightness_value = brightness_max
    else:
        try:
            requested_brightness = int(
                brightness_option
            )
        except ValueError as exc:
            raise ValueError(
                "--brightness는 'max' 또는 정수여야 합니다."
            ) from exc

        brightness_value = min(
            brightness_max,
            max(
                brightness_min,
                requested_brightness,
            ),
        )

        # 장치 step에 맞춰 가장 가까운 유효값으로 보정
        brightness_value = (
            brightness_min
            + round(
                (
                    brightness_value
                    - brightness_min
                )
                / brightness_step
            )
            * brightness_step
        )

        brightness_value = min(
            brightness_max,
            max(
                brightness_min,
                brightness_value,
            ),
        )

    device.set_int_property(
        OBPropertyID.OB_PROP_COLOR_BRIGHTNESS_INT,
        brightness_value,
    )

    return {
        "auto_exposure": device.get_bool_property(
            OBPropertyID.OB_PROP_COLOR_AUTO_EXPOSURE_BOOL
        ),
        "auto_white_balance": device.get_bool_property(
            OBPropertyID.OB_PROP_COLOR_AUTO_WHITE_BALANCE_BOOL
        ),
        "brightness_min": brightness_min,
        "brightness_max": brightness_max,
        "brightness_step": brightness_step,
        "brightness": device.get_int_property(
            OBPropertyID.OB_PROP_COLOR_BRIGHTNESS_INT
        ),
    }


def read_color_controls(device) -> dict[str, int]:
    """현재 Color Exposure/Gain/WB/Brightness 실제값을 읽는다."""

    return {
        "exposure": device.get_int_property(
            OBPropertyID.OB_PROP_COLOR_EXPOSURE_INT
        ),
        "gain": device.get_int_property(
            OBPropertyID.OB_PROP_COLOR_GAIN_INT
        ),
        "white_balance": device.get_int_property(
            OBPropertyID.OB_PROP_COLOR_WHITE_BALANCE_INT
        ),
        "brightness": device.get_int_property(
            OBPropertyID.OB_PROP_COLOR_BRIGHTNESS_INT
        ),
    }

def main() -> None:
    args = parse_arguments()

    if args.target_count <= 0:
        raise ValueError(
            "target-count는 1 이상이어야 합니다."
        )

    if args.average_frames <= 0:
        raise ValueError(
            "average-frames는 1 이상이어야 합니다."
        )

    output_dir = (
        PROJECT_ROOT
        / "data"
        / "raw"
        / args.category
        / args.session
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    metadata_path = (
        output_dir
        / "capture_metadata.csv"
    )

    next_index = find_next_index(
        output_dir=output_dir,
        category=args.category,
    )

    pipeline = Pipeline()
    config = Config()
    started = False

    try:
        profile = find_color_profile(
            pipeline
        )

        config.enable_stream(
            profile
        )

        print("=" * 70)
        print("Gemini 336L 데이터 수집")
        print(
            f"프로파일: "
            f"{profile.get_width()}x"
            f"{profile.get_height()} "
            f"@{profile.get_fps()}fps "
            f"{profile.get_format()}"
        )
        print(f"구분: {args.category}")
        print(f"세션: {args.session}")
        print(f"저장 폴더: {output_dir}")
        print(
            f"프레임 평균: "
            f"{args.average_frames}장"
        )
        print()
        print("SPACE : 평균 영상 저장")
        print("Q     : 촬영 종료")
        print("=" * 70)

        pipeline.start(config)
        started = True

        device = pipeline.get_device()

        # 자동 Exposure/Gain + 자동 White Balance를 사용하고,
        # Color Brightness는 장치가 허용하는 최대값으로 설정한다.
        control_config = configure_color_controls(
            device,
            args.brightness,
        )

        print(
            "Color Auto Exposure:",
            control_config["auto_exposure"],
        )
        print(
            "Color Auto White Balance:",
            control_config["auto_white_balance"],
        )
        print(
            "Color Brightness 범위:",
            f'{control_config["brightness_min"]} ~ '
            f'{control_config["brightness_max"]} '
            f'(step={control_config["brightness_step"]})',
        )
        print(
            "Color Brightness 설정:",
            control_config["brightness"],
            (
                "(MAX)"
                if args.brightness.lower() == "max"
                else ""
            ),
        )

        print(
            f"카메라 워밍업 중: "
            f"{args.warmup_frames}프레임"
        )

        for _ in range(
            args.warmup_frames
        ):
            wait_for_color_frame(
                pipeline
            )

        print("워밍업 완료")

        warmed_controls = read_color_controls(
            device
        )

        print(
            "워밍업 후 Exposure:",
            warmed_controls["exposure"],
        )
        print(
            "워밍업 후 Gain:",
            warmed_controls["gain"],
        )
        print(
            "워밍업 후 White Balance:",
            warmed_controls["white_balance"],
        )
        print(
            "워밍업 후 Brightness:",
            warmed_controls["brightness"],
        )

        saved_count = 0

        while saved_count < args.target_count:
            color_frame = wait_for_color_frame(
                pipeline
            )

            image = frame_to_bgr(
                color_frame
            )

            preview = make_preview(
                image=image,
                category=args.category,
                session=args.session,
                saved_count=saved_count,
                target_count=args.target_count,
            )

            cv2.imshow(
                "Gemini 336L Dataset Capture",
                preview,
            )

            key = (
                cv2.waitKey(1)
                & 0xFF
            )

            if key in (
                ord("q"),
                ord("Q"),
            ):
                print("사용자가 촬영을 종료했습니다.")
                break

            if key != 32:
                continue

            print()
            print(
                f"[{saved_count + 1}/"
                f"{args.target_count}] "
                "평균 영상 촬영 시작"
            )

            # Auto Exposure 사용 중이므로 실제 촬영 직전 값을 기록/확인한다.
            capture_controls = read_color_controls(
                device
            )
            print(
                "촬영 설정: "
                f'Exposure={capture_controls["exposure"]}, '
                f'Gain={capture_controls["gain"]}, '
                f'WB={capture_controls["white_balance"]}, '
                f'Brightness={capture_controls["brightness"]}'
            )

            (
                averaged_image,
                metrics,
            ) = capture_average_image(
                pipeline=pipeline,
                average_frame_count=(
                    args.average_frames
                ),
            )

            captured_at = datetime.now()

            filename = (
                f"{args.category}_"
                f"{next_index:04d}_"
                f"{captured_at.strftime('%Y%m%d_%H%M%S_%f')}_"
                f"{WIDTH}x{HEIGHT}_"
                f"mjpg_average"
                f"{args.average_frames}.png"
            )

            output_path = (
                output_dir
                / filename
            )

            success = cv2.imwrite(
                str(output_path),
                averaged_image,
            )

            if not success:
                raise RuntimeError(
                    f"영상 저장 실패: {output_path}"
                )

            append_metadata(
                metadata_path,
                {
                    "filename": filename,
                    "category": args.category,
                    "session": args.session,
                    "captured_at": (
                        captured_at.isoformat(
                            timespec="microseconds"
                        )
                    ),
                    "width": WIDTH,
                    "height": HEIGHT,
                    "requested_fps": FPS,
                    "format": "MJPG",
                    "average_frames": (
                        args.average_frames
                    ),
                    "brightness": (
                        f"{metrics['brightness']:.6f}"
                    ),
                    "overexposed_percent": (
                        f"{metrics['overexposed_percent']:.6f}"
                    ),
                    "sharpness": (
                        f"{metrics['sharpness']:.6f}"
                    ),
                    "temporal_noise": (
                        f"{metrics['temporal_noise']:.6f}"
                    ),
                },
            )

            saved_count += 1
            next_index += 1

            print(f"저장 완료: {output_path}")
            print(
                f"밝기="
                f"{metrics['brightness']:.2f}, "
                f"과노출="
                f"{metrics['overexposed_percent']:.3f}%, "
                f"선명도="
                f"{metrics['sharpness']:.2f}, "
                f"시간 노이즈="
                f"{metrics['temporal_noise']:.3f}"
            )
            print()
            print(
                "물체를 다시 배치한 뒤 "
                "화면이 안정되면 Space를 누르세요."
            )

        print()
        print("=" * 70)
        print(
            f"이번 실행 저장 수: "
            f"{saved_count}장"
        )
        print(f"저장 폴더: {output_dir}")
        print(f"메타데이터: {metadata_path}")
        print("=" * 70)

    finally:
        if started:
            pipeline.stop()

        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
