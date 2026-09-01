from __future__ import annotations
import re
from pyorbbecsdk import (
    Config,
    OBFormat,
    OBPropertyType,
    OBSensorType,
    Pipeline,
)


WIDTH = 1280
HEIGHT = 800
FPS = 8
PIXEL_FORMAT = OBFormat.MJPG

WARMUP_FRAME_COUNT = 60
FRAME_TIMEOUT_MS = 3000


def find_color_profile(
    pipeline: Pipeline,
):
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
        "프로파일을 찾지 못했습니다: "
        f"{WIDTH}x{HEIGHT} @{FPS}fps MJPG"
    )


def wait_for_color_frame(
    pipeline: Pipeline,
):
    while True:
        frames = pipeline.wait_for_frames(
            FRAME_TIMEOUT_MS
        )

        if frames is None:
            continue

        color_frame = frames.get_color_frame()

        if color_frame is not None:
            return color_frame


def is_relevant_property(
    name: str,
) -> bool:
    """
    OB_PROP_COLOR_EXPOSURE_INT처럼
    밑줄로 구성된 SDK 속성 이름을
    공백 형태로 정규화해 검색한다.
    """

    normalized = re.sub(
        r"[^a-z0-9]+",
        " ",
        str(name).lower(),
    ).strip()

    if "color" not in normalized:
        return False

    keywords = (
        "exposure",
        "gain",
        "white balance",
        "brightness",
        "sharpness",
        "gamma",
        "saturation",
        "contrast",
        "hue",
        "power line",
        "powerline",
        "backlight",
    )

    return any(
        keyword in normalized
        for keyword in keywords
    )


def read_property(
    device,
    item,
) -> tuple[str, str]:
    try:
        if (
            item.type
            == OBPropertyType.OB_BOOL_PROPERTY
        ):
            value = device.get_bool_property(
                item.id
            )

            return (
                str(bool(value)),
                "False ~ True",
            )

        if (
            item.type
            == OBPropertyType.OB_INT_PROPERTY
        ):
            value = device.get_int_property(
                item.id
            )

            value_range = (
                device.get_int_property_range(
                    item.id
                )
            )

            range_text = (
                f"{value_range.min} ~ "
                f"{value_range.max}, "
                f"step={value_range.step}"
            )

            return str(value), range_text

        if (
            item.type
            == OBPropertyType.OB_FLOAT_PROPERTY
        ):
            value = device.get_float_property(
                item.id
            )

            value_range = (
                device.get_float_property_range(
                    item.id
                )
            )

            range_text = (
                f"{value_range.min} ~ "
                f"{value_range.max}, "
                f"step={value_range.step}"
            )

            return str(value), range_text

        return "지원하지 않는 타입", "-"

    except Exception as error:
        return (
            f"읽기 실패: {error}",
            "-",
        )


def main() -> None:
    pipeline = Pipeline()
    config = Config()
    started = False

    try:
        profile = find_color_profile(
            pipeline
        )

        config.enable_stream(profile)

        print("=" * 80)
        print("Gemini 336L Color 설정 조회")
        print(
            f"프로파일: "
            f"{profile.get_width()}x"
            f"{profile.get_height()} "
            f"@{profile.get_fps()}fps "
            f"{profile.get_format()}"
        )
        print("=" * 80)

        pipeline.start(config)
        started = True

        print(
            f"자동 설정 안정화 중: "
            f"{WARMUP_FRAME_COUNT}프레임"
        )

        for _ in range(
            WARMUP_FRAME_COUNT
        ):
            wait_for_color_frame(
                pipeline
            )

        device = pipeline.get_device()

        property_count = (
            device.get_support_property_count()
        )

        found_count = 0

        print()
        print(
            f"{'ID':>5} | "
            f"{'속성 이름':<38} | "
            f"{'현재값':<14} | 범위"
        )
        print("-" * 110)

        for index in range(property_count):
            item = device.get_supported_property(
                index
            )

            if not is_relevant_property(
                item.name
            ):
                continue

            value_text, range_text = (
                read_property(
                    device,
                    item,
                )
            )

            print(
                f"{int(item.id):>5} | "
                f"{item.name:<38} | "
                f"{value_text:<14} | "
                f"{range_text}"
            )

            found_count += 1

        print("-" * 110)
        print(
            f"표시된 Color 관련 속성: "
            f"{found_count}개"
        )

    finally:
        if started:
            pipeline.stop()


if __name__ == "__main__":
    main()