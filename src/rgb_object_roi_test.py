from __future__ import annotations

import argparse
from dataclasses import dataclass
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


@dataclass(frozen=True)
class RoiResult:
    raw_mask: np.ndarray
    cleaned_mask: np.ndarray
    final_mask: np.ndarray
    bbox: tuple[int, int, int, int] | None
    area: int
    background_lab: tuple[float, float, float]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Gemini 336L Color 기반 동적 검사체 ROI 테스트 "
            "(검은 검사판 배경 기준)"
        )
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "results" / "rgb_roi_test",
        help="결과 저장 폴더",
    )
    parser.add_argument(
        "--warmup-frames",
        type=int,
        default=60,
    )
    parser.add_argument(
        "--brightness",
        type=int,
        default=48,
        help="Color brightness 값. 현재 실험 기본값 48",
    )
    parser.add_argument(
        "--luma-delta",
        type=float,
        default=18.0,
        help="검은 배경 대비 L 차이 threshold",
    )
    parser.add_argument(
        "--chroma-delta",
        type=float,
        default=14.0,
        help="검은 배경 대비 a/b 색차 threshold",
    )
    parser.add_argument(
        "--min-object-area",
        type=int,
        default=20000,
        help="최소 검사체 면적(pixel)",
    )
    parser.add_argument(
        "--open-size",
        type=int,
        default=5,
    )
    parser.add_argument(
        "--close-size",
        type=int,
        default=21,
    )
    parser.add_argument(
        "--erode-px",
        type=int,
        default=20,
        help="최종 검사 영역에서 물체 경계를 안쪽으로 제외할 픽셀",
    )
    parser.add_argument(
        "--search-roi",
        type=str,
        default=None,
        help="고정 탐색영역 x1,y1,x2,y2. 없으면 실행 중 S키로 선택",
    )

    return parser.parse_args()


def odd(value: int) -> int:
    if value <= 0:
        return 0
    return value if value % 2 else value + 1


def parse_roi(text: str | None) -> tuple[int, int, int, int] | None:
    if text is None:
        return None

    values = [int(v.strip()) for v in text.split(",")]

    if len(values) != 4:
        raise ValueError("--search-roi는 x1,y1,x2,y2 형식이어야 합니다.")

    x1, y1, x2, y2 = values

    x1 = max(0, min(WIDTH - 1, x1))
    y1 = max(0, min(HEIGHT - 1, y1))
    x2 = max(x1 + 1, min(WIDTH, x2))
    y2 = max(y1 + 1, min(HEIGHT, y2))

    return x1, y1, x2, y2


def find_color_profile(pipeline: Pipeline):
    profiles = pipeline.get_stream_profile_list(
        OBSensorType.COLOR_SENSOR
    )

    for index in range(profiles.get_count()):
        profile = profiles.get_stream_profile_by_index(index)

        if (
            profile.get_width() == WIDTH
            and profile.get_height() == HEIGHT
            and profile.get_fps() == FPS
            and profile.get_format() == PIXEL_FORMAT
        ):
            return profile

    raise RuntimeError(
        f"{WIDTH}x{HEIGHT} @{FPS}fps MJPG Color 프로파일을 찾지 못했습니다."
    )


def wait_for_color_frame(pipeline: Pipeline):
    while True:
        frames = pipeline.wait_for_frames(FRAME_TIMEOUT_MS)

        if frames is None:
            continue

        color_frame = frames.get_color_frame()

        if color_frame is not None:
            return color_frame


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
        raise RuntimeError("MJPG Color 디코딩 실패")

    return image


def configure_camera(device, brightness: int) -> None:
    # 촬영 조건은 현재 실험에서 가장 안정적이었던 자동 노출/게인 기반으로 유지
    device.set_bool_property(
        OBPropertyID.OB_PROP_COLOR_AUTO_EXPOSURE_BOOL,
        True,
    )
    device.set_bool_property(
        OBPropertyID.OB_PROP_COLOR_AUTO_WHITE_BALANCE_BOOL,
        True,
    )

    # 지원 범위 안에서 brightness 적용
    try:
        prop_range = device.get_int_property_range(
            OBPropertyID.OB_PROP_COLOR_BRIGHTNESS_INT
        )
        min_value = int(prop_range.min)
        max_value = int(prop_range.max)
        brightness = int(np.clip(brightness, min_value, max_value))

        device.set_int_property(
            OBPropertyID.OB_PROP_COLOR_BRIGHTNESS_INT,
            brightness,
        )

        print(
            f"Color Brightness 범위: {min_value} ~ {max_value} | "
            f"설정: {brightness}"
        )
    except Exception as exc:
        print(f"Brightness 설정 경고: {exc}")

    try:
        device.set_int_property(
            OBPropertyID.OB_PROP_COLOR_POWER_LINE_FREQUENCY_INT,
            2,
        )
    except Exception:
        pass


def get_camera_state(device) -> dict[str, int | bool]:
    result: dict[str, int | bool] = {}

    for name, prop, kind in (
        (
            "auto_exposure",
            OBPropertyID.OB_PROP_COLOR_AUTO_EXPOSURE_BOOL,
            "bool",
        ),
        (
            "exposure",
            OBPropertyID.OB_PROP_COLOR_EXPOSURE_INT,
            "int",
        ),
        (
            "gain",
            OBPropertyID.OB_PROP_COLOR_GAIN_INT,
            "int",
        ),
        (
            "brightness",
            OBPropertyID.OB_PROP_COLOR_BRIGHTNESS_INT,
            "int",
        ),
        (
            "white_balance",
            OBPropertyID.OB_PROP_COLOR_WHITE_BALANCE_INT,
            "int",
        ),
    ):
        try:
            if kind == "bool":
                result[name] = device.get_bool_property(prop)
            else:
                result[name] = device.get_int_property(prop)
        except Exception:
            pass

    return result


def make_border_sample_mask(
    roi_height: int,
    roi_width: int,
    border_ratio: float = 0.10,
) -> np.ndarray:
    """
    Search ROI의 가장자리 띠를 배경 샘플로 사용한다.
    검사체가 중앙에 있고 검은 검사판이 주변에 존재하는 환경을 전제로 한다.
    """
    by = max(4, int(round(roi_height * border_ratio)))
    bx = max(4, int(round(roi_width * border_ratio)))

    mask = np.zeros((roi_height, roi_width), dtype=bool)
    mask[:by, :] = True
    mask[-by:, :] = True
    mask[:, :bx] = True
    mask[:, -bx:] = True

    return mask


def estimate_black_background_lab(
    roi_bgr: np.ndarray,
) -> tuple[float, float, float]:
    lab = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)

    border_mask = make_border_sample_mask(
        roi_bgr.shape[0],
        roi_bgr.shape[1],
        border_ratio=0.10,
    )

    border_pixels = lab[border_mask]

    if border_pixels.size == 0:
        raise RuntimeError("배경 샘플을 얻지 못했습니다.")

    # Search ROI 가장자리에 일부 물체나 반사가 들어와도
    # 검은 판 쪽을 우선 사용하기 위해 L 하위 60%만 사용한다.
    l_values = border_pixels[:, 0]
    cutoff = np.percentile(l_values, 60.0)
    dark_pixels = border_pixels[l_values <= cutoff]

    if dark_pixels.shape[0] < 50:
        dark_pixels = border_pixels

    median = np.median(dark_pixels, axis=0)

    return (
        float(median[0]),
        float(median[1]),
        float(median[2]),
    )


def largest_component(
    mask: np.ndarray,
    min_area: int,
) -> np.ndarray:
    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask,
        connectivity=8,
    )

    valid = [
        index
        for index in range(1, count)
        if stats[index, cv2.CC_STAT_AREA] >= min_area
    ]

    output = np.zeros_like(mask)

    if not valid:
        return output

    label = max(
        valid,
        key=lambda idx: stats[idx, cv2.CC_STAT_AREA],
    )

    output[labels == label] = 255

    return output


def fill_external_contour(mask: np.ndarray) -> np.ndarray:
    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    if not contours:
        return np.zeros_like(mask)

    contour = max(contours, key=cv2.contourArea)

    output = np.zeros_like(mask)

    cv2.drawContours(
        output,
        [contour],
        -1,
        255,
        thickness=cv2.FILLED,
    )

    return output


def erode_by_pixels(mask: np.ndarray, pixels: int) -> np.ndarray:
    """
    정사각 kernel erosion 대신 distance transform으로 경계에서
    지정 pixel만큼 안쪽인 영역만 남긴다.
    """
    if pixels <= 0 or not np.any(mask):
        return mask.copy()

    distance = cv2.distanceTransform(
        mask,
        cv2.DIST_L2,
        5,
    )

    return (distance >= float(pixels)).astype(np.uint8) * 255


def get_bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    ys, xs = np.nonzero(mask)

    if xs.size == 0:
        return None

    return (
        int(xs.min()),
        int(ys.min()),
        int(xs.max()) + 1,
        int(ys.max()) + 1,
    )


def create_dynamic_object_mask(
    image_bgr: np.ndarray,
    search_roi: tuple[int, int, int, int],
    luma_delta: float,
    chroma_delta: float,
    min_object_area: int,
    open_size: int,
    close_size: int,
    erode_px: int,
) -> RoiResult:
    x1, y1, x2, y2 = search_roi

    roi = image_bgr[y1:y2, x1:x2]

    if roi.size == 0:
        raise RuntimeError("Search ROI가 비어 있습니다.")

    background_lab = estimate_black_background_lab(roi)

    lab = cv2.cvtColor(
        roi,
        cv2.COLOR_BGR2LAB,
    ).astype(np.float32)

    bg_l, bg_a, bg_b = background_lab

    luma_diff = np.abs(lab[..., 0] - bg_l)
    chroma_diff = np.sqrt(
        (lab[..., 1] - bg_a) ** 2
        + (lab[..., 2] - bg_b) ** 2
    )

    raw_local = (
        (luma_diff >= luma_delta)
        | (chroma_diff >= chroma_delta)
    ).astype(np.uint8) * 255

    # 작은 점 제거
    open_k = odd(open_size)

    cleaned = raw_local.copy()

    if open_k > 0:
        kernel = np.ones((open_k, open_k), dtype=np.uint8)
        cleaned = cv2.morphologyEx(
            cleaned,
            cv2.MORPH_OPEN,
            kernel,
        )

    # 반사/그림자 때문에 물체 내부가 끊기는 것을 어느 정도 연결
    close_k = odd(close_size)

    if close_k > 0:
        kernel = np.ones((close_k, close_k), dtype=np.uint8)
        cleaned = cv2.morphologyEx(
            cleaned,
            cv2.MORPH_CLOSE,
            kernel,
        )

    cleaned = largest_component(
        cleaned,
        min_area=min_object_area,
    )

    if np.any(cleaned):
        cleaned = fill_external_contour(cleaned)

    final_local = erode_by_pixels(
        cleaned,
        pixels=erode_px,
    )

    raw_mask = np.zeros(
        image_bgr.shape[:2],
        dtype=np.uint8,
    )
    cleaned_mask = np.zeros_like(raw_mask)
    final_mask = np.zeros_like(raw_mask)

    raw_mask[y1:y2, x1:x2] = raw_local
    cleaned_mask[y1:y2, x1:x2] = cleaned
    final_mask[y1:y2, x1:x2] = final_local

    return RoiResult(
        raw_mask=raw_mask,
        cleaned_mask=cleaned_mask,
        final_mask=final_mask,
        bbox=get_bbox(final_mask),
        area=int(np.count_nonzero(final_mask)),
        background_lab=background_lab,
    )


def make_overlay(
    image_bgr: np.ndarray,
    search_roi: tuple[int, int, int, int],
    result: RoiResult,
) -> np.ndarray:
    overlay = image_bgr.copy()

    x1, y1, x2, y2 = search_roi

    cv2.rectangle(
        overlay,
        (x1, y1),
        (x2 - 1, y2 - 1),
        (0, 255, 255),
        2,
    )

    contours, _ = cv2.findContours(
        result.final_mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    cv2.drawContours(
        overlay,
        contours,
        -1,
        (0, 255, 0),
        3,
    )

    if result.bbox is not None:
        bx1, by1, bx2, by2 = result.bbox

        cv2.rectangle(
            overlay,
            (bx1, by1),
            (bx2 - 1, by2 - 1),
            (255, 255, 0),
            2,
        )

    return overlay


def make_masked_color(
    image_bgr: np.ndarray,
    mask: np.ndarray,
) -> np.ndarray:
    output = np.zeros_like(image_bgr)
    output[mask > 0] = image_bgr[mask > 0]
    return output


def make_preview(
    image_bgr: np.ndarray,
    raw_mask: np.ndarray,
    overlay: np.ndarray,
) -> np.ndarray:
    mask_bgr = cv2.cvtColor(
        raw_mask,
        cv2.COLOR_GRAY2BGR,
    )

    items = []

    for image in (image_bgr, mask_bgr, overlay):
        items.append(
            cv2.resize(
                image,
                (480, 300),
                interpolation=cv2.INTER_AREA,
            )
        )

    preview = np.hstack(items)

    cv2.putText(
        preview,
        "S: select search ROI | SPACE: save | Q/ESC: quit",
        (15, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    return preview


def select_search_roi(
    image_bgr: np.ndarray,
) -> tuple[int, int, int, int] | None:
    display = cv2.resize(
        image_bgr,
        (960, 600),
        interpolation=cv2.INTER_AREA,
    )

    x, y, w, h = cv2.selectROI(
        "Select SEARCH AREA - ENTER confirm / ESC cancel",
        display,
        showCrosshair=True,
        fromCenter=False,
    )

    cv2.destroyWindow(
        "Select SEARCH AREA - ENTER confirm / ESC cancel"
    )

    if w <= 0 or h <= 0:
        return None

    scale_x = image_bgr.shape[1] / 960.0
    scale_y = image_bgr.shape[0] / 600.0

    x1 = int(round(x * scale_x))
    y1 = int(round(y * scale_y))
    x2 = int(round((x + w) * scale_x))
    y2 = int(round((y + h) * scale_y))

    return (
        max(0, x1),
        max(0, y1),
        min(image_bgr.shape[1], x2),
        min(image_bgr.shape[0], y2),
    )


def save_results(
    output_root: Path,
    image_bgr: np.ndarray,
    search_roi: tuple[int, int, int, int],
    result: RoiResult,
    overlay: np.ndarray,
    camera_state: dict[str, int | bool],
    args: argparse.Namespace,
) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = output_root / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)

    cv2.imwrite(
        str(output_dir / "original.png"),
        image_bgr,
    )
    cv2.imwrite(
        str(output_dir / "candidate_raw.png"),
        result.raw_mask,
    )
    cv2.imwrite(
        str(output_dir / "object_mask_filled.png"),
        result.cleaned_mask,
    )
    cv2.imwrite(
        str(output_dir / "final_inspection_mask.png"),
        result.final_mask,
    )
    cv2.imwrite(
        str(output_dir / "object_overlay.png"),
        overlay,
    )
    cv2.imwrite(
        str(output_dir / "masked_color.png"),
        make_masked_color(
            image_bgr,
            result.final_mask,
        ),
    )

    lines = [
        f"search_roi={search_roi}",
        f"object_bbox={result.bbox}",
        f"final_mask_area={result.area}",
        (
            "background_lab="
            f"{tuple(round(v, 3) for v in result.background_lab)}"
        ),
        f"luma_delta={args.luma_delta}",
        f"chroma_delta={args.chroma_delta}",
        f"min_object_area={args.min_object_area}",
        f"open_size={args.open_size}",
        f"close_size={args.close_size}",
        f"erode_px={args.erode_px}",
    ]

    for key, value in camera_state.items():
        lines.append(f"camera_{key}={value}")

    (output_dir / "summary.txt").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )

    return output_dir


def main() -> None:
    args = parse_args()

    search_roi = parse_roi(args.search_roi)

    pipeline = Pipeline()
    config = Config()
    started = False

    try:
        profile = find_color_profile(pipeline)
        config.enable_stream(profile)

        pipeline.start(config)
        started = True

        device = pipeline.get_device()

        configure_camera(
            device,
            brightness=args.brightness,
        )

        print("=" * 72)
        print("RGB Dynamic Object ROI Test")
        print(
            f"Color: {profile.get_width()}x{profile.get_height()} "
            f"@{profile.get_fps()} {profile.get_format()}"
        )
        print(f"Brightness: {args.brightness}")
        print("S: Search Area 선택/변경")
        print("SPACE: 현재 mask 결과 저장")
        print("Q/ESC: 종료")
        print("=" * 72)

        for _ in range(args.warmup_frames):
            wait_for_color_frame(pipeline)

        print("워밍업 완료")
        print("카메라 상태:", get_camera_state(device))

        while True:
            frame = wait_for_color_frame(pipeline)
            image = frame_to_bgr(frame)

            if search_roi is None:
                preview = image.copy()

                cv2.putText(
                    preview,
                    "Press S to select SEARCH AREA",
                    (30, 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.0,
                    (0, 255, 255),
                    2,
                    cv2.LINE_AA,
                )

                cv2.imshow(
                    "RGB Dynamic Object ROI Test",
                    cv2.resize(
                        preview,
                        (960, 600),
                        interpolation=cv2.INTER_AREA,
                    ),
                )

                key = cv2.waitKey(1) & 0xFF

                if key in (ord("q"), ord("Q"), 27):
                    break

                if key in (ord("s"), ord("S")):
                    selected = select_search_roi(image)

                    if selected is not None:
                        search_roi = selected
                        print("Search ROI:", search_roi)

                continue

            try:
                result = create_dynamic_object_mask(
                    image_bgr=image,
                    search_roi=search_roi,
                    luma_delta=args.luma_delta,
                    chroma_delta=args.chroma_delta,
                    min_object_area=args.min_object_area,
                    open_size=args.open_size,
                    close_size=args.close_size,
                    erode_px=args.erode_px,
                )
            except Exception as exc:
                print("ROI 계산 오류:", exc)
                result = RoiResult(
                    raw_mask=np.zeros(image.shape[:2], np.uint8),
                    cleaned_mask=np.zeros(image.shape[:2], np.uint8),
                    final_mask=np.zeros(image.shape[:2], np.uint8),
                    bbox=None,
                    area=0,
                    background_lab=(0.0, 0.0, 0.0),
                )

            overlay = make_overlay(
                image,
                search_roi,
                result,
            )

            preview = make_preview(
                image,
                result.raw_mask,
                overlay,
            )

            cv2.imshow(
                "RGB Dynamic Object ROI Test",
                preview,
            )

            key = cv2.waitKey(1) & 0xFF

            if key in (ord("q"), ord("Q"), 27):
                break

            if key in (ord("s"), ord("S")):
                selected = select_search_roi(image)

                if selected is not None:
                    search_roi = selected
                    print("Search ROI 변경:", search_roi)

                continue

            if key == 32:
                camera_state = get_camera_state(device)

                output_dir = save_results(
                    output_root=args.output_dir,
                    image_bgr=image,
                    search_roi=search_roi,
                    result=result,
                    overlay=overlay,
                    camera_state=camera_state,
                    args=args,
                )

                print("=" * 72)
                print("RGB ROI 저장 완료")
                print("Search ROI:", search_roi)
                print("Object bbox:", result.bbox)
                print("Final mask area:", result.area)
                print(
                    "Background LAB:",
                    tuple(round(v, 2) for v in result.background_lab),
                )
                print("Camera:", camera_state)
                print("저장:", output_dir.resolve())
                print("=" * 72)

    finally:
        if started:
            pipeline.stop()

        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
