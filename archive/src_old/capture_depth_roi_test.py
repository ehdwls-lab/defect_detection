from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

from pyorbbecsdk import (
    AlignFilter,
    Config,
    OBFormat,
    OBSensorType,
    OBStreamType,
    Pipeline,
)

from depth_roi_improved import (
    create_object_mask_debug,
    make_depth_visualization,
    make_mask_views,
)

WIDTH = 1280
HEIGHT = 800
FPS = 10
COLOR_FORMAT = OBFormat.MJPG
FRAME_TIMEOUT_MS = 3000


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Gemini 336L Depth ROI 진단 실험")
    p.add_argument("--output-dir", type=Path, default=Path("../results/depth_roi_test"))
    p.add_argument("--average-depth-frames", type=int, default=8)
    p.add_argument("--warmup-frames", type=int, default=30)
    p.add_argument("--min-height-mm", type=float, default=10.0)
    p.add_argument("--border-ratio", type=float, default=0.10)
    p.add_argument("--min-object-area", type=int, default=10000)
    p.add_argument("--morphology-size", type=int, default=7)
    p.add_argument("--bridge-size", type=int, default=21)
    p.add_argument("--mask-erode", type=int, default=15)
    return p.parse_args()


def find_color_profile(pipeline: Pipeline):
    profiles = pipeline.get_stream_profile_list(OBSensorType.COLOR_SENSOR)
    for i in range(profiles.get_count()):
        profile = profiles.get_stream_profile_by_index(i)
        if (
            profile.get_width() == WIDTH
            and profile.get_height() == HEIGHT
            and profile.get_fps() == FPS
            and profile.get_format() == COLOR_FORMAT
        ):
            return profile
    print("경고: 1280x800 @10fps MJPG를 찾지 못해 Color 기본 프로파일 사용")
    return profiles.get_default_video_stream_profile()


def color_frame_to_bgr(frame) -> np.ndarray:
    raw = np.frombuffer(frame.get_data(), dtype=np.uint8)
    fmt = frame.get_format()
    w, h = frame.get_width(), frame.get_height()

    if fmt == OBFormat.MJPG:
        image = cv2.imdecode(raw, cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError("MJPG Color 디코딩 실패")
        return image
    if fmt == OBFormat.RGB:
        return cv2.cvtColor(raw.reshape(h, w, 3), cv2.COLOR_RGB2BGR)
    if fmt == OBFormat.BGR:
        return raw.reshape(h, w, 3).copy()
    raise RuntimeError(f"지원하지 않는 Color format: {fmt}")


def depth_frame_to_mm(frame) -> np.ndarray:
    w, h = frame.get_width(), frame.get_height()
    raw = np.frombuffer(frame.get_data(), dtype=np.uint16)
    depth = raw.reshape(h, w).astype(np.float32)
    return depth * frame.get_depth_scale()


def wait_aligned_frames(pipeline: Pipeline, align_filter: AlignFilter):
    while True:
        frames = pipeline.wait_for_frames(FRAME_TIMEOUT_MS)
        if frames is None:
            continue
        aligned = align_filter.process(frames)
        if aligned is None:
            continue
        color = aligned.get_color_frame()
        depth = aligned.get_depth_frame()
        if color is not None and depth is not None:
            return color, depth


def robust_depth_median(depth_frames: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    stack = np.stack(depth_frames, axis=0).astype(np.float32)
    valid = np.isfinite(stack) & (stack > 0)
    valid_count = np.sum(valid, axis=0)
    safe = np.where(valid, stack, np.nan)
    with np.errstate(all="ignore"):
        median = np.nanmedian(safe, axis=0)
    median = np.where(np.isfinite(median), median, 0).astype(np.float32)
    valid_ratio = valid_count.astype(np.float32) / float(stack.shape[0])
    return median, valid_ratio


def save_result(output_dir: Path, color: np.ndarray, depth_mm: np.ndarray, valid_ratio: np.ndarray, args):
    output_dir.mkdir(parents=True, exist_ok=True)

    if depth_mm.shape[:2] != color.shape[:2]:
        target = (color.shape[1], color.shape[0])
        depth_mm = cv2.resize(depth_mm, target, interpolation=cv2.INTER_NEAREST)
        valid_ratio = cv2.resize(valid_ratio, target, interpolation=cv2.INTER_NEAREST)

    result, debug = create_object_mask_debug(
        depth_mm=depth_mm,
        min_height_mm=args.min_height_mm,
        border_ratio=args.border_ratio,
        min_object_area=args.min_object_area,
        morphology_size=args.morphology_size,
        bridge_size=args.bridge_size,
        mask_erode=args.mask_erode,
        contour_fill=True,
    )

    depth_vis = make_depth_visualization(depth_mm)
    overlay, bbox_overlay, masked = make_mask_views(color, result.mask, result.bbox)
    ratio_vis = cv2.applyColorMap(
        np.clip(valid_ratio * 255.0, 0, 255).astype(np.uint8),
        cv2.COLORMAP_TURBO,
    )

    cv2.imwrite(str(output_dir / "color.png"), color)
    np.save(str(output_dir / "depth_mm.npy"), depth_mm)
    cv2.imwrite(str(output_dir / "depth_visualization.png"), depth_vis)
    cv2.imwrite(str(output_dir / "valid_ratio.png"), ratio_vis)

    for name, image in debug.items():
        cv2.imwrite(str(output_dir / f"{name}.png"), image)

    cv2.imwrite(str(output_dir / "color_mask_overlay.png"), overlay)
    cv2.imwrite(str(output_dir / "bbox_overlay.png"), bbox_overlay)
    cv2.imwrite(str(output_dir / "masked_color.png"), masked)

    valid = np.isfinite(depth_mm) & (depth_mm > 0)
    lines = [
        f"floor_depth_mm={result.floor_depth_mm:.3f}",
        f"mask_area={result.mask_area}",
        f"bbox={result.bbox}",
        f"invalid_ratio_in_bbox={result.invalid_ratio_in_bbox:.6f}",
        f"whole_invalid_ratio={float(np.mean(~valid)):.6f}",
        f"mean_valid_ratio={float(np.mean(valid_ratio)):.6f}",
        f"min_height_mm={args.min_height_mm}",
        f"border_ratio={args.border_ratio}",
        f"min_object_area={args.min_object_area}",
        f"morphology_size={args.morphology_size}",
        f"bridge_size={args.bridge_size}",
        f"mask_erode={args.mask_erode}",
    ]
    (output_dir / "summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")

    print("=" * 70)
    print("Depth ROI 저장 완료")
    for line in lines:
        print(line)
    print(f"결과 폴더: {output_dir.resolve()}")
    print("=" * 70)


def main() -> None:
    args = parse_args()
    if args.average_depth_frames < 1:
        raise ValueError("--average-depth-frames는 1 이상이어야 합니다.")

    pipeline = Pipeline()
    config = Config()
    started = False

    color_profile = find_color_profile(pipeline)
    depth_profiles = pipeline.get_stream_profile_list(OBSensorType.DEPTH_SENSOR)
    depth_profile = depth_profiles.get_default_video_stream_profile()

    config.enable_stream(color_profile)
    config.enable_stream(depth_profile)

    align_filter = AlignFilter(align_to_stream=OBStreamType.COLOR_STREAM)

    try:
        pipeline.start(config)
        started = True

        print("=" * 70)
        print("Gemini 336L Depth ROI Test")
        print(
            f"Color: {color_profile.get_width()}x{color_profile.get_height()} "
            f"@{color_profile.get_fps()} {color_profile.get_format()}"
        )
        print(
            f"Depth: {depth_profile.get_width()}x{depth_profile.get_height()} "
            f"@{depth_profile.get_fps()} {depth_profile.get_format()}"
        )
        print("SPACE: Depth ROI 캡처/저장")
        print("Q/ESC: 종료")
        print("=" * 70)

        for _ in range(args.warmup_frames):
            wait_aligned_frames(pipeline, align_filter)
        print("워밍업 완료")

        while True:
            color_frame, depth_frame = wait_aligned_frames(pipeline, align_filter)
            color = color_frame_to_bgr(color_frame)
            depth_mm = depth_frame_to_mm(depth_frame)
            depth_vis = make_depth_visualization(depth_mm)

            if depth_vis.shape[:2] != color.shape[:2]:
                depth_vis = cv2.resize(
                    depth_vis,
                    (color.shape[1], color.shape[0]),
                    interpolation=cv2.INTER_NEAREST,
                )

            left = cv2.resize(color, (640, 400), interpolation=cv2.INTER_AREA)
            right = cv2.resize(depth_vis, (640, 400), interpolation=cv2.INTER_NEAREST)
            preview = np.hstack([left, right])
            cv2.putText(
                preview,
                "SPACE: capture ROI | Q/ESC: quit",
                (20, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            cv2.imshow("Depth ROI Test", preview)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), ord("Q"), 27):
                break
            if key != 32:
                continue

            print(f"Depth {args.average_depth_frames}프레임 median 합성 시작...")
            depth_frames = []
            latest_color = color

            for i in range(args.average_depth_frames):
                c_frame, d_frame = wait_aligned_frames(pipeline, align_filter)
                latest_color = color_frame_to_bgr(c_frame)
                depth_frames.append(depth_frame_to_mm(d_frame))
                print(
                    f"\rDepth 수집: {i + 1}/{args.average_depth_frames}",
                    end="",
                    flush=True,
                )
            print()

            median_depth, valid_ratio = robust_depth_median(depth_frames)
            save_result(args.output_dir, latest_color, median_depth, valid_ratio, args)
            print("결과 확인 후 추가 촬영은 SPACE, 종료는 Q")

    finally:
        if started:
            pipeline.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
