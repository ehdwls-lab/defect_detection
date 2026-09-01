from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from pyorbbecsdk import (
    AlignFilter,
    Config,
    OBFrameAggregateOutputMode,
    OBPropertyID,
    OBSensorType,
    OBStreamType,
    Pipeline,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.capture_dataset import find_color_profile, frame_to_bgr
from src.depth_roi import (
    create_object_mask,
    make_depth_visualization,
    make_mask_views,
)


FRAME_TIMEOUT_MS = 3000


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Gemini 336L aligned RGB-D Depth ROI preview/capture"
    )
    parser.add_argument("--preview", action="store_true", help="실시간 Mask 확인")
    parser.add_argument("--height-cm", type=int, default=None)
    parser.add_argument("--output-root", type=Path, default=PROJECT_ROOT / "data" / "depth_roi_test")
    parser.add_argument("--warmup-frames", type=int, default=60)
    parser.add_argument("--average-frames", type=int, default=4)
    parser.add_argument("--min-height-mm", type=float, default=5.0)
    parser.add_argument("--border-ratio", type=float, default=0.10)
    parser.add_argument("--min-object-area", type=int, default=10000)
    parser.add_argument("--mask-erode", type=int, default=3)
    parser.add_argument("--morphology-size", type=int, default=5)
    parser.add_argument(
        "--all-components", action="store_true",
        help="최대 component 하나 대신 최소 면적 이상의 모든 component 사용",
    )
    return parser.parse_args()


def configure_color_device(device) -> None:
    device.set_bool_property(OBPropertyID.OB_PROP_COLOR_AUTO_EXPOSURE_BOOL, True)
    device.set_bool_property(OBPropertyID.OB_PROP_COLOR_AUTO_WHITE_BALANCE_BOOL, True)
    device.set_int_property(OBPropertyID.OB_PROP_COLOR_AUTO_EXPOSURE_PRIORITY_INT, 0)
    device.set_int_property(OBPropertyID.OB_PROP_COLOR_POWER_LINE_FREQUENCY_INT, 2)
    device.set_int_property(OBPropertyID.OB_PROP_COLOR_AE_MAX_EXPOSURE_INT, 1249)
    device.set_int_property(OBPropertyID.OB_PROP_COLOR_BRIGHTNESS_INT, 0)


def get_aligned_rgbd(pipeline: Pipeline, align_filter: AlignFilter) -> tuple[np.ndarray, np.ndarray, float]:
    frames = pipeline.wait_for_frames(FRAME_TIMEOUT_MS)
    if frames is None:
        raise RuntimeError("RGB-D frame timeout")
    aligned = align_filter.process(frames)
    if aligned is None:
        raise RuntimeError("Depth-to-Color alignment 실패")
    color_frame = aligned.get_color_frame()
    depth_frame = aligned.get_depth_frame()
    if color_frame is None or depth_frame is None:
        raise RuntimeError("정렬된 Color/Depth frame이 없습니다.")
    color_bgr = frame_to_bgr(color_frame)
    raw = np.frombuffer(depth_frame.get_data(), dtype=np.uint16)
    depth_mm = raw.reshape(depth_frame.get_height(), depth_frame.get_width()).astype(np.float32)
    depth_scale = float(depth_frame.get_depth_scale())
    depth_mm *= depth_scale
    if depth_mm.shape != color_bgr.shape[:2]:
        raise RuntimeError(
            "정렬 후 RGB/Depth 크기가 다릅니다. Resize fallback은 사용하지 않습니다: "
            f"RGB={color_bgr.shape[1]}x{color_bgr.shape[0]}, "
            f"Depth={depth_mm.shape[1]}x{depth_mm.shape[0]}"
        )
    return color_bgr, depth_mm, depth_scale


def average_rgbd(
    pipeline: Pipeline, align_filter: AlignFilter, count: int
) -> tuple[np.ndarray, np.ndarray, float]:
    color_sum = None
    depth_sum = None
    depth_count = None
    depth_scale = 0.0
    for _ in range(count):
        color, depth, depth_scale = get_aligned_rgbd(pipeline, align_filter)
        if color_sum is None:
            color_sum = np.zeros_like(color, dtype=np.float64)
            depth_sum = np.zeros_like(depth, dtype=np.float64)
            depth_count = np.zeros_like(depth, dtype=np.uint16)
        color_sum += color
        valid = np.isfinite(depth) & (depth > 0)
        depth_sum[valid] += depth[valid]
        depth_count[valid] += 1
    color_avg = np.clip(color_sum / count, 0, 255).astype(np.uint8)
    depth_avg = np.zeros_like(depth_sum, dtype=np.float32)
    valid = depth_count > 0
    depth_avg[valid] = (depth_sum[valid] / depth_count[valid]).astype(np.float32)
    return color_avg, depth_avg, depth_scale


def process_mask(color_bgr: np.ndarray, depth_mm: np.ndarray, args: argparse.Namespace):
    result = create_object_mask(
        depth_mm=depth_mm,
        min_height_mm=args.min_height_mm,
        border_ratio=args.border_ratio,
        min_object_area=args.min_object_area,
        largest_component=not args.all_components,
        morphology_size=args.morphology_size,
        mask_erode=args.mask_erode,
    )
    depth_vis = make_depth_visualization(depth_mm)
    overlay, bbox_preview, masked = make_mask_views(color_bgr, result.mask, result.bbox)
    return result, depth_vis, overlay, bbox_preview, masked


def next_sample_dir(root: Path) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    indexes = []
    for path in root.glob("sample_*"):
        try:
            indexes.append(int(path.name.split("_")[-1]))
        except ValueError:
            pass
    return root / f"sample_{max(indexes, default=0) + 1:03d}"


def save_sample(
    output_dir: Path, color_bgr: np.ndarray, depth_mm: np.ndarray,
    depth_scale: float, result, depth_vis: np.ndarray,
    overlay: np.ndarray, bbox_preview: np.ndarray, masked: np.ndarray,
    args: argparse.Namespace,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=False)
    files = {
        "color.png": color_bgr,
        "depth_visualization.png": depth_vis,
        "object_mask.png": result.mask,
        "mask_overlay.png": overlay,
        "roi_bbox_preview.png": bbox_preview,
        "masked_rgb.png": masked,
    }
    for name, image in files.items():
        if not cv2.imwrite(str(output_dir / name), image):
            raise RuntimeError(f"저장 실패: {output_dir / name}")
    np.save(output_dir / "depth.npy", depth_mm, allow_pickle=False)
    bbox = None
    if result.bbox is not None:
        x1, y1, x2, y2 = result.bbox
        bbox = {"x_min": x1, "y_min": y1, "x_max": x2, "y_max": y2,
                "width": x2 - x1, "height": y2 - y1}
    metadata = {
        "timestamp": datetime.now().isoformat(timespec="microseconds"),
        "height_cm": args.height_cm,
        "color_width": int(color_bgr.shape[1]),
        "color_height": int(color_bgr.shape[0]),
        "depth_width": int(depth_mm.shape[1]),
        "depth_height": int(depth_mm.shape[0]),
        "depth_scale": depth_scale,
        "alignment": "Orbbec software Depth-to-Color AlignFilter",
        "floor_depth_mm": result.floor_depth_mm,
        "min_height_mm": args.min_height_mm,
        "border_ratio": args.border_ratio,
        "mask_area": result.mask_area,
        "bbox": bbox,
        "invalid_depth_ratio_in_bbox": result.invalid_ratio_in_bbox,
        "camera_settings": {"color": "1280x800 MJPG 8 FPS", "AE": True, "AWB": True},
    }
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def main() -> None:
    args = parse_arguments()
    if args.average_frames <= 0 or args.warmup_frames < 0:
        raise ValueError("average-frames는 양수, warmup-frames는 0 이상이어야 합니다.")
    output_root = args.output_root
    if args.height_cm is not None:
        if not 1 <= args.height_cm <= 999:
            raise ValueError("height-cm는 1~999 범위여야 합니다.")
        output_root = output_root / f"h{args.height_cm:03d}"

    pipeline = Pipeline()
    config = Config()
    color_profile = find_color_profile(pipeline)
    depth_profile = pipeline.get_stream_profile_list(
        OBSensorType.DEPTH_SENSOR
    ).get_default_video_stream_profile()
    config.enable_stream(color_profile)
    config.enable_stream(depth_profile)
    config.set_frame_aggregate_output_mode(OBFrameAggregateOutputMode.FULL_FRAME_REQUIRE)
    align_filter = AlignFilter(align_to_stream=OBStreamType.COLOR_STREAM)
    started = False
    try:
        pipeline.start(config)
        started = True
        try:
            pipeline.enable_frame_sync()
        except Exception as error:
            print(f"[WARNING] Frame sync를 활성화하지 못했습니다: {error}")
        configure_color_device(pipeline.get_device())
        print("Alignment: Orbbec SDK software Depth-to-Color (no cv2.resize fallback)")
        print("Q: 종료 | S: 현재 RGB-D 평균/Mask 저장")
        for _ in range(args.warmup_frames):
            get_aligned_rgbd(pipeline, align_filter)
        while True:
            color, depth, depth_scale = get_aligned_rgbd(pipeline, align_filter)
            result, depth_vis, overlay, bbox_preview, masked = process_mask(color, depth, args)
            preview = np.hstack([color, depth_vis, cv2.cvtColor(result.mask, cv2.COLOR_GRAY2BGR), overlay])
            preview = cv2.resize(preview, (1280, 200), interpolation=cv2.INTER_AREA)
            cv2.imshow("RGB | Depth | Object Mask | Overlay", preview)
            if result.invalid_ratio_in_bbox > 0.20:
                print(f"[WARNING] Object bbox invalid depth: {result.invalid_ratio_in_bbox:.1%}")
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), ord("Q"), 27):
                break
            if key in (ord("s"), ord("S")):
                color, depth, depth_scale = average_rgbd(pipeline, align_filter, args.average_frames)
                result, depth_vis, overlay, bbox_preview, masked = process_mask(color, depth, args)
                sample_dir = next_sample_dir(output_root)
                save_sample(sample_dir, color, depth, depth_scale, result,
                            depth_vis, overlay, bbox_preview, masked, args)
                print(f"저장 완료: {sample_dir}")
    finally:
        if started:
            pipeline.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
