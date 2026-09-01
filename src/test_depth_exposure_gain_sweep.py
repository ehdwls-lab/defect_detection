from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

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
FPS = 15
FORMAT = OBFormat.Y16
TIMEOUT_MS = 3000


@dataclass
class SettingResult:
    exposure: int
    gain: int
    valid_ratio: float
    stable_valid_ratio: float
    median_depth_mm: float
    temporal_std_median_mm: float
    temporal_std_p95_mm: float
    frame_delta_median_mm: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Gemini 336L Depth exposure/gain sweep. "
            "Depth AE 기준값에서 exposure/gain을 낮춰가며 "
            "유효 depth 비율과 temporal noise를 비교한다."
        )
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "results" / "depth_exposure_gain_sweep",
    )
    parser.add_argument("--frames-per-setting", type=int, default=24)
    parser.add_argument("--settle-frames", type=int, default=15)
    parser.add_argument("--auto-warmup-frames", type=int, default=40)

    parser.add_argument(
        "--exposure-factors",
        type=str,
        default="1.0,0.75,0.5,0.35,0.25",
    )
    parser.add_argument(
        "--gain-factors",
        type=str,
        default="1.0,0.75,0.5",
    )

    parser.add_argument(
        "--center-roi-ratio",
        type=float,
        default=0.70,
    )

    parser.add_argument(
        "--preset",
        type=str,
        default="",
        help='예: "High Accuracy". 빈 문자열이면 현재 preset 유지.',
    )

    return parser.parse_args()


def parse_factors(text: str) -> list[float]:
    values = []

    for token in text.split(","):
        token = token.strip()

        if not token:
            continue

        value = float(token)

        if value <= 0:
            raise ValueError("factor는 0보다 커야 합니다.")

        values.append(value)

    if not values:
        raise ValueError("factor 목록이 비었습니다.")

    return values


def find_depth_profile(pipeline: Pipeline):
    profiles = pipeline.get_stream_profile_list(
        OBSensorType.DEPTH_SENSOR
    )

    exact = None
    fallback = None

    for i in range(profiles.get_count()):
        p = profiles.get_stream_profile_by_index(i)

        if p.get_format() != FORMAT:
            continue

        if fallback is None:
            fallback = p

        if (
            p.get_width() == WIDTH
            and p.get_height() == HEIGHT
            and p.get_fps() == FPS
        ):
            exact = p
            break

    if exact is not None:
        return exact

    if fallback is not None:
        print(
            "[WARN] 1280x800@15 Y16 exact profile을 못 찾아 "
            f"{fallback.get_width()}x{fallback.get_height()}@"
            f"{fallback.get_fps()} profile을 사용합니다."
        )
        return fallback

    raise RuntimeError("Y16 Depth profile을 찾지 못했습니다.")


def wait_depth_frame(pipeline: Pipeline):
    while True:
        frames = pipeline.wait_for_frames(TIMEOUT_MS)

        if frames is None:
            continue

        frame = frames.get_depth_frame()

        if frame is not None:
            return frame


def depth_to_mm(depth_frame) -> np.ndarray:
    h = depth_frame.get_height()
    w = depth_frame.get_width()
    scale = float(depth_frame.get_depth_scale())

    raw = np.frombuffer(
        depth_frame.get_data(),
        dtype=np.uint16,
    ).reshape(h, w)

    return raw.astype(np.float32) * scale


def center_roi_slices(shape, ratio: float):
    h, w = shape

    ratio = float(
        np.clip(
            ratio,
            0.1,
            1.0,
        )
    )

    rw = int(round(w * ratio))
    rh = int(round(h * ratio))

    x1 = (w - rw) // 2
    y1 = (h - rh) // 2

    return (
        slice(y1, y1 + rh),
        slice(x1, x1 + rw),
    )


def get_int_range(device, prop):
    try:
        rng = device.get_int_property_range(prop)

        default_value = None
        for attr in ("def", "default"):
            try:
                default_value = int(getattr(rng, attr))
                break
            except Exception:
                pass

        step = 1
        try:
            step = int(rng.step)
        except Exception:
            pass

        return (
            int(rng.min),
            int(rng.max),
            step,
            default_value,
        )
    except Exception:
        return None


def clamp_to_range(
    value: int,
    prop_range,
) -> int:
    value = int(value)

    if prop_range is None:
        return value

    low, high, step, _ = prop_range

    value = int(
        np.clip(
            value,
            low,
            high,
        )
    )

    if step > 0:
        value = (
            low
            + round(
                (value - low)
                / step
            )
            * step
        )

        value = int(
            np.clip(
                value,
                low,
                high,
            )
        )

    return value


def maybe_load_preset(
    device,
    requested_name: str,
):
    try:
        presets = list(
            device.get_available_preset_list()
        )
    except Exception as exc:
        print(f"[PRESET] list 조회 실패: {exc}")
        return

    if presets:
        print("[PRESET] available:", presets)

    if not requested_name:
        try:
            print(
                "[PRESET] current:",
                device.get_current_preset_name(),
            )
        except Exception:
            pass
        return

    target = None

    for name in presets:
        if str(name).lower() == requested_name.lower():
            target = name
            break

    if target is None:
        print(
            f'[WARN] preset "{requested_name}"를 찾지 못해 현재 preset 유지'
        )
        return

    try:
        device.load_preset(target)
        print(
            "[PRESET] loaded:",
            device.get_current_preset_name(),
        )
    except Exception as exc:
        print(f"[WARN] preset load 실패: {exc}")


def set_depth_manual(
    device,
    exposure: int,
    gain: int,
):
    device.set_bool_property(
        OBPropertyID.OB_PROP_DEPTH_AUTO_EXPOSURE_BOOL,
        False,
    )

    device.set_int_property(
        OBPropertyID.OB_PROP_DEPTH_EXPOSURE_INT,
        int(exposure),
    )

    device.set_int_property(
        OBPropertyID.OB_PROP_DEPTH_GAIN_INT,
        int(gain),
    )


def auto_reference_values(
    pipeline: Pipeline,
    device,
    warmup_frames: int,
):
    device.set_bool_property(
        OBPropertyID.OB_PROP_DEPTH_AUTO_EXPOSURE_BOOL,
        True,
    )

    print(
        f"[AUTO] Depth AE ON -> {warmup_frames} frames warmup"
    )

    for _ in range(warmup_frames):
        wait_depth_frame(pipeline)

    exposure = device.get_int_property(
        OBPropertyID.OB_PROP_DEPTH_EXPOSURE_INT
    )

    gain = device.get_int_property(
        OBPropertyID.OB_PROP_DEPTH_GAIN_INT
    )

    print(
        f"[AUTO REFERENCE] exposure={exposure}, gain={gain}"
    )

    return int(exposure), int(gain)


def render_depth(
    depth_mm: np.ndarray,
):
    valid = depth_mm > 0

    if not np.any(valid):
        return np.zeros(
            (
                depth_mm.shape[0],
                depth_mm.shape[1],
                3,
            ),
            dtype=np.uint8,
        )

    vals = depth_mm[valid]

    min_mm = float(
        np.percentile(vals, 2)
    )

    max_mm = float(
        np.percentile(vals, 98)
    )

    if max_mm <= min_mm:
        max_mm = min_mm + 1.0

    norm = np.zeros_like(
        depth_mm,
        dtype=np.float32,
    )

    norm[valid] = (
        depth_mm[valid]
        - min_mm
    ) / (
        max_mm
        - min_mm
    )

    norm = np.clip(
        norm,
        0.0,
        1.0,
    )

    img = (
        norm * 255.0
    ).astype(np.uint8)

    color = cv2.applyColorMap(
        img,
        cv2.COLORMAP_JET,
    )

    color[
        ~valid
    ] = 0

    return color


def collect_setting(
    pipeline: Pipeline,
    device,
    exposure: int,
    gain: int,
    settle_frames: int,
    frames_per_setting: int,
    roi_ratio: float,
):
    set_depth_manual(
        device,
        exposure,
        gain,
    )

    for _ in range(settle_frames):
        wait_depth_frame(pipeline)

    frames = []

    for _ in range(frames_per_setting):
        frame = wait_depth_frame(
            pipeline
        )

        frames.append(
            depth_to_mm(frame)
        )

    stack = np.stack(
        frames,
        axis=0,
    ).astype(np.float32)

    ys, xs = center_roi_slices(
        stack.shape[1:],
        roi_ratio,
    )

    roi = stack[
        :,
        ys,
        xs,
    ]

    valid = roi > 0

    valid_ratio = float(
        np.mean(valid)
    )

    valid_frequency = np.mean(
        valid,
        axis=0,
    )

    stable_valid_ratio = float(
        np.mean(
            valid_frequency
            >= 0.8
        )
    )

    valid_values = roi[
        valid
    ]

    median_depth_mm = (
        float(np.median(valid_values))
        if valid_values.size
        else float("nan")
    )

    reshaped = roi.reshape(
        roi.shape[0],
        -1,
    )

    temporal_stds = []

    min_valid_frames = max(
        3,
        frames_per_setting // 2,
    )

    for idx in range(
        reshaped.shape[1]
    ):
        col = reshaped[:, idx]

        vals = col[
            col > 0
        ]

        if vals.size >= min_valid_frames:
            temporal_stds.append(
                float(
                    np.std(vals)
                )
            )

    if temporal_stds:
        temporal_std_median_mm = float(
            np.median(
                temporal_stds
            )
        )

        temporal_std_p95_mm = float(
            np.percentile(
                temporal_stds,
                95,
            )
        )
    else:
        temporal_std_median_mm = float("inf")
        temporal_std_p95_mm = float("inf")

    frame_deltas = []

    for i in range(
        1,
        roi.shape[0],
    ):
        a = roi[
            i - 1
        ]

        b = roi[
            i
        ]

        pair_valid = (
            (a > 0)
            & (b > 0)
        )

        if np.any(pair_valid):
            frame_deltas.append(
                float(
                    np.median(
                        np.abs(
                            b[pair_valid]
                            - a[pair_valid]
                        )
                    )
                )
            )

    frame_delta_median_mm = (
        float(
            np.median(
                frame_deltas
            )
        )
        if frame_deltas
        else float("inf")
    )

    median_frame = np.median(
        stack,
        axis=0,
    ).astype(np.float32)

    valid_mask = (
        np.mean(
            stack > 0,
            axis=0,
        )
        >= 0.8
    ).astype(np.uint8) * 255

    return (
        SettingResult(
            exposure=exposure,
            gain=gain,
            valid_ratio=valid_ratio,
            stable_valid_ratio=stable_valid_ratio,
            median_depth_mm=median_depth_mm,
            temporal_std_median_mm=temporal_std_median_mm,
            temporal_std_p95_mm=temporal_std_p95_mm,
            frame_delta_median_mm=frame_delta_median_mm,
        ),
        median_frame,
        valid_mask,
    )


def choose_recommended(
    results: list[SettingResult],
):
    if not results:
        return None

    max_stable = max(
        r.stable_valid_ratio
        for r in results
    )

    eligible = [
        r
        for r in results
        if r.stable_valid_ratio
        >= 0.80 * max_stable
    ]

    if not eligible:
        eligible = results

    eligible.sort(
        key=lambda r: (
            r.temporal_std_median_mm,
            r.frame_delta_median_mm,
            -r.stable_valid_ratio,
        )
    )

    return eligible[0]


def save_csv(
    path: Path,
    results: Iterable[SettingResult],
):
    with path.open(
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as f:
        writer = csv.writer(f)

        writer.writerow(
            [
                "exposure",
                "gain",
                "valid_ratio",
                "stable_valid_ratio",
                "median_depth_mm",
                "temporal_std_median_mm",
                "temporal_std_p95_mm",
                "frame_delta_median_mm",
            ]
        )

        for r in results:
            writer.writerow(
                [
                    r.exposure,
                    r.gain,
                    f"{r.valid_ratio:.6f}",
                    f"{r.stable_valid_ratio:.6f}",
                    f"{r.median_depth_mm:.3f}",
                    f"{r.temporal_std_median_mm:.3f}",
                    f"{r.temporal_std_p95_mm:.3f}",
                    f"{r.frame_delta_median_mm:.3f}",
                ]
            )


def main():
    args = parse_args()

    exposure_factors = parse_factors(
        args.exposure_factors
    )

    gain_factors = parse_factors(
        args.gain_factors
    )

    output_dir = args.output_dir

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    pipeline = Pipeline()
    config = Config()
    started = False

    try:
        profile = find_depth_profile(
            pipeline
        )

        config.enable_stream(
            profile
        )

        pipeline.start(
            config
        )

        started = True

        device = pipeline.get_device()

        maybe_load_preset(
            device,
            args.preset,
        )

        exp_range = get_int_range(
            device,
            OBPropertyID.OB_PROP_DEPTH_EXPOSURE_INT,
        )

        gain_range = get_int_range(
            device,
            OBPropertyID.OB_PROP_DEPTH_GAIN_INT,
        )

        print("=" * 86)
        print("Gemini 336L Depth Exposure/Gain Sweep")
        print(
            f"Depth profile: "
            f"{profile.get_width()}x{profile.get_height()} @"
            f"{profile.get_fps()}fps {profile.get_format()}"
        )
        print("Exposure range:", exp_range)
        print("Gain range:", gain_range)
        print(
            f"Evaluation center ROI ratio: {args.center_roi_ratio:.2f}"
        )
        print("=" * 86)

        auto_exp, auto_gain = auto_reference_values(
            pipeline,
            device,
            args.auto_warmup_frames,
        )

        setting_pairs = []

        for ef in exposure_factors:
            for gf in gain_factors:
                exp = clamp_to_range(
                    int(round(auto_exp * ef)),
                    exp_range,
                )

                gain = clamp_to_range(
                    int(round(auto_gain * gf)),
                    gain_range,
                )

                pair = (
                    exp,
                    gain,
                )

                if pair not in setting_pairs:
                    setting_pairs.append(
                        pair
                    )

        print(
            f"[SWEEP] {len(setting_pairs)} settings"
        )

        results = []

        for index, (
            exposure,
            gain,
        ) in enumerate(
            setting_pairs,
            start=1,
        ):
            print(
                f"\n[{index}/{len(setting_pairs)}] "
                f"exposure={exposure}, gain={gain}"
            )

            (
                result,
                median_frame,
                valid_mask,
            ) = collect_setting(
                pipeline=pipeline,
                device=device,
                exposure=exposure,
                gain=gain,
                settle_frames=args.settle_frames,
                frames_per_setting=args.frames_per_setting,
                roi_ratio=args.center_roi_ratio,
            )

            results.append(
                result
            )

            print(
                f"  valid={result.valid_ratio*100:.2f}% | "
                f"stable_valid={result.stable_valid_ratio*100:.2f}% | "
                f"temp_std_med={result.temporal_std_median_mm:.2f} mm | "
                f"temp_std_p95={result.temporal_std_p95_mm:.2f} mm | "
                f"frame_delta={result.frame_delta_median_mm:.2f} mm"
            )

            stem = (
                f"exp_{exposure}_gain_{gain}"
            )

            cv2.imwrite(
                str(
                    output_dir
                    / f"{stem}_depth.png"
                ),
                render_depth(
                    median_frame
                ),
            )

            cv2.imwrite(
                str(
                    output_dir
                    / f"{stem}_stable_valid.png"
                ),
                valid_mask,
            )

        save_csv(
            output_dir
            / "depth_sweep_results.csv",
            results,
        )

        recommended = choose_recommended(
            results
        )

        lines = [
            f"auto_reference_exposure={auto_exp}",
            f"auto_reference_gain={auto_gain}",
            f"exposure_range={exp_range}",
            f"gain_range={gain_range}",
            f"preset={args.preset or 'current'}",
            "",
        ]

        if recommended is not None:
            lines.extend(
                [
                    "recommended:",
                    f"  exposure={recommended.exposure}",
                    f"  gain={recommended.gain}",
                    (
                        "  stable_valid_ratio="
                        f"{recommended.stable_valid_ratio:.6f}"
                    ),
                    (
                        "  temporal_std_median_mm="
                        f"{recommended.temporal_std_median_mm:.3f}"
                    ),
                    (
                        "  temporal_std_p95_mm="
                        f"{recommended.temporal_std_p95_mm:.3f}"
                    ),
                    (
                        "  frame_delta_median_mm="
                        f"{recommended.frame_delta_median_mm:.3f}"
                    ),
                ]
            )

        (
            output_dir
            / "summary.txt"
        ).write_text(
            "\n".join(
                lines
            )
            + "\n",
            encoding="utf-8",
        )

        print("\n" + "=" * 86)

        if recommended is not None:
            print(
                "[RECOMMENDED] "
                f"Depth Exposure={recommended.exposure}, "
                f"Gain={recommended.gain}"
            )
            print(
                f"stable valid={recommended.stable_valid_ratio*100:.2f}% | "
                f"temporal std median={recommended.temporal_std_median_mm:.2f} mm"
            )

        print(
            "[SAVE]",
            (
                output_dir
                / "depth_sweep_results.csv"
            ).resolve(),
        )

        print("=" * 86)

    finally:
        if started:
            pipeline.stop()

        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
