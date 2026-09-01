from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
SUBSYSTEM_ROOT = REPOSITORY_ROOT / "서영 파트 파일"
if str(SUBSYSTEM_ROOT) not in sys.path:
    sys.path.insert(0, str(SUBSYSTEM_ROOT))

from structured_light_projector import (  # noqa: E402
    PHASES,
    PRODUCTION_AMPLITUDE,
    PRODUCTION_BASE,
    PRODUCTION_DIRECTION,
    PRODUCTION_PERIOD,
    production_phase_patterns,
    select_projector_monitor,
    xrandr_monitors,
)


WINDOW_NAME = "STRUCTURED LIGHT PROJECTOR DIAGNOSTIC"


def pattern_statistics(name: str, image: np.ndarray, phase: float | None = None) -> None:
    phase_text = "n/a" if phase is None else f"{np.degrees(phase):.0f} deg"
    print(
        f"pattern={name} period={PRODUCTION_PERIOD} phase={phase_text} "
        f"shape={image.shape} min={int(image.min())} max={int(image.max())} "
        f"mean={float(image.mean()):.3f}"
    )


def coverage_pattern(width: int, height: int) -> np.ndarray:
    image = np.zeros((height, width, 3), dtype=np.uint8)
    margin = max(20, min(width, height) // 30)
    cv2.rectangle(image, (margin, margin), (width // 3, height // 3), (255, 255, 255), -1)
    cv2.rectangle(
        image, (2 * width // 3, margin), (width - margin, height // 3), (128, 128, 128), -1,
    )
    for x in range(margin, width // 3, max(8, width // 100)):
        cv2.line(image, (x, 2 * height // 3), (x, height - margin), (255, 255, 255), 2)
    scale = max(1.0, min(width / 1920.0, height / 1080.0) * 2.0)
    cv2.putText(
        image, "PROJECTOR TEST", (2 * width // 3, 5 * height // 6),
        cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), max(2, int(scale * 2)), cv2.LINE_AA,
    )
    return image


def verify_and_save_patterns(
    width: int, height: int, save_dir: Path,
) -> dict[str, np.ndarray]:
    patterns = production_phase_patterns(width, height)
    save_dir.mkdir(parents=True, exist_ok=True)
    for phase, name in PHASES:
        image = patterns[name]
        if image.shape != (height, width, 3):
            raise RuntimeError(f"phase_{name} shape mismatch: {image.shape}")
        if int(image.max()) <= int(image.min()):
            raise RuntimeError(f"phase_{name} has no dynamic range")
        if not cv2.imwrite(str(save_dir / f"phase_{name}.png"), image):
            raise RuntimeError(f"phase_{name}.png 저장 실패")
        pattern_statistics(f"PHASE {name}", image, phase)
    for index, left in enumerate(patterns.values()):
        for right in list(patterns.values())[index + 1:]:
            if np.array_equal(left, right):
                raise RuntimeError("서로 동일한 phase pattern이 생성됐습니다")
    print(f"numeric verification: PASS, saved={save_dir}")
    return patterns


def prepare_window(monitor: dict[str, object]) -> None:
    black = np.zeros((int(monitor["h"]), int(monitor["w"]), 3), dtype=np.uint8)
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.imshow(WINDOW_NAME, black)
    cv2.waitKey(500)
    cv2.moveWindow(WINDOW_NAME, int(monitor["x"]), int(monitor["y"]))
    cv2.waitKey(500)
    cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    cv2.imshow(WINDOW_NAME, black)
    cv2.waitKey(1000)
    try:
        rect = cv2.getWindowImageRect(WINDOW_NAME)
        print(f"created window: rect={rect}, requested_fullscreen=True")
    except cv2.error as exc:
        print(f"created window: getWindowImageRect unavailable: {exc}")


def show_interactive(monitor: dict[str, object], patterns: dict[str, np.ndarray], delay_ms: int) -> None:
    width, height = int(monitor["w"]), int(monitor["h"])
    screens = {
        ord("1"): ("BLACK", np.zeros((height, width, 3), dtype=np.uint8)),
        ord("2"): ("WHITE", np.full((height, width, 3), 255, dtype=np.uint8)),
        ord("3"): ("GRAY 50%", np.full((height, width, 3), 128, dtype=np.uint8)),
        ord("4"): ("PHASE 000", patterns["000"]),
        ord("5"): ("PHASE 090", patterns["090"]),
        ord("6"): ("PHASE 180", patterns["180"]),
        ord("7"): ("PHASE 270", patterns["270"]),
        ord("9"): ("COVERAGE", coverage_pattern(width, height)),
    }
    print("keys: 1 black, 2 white, 3 gray, 4/5/6/7 phase, 8 auto sequence, 9 coverage, Q/ESC quit")
    current = screens[ord("1")]
    auto = False
    sequence_index = 0
    while True:
        if auto:
            name = PHASES[sequence_index][1]
            current = (f"AUTO PHASE {name}", patterns[name])
            sequence_index = (sequence_index + 1) % len(PHASES)
        cv2.imshow(WINDOW_NAME, current[1])
        pattern_statistics(current[0], current[1])
        key = cv2.waitKey(delay_ms if auto else 50) & 0xFF
        if key in (27, ord("q"), ord("Q")):
            break
        if key == ord("8"):
            auto = True
        elif key in screens:
            auto = False
            current = screens[key]
    cv2.destroyWindow(WINDOW_NAME)


def main() -> int:
    parser = argparse.ArgumentParser(description="카메라를 열지 않는 구조광 projector diagnostic")
    parser.add_argument("--monitor", default="auto", help="xrandr display name 또는 auto")
    parser.add_argument("--save-dir", type=Path, default=Path("/tmp/projector_test"))
    parser.add_argument("--array-only", action="store_true", help="GUI 없이 production pattern 배열만 검증")
    parser.add_argument("--width", type=int, default=1920, help="array-only fallback width")
    parser.add_argument("--height", type=int, default=1080, help="array-only fallback height")
    parser.add_argument("--sequence-delay-ms", type=int, default=800)
    args = parser.parse_args()

    print(f"session: XDG_SESSION_TYPE={os.environ.get('XDG_SESSION_TYPE')!r}, DISPLAY={os.environ.get('DISPLAY')!r}")
    print(
        f"production contract: direction={PRODUCTION_DIRECTION}, period={PRODUCTION_PERIOD}, "
        f"base={PRODUCTION_BASE}, amplitude={PRODUCTION_AMPLITUDE}"
    )
    monitors = xrandr_monitors()
    print("monitors:")
    for item in monitors:
        print(
            f"  {item['name']} {item['w']}x{item['h']} "
            f"({item['x']},{item['y']}) primary={item['primary']}"
        )
    selected = select_projector_monitor(monitors, args.monitor)
    if selected is None:
        if not args.array_only:
            raise RuntimeError(f"projector display를 선택할 수 없습니다: {args.monitor}")
        selected = {"name": "array-only", "w": args.width, "h": args.height, "x": 0, "y": 0, "primary": False}
    print(
        f"selected display: name={selected['name']} resolution={selected['w']}x{selected['h']} "
        f"x={selected['x']} y={selected['y']}"
    )
    patterns = verify_and_save_patterns(int(selected["w"]), int(selected["h"]), args.save_dir)
    if args.array_only:
        print("array-only mode: no projector window or camera was opened")
        return 0
    prepare_window(selected)
    print(f"actual pattern shape: {(int(selected['h']), int(selected['w']), 3)}")
    show_interactive(selected, patterns, max(50, args.sequence_delay_ms))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
