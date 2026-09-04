from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.integration.projector_controller import OpenCVProjectorController


def main() -> int:
    parser = argparse.ArgumentParser(description="Manual BLACK/WHITE/PHASE/BLACK projector diagnostic")
    parser.add_argument("--monitor", default="HDMI-0")
    parser.add_argument("--delay", type=float, default=1.0)
    args = parser.parse_args()
    subsystem = Path(__file__).resolve().parents[2] / "서영 파트 파일"
    sys.path.insert(0, str(subsystem))
    from structured_light_projector import select_projector_monitor, xrandr_monitors
    selected = select_projector_monitor(xrandr_monitors(), args.monitor)
    if selected is None:
        raise RuntimeError(f"projector monitor not found: {args.monitor}")
    controller = OpenCVProjectorController(selected)
    controller.open()
    try:
        controller.show_black(); print("BLACK"); time.sleep(args.delay)
        # WHITE is diagnostic-only and does not become part of the production interface.
        controller.show_white_diagnostic()
        print("WHITE"); time.sleep(args.delay)
        for name in controller.PHASE_ORDER:
            controller.show_phase(name); print(f"PHASE {name}"); time.sleep(args.delay)
        controller.show_black(); print("BLACK")
        import cv2
        cv2.waitKey(max(1, int(args.delay * 1000)))
        return 0
    finally:
        controller.show_black()
        controller.close()


if __name__ == "__main__":
    raise SystemExit(main())
