from __future__ import annotations

import argparse
import sys
from pathlib import Path

if __package__ in (None, ""):
    root = Path(__file__).resolve().parents[2]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

def main() -> int:
    parser = argparse.ArgumentParser(description="Explicit Orbbec RGB+Depth capture diagnostic")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--warmup-frames", type=int, default=30)
    args = parser.parse_args()
    import cv2
    import numpy as np
    from src.camera.orbbec_controller import OrbbecCameraController
    camera = OrbbecCameraController(warmup_frames=args.warmup_frames)
    try:
        camera.start()
        frame = camera.capture()
        args.output_dir.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(args.output_dir / "color.png"), frame.color_bgr)
        np.save(args.output_dir / "depth_mm.npy", frame.depth_mm)
        print(f"saved RGB+Depth diagnostic: {args.output_dir}")
        return 0
    finally:
        camera.close()


if __name__ == "__main__":
    raise SystemExit(main())
