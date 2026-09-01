from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

if __package__ in (None, ""):
    root = Path(__file__).resolve().parents[2]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

from src.platform.serial_controller import SerialPlatformConfig, SerialPlatformController


def main() -> int:
    parser = argparse.ArgumentParser(description="Read-only STM32 telemetry diagnostic")
    parser.add_argument("--port", required=True)
    parser.add_argument("--count", type=int, default=10)
    parser.add_argument("--timeout", type=float, default=2.0)
    args = parser.parse_args()
    if args.count <= 0:
        parser.error("--count must be positive")
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    controller = SerialPlatformController(SerialPlatformConfig(args.port, read_timeout_s=args.timeout))
    try:
        controller.connect()
        for _ in range(args.count):
            print(controller.read_telemetry(args.timeout))
        return 0
    finally:
        controller.close()


if __name__ == "__main__":
    raise SystemExit(main())
