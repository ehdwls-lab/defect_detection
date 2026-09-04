from __future__ import annotations

import argparse
import logging
import math
import sys
from pathlib import Path

if __package__ in (None, ""):
    root = Path(__file__).resolve().parents[2]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

from src.platform.serial_controller import SerialPlatformConfig, SerialPlatformController


def finite_arg(value: str) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise argparse.ArgumentTypeError("value must be finite")
    return number


def main() -> int:
    parser = argparse.ArgumentParser(description="Read-only STM32 telemetry diagnostic")
    parser.add_argument("--port", required=True)
    parser.add_argument("--count", type=int, default=10)
    parser.add_argument("--timeout", type=finite_arg, default=2.0)
    parser.add_argument(
        "--fresh-settle", type=finite_arg, default=0.10,
        help="diagnostic USB/CDC drain interval between two RX resets",
    )
    parser.add_argument("--snapshot", action="store_true", help="read one fresh telemetry packet")
    args = parser.parse_args()
    if args.count <= 0:
        parser.error("--count must be positive")
    if args.fresh_settle < 0:
        parser.error("--fresh-settle must be non-negative")
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    controller = SerialPlatformController(SerialPlatformConfig(args.port, read_timeout_s=args.timeout))
    try:
        controller.connect()
        count = 1 if args.snapshot else args.count
        print(controller.read_fresh_telemetry(args.timeout, settle_s=args.fresh_settle))
        for _ in range(count - 1):
            print(controller.read_telemetry(args.timeout))
        return 0
    finally:
        controller.close()


if __name__ == "__main__":
    raise SystemExit(main())
