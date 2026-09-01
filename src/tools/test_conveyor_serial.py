from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

if __package__ in (None, ""):
    root = Path(__file__).resolve().parents[2]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

from src.conveyor.serial_controller import SerialConveyorConfig, SerialConveyorController


def main() -> int:
    parser = argparse.ArgumentParser(description="Explicit conveyor movement diagnostic")
    parser.add_argument("--port", required=True)
    parser.add_argument("--direction", required=True, choices=("F", "B"))
    parser.add_argument("--steps", required=True, type=int)
    parser.add_argument("--timeout", type=float, default=30.0)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    config = SerialConveyorConfig(
        port=args.port,
        inspection_direction=args.direction,
        inspection_steps=args.steps,
        exit_direction=args.direction,
        exit_steps=args.steps,
        timeout_sec=args.timeout,
    )
    controller = SerialConveyorController(config)
    try:
        controller.connect()
        controller.move_steps(args.direction, args.steps)
        controller.wait_until_stopped()
        logging.info("[CONVEYOR] Target Reached confirmed")
        return 0
    finally:
        controller.close()


if __name__ == "__main__":
    raise SystemExit(main())
