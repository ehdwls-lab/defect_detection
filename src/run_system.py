from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

# Support the required direct invocation: ``python3 src/run_system.py``.
if __package__ in (None, ""):
    project_root = Path(__file__).resolve().parents[1]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

from src.system.factory import HardwareModeNotReadyError, build_system


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Integrated defect-detection system")
    parser.add_argument("--mode", choices=("mock", "hardware"), required=True)
    parser.add_argument("--once", action="store_true", help="run one object workflow")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    if not args.once:
        raise SystemExit("Phase 2 supports --once only")
    try:
        controller = build_system(args.mode)
    except HardwareModeNotReadyError as exc:
        logging.error(str(exc))
        return 2
    result = controller.run_once()
    print(json.dumps(result.to_dict(), ensure_ascii=False, indent=2))
    return 0 if result.success else 1


if __name__ == "__main__":
    raise SystemExit(main())
