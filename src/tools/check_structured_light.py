from __future__ import annotations

import argparse
import sys
from pathlib import Path

if __package__ in (None, ""):
    root = Path(__file__).resolve().parents[2]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

from src.integration.structured_light_runner import ShellStructuredLightConfig, ShellStructuredLightRunner


def main() -> int:
    parser = argparse.ArgumentParser(description="Structured-light preflight; execution requires --execute")
    parser.add_argument("--subsystem-root", type=Path, required=True)
    parser.add_argument("--result-root", type=Path, required=True)
    parser.add_argument("--timeout", type=float, default=900.0)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    runner = ShellStructuredLightRunner(ShellStructuredLightConfig(
        subsystem_root=args.subsystem_root, result_root=args.result_root, timeout_sec=args.timeout,
    ))
    issues = runner.preflight()
    if issues:
        for issue in issues:
            print(f"FAIL: {issue}")
        return 2
    print("Structured-light preflight passed")
    if not args.execute:
        print("No scan executed (pass --execute explicitly to run the shell pipeline)")
        return 0
    result = runner.run_scan()
    print(f"scan complete: {result.result_directory}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
