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

from src.integration.real_pose_planner import RealPosePlanner
from src.platform.motion_diagnostic import MotionDiagnosticError, MotionWaitConfig, PlatformMotionDiagnostic
from src.platform.serial_controller import SerialPlatformConfig, SerialPlatformController


def finite_arg(value: str) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise argparse.ArgumentTypeError("value must be finite")
    return number


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Hardware-gated STM32 platform motion diagnostic")
    parser.add_argument("--port", required=True)
    parser.add_argument("--baud", type=int, default=115200)
    parser.add_argument("--timeout", type=finite_arg, default=10.0)
    parser.add_argument("--post-command-guard", type=finite_arg, default=0.05)
    parser.add_argument("--stable-samples", type=int, default=3)
    parser.add_argument("--deadband-observation", type=finite_arg, default=0.20)
    parser.add_argument(
        "--fresh-settle", type=finite_arg, default=0.10,
        help="diagnostic USB/CDC drain interval between two RX resets (not a production limit)",
    )
    parser.add_argument("--z", type=finite_arg, help="explicit absolute Z target/safe Z in cm")
    parser.add_argument("--roll", type=finite_arg, help="explicit absolute roll target in degrees")
    parser.add_argument("--pitch", type=finite_arg, help="explicit absolute pitch target in degrees")
    parser.add_argument("--execute", action="store_true", help="allow direct Z/R/P test after confirmation")
    parser.add_argument("--snapshot", action="store_true", help="fresh read-only snapshot (also the default without motion)")
    parser.add_argument("--ack-safe-height", action="store_true")
    parser.add_argument("--pose-json")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--execute-pose", action="store_true", help="execute pose after explicit safe Z and confirmation")
    parser.add_argument("--log", help="save telemetry as .json or .csv")
    return parser


def _show(heading: str, telemetry) -> None:
    print(f"{heading}: {telemetry}")


def run(args: argparse.Namespace, *, controller=None, confirm=None) -> int:
    if args.execute and args.execute_pose:
        raise MotionDiagnosticError("choose --execute or --execute-pose, not both")
    if args.snapshot and (args.execute or args.execute_pose):
        raise MotionDiagnosticError("--snapshot cannot be combined with motion execution")
    plan = None
    if args.pose_json:
        plan = RealPosePlanner().plan(args.pose_json)
        if not plan.poses or not plan.metadata.get("platform_motion_allowed"):
            raise MotionDiagnosticError(
                "pose JSON has no reachable metric pose; legacy phase pose fallback is forbidden"
            )
        pose = plan.poses[0]
        print("Pose JSON target")
        print(f"roll: {pose.roll_deg:+.4f} deg")
        print(f"pitch: {pose.pitch_deg:+.4f} deg")
        print("Z: NOT PROVIDED")
        print("legacy Z: IGNORED")
        if not args.execute_pose:
            print("DRY RUN - no command sent")
    if args.execute_pose:
        if plan is None:
            raise MotionDiagnosticError("--execute-pose requires --pose-json")
        if args.z is None:
            raise MotionDiagnosticError("--execute-pose requires user-provided --z safe height")
        if not args.ack_safe_height:
            raise MotionDiagnosticError("--execute-pose requires --ack-safe-height")

    owned = controller is None
    controller = controller or SerialPlatformController(
        SerialPlatformConfig(args.port, baudrate=args.baud, read_timeout_s=min(args.timeout, 1.0))
    )
    wait_config = MotionWaitConfig(
        post_command_guard_s=args.post_command_guard,
        stable_sample_count=args.stable_samples,
        deadband_observation_s=args.deadband_observation,
        fresh_read_settle_s=args.fresh_settle,
    )
    diagnostic = PlatformMotionDiagnostic(
        controller, timeout_s=args.timeout, wait_config=wait_config, confirm=confirm,
    )
    try:
        controller.connect()
        before = diagnostic.read_before()
        _show("before", before)
        if args.execute_pose:
            pose = plan.poses[0]
            after = diagnostic.execute_pose(
                safe_z_cm=args.z, roll_deg=pose.roll_deg, pitch_deg=pose.pitch_deg,
                ack_safe_height=args.ack_safe_height,
            )
            _show("after", after)
        elif args.execute:
            if args.z is None and args.roll is None and args.pitch is None:
                raise MotionDiagnosticError("--execute requires --z, --roll, or --pitch")
            after = before
            if args.z is not None:
                after = diagnostic.execute_z(args.z)
                _show("after Z", after)
            if args.roll is not None or args.pitch is not None:
                after = diagnostic.execute_orientation(
                    roll_deg=args.roll, pitch_deg=args.pitch, before=after,
                    ack_safe_height=args.ack_safe_height,
                )
                _show("after orientation", after)
        else:
            if any(value is not None for value in (args.z, args.roll, args.pitch)):
                print("DRY RUN - requested target shown; --execute absent, no command sent")
            else:
                print("FRESH READ-ONLY SNAPSHOT - no command sent")
        if args.log:
            diagnostic.log.save(args.log)
            print(f"telemetry log: {args.log}")
        return 0
    finally:
        if owned:
            controller.close()


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    try:
        return run(args)
    except (MotionDiagnosticError, ValueError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    raise SystemExit(main())
