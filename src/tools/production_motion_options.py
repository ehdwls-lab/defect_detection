"""Shared production platform motion CLI and validated wait configuration."""

from __future__ import annotations

import argparse
import math

from src.platform.motion_diagnostic import (
    MotionWaitConfig, ORIENTATION_TARGET_REACHED_TOLERANCE_DEG,
    Z_TARGET_REACHED_TOLERANCE_CM,
)


def finite_number(raw: str) -> float:
    value = float(raw)
    if not math.isfinite(value):
        raise argparse.ArgumentTypeError("value must be finite")
    return value


def positive_integer(raw: str) -> int:
    value = int(raw)
    if value <= 0:
        raise argparse.ArgumentTypeError("value must be a positive integer")
    return value


def add_production_motion_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--platform-motion-timeout", type=finite_number, default=30.0)
    parser.add_argument("--post-command-guard", type=finite_number, default=0.05)
    parser.add_argument("--stable-samples", type=positive_integer, default=3)
    parser.add_argument("--deadband-observation", type=finite_number, default=0.20)
    parser.add_argument("--fresh-settle", type=finite_number, default=0.10)
    parser.add_argument(
        "--z-target-tolerance", type=finite_number,
        default=Z_TARGET_REACHED_TOLERANCE_CM,
    )
    parser.add_argument(
        "--orientation-target-tolerance", type=finite_number,
        default=ORIENTATION_TARGET_REACHED_TOLERANCE_DEG,
    )


def build_production_motion_wait_config(args: argparse.Namespace) -> MotionWaitConfig:
    config = MotionWaitConfig(
        post_command_guard_s=args.post_command_guard,
        stable_sample_count=args.stable_samples,
        deadband_observation_s=args.deadband_observation,
        fresh_read_settle_s=args.fresh_settle,
        z_target_tolerance_cm=args.z_target_tolerance,
        orientation_target_tolerance_deg=args.orientation_target_tolerance,
    )
    config.validate()
    if config.fresh_read_settle_s >= args.platform_motion_timeout:
        raise ValueError("fresh_read_settle_s must be less than platform_motion_timeout")
    return config
