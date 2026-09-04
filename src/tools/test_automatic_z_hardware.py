from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.camera.orbbec_controller import OrbbecCameraController
from src.inspection.hardware_z_search import (
    CandidateArtifactStore, HardwareAutomaticZSearch, HardwareZSearchConfig,
    SensorQualityConfig, SensorQualityEvaluator, SurfaceReadinessEvaluator,
)
from src.integration.projector_controller import OpenCVProjectorController
from src.platform.motion_diagnostic import DiagnosticZMover, MotionWaitConfig, PlatformMotionDiagnostic
from src.platform.serial_controller import SerialPlatformConfig, SerialPlatformController


def finite(value: str) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise argparse.ArgumentTypeError("value must be finite")
    return number


def candidates(value: str) -> tuple[float, ...]:
    try:
        parsed = tuple(float(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as exc:
        raise argparse.ArgumentTypeError("candidates must be comma-separated numbers") from exc
    if not parsed or not all(math.isfinite(item) for item in parsed):
        raise argparse.ArgumentTypeError("at least one finite candidate is required")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Dry-run-first RGB+Depth Automatic Z hardware diagnostic")
    parser.add_argument("--port", required=True)
    parser.add_argument("--candidates", required=True, type=candidates)
    parser.add_argument("--z-max", required=True, type=finite)
    parser.add_argument(
        "--selection-policy",
        choices=("highest_passing_readiness", "best_quality_score", "best_surface_coverage"),
        default="highest_passing_readiness",
    )
    parser.add_argument("--roll", type=finite)
    parser.add_argument("--pitch", type=finite)
    parser.add_argument("--quality-config", help="required threshold/weight JSON for execution")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--surface-area-weight", type=finite, default=0.6)
    parser.add_argument("--depth-valid-weight", type=finite, default=0.4)
    parser.add_argument("--result", type=Path)
    parser.add_argument("--monitor", default="HDMI-0")
    parser.add_argument("--timeout", type=finite, default=10.0)
    parser.add_argument("--post-command-guard", type=finite, default=0.05)
    parser.add_argument("--stable-samples", type=int, default=3)
    parser.add_argument("--deadband-observation", type=finite, default=0.20)
    parser.add_argument(
        "--fresh-settle", type=finite, default=0.10,
        help="diagnostic USB/CDC drain interval between two RX resets",
    )
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--ack-mechanical-z-range", action="store_true")
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    config = HardwareZSearchConfig(
        args.candidates, args.z_max, args.timeout, args.selection_policy,
        surface_area_weight=args.surface_area_weight,
        depth_valid_weight=args.depth_valid_weight,
    )
    wait_config = MotionWaitConfig(
        post_command_guard_s=args.post_command_guard,
        stable_sample_count=args.stable_samples,
        deadband_observation_s=args.deadband_observation,
        fresh_read_settle_s=args.fresh_settle,
    )
    try:
        config.validate()
        wait_config.validate()
        if wait_config.fresh_read_settle_s >= args.timeout:
            raise ValueError("fresh_read_settle_s must be less than timeout")
    except ValueError as exc:
        parser.error(str(exc))
    print(f"Z candidates: {list(args.candidates)}")
    print(f"user mechanical/software z_max: {args.z_max}")
    print(f"selection_policy: {args.selection_policy}")
    print("source: explicit CLI candidates; structured-light legacy Z is not used")
    if not args.execute:
        print("DRY RUN - no platform command and no camera capture")
        return 0
    if not args.ack_mechanical_z_range:
        parser.error("--execute requires --ack-mechanical-z-range")
    if args.roll is None or args.pitch is None:
        parser.error("--execute requires explicit --roll and --pitch")
    if not args.quality_config:
        parser.error("--execute requires --quality-config")
    try:
        quality_config = SensorQualityConfig.from_json(args.quality_config)
        quality_config.require_execution_ready(args.selection_policy)
    except (OSError, ValueError, TypeError) as exc:
        parser.error(str(exc))
    if input("Type EXECUTE to move through the listed absolute Z candidates: ").strip() != "EXECUTE":
        parser.error("execution cancelled; no hardware was opened")

    subsystem = Path(__file__).resolve().parents[2] / "서영 파트 파일"
    sys.path.insert(0, str(subsystem))
    from structured_light_projector import select_projector_monitor, xrandr_monitors
    monitor = select_projector_monitor(xrandr_monitors(), args.monitor)
    if monitor is None:
        parser.error(f"projector monitor not found: {args.monitor}")
    projector = OpenCVProjectorController(monitor)
    platform = SerialPlatformController(SerialPlatformConfig(args.port, read_timeout_s=1.0))
    camera = OrbbecCameraController()
    projector.open(); projector.show_black()
    platform.connect(); camera.start()
    try:
        diagnostic = PlatformMotionDiagnostic(
            platform, timeout_s=args.timeout,
            wait_config=wait_config,
            confirm=lambda _: True,
        )
        evaluator = (
            SurfaceReadinessEvaluator(quality_config)
            if args.selection_policy in {
                "highest_passing_readiness", "best_surface_coverage",
            }
            else SensorQualityEvaluator(quality_config)
        )
        diagnostic.read_before()
        search = HardwareAutomaticZSearch(
            platform=DiagnosticZMover(diagnostic), camera=camera, projector=projector,
            evaluator=evaluator,
            config=config,
            artifact_store=CandidateArtifactStore(args.output_dir) if args.output_dir else None,
        )
        result = search.run(pose_id="hardware_diagnostic", roll=args.roll, pitch=args.pitch)
        if args.result:
            result.save(args.result)
        if not result.success:
            raise RuntimeError("NoValidInspectionZ")
        print(f"best_z: {result.best_z}")
        return 0
    finally:
        try:
            projector.show_black()
        finally:
            camera.close(); platform.close(); projector.close()


if __name__ == "__main__":
    raise SystemExit(main())
