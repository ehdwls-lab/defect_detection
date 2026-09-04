from __future__ import annotations

import argparse
import logging
import math
import shutil
import sys
from dataclasses import replace
from pathlib import Path

if __package__ in (None, ""):
    repository_root = Path(__file__).resolve().parents[2]
    if str(repository_root) not in sys.path:
        sys.path.insert(0, str(repository_root))

from src.camera.orbbec_controller import OrbbecCameraController
from src.config import InspectionConfig
from src.conveyor.serial_controller import SerialConveyorConfig, SerialConveyorController
from src.inspection.hardware_z_search import (
    CandidateArtifactStore,
    HardwareAutomaticZSearch,
    HardwareZSearchConfig,
    SensorQualityConfig,
    SurfaceReadinessEvaluator,
)
from src.lighting.serial_controller import SerialLightingConfig, SerialLightingController
from src.integration.integrated_inspection_cycle import (
    IntegratedInspectionCycle,
    ManualLEDConfirmationError,
    create_timestamped_run_directory,
)
from src.integration.projector_controller import OpenCVProjectorController
from src.integration.platform_limits import ORIENTATION_SAFE_Z_MIN_CM
from src.integration.real_pose_planner import RealPosePlanner
from src.integration.structured_light_runner import (
    ShellStructuredLightConfig,
    ShellStructuredLightRunner,
    StructuredLightStatus,
)
from src.platform.motion_diagnostic import (
    DiagnosticZMover,
    MotionWaitConfig,
    PlatformMotionDiagnostic,
)
from src.platform.serial_controller import SerialPlatformConfig, SerialPlatformController
from src.anomaly.detector import ProductionAnomalyConfig, ProductionAnomalyDetector
from src.integration.final_capture import FINAL_RGB_WARMUP_FRAMES
from src.tools.production_motion_options import (
    add_production_motion_arguments, build_production_motion_wait_config,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SUBSYSTEM_ROOT = REPOSITORY_ROOT / "서영 파트 파일"
PRODUCTION_AUTOMATIC_Z_START_CM = ORIENTATION_SAFE_Z_MIN_CM
PRODUCTION_AUTOMATIC_Z_STEP_CM = 1.0


def finite_arg(value: str) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise argparse.ArgumentTypeError("value must be finite")
    return number


def positive_int(value: str) -> int:
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("value must be a positive integer")
    return number


def z_candidates(value: str) -> tuple[float, ...]:
    try:
        parsed = tuple(float(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as exc:
        raise argparse.ArgumentTypeError("Z candidates must be comma-separated numbers") from exc
    if not parsed or not all(math.isfinite(item) for item in parsed):
        raise argparse.ArgumentTypeError("at least one finite Z candidate is required")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Dry-run-first partial hardware inspection-cycle diagnostic",
    )
    parser.add_argument("--conveyor-port", required=True)
    parser.add_argument("--platform-port", required=True)
    parser.add_argument("--lighting-port", required=True)
    parser.add_argument("--lighting-startup-timeout", type=finite_arg, default=5.0)
    parser.add_argument("--cover-open-angle", type=int, choices=(0, 90))
    parser.add_argument("--cover-close-angle", type=int, choices=(0, 90))
    parser.add_argument(
        "--cover-cleanup-state", choices=("OPEN", "CLOSE", "NONE"), default="CLOSE",
    )
    parser.add_argument("--conveyor-steps", required=True, type=positive_int)
    parser.add_argument("--conveyor-direction", choices=("F", "B"), default="F")
    parser.add_argument("--conveyor-out-direction", choices=("F", "B"), default="F")
    parser.add_argument("--conveyor-out-steps", type=positive_int, default=10000)
    parser.add_argument("--monitor", required=True)
    parser.add_argument("--scan-z", required=True, type=finite_arg)
    parser.add_argument("--safe-z", required=True, type=finite_arg)
    parser.add_argument("--z-candidates", type=z_candidates)
    parser.add_argument("--z-start", type=finite_arg)
    parser.add_argument("--z-search-min", type=finite_arg, default=17.0)
    parser.add_argument("--z-coarse-step", type=finite_arg)
    parser.add_argument("--z-fine-step", type=finite_arg)
    parser.add_argument(
        "--pose-plan-mode", choices=("dominant_only", "all_valid_planes"),
        default="dominant_only",
    )
    parser.add_argument("--z-max", required=True, type=finite_arg)
    parser.add_argument(
        "--z-selection-policy",
        choices=("highest_passing_readiness", "best_surface_coverage"),
        default="best_surface_coverage",
    )
    parser.add_argument("--z-surface-area-weight", type=finite_arg, default=0.6)
    parser.add_argument("--z-depth-valid-weight", type=finite_arg, default=0.4)
    parser.add_argument("--quality-config", required=True, type=Path)
    parser.add_argument(
        "--output-root", type=Path,
        default=REPOSITORY_ROOT / "results" / "integrated_hardware",
    )
    parser.add_argument("--subsystem-root", type=Path, default=DEFAULT_SUBSYSTEM_ROOT)
    parser.add_argument("--structured-light-python", type=Path)
    parser.add_argument("--structured-light-timeout", type=finite_arg, default=900.0)
    parser.add_argument("--conveyor-timeout", type=finite_arg, default=30.0)
    parser.add_argument("--conveyor-startup-timeout", type=finite_arg, default=5.0)
    parser.add_argument(
        "--conveyor-rx-settle", type=finite_arg, default=0.10,
        help="diagnostic USB/serial drain interval; not a production safety limit",
    )
    add_production_motion_arguments(parser)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--ack-mechanical-range", action="store_true")
    parser.add_argument(
        "--anomaly-model", type=Path,
        default=REPOSITORY_ROOT / "models" / "best_autoencoder.pth",
    )
    parser.add_argument(
        "--anomaly-val-manifest", type=Path,
        default=REPOSITORY_ROOT / "data" / "manifests" / "val.csv",
    )
    parser.add_argument("--anomaly-surface-coverage", type=finite_arg, default=1.0)
    parser.add_argument(
        "--final-rgb-warmup-frames", type=int, default=FINAL_RGB_WARMUP_FRAMES,
    )
    parser.add_argument(
        "--legacy-inspection-roi", action="store_true",
        help="disable ArUco workspace selection and use fallback contour-fill workspace",
    )
    return parser


def _led_checkpoint() -> bool:
    print("=" * 48)
    print("Projector: BLACK")
    print("Platform pose ready")
    print("")
    print("Turn INSPECTION LED ON now.")
    print("")
    print("After confirming the LED is ON, type:")
    print("")
    print("LED_ON")
    print("=" * 48)
    while True:
        try:
            value = input("LED checkpoint: ").strip()
        except EOFError as exc:
            raise ManualLEDConfirmationError(
                "stdin closed before exact LED_ON confirmation"
            ) from exc
        if value == "LED_ON":
            return True
        print("Automatic Z remains blocked. Type exactly LED_ON after the LED is ON.")


def _validate_static(args: argparse.Namespace) -> tuple[
    SerialConveyorConfig, HardwareZSearchConfig, MotionWaitConfig, SensorQualityConfig,
    ShellStructuredLightConfig,
]:
    adaptive_values = (args.z_start, args.z_coarse_step, args.z_fine_step)
    if args.z_candidates is None and not any(value is not None for value in adaptive_values):
        args.z_start = 25.0
        args.z_coarse_step = PRODUCTION_AUTOMATIC_Z_STEP_CM
        args.z_fine_step = PRODUCTION_AUTOMATIC_Z_STEP_CM
        adaptive_values = (args.z_start, args.z_coarse_step, args.z_fine_step)
    adaptive_requested = any(value is not None for value in adaptive_values)
    if adaptive_requested and args.z_candidates is not None:
        raise ValueError("--z-candidates cannot be combined with adaptive Z options")
    if adaptive_requested and not all(value is not None for value in adaptive_values):
        raise ValueError("adaptive mode requires --z-start, --z-coarse-step, and --z-fine-step")
    if not adaptive_requested and args.z_candidates is None:
        raise ValueError("provide --z-candidates or all adaptive Z options")
    if args.scan_z > args.z_max:
        raise ValueError("scan_z must not exceed the user-provided z_max")
    if args.safe_z < ORIENTATION_SAFE_Z_MIN_CM:
        raise ValueError(
            "safe_z must be at least the production orientation minimum "
            f"{ORIENTATION_SAFE_Z_MIN_CM:g} cm"
        )
    if args.safe_z > args.z_max:
        raise ValueError("safe_z must not exceed the user-provided z_max")
    if args.z_search_min < 17.0 or args.z_search_min > 25.0:
        raise ValueError("z_search_min must be between 17 and 25 cm")
    if args.structured_light_timeout <= 0:
        raise ValueError("structured_light_timeout must be positive")
    if args.anomaly_surface_coverage != 1.0:
        raise ValueError("production anomaly_surface_coverage must be 1.0")
    if args.final_rgb_warmup_frames < 0:
        raise ValueError("final_rgb_warmup_frames must be non-negative")
    if args.conveyor_direction != "F" or args.conveyor_steps != 6325:
        raise ValueError("production conveyor IN must be F6325")
    if args.conveyor_out_direction != "F" or args.conveyor_out_steps != 10000:
        raise ValueError("production conveyor OUT must be F10000")
    conveyor_config = SerialConveyorConfig(
        port=args.conveyor_port,
        inspection_direction=args.conveyor_direction,
        inspection_steps=args.conveyor_steps,
        exit_direction=args.conveyor_out_direction,
        exit_steps=args.conveyor_out_steps,
        timeout_sec=args.conveyor_timeout,
        rx_settle_sec=args.conveyor_rx_settle,
        startup_timeout_sec=args.conveyor_startup_timeout,
    )
    conveyor_config.validate()
    z_config = HardwareZSearchConfig(
        () if adaptive_requested else args.z_candidates,
        args.z_max, args.platform_motion_timeout,
        args.z_selection_policy,
        "adaptive" if adaptive_requested else "explicit",
        args.z_start, args.z_coarse_step, args.z_fine_step,
        args.z_surface_area_weight, args.z_depth_valid_weight,
        stop_after_first_post_pass_failure=not adaptive_requested,
        search_min_z_cm=args.z_search_min if adaptive_requested else None,
    )
    if adaptive_requested and args.z_max < 25.0:
        raise ValueError("adaptive production search requires z_max >= 25 cm")
    z_config.validate()
    wait_config = build_production_motion_wait_config(args)
    quality_config = SensorQualityConfig.from_json(args.quality_config)
    quality_config.require_execution_ready(args.z_selection_policy)
    structured_config = ShellStructuredLightConfig(
        subsystem_root=args.subsystem_root,
        # Static preflight checks the user-selected integrated output root.
        # The actual scan is redirected into the timestamped run after the
        # final EXECUTE confirmation.
        result_root=args.output_root,
        python_path=args.structured_light_python,
        timeout_sec=args.structured_light_timeout,
        non_interactive=True,
        visualize=False,
        projector_monitor=args.monitor,
    )
    return conveyor_config, z_config, wait_config, quality_config, structured_config


def run(args: argparse.Namespace, *, confirmation_input=None) -> int:
    try:
        conveyor_config, z_config, wait_config, quality_config, structured_config = _validate_static(args)
    except (OSError, TypeError, ValueError) as exc:
        raise ValueError(f"configuration is not execution-ready: {exc}") from exc

    print(f"conveyor command: {args.conveyor_direction}{args.conveyor_steps}")
    print(f"structured-light scan/reference Z: {args.scan_z}")
    print(f"safe Z: {args.safe_z}")
    if args.z_candidates is not None:
        print(f"Z candidates: {list(args.z_candidates)}")
    else:
        print(
            f"adaptive Z: start={args.z_start}, max={args.z_max}, "
            f"min={args.z_search_min}, coarse_step={args.z_coarse_step}, fine_step={args.z_fine_step}"
        )
    print(f"z_max: {args.z_max}")
    print(f"pose plan mode: {args.pose_plan_mode}")
    print(f"anomaly surface patch coverage: {args.anomaly_surface_coverage}")
    print(f"final RGB warmup frames: {args.final_rgb_warmup_frames}")
    print(
        "anomaly ROI: "
        + ("Depth external contour fill (fallback workspace)" if args.legacy_inspection_roi
            else "Depth external contour fill (ArUco workspace, fallback workspace)")
    )
    print(f"selection policy: {args.z_selection_policy}")
    print("structured-light legacy Z: IGNORED")
    if args.conveyor_out_direction is None:
        print("cycle stop: anomaly inference complete (no conveyor OUT)")
    else:
        print(
            "optional conveyor OUT: "
            f"{args.conveyor_out_direction}{args.conveyor_out_steps}"
        )
    if not args.execute:
        print("DRY RUN - no serial, camera, projector GUI, or structured-light scan was opened")
        return 0
    if not args.ack_mechanical_range:
        raise ValueError("--execute requires --ack-mechanical-range")
    if args.cover_open_angle is None or args.cover_close_angle is None:
        raise ValueError(
            "--execute requires explicit --cover-open-angle and --cover-close-angle hardware mapping"
        )
    if args.cover_open_angle == args.cover_close_angle:
        raise ValueError("cover OPEN/CLOSE angles must differ")

    anomaly_detector = ProductionAnomalyDetector(ProductionAnomalyConfig(
        checkpoint_path=args.anomaly_model.expanduser().resolve(),
        validation_manifest_path=args.anomaly_val_manifest.expanduser().resolve(),
        surface_patch_coverage=args.anomaly_surface_coverage,
    ))
    anomaly_detector.validate_ready()

    preflight_runner = ShellStructuredLightRunner(structured_config)
    report = preflight_runner.preflight_report()
    if report.overall_status is not StructuredLightStatus.READY:
        details = "; ".join((*report.issues, *report.warnings))
        raise ValueError(f"structured-light preflight is not READY: {report.overall_status.value}: {details}")
    confirmation_input = input if confirmation_input is None else confirmation_input
    if confirmation_input(
        "This will move the conveyor and STM platform and capture from Orbbec.\n"
        "Type EXECUTE to start the partial hardware cycle: "
    ).strip() != "EXECUTE":
        raise ValueError("execution cancelled; no hardware was opened")
    anomaly_detector.prepare()

    subsystem = Path(structured_config.subsystem_root).expanduser().resolve()
    if str(subsystem) not in sys.path:
        sys.path.insert(0, str(subsystem))
    from structured_light_projector import select_projector_monitor, xrandr_monitors

    monitor = select_projector_monitor(xrandr_monitors(), args.monitor)
    if monitor is None:
        raise ValueError(f"projector monitor not found: {args.monitor}")

    run_directory = create_timestamped_run_directory(args.output_root)
    structured_config = replace(
        structured_config,
        result_root=run_directory / "structured_light" / "raw",
    )
    projector = OpenCVProjectorController(monitor)
    conveyor = SerialConveyorController(conveyor_config)
    platform = SerialPlatformController(
        SerialPlatformConfig(args.platform_port, baudrate=115200, read_timeout_s=1.0),
    )
    lighting = SerialLightingController(SerialLightingConfig(
        args.lighting_port, startup_timeout_sec=args.lighting_startup_timeout,
        projector_cover_open_angle_deg=args.cover_open_angle,
        projector_cover_close_angle_deg=args.cover_close_angle,
        projector_cover_cleanup_state=args.cover_cleanup_state,
    ))
    motion = PlatformMotionDiagnostic(
        platform, timeout_s=args.platform_motion_timeout, wait_config=wait_config,
        confirm=lambda _: True,
    )
    camera = OrbbecCameraController()
    runner = ShellStructuredLightRunner(structured_config, projector=projector)
    evaluator = SurfaceReadinessEvaluator(quality_config)
    z_search = HardwareAutomaticZSearch(
        platform=DiagnosticZMover(motion), camera=camera, projector=projector,
        evaluator=evaluator, config=z_config,
        artifact_store=CandidateArtifactStore(run_directory / "automatic_z"),
    )
    inspection_config = InspectionConfig.default()
    if args.legacy_inspection_roi:
        inspection_config = replace(
            inspection_config,
            hybrid_roi=replace(inspection_config.hybrid_roi, enabled=False),
        )
    cycle = IntegratedInspectionCycle(
        conveyor=conveyor,
        structured_light_runner=runner,
        pose_planner=RealPosePlanner(
            args.pose_plan_mode,
            inspection_roll_limit_deg=inspection_config.quality.inspection_roll_limit_deg,
            inspection_pitch_limit_deg=inspection_config.quality.inspection_pitch_limit_deg,
        ),
        projector=projector,
        platform=platform,
        motion_diagnostic=motion,
        camera=camera,
        automatic_z_search=z_search,
        scan_z=args.scan_z,
        safe_z=args.safe_z,
        run_directory=run_directory,
        lighting=lighting,
        anomaly_detector=anomaly_detector,
        final_capture_inspection_config=inspection_config,
        final_rgb_warmup_frames=args.final_rgb_warmup_frames,
        conveyor_out_enabled=True,
    )
    quality_snapshot = cycle.paths.automatic_z / "quality_config.json"
    if args.quality_config.expanduser().resolve() != quality_snapshot.resolve():
        shutil.copy2(args.quality_config, quality_snapshot)
    result = cycle.run()
    print(f"cycle stage: {result.stage}")
    print(f"overall status: {result.overall_status}")
    print(f"quality judgement: {result.quality_judgement}")
    print(f"cycle result: {cycle.paths.root / 'cycle_result.json'}")
    if result.success:
        print(f"best Z: {result.best_z}")
        print(
            "End-to-end inspection complete; conveyor OUT "
            + ("completed" if result.conveyor_out_executed else "was not run")
        )
        return 0
    if result.overall_status == "PARTIAL_COMPLETE":
        print(
            "End-to-end inspection partially complete: "
            f"completed={result.planes_completed}, failed={result.planes_failed}; "
            "conveyor OUT "
            + ("completed" if result.conveyor_out_executed else "was not run")
        )
        return 1
    print(f"cycle failed at {result.stage}: {result.error_type}: {result.error_message}")
    return 130 if result.interrupted else 1


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    try:
        return run(args)
    except ValueError as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    raise SystemExit(main())
