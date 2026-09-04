from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

if __package__ in (None, ""):
    repository_root = Path(__file__).resolve().parents[2]
    if str(repository_root) not in sys.path:
        sys.path.insert(0, str(repository_root))

from src.integration.projector_controller import OpenCVProjectorController
from src.integration.metric_pose_postprocess import postprocess_metric_pose_json
from src.integration.real_pose_planner import parse_pose_json
from src.integration.structured_light_runner import (
    ShellStructuredLightConfig,
    ShellStructuredLightRunner,
    StructuredLightRunInfo,
    StructuredLightStatus,
)
from src.lighting.serial_controller import SerialLightingConfig, SerialLightingController
from src.platform.serial_controller import SerialPlatformConfig, SerialPlatformController


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SUBSYSTEM_ROOT = REPOSITORY_ROOT / "서영 파트 파일"


class MetricPoseMeasurementError(RuntimeError):
    pass


def _finite(value: Any, name: str) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise MetricPoseMeasurementError(f"{name} must be finite")
    return number


class MetricPoseMeasurement:
    """Read-only platform telemetry plus one structured-light scan.

    This object intentionally has no conveyor, platform-motion, Automatic Z,
    inspection planner, or legacy pose dependency.
    """

    def __init__(
        self, *, platform: Any, structured_light_runner: Any, projector: Any,
        output_directory: str | Path, lighting: Any | None = None,
        telemetry_timeout_s: float = 2.0, telemetry_settle_s: float = 0.10,
        metric_pose_postprocessor: Callable[..., dict[str, Any]] = postprocess_metric_pose_json,
    ) -> None:
        self.platform = platform
        self.structured_light_runner = structured_light_runner
        self.projector = projector
        self.lighting = lighting
        self.output_directory = Path(output_directory).expanduser().resolve()
        self.telemetry_timeout_s = _finite(telemetry_timeout_s, "telemetry_timeout_s")
        self.telemetry_settle_s = _finite(telemetry_settle_s, "telemetry_settle_s")
        self.metric_pose_postprocessor = metric_pose_postprocessor
        if self.telemetry_timeout_s <= 0 or self.telemetry_settle_s < 0:
            raise ValueError("telemetry timeout must be positive and settle must be non-negative")

    @staticmethod
    def _pose_json(run_info: StructuredLightRunInfo) -> Path:
        if run_info.pose_json_path is None:
            raise MetricPoseMeasurementError("structured-light run did not expose pose_json_path")
        path = Path(run_info.pose_json_path).expanduser().resolve()
        root = Path(run_info.result_directory).expanduser().resolve()
        try:
            path.relative_to(root)
        except ValueError as exc:
            raise MetricPoseMeasurementError("pose JSON is outside the current scan directory") from exc
        if not path.is_file():
            raise MetricPoseMeasurementError(f"pose JSON does not exist: {path}")
        return path

    @staticmethod
    def _metric_planes(document: Any) -> tuple[list[dict[str, Any]], int]:
        planes: list[dict[str, Any]] = []
        metric_valid_count = 0
        for index, plane in enumerate(document.planes):
            metric = plane.metric_pose or {}
            status = str(metric.get("status", "DETECTED"))
            camera_roll = metric.get("camera_roll_deg")
            camera_pitch = metric.get("camera_pitch_deg")
            metric_valid = (
                metric.get("physical_metric") is True
                and status in {"METRIC_VALID", "REACHABLE", "UNREACHABLE"}
                and camera_roll is not None and camera_pitch is not None
            )
            if metric_valid:
                camera_roll = _finite(camera_roll, f"planes[{index}].camera_roll_deg")
                camera_pitch = _finite(camera_pitch, f"planes[{index}].camera_pitch_deg")
                metric_valid_count += 1
            planes.append({
                "source_plane_index": int(plane.metadata.get("source_plane_index", index)),
                "plane_name": plane.plane_name,
                "dominant": plane.dominant,
                "sl_points": plane.point_count,
                "metric_status": status,
                "metric_valid": metric_valid,
                "depth_points": int(metric.get("depth_points_count", 0)),
                "depth_coverage": float(metric.get("depth_coverage", 0.0)),
                "normal_xyz": metric.get("normal_xyz"),
                "center_xyz_mm": metric.get("center_xyz_mm"),
                "camera_roll_deg": camera_roll if metric_valid else None,
                "camera_pitch_deg": camera_pitch if metric_valid else None,
                "axis_contract": metric.get("axis_contract"),
                "reject_reason": metric.get("reject_reason"),
                "reason": metric.get("reason"),
                "reachable": metric.get("reachable") is True,
                "physical_plane_index": plane.metadata.get("physical_plane_index"),
                "merged_source_plane_indices": plane.metadata.get("merged_source_plane_indices", []),
                "merged_source_plane_names": plane.metadata.get("merged_source_plane_names", []),
                "platform_delta_roll_deg": metric.get("platform_delta_roll_deg"),
                "platform_delta_pitch_deg": metric.get("platform_delta_pitch_deg"),
                "current_platform_roll_deg": metric.get("current_platform_roll_deg"),
                "current_platform_pitch_deg": metric.get("current_platform_pitch_deg"),
                "target_platform_roll_deg": metric.get("target_platform_roll_deg"),
                "target_platform_pitch_deg": metric.get("target_platform_pitch_deg"),
                "calibration_id": metric.get("calibration_id"),
            })
        return planes, metric_valid_count

    @staticmethod
    def _print(
        telemetry: Any, planes: list[dict[str, Any]], metric_valid_count: int,
        *, raw_plane_count: int, physical_plane_count: int,
    ) -> None:
        print("\nMETRIC POSE MEASUREMENT")
        print("\nPlatform IMU")
        print(f"Z     = {float(telemetry.z_cm):.2f} cm")
        print(f"Roll  = {float(telemetry.roll_deg):+.2f} deg")
        print(f"Pitch = {float(telemetry.pitch_deg):+.2f} deg")
        print(f"stable = {telemetry.stable}")
        print(f"homing = {telemetry.homing}")
        print(f"\nDetected raw planes = {raw_plane_count}")
        print(f"Physical planes     = {physical_plane_count}")
        print(f"Metric-valid planes = {metric_valid_count}")
        for order, plane in enumerate(planes, 1):
            print(f"\nPlane {order}")
            print(f"name = {plane['plane_name']}")
            print(f"dominant = {plane['dominant']}")
            print(f"SL points = {plane['sl_points']}")
            print(f"Depth points = {plane['depth_points']}")
            print(f"Depth coverage = {plane['depth_coverage']:.6f}")
            print(f"normal = {plane['normal_xyz']}")
            print(f"Metric camera Roll = {plane['camera_roll_deg']}")
            print(f"Metric camera Pitch = {plane['camera_pitch_deg']}")
            print(f"Reachable = {plane['reachable']}")
            print(f"Target platform Roll = {plane['target_platform_roll_deg']}")
            print(f"Target platform Pitch = {plane['target_platform_pitch_deg']}")
            if not plane["metric_valid"]:
                print(f"Metric status = {plane['metric_status']}")
                print(f"Reason = {plane['reject_reason']}")

    def run(self) -> dict[str, Any]:
        self.output_directory.mkdir(parents=True, exist_ok=False)
        platform_connected = False
        lighting_connected = False
        projector_opened = False
        primary_error: BaseException | None = None
        try:
            self.platform.connect()
            platform_connected = True
            telemetry = self.platform.read_fresh_telemetry(
                timeout=self.telemetry_timeout_s,
                settle_s=self.telemetry_settle_s,
            )
            print("Platform fresh telemetry (READ ONLY)")
            print(f"Z={telemetry.z_cm:.2f} R={telemetry.roll_deg:+.2f} P={telemetry.pitch_deg:+.2f} "
                  f"stable={telemetry.stable} homing={telemetry.homing}")
            if telemetry.homing:
                raise MetricPoseMeasurementError("platform is homing; structured-light scan blocked")
            if not telemetry.stable:
                raise MetricPoseMeasurementError("platform is not stable; structured-light scan blocked")

            if self.lighting is not None:
                self.lighting.connect()
                lighting_connected = True
                self.lighting.inspection_off()
            self.projector.open()
            projector_opened = True
            self.projector.show_black()
            run_info = self.structured_light_runner.run_scan()
            self.projector.show_black()

            pose_json = self._pose_json(run_info)
            planning_telemetry = None

            def read_planning_telemetry():
                nonlocal planning_telemetry
                planning_telemetry = self.platform.read_fresh_telemetry(
                    timeout=self.telemetry_timeout_s,
                    settle_s=self.telemetry_settle_s,
                )
                return planning_telemetry

            postprocess = self.metric_pose_postprocessor(
                pose_json,
                fresh_telemetry_reader=read_planning_telemetry,
            )
            if planning_telemetry is not None:
                telemetry = planning_telemetry
            document = parse_pose_json(pose_json)
            planes, metric_valid_count = self._metric_planes(document)
            self._print(
                telemetry, planes, metric_valid_count,
                raw_plane_count=int(postprocess["raw_plane_count"]),
                physical_plane_count=int(postprocess["metric_physical_plane_count"]),
            )
            summary = {
                "schema_version": "metric_pose_measurement_v1",
                "platform_telemetry": {
                    "z_cm": float(telemetry.z_cm),
                    "roll_deg": float(telemetry.roll_deg),
                    "pitch_deg": float(telemetry.pitch_deg),
                    "stable": bool(telemetry.stable),
                    "homing": bool(telemetry.homing),
                },
                "structured_light": {
                    "run_id": run_info.run_id,
                    "result_directory": str(Path(run_info.result_directory).resolve()),
                    "pose_json_path": str(pose_json),
                    "manifest_path": str(run_info.manifest_path) if run_info.manifest_path else None,
                },
                "detected_planes": int(postprocess["raw_plane_count"]),
                "raw_plane_count": int(postprocess["raw_plane_count"]),
                "metric_physical_plane_count": int(postprocess["metric_physical_plane_count"]),
                "reachable_pose_count": int(postprocess["reachable_pose_count"]),
                "metric_valid_planes": metric_valid_count,
                "planes": planes,
                "safety": {
                    "platform_read_only": True,
                    "motion_commands_allowed": False,
                    "legacy_phase_pose_fallback": False,
                },
            }
            (self.output_directory / "summary.json").write_text(
                json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8",
            )
            return summary
        except BaseException as exc:
            primary_error = exc
            raise
        finally:
            cleanup_errors = []
            if projector_opened:
                try:
                    self.projector.show_black()
                except Exception as exc:
                    cleanup_errors.append(f"projector BLACK: {exc}")
            if lighting_connected:
                try:
                    self.lighting.inspection_off()
                except Exception as exc:
                    cleanup_errors.append(f"lighting OFF: {exc}")
            for label, resource in (
                ("projector", self.projector if projector_opened else None),
                ("lighting", self.lighting if lighting_connected else None),
                ("platform", self.platform if platform_connected else None),
            ):
                if resource is not None:
                    try:
                        resource.close()
                    except Exception as exc:
                        cleanup_errors.append(f"{label} close: {exc}")
            if cleanup_errors and primary_error is None:
                raise MetricPoseMeasurementError("; ".join(cleanup_errors))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Read-only platform + metric pose measurement")
    parser.add_argument("--platform-port", required=True)
    parser.add_argument("--lighting-port")
    parser.add_argument("--monitor", required=True)
    parser.add_argument("--subsystem-root", type=Path, default=DEFAULT_SUBSYSTEM_ROOT)
    parser.add_argument("--structured-light-python", type=Path)
    parser.add_argument("--structured-light-timeout", type=float, default=900.0)
    parser.add_argument("--telemetry-timeout", type=float, default=2.0)
    parser.add_argument("--telemetry-settle", type=float, default=0.10)
    parser.add_argument(
        "--output-root", type=Path,
        default=REPOSITORY_ROOT / "results" / "metric_pose_measurement",
    )
    parser.add_argument("--execute", action="store_true")
    return parser


def main(argv: list[str] | None = None, *, confirmation_input: Callable[[str], str] = input) -> int:
    args = build_parser().parse_args(argv)
    if not args.execute:
        raise MetricPoseMeasurementError(
            "--execute is required; no serial port, camera, or projector was opened"
        )
    if confirmation_input(
        "READ/SCAN ONLY: no conveyor or platform motion command will be sent. "
        "Type MEASURE to continue: "
    ).strip() != "MEASURE":
        raise MetricPoseMeasurementError("measurement cancelled; no hardware was opened")

    timestamp = datetime.now().strftime("run_%Y%m%d_%H%M%S")
    output_directory = args.output_root.expanduser().resolve() / timestamp
    subsystem = args.subsystem_root.expanduser().resolve()
    if str(subsystem) not in sys.path:
        sys.path.insert(0, str(subsystem))
    from structured_light_projector import select_projector_monitor, xrandr_monitors
    monitor = select_projector_monitor(xrandr_monitors(), args.monitor)
    if monitor is None:
        raise MetricPoseMeasurementError(f"projector monitor not found: {args.monitor}")

    structured_config = ShellStructuredLightConfig(
        subsystem_root=subsystem,
        result_root=output_directory / "structured_light" / "raw",
        python_path=args.structured_light_python,
        timeout_sec=args.structured_light_timeout,
        projector_monitor=args.monitor,
    )
    runner = ShellStructuredLightRunner(structured_config)
    report = runner.preflight_report()
    if report.overall_status is not StructuredLightStatus.READY:
        details = "; ".join((*report.issues, *report.warnings))
        raise MetricPoseMeasurementError(
            f"structured-light preflight is not READY: {report.overall_status.value}: {details}"
        )
    platform = SerialPlatformController(
        SerialPlatformConfig(args.platform_port, read_timeout_s=args.telemetry_timeout),
    )
    lighting = (
        SerialLightingController(SerialLightingConfig(args.lighting_port))
        if args.lighting_port else None
    )
    projector = OpenCVProjectorController(monitor)
    measurement = MetricPoseMeasurement(
        platform=platform, structured_light_runner=runner, projector=projector,
        lighting=lighting, output_directory=output_directory,
        telemetry_timeout_s=args.telemetry_timeout,
        telemetry_settle_s=args.telemetry_settle,
    )
    measurement.run()
    print(f"\nsummary: {output_directory / 'summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
