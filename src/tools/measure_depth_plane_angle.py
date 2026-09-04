from __future__ import annotations

import argparse
import json
import math
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

import cv2
import numpy as np

if __package__ in (None, ""):
    repository_root = Path(__file__).resolve().parents[2]
    if str(repository_root) not in sys.path:
        sys.path.insert(0, str(repository_root))

from src.camera.orbbec_controller import OrbbecCameraController
from src.config import InspectionConfig
from src.integration.metric_pose import (
    CameraIntrinsics,
    MetricFitConfig,
    backproject_depth_pixels,
    camera_tilt_degrees,
    fit_metric_plane_ransac,
)
from src.integration.orbbec_intrinsics import build_d2c_intrinsics_payload
from src.platform.serial_controller import SerialPlatformConfig, SerialPlatformController


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


class DepthPlaneMeasurementError(RuntimeError):
    pass


def roi_arg(values: list[str] | tuple[int, ...]) -> tuple[int, int, int, int]:
    try:
        roi = tuple(int(value) for value in values)
    except (TypeError, ValueError) as exc:
        raise argparse.ArgumentTypeError("ROI requires four integers: x1 y1 x2 y2") from exc
    if len(roi) != 4:
        raise argparse.ArgumentTypeError("ROI requires four integers: x1 y1 x2 y2")
    return roi  # image bounds are validated against the captured frame


def validate_roi(roi: tuple[int, int, int, int], width: int, height: int) -> None:
    x1, y1, x2, y2 = roi
    if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
        raise DepthPlaneMeasurementError(
            f"ROI {roi} is outside image bounds {width}x{height} or is empty"
        )


class OrbbecAlignedDepthSource:
    """Tool-local adapter; it does not alter the production camera controller."""

    def __init__(self, controller: OrbbecCameraController | None = None) -> None:
        self.controller = controller or OrbbecCameraController()

    def start(self) -> None:
        self.controller.start()

    def capture(self):
        return self.controller.capture()

    def color_intrinsics(self, width: int, height: int) -> CameraIntrinsics:
        pipeline = self.controller._pipeline
        if pipeline is None:
            raise DepthPlaneMeasurementError("Orbbec camera is not started")
        payload = build_d2c_intrinsics_payload(
            pipeline.get_camera_param(),
            depth_grid_width=width,
            depth_grid_height=height,
        )
        raw = payload["color_intrinsics"]
        return CameraIntrinsics(
            fx=float(raw["fx"]), fy=float(raw["fy"]),
            cx=float(raw["cx"]), cy=float(raw["cy"]),
            width=int(raw["width"]), height=int(raw["height"]),
            source=str(payload["intrinsic_source"]), aligned_to="color",
        )

    def close(self) -> None:
        self.controller.close()


class DepthPlaneAngleMeasurement:
    """STM read-only gate followed by one aligned RGB+Depth capture."""

    def __init__(
        self, *, platform: Any, camera: Any, roi: tuple[int, int, int, int],
        output_directory: str | Path, save_preview: bool = False,
        telemetry_timeout_s: float = 2.0, telemetry_settle_s: float = 0.10,
        fit_config: MetricFitConfig | None = None,
    ) -> None:
        self.platform = platform
        self.camera = camera
        self.roi = tuple(roi)
        self.output_directory = Path(output_directory).expanduser().resolve()
        self.save_preview = bool(save_preview)
        self.telemetry_timeout_s = float(telemetry_timeout_s)
        self.telemetry_settle_s = float(telemetry_settle_s)
        self.fit_config = fit_config or MetricFitConfig()
        if not math.isfinite(self.telemetry_timeout_s) or self.telemetry_timeout_s <= 0:
            raise ValueError("telemetry timeout must be positive and finite")
        if not math.isfinite(self.telemetry_settle_s) or self.telemetry_settle_s < 0:
            raise ValueError("telemetry settle must be non-negative and finite")

    @staticmethod
    def _depth_preview(depth: np.ndarray) -> np.ndarray:
        valid = np.isfinite(depth) & (depth > 0)
        depth_u8 = np.zeros(depth.shape, dtype=np.uint8)
        if np.any(valid):
            low, high = np.percentile(depth[valid], (2, 98))
            if high > low:
                depth_u8[valid] = np.clip((depth[valid] - low) / (high - low) * 255, 0, 255)
        depth_preview = cv2.applyColorMap(depth_u8, cv2.COLORMAP_TURBO)
        depth_preview[~valid] = 0
        return depth_preview

    def _save_full_previews(self, color: np.ndarray, depth: np.ndarray) -> None:
        cv2.imwrite(
            str(self.output_directory / "color_full_preview.png"),
            np.asarray(color),
        )
        cv2.imwrite(
            str(self.output_directory / "depth_full_preview.png"),
            self._depth_preview(depth),
        )

    def _save_previews(self, color: np.ndarray, depth: np.ndarray) -> None:
        x1, y1, x2, y2 = self.roi
        color_preview = np.asarray(color).copy()
        cv2.rectangle(color_preview, (x1, y1), (x2 - 1, y2 - 1), (0, 255, 0), 2)
        cv2.imwrite(str(self.output_directory / "color_roi_preview.png"), color_preview)
        depth_preview = self._depth_preview(depth)
        cv2.rectangle(depth_preview, (x1, y1), (x2 - 1, y2 - 1), (255, 255, 255), 2)
        cv2.imwrite(str(self.output_directory / "depth_roi_preview.png"), depth_preview)

    def run(self) -> dict[str, Any]:
        started = time.monotonic()
        self.output_directory.mkdir(parents=True, exist_ok=False)
        platform_connected = False
        camera_started = False
        try:
            self.platform.connect()
            platform_connected = True
            telemetry = self.platform.read_fresh_telemetry(
                timeout=self.telemetry_timeout_s, settle_s=self.telemetry_settle_s,
            )
            if telemetry.homing:
                raise DepthPlaneMeasurementError("platform is homing; depth capture blocked")
            if not telemetry.stable:
                raise DepthPlaneMeasurementError("platform is not stable; depth capture blocked")

            self.camera.start()
            camera_started = True
            frame = self.camera.capture()
            depth = np.asarray(frame.depth_mm, dtype=np.float32)
            color = np.asarray(frame.color_bgr)
            if self.save_preview:
                self._save_full_previews(color, depth)
            if depth.ndim != 2 or color.shape[:2] != depth.shape:
                raise DepthPlaneMeasurementError("aligned color/depth grids do not match")
            height, width = depth.shape
            validate_roi(self.roi, width, height)
            intrinsics = self.camera.color_intrinsics(width, height)
            x1, y1, x2, y2 = self.roi
            vv, uu = np.mgrid[y1:y2, x1:x2]
            pixels = np.column_stack((uu.ravel(), vv.ravel()))
            xyz, valid_count = backproject_depth_pixels(
                depth, pixels, intrinsics, self.fit_config,
            )
            normal, center, residual, inlier_ratio = fit_metric_plane_ransac(
                xyz, self.fit_config,
            )
            camera_roll, camera_pitch = camera_tilt_degrees(normal)
            if self.save_preview:
                self._save_previews(color, depth)
            elapsed = time.monotonic() - started
            summary = {
                "schema_version": "depth_plane_calibration_v1",
                "platform_telemetry": {
                    "z_cm": float(telemetry.z_cm), "roll_deg": float(telemetry.roll_deg),
                    "pitch_deg": float(telemetry.pitch_deg), "stable": bool(telemetry.stable),
                    "homing": bool(telemetry.homing),
                },
                "depth": {
                    "alignment": "color", "unit": "mm", "roi": list(self.roi),
                    "valid_points": valid_count, "inlier_ratio": inlier_ratio,
                    "ransac_median_residual_mm": residual,
                    "intrinsics_source": intrinsics.source,
                },
                "plane": {
                    "normal_xyz": normal.tolist(), "center_xyz_mm": center.tolist(),
                    "camera_roll_deg": camera_roll, "camera_pitch_deg": camera_pitch,
                },
                "elapsed_seconds": elapsed,
                "safety": {
                    "platform_read_only": True, "structured_light_used": False,
                    "projector_used": False, "conveyor_used": False,
                    "automatic_z_used": False,
                },
            }
            (self.output_directory / "summary.json").write_text(
                json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8",
            )
            print("\nFAST DEPTH PLANE ANGLE")
            print("\nPlatform IMU")
            print(f"Z = {telemetry.z_cm:.2f} cm")
            print(f"R = {telemetry.roll_deg:+.2f} deg")
            print(f"P = {telemetry.pitch_deg:+.2f} deg")
            print("\nDepth ROI")
            print(f"ROI = {self.roi}")
            print(f"valid points = {valid_count}")
            print(f"inlier ratio = {inlier_ratio:.6f}")
            print(f"normal = {normal.tolist()}")
            print(f"camera Roll = {camera_roll:+.6f} deg")
            print(f"camera Pitch = {camera_pitch:+.6f} deg")
            print(f"elapsed = {elapsed:.3f} s")
            return summary
        finally:
            if camera_started:
                self.camera.close()
            if platform_connected:
                self.platform.close()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Fast read-only depth plane angle calibration")
    parser.add_argument("--platform-port", required=True)
    parser.add_argument("--roi", nargs=4, required=True, metavar=("X1", "Y1", "X2", "Y2"))
    parser.add_argument("--save-preview", action="store_true")
    parser.add_argument("--telemetry-timeout", type=float, default=2.0)
    parser.add_argument("--telemetry-settle", type=float, default=0.10)
    parser.add_argument(
        "--output-root", type=Path,
        default=REPOSITORY_ROOT / "results" / "depth_plane_calibration",
    )
    parser.add_argument("--execute", action="store_true")
    return parser


def main(argv: list[str] | None = None, *, confirmation_input: Callable[[str], str] = input) -> int:
    args = build_parser().parse_args(argv)
    if not args.execute:
        raise DepthPlaneMeasurementError(
            "--execute is required; no serial port or camera was opened"
        )
    if confirmation_input(
        "READ + ALIGNED DEPTH ONLY; no platform command will be sent. "
        "Type MEASURE to continue: "
    ).strip() != "MEASURE":
        raise DepthPlaneMeasurementError("measurement cancelled; no hardware was opened")
    roi = roi_arg(args.roi)
    run_dir = args.output_root.expanduser().resolve() / datetime.now().strftime("run_%Y%m%d_%H%M%S")
    config = InspectionConfig.default()
    fit = MetricFitConfig(
        min_depth_mm=config.depth.min_mm, max_depth_mm=config.depth.max_mm,
        ransac_threshold_mm=config.depth.plane_ransac_mm,
        ransac_iterations=config.depth.plane_ransac_iters,
        min_points=config.depth.plane_min_points,
        max_points=config.depth.plane_max_points,
    )
    measurement = DepthPlaneAngleMeasurement(
        platform=SerialPlatformController(SerialPlatformConfig(args.platform_port)),
        camera=OrbbecAlignedDepthSource(), roi=roi, output_directory=run_dir,
        save_preview=args.save_preview, telemetry_timeout_s=args.telemetry_timeout,
        telemetry_settle_s=args.telemetry_settle, fit_config=fit,
    )
    measurement.run()
    print(f"summary: {run_dir / 'summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
