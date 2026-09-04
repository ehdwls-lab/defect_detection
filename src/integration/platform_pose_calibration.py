from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .platform_limits import PLATFORM_RP_LIMIT_DEG


class PlatformPoseCalibrationError(ValueError):
    pass


@dataclass(frozen=True)
class CameraPlatformRPCalibration:
    calibration_id: str
    calibration_z_cm: float
    roi: tuple[int, int, int, int]
    camera_from_platform_J: tuple[tuple[float, float], tuple[float, float]]
    platform_correction_from_camera_K: tuple[tuple[float, float], tuple[float, float]]
    platform_limit_deg: float = PLATFORM_RP_LIMIT_DEG

    def validate(self) -> None:
        if not self.calibration_id.strip():
            raise PlatformPoseCalibrationError("calibration_id is required")
        if not math.isfinite(self.calibration_z_cm) or self.calibration_z_cm <= 0:
            raise PlatformPoseCalibrationError("calibration_z_cm must be positive and finite")
        if len(self.roi) != 4 or not all(isinstance(value, int) for value in self.roi):
            raise PlatformPoseCalibrationError("calibration ROI must contain four integers")
        j = np.asarray(self.camera_from_platform_J, dtype=np.float64)
        k = np.asarray(self.platform_correction_from_camera_K, dtype=np.float64)
        if j.shape != (2, 2) or k.shape != (2, 2):
            raise PlatformPoseCalibrationError("J and K must be 2x2 matrices")
        if not np.all(np.isfinite(j)) or not np.all(np.isfinite(k)):
            raise PlatformPoseCalibrationError("J and K must be finite")
        if abs(float(np.linalg.det(j))) <= 1e-9:
            raise PlatformPoseCalibrationError("camera_from_platform_J is singular")
        if not np.allclose(k, -np.linalg.inv(j), rtol=0.0, atol=2e-6):
            raise PlatformPoseCalibrationError("K is inconsistent with -inv(J)")
        if not math.isfinite(self.platform_limit_deg) or self.platform_limit_deg <= 0:
            raise PlatformPoseCalibrationError("platform_limit_deg must be positive and finite")

    def metadata(self) -> dict[str, Any]:
        value = asdict(self)
        value["roi"] = list(self.roi)
        value["camera_from_platform_J"] = [list(row) for row in self.camera_from_platform_J]
        value["platform_correction_from_camera_K"] = [
            list(row) for row in self.platform_correction_from_camera_K
        ]
        return value


CAMERA_PLATFORM_RP_20260903 = CameraPlatformRPCalibration(
    calibration_id="camera_platform_rp_20260903",
    calibration_z_cm=15.0,
    roi=(450, 180, 950, 520),
    camera_from_platform_J=(
        (-1.027901, -0.053029),
        (-0.039609, 1.046716),
    ),
    platform_correction_from_camera_K=(
        (0.970961, 0.049191),
        (0.036743, -0.953507),
    ),
)


def calibrated_platform_target(
    camera_roll_deg: float,
    camera_pitch_deg: float,
    current_platform_roll_deg: float,
    current_platform_pitch_deg: float,
    calibration: CameraPlatformRPCalibration | None,
) -> dict[str, Any]:
    values = (
        camera_roll_deg, camera_pitch_deg,
        current_platform_roll_deg, current_platform_pitch_deg,
    )
    if not all(math.isfinite(float(value)) for value in values):
        raise PlatformPoseCalibrationError("camera and current platform angles must be finite")
    if calibration is None:
        raise PlatformPoseCalibrationError("camera/platform R/P calibration is not loaded")
    calibration.validate()
    camera = np.array([camera_roll_deg, camera_pitch_deg], dtype=np.float64)
    delta = np.asarray(calibration.platform_correction_from_camera_K) @ camera
    current = np.array(
        [current_platform_roll_deg, current_platform_pitch_deg], dtype=np.float64,
    )
    target = current + delta
    reachable = bool(np.all(np.abs(target) <= calibration.platform_limit_deg + 1e-9))
    reason = (
        (
            f"target within platform R/P limit ±{calibration.platform_limit_deg:g} deg: "
            f"roll={target[0]:+.6f}, pitch={target[1]:+.6f}"
        )
        if reachable
        else (
            f"target platform angle exceeds ±{calibration.platform_limit_deg:g} deg: "
            f"roll={target[0]:+.6f}, pitch={target[1]:+.6f}"
        )
    )
    return {
        "camera_roll_deg": float(camera[0]),
        "camera_pitch_deg": float(camera[1]),
        "platform_delta_roll_deg": float(delta[0]),
        "platform_delta_pitch_deg": float(delta[1]),
        "current_platform_roll_deg": float(current[0]),
        "current_platform_pitch_deg": float(current[1]),
        "target_platform_roll_deg": float(target[0]),
        "target_platform_pitch_deg": float(target[1]),
        "calibration_id": calibration.calibration_id,
        "calibration_z_cm": calibration.calibration_z_cm,
        "roi": list(calibration.roi),
        "camera_from_platform_J": [list(row) for row in calibration.camera_from_platform_J],
        "platform_correction_from_camera_K": [
            list(row) for row in calibration.platform_correction_from_camera_K
        ],
        "calibration": calibration.metadata(),
        "reachable": reachable,
        "reason": reason,
    }


def predicted_board_normal_from_platform_pose(
    current_platform_roll_deg: float,
    current_platform_pitch_deg: float,
    commanded_platform_roll_deg: float,
    commanded_platform_pitch_deg: float,
    calibration: CameraPlatformRPCalibration | None = CAMERA_PLATFORM_RP_20260903,
) -> np.ndarray:
    """Predict the front-facing board normal at the commanded platform pose."""
    values = (
        current_platform_roll_deg, current_platform_pitch_deg,
        commanded_platform_roll_deg, commanded_platform_pitch_deg,
    )
    if not all(math.isfinite(float(value)) for value in values):
        raise PlatformPoseCalibrationError("platform R/P values must be finite")
    if calibration is None:
        raise PlatformPoseCalibrationError("camera/platform R/P calibration is not loaded")
    calibration.validate()
    commanded = np.asarray(
        [commanded_platform_roll_deg, commanded_platform_pitch_deg], dtype=np.float64,
    )
    camera_tilt = np.asarray(calibration.camera_from_platform_J, dtype=np.float64) @ commanded
    roll, pitch = np.radians(camera_tilt)
    normal = np.asarray([np.tan(roll), np.tan(pitch), -1.0], dtype=np.float64)
    return normal / np.linalg.norm(normal)


def apply_calibration_to_pose_json(
    path: str | Path,
    *, current_platform_roll_deg: float,
    current_platform_pitch_deg: float,
    calibration: CameraPlatformRPCalibration | None,
) -> None:
    pose_path = Path(path).expanduser().resolve()
    try:
        payload = json.loads(pose_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PlatformPoseCalibrationError(f"failed to read pose JSON: {pose_path}") from exc
    planes = payload.get("planes")
    if not isinstance(planes, list):
        raise PlatformPoseCalibrationError("pose JSON planes must be a list")
    for index, plane in enumerate(planes):
        metric = plane.get("metric_pose") if isinstance(plane, dict) else None
        if not isinstance(metric, dict):
            continue
        try:
            target = calibrated_platform_target(
                float(metric["camera_roll_deg"]),
                float(metric["camera_pitch_deg"]),
                current_platform_roll_deg,
                current_platform_pitch_deg,
                calibration,
            )
        except (KeyError, TypeError, ValueError, PlatformPoseCalibrationError) as exc:
            metric.update(
                status="METRIC_VALID" if metric.get("physical_metric") is True else "DETECTED",
                reachable=False,
                reason=f"platform calibration unavailable: {exc}",
                reject_reason=f"platform calibration unavailable: {exc}",
            )
            continue
        metric.update(target)
        metric["roll_deg"] = target["target_platform_roll_deg"]
        metric["pitch_deg"] = target["target_platform_pitch_deg"]
        metric["status"] = "REACHABLE" if target["reachable"] else "UNREACHABLE"
        metric["reject_reason"] = None if target["reachable"] else target["reason"]
    temporary = pose_path.with_name(f".{pose_path.name}.calibration.tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(pose_path)
