from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Callable

import numpy as np

from .platform_limits import PLATFORM_RP_LIMIT_DEG
from .platform_pose_calibration import CAMERA_PLATFORM_RP_20260903


def predicted_camera_residual_angle_deg(camera_roll_deg: float, camera_pitch_deg: float) -> float:
    """Angle between the predicted residual normal and camera boresight."""
    roll = math.radians(camera_roll_deg)
    pitch = math.radians(camera_pitch_deg)
    return math.degrees(math.atan(math.hypot(math.tan(roll), math.tan(pitch))))


def _golden_minimum(function: Callable[[float], float], lower: float, upper: float) -> float:
    ratio = (math.sqrt(5.0) - 1.0) / 2.0
    left, right = float(lower), float(upper)
    x1 = right - ratio * (right - left)
    x2 = left + ratio * (right - left)
    f1, f2 = function(x1), function(x2)
    for _ in range(80):
        if f1 <= f2:
            right, x2, f2 = x2, x1, f1
            x1 = right - ratio * (right - left)
            f1 = function(x1)
        else:
            left, x1, f1 = x1, x2, f2
            x2 = left + ratio * (right - left)
            f2 = function(x2)
    return (left + right) / 2.0


def plan_platform_alignment(
    *, camera_roll_deg: float, camera_pitch_deg: float,
    current_platform_roll_deg: float, current_platform_pitch_deg: float,
    desired_target_roll_deg: float, desired_target_pitch_deg: float,
    limit_deg: float = PLATFORM_RP_LIMIT_DEG,
) -> dict[str, Any]:
    values = (
        camera_roll_deg, camera_pitch_deg, current_platform_roll_deg,
        current_platform_pitch_deg, desired_target_roll_deg,
        desired_target_pitch_deg, limit_deg,
    )
    if not all(math.isfinite(float(value)) for value in values) or limit_deg <= 0:
        raise ValueError("alignment inputs and limit must be finite; limit must be positive")
    j = np.asarray(CAMERA_PLATFORM_RP_20260903.camera_from_platform_J, dtype=np.float64)
    camera = np.asarray([camera_roll_deg, camera_pitch_deg], dtype=np.float64)
    current = np.asarray(
        [current_platform_roll_deg, current_platform_pitch_deg], dtype=np.float64,
    )
    desired = np.asarray([desired_target_roll_deg, desired_target_pitch_deg], dtype=np.float64)

    def residual(target: np.ndarray) -> float:
        predicted = camera + j @ (target - current)
        return predicted_camera_residual_angle_deg(float(predicted[0]), float(predicted[1]))

    if np.all(np.abs(desired) <= limit_deg + 1e-9):
        commanded = desired
        mode = "FULL"
    else:
        candidates: list[np.ndarray] = []
        for fixed_axis in (0, 1):
            free_axis = 1 - fixed_axis
            for fixed_value in (-limit_deg, limit_deg):
                def edge_objective(free_value: float, *, axis=fixed_axis,
                                   free=free_axis, fixed=fixed_value) -> float:
                    target = np.empty(2, dtype=np.float64)
                    target[axis], target[free] = fixed, free_value
                    return residual(target)

                free_value = _golden_minimum(edge_objective, -limit_deg, limit_deg)
                target = np.empty(2, dtype=np.float64)
                target[fixed_axis], target[free_axis] = fixed_value, free_value
                candidates.append(target)
        candidates.extend(
            np.asarray([roll, pitch], dtype=np.float64)
            for roll in (-limit_deg, limit_deg)
            for pitch in (-limit_deg, limit_deg)
        )
        commanded = min(candidates, key=residual)
        mode = "PARTIAL"
    return {
        "alignment_mode": mode,
        "desired_target_roll_deg": float(desired[0]),
        "desired_target_pitch_deg": float(desired[1]),
        "commanded_target_roll_deg": float(commanded[0]),
        "commanded_target_pitch_deg": float(commanded[1]),
        "predicted_residual_angle_deg": float(residual(commanded)),
        "platform_rp_limit_deg": float(limit_deg),
    }


def apply_alignment_policy_to_pose_json(path: str | Path) -> None:
    pose_path = Path(path).expanduser().resolve()
    payload = json.loads(pose_path.read_text(encoding="utf-8"))
    for plane in payload.get("planes", []):
        metric = plane.get("metric_pose") if isinstance(plane, dict) else None
        if not isinstance(metric, dict):
            continue
        try:
            if metric.get("calibration_id") != CAMERA_PLATFORM_RP_20260903.calibration_id:
                raise ValueError("verified camera/platform calibration is missing")
            alignment = plan_platform_alignment(
                camera_roll_deg=float(metric["camera_roll_deg"]),
                camera_pitch_deg=float(metric["camera_pitch_deg"]),
                current_platform_roll_deg=float(metric["current_platform_roll_deg"]),
                current_platform_pitch_deg=float(metric["current_platform_pitch_deg"]),
                desired_target_roll_deg=float(metric["target_platform_roll_deg"]),
                desired_target_pitch_deg=float(metric["target_platform_pitch_deg"]),
            )
        except (KeyError, TypeError, ValueError) as exc:
            metric.update(
                alignment_mode="REJECTED", reachable=False,
                status="METRIC_VALID" if metric.get("physical_metric") is True else "DETECTED",
                reason=f"alignment planning rejected: {exc}",
                reject_reason=f"alignment planning rejected: {exc}",
            )
            continue
        metric.update(alignment)
        metric.update(
            status="REACHABLE", reachable=True,
            reason=(
                "full camera-normal alignment within platform limits"
                if alignment["alignment_mode"] == "FULL"
                else "partial best-effort camera-normal alignment within platform limits"
            ),
            reject_reason=None,
            roll_deg=alignment["commanded_target_roll_deg"],
            pitch_deg=alignment["commanded_target_pitch_deg"],
        )
    temporary = pose_path.with_name(f".{pose_path.name}.alignment.tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(pose_path)
