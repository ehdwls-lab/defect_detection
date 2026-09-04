from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any


def combined_tilt_deg(roll_deg: float, pitch_deg: float) -> float:
    """Return the angle between the tilted surface normal and camera Z."""
    value = math.cos(math.radians(roll_deg)) * math.cos(math.radians(pitch_deg))
    return math.degrees(math.acos(max(-1.0, min(1.0, value))))


def max_tilt_for_z(z_cm: float, envelope: tuple[tuple[float, float], ...]) -> float:
    points = sorted((float(z), float(limit)) for z, limit in envelope)
    if not points or not math.isfinite(z_cm):
        raise ValueError("tilt envelope and Z must be finite")
    if z_cm <= points[0][0]:
        return points[0][1]
    if z_cm >= points[-1][0]:
        return points[-1][1]
    for (left_z, left_limit), (right_z, right_limit) in zip(points, points[1:]):
        if left_z <= z_cm <= right_z:
            fraction = (z_cm - left_z) / (right_z - left_z)
            return left_limit + fraction * (right_limit - left_limit)
    return points[-1][1]


@dataclass(frozen=True)
class AdaptivePose:
    z_cm: float
    requested_roll_deg: float
    requested_pitch_deg: float
    applied_roll_deg: float
    applied_pitch_deg: float
    combined_tilt_deg: float
    max_combined_tilt_deg: float
    tilt_scale: float
    clamped: bool


def adaptive_pose_for_z(
    z_cm: float,
    requested_roll_deg: float,
    requested_pitch_deg: float,
    *,
    roll_limit_deg: float,
    pitch_limit_deg: float,
    envelope: tuple[tuple[float, float], ...],
) -> AdaptivePose:
    base_roll = max(-roll_limit_deg, min(roll_limit_deg, float(requested_roll_deg)))
    base_pitch = max(-pitch_limit_deg, min(pitch_limit_deg, float(requested_pitch_deg)))
    allowed = max_tilt_for_z(float(z_cm), envelope)
    scale = 1.0
    if combined_tilt_deg(base_roll, base_pitch) > allowed:
        low, high = 0.0, 1.0
        for _ in range(60):
            middle = (low + high) / 2.0
            if combined_tilt_deg(base_roll * middle, base_pitch * middle) <= allowed:
                low = middle
            else:
                high = middle
        scale = low
    applied_roll = base_roll * scale
    applied_pitch = base_pitch * scale
    return AdaptivePose(
        z_cm=float(z_cm),
        requested_roll_deg=float(requested_roll_deg),
        requested_pitch_deg=float(requested_pitch_deg),
        applied_roll_deg=applied_roll,
        applied_pitch_deg=applied_pitch,
        combined_tilt_deg=combined_tilt_deg(applied_roll, applied_pitch),
        max_combined_tilt_deg=allowed,
        tilt_scale=scale,
        clamped=(base_roll != requested_roll_deg or base_pitch != requested_pitch_deg or scale < 1.0),
    )


def apply_adaptive_pose_transition(
    motion: Any, target: AdaptivePose, previous: AdaptivePose | None,
) -> Any:
    """Apply the production-safe adaptive R/P/Z ordering."""
    before = motion.read_before()
    if previous is None or target.z_cm <= previous.z_cm:
        if previous is None or not math.isclose(
            target.applied_roll_deg, previous.applied_roll_deg, abs_tol=1e-9,
        ) or not math.isclose(
            target.applied_pitch_deg, previous.applied_pitch_deg, abs_tol=1e-9,
        ):
            return motion.execute_orientation(
                roll_deg=target.applied_roll_deg,
                pitch_deg=target.applied_pitch_deg,
                before=before, ack_safe_height=True,
            )
        return before
    motion.execute_z(target.z_cm)
    return motion.execute_orientation(
        roll_deg=target.applied_roll_deg,
        pitch_deg=target.applied_pitch_deg,
        before=motion.read_before(), ack_safe_height=True,
    )
