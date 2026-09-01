from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PlatformPoseCommand:
    z_cm: float
    roll_deg: float
    pitch_deg: float


@dataclass(frozen=True)
class PlatformTelemetry:
    z_cm: float
    roll_deg: float
    pitch_deg: float
    target_reached: bool
    homing: bool
    motor_1: int | None = None
    motor_2: int | None = None
    motor_3: int | None = None
    imu_mode: int | None = None
    timestamp: float | None = None
