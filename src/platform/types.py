from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PlatformPoseCommand:
    z_cm: float
    roll_deg: float
    pitch_deg: float


@dataclass(frozen=True)
class PlatformLimits:
    """Optional host-side limits.

    ``None`` means that a limit has not been calibrated.  Hardware defaults
    deliberately remain unset; callers must provide verified values.
    """

    z_min_cm: float | None = None
    z_max_cm: float | None = None
    roll_min_deg: float | None = None
    roll_max_deg: float | None = None
    pitch_min_deg: float | None = None
    pitch_max_deg: float | None = None

    def validate(self, command: PlatformPoseCommand) -> None:
        checks = (
            ("z_cm", command.z_cm, self.z_min_cm, self.z_max_cm),
            ("roll_deg", command.roll_deg, self.roll_min_deg, self.roll_max_deg),
            ("pitch_deg", command.pitch_deg, self.pitch_min_deg, self.pitch_max_deg),
        )
        for name, value, minimum, maximum in checks:
            if minimum is not None and value < minimum:
                raise ValueError(f"{name}={value} is below configured minimum {minimum}")
            if maximum is not None and value > maximum:
                raise ValueError(f"{name}={value} is above configured maximum {maximum}")


@dataclass(frozen=True)
class PlatformTelemetry:
    z_cm: float
    roll_deg: float
    pitch_deg: float
    stable: bool
    homing: bool
    motor1: int | None = None
    motor2: int | None = None
    motor3: int | None = None
    imu_mode: int | None = None
    control_mode: int | None = None
    roll_rate_deg_s: float | None = None
    pitch_rate_deg_s: float | None = None
    timestamp: float | None = None

    @property
    def target_reached(self) -> bool:
        """Compatibility alias; firmware S means stable, not measured reach."""
        return self.stable

    @property
    def motor_1(self) -> int | None:
        return self.motor1

    @property
    def motor_2(self) -> int | None:
        return self.motor2

    @property
    def motor_3(self) -> int | None:
        return self.motor3
