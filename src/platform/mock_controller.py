from __future__ import annotations

import time

from .types import PlatformPoseCommand, PlatformTelemetry


class MockPlatformController:
    """Stateful mock used for orchestration tests without hardware.

    This is intentionally not a serial implementation. It only simulates pose state
    and telemetry so higher-level flow logic can be validated.
    """

    def __init__(self, start_z_cm: float = 20.0) -> None:
        self._z_cm = start_z_cm
        self._roll_deg = 0.0
        self._pitch_deg = 0.0
        self._target_reached = True
        self._homing = False

    def move_to(self, command: PlatformPoseCommand) -> None:
        self._z_cm = float(command.z_cm)
        self._roll_deg = float(command.roll_deg)
        self._pitch_deg = float(command.pitch_deg)
        self._target_reached = False

    def get_telemetry(self) -> PlatformTelemetry:
        return PlatformTelemetry(
            z_cm=self._z_cm,
            roll_deg=self._roll_deg,
            pitch_deg=self._pitch_deg,
            target_reached=self._target_reached,
            homing=self._homing,
            motor_1=1,
            motor_2=1,
            motor_3=1,
            imu_mode=0,
            timestamp=time.time(),
        )

    def wait_until_reached(self, timeout: float) -> PlatformTelemetry:
        time.sleep(min(0.05, max(0.01, timeout / 100.0)))
        self._target_reached = True
        return self.get_telemetry()
