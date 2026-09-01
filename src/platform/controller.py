from __future__ import annotations

from typing import Protocol

from .types import PlatformPoseCommand, PlatformTelemetry


class PlatformController(Protocol):
    """Platform controller interface.

    This interface is intentionally hardware-agnostic. Actual STM serial transport
    will be implemented only after the firmware protocol has been verified.
    """

    def move_to(self, command: PlatformPoseCommand) -> None:
        ...

    def get_telemetry(self) -> PlatformTelemetry:
        ...

    def wait_until_reached(self, timeout: float) -> PlatformTelemetry:
        ...


class PlatformControllerProtocol(PlatformController):
    """Marker protocol placeholder for compatibility and type checking."""

    def move_to(self, command: PlatformPoseCommand) -> None:
        raise NotImplementedError

    def get_telemetry(self) -> PlatformTelemetry:
        raise NotImplementedError

    def wait_until_reached(self, timeout: float) -> PlatformTelemetry:
        raise NotImplementedError
