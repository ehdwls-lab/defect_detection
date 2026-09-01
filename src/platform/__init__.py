"""Platform motion interfaces and DTOs for the integration layer."""

from .controller import PlatformController, PlatformControllerProtocol
from .types import PlatformLimits, PlatformPoseCommand, PlatformTelemetry
from .serial_controller import SerialPlatformConfig, SerialPlatformController

__all__ = [
    "PlatformPoseCommand",
    "PlatformLimits",
    "PlatformTelemetry",
    "PlatformController",
    "PlatformControllerProtocol",
    "SerialPlatformConfig",
    "SerialPlatformController",
]
