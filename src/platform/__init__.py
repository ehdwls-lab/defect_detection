"""Platform motion interfaces and DTOs for the integration layer."""

from .controller import PlatformController, PlatformControllerProtocol
from .types import PlatformPoseCommand, PlatformTelemetry

__all__ = [
    "PlatformPoseCommand",
    "PlatformTelemetry",
    "PlatformController",
    "PlatformControllerProtocol",
]
