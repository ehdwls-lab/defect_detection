"""UI-only event contract; production cycles do not depend on this module."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class InspectionUIEventType(str, Enum):
    CYCLE_STARTED = "CYCLE_STARTED"
    STAGE_CHANGED = "STAGE_CHANGED"
    STRUCTURED_LIGHT_READY = "STRUCTURED_LIGHT_READY"
    POSE_SELECTED = "POSE_SELECTED"
    AUTO_Z_UPDATED = "AUTO_Z_UPDATED"
    ROI_READY = "ROI_READY"
    RGB_CAPTURED = "RGB_CAPTURED"
    ANOMALY_RESULT = "ANOMALY_RESULT"
    POSE_COMPLETE = "POSE_COMPLETE"
    CYCLE_COMPLETE = "CYCLE_COMPLETE"
    HARDWARE_ERROR = "HARDWARE_ERROR"


@dataclass(frozen=True)
class InspectionUIEvent:
    type: InspectionUIEventType
    elapsed_s: float
    payload: dict[str, Any] = field(default_factory=dict)

