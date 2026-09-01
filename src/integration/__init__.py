"""Integration contracts between the structured-light subsystem and the defect-detection runtime."""

from .coordinate_contract import (
    CloudType,
    CoordinateConvention,
    PointCloudMetadata,
    StructuredLightResult,
    StructuredLightPaths,
)
from .inspection_plan import InspectionPlan, PosePlanningInput, PoseTarget
from .pose_planner import PosePlanner
from .structured_light_adapter import StructuredLightAdapter, StructuredLightLoadError, UnsupportedPLYFormatError

__all__ = [
    "CloudType",
    "CoordinateConvention",
    "PointCloudMetadata",
    "StructuredLightResult",
    "StructuredLightPaths",
    "PoseTarget",
    "PosePlanningInput",
    "InspectionPlan",
    "PosePlanner",
    "StructuredLightAdapter",
    "StructuredLightLoadError",
    "UnsupportedPLYFormatError",
]
