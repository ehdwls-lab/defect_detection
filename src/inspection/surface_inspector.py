from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

from src.integration.inspection_plan import PoseTarget


@dataclass(frozen=True)
class SurfaceInspectionResult:
    pose_id: str
    ready: bool
    valid_patch_count: int
    surface_area_px: int
    metadata: dict[str, Any] = field(default_factory=dict)


class SurfaceInspector(Protocol):
    def inspect(self, pose: PoseTarget, z_cm: float) -> SurfaceInspectionResult: ...


class MockSurfaceInspector:
    def inspect(self, pose: PoseTarget, z_cm: float) -> SurfaceInspectionResult:
        return SurfaceInspectionResult(
            pose_id=pose.pose_id,
            ready=True,
            valid_patch_count=32,
            surface_area_px=131072,
            metadata={"source": "mock", "mock": True, "z_cm": z_cm},
        )
