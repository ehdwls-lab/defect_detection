from __future__ import annotations

from .coordinate_contract import StructuredLightResult
from .inspection_plan import InspectionPlan, PoseTarget


class MockPosePlanner:
    def __init__(self, poses: tuple[PoseTarget, ...] | None = None) -> None:
        self.poses = poses or (
            PoseTarget("pose_01", pitch_deg=0.0, roll_deg=0.0, source="mock", metadata={"mock": True}),
            PoseTarget("pose_02", pitch_deg=0.0, roll_deg=5.0, source="mock", metadata={"mock": True}),
        )

    def plan(self, result: StructuredLightResult) -> InspectionPlan:
        result.validate()
        return InspectionPlan(
            object_id=result.run_id,
            poses=list(self.poses),
            source_ply=result.ply_path,
            metadata={"source": "mock", "mock": True},
        )
