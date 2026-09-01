from __future__ import annotations

from .inspection_plan import InspectionPlan, PosePlanningInput, PoseTarget


class PosePlanner:
    """Placeholder interface for pose selection.

    This layer is intentionally thin. It should accept a structured-light result and
    return an inspection plan, but it must not perform platform control or final
    anomaly detection.
    """

    @staticmethod
    def plan(input_data: PosePlanningInput) -> InspectionPlan:
        if input_data.cloud is None:
            raise ValueError("Pose planner input is missing the cloud metadata.")

        pose = PoseTarget(
            pose_id="pose_01",
            pitch_deg=None,
            roll_deg=None,
            target_surface_id=None,
            confidence=None,
            source="placeholder_pose_planner",
            metadata={"status": "interface_only", "notes": "Pose algorithm not implemented yet."},
        )

        return InspectionPlan(
            object_id="structured_light_object",
            poses=[pose],
            source_ply=input_data.ply_path,
            metadata={
                "cloud_type": getattr(input_data.cloud, "cloud_type", None),
                "coordinate": getattr(input_data.cloud, "coordinate", None),
                "input_source": "structured_light_adapter",
            },
        )
