from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class PoseTarget:
    pose_id: str
    pitch_deg: float | None = None
    roll_deg: float | None = None
    target_surface_id: str | None = None
    confidence: float | None = None
    source: str = "pose_planner"
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class PosePlanningInput:
    cloud: Any
    ply_path: Path
    object_mask_path: Path | None = None
    coordinate: Any | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class InspectionPlan:
    object_id: str
    poses: list[PoseTarget]
    source_ply: Path
    metadata: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        if not self.poses:
            raise ValueError("Inspection plan requires at least one pose target.")
        if not self.source_ply.exists():
            raise FileNotFoundError(f"Source PLY does not exist: {self.source_ply}")
