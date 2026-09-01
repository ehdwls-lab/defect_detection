from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from .coordinate_contract import StructuredLightResult
from .inspection_plan import InspectionPlan, PoseTarget


class PoseJSONError(ValueError):
    """Raised when a structured-light pose document is absent or unsafe."""


@dataclass(frozen=True)
class StructuredLightPlane:
    plane_name: str
    dominant: bool
    point_count: int
    point_ratio: float
    roll_deg: float
    pitch_deg: float
    raw_roll_deg: float
    raw_pitch_deg: float
    legacy_relative_z: dict[str, Any] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class StructuredLightPoseDocument:
    pose_json_path: Path
    input_ply: Path
    schema_version: str
    planes: tuple[StructuredLightPlane, ...]
    stm_z_command_allowed: bool
    coordinate_contract: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


def _finite_number(value: Any, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise PoseJSONError(f"{field_name} must be a number")
    result = float(value)
    if not math.isfinite(result):
        raise PoseJSONError(f"{field_name} must be finite")
    return result


def parse_pose_json(path: str | Path) -> StructuredLightPoseDocument:
    pose_path = Path(path).expanduser().resolve()
    try:
        with pose_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise PoseJSONError(f"Failed to read pose JSON: {pose_path}") from exc

    if not isinstance(payload, dict):
        raise PoseJSONError("Pose JSON root must be an object")
    raw_planes = payload.get("planes")
    if not isinstance(raw_planes, list) or not raw_planes:
        raise PoseJSONError("Pose JSON requires at least one plane")

    counts: list[int] = []
    for index, plane in enumerate(raw_planes):
        if not isinstance(plane, dict):
            raise PoseJSONError(f"planes[{index}] must be an object")
        count = plane.get("points_count")
        if isinstance(count, bool) or not isinstance(count, int) or count < 0:
            raise PoseJSONError(f"planes[{index}].points_count must be a non-negative integer")
        counts.append(count)
    total_count = sum(counts)
    if total_count <= 0:
        raise PoseJSONError("Plane point total must be positive")

    planes: list[StructuredLightPlane] = []
    for index, (raw, count) in enumerate(zip(raw_planes, counts)):
        prefix = f"planes[{index}]"
        if "roll_deg" not in raw or "pitch_deg" not in raw:
            raise PoseJSONError(f"{prefix} requires roll_deg and pitch_deg")
        roll = _finite_number(raw["roll_deg"], f"{prefix}.roll_deg")
        pitch = _finite_number(raw["pitch_deg"], f"{prefix}.pitch_deg")
        raw_roll = _finite_number(raw.get("raw_roll_deg", roll), f"{prefix}.raw_roll_deg")
        raw_pitch = _finite_number(raw.get("raw_pitch_deg", pitch), f"{prefix}.raw_pitch_deg")
        legacy = raw.get("legacy_relative_z")
        if legacy is not None:
            if not isinstance(legacy, dict):
                raise PoseJSONError(f"{prefix}.legacy_relative_z must be an object")
            if legacy.get("metric") is not False or legacy.get("stm_compatible") is not False:
                raise PoseJSONError(
                    f"{prefix}.legacy_relative_z must declare metric=false and stm_compatible=false"
                )
        known = {
            "plane_name", "dominant", "points_count", "pitch_deg", "roll_deg",
            "raw_pitch_deg", "raw_roll_deg", "legacy_relative_z",
        }
        planes.append(StructuredLightPlane(
            plane_name=str(raw.get("plane_name", f"Shot {index + 1}")),
            dominant=raw.get("dominant") is True,
            point_count=count,
            point_ratio=count / total_count,
            roll_deg=roll,
            pitch_deg=pitch,
            raw_roll_deg=raw_roll,
            raw_pitch_deg=raw_pitch,
            legacy_relative_z=legacy,
            metadata={key: value for key, value in raw.items() if key not in known},
        ))

    input_ply_value = payload.get("input_ply")
    if not isinstance(input_ply_value, str) or not input_ply_value:
        raise PoseJSONError("Pose JSON requires input_ply")
    input_ply = Path(input_ply_value).expanduser()
    if not input_ply.is_absolute():
        input_ply = pose_path.parent / input_ply

    # Missing permission is treated as denied. An explicit false is therefore
    # always preserved and can never be promoted into a motion command.
    stm_allowed = payload.get("stm_z_command_allowed", False)
    if not isinstance(stm_allowed, bool):
        raise PoseJSONError("stm_z_command_allowed must be boolean")
    known_root = {
        "schema_version", "input_ply", "segmented_ply", "coordinate_contract",
        "planes", "stm_z_command_allowed",
    }
    return StructuredLightPoseDocument(
        pose_json_path=pose_path,
        input_ply=input_ply.resolve(),
        schema_version=str(payload.get("schema_version", "")),
        planes=tuple(planes),
        stm_z_command_allowed=stm_allowed,
        coordinate_contract=dict(payload.get("coordinate_contract", {})),
        metadata={key: value for key, value in payload.items() if key not in known_root},
    )


class RealPosePlanner:
    """Convert pose-analysis JSON into a motion-safe V0 inspection plan.

    V0 emits one dominant roll/pitch target. It intentionally has no Z target.
    Parsed documents retain every plane for future multi-plane policy support.
    """

    def __init__(self, selection_policy: Literal["dominant_only", "multi_plane"] = "dominant_only") -> None:
        if selection_policy not in {"dominant_only", "multi_plane"}:
            raise ValueError(f"Unsupported selection policy: {selection_policy}")
        self.selection_policy = selection_policy

    @staticmethod
    def pose_json_for_result(result: StructuredLightResult) -> Path:
        configured = result.metadata.get("pose_json_path")
        if configured:
            return Path(configured).expanduser().resolve()
        return result.ply_path.with_name(f"{result.ply_path.stem}_pose.json")

    def parse(self, source: StructuredLightResult | str | Path) -> StructuredLightPoseDocument:
        if isinstance(source, StructuredLightResult):
            source.validate()
            return parse_pose_json(self.pose_json_for_result(source))
        return parse_pose_json(source)

    @staticmethod
    def select_dominant(document: StructuredLightPoseDocument) -> StructuredLightPlane:
        # point_count, then source order: deterministic even if dominant flags are
        # missing, duplicated, or inconsistent with the measured largest plane.
        return max(enumerate(document.planes), key=lambda item: (item[1].point_count, -item[0]))[1]

    def plan(self, source: StructuredLightResult | str | Path) -> InspectionPlan:
        document = self.parse(source)
        if self.selection_policy == "multi_plane":
            raise NotImplementedError("multi_plane planning is reserved for a future policy version")
        plane = self.select_dominant(document)
        legacy_present = plane.legacy_relative_z is not None
        pose = PoseTarget(
            pose_id=plane.plane_name,
            roll_deg=plane.roll_deg,
            pitch_deg=plane.pitch_deg,
            target_surface_id=plane.plane_name,
            confidence=plane.point_ratio,
            source="structured_light",
            metadata={
                "source": "structured_light",
                "plane_role": "dominant",
                "point_count": plane.point_count,
                "point_ratio": plane.point_ratio,
                "raw_roll": plane.raw_roll_deg,
                "raw_pitch": plane.raw_pitch_deg,
                "legacy_z_ignored": legacy_present,
                "stm_z_command_allowed": document.stm_z_command_allowed,
            },
        )
        return InspectionPlan(
            object_id=document.input_ply.stem,
            poses=[pose],
            source_ply=document.input_ply,
            metadata={
                "source": "structured_light",
                "schema_version": document.schema_version,
                "selection_policy": self.selection_policy,
                "pose_json_path": str(document.pose_json_path),
                "parsed_plane_count": len(document.planes),
                "stm_z_command_allowed": document.stm_z_command_allowed,
                "z_provided": False,
                "legacy_z_ignored": legacy_present,
                "platform_motion_allowed": False,
                "automatic_z_allowed": False,
            },
        )
