from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any


class CloudType(str, Enum):
    OBJECT_ONLY = "OBJECT_ONLY"
    OBJECT_AND_PLATFORM = "OBJECT_AND_PLATFORM"
    OBJECT_PLATFORM_FLOOR = "OBJECT_PLATFORM_FLOOR"
    UNKNOWN = "UNKNOWN"


@dataclass(frozen=True)
class CoordinateConvention:
    x_axis: str
    y_axis: str
    z_axis: str
    xy_unit: str
    z_unit: str
    origin_description: str
    image_width: int | None = None
    image_height: int | None = None


@dataclass(frozen=True)
class PointCloudMetadata:
    ply_path: Path
    point_count: int
    has_color: bool
    has_normals: bool
    includes_object: bool
    includes_platform: bool
    includes_floor: bool
    cloud_type: CloudType
    coordinate: CoordinateConvention


@dataclass(frozen=True)
class StructuredLightPaths:
    root: Path
    initial_setup_dir: Path | None = None
    current_run_dir: Path | None = None


@dataclass(frozen=True)
class StructuredLightResult:
    run_id: str
    ply_path: Path
    object_mask_path: Path | None = None
    phase_path: Path | None = None
    depth_path: Path | None = None
    cloud: PointCloudMetadata | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        if not self.ply_path.exists():
            raise FileNotFoundError(f"Structured-light PLY not found: {self.ply_path}")
        if self.cloud is None:
            raise ValueError("Structured-light result is missing cloud metadata.")
        if self.cloud.point_count <= 0:
            raise ValueError(f"Structured-light cloud is empty: {self.ply_path}")


DEFAULT_COORDINATE_CONVENTION = CoordinateConvention(
    x_axis="image-centered horizontal axis, derived from pixel column relative to image center",
    y_axis="image-centered vertical axis, derived from pixel row relative to image center",
    z_axis="relative phase-derived height; not guaranteed to be calibrated mm unless proven in the external pipeline",
    xy_unit="pixel-relative",
    z_unit="relative_phase_scale",
    origin_description="image center / structured-light reconstruction center",
    image_width=None,
    image_height=None,
)
