from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from .coordinate_contract import (
    CloudType,
    CoordinateConvention,
    DEFAULT_COORDINATE_CONVENTION,
    PointCloudMetadata,
    StructuredLightResult,
    StructuredLightPaths,
)


class StructuredLightLoadError(RuntimeError):
    """Raised when a structured-light run cannot be resolved into a valid result."""


class PLYReadError(StructuredLightLoadError):
    """Raised when a point cloud file cannot be parsed."""


class UnsupportedCloudTypeError(StructuredLightLoadError):
    """Raised when the cloud type cannot be confidently classified."""


class StructuredLightAdapter:
    """Normalize external structured-light outputs into a stable defect-detection contract.

    This adapter intentionally does not recreate any structured-light algorithm. It only
    discovers the right artifact set, validates it, and exposes a clear metadata contract.
    """

    @staticmethod
    def from_directory(directory: str | Path, *, paths: StructuredLightPaths | None = None) -> StructuredLightResult:
        run_dir = Path(directory).expanduser().resolve()
        if not run_dir.exists():
            raise StructuredLightLoadError(f"Structured-light directory does not exist: {run_dir}")

        candidate_ply = StructuredLightAdapter._find_latest_ply(run_dir)
        if candidate_ply is None:
            raise PLYReadError(f"No final PLY found in structured-light run directory: {run_dir}")

        return StructuredLightAdapter.from_ply(candidate_ply, run_id=run_dir.name, paths=paths)

    @staticmethod
    def from_ply(
        ply_path: str | Path,
        *,
        run_id: str | None = None,
        paths: StructuredLightPaths | None = None,
    ) -> StructuredLightResult:
        ply = Path(ply_path).expanduser().resolve()
        if not ply.exists():
            raise PLYReadError(f"PLY file does not exist: {ply}")

        metadata = StructuredLightAdapter._read_ply_header(ply)
        cloud_type = StructuredLightAdapter._infer_cloud_type(ply, metadata)
        coordinate = StructuredLightAdapter._coordinate_convention(ply, metadata)

        point_count = metadata.get("vertex_count", 0)
        cloud = PointCloudMetadata(
            ply_path=ply,
            point_count=point_count,
            has_color=metadata.get("has_color", True),
            has_normals=metadata.get("has_normals", False),
            includes_object=cloud_type in {CloudType.OBJECT_ONLY, CloudType.OBJECT_AND_PLATFORM, CloudType.OBJECT_PLATFORM_FLOOR},
            includes_platform=cloud_type in {CloudType.OBJECT_AND_PLATFORM, CloudType.OBJECT_PLATFORM_FLOOR},
            includes_floor=cloud_type is CloudType.OBJECT_PLATFORM_FLOOR,
            cloud_type=cloud_type,
            coordinate=coordinate,
        )

        object_mask = None
        phase_path = None
        depth_path = None

        if paths is not None:
            if paths.current_run_dir is not None:
                object_mask = StructuredLightAdapter._find_object_mask(paths.current_run_dir)
                phase_path = StructuredLightAdapter._find_phase_artifact(paths.current_run_dir)
                depth_path = StructuredLightAdapter._find_depth_artifact(paths.current_run_dir)

        return StructuredLightResult(
            run_id=run_id or ply.stem,
            ply_path=ply,
            object_mask_path=object_mask,
            phase_path=phase_path,
            depth_path=depth_path,
            cloud=cloud,
            metadata={
                "schema_version": "structured_light_interface_v1",
                "source": "adapter",
                "source_file": str(ply),
                "cloud_type": cloud_type.value,
                "header": metadata,
            },
        )

    @staticmethod
    def _find_latest_ply(run_dir: Path) -> Path | None:
        candidates = sorted(run_dir.rglob("*.ply"), key=lambda p: p.stat().st_mtime, reverse=True)
        return candidates[0] if candidates else None

    @staticmethod
    def _read_ply_header(ply_path: Path) -> dict[str, Any]:
        try:
            with ply_path.open("r", encoding="ascii") as handle:
                lines = handle.readlines()
        except OSError as exc:
            raise PLYReadError(f"Failed to read PLY header: {ply_path}") from exc

        vertex_count = 0
        has_color = False
        has_normals = False
        seen_end_header = False

        for line in lines:
            stripped = line.strip()
            if stripped.startswith("element vertex"):
                try:
                    vertex_count = int(stripped.split()[-1])
                except ValueError:
                    vertex_count = 0
            if stripped.startswith("property uchar red"):
                has_color = True
            if stripped.startswith("property float nx") or stripped.startswith("property float nx"):
                has_normals = True
            if stripped == "end_header":
                seen_end_header = True
                break

        if not seen_end_header:
            raise PLYReadError(f"PLY header missing end_header marker: {ply_path}")

        return {
            "vertex_count": vertex_count,
            "has_color": has_color,
            "has_normals": has_normals,
        }

    @staticmethod
    def _infer_cloud_type(ply_path: Path, metadata: dict[str, Any]) -> CloudType:
        name = ply_path.name.lower()
        if "with_floor" in name:
            return CloudType.OBJECT_PLATFORM_FLOOR
        if "platform" in name or "floor" in name:
            return CloudType.OBJECT_AND_PLATFORM
        if "final" in name or "object" in name:
            return CloudType.OBJECT_ONLY
        if metadata.get("vertex_count", 0) <= 0:
            return CloudType.UNKNOWN
        return CloudType.OBJECT_ONLY

    @staticmethod
    def _coordinate_convention(ply_path: Path, metadata: dict[str, Any]) -> CoordinateConvention:
        # The current structured-light analysis is intentionally conservative: no arbitrary
        # metric conversion is assumed unless the external pipeline proved it.
        return CoordinateConvention(
            x_axis="image-centered x coordinate, relative to the reconstructed image center",
            y_axis="image-centered y coordinate, relative to the reconstructed image center",
            z_axis="relative phase-derived height, not guaranteed to be calibrated mm",
            xy_unit="pixel-relative",
            z_unit="relative_phase_scale",
            origin_description="image center / structured-light reconstruction center",
            image_width=None,
            image_height=None,
        )

    @staticmethod
    def _find_object_mask(run_dir: Path) -> Path | None:
        candidates = sorted(run_dir.rglob("*mask*.npy"), key=lambda p: p.stat().st_mtime, reverse=True)
        return candidates[0] if candidates else None

    @staticmethod
    def _find_phase_artifact(run_dir: Path) -> Path | None:
        phase_files = sorted(run_dir.rglob("phase_*.png"), key=lambda p: p.stat().st_mtime, reverse=True)
        return phase_files[0] if phase_files else None

    @staticmethod
    def _find_depth_artifact(run_dir: Path) -> Path | None:
        depth_files = sorted(run_dir.rglob("*depth*.npy"), key=lambda p: p.stat().st_mtime, reverse=True)
        if depth_files:
            return depth_files[0]
        depth_png = sorted(run_dir.rglob("*depth*.png"), key=lambda p: p.stat().st_mtime, reverse=True)
        return depth_png[0] if depth_png else None
