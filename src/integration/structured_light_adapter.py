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


class UnsupportedPLYFormatError(StructuredLightLoadError):
    """Raised for PLY encodings not supported by this Phase 2 adapter."""


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

        artifacts = StructuredLightAdapter.discover_artifacts(run_dir)
        candidate_ply = StructuredLightAdapter.select_artifact(artifacts)
        if candidate_ply is None:
            raise PLYReadError(f"No final PLY found in structured-light run directory: {run_dir}")

        result = StructuredLightAdapter.from_ply(candidate_ply, run_id=run_dir.name, paths=paths)
        result.metadata["available_artifacts"] = [
            {"path": str(path), "cloud_type": cloud_type.value}
            for path, cloud_type in artifacts
        ]
        return result

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

        metadata = StructuredLightAdapter.parse_ply_header(ply)
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
    def discover_artifacts(run_dir: str | Path) -> list[tuple[Path, CloudType]]:
        root = Path(run_dir)
        return [
            (path, StructuredLightAdapter._infer_cloud_type(path, {}))
            for path in sorted(root.rglob("*.ply"))
        ]

    @staticmethod
    def select_artifact(artifacts: list[tuple[Path, CloudType]], preferred: CloudType = CloudType.OBJECT_ONLY) -> Path | None:
        priorities = {
            CloudType.OBJECT_ONLY: 0,
            CloudType.OBJECT_AND_PLATFORM: 1,
            CloudType.OBJECT_PLATFORM_FLOOR: 2,
            CloudType.UNKNOWN: 3,
        }
        matches = [item for item in artifacts if item[1] is preferred]
        if matches:
            return sorted(matches, key=lambda item: (priorities[item[1]], item[0].name))[0][0]
        known = [item for item in artifacts if item[1] is not CloudType.UNKNOWN]
        if not known:
            return None
        return sorted(known, key=lambda item: (priorities[item[1]], item[0].name))[0][0]

    @staticmethod
    def parse_ply_header(ply_path: str | Path) -> dict[str, Any]:
        ply_path = Path(ply_path)
        try:
            with ply_path.open("r", encoding="ascii") as handle:
                lines = []
                for _ in range(256):
                    line = handle.readline()
                    if not line:
                        break
                    lines.append(line)
                    if line.strip() == "end_header":
                        break
        except (OSError, UnicodeDecodeError) as exc:
            raise PLYReadError(f"Failed to read PLY header: {ply_path}") from exc

        vertex_count = 0
        has_color = False
        has_normals = False
        seen_end_header = False
        ply_format = None
        properties: set[str] = set()

        for line in lines:
            stripped = line.strip()
            if stripped.startswith("format "):
                parts = stripped.split()
                ply_format = parts[1] if len(parts) >= 2 else None
            if stripped.startswith("element vertex"):
                try:
                    vertex_count = int(stripped.split()[-1])
                except ValueError:
                    vertex_count = 0
            if stripped.startswith("property "):
                properties.add(stripped.split()[-1])
            if stripped == "end_header":
                seen_end_header = True
                break

        if not seen_end_header:
            raise PLYReadError(f"PLY header missing end_header marker: {ply_path}")
        if ply_format != "ascii":
            raise UnsupportedPLYFormatError(f"Only ASCII PLY is supported, got {ply_format!r}: {ply_path}")

        required_xyz = {"x", "y", "z"}
        if not required_xyz.issubset(properties):
            raise PLYReadError(f"PLY vertex properties must contain x/y/z: {ply_path}")
        has_color = {"red", "green", "blue"}.issubset(properties)
        has_normals = {"nx", "ny", "nz"}.issubset(properties)

        return {
            "vertex_count": vertex_count,
            "has_color": has_color,
            "has_normals": has_normals,
            "format": ply_format,
            "properties": sorted(properties),
        }

    @staticmethod
    def _infer_cloud_type(ply_path: Path, metadata: dict[str, Any]) -> CloudType:
        name = ply_path.name.lower()
        if "with_floor" in name or "바닥" in name:
            return CloudType.OBJECT_PLATFORM_FLOOR
        if "물체+플랫폼" in name or "object+platform" in name or "object_and_platform" in name:
            return CloudType.OBJECT_AND_PLATFORM
        if "물체만" in name or "object_only" in name:
            return CloudType.OBJECT_ONLY
        if name.startswith("final_dc_mask_phase") and "with_floor" not in name:
            return CloudType.OBJECT_ONLY
        # Segmented output is a visualization/analysis artifact. Its content
        # type is not inferred without an explicit manifest.
        return CloudType.UNKNOWN

    @staticmethod
    def _coordinate_convention(ply_path: Path, metadata: dict[str, Any]) -> CoordinateConvention:
        # The current structured-light analysis is intentionally conservative: no arbitrary
        # metric conversion is assumed unless the external pipeline proved it.
        return CoordinateConvention(
            x_axis="image-centered x coordinate, relative to the reconstructed image center",
            y_axis="image-centered y coordinate, relative to the reconstructed image center",
            z_axis="relative phase-derived height, not guaranteed to be calibrated mm",
            xy_unit="pixel",
            z_unit="phase_relative",
            origin_description="image center / structured-light reconstruction center",
            image_width=None,
            image_height=None,
            z_scale=None,
            z_sign=None,
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
