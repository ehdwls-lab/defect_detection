from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .metric_pose import (
    MetricFitConfig,
    MetricPoseError,
    backproject_depth_pixels,
    camera_tilt_degrees,
    fit_metric_plane_ransac,
    load_intrinsics,
)


METRIC_PLANE_NORMAL_THRESHOLD_DEG = 2.0
METRIC_PLANE_COPLANAR_DISTANCE_MM = 5.0


class MetricPlaneMergeError(ValueError):
    pass


@dataclass(frozen=True)
class MetricPlaneMergeConfig:
    normal_threshold_deg: float = METRIC_PLANE_NORMAL_THRESHOLD_DEG
    coplanar_distance_mm: float = METRIC_PLANE_COPLANAR_DISTANCE_MM

    def validate(self) -> None:
        if not math.isfinite(self.normal_threshold_deg) or not 0 <= self.normal_threshold_deg <= 90:
            raise MetricPlaneMergeError("normal_threshold_deg must be within [0, 90]")
        if not math.isfinite(self.coplanar_distance_mm) or self.coplanar_distance_mm < 0:
            raise MetricPlaneMergeError("coplanar_distance_mm must be non-negative")


@dataclass(frozen=True)
class MetricPlaneCandidate:
    source_plane_index: int
    source_plane_name: str
    dominant: bool
    structured_light_points: int
    pixels_uv: np.ndarray
    xyz_mm: np.ndarray
    normal: np.ndarray
    center_mm: np.ndarray


def _same_physical_plane(
    first: MetricPlaneCandidate,
    second: MetricPlaneCandidate,
    config: MetricPlaneMergeConfig,
) -> bool:
    n1 = np.asarray(first.normal, dtype=np.float64)
    n2 = np.asarray(second.normal, dtype=np.float64)
    n1 /= np.linalg.norm(n1)
    n2 /= np.linalg.norm(n2)
    angular_difference = math.degrees(math.acos(float(np.clip(abs(n1 @ n2), -1.0, 1.0))))
    if angular_difference > config.normal_threshold_deg:
        return False
    center_delta = np.asarray(second.center_mm) - np.asarray(first.center_mm)
    # Symmetric center-to-plane distance prevents parallel surfaces at
    # different metric depths from being merged.
    first_to_second = abs(float(center_delta @ n1))
    second_to_first = abs(float(center_delta @ n2))
    return max(first_to_second, second_to_first) <= config.coplanar_distance_mm


def group_metric_plane_candidates(
    candidates: list[MetricPlaneCandidate],
    config: MetricPlaneMergeConfig | None = None,
) -> list[list[MetricPlaneCandidate]]:
    config = config or MetricPlaneMergeConfig()
    config.validate()
    parents = list(range(len(candidates)))

    def find(index: int) -> int:
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index

    def union(a: int, b: int) -> None:
        a_root, b_root = find(a), find(b)
        if a_root != b_root:
            parents[b_root] = a_root

    for first in range(len(candidates)):
        for second in range(first + 1, len(candidates)):
            if _same_physical_plane(candidates[first], candidates[second], config):
                union(first, second)
    grouped: dict[int, list[MetricPlaneCandidate]] = {}
    for index, candidate in enumerate(candidates):
        grouped.setdefault(find(index), []).append(candidate)
    return list(grouped.values())


def refit_metric_plane_group(
    group: list[MetricPlaneCandidate],
    physical_plane_index: int,
    fit_config: MetricFitConfig,
) -> dict[str, Any]:
    if not group:
        raise MetricPlaneMergeError("cannot refit an empty physical-plane group")
    pixels = np.concatenate([candidate.pixels_uv for candidate in group], axis=0)
    xyz_all = np.concatenate([candidate.xyz_mm for candidate in group], axis=0)
    if len(pixels) == len(xyz_all):
        _, unique_indices = np.unique(pixels, axis=0, return_index=True)
        xyz = xyz_all[np.sort(unique_indices)]
    else:
        xyz = xyz_all
    normal, center, residual, inlier_ratio = fit_metric_plane_ransac(xyz, fit_config)
    camera_roll, camera_pitch = camera_tilt_degrees(normal)
    source_indices = [candidate.source_plane_index for candidate in group]
    source_names = [candidate.source_plane_name for candidate in group]
    total_membership = len(np.unique(pixels, axis=0))
    total_depth_points = len(xyz)
    coverage = total_depth_points / total_membership if total_membership else 0.0
    return {
        "physical_plane_index": physical_plane_index,
        "merged_source_plane_indices": source_indices,
        "merged_source_plane_names": source_names,
        "dominant": any(candidate.dominant for candidate in group),
        "points_count": sum(candidate.structured_light_points for candidate in group),
        "total_depth_points": total_depth_points,
        "depth_coverage": coverage,
        "metric_normal": normal.tolist(),
        "metric_center_xyz_mm": center.tolist(),
        "camera_roll_deg": camera_roll,
        "camera_pitch_deg": camera_pitch,
        "ransac_median_residual_mm": residual,
        "ransac_inlier_ratio": inlier_ratio,
    }


def merge_metric_physical_planes_in_pose_json(
    path: str | Path,
    *, merge_config: MetricPlaneMergeConfig | None = None,
    fit_config: MetricFitConfig | None = None,
) -> None:
    """Replace raw planes with metric physical planes using saved depth membership."""
    pose_path = Path(path).expanduser().resolve()
    try:
        payload = json.loads(pose_path.read_text(encoding="utf-8"))
        planes = payload["planes"]
        contract = payload["metric_pose_contract"]
        depth_path = Path(contract["depth_path"]).expanduser().resolve()
        intrinsics_path = Path(contract["intrinsics_path"]).expanduser().resolve()
        membership_path = Path(contract["plane_membership_path"]).expanduser().resolve()
        depth = np.load(depth_path).astype(np.float32)
        intrinsics = load_intrinsics(intrinsics_path)
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise MetricPlaneMergeError(f"metric physical-plane artifacts are unavailable: {exc}") from exc
    if not isinstance(planes, list) or not planes:
        raise MetricPlaneMergeError("pose JSON requires raw planes for physical merge")
    config = fit_config or MetricFitConfig()
    candidates: list[MetricPlaneCandidate] = []
    try:
        with np.load(membership_path) as memberships:
            for fallback_index, plane in enumerate(planes):
                metric = plane.get("metric_pose", {})
                membership = plane.get("pixel_membership", {})
                if metric.get("physical_metric") is not True or metric.get("status") not in {
                    "METRIC_VALID", "REACHABLE", "UNREACHABLE",
                }:
                    continue
                pixels = np.asarray(memberships[membership["sidecar_key"]], dtype=np.int64)
                xyz, _ = backproject_depth_pixels(depth, pixels, intrinsics, config)
                if len(xyz) < config.min_points:
                    continue
                candidates.append(MetricPlaneCandidate(
                    source_plane_index=int(plane.get("source_plane_index", fallback_index)),
                    source_plane_name=str(plane.get("plane_name", f"Shot {fallback_index + 1}")),
                    dominant=plane.get("dominant") is True,
                    structured_light_points=int(plane.get("points_count", 0)),
                    pixels_uv=pixels,
                    xyz_mm=xyz,
                    normal=np.asarray(metric["normal_xyz"], dtype=np.float64),
                    center_mm=np.asarray(metric["center_xyz_mm"], dtype=np.float64),
                ))
    except (OSError, KeyError, TypeError, ValueError) as exc:
        raise MetricPlaneMergeError(f"invalid metric plane membership: {exc}") from exc
    if not candidates:
        raise MetricPlaneMergeError("no metric-valid raw planes are available for physical merge")

    groups = group_metric_plane_candidates(candidates, merge_config)
    merged_planes = []
    for physical_index, group in enumerate(groups):
        refit = refit_metric_plane_group(group, physical_index, config)
        representative = next(
            plane for fallback_index, plane in enumerate(planes)
            if int(plane.get("source_plane_index", fallback_index)) == group[0].source_plane_index
        )
        metric_pose = dict(representative.get("metric_pose", {}))
        metric_pose.update(
            status="METRIC_VALID", reachable=False, reject_reason="platform calibration pending",
            normal_xyz=refit["metric_normal"], center_xyz_mm=refit["metric_center_xyz_mm"],
            depth_points_count=refit["total_depth_points"], depth_coverage=refit["depth_coverage"],
            camera_roll_deg=refit["camera_roll_deg"], camera_pitch_deg=refit["camera_pitch_deg"],
            ransac_median_residual_mm=refit["ransac_median_residual_mm"],
            ransac_inlier_ratio=refit["ransac_inlier_ratio"],
        )
        merged_planes.append({
            "source_plane_index": refit["physical_plane_index"],
            "physical_plane_index": refit["physical_plane_index"],
            "plane_name": f"Physical Plane {physical_index + 1}",
            "dominant": refit["dominant"],
            "points_count": refit["points_count"],
            "pitch_deg": float(representative.get("pitch_deg", 0.0)),
            "roll_deg": float(representative.get("roll_deg", 0.0)),
            "raw_pitch_deg": float(representative.get("raw_pitch_deg", 0.0)),
            "raw_roll_deg": float(representative.get("raw_roll_deg", 0.0)),
            "legacy_pose_semantics": "phase_space_heuristic_metadata_only",
            "merged_source_plane_indices": refit["merged_source_plane_indices"],
            "merged_source_plane_names": refit["merged_source_plane_names"],
            "total_depth_points": refit["total_depth_points"],
            "depth_coverage": refit["depth_coverage"],
            "metric_normal": refit["metric_normal"],
            "camera_roll_deg": refit["camera_roll_deg"],
            "camera_pitch_deg": refit["camera_pitch_deg"],
            "pixel_membership": None,
            "metric_pose": metric_pose,
            "legacy_relative_z": representative.get("legacy_relative_z"),
        })
    payload["raw_plane_count"] = len(planes)
    payload["metric_physical_plane_count"] = len(merged_planes)
    payload["metric_plane_merge"] = {
        "normal_threshold_deg": (merge_config or MetricPlaneMergeConfig()).normal_threshold_deg,
        "coplanar_distance_mm": (merge_config or MetricPlaneMergeConfig()).coplanar_distance_mm,
        "method": "metric normal + symmetric center-to-plane distance; union XYZ RANSAC/SVD refit",
    }
    for plane in merged_planes:
        plane["raw_plane_count"] = len(planes)
        plane["metric_physical_plane_count"] = len(merged_planes)
    payload["planes"] = merged_planes
    temporary = pose_path.with_name(f".{pose_path.name}.physical-merge.tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(pose_path)
