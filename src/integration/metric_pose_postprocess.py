from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable

from .metric_plane_merge import (
    MetricPlaneMergeConfig,
    merge_metric_physical_planes_in_pose_json,
)
from .metric_pose import MetricFitConfig
from .platform_pose_calibration import (
    CAMERA_PLATFORM_RP_20260903,
    CameraPlatformRPCalibration,
    apply_calibration_to_pose_json,
)
from .platform_alignment import apply_alignment_policy_to_pose_json


class MetricPosePostprocessError(RuntimeError):
    pass


def postprocess_metric_pose_json(
    path: str | Path,
    *,
    current_platform_roll_deg: float | None = None,
    current_platform_pitch_deg: float | None = None,
    fresh_telemetry_reader: Callable[[], Any] | None = None,
    calibration: CameraPlatformRPCalibration | None = CAMERA_PLATFORM_RP_20260903,
    merge_config: MetricPlaneMergeConfig | None = None,
    fit_config: MetricFitConfig | None = None,
) -> dict[str, Any]:
    """Create the motion-planning pose artifact from raw metric SL planes.

    The entry point is intentionally shared by production integration and the
    read/scan-only measurement tool. Existing raw reachability and axis-contract
    rejection fields are discarded by merge/refit and recalculated here.
    """
    pose_path = Path(path).expanduser().resolve()
    try:
        before = json.loads(pose_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise MetricPosePostprocessError(f"failed to read pose JSON: {pose_path}") from exc
    raw_plane_count = int(before.get("raw_plane_count", len(before.get("planes", []))))
    # Reprocessing an already merged artifact must not try to consume raw
    # memberships again. It may still be recalibrated from newer telemetry.
    if "metric_physical_plane_count" not in before:
        merge_metric_physical_planes_in_pose_json(
            pose_path, merge_config=merge_config, fit_config=fit_config,
        )
    if fresh_telemetry_reader is not None:
        telemetry = fresh_telemetry_reader()
        if getattr(telemetry, "homing", False):
            raise MetricPosePostprocessError("fresh platform telemetry reports homing")
        if getattr(telemetry, "stable", False) is not True:
            raise MetricPosePostprocessError("fresh platform telemetry is not stable")
        current_platform_roll_deg = float(telemetry.roll_deg)
        current_platform_pitch_deg = float(telemetry.pitch_deg)
    if current_platform_roll_deg is None or current_platform_pitch_deg is None:
        raise MetricPosePostprocessError(
            "fresh platform roll/pitch telemetry is required for production postprocess"
        )
    apply_calibration_to_pose_json(
        pose_path,
        current_platform_roll_deg=current_platform_roll_deg,
        current_platform_pitch_deg=current_platform_pitch_deg,
        calibration=calibration,
    )
    apply_alignment_policy_to_pose_json(pose_path)
    try:
        final = json.loads(pose_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise MetricPosePostprocessError(f"failed to read final pose JSON: {pose_path}") from exc
    planes = final.get("planes", [])
    return {
        "pose_json_path": str(pose_path),
        "raw_plane_count": raw_plane_count,
        "metric_physical_plane_count": int(
            final.get("metric_physical_plane_count", len(planes))
        ),
        "reachable_pose_count": sum(
            isinstance(plane, dict)
            and isinstance(plane.get("metric_pose"), dict)
            and plane["metric_pose"].get("reachable") is True
            for plane in planes
        ),
        "current_platform_roll_deg": float(current_platform_roll_deg),
        "current_platform_pitch_deg": float(current_platform_pitch_deg),
    }
