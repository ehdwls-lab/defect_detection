from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .platform_limits import PLATFORM_RP_LIMIT_DEG


class MetricPoseError(ValueError):
    """Raised when metric pose provenance or geometry is unsafe."""


@dataclass(frozen=True)
class CameraIntrinsics:
    fx: float
    fy: float
    cx: float
    cy: float
    width: int
    height: int
    source: str
    aligned_to: str = "color"

    def validate(self) -> None:
        values = (self.fx, self.fy, self.cx, self.cy)
        if not all(math.isfinite(value) for value in values):
            raise MetricPoseError("camera intrinsics must be finite")
        if self.fx <= 0 or self.fy <= 0 or self.width <= 0 or self.height <= 0:
            raise MetricPoseError("camera intrinsics dimensions/focal lengths must be positive")
        if self.aligned_to != "color":
            raise MetricPoseError("only depth aligned to the color grid is supported")


@dataclass(frozen=True)
class PlatformAxisContract:
    verified: bool
    roll_source: str = "camera_roll"
    pitch_source: str = "camera_pitch"
    roll_sign: int = 1
    pitch_sign: int = 1
    reference: str = ""

    def validate(self) -> None:
        if self.roll_source not in {"camera_roll", "camera_pitch"}:
            raise MetricPoseError("unsupported roll_source")
        if self.pitch_source not in {"camera_roll", "camera_pitch"}:
            raise MetricPoseError("unsupported pitch_source")
        if self.roll_source == self.pitch_source:
            raise MetricPoseError("platform roll and pitch cannot use the same camera axis")
        if self.roll_sign not in {-1, 1} or self.pitch_sign not in {-1, 1}:
            raise MetricPoseError("axis signs must be -1 or +1")


@dataclass(frozen=True)
class MetricFitConfig:
    min_depth_mm: float = 80.0
    max_depth_mm: float = 2000.0
    ransac_threshold_mm: float = 2.5
    ransac_iterations: int = 160
    min_points: int = 500
    max_points: int = 7000
    min_coverage: float = 0.25
    max_tilt_deg: float = PLATFORM_RP_LIMIT_DEG


def load_intrinsics(path: str | Path) -> CameraIntrinsics:
    source_path = Path(path).expanduser().resolve()
    try:
        payload = json.loads(source_path.read_text(encoding="utf-8"))
        raw = payload.get("color_intrinsics", payload)
        intrinsics = CameraIntrinsics(
            fx=float(raw["fx"]), fy=float(raw["fy"]),
            cx=float(raw["cx"]), cy=float(raw["cy"]),
            width=int(raw["width"]), height=int(raw["height"]),
            source=str(payload.get("source", source_path)),
            aligned_to=str(payload.get("depth_alignment", "")),
        )
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise MetricPoseError(f"invalid intrinsics artifact: {source_path}") from exc
    intrinsics.validate()
    return intrinsics


def load_axis_contract(path: str | Path | None) -> PlatformAxisContract:
    if path is None:
        return PlatformAxisContract(verified=False, reference="missing axis/sign calibration")
    source_path = Path(path).expanduser().resolve()
    try:
        raw = json.loads(source_path.read_text(encoding="utf-8"))
        contract = PlatformAxisContract(
            verified=raw.get("verified") is True,
            roll_source=str(raw.get("roll_source", "camera_roll")),
            pitch_source=str(raw.get("pitch_source", "camera_pitch")),
            roll_sign=int(raw.get("roll_sign", 1)),
            pitch_sign=int(raw.get("pitch_sign", 1)),
            reference=str(raw.get("reference", source_path)),
        )
    except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise MetricPoseError(f"invalid platform axis contract: {source_path}") from exc
    contract.validate()
    return contract


def ply_xy_to_depth_pixels(
    points_xyz: np.ndarray, width: int, height: int, *, transform: str = "rotate_180"
) -> np.ndarray:
    """Recover integer pixels from X=u-w/2, Y=h/2-v and map to D2C depth."""
    points = np.asarray(points_xyz, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] < 2:
        raise MetricPoseError("plane points must have shape Nx3")
    u_sl = np.rint(points[:, 0] + width / 2.0).astype(np.int64)
    v_sl = np.rint(height / 2.0 - points[:, 1]).astype(np.int64)
    if transform == "rotate_180":
        u, v = width - 1 - u_sl, height - 1 - v_sl
    elif transform == "identity":
        u, v = u_sl, v_sl
    else:
        raise MetricPoseError(f"unsupported pixel transform: {transform}")
    pixels = np.column_stack((u, v))
    valid = (u >= 0) & (u < width) & (v >= 0) & (v < height)
    return np.unique(pixels[valid], axis=0)


def backproject_depth_pixels(
    depth_mm: np.ndarray, pixels_uv: np.ndarray, intrinsics: CameraIntrinsics,
    config: MetricFitConfig,
) -> tuple[np.ndarray, int]:
    depth = np.asarray(depth_mm)
    if depth.ndim != 2 or depth.shape != (intrinsics.height, intrinsics.width):
        raise MetricPoseError(
            f"depth/intrinsics grid mismatch: depth={depth.shape}, "
            f"intrinsics={(intrinsics.height, intrinsics.width)}"
        )
    pixels = np.asarray(pixels_uv, dtype=np.int64)
    if pixels.ndim != 2 or pixels.shape[1] != 2:
        raise MetricPoseError("pixels must have shape Nx2")
    if not len(pixels):
        return np.empty((0, 3), dtype=np.float64), 0
    u, v = pixels[:, 0], pixels[:, 1]
    inside = (u >= 0) & (u < intrinsics.width) & (v >= 0) & (v < intrinsics.height)
    u, v = u[inside], v[inside]
    z = depth[v, u].astype(np.float64)
    valid = np.isfinite(z) & (z >= config.min_depth_mm) & (z <= config.max_depth_mm)
    u, v, z = u[valid], v[valid], z[valid]
    x = (u.astype(np.float64) - intrinsics.cx) * z / intrinsics.fx
    y = (v.astype(np.float64) - intrinsics.cy) * z / intrinsics.fy
    return np.column_stack((x, y, z)), int(np.count_nonzero(valid))


def fit_metric_plane_ransac(
    points: np.ndarray, config: MetricFitConfig,
) -> tuple[np.ndarray, np.ndarray, float, float]:
    """Fit a metric plane and return normal, center, median residual and inlier ratio."""
    if len(points) < config.min_points:
        raise MetricPoseError(f"insufficient valid depth points: {len(points)} < {config.min_points}")
    rng = np.random.default_rng(42)
    sample = points
    if len(sample) > config.max_points:
        sample = sample[rng.choice(len(sample), config.max_points, replace=False)]
    best = None
    for _ in range(max(20, config.ransac_iterations)):
        a, b, c = sample[rng.choice(len(sample), 3, replace=False)]
        normal = np.cross(b - a, c - a)
        norm = np.linalg.norm(normal)
        if norm <= 1e-12:
            continue
        normal /= norm
        residual = np.abs((sample - a) @ normal)
        inliers = residual <= config.ransac_threshold_mm
        if best is None or np.count_nonzero(inliers) > np.count_nonzero(best):
            best = inliers
    if best is None or np.count_nonzero(best) < config.min_points:
        raise MetricPoseError("metric RANSAC did not find enough plane inliers")
    inlier_points = sample[best]
    center = inlier_points.mean(axis=0)
    _, _, vh = np.linalg.svd(inlier_points - center, full_matrices=False)
    normal = vh[-1]
    normal /= np.linalg.norm(normal)
    if normal[2] > 0:  # face the camera; Orbbec +Z points away from camera
        normal = -normal
    residual = np.abs((inlier_points - center) @ normal)
    return normal, center, float(np.median(residual)), float(np.mean(best))


def camera_tilt_degrees(normal_xyz: np.ndarray) -> tuple[float, float]:
    nx, ny, nz = (float(value) for value in normal_xyz)
    if nz > 0:
        nx, ny, nz = -nx, -ny, -nz
    camera_roll = math.degrees(math.atan2(nx, -nz))
    camera_pitch = math.degrees(math.atan2(ny, -nz))
    return camera_roll, camera_pitch


def build_metric_pose(
    depth_mm: np.ndarray,
    pixels_uv: np.ndarray,
    intrinsics: CameraIntrinsics,
    axis_contract: PlatformAxisContract,
    config: MetricFitConfig | None = None,
) -> dict[str, Any]:
    config = config or MetricFitConfig()
    intrinsics.validate()
    axis_contract.validate()
    total = int(len(pixels_uv))
    result: dict[str, Any] = {
        "source": "orbbec_depth", "physical_metric": True,
        "depth_unit": "mm", "depth_points_count": 0,
        "depth_coverage": 0.0, "status": "DETECTED",
        "reachable": False, "reject_reason": None,
        "intrinsics_source": intrinsics.source,
        "axis_contract": asdict(axis_contract),
    }
    try:
        xyz, valid_count = backproject_depth_pixels(depth_mm, pixels_uv, intrinsics, config)
        coverage = valid_count / total if total else 0.0
        result.update(depth_points_count=valid_count, depth_coverage=coverage)
        if coverage < config.min_coverage:
            raise MetricPoseError(
                f"depth coverage {coverage:.6f} below minimum {config.min_coverage:.6f}"
            )
        normal, center, residual, inlier_ratio = fit_metric_plane_ransac(xyz, config)
        camera_roll, camera_pitch = camera_tilt_degrees(normal)
        camera_angles = {"camera_roll": camera_roll, "camera_pitch": camera_pitch}
        roll = axis_contract.roll_sign * camera_angles[axis_contract.roll_source]
        pitch = axis_contract.pitch_sign * camera_angles[axis_contract.pitch_source]
        result.update(
            status="METRIC_VALID", normal_xyz=normal.tolist(), center_xyz_mm=center.tolist(),
            ransac_median_residual_mm=residual, camera_roll_deg=camera_roll,
            camera_pitch_deg=camera_pitch, ransac_inlier_ratio=inlier_ratio,
            roll_deg=roll, pitch_deg=pitch,
        )
        if not axis_contract.verified:
            result["reject_reason"] = "platform axis/sign contract is unresolved"
        elif (
            abs(roll) > config.max_tilt_deg + 1e-4
            or abs(pitch) > config.max_tilt_deg + 1e-4
        ):
            result.update(status="UNREACHABLE", reject_reason="platform tilt limit exceeded")
        else:
            result.update(status="REACHABLE", reachable=True)
    except MetricPoseError as exc:
        result["reject_reason"] = str(exc)
    return result
