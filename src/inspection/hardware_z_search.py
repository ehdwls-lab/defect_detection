from __future__ import annotations

import csv
import json
import math
import warnings
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any, Callable, Protocol

import numpy as np

from src.camera.controller import RGBDepthFrame
from src.config import InspectionConfig
from src.core.surface_geometry import (
    evaluate_surface_geometry_readiness,
    extract_surface_geometry,
)
from src.inspection.automatic_z_search import AutomaticZSearch
from src.inspection.z_search_types import InspectionQualitySample
from src.integration.projector_controller import ProjectorState
from src.inspection.adaptive_pose import AdaptivePose, adaptive_pose_for_z


@dataclass(frozen=True)
class SensorQualityConfig:
    depth_min_mm: float | None = None
    depth_max_mm: float | None = None
    min_depth_valid_ratio: float | None = None
    min_roi_depth_coverage: float | None = None
    max_invalid_ratio: float | None = None
    max_saturation_ratio: float | None = None
    max_dark_ratio: float | None = None
    min_sharpness: float | None = None
    min_contrast: float | None = None
    max_edge_occupancy_ratio: float | None = None
    saturation_value: int | None = None
    dark_value: int | None = None
    score_weights: dict[str, float] = field(default_factory=dict)
    schema_version: str = "automatic_z_quality_v1"
    profile: str | None = None
    provenance: dict[str, Any] = field(default_factory=dict)
    unresolved: tuple[str, ...] = field(default_factory=tuple)

    def validate(self) -> None:
        numeric = (
            self.depth_min_mm, self.depth_max_mm,
            self.min_depth_valid_ratio, self.min_roi_depth_coverage,
            self.max_invalid_ratio, self.max_saturation_ratio, self.max_dark_ratio,
            self.min_sharpness, self.min_contrast, self.max_edge_occupancy_ratio,
            *self.score_weights.values(),
        )
        if not all(value is None or math.isfinite(float(value)) for value in numeric):
            raise ValueError("quality thresholds and weights must be finite")
        for name in (
            "min_depth_valid_ratio", "min_roi_depth_coverage", "max_invalid_ratio",
            "max_saturation_ratio", "max_dark_ratio", "max_edge_occupancy_ratio",
        ):
            value = getattr(self, name)
            if value is not None and not 0 <= value <= 1:
                raise ValueError(f"{name} must be between 0 and 1")
        for name in ("saturation_value", "dark_value"):
            value = getattr(self, name)
            if value is not None and not 0 <= value <= 255:
                raise ValueError(f"{name} must be between 0 and 255")
        if (self.depth_min_mm is not None and self.depth_max_mm is not None
                and self.depth_min_mm >= self.depth_max_mm):
            raise ValueError("depth_min_mm must be lower than depth_max_mm")

    def execution_blockers(self, selection_policy: str = "best_quality_score") -> tuple[str, ...]:
        blockers: list[str] = []
        if selection_policy == "best_quality_score" and not self.score_weights:
            blockers.append("quality score/weights")
        if selection_policy in {"highest_passing_readiness", "best_surface_coverage"}:
            for name in ("depth_min_mm", "depth_max_mm", "min_depth_valid_ratio"):
                if getattr(self, name) is None:
                    blockers.append(name)
        return tuple(blockers)

    def require_execution_ready(self, selection_policy: str = "best_quality_score") -> None:
        blockers = self.execution_blockers(selection_policy)
        if blockers:
            raise ValueError("quality config is not execution-ready; unresolved: " + ", ".join(blockers))

    @classmethod
    def from_json(cls, path: str | Path) -> "SensorQualityConfig":
        with Path(path).expanduser().open("r", encoding="utf-8") as handle:
            result = cls(**json.load(handle))
        result.validate()
        return result


@dataclass(frozen=True)
class HardwareZSearchConfig:
    candidates: tuple[float, ...]
    z_max: float
    stable_timeout_s: float
    selection_policy: str = "best_quality_score"
    search_mode: str = "explicit"
    z_start: float | None = None
    coarse_step: float | None = None
    fine_step: float | None = None
    surface_area_weight: float = 0.6
    depth_valid_weight: float = 0.4
    stop_after_first_post_pass_failure: bool = True
    search_min_z_cm: float | None = None

    def validate(self) -> None:
        values = (
            *self.candidates, self.z_max, self.stable_timeout_s,
            self.surface_area_weight, self.depth_valid_weight,
        )
        if self.search_mode == "adaptive":
            if self.z_start is None or self.coarse_step is None or self.fine_step is None:
                raise ValueError("adaptive search requires z_start, coarse_step, and fine_step")
            values += (self.z_start, self.coarse_step, self.fine_step)
            if self.z_start > self.z_max or self.coarse_step <= 0 or self.fine_step <= 0:
                raise ValueError("adaptive Z bounds and steps are invalid")
            if self.search_min_z_cm is not None:
                values += (self.search_min_z_cm,)
                if self.search_min_z_cm > self.z_start:
                    raise ValueError("adaptive search minimum must not exceed start Z")
        elif not self.candidates:
            raise ValueError("at least one explicit Z candidate is required")
        elif self.search_mode != "explicit":
            raise ValueError(f"unsupported Automatic Z search mode: {self.search_mode}")
        if not all(math.isfinite(value) for value in values):
            raise ValueError("Z candidates, z_max, and timeout must be finite")
        if self.stable_timeout_s <= 0:
            raise ValueError("stable timeout must be positive")
        if any(value > self.z_max for value in self.candidates):
            raise ValueError("Z candidate exceeds the user-provided z_max")
        if self.search_mode == "adaptive" and self.candidates:
            raise ValueError("adaptive search cannot also define explicit Z candidates")
        if len(set(self.candidates)) != len(self.candidates):
            raise ValueError("Z candidates must be unique")
        if any(right <= left for left, right in zip(self.candidates, self.candidates[1:])):
            raise ValueError("Z candidates must be strictly ascending")
        if self.selection_policy not in {
            "highest_passing_readiness", "best_quality_score", "best_surface_coverage",
        }:
            raise ValueError(f"unsupported Automatic Z selection policy: {self.selection_policy}")
        if self.surface_area_weight < 0 or self.depth_valid_weight < 0:
            raise ValueError("surface coverage score weights must be non-negative")
        if self.surface_area_weight + self.depth_valid_weight <= 0:
            raise ValueError("at least one surface coverage score weight must be positive")


@dataclass(frozen=True)
class HardwareZCandidateResult:
    z_command: float
    roll: float
    pitch: float
    rgb_path: str | None
    depth_path: str | None
    depth_median: float | None
    depth_valid_ratio: float
    roi_depth_coverage: float
    invalid_ratio: float
    saturation_ratio: float | None
    dark_ratio: float | None
    sharpness: float
    contrast: float
    edge_occupancy_ratio: float
    quality_score: float | None
    accepted: bool
    rejection_reason: tuple[str, ...] = field(default_factory=tuple)
    readiness_pass: bool | None = None
    readiness_frames: int = 1
    plane_inlier_ratio: float | None = None
    plane_residual: float | None = None
    surface_patch_count: int | None = None
    object_area_px: int | None = None
    surface_area_px: int | None = None
    surface_ratio: float | None = None
    usable_patch_count: int | None = None
    fov_edge_contact: bool | None = None
    diagnostic_dir: str | None = None
    quality_score_components: dict[str, float] | None = None
    quality_score_formula: str | None = None
    depth_p05_mm: float | None = None
    depth_p95_mm: float | None = None
    board_roi_depth_valid_ratio: float | None = None
    requested_roll_deg: float | None = None
    requested_pitch_deg: float | None = None
    applied_roll_deg: float | None = None
    applied_pitch_deg: float | None = None
    combined_tilt_deg: float | None = None
    max_combined_tilt_deg: float | None = None
    tilt_scale: float | None = None

    def diagnostic_dict(self) -> dict[str, Any]:
        """Serialize diagnostics with an explicit millimetre median alias."""
        payload = asdict(self)
        payload["depth_median_mm"] = self.depth_median
        return payload

    def selection_sample(self) -> InspectionQualitySample:
        return InspectionQualitySample(
            z_cm=self.z_command, depth_valid_ratio=self.depth_valid_ratio,
            plane_inlier_ratio=float(self.plane_inlier_ratio or 0.0),
            plane_residual_mm=float(self.plane_residual or 0.0),
            object_area_px=int(self.object_area_px or 0),
            surface_area_px=int(self.surface_area_px or 0),
            valid_patch_count=int(self.usable_patch_count or 0),
            touches_fov_edge=self.edge_occupancy_ratio > 0,
            rgb_saturated_ratio=self.saturation_ratio, rgb_sharpness=self.sharpness,
            gate_passed=self.accepted, quality_score=self.quality_score,
            reasons=self.rejection_reason,
        )


@dataclass(frozen=True)
class HardwareZSearchResult:
    success: bool
    candidates: tuple[HardwareZCandidateResult, ...]
    best_z: float | None
    best_rgb: str | None
    best_depth: str | None
    best_metrics: dict[str, Any] | None
    failure_reason: str | None = None
    selection_policy: str = "best_quality_score"
    stop_reason: str | None = None
    search_mode: str = "explicit"
    coarse_step: float | None = None
    fine_step: float | None = None
    coarse_candidates: tuple[float, ...] = field(default_factory=tuple)
    fine_candidates: tuple[float, ...] = field(default_factory=tuple)
    last_pass_z: float | None = None
    first_fail_z: float | None = None
    selected_best_z_quality_score: float | None = None
    quality_score_formula: str | None = None
    quality_score_weights: dict[str, float] | None = None

    def save(self, path: str | Path) -> None:
        target = Path(path).expanduser()
        target.parent.mkdir(parents=True, exist_ok=True)
        payload = asdict(self)
        payload["candidates"] = [item.diagnostic_dict() for item in self.candidates]
        if payload["best_metrics"] is not None:
            payload["best_metrics"]["depth_median_mm"] = payload["best_metrics"].get(
                "depth_median"
            )
        if target.suffix.lower() == ".json":
            target.write_text(json.dumps(payload, indent=2), encoding="utf-8")
            return
        if target.suffix.lower() == ".csv":
            rows = [item.diagnostic_dict() for item in self.candidates]
            with target.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(rows[0]) if rows else ["z_command"])
                writer.writeheader(); writer.writerows(rows)
            return
        raise ValueError("result path must end in .json or .csv")


class ZMover(Protocol):
    def move_and_wait(self, z_cm: float, timeout_s: float) -> Any: ...


class SensorQualityEvaluator:
    def __init__(self, config: SensorQualityConfig) -> None:
        config.validate()
        self.config = config

    def evaluate(self, frame: RGBDepthFrame, *, z_command: float, roll: float, pitch: float,
                 rgb_path: str | None = None, depth_path: str | None = None) -> HardwareZCandidateResult:
        color = np.asarray(frame.color_bgr)
        depth = np.asarray(frame.depth_mm, dtype=np.float32)
        if color.ndim != 3 or color.shape[:2] != depth.shape:
            raise ValueError("aligned RGB and depth shapes are required")
        valid = np.isfinite(depth) & (depth > 0)
        if self.config.depth_min_mm is not None:
            valid &= depth >= self.config.depth_min_mm
        if self.config.depth_max_mm is not None:
            valid &= depth <= self.config.depth_max_mm
        total = max(1, depth.size)
        valid_ratio = float(valid.sum() / total)
        invalid_ratio = 1.0 - valid_ratio
        median = float(np.median(depth[valid])) if valid.any() else None
        depth_p05 = float(np.percentile(depth[valid], 5)) if valid.any() else None
        depth_p95 = float(np.percentile(depth[valid], 95)) if valid.any() else None
        roi_coverage = valid_ratio
        import cv2
        gray = cv2.cvtColor(color, cv2.COLOR_BGR2GRAY)
        saturation = (float((color.max(axis=2) >= self.config.saturation_value).mean())
                      if self.config.saturation_value is not None else None)
        dark = (float((gray <= self.config.dark_value).mean())
                if self.config.dark_value is not None else None)
        contrast = float(gray.std())
        sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        edge_pixels = np.concatenate((valid[0], valid[-1], valid[:, 0], valid[:, -1]))
        edge_occupancy = float(edge_pixels.mean())
        reasons: list[str] = []
        checks = (
            (self.config.min_depth_valid_ratio is not None
             and valid_ratio < self.config.min_depth_valid_ratio, "depth_valid_ratio"),
            (self.config.min_roi_depth_coverage is not None
             and roi_coverage < self.config.min_roi_depth_coverage, "roi_depth_coverage"),
            (self.config.max_invalid_ratio is not None
             and invalid_ratio > self.config.max_invalid_ratio, "invalid_depth"),
            (self.config.max_saturation_ratio is not None and saturation is not None
             and saturation > self.config.max_saturation_ratio, "rgb_saturation"),
            (self.config.max_dark_ratio is not None and dark is not None
             and dark > self.config.max_dark_ratio, "rgb_dark"),
            (self.config.min_sharpness is not None
             and sharpness < self.config.min_sharpness, "rgb_sharpness"),
            (self.config.min_contrast is not None
             and contrast < self.config.min_contrast, "rgb_contrast"),
            (self.config.max_edge_occupancy_ratio is not None
             and edge_occupancy > self.config.max_edge_occupancy_ratio, "roi_clipping"),
        )
        reasons.extend(reason for rejected, reason in checks if rejected)
        metrics: dict[str, float | None] = {
            "depth_valid_ratio": valid_ratio, "roi_depth_coverage": roi_coverage,
            "invalid_ratio": invalid_ratio, "saturation_ratio": saturation,
            "dark_ratio": dark, "sharpness": sharpness, "contrast": contrast,
            "edge_occupancy_ratio": edge_occupancy,
        }
        score = (sum(float(weight) * float(metrics[name])
                     for name, weight in self.config.score_weights.items()
                     if name in metrics and metrics[name] is not None)
                 if self.config.score_weights else None)
        accepted = not reasons
        return HardwareZCandidateResult(
            z_command, roll, pitch, rgb_path, depth_path, median, valid_ratio,
            roi_coverage, invalid_ratio, saturation, dark, sharpness, contrast,
            edge_occupancy, score if accepted else None, accepted, tuple(reasons),
            depth_p05_mm=depth_p05, depth_p95_mm=depth_p95,
        )


class SurfaceReadinessEvaluator:
    """Adapter for the existing Manual-Z surface readiness pipeline."""

    def __init__(self, sensor_config: SensorQualityConfig,
                 inspection_config: InspectionConfig | None = None) -> None:
        self.sensor_metrics = SensorQualityEvaluator(sensor_config)
        self.config = inspection_config or InspectionConfig.default()

    def _evaluate_frame(self, frame: RGBDepthFrame, depth_mm: np.ndarray, *,
                        z_command: float, roll: float, pitch: float) -> HardwareZCandidateResult:
        cfg = self.config
        geometry = extract_surface_geometry(depth_mm, frame.color_bgr.shape, cfg)
        object_mask = geometry.object_mask
        surface_mask = geometry.surface_mask
        patches = list(geometry.patches)
        depth_valid_ratio = geometry.depth_valid_ratio
        plane_inlier_ratio = geometry.plane_inlier_ratio
        plane_residual = geometry.plane_residual
        print(
            f"[AUTO-Z DEBUG] "
            f"Z={z_command:.1f} "
            f"depth_valid={depth_valid_ratio:.3f} "
            f"plane_inlier={plane_inlier_ratio:.3f} "
            f"plane_residual={plane_residual:.3f} "
            f"object_px={0 if object_mask is None else int(np.count_nonzero(object_mask))} "
            f"surface_px={0 if surface_mask is None else int(np.count_nonzero(surface_mask))} "
            f"patches={len(patches)}"
        )
        # Automatic-Z surface patches are diagnostic, not a size-dependent
        # readiness gate. Final capture uses this same geometry-only contract.
        readiness = evaluate_surface_geometry_readiness(geometry, cfg)
        metric_frame = RGBDepthFrame(frame.color_bgr, depth_mm, frame.timestamp)
        base = self.sensor_metrics.evaluate(
            metric_frame, z_command=z_command, roll=roll, pitch=pitch,
        )
        edge_contact = geometry.fov_edge_contact
        object_area_px = int(readiness.metrics["object_area_px"])
        surface_area_px = int(readiness.metrics["surface_area_px"])

        print(
            f"[AUTO-Z] Z={z_command:.1f} "
            f"depth_valid={depth_valid_ratio:.3f} "
            f"plane_inlier={plane_inlier_ratio:.3f} "
            f"plane_residual={plane_residual:.3f} "
            f"object_px={0 if object_mask is None else int(np.count_nonzero(object_mask))} "
            f"surface_px={0 if surface_mask is None else int(np.count_nonzero(surface_mask))} "
            f"patches={len(patches)} "
            f"ready={readiness.ready} "
            f"reasons={list(readiness.reasons)}"
        )

        return replace(
            base, accepted=readiness.ready, rejection_reason=tuple(readiness.reasons),
            readiness_pass=readiness.ready, plane_inlier_ratio=plane_inlier_ratio,
            plane_residual=plane_residual, surface_patch_count=len(patches),
            object_area_px=object_area_px, surface_area_px=surface_area_px,
            surface_ratio=readiness.metrics["surface_ratio"],
            usable_patch_count=int(readiness.metrics["usable_patch_count"]),
            fov_edge_contact=edge_contact, depth_valid_ratio=depth_valid_ratio,
            quality_score=None,
        )

    def evaluate_candidate(self, camera: Any, *, z_command: float, roll: float, pitch: float,
                           candidate_index: int,
                           artifact_store: "CandidateArtifactStore | None") -> HardwareZCandidateResult:
        required_streak = self.config.quality.ready_streak_frames
        max_attempts = required_streak * 2
        depth_history: list[np.ndarray] = []
        streak = 0
        records: list[tuple[HardwareZCandidateResult, RGBDepthFrame, np.ndarray]] = []
        final_pass_streak: list[tuple[HardwareZCandidateResult, RGBDepthFrame, np.ndarray]] = []
        for _ in range(max_attempts):
            last_frame = camera.capture()
            depth_history.append(np.asarray(last_frame.depth_mm, dtype=np.float32))
            depth_history = depth_history[-self.config.depth.median_frames:]
            stack = np.stack(depth_history).astype(np.float32)
            stack[stack <= 0] = np.nan
            with warnings.catch_warnings(), np.errstate(invalid="ignore"):
                warnings.filterwarnings(
                    "ignore", message="All-NaN slice encountered", category=RuntimeWarning,
                )
                median_depth = np.nanmedian(stack, axis=0)
            median_depth = np.nan_to_num(median_depth, nan=0.0).astype(np.float32)
            last = self._evaluate_frame(
                last_frame, median_depth, z_command=z_command, roll=roll, pitch=pitch,
            )
            record = (last, last_frame, median_depth.copy())
            records.append(record)
            if last.readiness_pass:
                streak += 1
                final_pass_streak.append(record)
            else:
                streak = 0
                final_pass_streak = []
            if streak >= required_streak:
                break
        assert records
        passed = streak >= required_streak
        def residual_rank(item: tuple[HardwareZCandidateResult, RGBDepthFrame, np.ndarray]) -> float:
            value = item[0].plane_residual
            return -float(value) if value is not None and math.isfinite(value) else -math.inf

        if passed:
            representative = max(
                final_pass_streak,
                key=lambda item: (
                    int(item[0].surface_area_px or 0), item[0].depth_valid_ratio,
                    float(item[0].plane_inlier_ratio or 0.0),
                    residual_rank(item),
                ),
            )
        else:
            representative = max(
                records,
                key=lambda item: (
                    item[0].depth_valid_ratio, int(item[0].surface_area_px or 0),
                    int(item[0].object_area_px or 0),
                    float(item[0].plane_inlier_ratio or 0.0),
                    residual_rank(item),
                ),
            )
        last, last_frame, representative_depth = representative
        reasons = last.rejection_reason if not passed else ()
        if not passed and not reasons:
            reasons = (f"readiness_streak={streak}/{required_streak}",)
        rgb_path = depth_path = diagnostic_dir = None
        if artifact_store is not None:
            geometry = extract_surface_geometry(
                representative_depth, last_frame.color_bgr.shape, self.config,
            )
            rgb_path, depth_path, diagnostic_dir = artifact_store.save_geometry(
                z_command, last_frame, representative_depth, geometry,
                replace(
                    last, accepted=passed, readiness_pass=passed,
                    readiness_frames=len(records), rejection_reason=reasons,
                ),
            )
        return replace(
            last, accepted=passed, readiness_pass=passed,
            readiness_frames=len(records), rejection_reason=reasons,
            rgb_path=rgb_path, depth_path=depth_path, quality_score=None,
            diagnostic_dir=diagnostic_dir,
        )


class CandidateArtifactStore:
    def __init__(self, root: str | Path) -> None:
        self.root = Path(root).expanduser()
        self.root.mkdir(parents=True, exist_ok=True)

    def save(self, index: int, frame: RGBDepthFrame) -> tuple[str, str]:
        import cv2
        rgb = self.root / f"candidate_{index:03d}_rgb.png"
        depth = self.root / f"candidate_{index:03d}_depth.npy"
        if not cv2.imwrite(str(rgb), frame.color_bgr):
            raise RuntimeError(f"failed to save RGB frame: {rgb}")
        np.save(depth, frame.depth_mm)
        return str(rgb), str(depth)

    @staticmethod
    def _z_directory_name(z_cm: float) -> str:
        value = f"{float(z_cm):g}".replace("-", "minus_").replace(".", "_")
        return f"z_{value}"

    def save_geometry(
        self, z_cm: float, frame: RGBDepthFrame, median_depth: np.ndarray,
        geometry: Any, result: HardwareZCandidateResult,
    ) -> tuple[str, str, str]:
        import cv2

        root = self.root / self._z_directory_name(z_cm)
        root.mkdir(parents=True, exist_ok=True)
        rgb_path = root / "representative_rgb.png"
        depth_path = root / "representative_depth.npy"
        object_path = root / "object_mask.png"
        surface_path = root / "surface_mask.png"
        overlay_path = root / "surface_overlay.png"
        color = np.asarray(frame.color_bgr)
        shape = np.asarray(median_depth).shape
        object_mask = (
            np.zeros(shape, dtype=np.uint8) if geometry.object_mask is None
            else np.where(np.asarray(geometry.object_mask) > 0, 255, 0).astype(np.uint8)
        )
        surface_mask = (
            np.zeros(shape, dtype=np.uint8) if geometry.surface_mask is None
            else np.where(np.asarray(geometry.surface_mask) > 0, 255, 0).astype(np.uint8)
        )
        if not cv2.imwrite(str(rgb_path), color):
            raise RuntimeError(f"failed to save representative RGB: {rgb_path}")
        np.save(depth_path, np.asarray(median_depth, dtype=np.float32))
        for path, mask in ((object_path, object_mask), (surface_path, surface_mask)):
            if not cv2.imwrite(str(path), mask):
                raise RuntimeError(f"failed to save Automatic Z mask: {path}")
        overlay = color.copy()
        tint = np.zeros_like(overlay)
        tint[..., 1] = surface_mask
        overlay = np.where(
            (surface_mask > 0)[..., None],
            cv2.addWeighted(overlay, 0.65, tint, 0.35, 0.0), overlay,
        )
        if not cv2.imwrite(str(overlay_path), overlay):
            raise RuntimeError(f"failed to save Automatic Z overlay: {overlay_path}")
        self._write_diagnostics(root, result)
        return str(rgb_path), str(depth_path), str(root)

    @staticmethod
    def _write_diagnostics(root: Path, result: HardwareZCandidateResult) -> None:
        payload = {
            "z_cm": result.z_command,
            "readiness_pass": result.readiness_pass,
            "depth_valid_ratio": result.depth_valid_ratio,
            "plane_inlier_ratio": result.plane_inlier_ratio,
            "plane_residual": result.plane_residual,
            "object_area_px": result.object_area_px,
            "surface_area_px": result.surface_area_px,
            "surface_ratio": result.surface_ratio,
            "usable_patch_count": result.usable_patch_count,
            "fov_edge_contact": result.fov_edge_contact,
            "readiness_reasons": list(result.rejection_reason),
            "quality_score": result.quality_score,
            "quality_score_components": result.quality_score_components,
            "quality_score_formula": result.quality_score_formula,
            "depth_p05_mm": result.depth_p05_mm,
            "depth_median_mm": result.depth_median,
            "depth_p95_mm": result.depth_p95_mm,
            "board_roi_depth_valid_ratio": result.board_roi_depth_valid_ratio,
        }
        (root / "diagnostics.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8",
        )

    def update_diagnostics(self, result: HardwareZCandidateResult) -> None:
        if result.diagnostic_dir is not None:
            self._write_diagnostics(Path(result.diagnostic_dir), result)


class HardwareAutomaticZSearch:
    def __init__(self, *, platform: ZMover, camera: Any, projector: Any,
                 evaluator: SensorQualityEvaluator, config: HardwareZSearchConfig,
                 artifact_store: CandidateArtifactStore | None = None,
                 pose_for_z: Callable[[float], AdaptivePose] | None = None,
                 before_z: Callable[[AdaptivePose, AdaptivePose | None], None] | None = None) -> None:
        config.validate()
        self.platform, self.camera, self.projector = platform, camera, projector
        self.evaluator, self.config, self.artifact_store = evaluator, config, artifact_store
        self.pose_for_z = pose_for_z
        self.before_z = before_z

    def run(self, *, pose_id: str, roll: float, pitch: float) -> HardwareZSearchResult:
        self.projector.show_black()
        results: list[HardwareZCandidateResult] = []
        coarse_candidates = self._coarse_candidates()
        fine_candidates: list[float] = []
        candidates = coarse_candidates
        first_fail_z: float | None = None
        last_pass_z: float | None = None
        current_pose: AdaptivePose | None = None
        for index, z_cm in enumerate(candidates):
            if z_cm > self.config.z_max:  # defensive check even after config validation
                raise ValueError("candidate traversal attempted to exceed z_max")
            target_pose = self.pose_for_z(z_cm) if self.pose_for_z is not None else None
            if target_pose is not None and self.before_z is not None:
                self.before_z(target_pose, current_pose)
            self.projector.show_black()
            if self.projector.state is not ProjectorState.BLACK:
                raise RuntimeError("projector must be BLACK before Automatic Z motion")
            self.platform.move_and_wait(z_cm, self.config.stable_timeout_s)
            self.projector.show_black()
            if self.projector.state is not ProjectorState.BLACK:
                raise RuntimeError("projector must be BLACK before RGB/depth capture")
            if hasattr(self.evaluator, "evaluate_candidate"):
                evaluated = self.evaluator.evaluate_candidate(
                    self.camera, z_command=z_cm,
                    roll=(target_pose.applied_roll_deg if target_pose is not None else roll),
                    pitch=(target_pose.applied_pitch_deg if target_pose is not None else pitch),
                    candidate_index=index, artifact_store=self.artifact_store,
                )
            else:
                frame = self.camera.capture()
                rgb_path = depth_path = None
                if self.artifact_store is not None:
                    rgb_path, depth_path = self.artifact_store.save(index, frame)
                evaluated = self.evaluator.evaluate(
                    frame, z_command=z_cm,
                    roll=(target_pose.applied_roll_deg if target_pose is not None else roll),
                    pitch=(target_pose.applied_pitch_deg if target_pose is not None else pitch),
                    rgb_path=rgb_path, depth_path=depth_path,
                )
            results.append(evaluated)
            if target_pose is not None:
                evaluated = replace(
                    evaluated,
                    requested_roll_deg=target_pose.requested_roll_deg,
                    requested_pitch_deg=target_pose.requested_pitch_deg,
                    applied_roll_deg=target_pose.applied_roll_deg,
                    applied_pitch_deg=target_pose.applied_pitch_deg,
                    combined_tilt_deg=target_pose.combined_tilt_deg,
                    max_combined_tilt_deg=target_pose.max_combined_tilt_deg,
                    tilt_scale=target_pose.tilt_scale,
                )
                results[-1] = evaluated
                current_pose = target_pose
                current_z = z_cm
            if self.config.selection_policy != "best_surface_coverage":
                print(
                    "[AUTO-Z CANDIDATE] "
                    f"Z={z_cm:g} {'PASS' if evaluated.accepted else 'FAIL'} "
                    f"depth_valid={evaluated.depth_valid_ratio:.3f} "
                    f"surface_px={int(evaluated.surface_area_px or 0)} "
                    f"quality_score={evaluated.quality_score}"
                )
            if not evaluated.accepted and (
                self.config.selection_policy == "highest_passing_readiness"
                or (
                    self.config.selection_policy == "best_surface_coverage"
                    and last_pass_z is not None
                    and self.config.stop_after_first_post_pass_failure
                )
            ):
                first_fail_z = z_cm
                break
            if evaluated.accepted:
                last_pass_z = z_cm

        if (self.config.search_mode == "adaptive"
                and self.config.selection_policy == "highest_passing_readiness"
                and first_fail_z is not None and last_pass_z is not None):
            fine_candidates = self._fine_candidates(last_pass_z, first_fail_z)
            for offset, z_cm in enumerate(fine_candidates, start=len(results)):
                target_pose = self.pose_for_z(z_cm) if self.pose_for_z is not None else None
                if target_pose is not None and self.before_z is not None:
                    self.before_z(target_pose, current_pose)
                evaluated = self._evaluate_candidate(
                    z_cm, offset, pose_id=pose_id, roll=roll, pitch=pitch,
                )
                results.append(evaluated)
                if target_pose is not None:
                    evaluated = replace(
                        evaluated,
                        requested_roll_deg=target_pose.requested_roll_deg,
                        requested_pitch_deg=target_pose.requested_pitch_deg,
                        applied_roll_deg=target_pose.applied_roll_deg,
                        applied_pitch_deg=target_pose.applied_pitch_deg,
                        combined_tilt_deg=target_pose.combined_tilt_deg,
                        max_combined_tilt_deg=target_pose.max_combined_tilt_deg,
                        tilt_scale=target_pose.tilt_scale,
                    )
                    results[-1] = evaluated
                    current_pose = target_pose
                if not evaluated.accepted:
                    first_fail_z = z_cm
                    break
                last_pass_z = z_cm

        score_formula = None
        score_weights = None
        if self.config.selection_policy == "best_surface_coverage":
            results = self._apply_surface_coverage_scores(results)
            score_formula = (
                "surface_area_weight*(surface_area_px/max_pass_surface_area_px) + "
                "depth_valid_weight*(depth_valid_ratio/max_pass_depth_valid_ratio), "
                "divided by weight sum; ties select lower Z"
            )
            score_weights = {
                "surface_area_weight": self.config.surface_area_weight,
                "depth_valid_weight": self.config.depth_valid_weight,
            }
            if self.artifact_store is not None:
                for item in results:
                    self.artifact_store.update_diagnostics(item)
            for item in results:
                print(
                    "[AUTO-Z CANDIDATE] "
                    f"Z={item.z_command:g} {'PASS' if item.accepted else 'FAIL'} "
                    f"depth_valid={item.depth_valid_ratio:.3f} "
                    f"surface_px={int(item.surface_area_px or 0)} "
                    f"quality_score={item.quality_score}"
                )

        if self.config.selection_policy == "highest_passing_readiness":
            passing = [item for item in results if item.accepted]
            if not passing:
                return HardwareZSearchResult(
                    success=False, candidates=tuple(results), best_z=None,
                    best_rgb=None, best_depth=None, best_metrics=None,
                    failure_reason="NoValidInspectionZ",
                    selection_policy=self.config.selection_policy,
                    stop_reason="first_candidate_failed_readiness",
                    search_mode=self.config.search_mode,
                    coarse_step=self.config.coarse_step,
                    fine_step=self.config.fine_step,
                    coarse_candidates=tuple(coarse_candidates),
                    fine_candidates=tuple(fine_candidates),
                    first_fail_z=first_fail_z,
                )
            best = passing[-1]
            stop_reason = (
                "fine_candidate_failed_readiness"
                if self.config.search_mode == "adaptive" and fine_candidates and not results[-1].accepted
                else "next_candidate_failed_readiness"
                if not results[-1].accepted else "highest_candidate_passed"
            )
        else:
            selected = AutomaticZSearch.select_best(
                pose_id=pose_id, samples=[item.selection_sample() for item in results],
            )
            if not selected.success or selected.best_z_cm is None:
                return HardwareZSearchResult(
                    success=False, candidates=tuple(results), best_z=None,
                    best_rgb=None, best_depth=None, best_metrics=None,
                    failure_reason="NoValidInspectionZ",
                    selection_policy=self.config.selection_policy,
                    stop_reason="no_passing_quality_score",
                )
            best = next(item for item in results if item.z_command == selected.best_z_cm)
            stop_reason = self.config.selection_policy
        print(
            "[AUTO-Z SELECT] "
            f"policy={self.config.selection_policy} selected_z={best.z_command:g} "
            f"score={'N/A' if best.quality_score is None else f'{best.quality_score:.6f}'}"
        )
        if results[-1].z_command != best.z_command:
            if self.pose_for_z is not None and self.before_z is not None:
                target_pose = self.pose_for_z(best.z_command)
                self.before_z(target_pose, current_pose)
            self.platform.move_and_wait(best.z_command, self.config.stable_timeout_s)
        return HardwareZSearchResult(
            success=True, candidates=tuple(results), best_z=best.z_command,
            best_rgb=best.rgb_path, best_depth=best.depth_path,
            best_metrics=best.diagnostic_dict(), failure_reason=None,
            selection_policy=self.config.selection_policy, stop_reason=stop_reason,
            search_mode=self.config.search_mode,
            coarse_step=self.config.coarse_step, fine_step=self.config.fine_step,
            coarse_candidates=tuple(coarse_candidates), fine_candidates=tuple(fine_candidates),
            last_pass_z=best.z_command, first_fail_z=first_fail_z,
            selected_best_z_quality_score=best.quality_score,
            quality_score_formula=score_formula, quality_score_weights=score_weights,
        )

    def _apply_surface_coverage_scores(
        self, results: list[HardwareZCandidateResult],
    ) -> list[HardwareZCandidateResult]:
        passing = [item for item in results if item.accepted]
        if not passing:
            return results
        max_surface = max(float(item.surface_area_px or 0) for item in passing)
        max_depth = max(float(item.depth_valid_ratio) for item in passing)
        weight_sum = self.config.surface_area_weight + self.config.depth_valid_weight
        formula = "normalized_surface_area_and_depth_valid_ratio"
        scored: list[HardwareZCandidateResult] = []
        for item in results:
            if not item.accepted:
                scored.append(replace(item, quality_score=None, quality_score_formula=formula))
                continue
            surface_normalized = (
                float(item.surface_area_px or 0) / max_surface if max_surface > 0 else 0.0
            )
            depth_normalized = item.depth_valid_ratio / max_depth if max_depth > 0 else 0.0
            score = (
                self.config.surface_area_weight * surface_normalized
                + self.config.depth_valid_weight * depth_normalized
            ) / weight_sum
            scored.append(replace(
                item, quality_score=float(score),
                quality_score_components={
                    "surface_area_normalized": float(surface_normalized),
                    "depth_valid_normalized": float(depth_normalized),
                },
                quality_score_formula=formula,
            ))
        return scored

    def _coarse_candidates(self) -> list[float]:
        if self.config.search_mode == "explicit":
            return list(self.config.candidates)
        assert self.config.z_start is not None and self.config.coarse_step is not None
        values: list[float] = []
        current = self.config.z_start
        if self.config.search_min_z_cm is not None:
            while current >= self.config.search_min_z_cm - 1e-9:
                values.append(round(current, 10))
                current -= self.config.coarse_step
            if not values or values[-1] != self.config.search_min_z_cm:
                values.append(self.config.search_min_z_cm)
        else:
            while current <= self.config.z_max + 1e-9:
                values.append(round(current, 10))
                current += self.config.coarse_step
            if not values or values[-1] != self.config.z_max:
                values.append(self.config.z_max)
        return values

    def _fine_candidates(self, last_pass: float, first_fail: float) -> list[float]:
        assert self.config.fine_step is not None
        values: list[float] = []
        current = last_pass + self.config.fine_step
        while current < first_fail - 1e-9 and current <= self.config.z_max + 1e-9:
            values.append(round(current, 10))
            current += self.config.fine_step
        return values

    def _evaluate_candidate(self, z_cm: float, index: int, *, pose_id: str,
                            roll: float, pitch: float) -> HardwareZCandidateResult:
        self.projector.show_black()
        if self.projector.state is not ProjectorState.BLACK:
            raise RuntimeError("projector must be BLACK before Automatic Z motion")
        self.platform.move_and_wait(z_cm, self.config.stable_timeout_s)
        self.projector.show_black()
        if self.projector.state is not ProjectorState.BLACK:
            raise RuntimeError("projector must be BLACK before RGB/depth capture")
        if hasattr(self.evaluator, "evaluate_candidate"):
            return self.evaluator.evaluate_candidate(
                self.camera, z_command=z_cm, roll=roll, pitch=pitch,
                candidate_index=index, artifact_store=self.artifact_store,
            )
        frame = self.camera.capture()
        rgb_path = depth_path = None
        if self.artifact_store is not None:
            rgb_path, depth_path = self.artifact_store.save(index, frame)
        return self.evaluator.evaluate(
            frame, z_command=z_cm, roll=roll, pitch=pitch,
            rgb_path=rgb_path, depth_path=depth_path,
        )
