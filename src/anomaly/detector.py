from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from src.config import InspectionConfig
from src.core.surface_geometry import SurfaceGeometryResult, extract_surface_geometry
from src.inspection.surface_inspector import SurfaceInspectionResult
from src.integration.inspection_failures import AnomalyInputDataError


@dataclass(frozen=True)
class AnomalyResult:
    status: str
    is_mock: bool
    score: float | None = None
    threshold: float | None = None
    classification: str | None = None
    judgement: str | None = None
    heatmap_path: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class AnomalyDetector(Protocol):
    def inspect(self, surface: SurfaceInspectionResult) -> AnomalyResult: ...


class MockAnomalyDetector:
    def inspect(self, surface: SurfaceInspectionResult) -> AnomalyResult:
        return AnomalyResult(
            status="MOCK_NORMAL",
            is_mock=True,
            metadata={"source": "mock", "pose_id": surface.pose_id},
        )


class AnomalyModelNotReadyError(RuntimeError):
    pass


@dataclass(frozen=True)
class ProductionAnomalyConfig:
    checkpoint_path: Path
    validation_manifest_path: Path
    threshold_percentile: float = 99.0
    score_method: str = "mean_mse"
    batch_size: int = 32
    surface_patch_coverage: float = 1.0


class ProductionAnomalyDetector:
    """In-process adapter over the existing ``infer_anomaly.py`` model API."""

    def __init__(self, config: ProductionAnomalyConfig,
                 inspection_config: InspectionConfig | None = None) -> None:
        self.config = config
        self.inspection_config = inspection_config or InspectionConfig.default()
        self._runtime: dict[str, Any] | None = None

    def validate_ready(self) -> None:
        missing = [
            str(path) for path in (
                self.config.checkpoint_path, self.config.validation_manifest_path,
            ) if not Path(path).expanduser().is_file()
        ]
        if missing:
            raise AnomalyModelNotReadyError(
                "ANOMALY_MODEL_NOT_READY: missing artifact(s): " + ", ".join(missing)
            )
        if not 0 <= self.config.threshold_percentile <= 100:
            raise AnomalyModelNotReadyError("ANOMALY_MODEL_NOT_READY: invalid threshold percentile")
        if self.config.batch_size <= 0:
            raise AnomalyModelNotReadyError("ANOMALY_MODEL_NOT_READY: invalid batch size")
        if self.config.surface_patch_coverage != 1.0:
            raise AnomalyModelNotReadyError(
                "ANOMALY_MODEL_NOT_READY: production surface_patch_coverage must be 1.0"
            )

    def _load(self) -> dict[str, Any]:
        if self._runtime is not None:
            return self._runtime
        self.validate_ready()
        try:
            import torch
            from src.infer_anomaly import (
                SCORE_METHODS, inspect_image, load_bgr_image, load_model,
                read_validation_entries, validation_region_positions,
            )
            if self.config.score_method not in SCORE_METHODS:
                raise ValueError(f"unsupported score method: {self.config.score_method}")
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            checkpoint = torch.load(
                self.config.checkpoint_path, map_location=device, weights_only=False,
            )
            model = load_model(checkpoint, device)
            model_config = checkpoint["config"]
            patch_size = int(checkpoint.get("patch_size", model_config["patch_size"]))
            stride = int(checkpoint.get("stride", model_config["stride"]))
            preprocessing = {
                "gamma": float(checkpoint.get("gamma", model_config.get("gamma", 0.82))),
                "clahe_clip": float(checkpoint.get("clahe_clip", model_config.get("clahe_clip", 1.5))),
                "unsharp_amount": float(checkpoint.get("unsharp_amount", model_config.get("unsharp_amount", 0.30))),
            }
            validation_values: list[float] = []
            validation_region_patch_count = 0
            validation_full_image_count = 0
            for entry in read_validation_entries(self.config.validation_manifest_path):
                image_path = entry["path"]
                assert image_path is not None
                image = load_bgr_image(image_path)
                allowed_positions = validation_region_positions(
                    entry, image.shape, patch_size, stride,
                )
                if allowed_positions is None:
                    validation_full_image_count += 1
                else:
                    validation_region_patch_count += len(allowed_positions)
                scores, *_ = inspect_image(
                    model, image, patch_size, stride,
                    self.config.batch_size, device, preprocessing,
                    allowed_positions=allowed_positions,
                )
                validation_values.extend(scores[self.config.score_method].tolist())
            if not validation_values:
                raise ValueError("normal validation scores are empty")
            validation = np.asarray(validation_values, dtype=np.float32)
            self._runtime = {
                "model": model, "device": device, "patch_size": patch_size,
                "stride": stride, "preprocessing": preprocessing,
                "validation": validation,
                "threshold": float(np.percentile(validation, self.config.threshold_percentile)),
                "inspect_image": inspect_image,
                "validation_region_patch_count": validation_region_patch_count,
                "validation_full_image_count": validation_full_image_count,
            }
            return self._runtime
        except AnomalyModelNotReadyError:
            raise
        except Exception as exc:
            raise AnomalyModelNotReadyError(f"ANOMALY_MODEL_NOT_READY: {exc}") from exc

    def prepare(self) -> None:
        """Fully load checkpoint and validation thresholds before hardware opens."""
        self._load()

    def inspect_frame(
        self, frame: Any, *, pose_id: str, output_directory: str | Path,
        rgb_path: str, depth_path: str, ir_path: str | None,
        platform_telemetry: Any,
        surface_geometry: SurfaceGeometryResult | None = None,
        final_capture_metadata: dict[str, Any] | None = None,
        geometry_capture_metadata: dict[str, Any] | None = None,
        inspection_mask: np.ndarray | None = None,
        filled_object_mask: np.ndarray | None = None,
    ) -> AnomalyResult:
        runtime = self._load()
        import cv2
        from src.infer_anomaly import (
            build_anomaly_mask, build_inspected_mask, calculate_positions,
            make_heatmap, make_threshold_overlay, select_patch_positions,
        )
        color = np.asarray(frame.color_bgr)
        if color.ndim != 3 or color.shape[2] != 3 or color.size == 0:
            raise AnomalyInputDataError("final RGB frame is empty or not BGR")
        if color.shape[0] < runtime["patch_size"] or color.shape[1] < runtime["patch_size"]:
            raise AnomalyInputDataError(
                "final RGB frame is smaller than the anomaly model patch size"
            )
        if surface_geometry is None:
            depth = np.asarray(frame.depth_mm, dtype=np.float32)
            if depth.ndim != 2 or depth.shape != color.shape[:2]:
                raise AnomalyInputDataError(
                    f"fresh aligned RGB/depth shape mismatch: RGB={color.shape[:2]}, "
                    f"Depth={depth.shape}"
                )
            try:
                geometry = extract_surface_geometry(
                    depth, color.shape, self.inspection_config,
                    patch_min_coverage=self.config.surface_patch_coverage,
                )
            except ValueError as exc:
                raise AnomalyInputDataError(f"invalid fresh depth geometry: {exc}") from exc
        else:
            geometry = surface_geometry
            for name, frozen_mask in (
                ("object_mask", geometry.object_mask),
                ("surface_mask", geometry.surface_mask),
            ):
                if frozen_mask is not None and np.asarray(frozen_mask).shape != color.shape[:2]:
                    raise AnomalyInputDataError(
                        f"frozen {name} / final RGB shape mismatch: "
                        f"mask={np.asarray(frozen_mask).shape}, RGB={color.shape[:2]}"
                    )
        if geometry.object_area_px == 0 or geometry.object_mask is None:
            raise AnomalyInputDataError(
                "fresh depth object surface was not found "
                f"(depth_valid_ratio={geometry.depth_valid_ratio:.6f}, "
                f"plane_inlier_ratio={geometry.plane_inlier_ratio:.6f}, "
                f"plane_residual={geometry.plane_residual})"
            )
        if geometry.surface_area_px == 0 or geometry.surface_mask is None:
            raise AnomalyInputDataError("fresh depth surface mask is empty")
        roi_mask = geometry.surface_mask
        anomaly_roi_type = "surface_mask"
        if inspection_mask is not None:
            roi_mask = np.asarray(inspection_mask)
            if roi_mask.ndim != 2 or roi_mask.shape != color.shape[:2]:
                raise AnomalyInputDataError(
                    "inspection mask / final RGB shape mismatch: "
                    f"mask={roi_mask.shape}, RGB={color.shape[:2]}"
                )
            roi_mask = np.where(roi_mask > 0, 255, 0).astype(np.uint8)
            roi_area = int(np.count_nonzero(roi_mask))
            if roi_area == 0 or roi_area == roi_mask.size:
                raise AnomalyInputDataError("inspection mask is empty or full-frame")
            anomaly_roi_type = "inspection_mask"
        try:
            selected_positions = select_patch_positions(
                color.shape, runtime["patch_size"], runtime["stride"],
                surface_mask=roi_mask,
                min_surface_coverage=self.config.surface_patch_coverage,
            )
        except ValueError as exc:
            raise AnomalyInputDataError(f"invalid fresh depth geometry: {exc}") from exc
        if not selected_positions:
            raise AnomalyInputDataError("selected_surface_patch_count == 0")
        try:
            scores, positions, _, score_maps, _ = runtime["inspect_image"](
                runtime["model"], color, runtime["patch_size"],
                runtime["stride"], self.config.batch_size, runtime["device"],
                runtime["preprocessing"],
                surface_mask=roi_mask,
                min_surface_coverage=self.config.surface_patch_coverage,
            )
            method_scores = np.asarray(scores[self.config.score_method])
        except (ValueError, IndexError, KeyError) as exc:
            raise AnomalyInputDataError(f"invalid anomaly inference input/data: {exc}") from exc
        if not positions:
            raise AnomalyInputDataError("selected_surface_patch_count == 0")
        if method_scores.size == 0 or not np.all(np.isfinite(method_scores)):
            raise AnomalyInputDataError("anomaly inference produced no finite patch scores")
        threshold = float(runtime["threshold"])
        score = float(np.max(method_scores))
        classification = "DEFECT" if bool(np.any(method_scores > threshold)) else "NORMAL"
        judgement = "NG" if classification == "DEFECT" else "OK"
        output = Path(output_directory).expanduser().resolve()
        output.mkdir(parents=True, exist_ok=True)
        inspected_mask = build_inspected_mask(
            color.shape[:2], positions, runtime["patch_size"], roi_mask,
        )
        heatmap = make_heatmap(
            score_maps[self.config.score_method], runtime["validation"], method_scores,
            inspected_mask,
        )
        heatmap_path = output / "anomaly_heatmap.png"
        mask = build_anomaly_mask(
            color.shape[:2], positions, method_scores, threshold,
            runtime["patch_size"], roi_mask,
        )
        overlay_path = output / "anomaly_overlay.png"
        surface_mask_path = output / "surface_mask.png"
        object_mask_path = output / "object_mask.png"
        surface_patch_overlay_path = output / "surface_patch_overlay.png"
        inspection_mask_path = output / "inspection_mask.png"
        patch_overlay = color.copy()
        if filled_object_mask is not None:
            filled_binary = np.asarray(filled_object_mask)
            if filled_binary.ndim != 2 or filled_binary.shape != color.shape[:2]:
                raise AnomalyInputDataError(
                    "filled object mask / final RGB shape mismatch: "
                    f"mask={filled_binary.shape}, RGB={color.shape[:2]}"
                )
            filled_binary = filled_binary > 0
            magenta = np.zeros_like(patch_overlay)
            magenta[..., 0] = 255
            magenta[..., 2] = 255
            yellow = np.zeros_like(patch_overlay)
            yellow[..., 1] = 255
            yellow[..., 2] = 255
            patch_overlay = np.where(
                filled_binary[..., None],
                cv2.addWeighted(patch_overlay, 0.70, magenta, 0.30, 0),
                patch_overlay,
            )
            patch_overlay = np.where(
                (roi_mask > 0)[..., None],
                cv2.addWeighted(patch_overlay, 0.65, yellow, 0.35, 0),
                patch_overlay,
            )
        for x, y in positions:
            cv2.rectangle(
                patch_overlay, (x, y),
                (x + runtime["patch_size"] - 1, y + runtime["patch_size"] - 1),
                ((255, 255, 0) if filled_object_mask is not None else (0, 255, 0)), 1,
            )
        if not cv2.imwrite(str(heatmap_path), heatmap):
            raise RuntimeError(f"failed to save anomaly heatmap: {heatmap_path}")
        if not cv2.imwrite(str(overlay_path), make_threshold_overlay(frame.color_bgr, mask)):
            raise RuntimeError(f"failed to save anomaly overlay: {overlay_path}")
        for artifact_path, artifact in (
            (surface_mask_path, geometry.surface_mask),
            (object_mask_path, geometry.object_mask),
            (inspection_mask_path, roi_mask),
            (surface_patch_overlay_path, patch_overlay),
        ):
            if not cv2.imwrite(str(artifact_path), artifact):
                raise RuntimeError(f"failed to save anomaly artifact: {artifact_path}")
        total_grid_patch_count = (
            len(calculate_positions(color.shape[0], runtime["patch_size"], runtime["stride"]))
            * len(calculate_positions(color.shape[1], runtime["patch_size"], runtime["stride"]))
        )
        metadata = {
            "pose_id": pose_id, "score_method": self.config.score_method,
            "checkpoint_path": str(Path(self.config.checkpoint_path).resolve()),
            "validation_manifest_path": str(Path(self.config.validation_manifest_path).resolve()),
            "rgb_path": rgb_path, "depth_path": depth_path, "ir_path": ir_path,
            "overlay_path": str(overlay_path),
            "surface_only_inference": True,
            "anomaly_roi_type": anomaly_roi_type,
            "surface_mask_path": str(surface_mask_path),
            "object_mask_path": str(object_mask_path),
            "surface_patch_overlay_path": str(surface_patch_overlay_path),
            "inspection_mask_path": str(inspection_mask_path),
            "surface_patch_coverage_threshold": self.config.surface_patch_coverage,
            "total_grid_patch_count": total_grid_patch_count,
            "selected_surface_patch_count": len(positions),
            "selected_inspection_patch_count": len(positions),
            "inspection_area_px": int(np.count_nonzero(roi_mask)),
            "inspection_to_surface_ratio": (
                float(np.count_nonzero(roi_mask) / geometry.surface_area_px)
                if geometry.surface_area_px > 0 else None
            ),
            "inspection_to_object_ratio": (
                float(np.count_nonzero(roi_mask) / geometry.object_area_px)
                if geometry.object_area_px > 0 else None
            ),
            "object_area_px": geometry.object_area_px,
            "surface_area_px": geometry.surface_area_px,
            "surface_ratio": geometry.surface_ratio,
            "depth_valid_ratio": geometry.depth_valid_ratio,
            "plane_inlier_ratio": geometry.plane_inlier_ratio,
            "plane_residual": geometry.plane_residual,
            "validation_region_patch_count": runtime.get(
                "validation_region_patch_count", 0,
            ),
            "validation_full_image_count": runtime.get(
                "validation_full_image_count", 0,
            ),
            "actual_platform_roll_deg": float(platform_telemetry.roll_deg),
            "actual_platform_pitch_deg": float(platform_telemetry.pitch_deg),
            "actual_platform_z_cm": float(platform_telemetry.z_cm),
        }
        metadata.update(final_capture_metadata or {})
        metadata.update(geometry_capture_metadata or {})
        print(f"[ANOMALY] roi_type={anomaly_roi_type}")
        print(f"[ANOMALY] selected_inspection_patches={len(positions)}")
        print(f"[ANOMALY] score={score}")
        print(f"[ANOMALY] threshold={threshold}")
        print(f"[ANOMALY] classification={classification}")
        print(f"[ANOMALY] judgement={judgement}")
        return AnomalyResult(
            status="ANOMALY_COMPLETE", is_mock=False, score=score,
            threshold=threshold, classification=classification, judgement=judgement,
            heatmap_path=str(heatmap_path),
            metadata=metadata,
        )
