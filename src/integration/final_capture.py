from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np

from src.config import InspectionConfig
from src.core.inspection_mask import InspectionMaskResult
from src.core.surface_geometry import (
    SurfaceGeometryResult,
    evaluate_surface_geometry_readiness,
    extract_surface_geometry,
)
from src.integration.inspection_failures import (
    FinalCaptureQualityError,
    FinalRGBCaptureError,
)


FINAL_CAPTURE_MAX_ATTEMPTS = 8
FINAL_GEOMETRY_MAX_ATTEMPTS = FINAL_CAPTURE_MAX_ATTEMPTS
FINAL_RGB_WARMUP_FRAMES = 3


@dataclass(frozen=True)
class FinalCaptureArtifacts:
    rgb_path: str
    depth_path: str
    ir_path: str | None


@dataclass(frozen=True)
class FinalGeometryArtifacts:
    depth_path: str
    object_mask_path: str
    surface_mask_path: str
    surface_geometry_overlay_path: str | None
    inspection_mask_path: str | None
    inspection_mask_overlay_path: str | None


@dataclass(frozen=True)
class FinalRGBArtifacts:
    rgb_path: str
    ir_path: str | None


@dataclass(frozen=True)
class FinalCaptureAttemptDiagnostic:
    attempt: int
    depth_valid_ratio: float | None
    plane_inlier_ratio: float | None
    plane_residual: float | None
    object_area_px: int
    surface_area_px: int
    ready: bool
    reasons: tuple[str, ...]


@dataclass(frozen=True)
class GeometryReadyFinalCapture:
    frame: Any
    geometry: SurfaceGeometryResult
    attempts: tuple[FinalCaptureAttemptDiagnostic, ...]
    accepted_attempt: int

    def metadata(self) -> dict[str, Any]:
        accepted = self.attempts[self.accepted_attempt - 1]
        metadata = _capture_metadata(self.attempts, accepted)
        metadata["surface_ratio"] = self.geometry.surface_ratio
        return metadata


# The old public name is retained because diagnostic callers already use it,
# but the object now represents the LED-OFF final geometry capture.
FinalGeometryCapture = GeometryReadyFinalCapture


def _finite_or_none(value: float) -> float | None:
    return float(value) if np.isfinite(value) else None


def _format_metric(value: float | None) -> str:
    return "N/A" if value is None else f"{value:.3f}"


def _capture_metadata(
    attempts: tuple[FinalCaptureAttemptDiagnostic, ...] | list[FinalCaptureAttemptDiagnostic],
    accepted: FinalCaptureAttemptDiagnostic | None,
) -> dict[str, Any]:
    reference = accepted if accepted is not None else (attempts[-1] if attempts else None)
    return {
        "geometry_capture_attempts": len(attempts),
        "geometry_accepted_attempt": None if accepted is None else accepted.attempt,
        "depth_valid_ratio": None if reference is None else reference.depth_valid_ratio,
        "plane_inlier_ratio": None if reference is None else reference.plane_inlier_ratio,
        "plane_residual": None if reference is None else reference.plane_residual,
        "object_area_px": 0 if reference is None else reference.object_area_px,
        "surface_area_px": 0 if reference is None else reference.surface_area_px,
        "surface_ratio": (
            None
            if reference is None or reference.object_area_px <= 0
            else float(reference.surface_area_px / reference.object_area_px)
        ),
        "geometry_capture_attempt_diagnostics": [asdict(item) for item in attempts],
        # Backward-compatible field names retained for existing result consumers.
        "final_capture_attempts": len(attempts),
        "final_capture_accepted_attempt": None if accepted is None else accepted.attempt,
        "final_capture_depth_valid_ratio": (
            None if reference is None else reference.depth_valid_ratio
        ),
        "final_capture_plane_inlier_ratio": (
            None if reference is None else reference.plane_inlier_ratio
        ),
        "final_capture_plane_residual": (
            None if reference is None else reference.plane_residual
        ),
        "final_capture_object_area_px": 0 if reference is None else reference.object_area_px,
        "final_capture_surface_area_px": 0 if reference is None else reference.surface_area_px,
        "final_capture_attempt_diagnostics": [asdict(item) for item in attempts],
    }


def acquire_geometry_ready_final_frame(
    camera: Any,
    inspection_config: InspectionConfig,
    *,
    max_attempts: int = FINAL_CAPTURE_MAX_ATTEMPTS,
    geometry_extractor: Callable[..., SurfaceGeometryResult] = extract_surface_geometry,
) -> GeometryReadyFinalCapture:
    """Select the first LED-OFF fresh depth frame whose geometry is ready."""
    if isinstance(max_attempts, bool) or not isinstance(max_attempts, int) or max_attempts <= 0:
        raise ValueError("max_attempts must be a positive integer")
    attempts: list[FinalCaptureAttemptDiagnostic] = []
    for attempt in range(1, max_attempts + 1):
        frame = camera.capture()
        try:
            geometry = geometry_extractor(
                frame.depth_mm, frame.color_bgr.shape, inspection_config,
            )
            readiness = evaluate_surface_geometry_readiness(geometry, inspection_config)
            diagnostic = FinalCaptureAttemptDiagnostic(
                attempt=attempt,
                depth_valid_ratio=_finite_or_none(geometry.depth_valid_ratio),
                plane_inlier_ratio=_finite_or_none(geometry.plane_inlier_ratio),
                plane_residual=_finite_or_none(geometry.plane_residual),
                object_area_px=geometry.object_area_px,
                surface_area_px=geometry.surface_area_px,
                ready=readiness.ready,
                reasons=tuple(readiness.reasons),
            )
        except ValueError as exc:
            diagnostic = FinalCaptureAttemptDiagnostic(
                attempt=attempt,
                depth_valid_ratio=None,
                plane_inlier_ratio=None,
                plane_residual=None,
                object_area_px=0,
                surface_area_px=0,
                ready=False,
                reasons=(f"invalid geometry input: {exc}",),
            )
            geometry = None
        attempts.append(diagnostic)
        print("[FINAL GEOMETRY]")
        if attempt == 1:
            print("LED=OFF")
        print(f"attempt={attempt}/{max_attempts}")
        print(f"depth_valid={_format_metric(diagnostic.depth_valid_ratio)}")
        print(f"plane_inlier={_format_metric(diagnostic.plane_inlier_ratio)}")
        print(f"plane_residual={_format_metric(diagnostic.plane_residual)}")
        print(f"object_px={diagnostic.object_area_px}")
        print(f"surface_px={diagnostic.surface_area_px}")
        print(f"ready={diagnostic.ready}")
        if diagnostic.ready:
            assert geometry is not None
            print(f"[FINAL GEOMETRY] accepted={attempt}")
            return GeometryReadyFinalCapture(
                frame=frame,
                geometry=geometry,
                attempts=tuple(attempts),
                accepted_attempt=attempt,
            )
    metadata = _capture_metadata(attempts, None)
    raise FinalCaptureQualityError(
        f"no geometry-ready fresh final frame in {max_attempts} attempts",
        metadata=metadata,
    )


def acquire_warmed_final_rgb_frame(
    camera: Any,
    *,
    warmup_frames: int = FINAL_RGB_WARMUP_FRAMES,
    expected_shape: tuple[int, int] | None = None,
) -> Any:
    """Flush buffered frames after LED ON, then return one valid RGB frame."""
    if isinstance(warmup_frames, bool) or not isinstance(warmup_frames, int):
        raise ValueError("warmup_frames must be a non-negative integer")
    if warmup_frames < 0:
        raise ValueError("warmup_frames must be a non-negative integer")
    for index in range(1, warmup_frames + 1):
        camera.capture()
        print(f"[FINAL RGB] warmup={index}/{warmup_frames} discard")
    frame = camera.capture()
    color = np.asarray(getattr(frame, "color_bgr", None))
    if color.ndim != 3 or color.shape[2] != 3 or color.size == 0:
        raise FinalRGBCaptureError("final LED-ON RGB frame is empty or not BGR")
    if expected_shape is not None and tuple(color.shape[:2]) != tuple(expected_shape):
        raise FinalRGBCaptureError(
            "frozen surface mask / final RGB shape mismatch: "
            f"mask={tuple(expected_shape)}, RGB={tuple(color.shape[:2])}"
        )
    print("[FINAL RGB] captured")
    return frame


def _binary_mask(mask: np.ndarray | None, *, name: str) -> np.ndarray:
    if mask is None:
        raise ValueError(f"{name} is missing")
    value = np.asarray(mask)
    if value.ndim != 2 or value.size == 0:
        raise ValueError(f"{name} must be a non-empty 2D mask")
    return np.where(value > 0, 255, 0).astype(np.uint8)


def save_final_geometry_capture(
    capture: GeometryReadyFinalCapture,
    output_directory: str | Path,
    *,
    inspection_mask: InspectionMaskResult | None = None,
) -> FinalGeometryArtifacts:
    """Persist only the accepted LED-OFF depth and its frozen geometry."""
    import cv2

    root = Path(output_directory).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    depth = np.asarray(capture.frame.depth_mm, dtype=np.float32)
    object_mask = _binary_mask(capture.geometry.object_mask, name="object_mask")
    surface_mask = _binary_mask(capture.geometry.surface_mask, name="surface_mask")
    if depth.ndim != 2 or depth.shape != surface_mask.shape or object_mask.shape != surface_mask.shape:
        raise ValueError(
            "accepted geometry artifact shape mismatch: "
            f"depth={depth.shape}, object={object_mask.shape}, surface={surface_mask.shape}"
        )
    depth_path = root / "final_depth.npy"
    object_mask_path = root / "object_mask.png"
    surface_mask_path = root / "surface_mask.png"
    overlay_path = root / "surface_geometry_overlay.png"
    np.save(depth_path, depth)
    if not cv2.imwrite(str(object_mask_path), object_mask):
        raise RuntimeError(f"failed to save final object mask: {object_mask_path}")
    if not cv2.imwrite(str(surface_mask_path), surface_mask):
        raise RuntimeError(f"failed to save final surface mask: {surface_mask_path}")
    overlay_saved: str | None = None
    color = np.asarray(getattr(capture.frame, "color_bgr", None))
    if color.ndim == 3 and color.shape[:2] == surface_mask.shape:
        overlay = color.copy()
        tint = np.zeros_like(overlay)
        tint[..., 1] = surface_mask
        overlay = np.where(
            (surface_mask > 0)[..., None],
            cv2.addWeighted(overlay, 0.65, tint, 0.35, 0.0),
            overlay,
        )
        if not cv2.imwrite(str(overlay_path), overlay):
            raise RuntimeError(f"failed to save surface geometry overlay: {overlay_path}")
        overlay_saved = str(overlay_path)
    inspection_mask_saved: str | None = None
    inspection_overlay_saved: str | None = None
    if inspection_mask is not None:
        inspection_binary = _binary_mask(inspection_mask.mask, name="inspection_mask")
        if inspection_binary.shape != surface_mask.shape:
            raise ValueError(
                "inspection mask / accepted geometry shape mismatch: "
                f"inspection={inspection_binary.shape}, surface={surface_mask.shape}"
            )
        inspection_path = root / "inspection_mask.png"
        inspection_overlay_path = root / "inspection_mask_overlay.png"
        if not cv2.imwrite(str(inspection_path), inspection_binary):
            raise RuntimeError(f"failed to save inspection mask: {inspection_path}")
        inspection_mask_saved = str(inspection_path)
        if color.ndim == 3 and color.shape[:2] == inspection_binary.shape:
            inspection_overlay = color.copy()
            contours, _ = cv2.findContours(
                inspection_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE,
            )
            cv2.drawContours(inspection_overlay, contours, -1, (0, 255, 255), 2)
            if not cv2.imwrite(str(inspection_overlay_path), inspection_overlay):
                raise RuntimeError(
                    f"failed to save inspection mask overlay: {inspection_overlay_path}"
                )
            inspection_overlay_saved = str(inspection_overlay_path)
    return FinalGeometryArtifacts(
        str(depth_path), str(object_mask_path), str(surface_mask_path), overlay_saved,
        inspection_mask_saved, inspection_overlay_saved,
    )


def save_final_rgb_capture(frame: Any, output_directory: str | Path) -> FinalRGBArtifacts:
    """Persist the post-warmup LED-ON RGB frame; its depth is intentionally ignored."""
    import cv2

    root = Path(output_directory).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    rgb_path = root / "final_rgb.png"
    if not cv2.imwrite(str(rgb_path), np.asarray(frame.color_bgr)):
        raise RuntimeError(f"failed to save final RGB: {rgb_path}")
    infrared = getattr(frame, "infrared", None)
    ir_path = None
    if infrared is not None:
        ir_target = root / "final_ir.png"
        if not cv2.imwrite(str(ir_target), np.asarray(infrared)):
            raise RuntimeError(f"failed to save final IR: {ir_target}")
        ir_path = str(ir_target)
    return FinalRGBArtifacts(str(rgb_path), ir_path)


def save_final_capture(frame, output_directory: str | Path) -> FinalCaptureArtifacts:
    import cv2
    root = Path(output_directory).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    rgb_path = root / "final_rgb.png"
    depth_path = root / "final_depth.npy"
    if not cv2.imwrite(str(rgb_path), np.asarray(frame.color_bgr)):
        raise RuntimeError(f"failed to save final RGB: {rgb_path}")
    np.save(depth_path, np.asarray(frame.depth_mm, dtype=np.float32))
    infrared = getattr(frame, "infrared", None)
    ir_path = None
    if infrared is not None:
        ir_target = root / "final_ir.png"
        if not cv2.imwrite(str(ir_target), np.asarray(infrared)):
            raise RuntimeError(f"failed to save final IR: {ir_target}")
        ir_path = str(ir_target)
    return FinalCaptureArtifacts(str(rgb_path), str(depth_path), ir_path)
