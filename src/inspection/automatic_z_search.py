from __future__ import annotations

from typing import Protocol

from .z_search_types import BestZResult, InspectionQualitySample


class ZSearchController(Protocol):
    def search(self, *, pose_id: str, start_z_cm: float, end_z_cm: float, step_z_cm: float) -> BestZResult:
        ...


class AutomaticZSearch:
    """Thin interface skeleton for automatic Z search.

    This does not contain a hardware serial implementation or an arbitrary
    heuristic. It defines the interface and expected return contract only.
    """

    def __init__(self, *, start_z_cm: float = 18.0, end_z_cm: float = 30.0, step_z_cm: float = 0.5) -> None:
        self.start_z_cm = float(start_z_cm)
        self.end_z_cm = float(end_z_cm)
        self.step_z_cm = float(step_z_cm)

    def search(self, *, pose_id: str, start_z_cm: float | None = None, end_z_cm: float | None = None, step_z_cm: float | None = None) -> BestZResult:
        z0 = self.start_z_cm if start_z_cm is None else float(start_z_cm)
        z1 = self.end_z_cm if end_z_cm is None else float(end_z_cm)
        step = self.step_z_cm if step_z_cm is None else float(step_z_cm)

        if step <= 0:
            raise ValueError("Automatic Z search step size must be positive.")

        sample = InspectionQualitySample(
            z_cm=z0,
            depth_valid_ratio=0.0,
            plane_inlier_ratio=0.0,
            plane_residual_mm=0.0,
            object_area_px=0,
            surface_area_px=0,
            valid_patch_count=0,
            touches_fov_edge=False,
            gate_passed=False,
            quality_score=None,
            reasons=("Automatic Z Search is a placeholder interface only.",),
        )

        return BestZResult(
            success=False,
            best_z_cm=None,
            best_quality=None,
            samples=(sample,),
            failure_reason="Automatic Z Search not implemented yet; interface contract only.",
            pose_id=pose_id,
        )
