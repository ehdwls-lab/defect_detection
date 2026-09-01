from __future__ import annotations

import logging

from src.inspection.z_search_types import InspectionQualitySample
from src.integration.inspection_plan import PoseTarget


class MockQualitySampler:
    def __init__(self, *, all_fail: bool = False) -> None:
        self.all_fail = all_fail

    def sample(self, pose: PoseTarget) -> tuple[InspectionQualitySample, ...]:
        values = ((20.0, False, None), (20.5, True, .68), (21.0, True, .81),
                  (21.5, True, .87), (22.0, True, .74))
        samples = []
        for z_cm, passed, score in values:
            if self.all_fail:
                passed, score = False, None
            item = InspectionQualitySample(
                z_cm=z_cm, depth_valid_ratio=.8, plane_inlier_ratio=.85,
                plane_residual_mm=1.0, object_area_px=140000,
                surface_area_px=120000, valid_patch_count=32,
                touches_fov_edge=False, rgb_mean_brightness=128.0,
                rgb_saturated_ratio=.01, rgb_sharpness=95.0,
                gate_passed=passed, quality_score=score,
                reasons=() if passed else ("mock gate failure",),
            )
            logging.getLogger(__name__).info(
                "[Z_SEARCH] pose=%s candidate=%.1f gate=%s score=%s",
                pose.pose_id, z_cm, passed, score,
            )
            samples.append(item)
        return tuple(samples)
