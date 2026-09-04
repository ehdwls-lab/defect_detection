from __future__ import annotations

import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import numpy as np

from src.config import InspectionConfig
from src.core.depth_contour_roi import build_depth_external_contour_roi
from src.integration.metric_pose import CameraIntrinsics
from src.tools.live_roi_debug import (
    cleanup_lighting, main, patch_overlay, save_snapshot, stage_images,
)


class LiveROIDebugTests(unittest.TestCase):
    def setUp(self):
        base = InspectionConfig.default()
        self.config = replace(
            base,
            depth=replace(base.depth, plane_min_points=100, object_open_size=1),
            surface_roi=replace(
                base.surface_roi, fallback_workspace_margin_px=10,
                min_object_area=100,
            ),
        )
        self.depth = np.full((240, 240), 500, dtype=np.float32)
        self.depth[70:170, 70:170] = 450
        self.color = np.full((240, 240, 3), 80, dtype=np.uint8)
        self.intrinsics = CameraIntrinsics(
            200, 200, 120, 120, 240, 240, "synthetic", "color",
        )

    def result(self):
        return build_depth_external_contour_roi(
            self.depth, self.color.shape, self.config,
            intrinsics=self.intrinsics, depth_frames=[self.depth] * 5,
            min_votes=2, current_platform_roll_deg=0,
            current_platform_pitch_deg=0, commanded_platform_roll_deg=0,
            commanded_platform_pitch_deg=0,
        )

    def test_default_invocation_is_hardware_free_dry_run(self):
        self.assertEqual(main(["--roll", "26", "--pitch", "0"]), 0)

    def test_failed_lighting_connect_does_not_run_cleanup(self):
        class Lighting:
            def __init__(self):
                self.off_calls = 0
                self.close_calls = 0
            def inspection_off(self): self.off_calls += 1
            def close(self): self.close_calls += 1

        lighting = Lighting()
        cleanup_lighting(lighting, connected=False)
        self.assertEqual((lighting.off_calls, lighting.close_calls), (0, 0))

    def test_stage_images_and_snapshot_use_production_result(self):
        result = self.result()
        overlay, patches = patch_overlay(self.color, result.inspection_mask, self.config)
        for patch in patches:
            x, y, w, h = (int(patch[key]) for key in ("x", "y", "w", "h"))
            self.assertTrue(np.all(result.inspection_mask[y:y + h, x:x + w] > 0))
        images = stage_images(self.depth, self.color, result, overlay)
        self.assertEqual(set(images), {
            "aligned_depth", "board_plane_inliers", "signed_height",
            "per_frame_candidate", "vote_fusion", "main_component",
            "closed_component", "contour_filled", "inspection_mask",
            "patch_overlay",
        })
        with tempfile.TemporaryDirectory() as raw:
            saved = save_snapshot(
                Path(raw), color=self.color, depth=self.depth, result=result,
                patches=patches, patch_image=overlay, legacy_mask=None,
            )
            for name in (
                "rgb.png", "depth_visualization.png", "board_plane_overlay.png",
                "depth_signed_height.png", "object_candidate.png", "vote_count.png",
                "main_component.png", "closed_component.png", "contour_filled.png",
                "inspection_mask.png", "surface_patch_overlay.png", "diagnostics.json",
            ):
                self.assertTrue((saved / name).is_file(), name)


if __name__ == "__main__":
    unittest.main()
