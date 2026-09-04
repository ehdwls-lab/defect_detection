from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from src.camera.controller import RGBDepthFrame
from src.config import InspectionConfig
from src.core.surface_geometry import SurfaceGeometryResult
from src.integration.final_capture import (
    FINAL_CAPTURE_MAX_ATTEMPTS,
    acquire_warmed_final_rgb_frame,
    acquire_geometry_ready_final_frame,
    save_final_geometry_capture,
    save_final_rgb_capture,
    save_final_capture,
)
from src.integration.inspection_failures import FinalCaptureQualityError


class FakeCamera:
    def __init__(self, frames):
        self.frames = list(frames)
        self.captures = 0

    def capture(self):
        self.captures += 1
        return self.frames.pop(0)


def frame(marker: int) -> RGBDepthFrame:
    return RGBDepthFrame(
        np.full((128, 128, 3), marker, dtype=np.uint8),
        np.full((128, 128), 500 + marker, dtype=np.float32),
        float(marker),
    )


def geometry(depth_valid_ratio: float, *, patches=()) -> SurfaceGeometryResult:
    mask = np.zeros((128, 128), dtype=np.uint8)
    mask[32:96, 32:96] = 255
    return SurfaceGeometryResult(
        object_mask=mask.copy(), surface_mask=mask.copy(), patches=tuple(patches),
        object_area_px=4096, surface_area_px=4096, surface_ratio=1.0,
        depth_valid_ratio=depth_valid_ratio, plane_inlier_ratio=.8,
        plane_residual=1.0, fov_edge_contact=False,
    )


def extractor_for(ratios):
    values = iter(ratios)

    def extract(depth_mm, image_shape, inspection_config):
        del depth_mm, image_shape, inspection_config
        return geometry(next(values))

    return extract


class FinalCaptureReadinessTests(unittest.TestCase):
    def setUp(self):
        self.config = InspectionConfig.default()

    def test_selects_third_fresh_pair_after_two_low_depth_frames(self):
        frames = [frame(1), frame(2), frame(3)]
        camera = FakeCamera(frames)
        selected = acquire_geometry_ready_final_frame(
            camera, self.config, max_attempts=8,
            geometry_extractor=extractor_for((.18, .22, .27)),
        )
        self.assertIs(selected.frame, frames[2])
        self.assertEqual(selected.accepted_attempt, 3)
        self.assertEqual(camera.captures, 3)
        self.assertEqual(selected.frame.color_bgr[0, 0, 0], 3)
        self.assertEqual(selected.frame.depth_mm[0, 0], 503)
        self.assertEqual(selected.metadata()["final_capture_depth_valid_ratio"], .27)

    def test_led_off_geometry_selects_second_frame_at_existing_threshold(self):
        camera = FakeCamera([frame(1), frame(2)])
        selected = acquire_geometry_ready_final_frame(
            camera, self.config,
            geometry_extractor=extractor_for((.18, .27)),
        )
        self.assertEqual(selected.accepted_attempt, 2)
        self.assertEqual(selected.metadata()["geometry_capture_attempts"], 2)
        self.assertEqual(selected.metadata()["geometry_accepted_attempt"], 2)

    def test_first_geometry_valid_frame_returns_immediately(self):
        camera = FakeCamera([frame(1), frame(2)])
        selected = acquire_geometry_ready_final_frame(
            camera, self.config,
            geometry_extractor=extractor_for((.27, .30)),
        )
        self.assertEqual(selected.accepted_attempt, 1)
        self.assertEqual(camera.captures, 1)

    def test_all_eight_invalid_frames_raise_recoverable_quality_failure(self):
        camera = FakeCamera([frame(index) for index in range(8)])
        with self.assertRaises(FinalCaptureQualityError) as caught:
            acquire_geometry_ready_final_frame(
                camera, self.config,
                geometry_extractor=extractor_for((.18,) * FINAL_CAPTURE_MAX_ATTEMPTS),
            )
        self.assertEqual(camera.captures, 8)
        self.assertEqual(caught.exception.stage, "FINAL_GEOMETRY_CAPTURE")
        self.assertEqual(caught.exception.metadata["final_capture_attempts"], 8)
        self.assertIsNone(caught.exception.metadata["final_capture_accepted_attempt"])
        self.assertEqual(len(
            caught.exception.metadata["final_capture_attempt_diagnostics"]
        ), 8)

    def test_only_accepted_frame_is_saved_as_final_artifact(self):
        frames = [frame(11), frame(22), frame(33)]
        selected = acquire_geometry_ready_final_frame(
            FakeCamera(frames), self.config,
            geometry_extractor=extractor_for((.18, .22, .27)),
        )
        with tempfile.TemporaryDirectory() as temporary:
            artifacts = save_final_capture(selected.frame, temporary)
            rgb = cv2.imread(artifacts.rgb_path)
            depth = np.load(artifacts.depth_path)
            files = sorted(path.name for path in Path(temporary).iterdir())
        self.assertTrue(np.all(rgb == 33))
        self.assertTrue(np.all(depth == 533))
        self.assertEqual(files, ["final_depth.npy", "final_rgb.png"])

    def test_patch_count_is_not_a_final_capture_readiness_gate(self):
        selected = acquire_geometry_ready_final_frame(
            FakeCamera([frame(1)]), self.config,
            geometry_extractor=lambda *_: geometry(.27, patches=()),
        )
        self.assertEqual(selected.accepted_attempt, 1)
        self.assertEqual(selected.geometry.patches, ())

    def test_rgb_warmup_discards_three_and_returns_fourth_frame(self):
        frames = [frame(1), frame(2), frame(3), frame(4)]
        camera = FakeCamera(frames)
        selected = acquire_warmed_final_rgb_frame(
            camera, warmup_frames=3, expected_shape=(128, 128),
        )
        self.assertIs(selected, frames[3])
        self.assertEqual(camera.captures, 4)

    def test_geometry_and_rgb_artifacts_are_saved_from_separate_frames(self):
        geometry_frame = frame(7)
        selected = acquire_geometry_ready_final_frame(
            FakeCamera([geometry_frame]), self.config,
            geometry_extractor=extractor_for((.27,)),
        )
        rgb_frame = frame(99)
        rgb_frame = RGBDepthFrame(
            rgb_frame.color_bgr, np.zeros_like(rgb_frame.depth_mm), rgb_frame.timestamp,
        )
        with tempfile.TemporaryDirectory() as temporary:
            geometry_artifacts = save_final_geometry_capture(selected, temporary)
            rgb_artifacts = save_final_rgb_capture(rgb_frame, temporary)
            saved_depth = np.load(geometry_artifacts.depth_path)
            saved_rgb = cv2.imread(rgb_artifacts.rgb_path)
            files = {path.name for path in Path(temporary).iterdir()}
        self.assertTrue(np.all(saved_depth == 507))
        self.assertTrue(np.all(saved_rgb == 99))
        self.assertIn("object_mask.png", files)
        self.assertIn("surface_mask.png", files)
        self.assertIn("surface_geometry_overlay.png", files)


if __name__ == "__main__":
    unittest.main()
