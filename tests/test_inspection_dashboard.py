from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import cv2

from src.tools.run_inspection_ui import build_parser
from src.ui.image_utils import (anomaly_localization_overlay, depth_preview,
                                roi_contour_overlay, threshold_relative_heatmap)
from src.ui.inspection_presenter import display_judgement, load_inspection_view
from src.ui.run_replay import ReplayCursor, build_replay_events


class InspectionDashboardTests(unittest.TestCase):
    def test_presenter_reads_production_result_without_control_imports(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); (root / "rgb.png").touch()
            (root / "cycle_result.json").write_text(json.dumps({
                "overall_status": "COMPLETE", "final_judgement": "OK",
                "stage": "COMPLETE", "stage_history": ["CONNECT", "COMPLETE"],
                "conveyor_out_executed": True,
                "inspection_planes": [{
                    "plane_name": "Physical Plane 1", "status": "COMPLETE",
                    "inspection_judgement": "OK", "actual_platform_roll_deg": -2,
                    "actual_platform_pitch_deg": 1, "actual_platform_z_cm": 19,
                    "anomaly_result": {"score": .01, "threshold": .02,
                                       "classification": "NORMAL",
                                       "metadata": {"rgb_path": "rgb.png"}},
                }],
            }), encoding="utf-8")
            view = load_inspection_view(root)
        self.assertEqual((view.status, view.judgement, view.transport_complete),
                         ("COMPLETE", "OK", True))
        self.assertEqual((view.poses[0].score, view.poses[0].threshold), (.01, .02))
        self.assertEqual(view.poses[0].rgb, root / "rgb.png")

    def test_missing_and_malformed_result_are_explicit(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with self.assertRaises(FileNotFoundError): load_inspection_view(root)
            (root / "cycle_result.json").write_text("{", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "malformed"): load_inspection_view(root)

    def test_depth_projection_fallback_handles_invalid_pixels(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "depth.npy"
            np.save(path, np.array([[0, 100], [200, np.nan]], dtype=np.float32))
            preview = depth_preview(path)
        self.assertEqual(preview.shape, (2, 2, 3))
        self.assertTrue(np.all(preview[0, 0] == 0))

    def test_mask_overlay_draws_green_contour_and_missing_mask_is_safe(self):
        with tempfile.TemporaryDirectory() as temporary:
            mask_path = Path(temporary) / "mask.png"
            mask = np.zeros((32, 32), dtype=np.uint8); mask[8:24, 8:24] = 255
            cv2.imwrite(str(mask_path), mask)
            rgb = np.zeros((32, 32, 3), dtype=np.uint8)
            overlay = roi_contour_overlay(rgb, mask_path)
            missing = roi_contour_overlay(rgb, Path(temporary) / "missing.png")
        self.assertGreater(np.count_nonzero(overlay[:, :, 1]), 0)
        self.assertTrue(np.array_equal(missing, rgb))

    def test_heatmap_uses_threshold_relative_visual_scale_only(self):
        raw = np.full((16, 16, 3), 200, dtype=np.uint8)
        below = threshold_relative_heatmap(raw, .5, 1.0)
        above = threshold_relative_heatmap(raw, 1.5, 1.0)
        self.assertFalse(np.array_equal(below, above))
        self.assertEqual(display_judgement("OK"), "NORMAL")
        self.assertEqual(display_judgement("NG"), "DEFECT")
        self.assertEqual(display_judgement("PARTIAL_COMPLETE"), "RECHECK")
        localized = anomaly_localization_overlay(
            np.zeros((32, 48, 3), dtype=np.uint8), raw, 1.5, 1.0,
        )
        self.assertEqual(localized.shape, (32, 48, 3))

    def test_live_mode_is_observer_only_and_requires_existing_run_argument(self):
        args = build_parser().parse_args(["--mode", "live", "--run", "/tmp/run",
                                          "--screenshot", "/tmp/ui.png"])
        self.assertEqual((args.mode, args.run), ("live", Path("/tmp/run")))
        self.assertEqual(args.screenshot, Path("/tmp/ui.png"))

    def test_replay_timeline_is_ordered_and_restartable(self):
        events = build_replay_events(11)
        self.assertEqual([event.payload["stage"] for event in events],
                         ["IN", "3D SCAN", "POSE", "INSPECTION", "ANALYSIS", "JUDGEMENT", "OUT"])
        cursor = ReplayCursor(11)
        self.assertEqual([event.payload["stage"] for event in cursor.advance(5)],
                         ["IN", "3D SCAN", "POSE", "INSPECTION"])
        cursor.restart()
        self.assertEqual(cursor.advance(0)[0].payload["stage"], "IN")


if __name__ == "__main__":
    unittest.main()
