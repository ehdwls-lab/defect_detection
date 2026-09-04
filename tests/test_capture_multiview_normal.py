from __future__ import annotations

import csv
from types import SimpleNamespace
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.tools.capture_multiview_normal import (
    STANDARD_10, _park_platform_and_open_cover, _write_manifests, build_multiview_motion,
    build_multiview_z_config, build_parser, parse_custom_poses, run,
)
from src.tools.test_integrated_inspection_cycle import build_parser as integrated_parser


class CaptureMultiviewNormalTests(unittest.TestCase):
    def args(self, *extra):
        return build_parser().parse_args([
            "--session", "GRAY_01", "--material", "gray",
            "--platform-port", "/dev/never-platform",
            "--lighting-port", "/dev/never-lighting",
            "--cover-open-angle", "90", "--cover-close-angle", "0",
            "--quality-config", "config/automatic_z_quality.json", *extra,
        ])

    def test_dry_run_opens_no_hardware(self):
        with patch(
            "src.tools.capture_multiview_normal.execute_capture",
        ) as execute:
            self.assertEqual(run(self.args()), 0)
        execute.assert_not_called()

    def test_standard_preset_order_and_split_are_frozen(self):
        self.assertEqual(
            [(p.roll_deg, p.pitch_deg) for p in STANDARD_10],
            [(0, 0), (10, -1), (20, -3), (25, -4),
             (-10, 1), (-20, 1), (-25, 2), (0, 3), (15, -2), (-15, 2)],
        )
        self.assertEqual([p.split for p in STANDARD_10].count("train"), 8)
        self.assertEqual([p.split for p in STANDARD_10].count("val"), 2)

    def test_custom_pose_list_supports_source_level_splits(self):
        poses = parse_custom_poses("10,-1,train;-15,2,val")
        self.assertEqual([(p.roll_deg, p.pitch_deg, p.split) for p in poses],
                         [(10, -1, "train"), (-15, 2, "val")])

    def test_manifest_writer_keeps_only_supplied_accepted_pose_rows(self):
        rows = [{
            "path": "/capture/final_rgb.png", "split": "train", "label": "normal",
            "session": "GRAY_01", "mask": "/capture/inspection_mask.png",
            "source_run": "run", "plane": "pose_00", "material": "gray",
            "view": "production_pose", "notes": "patches=1;coverage=1.0",
        }]
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            _write_manifests(rows, output)
            with (output / "train.csv").open(encoding="utf-8") as handle:
                train = list(csv.DictReader(handle))
            with (output / "val.csv").open(encoding="utf-8") as handle:
                val = list(csv.DictReader(handle))
        self.assertEqual(len(train), 1)
        self.assertEqual(val, [])
        self.assertEqual(train[0]["mask"], "/capture/inspection_mask.png")

    def test_execute_still_requires_explicit_confirmation(self):
        args = self.args("--execute")
        with patch(
            "src.tools.capture_multiview_normal.execute_capture",
        ) as execute, self.assertRaisesRegex(ValueError, "cancelled"):
            run(args, confirmation_input=lambda _: "NO")
        execute.assert_not_called()

    def test_motion_defaults_match_integrated_production_and_not_implicit_ten_seconds(self):
        multiview = self.args()
        integrated = integrated_parser().parse_args([
            "--conveyor-port", "c", "--platform-port", "p",
            "--lighting-port", "l", "--conveyor-steps", "6325",
            "--monitor", "HDMI-0", "--scan-z", "0", "--safe-z", "15",
            "--z-start", "25", "--z-coarse-step", "1", "--z-fine-step", "1",
            "--z-max", "25", "--quality-config", "config/automatic_z_quality.json",
        ])
        names = (
            "platform_motion_timeout", "post_command_guard", "stable_samples",
            "deadband_observation", "fresh_settle", "z_target_tolerance",
            "orientation_target_tolerance",
        )
        self.assertEqual([getattr(multiview, n) for n in names],
                         [getattr(integrated, n) for n in names])
        motion = build_multiview_motion(object(), multiview)
        self.assertEqual(motion.timeout_s, 30.0)
        self.assertNotEqual(motion.timeout_s, 10.0)
        self.assertEqual(motion.wait_config.stable_sample_count, 3)

    def test_motion_cli_and_auto_z_cli_are_wired_to_runtime_configs(self):
        args = self.args(
            "--platform-motion-timeout", "47", "--post-command-guard", ".7",
            "--stable-samples", "5", "--deadband-observation", ".8",
            "--fresh-settle", ".3", "--z-target-tolerance", ".2",
            "--orientation-target-tolerance", ".15", "--z-start", "24",
            "--z-search-min", "18", "--z-max", "25",
            "--z-coarse-step", "2", "--z-fine-step", ".5",
        )
        motion = build_multiview_motion(object(), args)
        self.assertEqual(motion.timeout_s, 47)
        self.assertEqual(motion.wait_config.post_command_guard_s, .7)
        self.assertEqual(motion.wait_config.stable_sample_count, 5)
        self.assertEqual(motion.wait_config.deadband_observation_s, .8)
        self.assertEqual(motion.wait_config.fresh_read_settle_s, .3)
        self.assertEqual(motion.wait_config.z_target_tolerance_cm, .2)
        self.assertEqual(motion.wait_config.orientation_target_tolerance_deg, .15)
        z = build_multiview_z_config(args)
        self.assertEqual(
            (z.z_start, z.search_min_z_cm, z.z_max, z.coarse_step, z.fine_step,
             z.stable_timeout_s), (24, 18, 25, 2, .5, 47),
        )

    def test_final_cleanup_parks_before_opening_cover(self):
        events = []

        class Motion:
            def read_before(self):
                events.append("telemetry")
                return SimpleNamespace(roll_deg=1, pitch_deg=-1, z_cm=20)

            def execute_orientation(self, **kwargs):
                events.append(("orientation", kwargs["roll_deg"], kwargs["pitch_deg"]))

            def execute_z(self, z):
                events.append(("z", z))
                return SimpleNamespace(roll_deg=0, pitch_deg=0, z_cm=0)

        class Lighting:
            def projector_cover_open(self):
                events.append("cover_open")

        _park_platform_and_open_cover(Motion(), Lighting())
        self.assertEqual(events, [
            "telemetry", ("orientation", 0, 0), ("z", 0), "cover_open",
        ])

    def test_unsafe_park_never_opens_cover(self):
        class Motion:
            def read_before(self):
                return SimpleNamespace(roll_deg=0, pitch_deg=0, z_cm=20)

            def execute_orientation(self, **kwargs):
                return None

            def execute_z(self, z):
                return SimpleNamespace(roll_deg=0, pitch_deg=0, z_cm=1)

        lighting = unittest.mock.Mock()
        with self.assertRaisesRegex(RuntimeError, "park telemetry mismatch"):
            _park_platform_and_open_cover(Motion(), lighting)
        lighting.projector_cover_open.assert_not_called()


if __name__ == "__main__":
    unittest.main()
