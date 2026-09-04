from __future__ import annotations

import json
import tempfile
import unittest
import warnings
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import numpy as np

from src.camera.controller import RGBDepthFrame
from src.inspection.hardware_z_search import (
    CandidateArtifactStore, HardwareAutomaticZSearch, HardwareZCandidateResult,
    HardwareZSearchConfig,
    SensorQualityConfig, SensorQualityEvaluator, SurfaceReadinessEvaluator,
)
from src.core.surface_geometry import SurfaceGeometryResult
from src.integration.projector_controller import ProjectorState
from src.core.inspection_quality import evaluate_inspection_readiness
from src.tools.test_automatic_z_hardware import main as hardware_cli_main


class FakePlatform:
    def __init__(self, projector=None): self.moves = []; self.projector = projector
    def move_and_wait(self, z, timeout):
        if self.projector is not None:
            assert self.projector.state is ProjectorState.BLACK
        self.moves.append(z)


class FakeCamera:
    def __init__(self, frames): self.frames = list(frames); self.captures = 0
    def capture(self): self.captures += 1; return self.frames.pop(0)


class FakeProjector:
    def __init__(self): self.state = ProjectorState.PHASE; self.events = []
    def show_black(self): self.state = ProjectorState.BLACK; self.events.append("BLACK")


def frame(value=100, depth=500):
    return RGBDepthFrame(
        np.full((6, 6, 3), value, dtype=np.uint8),
        np.full((6, 6), depth, dtype=np.float32), 1.0,
    )


def candidate(z, score, accepted=True):
    return HardwareZCandidateResult(
        z, 1, 2, None, None, 500, .9, .9, .1, .01, .01, 10, 10, 0,
        score if accepted else None, accepted, () if accepted else ("rejected",),
    )


class FakeEvaluator:
    def __init__(self, values): self.values = dict(values); self.projector = None
    def evaluate(self, frame, *, z_command, roll, pitch, rgb_path=None, depth_path=None):
        if self.projector is not None:
            assert self.projector.state is ProjectorState.BLACK
        score = self.values[z_command]
        return candidate(z_command, score, accepted=score is not None)


class HardwareZSearchTests(unittest.TestCase):
    def search(self, scores, candidates=(20, 22, 24)):
        projector = FakeProjector()
        platform = FakePlatform(projector)
        camera = FakeCamera([frame()] * len(candidates))
        evaluator = FakeEvaluator(scores); evaluator.projector = projector
        search = HardwareAutomaticZSearch(
            platform=platform, camera=camera, projector=projector, evaluator=evaluator,
            config=HardwareZSearchConfig(tuple(candidates), max(candidates), 1),
        )
        return search, platform, camera, projector

    def test_candidate_traversal_rejection_best_and_return(self):
        search, platform, camera, projector = self.search({20: .5, 22: .9, 24: None})
        result = search.run(pose_id="p", roll=1, pitch=2)
        self.assertTrue(result.success)
        self.assertEqual(result.best_z, 22)
        self.assertEqual(platform.moves, [20, 22, 24, 22])
        self.assertFalse(result.candidates[2].accepted)
        self.assertEqual(camera.captures, 3)
        self.assertTrue(all(event == "BLACK" for event in projector.events))

    def test_candidate_result_json_contains_surface_diagnostics(self):
        search, _, _, _ = self.search({20: .5}, candidates=(20,))
        result = search.run(pose_id="p", roll=1, pitch=2)
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "result.json"
            result.save(path)
            payload = json.loads(path.read_text(encoding="utf-8"))
        for container in (payload["candidates"][0], payload["best_metrics"]):
            for key in (
                "object_area_px", "surface_area_px", "surface_ratio",
                "usable_patch_count", "depth_p05_mm", "depth_median_mm",
                "depth_p95_mm", "board_roi_depth_valid_ratio",
            ):
                self.assertIn(key, container)

    def test_no_valid_z(self):
        search, platform, _, _ = self.search({20: None, 22: None, 24: None})
        result = search.run(pose_id="p", roll=0, pitch=0)
        self.assertFalse(result.success)
        self.assertEqual(result.failure_reason, "NoValidInspectionZ")
        self.assertEqual(platform.moves, [20, 22, 24])

    def test_best_surface_coverage_selects_quality_peak_not_highest_pass(self):
        projector = FakeProjector()
        platform = FakePlatform(projector)
        values = {
            15: (600, .50, True), 16: (800, .60, True),
            17: (1000, .70, True), 18: (900, .65, True), 19: (0, .10, False),
        }

        class Evaluator:
            def evaluate(self, frame, *, z_command, roll, pitch, **kwargs):
                area, depth_ratio, accepted = values[z_command]
                return replace(
                    candidate(z_command, None, accepted),
                    surface_area_px=area, object_area_px=area,
                    surface_ratio=1.0 if area else None,
                    depth_valid_ratio=depth_ratio,
                )

        search = HardwareAutomaticZSearch(
            platform=platform, camera=FakeCamera([frame()] * 5), projector=projector,
            evaluator=Evaluator(),
            config=HardwareZSearchConfig(
                (15, 16, 17, 18, 19), 19, 1, "best_surface_coverage",
            ),
        )
        result = search.run(pose_id="p", roll=0, pitch=0)
        self.assertTrue(result.success)
        self.assertEqual(result.best_z, 17)
        self.assertAlmostEqual(result.selected_best_z_quality_score, 1.0)
        self.assertEqual([item.z_command for item in result.candidates], [15, 16, 17, 18, 19])
        self.assertEqual(platform.moves, [15, 16, 17, 18, 19, 17])
        self.assertEqual(result.quality_score_weights["surface_area_weight"], .6)

    def test_best_surface_coverage_searches_past_initial_failures(self):
        projector = FakeProjector()
        platform = FakePlatform(projector)
        values = {15: None, 16: None, 17: .4, 18: .5, 19: None, 20: .9}

        class Evaluator(FakeEvaluator):
            def evaluate(self, frame, *, z_command, roll, pitch, **kwargs):
                base = super().evaluate(
                    frame, z_command=z_command, roll=roll, pitch=pitch, **kwargs,
                )
                return replace(
                    base, surface_area_px=int((self.values[z_command] or 0) * 1000),
                )

        search = HardwareAutomaticZSearch(
            platform=platform, camera=FakeCamera([frame()] * 5), projector=projector,
            evaluator=Evaluator(values),
            config=HardwareZSearchConfig(
                (15, 16, 17, 18, 19, 20), 20, 1, "best_surface_coverage",
            ),
        )
        result = search.run(pose_id="p", roll=0, pitch=0)
        self.assertTrue(result.success)
        self.assertEqual(result.best_z, 18)
        self.assertEqual([item.z_command for item in result.candidates], [15, 16, 17, 18, 19])
        self.assertNotIn(20, platform.moves)

    def test_surface_candidate_artifacts_include_geometry_and_diagnostics(self):
        sensor_config = SensorQualityConfig.from_json("config/automatic_z_quality.json")
        evaluator = SurfaceReadinessEvaluator(sensor_config)
        required = evaluator.config.quality.ready_streak_frames
        mask = np.zeros((32, 32), dtype=np.uint8)
        mask[8:24, 8:24] = 255
        area = int(np.count_nonzero(mask))
        geometry = SurfaceGeometryResult(
            mask.copy(), mask.copy(), (), area, area, 1.0, .4, .8, 1.0, False,
        )
        evaluated = [
            replace(
                candidate(15, None, True), readiness_pass=True,
                object_area_px=area, surface_area_px=area, surface_ratio=1.0,
                plane_inlier_ratio=.8, plane_residual=1.0,
                usable_patch_count=2, fov_edge_contact=False,
            )
            for _ in range(required)
        ]
        camera = FakeCamera([
            RGBDepthFrame(
                np.zeros((32, 32, 3), dtype=np.uint8),
                np.full((32, 32), 500, dtype=np.float32), 1,
            )
        ] * required)
        with tempfile.TemporaryDirectory() as temporary, patch.object(
            evaluator, "_evaluate_frame", side_effect=evaluated,
        ), patch(
            "src.inspection.hardware_z_search.extract_surface_geometry",
            return_value=geometry,
        ):
            result = evaluator.evaluate_candidate(
                camera, z_command=15, roll=0, pitch=0, candidate_index=0,
                artifact_store=CandidateArtifactStore(temporary),
            )
            root = Path(temporary) / "z_15"
            for name in (
                "representative_rgb.png", "representative_depth.npy",
                "object_mask.png", "surface_mask.png", "surface_overlay.png",
                "diagnostics.json",
            ):
                self.assertTrue((root / name).is_file(), name)
            payload = json.loads((root / "diagnostics.json").read_text(encoding="utf-8"))
        self.assertTrue(payload["readiness_pass"])
        self.assertEqual(payload["surface_area_px"], area)
        self.assertIn("depth_p05_mm", payload)
        self.assertIn("depth_median_mm", payload)
        self.assertIn("depth_p95_mm", payload)
        self.assertIn("board_roi_depth_valid_ratio", payload)
        self.assertEqual(result.diagnostic_dir, str(root))

    def test_highest_readiness_pass_pass_fail_returns_second(self):
        search, platform, _, _ = self.search({20: .1, 22: .2, 24: None})
        search.config = HardwareZSearchConfig((20, 22, 24), 24, 1, "highest_passing_readiness")
        result = search.run(pose_id="p", roll=0, pitch=0)
        self.assertEqual(result.best_z, 22)
        self.assertEqual(result.stop_reason, "next_candidate_failed_readiness")
        self.assertEqual(platform.moves, [20, 22, 24, 22])

    def test_highest_readiness_all_pass_selects_highest(self):
        search, platform, _, _ = self.search({20: .1, 22: .2, 24: .3})
        search.config = HardwareZSearchConfig((20, 22, 24), 24, 1, "highest_passing_readiness")
        result = search.run(pose_id="p", roll=0, pitch=0)
        self.assertEqual(result.best_z, 24)
        self.assertEqual(result.stop_reason, "highest_candidate_passed")
        self.assertEqual(platform.moves, [20, 22, 24])

    def test_highest_readiness_first_fail_stops_immediately(self):
        search, platform, camera, _ = self.search({20: None, 22: .2, 24: .3})
        search.config = HardwareZSearchConfig((20, 22, 24), 24, 1, "highest_passing_readiness")
        result = search.run(pose_id="p", roll=0, pitch=0)
        self.assertFalse(result.success)
        self.assertEqual(result.failure_reason, "NoValidInspectionZ")
        self.assertEqual(platform.moves, [20])
        self.assertEqual(camera.captures, 1)

    def test_adaptive_search_refines_between_first_coarse_failure_and_last_pass(self):
        projector = FakeProjector()
        platform = FakePlatform(projector)
        camera = FakeCamera([frame()] * 6)
        evaluator = FakeEvaluator({20: .1, 25: None, 21: .2, 22: .3, 23: .4, 24: None})
        config = HardwareZSearchConfig(
            (), 30, 1, "highest_passing_readiness", "adaptive", 20, 5, 1,
        )
        search = HardwareAutomaticZSearch(
            platform=platform, camera=camera, projector=projector, evaluator=evaluator,
            config=config,
        )
        result = search.run(pose_id="p", roll=0, pitch=0)
        self.assertTrue(result.success)
        self.assertEqual(result.coarse_candidates, (20, 25, 30))
        self.assertEqual(result.fine_candidates, (21, 22, 23, 24))
        self.assertEqual([item.z_command for item in result.candidates], [20, 25, 21, 22, 23, 24])
        self.assertEqual(result.last_pass_z, 23)
        self.assertEqual(result.first_fail_z, 24)
        self.assertEqual(result.best_z, 23)
        self.assertEqual(result.search_mode, "adaptive")
        self.assertEqual(platform.moves, [20, 25, 21, 22, 23, 24, 23])

    def test_adaptive_15_pass_20_fail_refines_to_18_before_19_fail(self):
        projector = FakeProjector()
        platform = FakePlatform(projector)
        values = {15: .1, 20: None, 16: .2, 17: .3, 18: .4, 19: None}
        search = HardwareAutomaticZSearch(
            platform=platform, camera=FakeCamera([frame()] * 6), projector=projector,
            evaluator=FakeEvaluator(values),
            config=HardwareZSearchConfig(
                (), 30, 1, "highest_passing_readiness", "adaptive", 15, 5, 1,
            ),
        )
        result = search.run(pose_id="p", roll=0, pitch=0)
        self.assertTrue(result.success)
        self.assertEqual([item.z_command for item in result.candidates], [15, 20, 16, 17, 18, 19])
        self.assertEqual(result.best_z, 18)
        self.assertEqual(platform.moves, [15, 20, 16, 17, 18, 19, 18])

    def test_adaptive_one_cm_step_is_sequential_and_needs_no_fine_candidates(self):
        projector = FakeProjector()
        platform = FakePlatform(projector)
        search = HardwareAutomaticZSearch(
            platform=platform, camera=FakeCamera([frame()] * 3), projector=projector,
            evaluator=FakeEvaluator({15: .1, 16: .2, 17: None}),
            config=HardwareZSearchConfig(
                (), 30, 1, "highest_passing_readiness", "adaptive", 15, 1, 1,
            ),
        )
        result = search.run(pose_id="p", roll=0, pitch=0)
        self.assertTrue(result.success)
        self.assertEqual(result.coarse_candidates, tuple(range(15, 31)))
        self.assertEqual(result.fine_candidates, ())
        self.assertEqual([item.z_command for item in result.candidates], [15, 16, 17])
        self.assertEqual(result.first_fail_z, 17)
        self.assertEqual(result.best_z, 16)

    def test_production_adaptive_search_descends_from_25_to_17(self):
        projector = FakeProjector()
        platform = FakePlatform(projector)
        values = {z: (0.8 if z >= 21 else 0.7) for z in range(17, 26)}

        class Evaluator:
            def evaluate(self, frame, *, z_command, roll, pitch, **kwargs):
                del frame, roll, pitch, kwargs
                return candidate(z_command, values[z_command], True)

        search = HardwareAutomaticZSearch(
            platform=platform, camera=FakeCamera([frame()] * 9), projector=projector,
            evaluator=Evaluator(),
            config=HardwareZSearchConfig(
                (), 25, 1, "best_surface_coverage", "adaptive",
                25, 1, 1, search_min_z_cm=17,
            ),
        )
        result = search.run(pose_id="p", roll=25, pitch=0)
        self.assertEqual(result.coarse_candidates, tuple(range(25, 16, -1)))
        self.assertEqual([item.z_command for item in result.candidates], list(range(25, 16, -1)))

    def test_adaptive_all_coarse_passes_returns_highest_without_fine(self):
        projector = FakeProjector()
        platform = FakePlatform(projector)
        evaluator = FakeEvaluator({20: .1, 25: .2, 30: .3})
        search = HardwareAutomaticZSearch(
            platform=platform, camera=FakeCamera([frame()] * 3), projector=projector,
            evaluator=evaluator,
            config=HardwareZSearchConfig((), 30, 1, "highest_passing_readiness", "adaptive", 20, 5, 1),
        )
        result = search.run(pose_id="p", roll=0, pitch=0)
        self.assertEqual(result.best_z, 30)
        self.assertEqual(result.fine_candidates, ())
        self.assertEqual(platform.moves, [20, 25, 30])

    def test_adaptive_first_coarse_fail_preserves_no_valid_policy(self):
        projector = FakeProjector()
        platform = FakePlatform(projector)
        search = HardwareAutomaticZSearch(
            platform=platform, camera=FakeCamera([frame()]), projector=projector,
            evaluator=FakeEvaluator({20: None}),
            config=HardwareZSearchConfig((), 30, 1, "highest_passing_readiness", "adaptive", 20, 5, 1),
        )
        result = search.run(pose_id="p", roll=0, pitch=0)
        self.assertFalse(result.success)
        self.assertEqual(result.failure_reason, "NoValidInspectionZ")
        self.assertEqual(platform.moves, [20])

    def test_z_max_never_exceeded(self):
        with self.assertRaises(ValueError):
            HardwareZSearchConfig((20, 26), 24, 1).validate()

    def test_candidates_must_be_strictly_ascending(self):
        for values in ((22, 20), (20, 20)):
            with self.subTest(values=values), self.assertRaises(ValueError):
                HardwareZSearchConfig(values, 24, 1, "highest_passing_readiness").validate()

    def test_cli_without_execute_is_hardware_free_dry_run(self):
        argv = ["test_automatic_z_hardware.py", "--port", "/dev/never-open",
                "--candidates", "20,22", "--z-max", "22"]
        with patch("sys.argv", argv):
            self.assertEqual(hardware_cli_main(), 0)

    def test_invalid_fresh_settle_blocks_before_hardware_open(self):
        argv = [
            "test_automatic_z_hardware.py", "--port", "/dev/never-open",
            "--candidates", "20,22", "--z-max", "22",
            "--timeout", "0.05", "--fresh-settle", "0.10", "--execute",
        ]
        with patch("sys.argv", argv), patch(
            "src.tools.test_automatic_z_hardware.SerialPlatformController",
        ) as serial_controller:
            with self.assertRaises(SystemExit):
                hardware_cli_main()
        serial_controller.assert_not_called()

    def test_rgb_depth_metrics_and_rejection(self):
        config = SensorQualityConfig(
            depth_min_mm=200, depth_max_mm=1000,
            min_depth_valid_ratio=.5, min_roi_depth_coverage=.5,
            max_invalid_ratio=.5, max_saturation_ratio=.2, max_dark_ratio=.2,
            min_sharpness=1, min_contrast=1, max_edge_occupancy_ratio=1,
            saturation_value=250, dark_value=5,
            score_weights={"depth_valid_ratio": 1, "sharpness": .01, "contrast": .01},
        )
        checker = np.indices((8, 8)).sum(axis=0) % 2 * 200 + 20
        color = np.repeat(checker[..., None], 3, axis=2).astype(np.uint8)
        accepted = SensorQualityEvaluator(config).evaluate(
            RGBDepthFrame(color, np.full((8, 8), 500, dtype=np.float32), 1),
            z_command=20, roll=0, pitch=0,
        )
        self.assertTrue(accepted.accepted)
        self.assertEqual(accepted.depth_p05_mm, 500.0)
        self.assertEqual(accepted.depth_median, 500.0)
        self.assertEqual(accepted.depth_p95_mm, 500.0)
        rejected = SensorQualityEvaluator(config).evaluate(
            RGBDepthFrame(np.full((8, 8, 3), 255, dtype=np.uint8), np.zeros((8, 8)), 1),
            z_command=22, roll=0, pitch=0,
        )
        self.assertFalse(rejected.accepted)
        self.assertIn("invalid_depth", rejected.rejection_reason)
        self.assertIn("rgb_saturation", rejected.rejection_reason)

    def test_recovered_config_is_parseable_but_blocks_hardware_execution(self):
        config = SensorQualityConfig.from_json("config/automatic_z_quality.json")
        self.assertEqual(config.depth_min_mm, 80)
        self.assertEqual(config.depth_max_mm, 2000)
        self.assertEqual(config.min_depth_valid_ratio, .25)
        with self.assertRaisesRegex(ValueError, "quality score/weights"):
            config.require_execution_ready()
        config.require_execution_ready("highest_passing_readiness")

    def test_incomplete_recovered_config_blocks_before_hardware_open(self):
        argv = [
            "test_automatic_z_hardware.py", "--port", "/dev/never-open",
            "--candidates", "20,22", "--z-max", "22", "--roll", "0",
            "--pitch", "0", "--quality-config", "config/automatic_z_quality.json",
            "--ack-mechanical-z-range", "--execute",
        ]
        with patch("sys.argv", argv), patch("builtins.input", return_value="NO"):
            with self.assertRaises(SystemExit):
                hardware_cli_main()

    def test_surface_readiness_requires_configured_consecutive_frames(self):
        sensor_config = SensorQualityConfig.from_json("config/automatic_z_quality.json")
        evaluator = SurfaceReadinessEvaluator(sensor_config)
        required = evaluator.config.quality.ready_streak_frames
        camera = FakeCamera([frame()] * required)
        outcomes = [True] * required
        evaluated = [replace(candidate(20, None, value), readiness_pass=value) for value in outcomes]
        with patch.object(evaluator, "_evaluate_frame", side_effect=evaluated):
            result = evaluator.evaluate_candidate(
                camera, z_command=20, roll=0, pitch=0,
                candidate_index=0, artifact_store=None,
            )
        self.assertTrue(result.readiness_pass)
        self.assertEqual(result.readiness_frames, required)
        self.assertEqual(camera.captures, required)

    def test_surface_readiness_two_fails_then_four_passes(self):
        sensor_config = SensorQualityConfig.from_json("config/automatic_z_quality.json")
        evaluator = SurfaceReadinessEvaluator(sensor_config)
        required = evaluator.config.quality.ready_streak_frames
        outcomes = [False, False] + [True] * required
        camera = FakeCamera([frame()] * (required * 2))
        evaluated = [
            replace(candidate(20, None, value), readiness_pass=value, rejection_reason=())
            for value in outcomes
        ]
        with patch.object(evaluator, "_evaluate_frame", side_effect=evaluated):
            result = evaluator.evaluate_candidate(
                camera, z_command=20, roll=0, pitch=0,
                candidate_index=0, artifact_store=None,
            )
        self.assertTrue(result.readiness_pass)
        self.assertEqual(camera.captures, required + 2)

    def test_surface_readiness_stops_after_double_streak_attempts(self):
        sensor_config = SensorQualityConfig.from_json("config/automatic_z_quality.json")
        evaluator = SurfaceReadinessEvaluator(sensor_config)
        required = evaluator.config.quality.ready_streak_frames
        max_attempts = required * 2
        outcomes = [index % 2 == 0 for index in range(max_attempts)]
        camera = FakeCamera([frame()] * max_attempts)
        evaluated = [
            replace(candidate(20, None, value), readiness_pass=value, rejection_reason=())
            for value in outcomes
        ]
        with patch.object(evaluator, "_evaluate_frame", side_effect=evaluated):
            result = evaluator.evaluate_candidate(
                camera, z_command=20, roll=0, pitch=0,
                candidate_index=0, artifact_store=None,
            )
        self.assertFalse(result.readiness_pass)
        self.assertEqual(camera.captures, max_attempts)
        self.assertIn(f"readiness_streak=0/{required}", result.rejection_reason)

    def test_automatic_z_patch_count_is_diagnostic_not_hard_gate(self):
        object_mask = np.zeros((100, 100), dtype=np.uint8)
        object_mask[25:75, 25:75] = 255
        surface_mask = object_mask.copy()
        result = evaluate_inspection_readiness(
            object_mask, surface_mask, [], .30, .80, 1.0,
            min_valid_patches=None, fov_edge_margin_px=10,
        )
        self.assertTrue(result.ready)
        self.assertEqual(result.metrics["usable_patch_count"], 0)
        self.assertEqual(result.metrics["object_area_px"], 2500)
        self.assertEqual(result.metrics["surface_area_px"], 2500)
        self.assertEqual(result.metrics["surface_ratio"], 1.0)

    def test_missing_or_empty_surface_masks_fail_readiness(self):
        mask = np.zeros((100, 100), dtype=np.uint8)
        mask[25:75, 25:75] = 255
        missing_object = evaluate_inspection_readiness(
            None, mask, [], .30, .80, 1.0, min_valid_patches=None,
        )
        empty_surface = evaluate_inspection_readiness(
            mask, np.zeros_like(mask), [], .30, .80, 1.0, min_valid_patches=None,
        )
        self.assertIn("Object surface not found", missing_object.reasons)
        self.assertIn("Surface-only mask not found", empty_surface.reasons)

    def test_all_nan_depth_median_warning_is_suppressed(self):
        sensor_config = SensorQualityConfig.from_json("config/automatic_z_quality.json")
        evaluator = SurfaceReadinessEvaluator(sensor_config)
        required = evaluator.config.quality.ready_streak_frames
        camera = FakeCamera([frame(depth=0)] * required)
        evaluated = [
            replace(candidate(20, None, True), readiness_pass=True)
            for _ in range(required)
        ]
        with warnings.catch_warnings(record=True) as caught, patch.object(
            evaluator, "_evaluate_frame", side_effect=evaluated,
        ):
            warnings.simplefilter("always")
            evaluator.evaluate_candidate(
                camera, z_command=20, roll=0, pitch=0,
                candidate_index=0, artifact_store=None,
            )
        self.assertFalse(any("All-NaN slice" in str(item.message) for item in caught))


if __name__ == "__main__":
    unittest.main()
