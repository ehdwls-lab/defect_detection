from __future__ import annotations

import unittest
from dataclasses import replace

import numpy as np

from src.config import InspectionConfig
from src.core.depth_contour_roi import (
    DepthExternalContourROIError,
    build_depth_external_contour_roi as _build_depth_external_contour_roi,
)
from src.core.depth_processing import (
    close_object_component,
    fill_external_object_contour,
    guarded_convex_hull,
    select_main_object_component,
)
from src.core.patch_extractor import generate_surface_patches
from src.core.rgb_seeded_roi import build_rgb_seeded_roi
from src.integration.platform_pose_calibration import (
    CAMERA_PLATFORM_RP_20260903,
    predicted_board_normal_from_platform_pose,
)
from src.integration.metric_pose import CameraIntrinsics


def build_depth_external_contour_roi(depth, image_shape, cfg, **kwargs):
    kwargs.setdefault("intrinsics", CameraIntrinsics(
        200.0, 200.0, depth.shape[1] / 2, depth.shape[0] / 2,
        depth.shape[1], depth.shape[0], "synthetic", "color",
    ))
    return _build_depth_external_contour_roi(depth, image_shape, cfg, **kwargs)


def config() -> InspectionConfig:
    base = InspectionConfig.default()
    return replace(
        base,
        depth=replace(
            base.depth, plane_min_points=100, object_open_size=1,
            object_close_size=3, object_close_iterations=1,
        ),
        surface_roi=replace(
            base.surface_roi, min_object_area=100,
            fallback_workspace_margin_px=10, fallback_plane_ring_px=20,
        ),
        hybrid_roi=replace(base.hybrid_roi, board_plane_border_fraction=.12),
    )


def scene() -> tuple[np.ndarray, np.ndarray]:
    depth = np.full((240, 240), 500, dtype=np.float32)
    depth[70:170, 70:170] = 450
    color = np.full((240, 240, 3), 100, dtype=np.uint8)
    return depth, color


class DepthExternalContourROITests(unittest.TestCase):
    def test_solid_component_fill_is_unchanged(self):
        component = np.zeros((80, 80), dtype=np.uint8)
        component[20:60, 25:55] = 255
        filled = fill_external_object_contour(
            component, np.full_like(component, 255),
        )
        np.testing.assert_array_equal(filled, component)

    def test_large_internal_hole_is_filled(self):
        component = np.zeros((100, 100), dtype=np.uint8)
        component[15:85, 15:85] = 255
        component[35:65, 35:65] = 0
        filled = fill_external_object_contour(
            component, np.full_like(component, 255),
        )
        self.assertEqual(int(filled[50, 50]), 255)

    def test_multiple_internal_holes_are_filled(self):
        component = np.zeros((100, 100), dtype=np.uint8)
        component[10:90, 10:90] = 255
        component[25:40, 25:40] = 0
        component[55:75, 55:75] = 0
        filled = fill_external_object_contour(
            component, np.full_like(component, 255),
        )
        self.assertEqual(int(filled[30, 30]), 255)
        self.assertEqual(int(filled[65, 65]), 255)

    def test_open_background_not_replaced_with_bounding_box(self):
        component = np.zeros((100, 100), dtype=np.uint8)
        component[20:80, 20:30] = 255
        component[70:80, 20:80] = 255
        filled = fill_external_object_contour(
            component, np.full_like(component, 255),
        )
        self.assertEqual(int(filled[40, 60]), 0)

    def test_largest_guarded_component_selected_and_noise_removed(self):
        candidate = np.zeros((120, 120), dtype=np.uint8)
        candidate[30:90, 30:90] = 255
        candidate[5:10, 5:10] = 255
        cfg = replace(config().surface_roi, min_object_area=100)
        main = select_main_object_component(
            candidate, np.full_like(candidate, 255), cfg,
        )
        self.assertEqual(int(main[50, 50]), 255)
        self.assertEqual(int(main[7, 7]), 0)

    def test_fallback_workspace_plane_fill_erosion_and_hole_patch(self):
        depth, color = scene()
        depth[100:140, 100:140] = 0
        result = build_depth_external_contour_roi(
            depth, color.shape, config(), board_quad=None,
        )
        self.assertEqual(result.workspace_source, "fallback")
        self.assertEqual(int(result.depth_main_component_mask[120, 120]), 0)
        self.assertEqual(int(result.depth_object_contour_filled[120, 120]), 255)
        self.assertEqual(int(result.inspection_mask[120, 120]), 255)
        self.assertGreater(result.fill_gain_px, 0)
        self.assertTrue(np.all(
            (result.inspection_mask == 0)
            | (result.depth_object_contour_filled > 0)
        ))
        patches = generate_surface_patches(result.inspection_mask, 64, 32, .8)
        self.assertTrue(any(
            item["x"] <= 120 < item["x"] + 64
            and item["y"] <= 120 < item["y"] + 64
            for item in patches
        ))

    def test_aruco_workspace_is_used_and_output_is_clipped(self):
        depth, color = scene()
        quad = np.array([[20, 20], [220, 20], [220, 220], [20, 220]], np.float32)
        result = build_depth_external_contour_roi(
            depth, color.shape, config(), board_quad=quad,
        )
        self.assertEqual(result.workspace_source, "aruco")
        self.assertFalse(np.any(
            (result.depth_object_contour_filled > 0)
            & (result.workspace_mask == 0)
        ))

    def test_full_workspace_sized_object_is_rejected(self):
        depth, color = scene()
        strict = config()
        strict = replace(
            strict,
            surface_roi=replace(strict.surface_roi, max_object_area_ratio=.10),
        )
        with self.assertRaises(DepthExternalContourROIError):
            build_depth_external_contour_roi(
                depth, color.shape, strict, board_quad=None,
            )

    def test_spatial_board_plane_is_not_pulled_to_central_object(self):
        depth, color = scene()
        result = build_depth_external_contour_roi(depth, color.shape, config())
        self.assertEqual(result.board_plane_source, "depth_spatial_multi_plane")
        self.assertGreater(result.board_plane_point_count, 100)
        self.assertGreater(result.candidate_signed_height_median_mm, 4.0)
        self.assertGreater(result.candidate_signed_height_p05_mm, 4.0)

    def test_vote_fusion_recovers_different_edge_dropouts_and_rejects_noise(self):
        depth, color = scene()
        frames = []
        for dropout in (slice(70, 80), slice(160, 170), slice(90, 100), slice(140, 150), None):
            frame = depth.copy()
            if dropout is not None:
                frame[70:170, dropout] = 0
            frame[10:13, 10:13] = 450
            frames.append(frame)
        result = build_depth_external_contour_roi(
            frames[0], color.shape, config(), depth_frames=frames, min_votes=2,
        )
        self.assertEqual(result.roi_depth_frame_count, 5)
        self.assertEqual(result.roi_min_votes, 2)
        self.assertGreater(result.fused_candidate_area_px, 9000)
        self.assertEqual(int(result.depth_main_component_mask[11, 11]), 0)
        self.assertEqual(int(result.depth_object_contour_filled[120, 120]), 255)

    def test_small_gap_closes_inside_workspace_but_hull_is_not_default(self):
        component = np.zeros((120, 120), dtype=np.uint8)
        component[25:95, 25:95] = 255
        component[58:63, 25:35] = 0
        workspace = np.zeros_like(component)
        workspace[10:110, 10:110] = 255
        cfg = replace(config().surface_roi, inspection_close_size_px=9)
        closed = close_object_component(component, workspace, cfg)
        self.assertEqual(int(closed[60, 30]), 255)
        depth, color = scene()
        result = build_depth_external_contour_roi(depth, color.shape, config())
        self.assertFalse(result.hull_used)
        self.assertTrue(np.all((closed > 0) <= (workspace > 0)))

    def test_reversed_plane_does_not_create_object_candidate(self):
        depth, color = scene()
        reversed_depth = np.full_like(depth, 450)
        reversed_depth[70:170, 70:170] = 500
        with self.assertRaises(DepthExternalContourROIError):
            build_depth_external_contour_roi(
                reversed_depth, color.shape, config(),
            )

    def test_platform_pose_predicts_front_facing_board_normal(self):
        normal = predicted_board_normal_from_platform_pose(0.0, 0.0, 4.0, -3.0)
        camera_tilt = np.asarray(
            CAMERA_PLATFORM_RP_20260903.camera_from_platform_J,
        ) @ np.asarray([4.0, -3.0])
        expected = np.asarray([
            np.tan(np.radians(camera_tilt[0])),
            np.tan(np.radians(camera_tilt[1])),
            -1.0,
        ])
        expected /= np.linalg.norm(expected)
        np.testing.assert_allclose(normal, expected, atol=1e-12)

    def test_matching_board_normal_prior_is_recorded(self):
        depth, color = scene()
        result = build_depth_external_contour_roi(
            depth, color.shape, config(),
            current_platform_roll_deg=0.0,
            current_platform_pitch_deg=0.0,
            commanded_platform_roll_deg=0.0,
            commanded_platform_pitch_deg=0.0,
        )
        np.testing.assert_allclose(
            result.predicted_board_normal,
            predicted_board_normal_from_platform_pose(0.0, 0.0, 0.0, 0.0),
        )
        np.testing.assert_allclose(result.plane_normal, [0.0, 0.0, -1.0], atol=1e-6)
        self.assertAlmostEqual(result.normal_angle_error_deg, 0.0, places=5)
        self.assertGreaterEqual(len(result.plane_hypotheses), 2)
        self.assertEqual(result.board_plane_source, "depth_spatial_multi_plane")

    def test_metric_tilted_board_normal_and_spatial_recovery(self):
        cfg = config()
        intr = CameraIntrinsics(200, 200, 120, 120, 240, 240, "synthetic", "color")
        yy, xx = np.mgrid[0:240, 0:240]

        def plane(normal, center_z):
            normal = np.asarray(normal, dtype=np.float64)
            normal /= np.linalg.norm(normal)
            rays_x = (xx - intr.cx) / intr.fx
            rays_y = (yy - intr.cy) / intr.fy
            return (normal[2] * center_z / (
                normal[0] * rays_x + normal[1] * rays_y + normal[2]
            )).astype(np.float32)

        board_normal = np.array([0.0, 0.0, -1.0])
        depth = plane(board_normal, 500)
        depth[80:160, 80:160] = 450
        wall_normal = np.array([0.8, 0.0, -0.6])
        wall = plane(wall_normal, 500)
        depth[10:60] = wall[10:60]
        result = _build_depth_external_contour_roi(
            depth, (240, 240, 3), cfg, intrinsics=intr,
            current_platform_roll_deg=0, current_platform_pitch_deg=0,
            commanded_platform_roll_deg=0, commanded_platform_pitch_deg=0,
        )
        np.testing.assert_allclose(result.plane_normal, board_normal, atol=1e-3)
        self.assertGreaterEqual(len(result.plane_hypotheses), 2)
        self.assertTrue(any(
            item["rejected_reason"] == "metric normal exceeds board-normal guard"
            for item in result.plane_hypotheses
        ))

    def test_partial_aruco_one_two_three_markers_use_local_depth(self):
        depth, color = scene()
        corners = {
            0: np.array([[25, 25], [40, 25], [40, 40], [25, 40]], np.float32),
            1: np.array([[200, 25], [215, 25], [215, 40], [200, 40]], np.float32),
            2: np.array([[25, 200], [40, 200], [40, 215], [25, 215]], np.float32),
        }
        for count in (1, 2, 3):
            result = build_depth_external_contour_roi(
                depth, color.shape, config(), marker_map={i: corners[i] for i in range(count)},
            )
            self.assertEqual(result.board_plane_source, "aruco_partial_local_depth")
            self.assertEqual(result.aruco_detected_count, count)
            self.assertGreater(result.partial_aruco_sample_px, 0)

    def test_wall_like_plane_is_rejected_by_normal_prior(self):
        depth, color = scene()
        strict = replace(
            config(),
            surface_roi=replace(config().surface_roi, board_normal_prior_max_error_deg=5.0),
        )
        wall = np.tile(np.linspace(450, 550, depth.shape[1], dtype=np.float32), (depth.shape[0], 1))
        with self.assertRaises(DepthExternalContourROIError):
            build_depth_external_contour_roi(
                wall, color.shape, strict,
                current_platform_roll_deg=0.0,
                current_platform_pitch_deg=0.0,
                commanded_platform_roll_deg=0.0,
                commanded_platform_pitch_deg=0.0,
            )

    def test_rgb_fallback_uses_depth_seed_and_excludes_board_noise(self):
        workspace = np.zeros((160, 160), dtype=np.uint8)
        workspace[10:150, 10:150] = 255
        board = workspace.copy()
        color = np.zeros((160, 160, 3), dtype=np.uint8)
        color[10:150, 10:150] = 25
        color[45:115, 45:115] = 135
        color[15:20, 15:20] = 135
        seed = np.zeros_like(workspace)
        seed[55:105, 55:105] = 255
        cfg = replace(config().surface_roi, rgb_fallback_lab_distance=20)
        result = build_rgb_seeded_roi(color, workspace, board, seed, cfg)
        self.assertIsNotNone(result)
        assert result is not None
        self.assertGreater(result.seed_overlap_px, 0)
        self.assertEqual(int(result.object_mask[80, 80]), 255)
        self.assertEqual(int(result.object_mask[17, 17]), 0)


if __name__ == "__main__":
    unittest.main()
