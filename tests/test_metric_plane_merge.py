from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from src.integration.metric_plane_merge import (
    METRIC_PLANE_COPLANAR_DISTANCE_MM,
    METRIC_PLANE_NORMAL_THRESHOLD_DEG,
    MetricPlaneCandidate,
    group_metric_plane_candidates,
    merge_metric_physical_planes_in_pose_json,
    refit_metric_plane_group,
)
from src.integration.metric_pose import MetricFitConfig, camera_tilt_degrees
from src.integration.metric_pose_postprocess import postprocess_metric_pose_json


class MetricPlaneMergeTests(unittest.TestCase):
    def setUp(self):
        self.fit_config = MetricFitConfig(
            min_depth_mm=1, max_depth_mm=5000, ransac_threshold_mm=1.0,
            ransac_iterations=100, min_points=30, max_points=20000,
        )

    @staticmethod
    def candidate(index, *, z=500.0, slope_x=0.1, x_start=0.0,
                  count=200, dominant=False):
        x = np.linspace(x_start, x_start + 40.0, count)
        y = np.linspace(-20.0, 20.0, count)
        # Non-collinear deterministic grid.
        x, y = np.meshgrid(x[::10], y[:10])
        x, y = x.ravel(), y.ravel()
        zz = z + slope_x * x
        xyz = np.column_stack((x, y, zz))
        normal = np.array([slope_x, 0.0, -1.0])
        normal /= np.linalg.norm(normal)
        pixels = np.column_stack((np.arange(len(xyz)) + index * 10000,
                                  np.full(len(xyz), index)))
        return MetricPlaneCandidate(
            index, f"Raw {index}", dominant, len(xyz), pixels, xyz,
            normal, xyz.mean(axis=0),
        )

    def test_threshold_constants(self):
        self.assertEqual(METRIC_PLANE_NORMAL_THRESHOLD_DEG, 2.0)
        self.assertEqual(METRIC_PLANE_COPLANAR_DISTANCE_MM, 5.0)

    def test_coplanar_similar_normals_merge(self):
        first = self.candidate(0, x_start=0)
        second = self.candidate(1, x_start=45)
        self.assertEqual(len(group_metric_plane_candidates([first, second])), 1)

    def test_parallel_planes_at_different_depth_do_not_merge(self):
        first = self.candidate(0, z=500)
        second = self.candidate(1, z=520)
        self.assertEqual(len(group_metric_plane_candidates([first, second])), 2)

    def test_three_duplicate_planes_merge_into_one_group(self):
        planes = [self.candidate(index, x_start=index * 45) for index in range(3)]
        groups = group_metric_plane_candidates(planes)
        self.assertEqual(len(groups), 1)
        self.assertEqual(len(groups[0]), 3)

    def test_union_xyz_is_refit_instead_of_averaging_angles(self):
        small = self.candidate(0, slope_x=0.08, count=100)
        large = self.candidate(1, slope_x=0.10, count=1000)
        raw_angles = [camera_tilt_degrees(item.normal)[0] for item in (small, large)]
        result = refit_metric_plane_group([small, large], 0, self.fit_config)
        arithmetic_angle_average = sum(raw_angles) / 2
        self.assertGreater(abs(result["camera_roll_deg"] - arithmetic_angle_average), 0.2)
        self.assertGreater(result["total_depth_points"], len(small.xyz_mm))

    def test_dominant_and_source_provenance_survive_merge(self):
        group = [self.candidate(2, dominant=False), self.candidate(7, dominant=True)]
        result = refit_metric_plane_group(group, 3, self.fit_config)
        self.assertTrue(result["dominant"])
        self.assertEqual(result["physical_plane_index"], 3)
        self.assertEqual(result["merged_source_plane_indices"], [2, 7])
        self.assertEqual(result["merged_source_plane_names"], ["Raw 2", "Raw 7"])

    def test_pose_json_records_physical_plane_provenance(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            width, height = 40, 30
            depth = np.full((height, width), 500.0, dtype=np.float32)
            depth_path = root / "depth.npy"
            np.save(depth_path, depth)
            intrinsics_path = root / "intrinsics.json"
            intrinsics_path.write_text(json.dumps({
                "fx": 100.0, "fy": 100.0, "cx": 20.0, "cy": 15.0,
                "width": width, "height": height,
                "source": "test", "depth_alignment": "color",
            }), encoding="utf-8")
            pixels_a = np.array([(u, v) for v in range(5, 25) for u in range(2, 18)])
            pixels_b = np.array([(u, v) for v in range(5, 25) for u in range(22, 38)])
            memberships_path = root / "memberships.npz"
            np.savez_compressed(memberships_path, plane_a=pixels_a, plane_b=pixels_b)
            pose_path = root / "pose.json"
            planes = []
            for index, (name, key, pixels, dominant) in enumerate((
                ("A", "plane_a", pixels_a, False),
                ("B", "plane_b", pixels_b, True),
            )):
                xyz_z = 500.0
                planes.append({
                    "source_plane_index": index, "plane_name": name,
                    "dominant": dominant, "points_count": len(pixels),
                    "roll_deg": 77.0, "pitch_deg": -66.0,
                    "raw_roll_deg": 77.0, "raw_pitch_deg": -66.0,
                    "pixel_membership": {"sidecar_key": key},
                    "metric_pose": {
                        "physical_metric": True, "status": "METRIC_VALID",
                        "reachable": False,
                        "reject_reason": "platform axis/sign contract is unresolved",
                        "normal_xyz": [0.0, 0.0, -1.0],
                        "center_xyz_mm": [0.0, 0.0, xyz_z],
                        "camera_roll_deg": 0.0, "camera_pitch_deg": 0.0,
                    },
                })
            pose_path.write_text(json.dumps({
                "planes": planes,
                "metric_pose_contract": {
                    "depth_path": str(depth_path),
                    "intrinsics_path": str(intrinsics_path),
                    "plane_membership_path": str(memberships_path),
                },
            }), encoding="utf-8")
            postprocess_metric_pose_json(
                pose_path, fit_config=self.fit_config,
                current_platform_roll_deg=1.0,
                current_platform_pitch_deg=2.0,
            )
            saved = json.loads(pose_path.read_text(encoding="utf-8"))
            self.assertEqual(saved["raw_plane_count"], 2)
            self.assertEqual(saved["metric_physical_plane_count"], 1)
            physical = saved["planes"][0]
            self.assertEqual(physical["merged_source_plane_indices"], [0, 1])
            self.assertEqual(physical["merged_source_plane_names"], ["A", "B"])
            self.assertTrue(physical["dominant"])
            self.assertAlmostEqual(physical["camera_roll_deg"], 0.0, places=6)
            self.assertNotEqual(physical["camera_roll_deg"], 77.0)
            metric = physical["metric_pose"]
            self.assertEqual(metric["calibration_id"], "camera_platform_rp_20260903")
            self.assertEqual(metric["current_platform_roll_deg"], 1.0)
            self.assertEqual(metric["current_platform_pitch_deg"], 2.0)
            self.assertTrue(metric["reachable"])
            self.assertIsNone(metric["reject_reason"])
            self.assertNotIn("axis/sign contract is unresolved", json.dumps(saved))


if __name__ == "__main__":
    unittest.main()
