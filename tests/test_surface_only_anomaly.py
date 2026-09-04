from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np
import torch

from src.anomaly.detector import ProductionAnomalyConfig, ProductionAnomalyDetector
from src.camera.controller import RGBDepthFrame
from src.core.surface_geometry import SurfaceGeometryResult, extract_surface_geometry
from src.infer_anomaly import (
    inspect_image, load_bgr_image, make_patch_tensor, read_validation_entries,
    select_patch_positions, validation_region_positions,
)
from src.integration.inspection_failures import AnomalyInputDataError
from src.patch_dataset import PatchDataset


PREPROCESSING = {"gamma": 1.0, "clahe_clip": 1.0, "unsharp_amount": 0.0}


class ZeroModel(torch.nn.Module):
    def forward(self, value):
        return torch.zeros_like(value)


class Telemetry:
    roll_deg = 1.0
    pitch_deg = 2.0
    z_cm = 18.0


def geometry(mask: np.ndarray) -> SurfaceGeometryResult:
    area = int(np.count_nonzero(mask))
    return SurfaceGeometryResult(
        object_mask=mask.copy(), surface_mask=mask.copy(), patches=(),
        object_area_px=area, surface_area_px=area, surface_ratio=1.0,
        depth_valid_ratio=.8, plane_inlier_ratio=.9, plane_residual=.5,
        fov_edge_contact=False,
    )


def detector() -> ProductionAnomalyDetector:
    instance = ProductionAnomalyDetector(ProductionAnomalyConfig(
        Path("unused.pth"), Path("unused.csv"), surface_patch_coverage=1.0,
    ))
    instance._runtime = {
        "model": ZeroModel(), "device": torch.device("cpu"),
        "patch_size": 64, "stride": 32, "preprocessing": PREPROCESSING,
        "validation": np.zeros(4, dtype=np.float32), "threshold": .1,
        "inspect_image": inspect_image, "validation_region_patch_count": 4,
        "validation_full_image_count": 0,
    }
    return instance


class SurfaceOnlyInferenceTests(unittest.TestCase):
    def test_surface_mask_none_preserves_full_grid(self):
        image = np.zeros((128, 128, 3), dtype=np.uint8)
        tensor, positions = make_patch_tensor(image, 64, 32, PREPROCESSING)
        self.assertEqual(tensor.shape[0], 9)
        self.assertEqual(len(positions), 9)

    def test_surface_coverage_selects_only_central_patch(self):
        mask = np.zeros((128, 128), dtype=np.uint8)
        mask[32:96, 32:96] = 255
        positions = select_patch_positions(
            (128, 128, 3), 64, 32, surface_mask=mask,
            min_surface_coverage=.8,
        )
        self.assertEqual(positions, [(32, 32)])
        tensor, tensor_positions = make_patch_tensor(
            np.zeros((128, 128, 3), dtype=np.uint8), 64, 32, PREPROCESSING,
            surface_mask=mask, min_surface_coverage=.8,
        )
        self.assertEqual(tensor.shape[0], 1)
        self.assertEqual(tensor_positions, [(32, 32)])

    def test_full_coverage_accepts_exact_64_square(self):
        mask = np.full((64, 64), 255, dtype=np.uint8)
        self.assertEqual(select_patch_positions(
            (64, 64, 3), 64, 32, surface_mask=mask,
            min_surface_coverage=1.0,
        ), [(0, 0)])

    def test_99_percent_coverage_is_rejected_at_full_purity(self):
        mask = np.full((64, 64), 255, dtype=np.uint8)
        mask.flat[:41] = 0
        self.assertEqual(select_patch_positions(
            (64, 64, 3), 64, 32, surface_mask=mask,
            min_surface_coverage=1.0,
        ), [])

    def test_80_percent_coverage_is_rejected_at_full_purity(self):
        mask = np.full((64, 64), 255, dtype=np.uint8)
        mask.flat[:819] = 0
        self.assertEqual(select_patch_positions(
            (64, 64, 3), 64, 32, surface_mask=mask,
            min_surface_coverage=1.0,
        ), [])

    def test_patch_touching_boundary_is_accepted_when_every_pixel_is_inside(self):
        mask = np.zeros((96, 96), dtype=np.uint8)
        mask[:64, :64] = 255
        positions = select_patch_positions(
            (96, 96, 3), 64, 32, surface_mask=mask,
            min_surface_coverage=1.0,
        )
        self.assertEqual(positions, [(0, 0)])

    def test_one_outside_pixel_rejects_patch(self):
        mask = np.full((64, 64), 255, dtype=np.uint8)
        mask[63, 63] = 0
        self.assertEqual(select_patch_positions(
            (64, 64, 3), 64, 32, surface_mask=mask,
            min_surface_coverage=1.0,
        ), [])

    def test_every_selected_trapezoid_patch_is_pixel_pure(self):
        mask = np.zeros((256, 256), dtype=np.uint8)
        cv2.fillConvexPoly(
            mask,
            np.array([[48, 24], [208, 48], [224, 224], [24, 208]], np.int32),
            255,
        )
        positions = select_patch_positions(
            (256, 256, 3), 64, 32, surface_mask=mask,
            min_surface_coverage=1.0,
        )
        self.assertTrue(positions)
        for x, y in positions:
            self.assertTrue(np.all(mask[y:y + 64, x:x + 64] > 0))

    def _inspect(self, image: np.ndarray, mask: np.ndarray, root: Path):
        frame = RGBDepthFrame(image, np.full(mask.shape, 500, dtype=np.float32), 1.0)
        with patch("src.anomaly.detector.extract_surface_geometry", return_value=geometry(mask)):
            return detector().inspect_frame(
                frame, pose_id="plane", output_directory=root,
                rgb_path="rgb.png", depth_path="depth.npy", ir_path=None,
                platform_telemetry=Telemetry(),
            )

    def test_high_anomaly_background_outside_surface_does_not_change_classification(self):
        image = np.full((128, 128, 3), 255, dtype=np.uint8)
        image[32:96, 32:96] = 0
        mask = np.zeros((128, 128), dtype=np.uint8)
        mask[32:96, 32:96] = 255
        with tempfile.TemporaryDirectory() as temporary:
            result = self._inspect(image, mask, Path(temporary))
        self.assertEqual(result.classification, "NORMAL")
        self.assertEqual(result.metadata["selected_surface_patch_count"], 1)
        self.assertEqual(result.metadata["total_grid_patch_count"], 9)

    def test_surface_patch_above_threshold_is_defect_and_artifacts_are_masked(self):
        image = np.zeros((128, 128, 3), dtype=np.uint8)
        image[32:96, 32:96] = 255
        mask = np.zeros((128, 128), dtype=np.uint8)
        mask[32:96, 32:96] = 255
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            result = self._inspect(image, mask, root)
            heatmap = cv2.imread(str(root / "anomaly_heatmap.png"))
            overlay = cv2.imread(str(root / "anomaly_overlay.png"))
            self.assertTrue((root / "surface_mask.png").is_file())
            self.assertTrue((root / "object_mask.png").is_file())
            self.assertTrue((root / "surface_patch_overlay.png").is_file())
        self.assertEqual(result.classification, "DEFECT")
        self.assertTrue(np.array_equal(heatmap[0, 0], [0, 0, 0]))
        self.assertTrue(np.array_equal(overlay[0, 0], image[0, 0]))
        self.assertTrue(result.metadata["surface_only_inference"])
        self.assertEqual(result.metadata["surface_patch_coverage_threshold"], 1.0)
        for key in (
            "surface_area_px", "object_area_px", "surface_ratio",
            "depth_valid_ratio", "plane_inlier_ratio", "plane_residual",
        ):
            self.assertIn(key, result.metadata)

    def test_expanded_inspection_mask_selects_patch_outside_surface_mask(self):
        image = np.zeros((128, 128, 3), dtype=np.uint8)
        surface = np.zeros((128, 128), dtype=np.uint8)
        surface[32:96, 32:96] = 255
        inspection = np.zeros_like(surface)
        inspection[32:96, 0:96] = 255
        frame = RGBDepthFrame(image, np.zeros((1, 1), dtype=np.float32), 1)
        with tempfile.TemporaryDirectory() as temporary:
            result = detector().inspect_frame(
                frame, pose_id="plane", output_directory=temporary,
                rgb_path="rgb.png", depth_path="depth.npy", ir_path=None,
                platform_telemetry=Telemetry(), surface_geometry=geometry(surface),
                inspection_mask=inspection,
                final_capture_metadata={
                    "anomaly_roi_type": "aruco_depth_rgb_hybrid",
                },
            )
        self.assertEqual(result.metadata["anomaly_roi_type"], "aruco_depth_rgb_hybrid")
        self.assertEqual(result.metadata["selected_inspection_patch_count"], 2)
        self.assertEqual(result.metadata["selected_surface_patch_count"], 2)

    def test_patch_outside_inspection_mask_is_excluded_at_full_coverage(self):
        image = np.zeros((128, 128, 3), dtype=np.uint8)
        surface = np.zeros((128, 128), dtype=np.uint8)
        surface[16:112, 16:112] = 255
        inspection = np.zeros_like(surface)
        inspection[32:96, 32:96] = 255
        frame = RGBDepthFrame(image, np.zeros((1, 1), dtype=np.float32), 1)
        with tempfile.TemporaryDirectory() as temporary:
            result = detector().inspect_frame(
                frame, pose_id="plane", output_directory=temporary,
                rgb_path="rgb.png", depth_path="depth.npy", ir_path=None,
                platform_telemetry=Telemetry(), surface_geometry=geometry(surface),
                inspection_mask=inspection,
            )
        self.assertEqual(result.metadata["selected_inspection_patch_count"], 1)
        self.assertEqual(result.metadata["surface_patch_coverage_threshold"], 1.0)

    def test_zero_selected_inspection_patches_is_recoverable(self):
        image = np.zeros((128, 128, 3), dtype=np.uint8)
        surface = np.zeros((128, 128), dtype=np.uint8)
        surface[16:112, 16:112] = 255
        inspection = np.zeros_like(surface)
        inspection[60:68, 60:68] = 255
        with tempfile.TemporaryDirectory() as temporary, self.assertRaisesRegex(
            AnomalyInputDataError, "selected_surface_patch_count == 0",
        ):
            detector().inspect_frame(
                RGBDepthFrame(image, np.zeros((1, 1), dtype=np.float32), 1),
                pose_id="plane", output_directory=temporary,
                rgb_path="rgb.png", depth_path="depth.npy", ir_path=None,
                platform_telemetry=Telemetry(), surface_geometry=geometry(surface),
                inspection_mask=inspection,
            )

    def test_precomputed_accepted_geometry_is_reused_with_capture_metadata(self):
        image = np.zeros((128, 128, 3), dtype=np.uint8)
        mask = np.zeros((128, 128), dtype=np.uint8)
        mask[32:96, 32:96] = 255
        accepted_geometry = geometry(mask)
        frame = RGBDepthFrame(
            image, np.full((128, 128), 500, dtype=np.float32), 1,
        )
        capture_metadata = {
            "final_capture_attempts": 3,
            "final_capture_accepted_attempt": 3,
            "final_capture_depth_valid_ratio": .27,
        }
        with tempfile.TemporaryDirectory() as temporary, patch(
            "src.anomaly.detector.extract_surface_geometry",
        ) as extract:
            result = detector().inspect_frame(
                frame, pose_id="plane", output_directory=temporary,
                rgb_path="rgb.png", depth_path="depth.npy", ir_path=None,
                platform_telemetry=Telemetry(),
                surface_geometry=accepted_geometry,
                final_capture_metadata=capture_metadata,
            )
        extract.assert_not_called()
        self.assertEqual(result.metadata["final_capture_attempts"], 3)
        self.assertEqual(result.metadata["final_capture_accepted_attempt"], 3)
        self.assertEqual(result.metadata["final_capture_depth_valid_ratio"], .27)

    def test_led_on_depth_is_ignored_when_frozen_geometry_is_supplied(self):
        image = np.zeros((128, 128, 3), dtype=np.uint8)
        mask = np.zeros((128, 128), dtype=np.uint8)
        mask[32:96, 32:96] = 255
        frame = RGBDepthFrame(image, np.zeros((1, 1), dtype=np.float32), 2.0)
        with tempfile.TemporaryDirectory() as temporary, patch(
            "src.anomaly.detector.extract_surface_geometry",
        ) as extract:
            result = detector().inspect_frame(
                frame, pose_id="plane", output_directory=temporary,
                rgb_path="final_rgb.png", depth_path="final_depth.npy", ir_path=None,
                platform_telemetry=Telemetry(), surface_geometry=geometry(mask),
                geometry_capture_metadata={"geometry_accepted_attempt": 2},
            )
        extract.assert_not_called()
        self.assertEqual(result.classification, "NORMAL")
        self.assertEqual(result.judgement, "OK")
        self.assertEqual(result.metadata["geometry_accepted_attempt"], 2)

    def test_frozen_mask_rgb_shape_mismatch_is_recoverable_without_resize(self):
        image = np.zeros((96, 128, 3), dtype=np.uint8)
        mask = np.zeros((128, 128), dtype=np.uint8)
        mask[32:96, 32:96] = 255
        with tempfile.TemporaryDirectory() as temporary, self.assertRaisesRegex(
            AnomalyInputDataError, "frozen object_mask / final RGB shape mismatch",
        ):
            detector().inspect_frame(
                RGBDepthFrame(image, np.zeros((1, 1), dtype=np.float32), 2.0),
                pose_id="plane", output_directory=temporary,
                rgb_path="final_rgb.png", depth_path="final_depth.npy", ir_path=None,
                platform_telemetry=Telemetry(), surface_geometry=geometry(mask),
            )

    def test_zero_selected_surface_patches_is_recoverable(self):
        image = np.zeros((128, 128, 3), dtype=np.uint8)
        mask = np.zeros((128, 128), dtype=np.uint8)
        mask[60:68, 60:68] = 255
        with tempfile.TemporaryDirectory() as temporary, patch(
            "src.anomaly.detector.extract_surface_geometry", return_value=geometry(mask),
        ), self.assertRaisesRegex(AnomalyInputDataError, "selected_surface_patch_count"):
            detector().inspect_frame(
                RGBDepthFrame(image, np.full((128, 128), 500, dtype=np.float32), 1),
                pose_id="plane", output_directory=temporary, rgb_path="rgb.png",
                depth_path="depth.npy", ir_path=None, platform_telemetry=Telemetry(),
            )

    def test_rgb_depth_shape_mismatch_is_recoverable_frame_data_error(self):
        frame = RGBDepthFrame(
            np.zeros((128, 128, 3), dtype=np.uint8),
            np.zeros((64, 64), dtype=np.float32), 1,
        )
        with tempfile.TemporaryDirectory() as temporary, self.assertRaisesRegex(
            AnomalyInputDataError, "shape mismatch",
        ):
            detector().inspect_frame(
                frame, pose_id="plane", output_directory=temporary,
                rgb_path="rgb.png", depth_path="depth.npy", ir_path=None,
                platform_telemetry=Telemetry(),
            )

    def test_validation_regions_match_patch_dataset_coordinate_contract(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            image_path = root / "normal.png"
            region_path = root / "normal.json"
            manifest_path = root / "val.csv"
            cv2.imwrite(str(image_path), np.zeros((128, 128, 3), dtype=np.uint8))
            region_path.write_text(json.dumps({
                "regions": [{"x1": 16, "y1": 8, "x2": 112, "y2": 104}],
            }), encoding="utf-8")
            with manifest_path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(
                    handle, fieldnames=("path", "split", "label", "regions"),
                )
                writer.writeheader()
                writer.writerow({
                    "path": str(image_path), "split": "val", "label": "normal",
                    "regions": str(region_path),
                })
            entry = read_validation_entries(manifest_path)[0]
            positions = validation_region_positions(entry, (128, 128, 3), 64, 32)
            dataset = PatchDataset(
                manifest_path=manifest_path, split="val", patch_size=64, stride=32,
            )
        self.assertEqual(positions, [(16, 8), (48, 8), (16, 40), (48, 40)])
        self.assertEqual(
            [(left, top) for _, _, left, top in dataset.samples], positions,
        )

    def test_validation_mask_precedes_regions_and_uses_production_selector(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            mask_path = root / "mask.png"
            region_path = root / "regions.json"
            mask = np.zeros((128, 128), dtype=np.uint8)
            mask[:64, :64] = 255
            cv2.imwrite(str(mask_path), mask)
            region_path.write_text(json.dumps({
                "regions": [{"x1": 0, "y1": 0, "x2": 128, "y2": 128}],
            }), encoding="utf-8")
            entry = {"path": None, "mask": mask_path, "regions": region_path}
            with patch(
                "src.infer_anomaly.select_patch_positions",
                wraps=select_patch_positions,
            ) as selector:
                positions = validation_region_positions(
                    entry, (128, 128, 3), 64, 32,
                )
        self.assertEqual(positions, [(0, 0)])
        selector.assert_called_once()
        self.assertEqual(selector.call_args.kwargs["min_surface_coverage"], 1.0)

    def test_gray_v1_validation_masks_select_exactly_152_patches(self):
        manifest = Path("data/manifests/gray_v1/val.csv").resolve()
        counts = []
        for entry in read_validation_entries(manifest):
            image_path = entry["path"]
            assert image_path is not None
            image = load_bgr_image(image_path)
            counts.append(len(validation_region_positions(
                entry, image.shape, 64, 32,
            )))
        self.assertEqual(counts, [81, 71])
        self.assertEqual(sum(counts), 152)

    def test_validation_mask_missing_or_malformed_is_explicit_error(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            missing = root / "missing.png"
            with self.assertRaisesRegex(FileNotFoundError, "validation mask PNG"):
                validation_region_positions(
                    {"path": None, "mask": missing, "regions": None},
                    (128, 128, 3), 64, 32,
                )
            malformed = root / "malformed.png"
            malformed.write_text("not a png", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "읽을 수 없습니다"):
                validation_region_positions(
                    {"path": None, "mask": malformed, "regions": None},
                    (128, 128, 3), 64, 32,
                )

    def test_validation_without_mask_or_regions_keeps_full_image_fallback(self):
        self.assertIsNone(validation_region_positions(
            {"path": None, "mask": None, "regions": None},
            (128, 128, 3), 64, 32,
        ))

    def test_production_threshold_uses_only_gray_v1_mask_selected_scores(self):
        with tempfile.TemporaryDirectory() as temporary:
            checkpoint_path = Path(temporary) / "model.pth"
            checkpoint_path.touch()
            manifest = Path("data/manifests/gray_v1/val.csv").resolve()
            instance = ProductionAnomalyDetector(ProductionAnomalyConfig(
                checkpoint_path, manifest, threshold_percentile=99.0,
                score_method="mean_mse", surface_patch_coverage=1.0,
            ))
            emitted: list[float] = []

            def fake_inspect(model, image, patch_size, stride, batch_size, device,
                             preprocessing, *, allowed_positions=None, **kwargs):
                del model, image, patch_size, stride, batch_size, device
                del preprocessing, kwargs
                assert allowed_positions is not None
                values = np.arange(
                    len(emitted), len(emitted) + len(allowed_positions),
                    dtype=np.float32,
                )
                emitted.extend(values.tolist())
                return {"mean_mse": values}, allowed_positions, None, None, None

            checkpoint = {
                "config": {
                    "in_channels": 3, "latent_channels": 1,
                    "patch_size": 64, "stride": 32,
                },
                "patch_size": 64, "stride": 32,
            }
            with patch("torch.load", return_value=checkpoint), patch(
                "src.infer_anomaly.load_model", return_value=ZeroModel(),
            ), patch("src.infer_anomaly.inspect_image", side_effect=fake_inspect):
                runtime = instance._load()
        self.assertEqual(runtime["validation_region_patch_count"], 152)
        self.assertEqual(runtime["validation_full_image_count"], 0)
        self.assertEqual(len(runtime["validation"]), 152)
        self.assertAlmostEqual(runtime["threshold"], float(np.percentile(
            np.arange(152, dtype=np.float32), 99.0,
        )))

    def test_shared_surface_geometry_extracts_object_from_depth(self):
        depth = np.full((800, 1280), 500, dtype=np.float32)
        depth[300:500, 500:800] = 450
        result = extract_surface_geometry(depth, (800, 1280, 3), detector().inspection_config)
        self.assertGreater(result.object_area_px, 0)
        self.assertGreater(result.surface_area_px, 0)
        self.assertIsNotNone(result.surface_ratio)


if __name__ == "__main__":
    unittest.main()
