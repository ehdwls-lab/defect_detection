from __future__ import annotations

import types
import unittest

from src.camera.orbbec_controller import OrbbecCameraController
from src.integration.orbbec_intrinsics import (
    OrbbecIntrinsicsError,
    build_d2c_intrinsics_payload,
    select_color_intrinsic,
)


def intrinsic(*, fx=600.0, fy=601.0, width=1280, height=800):
    return types.SimpleNamespace(
        fx=fx, fy=fy, cx=639.5, cy=399.5, width=width, height=height,
    )


class OrbbecIntrinsicsTests(unittest.TestCase):
    def test_v2_rgb_intrinsic_is_preferred_and_provenance_is_saved(self):
        rgb = intrinsic()
        old = intrinsic(fx=1.0)
        param = types.SimpleNamespace(rgb_intrinsic=rgb, color_intrinsic=old)
        selected, field = select_color_intrinsic(param)
        self.assertIs(selected, rgb)
        self.assertEqual(field, "rgb_intrinsic")
        payload = build_d2c_intrinsics_payload(
            param, depth_grid_width=1280, depth_grid_height=800,
        )
        self.assertEqual(
            payload["intrinsic_source"],
            "Pipeline.get_camera_param().rgb_intrinsic",
        )
        self.assertEqual(payload["source_field"], "rgb_intrinsic")
        self.assertEqual(payload["color_intrinsics"]["fx"], 600.0)

    def test_older_color_intrinsic_is_supported(self):
        param = types.SimpleNamespace(color_intrinsic=intrinsic())
        payload = build_d2c_intrinsics_payload(
            param, depth_grid_width=1280, depth_grid_height=800,
        )
        self.assertEqual(payload["source_field"], "color_intrinsic")

    def test_missing_sdk_fields_is_explicit_error(self):
        with self.assertRaisesRegex(OrbbecIntrinsicsError, "neither"):
            select_color_intrinsic(types.SimpleNamespace())

    def test_non_positive_intrinsics_are_rejected(self):
        for name, value in (("fx", 0), ("fy", -1), ("width", 0), ("height", 0)):
            values = {name: value}
            param = types.SimpleNamespace(rgb_intrinsic=intrinsic(**values))
            with self.subTest(field=name), self.assertRaises(OrbbecIntrinsicsError):
                build_d2c_intrinsics_payload(
                    param, depth_grid_width=1280, depth_grid_height=800,
                )

    def test_d2c_grid_mismatch_is_rejected(self):
        param = types.SimpleNamespace(rgb_intrinsic=intrinsic())
        with self.assertRaisesRegex(OrbbecIntrinsicsError, "does not match"):
            build_d2c_intrinsics_payload(
                param, depth_grid_width=640, depth_grid_height=400,
            )

    def test_camera_controller_exposes_validated_rgb_intrinsics_without_capture(self):
        controller = OrbbecCameraController()
        controller._pipeline = types.SimpleNamespace(
            get_camera_param=lambda: types.SimpleNamespace(
                rgb_intrinsic=intrinsic(),
            ),
        )
        result = controller.color_intrinsics(1280, 800)
        self.assertEqual(result.fx, 600.0)
        self.assertEqual(result.fy, 601.0)
        self.assertEqual(result.width, 1280)
        self.assertEqual(result.height, 800)
        self.assertEqual(
            result.source,
            "Pipeline.get_camera_param().rgb_intrinsic",
        )
        self.assertEqual(result.aligned_to, "color")


if __name__ == "__main__":
    unittest.main()
