from __future__ import annotations

import math
from typing import Any


class OrbbecIntrinsicsError(RuntimeError):
    """Raised when SDK color intrinsics cannot safely describe the D2C grid."""


def select_color_intrinsic(camera_param: Any) -> tuple[Any, str]:
    """Select the v2 field first while retaining compatibility with older wrappers."""
    if hasattr(camera_param, "rgb_intrinsic"):
        return camera_param.rgb_intrinsic, "rgb_intrinsic"
    if hasattr(camera_param, "color_intrinsic"):
        return camera_param.color_intrinsic, "color_intrinsic"
    raise OrbbecIntrinsicsError(
        "OBCameraParam exposes neither rgb_intrinsic nor color_intrinsic"
    )


def build_d2c_intrinsics_payload(
    camera_param: Any, *, depth_grid_width: int, depth_grid_height: int,
) -> dict[str, Any]:
    intrinsic, source_field = select_color_intrinsic(camera_param)
    try:
        fx = float(intrinsic.fx)
        fy = float(intrinsic.fy)
        cx = float(intrinsic.cx)
        cy = float(intrinsic.cy)
        intrinsic_width = int(intrinsic.width)
        intrinsic_height = int(intrinsic.height)
        grid_width = int(depth_grid_width)
        grid_height = int(depth_grid_height)
    except (AttributeError, TypeError, ValueError) as exc:
        raise OrbbecIntrinsicsError(
            f"OBCameraParam.{source_field} is incomplete or invalid"
        ) from exc

    if not all(math.isfinite(value) for value in (fx, fy, cx, cy)):
        raise OrbbecIntrinsicsError("Orbbec color intrinsics must be finite")
    if fx <= 0 or fy <= 0 or intrinsic_width <= 0 or intrinsic_height <= 0:
        raise OrbbecIntrinsicsError(
            "Orbbec color intrinsics require fx>0, fy>0, width>0 and height>0"
        )
    if grid_width <= 0 or grid_height <= 0:
        raise OrbbecIntrinsicsError("D2C aligned depth grid dimensions must be positive")
    if (intrinsic_width, intrinsic_height) != (grid_width, grid_height):
        raise OrbbecIntrinsicsError(
            "D2C aligned depth grid does not match color intrinsics: "
            f"depth={grid_width}x{grid_height}, "
            f"intrinsics={intrinsic_width}x{intrinsic_height}"
        )

    intrinsic_source = f"Pipeline.get_camera_param().{source_field}"
    return {
        "schema_version": "orbbec_d2c_intrinsics_v1",
        "source": intrinsic_source,
        "intrinsic_source": intrinsic_source,
        "source_field": source_field,
        "depth_alignment": "color",
        "depth_unit": "mm",
        "structured_light_to_depth_pixel_transform": "rotate_180",
        "color_intrinsics": {
            "fx": fx, "fy": fy, "cx": cx, "cy": cy,
            "width": intrinsic_width, "height": intrinsic_height,
        },
    }
