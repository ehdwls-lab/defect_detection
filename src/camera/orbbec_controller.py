from __future__ import annotations

import time

from src.camera.controller import RGBDepthFrame
from src.config import InspectionConfig


class OrbbecCameraController:
    """Lifecycle wrapper around the verified Gemini 336L capture helpers.

    SDK and experimental helper imports are lazy, so mock/system imports do not
    initialize camera hardware. Surface processing remains in ``src.core``.
    """

    def __init__(self, config: InspectionConfig | None = None, warmup_frames: int = 30) -> None:
        self.config = config or InspectionConfig.default()
        self.warmup_frames = max(0, int(warmup_frames))
        self._pipeline = None
        self._align_filter = None

    def start(self) -> None:
        if self._pipeline is not None:
            return
        from pyorbbecsdk import AlignFilter, Config, OBFrameAggregateOutputMode, OBStreamType, Pipeline
        from src.test_surface_only_pose_inspection import (
            configure_camera, find_color_profile, find_depth_profile, wait_for_aligned_pair,
        )
        pipeline = Pipeline()
        sdk_config = Config()
        sdk_config.enable_stream(find_color_profile(pipeline))
        sdk_config.enable_stream(find_depth_profile(pipeline))
        try:
            sdk_config.set_frame_aggregate_output_mode(OBFrameAggregateOutputMode.FULL_FRAME_REQUIRE)
        except Exception:
            pass
        pipeline.start(sdk_config)
        align_filter = AlignFilter(align_to_stream=OBStreamType.COLOR_STREAM)
        try:
            configure_camera(pipeline.get_device(), type("Args", (), {
                "brightness": self.config.camera.brightness,
                "exposure": self.config.camera.exposure,
                "gain": self.config.camera.gain,
                "white_balance": self.config.camera.white_balance,
                "depth_exposure": self.config.depth.exposure,
                "depth_gain": self.config.depth.gain,
                "depth_median_frames": self.config.depth.median_frames,
            })())
            for _ in range(self.warmup_frames):
                wait_for_aligned_pair(pipeline, align_filter)
        except Exception:
            pipeline.stop()
            raise
        self._pipeline = pipeline
        self._align_filter = align_filter

    def capture(self) -> RGBDepthFrame:
        if self._pipeline is None or self._align_filter is None:
            raise RuntimeError("Orbbec camera is not started")
        from src.test_surface_only_pose_inspection import wait_for_aligned_pair
        color_bgr, depth_mm = wait_for_aligned_pair(self._pipeline, self._align_filter)
        return RGBDepthFrame(color_bgr=color_bgr, depth_mm=depth_mm, timestamp=time.time())

    def close(self) -> None:
        if self._pipeline is not None:
            self._pipeline.stop()
        self._pipeline = None
        self._align_filter = None
