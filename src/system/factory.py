from __future__ import annotations

from src.anomaly.detector import MockAnomalyDetector
from src.conveyor.mock_controller import MockConveyorController
from src.inspection.automatic_z_search import AutomaticZSearch
from src.inspection.mock_quality_sampler import MockQualitySampler
from src.inspection.surface_inspector import MockSurfaceInspector
from src.integration.mock_pose_planner import MockPosePlanner
from src.integration.structured_light_adapter import StructuredLightAdapter
from src.integration.structured_light_runner import MockStructuredLightRunner
from src.platform.mock_controller import MockPlatformController
from src.platform.types import PlatformLimits
from src.system.controller import SystemController


class HardwareModeNotReadyError(RuntimeError):
    pass


def build_system(mode: str) -> SystemController:
    if mode == "hardware":
        raise HardwareModeNotReadyError("Hardware mode is not ready; no fallback to mock was performed")
    if mode != "mock":
        raise ValueError(f"Unsupported mode: {mode}")
    # These are test-only limits and safe Z values for the in-memory mock.
    limits = PlatformLimits(z_min_cm=0.0, z_max_cm=40.0, roll_min_deg=-30.0,
                            roll_max_deg=30.0, pitch_min_deg=-30.0, pitch_max_deg=30.0)
    return SystemController(
        conveyor=MockConveyorController(),
        structured_light_runner=MockStructuredLightRunner(),
        structured_light_adapter=StructuredLightAdapter,
        pose_planner=MockPosePlanner(),
        platform=MockPlatformController(limits=limits),
        automatic_z_search=AutomaticZSearch(),
        quality_sampler=MockQualitySampler(),
        surface_inspector=MockSurfaceInspector(),
        anomaly_detector=MockAnomalyDetector(),
        safe_z_cm=18.0,
        mock=True,
    )
