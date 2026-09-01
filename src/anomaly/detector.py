from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

from src.inspection.surface_inspector import SurfaceInspectionResult


@dataclass(frozen=True)
class AnomalyResult:
    status: str
    is_mock: bool
    score: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class AnomalyDetector(Protocol):
    def inspect(self, surface: SurfaceInspectionResult) -> AnomalyResult: ...


class MockAnomalyDetector:
    def inspect(self, surface: SurfaceInspectionResult) -> AnomalyResult:
        return AnomalyResult(
            status="MOCK_NORMAL",
            is_mock=True,
            metadata={"source": "mock", "pose_id": surface.pose_id},
        )
