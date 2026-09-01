from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class PoseInspectionResult:
    pose_id: str
    target_roll_deg: float | None
    target_pitch_deg: float | None
    best_z_cm: float | None
    z_search_samples: list[dict[str, Any]] = field(default_factory=list)
    anomaly_result: dict[str, Any] | None = None


@dataclass
class SystemInspectionResult:
    success: bool
    started_at: str
    finished_at: str
    final_status: str
    failed_state: str | None = None
    error_type: str | None = None
    error_message: str | None = None
    pose_results: list[PoseInspectionResult] = field(default_factory=list)
    state_history: list[str] = field(default_factory=list)
    mock: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
