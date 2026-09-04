"""Deterministic replay timeline for a completed production run."""

from __future__ import annotations

from src.ui.events import InspectionUIEvent, InspectionUIEventType as T


REPLAY_STAGES = ("IN", "3D SCAN", "POSE", "INSPECTION", "ANALYSIS", "JUDGEMENT", "OUT")


def build_replay_events(duration_s: float = 11.0) -> tuple[InspectionUIEvent, ...]:
    if duration_s <= 0:
        raise ValueError("replay duration must be positive")
    scale = duration_s / 11.0
    definitions = (
        (0, T.CYCLE_STARTED, "IN"), (1, T.STRUCTURED_LIGHT_READY, "3D SCAN"),
        (3, T.POSE_SELECTED, "POSE"), (5, T.ROI_READY, "INSPECTION"),
        (7, T.ANOMALY_RESULT, "ANALYSIS"), (9, T.POSE_COMPLETE, "JUDGEMENT"),
        (11, T.CYCLE_COMPLETE, "OUT"),
    )
    return tuple(InspectionUIEvent(kind, second * scale, {"stage": stage})
                 for second, kind, stage in definitions)


class ReplayCursor:
    def __init__(self, duration_s: float = 11.0):
        self.events = build_replay_events(duration_s); self.index = -1

    def restart(self) -> None: self.index = -1

    def advance(self, elapsed_s: float) -> tuple[InspectionUIEvent, ...]:
        start = self.index + 1
        while self.index + 1 < len(self.events) and self.events[self.index + 1].elapsed_s <= elapsed_s:
            self.index += 1
        return self.events[start:self.index + 1]

