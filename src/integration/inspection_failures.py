"""Explicit per-plane failures that are safe to isolate and continue past."""

from __future__ import annotations


class RecoverablePlaneInspectionError(RuntimeError):
    """A data/quality failure limited to the currently inspected plane."""

    def __init__(self, stage: str, reason: str) -> None:
        super().__init__(reason)
        self.stage = stage
        self.reason = reason


class FinalCaptureQualityError(RecoverablePlaneInspectionError):
    def __init__(self, reason: str, *, metadata: dict | None = None) -> None:
        super().__init__("FINAL_GEOMETRY_CAPTURE", reason)
        self.metadata = dict(metadata or {})


class FinalRGBCaptureError(RecoverablePlaneInspectionError):
    def __init__(self, reason: str) -> None:
        super().__init__("FINAL_RGB_CAPTURE", reason)


class AnomalyInputDataError(RecoverablePlaneInspectionError):
    def __init__(self, reason: str) -> None:
        super().__init__("ANOMALY_INFERENCE", reason)
