from __future__ import annotations

import sys
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Protocol

import numpy as np


class ProjectorState(str, Enum):
    CLOSED = "CLOSED"
    BLACK = "BLACK"
    PHASE = "PHASE"


class ProjectorController(Protocol):
    state: ProjectorState

    def open(self) -> None: ...
    def show_black(self) -> None: ...
    def show_phase(self, phase_name: str) -> None: ...
    def close(self) -> None: ...


class OpenCVProjectorController:
    """Persistent fullscreen window using the verified production phase generator."""

    PHASE_ORDER = ("000", "090", "180", "270")

    def __init__(self, monitor: dict[str, Any], *, window_name: str = "STRUCTURED LIGHT PROJECTOR",
                 display_delay_ms: int = 1) -> None:
        self.monitor = monitor
        self.window_name = window_name
        self.display_delay_ms = display_delay_ms
        self.state = ProjectorState.CLOSED
        self.current_phase: str | None = None
        self._patterns: dict[str, np.ndarray] | None = None

    @staticmethod
    def _generator() -> Callable[[int, int], dict[str, np.ndarray]]:
        subsystem = Path(__file__).resolve().parents[2] / "서영 파트 파일"
        if str(subsystem) not in sys.path:
            sys.path.insert(0, str(subsystem))
        from structured_light_projector import production_phase_patterns
        return production_phase_patterns

    def open(self) -> None:
        if self.state is not ProjectorState.CLOSED:
            return
        import cv2
        width, height = int(self.monitor["w"]), int(self.monitor["h"])
        self._patterns = self._generator()(width, height)
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.moveWindow(self.window_name, int(self.monitor["x"]), int(self.monitor["y"]))
        cv2.setWindowProperty(self.window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        self.state = ProjectorState.BLACK
        self.show_black()

    def _show(self, image: np.ndarray) -> None:
        if self.state is ProjectorState.CLOSED:
            raise RuntimeError("projector window is not open")
        import cv2
        cv2.imshow(self.window_name, image)
        cv2.waitKey(self.display_delay_ms)

    def show_black(self) -> None:
        if self.state is ProjectorState.CLOSED:
            raise RuntimeError("projector window is not open")
        image = np.zeros((int(self.monitor["h"]), int(self.monitor["w"]), 3), dtype=np.uint8)
        self._show(image)
        self.state = ProjectorState.BLACK
        self.current_phase = None

    def show_phase(self, phase_name: str) -> None:
        if phase_name not in self.PHASE_ORDER:
            raise ValueError(f"unsupported production phase: {phase_name}")
        if self._patterns is None:
            raise RuntimeError("projector window is not open")
        self._show(self._patterns[phase_name])
        self.state = ProjectorState.PHASE
        self.current_phase = phase_name

    def show_white_diagnostic(self) -> None:
        if self.state is ProjectorState.CLOSED:
            raise RuntimeError("projector window is not open")
        image = np.full((int(self.monitor["h"]), int(self.monitor["w"]), 3), 255, dtype=np.uint8)
        self._show(image)

    def close(self) -> None:
        if self.state is ProjectorState.CLOSED:
            return
        try:
            self.show_black()
        finally:
            import cv2
            cv2.destroyWindow(self.window_name)
            self.state = ProjectorState.CLOSED
            self.current_phase = None


class FourPhaseProjectorSession:
    PHASE_ORDER = ("000", "090", "180", "270")

    def __init__(self, projector: ProjectorController) -> None:
        self.projector = projector

    def capture(self, capture_phase: Callable[[str], Any]) -> dict[str, Any]:
        """Keep the persistent window black outside the four capture callbacks."""
        self.projector.show_black()
        captured: dict[str, Any] = {}
        try:
            for phase_name in self.PHASE_ORDER:
                self.projector.show_phase(phase_name)
                captured[phase_name] = capture_phase(phase_name)
            return captured
        finally:
            self.projector.show_black()
