from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np


@dataclass(frozen=True)
class RGBDepthFrame:
    color_bgr: np.ndarray
    depth_mm: np.ndarray
    timestamp: float


class CameraController(Protocol):
    def start(self) -> None: ...
    def capture(self) -> RGBDepthFrame: ...
    def close(self) -> None: ...
