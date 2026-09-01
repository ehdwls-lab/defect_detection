"""Shared, side-effect-free projector selection and production pattern helpers."""

from __future__ import annotations

import re
import subprocess
from typing import Any

import numpy as np


PRODUCTION_PERIOD = 80
PRODUCTION_DIRECTION = "horizontal"
PRODUCTION_BASE = 128.0
PRODUCTION_AMPLITUDE = 127.0
PHASES = (
    (0.0, "000"),
    (np.pi / 2.0, "090"),
    (np.pi, "180"),
    (3.0 * np.pi / 2.0, "270"),
)


def parse_xrandr_monitors(output: str) -> list[dict[str, Any]]:
    pattern = re.compile(
        r"^(?P<name>\S+)\s+connected(?:\s+primary)?\s+"
        r"(?P<w>\d+)x(?P<h>\d+)\+(?P<x>-?\d+)\+(?P<y>-?\d+)"
    )
    monitors: list[dict[str, Any]] = []
    for line in output.splitlines():
        match = pattern.search(line)
        if match:
            monitors.append({
                "name": match.group("name"),
                "w": int(match.group("w")),
                "h": int(match.group("h")),
                "x": int(match.group("x")),
                "y": int(match.group("y")),
                "primary": " primary " in f" {line} ",
            })
    return monitors


def xrandr_monitors() -> list[dict[str, Any]]:
    try:
        result = subprocess.run(
            ["xrandr", "--query"], capture_output=True, text=True, check=True,
        )
    except (OSError, subprocess.SubprocessError):
        return []
    return parse_xrandr_monitors(result.stdout)


def select_projector_monitor(
    monitors: list[dict[str, Any]], monitor_name: str = "auto",
) -> dict[str, Any] | None:
    if monitor_name != "auto":
        return next((item for item in monitors if item["name"] == monitor_name), None)
    selected = next(
        (item for item in monitors if "HDMI" in str(item["name"]).upper()), None,
    )
    if selected is None:
        selected = next((item for item in monitors if not item["primary"]), None)
    return selected


def pattern_coordinates(width: int, height: int, direction: str) -> np.ndarray:
    if direction == "vertical":
        return np.tile(np.arange(width, dtype=np.float32), (height, 1))
    if direction == "horizontal":
        return np.tile(
            np.arange(height, dtype=np.float32).reshape(height, 1), (1, width),
        )
    raise ValueError(f"지원하지 않는 패턴 방향입니다: {direction}")


def sine_brightness(
    coordinates: np.ndarray, period: int, base: float, amplitude: float, phase: float,
) -> np.ndarray:
    pattern = base + amplitude * np.cos(
        2.0 * np.pi * coordinates / float(period) + phase
    )
    return np.clip(pattern, 0, 255).astype(np.uint8)


def color_pattern(gray_pattern: np.ndarray, color_name: str) -> np.ndarray:
    zeros = np.zeros_like(gray_pattern)
    if color_name == "white":
        return np.dstack((gray_pattern, gray_pattern, gray_pattern))
    if color_name == "green":
        return np.dstack((zeros, gray_pattern, zeros))
    if color_name == "red":
        return np.dstack((zeros, zeros, gray_pattern))
    if color_name == "blue":
        return np.dstack((gray_pattern, zeros, zeros))
    raise ValueError(f"지원하지 않는 색상입니다: {color_name}")


def production_phase_patterns(width: int, height: int) -> dict[str, np.ndarray]:
    coordinates = pattern_coordinates(width, height, PRODUCTION_DIRECTION)
    return {
        name: color_pattern(
            sine_brightness(
                coordinates, PRODUCTION_PERIOD, PRODUCTION_BASE,
                PRODUCTION_AMPLITUDE, phase,
            ),
            "white",
        )
        for phase, name in PHASES
    }
