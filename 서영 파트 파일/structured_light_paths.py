"""Portable filesystem contract for the structured-light subsystem.

This module contains paths only.  It deliberately does not import camera or
projector libraries, so preflight checks can use it without opening hardware.
"""

from __future__ import annotations

import os
from pathlib import Path


ENV_ROOT = "STRUCTURED_LIGHT_ROOT"


def structured_light_root() -> Path:
    configured = os.environ.get(ENV_ROOT)
    if configured:
        return Path(configured).expanduser().resolve()
    return Path(__file__).resolve().parent


ROOT = structured_light_root()
PROJECTOR_CALIBRATION_DIR = ROOT / "프로젝터 수동 범위 확인"
PROJECTOR_RANGE_JSON = PROJECTOR_CALIBRATION_DIR / "프로젝터_세로범위.json"
PLATFORM_ROOT = ROOT / "플랫폼 바닥 따기"
CALIBRATION_ROOT = PLATFORM_ROOT / "현재배치_기준데이터" / "active"
DEPTH_CALIBRATION_DIR = CALIBRATION_ROOT / "E1999_G64"
PLATFORM_DEPTH_NPY = DEPTH_CALIBRATION_DIR / "플랫폼_바닥_depth.npy"
REFERENCE_4PHASE_DIR = CALIBRATION_ROOT / "E480_G16" / "Reference_4위상"
PREPROCESS_ROOT = PLATFORM_ROOT / "구조광_전처리"
SAMPLE_ROOT = PREPROCESS_ROOT / "샘플"
CAMERA_SETTINGS_JSON = ROOT / "구조광_전처리_촬영" / "공통_카메라_고정값.json"


def ensure_output_directories() -> None:
    """Create output containers, never calibration input files."""
    SAMPLE_ROOT.mkdir(parents=True, exist_ok=True)
    PROJECTOR_CALIBRATION_DIR.mkdir(parents=True, exist_ok=True)
    DEPTH_CALIBRATION_DIR.mkdir(parents=True, exist_ok=True)

