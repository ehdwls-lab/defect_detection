#!/usr/bin/env python3
"""Canonical launcher for the preserved final structured-light implementation."""

from pathlib import Path
import runpy


IMPLEMENTATION = Path(__file__).with_name(
    "구조광_전처리_최종_v2_현재프레임플랫폼기준_Depth홀위상보강_경로수정_0822 (1).py"
)

if not IMPLEMENTATION.is_file():
    raise FileNotFoundError(f"구조광 최종 구현 파일이 없습니다: {IMPLEMENTATION}")

runpy.run_path(str(IMPLEMENTATION), run_name="__main__")
