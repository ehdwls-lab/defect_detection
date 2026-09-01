from __future__ import annotations

import argparse
import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.integration.real_pose_planner import RealPosePlanner


def main() -> int:
    parser = argparse.ArgumentParser(description="Inspect a structured-light pose JSON without moving hardware")
    parser.add_argument("--pose-json", required=True, help="Path to structured_light_pose_v1 JSON")
    args = parser.parse_args()

    plan = RealPosePlanner().plan(args.pose_json)
    pose = plan.poses[0]
    print("Selected inspection pose")
    print(f"source: {pose.source}")
    print(f"plane: {pose.metadata['plane_role']}")
    print(f"roll: {pose.roll_deg:+.2f} deg")
    print(f"pitch: {pose.pitch_deg:+.2f} deg")
    print("Z: NOT PROVIDED")
    print(f"legacy Z: {'IGNORED' if pose.metadata['legacy_z_ignored'] else 'NOT PRESENT'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
