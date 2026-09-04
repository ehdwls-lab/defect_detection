from __future__ import annotations

import argparse
import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.integration.real_pose_planner import RealPosePlanner


def print_plan_diagnostic(pose_json: str | Path) -> int:
    plan = RealPosePlanner().plan(pose_json)
    total_physical = int(
        plan.metadata.get("metric_physical_plane_count", plan.metadata["parsed_plane_count"])
    )
    print("POSE PLANNER DIAGNOSTIC")
    print(f"Total physical planes = {total_physical}")
    print(f"Reachable poses = {len(plan.poses)}")
    rejected = plan.metadata.get("rejected_planes", [])
    if rejected:
        print("Rejected planes")
        for plane in rejected:
            print(
                f"- {plane.get('plane_name', 'unknown')}: "
                f"{plane.get('reject_reason', 'not reachable')}"
            )
    if not plan.poses:
        print("Selected inspection pose = NONE")
        return 0
    pose = plan.poses[0]
    print("Selected inspection pose")
    print(f"source: {pose.source}")
    print(f"plane: {pose.metadata['plane_role']}")
    print(f"roll: {pose.roll_deg:+.2f} deg")
    print(f"pitch: {pose.pitch_deg:+.2f} deg")
    print("Z: NOT PROVIDED")
    print(f"legacy Z: {'IGNORED' if pose.metadata['legacy_z_ignored'] else 'NOT PRESENT'}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Inspect a structured-light pose JSON without moving hardware")
    parser.add_argument("--pose-json", required=True, help="Path to structured_light_pose_v1 JSON")
    args = parser.parse_args(argv)
    return print_plan_diagnostic(args.pose_json)


if __name__ == "__main__":
    raise SystemExit(main())
