from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from src.integration.structured_light_runner import StructuredLightRunInfo
from src.platform.types import PlatformTelemetry
from src.tools.measure_metric_pose import MetricPoseMeasurement


def telemetry(roll: float, pitch: float) -> PlatformTelemetry:
    return PlatformTelemetry(
        0.0, roll, pitch, True, False, 0, 0, 0, 1, 1, 0.0, 0.0, 1.0,
    )


class FakePlatform:
    def __init__(self):
        self.reads = [telemetry(1.0, 2.0), telemetry(3.0, 4.0)]
        self.read_count = 0
        self.motion_commands = []

    def connect(self): pass
    def close(self): pass
    def read_fresh_telemetry(self, **_kwargs):
        value = self.reads[self.read_count]
        self.read_count += 1
        return value


class FakeProjector:
    def open(self): pass
    def show_black(self): pass
    def close(self): pass


class FakeRunner:
    def __init__(self, root: Path, pose: Path):
        self.root, self.pose = root, pose

    def run_scan(self):
        return StructuredLightRunInfo("scan", self.root, pose_json_path=self.pose)


class MeasureMetricPosePostprocessTests(unittest.TestCase):
    def test_read_only_measurement_calls_shared_postprocess_with_fresh_telemetry(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            scan = root / "scan"
            scan.mkdir()
            ply = scan / "input.ply"
            ply.write_text("ply", encoding="ascii")
            pose = scan / "pose.json"
            pose.write_text(json.dumps({
                "input_ply": str(ply),
                "planes": [{
                    "plane_name": "Raw", "points_count": 100,
                    "roll_deg": 80.0, "pitch_deg": -70.0,
                    "raw_roll_deg": 80.0, "raw_pitch_deg": -70.0,
                    "metric_pose": {
                        "physical_metric": True, "status": "METRIC_VALID",
                        "camera_roll_deg": 0.0, "camera_pitch_deg": 0.0,
                    },
                }],
            }), encoding="utf-8")
            calls = []

            def postprocessor(path, *, fresh_telemetry_reader):
                current = fresh_telemetry_reader()
                calls.append((Path(path), current.roll_deg, current.pitch_deg))
                payload = json.loads(Path(path).read_text(encoding="utf-8"))
                payload.update(raw_plane_count=1, metric_physical_plane_count=1)
                plane = payload["planes"][0]
                plane.update(
                    physical_plane_index=0,
                    merged_source_plane_indices=[0],
                    merged_source_plane_names=["Raw"],
                )
                plane["metric_pose"].update(
                    status="REACHABLE", reachable=True, reject_reason=None,
                    reason="within limit", normal_xyz=[0, 0, -1],
                    center_xyz_mm=[0, 0, 500], depth_points_count=100,
                    depth_coverage=1.0, calibration_id="camera_platform_rp_20260903",
                    current_platform_roll_deg=current.roll_deg,
                    current_platform_pitch_deg=current.pitch_deg,
                    target_platform_roll_deg=current.roll_deg,
                    target_platform_pitch_deg=current.pitch_deg,
                )
                Path(path).write_text(json.dumps(payload), encoding="utf-8")
                return {
                    "raw_plane_count": 1, "metric_physical_plane_count": 1,
                    "reachable_pose_count": 1,
                }

            platform = FakePlatform()
            measurement = MetricPoseMeasurement(
                platform=platform, structured_light_runner=FakeRunner(scan, pose),
                projector=FakeProjector(), output_directory=root / "output",
                metric_pose_postprocessor=postprocessor,
            )
            summary = measurement.run()
            self.assertEqual(platform.read_count, 2)
            self.assertEqual(platform.motion_commands, [])
            self.assertEqual(calls, [(pose.resolve(), 3.0, 4.0)])
            self.assertEqual(summary["platform_telemetry"]["roll_deg"], 3.0)
            self.assertEqual(summary["metric_physical_plane_count"], 1)
            saved = json.loads(pose.read_text(encoding="utf-8"))
            self.assertEqual(saved["planes"][0]["metric_pose"]["current_platform_pitch_deg"], 4.0)


if __name__ == "__main__":
    unittest.main()
