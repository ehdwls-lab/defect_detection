from __future__ import annotations

import unittest

from src.platform.protocol import (
    MalformedTelemetryError,
    PlatformProtocolError,
    format_mode_command,
    format_orientation_command,
    format_pose_command,
    format_reset_command,
    format_z_command,
    parse_telemetry,
)


class PlatformProtocolTests(unittest.TestCase):
    def test_command_formatters(self):
        self.assertEqual(format_pose_command(z_cm=20, roll_deg=23, pitch_deg=0), "Z:20.00 R:23.00 P:0.00\r\n")
        self.assertEqual(format_z_command(21.5), "Z:21.50\r\n")
        self.assertEqual(format_orientation_command(5, -2.5), "R:5.00 P:-2.50\r\n")
        self.assertEqual(format_reset_command(), "RST\r\n")
        self.assertEqual(format_mode_command(1), "MODE:1\r\n")
        with self.assertRaises(PlatformProtocolError):
            format_mode_command(3)

    def test_telemetry(self):
        telemetry = parse_telemetry(
            "TLM:Z=20.00,R=1.25,P=-0.42,S=1,M1=0,M2=0,M3=0,H=0,G=1,C=1,VR=0.02,VP=-0.01",
            timestamp=123.0,
        )
        self.assertEqual(telemetry.z_cm, 20.0)
        self.assertEqual(telemetry.roll_deg, 1.25)
        self.assertEqual(telemetry.pitch_deg, -0.42)
        self.assertTrue(telemetry.stable)
        self.assertFalse(telemetry.homing)
        self.assertEqual(telemetry.imu_mode, 1)
        self.assertEqual(telemetry.control_mode, 1)
        self.assertEqual(telemetry.roll_rate_deg_s, 0.02)
        self.assertEqual(telemetry.pitch_rate_deg_s, -0.01)

    def test_malformed_telemetry(self):
        for line in ("", "Z=1", "TLM:Z=nope", "TLM:Z=1,R=2"):
            with self.subTest(line=line), self.assertRaises(MalformedTelemetryError):
                parse_telemetry(line)


if __name__ == "__main__":
    unittest.main()
