from __future__ import annotations

import unittest

from src.platform.protocol import MalformedTelemetryError
from src.platform.serial_controller import SerialPlatformConfig, SerialPlatformController
from src.platform.types import PlatformPoseCommand


class FakeSerial:
    def __init__(self, lines=(), **kwargs):
        self.lines = list(lines)
        self.writes = []
        self.is_open = True
    def readline(self): return self.lines.pop(0) if self.lines else b""
    def write(self, value): self.writes.append(value)
    def flush(self): pass
    def reset_input_buffer(self): self.lines.clear()
    def close(self): self.is_open = False


class FreshSequenceSerial(FakeSerial):
    def __init__(self, stale_lines, after_second_reset):
        super().__init__(stale_lines)
        self.after_second_reset = list(after_second_reset)
        self.reset_count = 0

    def reset_input_buffer(self):
        self.reset_count += 1
        self.lines.clear()
        if self.reset_count == 1:
            # Simulate a USB/CDC packet reaching the host during settle.
            self.lines.append(b"TLM:Z=20.00,R=0,P=0,S=1,M1=0,M2=0,M3=0,H=0,G=1,C=1,VR=0,VP=0\n")
        elif self.reset_count == 2:
            self.lines.extend(self.after_second_reset)


class PlatformSerialControllerTests(unittest.TestCase):
    def test_read_only_telemetry(self):
        fake = FakeSerial([b"boot message\n", b"TLM:Z=20,R=1,P=2,S=1,M1=0,M2=0,M3=0,H=0,G=1,C=1,VR=.1,VP=.2\n"])
        controller = SerialPlatformController(
            SerialPlatformConfig("FAKE", read_timeout_s=.01), serial_factory=lambda **kwargs: fake
        )
        controller.connect()
        telemetry = controller.read_telemetry()
        self.assertTrue(telemetry.stable)
        self.assertEqual(fake.writes, [])

    def test_move_formats_verified_packet(self):
        fake = FakeSerial()
        controller = SerialPlatformController(SerialPlatformConfig("FAKE"), serial_factory=lambda **kwargs: fake)
        controller.connect()
        controller.move_to(PlatformPoseCommand(20, 1, -2))
        self.assertEqual(fake.writes, [b"Z:20.00 R:1.00 P:-2.00\r\n"])

    def test_partial_motion_formats_verified_packets(self):
        fake = FakeSerial()
        controller = SerialPlatformController(SerialPlatformConfig("FAKE"), serial_factory=lambda **kwargs: fake)
        controller.connect()
        controller.move_z(5)
        controller.move_orientation(2, 0)
        self.assertEqual(fake.writes, [b"Z:5.00\r\n", b"R:2.00 P:0.00\r\n"])

    def test_discard_stale_input_uses_transport_buffer_reset(self):
        fake = FakeSerial([b"TLM:Z=0,R=0,P=0,S=1\n"])
        controller = SerialPlatformController(SerialPlatformConfig("FAKE"), serial_factory=lambda **kwargs: fake)
        controller.connect()
        controller.discard_stale_input()
        self.assertEqual(fake.lines, [])

    def test_fresh_read_discards_old_motion_sequence_and_late_usb_packet(self):
        old = [
            b"TLM:Z=28.77,R=0,P=0,S=0,M1=-1000,M2=-1000,M3=-1000,H=0,G=1,C=1,VR=0,VP=0\n",
            b"TLM:Z=28.39,R=0,P=0,S=0,M1=-1000,M2=-1000,M3=-1000,H=0,G=1,C=1,VR=0,VP=0\n",
            b"TLM:Z=20.00,R=0,P=0,S=1,M1=0,M2=0,M3=0,H=0,G=1,C=1,VR=0,VP=0\n",
        ]
        fresh = b"TLM:Z=20.00,R=-0.03,P=0.01,S=1,M1=0,M2=0,M3=0,H=0,G=1,C=1,VR=0,VP=0\n"
        fake = FreshSequenceSerial(old, [fresh])
        controller = SerialPlatformController(
            SerialPlatformConfig("FAKE", read_timeout_s=.01), serial_factory=lambda **kwargs: fake,
        )
        controller.connect()
        result = controller.read_fresh_telemetry(.01, settle_s=0)
        self.assertEqual(fake.reset_count, 2)
        self.assertEqual(result.z_cm, 20.0)
        self.assertAlmostEqual(result.roll_deg, -.03)
        self.assertTrue(result.stable)

    def test_fresh_read_skips_malformed_then_returns_valid_but_strict_read_does_not(self):
        malformed = b"TLM:Z=bad,R=0,P=0,S=1\n"
        valid = b"TLM:Z=20,R=0,P=0,S=1,M1=0,M2=0,M3=0,H=0,G=1,C=1,VR=0,VP=0\n"
        fake = FreshSequenceSerial([], [malformed, valid])
        controller = SerialPlatformController(
            SerialPlatformConfig("FAKE", read_timeout_s=.01), serial_factory=lambda **kwargs: fake,
        )
        controller.connect()
        self.assertEqual(controller.read_fresh_telemetry(.01, settle_s=0).z_cm, 20)

        strict_fake = FakeSerial([malformed, valid])
        strict = SerialPlatformController(
            SerialPlatformConfig("FAKE", read_timeout_s=.01), serial_factory=lambda **kwargs: strict_fake,
        )
        strict.connect()
        with self.assertRaises(MalformedTelemetryError):
            strict.read_telemetry(.01)

    def test_non_finite_serial_timeouts_are_rejected(self):
        for value in (float("nan"), float("inf"), float("-inf")):
            with self.subTest(value=value), self.assertRaises(ValueError):
                SerialPlatformController(SerialPlatformConfig("FAKE", read_timeout_s=value))

        fake = FakeSerial()
        controller = SerialPlatformController(
            SerialPlatformConfig("FAKE", read_timeout_s=.01),
            serial_factory=lambda **kwargs: fake,
        )
        controller.connect()
        with self.assertRaises(ValueError):
            controller.read_telemetry(float("inf"))


if __name__ == "__main__":
    unittest.main()
