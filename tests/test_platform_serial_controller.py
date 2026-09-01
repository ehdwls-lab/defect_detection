from __future__ import annotations

import unittest

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
    def close(self): self.is_open = False


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


if __name__ == "__main__":
    unittest.main()
