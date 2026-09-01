from __future__ import annotations

import unittest

from src.conveyor.serial_controller import (
    ConveyorTimeoutError,
    SerialConveyorConfig,
    SerialConveyorController,
    format_move_command,
)


class FakeSerial:
    def __init__(self, lines=(), **kwargs):
        self.lines = list(lines)
        self.writes = []
        self.is_open = True

    def write(self, data): self.writes.append(data)
    def flush(self): pass
    def close(self): self.is_open = False
    def readline(self): return self.lines.pop(0) if self.lines else b""


def config(timeout=.01):
    return SerialConveyorConfig("FAKE", "F", 5000, "B", 2000, timeout_sec=timeout)


class ConveyorSerialControllerTests(unittest.TestCase):
    def test_format_commands(self):
        self.assertEqual(format_move_command("F", 5000), "F5000\n")
        self.assertEqual(format_move_command("b", 2000), "B2000\n")

    def test_forward_backward_and_completion(self):
        fake = FakeSerial([b"moving\n", b"=== STOP (Target Reached) ===\n",
                           b"noise\n", b"=== STOP (Target Reached) ===\n"])
        controller = SerialConveyorController(config(), serial_factory=lambda **kwargs: fake)
        controller.connect()
        controller.move_to_inspection()
        controller.wait_until_stopped()
        controller.move_out()
        controller.wait_until_stopped()
        self.assertEqual(fake.writes, [b"F5000\n", b"B2000\n"])

    def test_malformed_output_times_out(self):
        fake = FakeSerial([b"STOP maybe\n", b"bad bytes \xff\n"])
        controller = SerialConveyorController(config(.001), serial_factory=lambda **kwargs: fake)
        controller.connect()
        controller.move_to_inspection()
        with self.assertRaises(ConveyorTimeoutError):
            controller.wait_until_stopped()


if __name__ == "__main__":
    unittest.main()
