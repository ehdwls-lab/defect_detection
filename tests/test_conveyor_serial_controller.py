from __future__ import annotations

import unittest

from src.conveyor.serial_controller import (
    ConveyorTimeoutError,
    SerialConveyorConfig,
    SerialConveyorController,
    format_move_ack,
    format_move_command,
)


class FakeSerial:
    def __init__(self, lines=(), responses=(), **kwargs):
        self.lines = list(lines)
        self.responses = [list(items) for items in responses]
        self.writes = []
        self.is_open = True
        self.reset_count = 0

    def write(self, data):
        self.writes.append(data)
        if self.responses:
            self.lines.extend(self.responses.pop(0))
    def flush(self): pass
    def close(self): self.is_open = False
    def readline(self): return self.lines.pop(0) if self.lines else b""
    def reset_input_buffer(self):
        self.reset_count += 1
        self.lines.clear()


def startup_banner():
    return [
        b"========================\n",
        b"DUAL STEP MOTOR TEST MODE\n",
        b"========================\n",
        b"Usage: F[steps] or B[steps]\n",
        b"Example: F5000 (Forward 5000 steps)\n",
        b"Example: B2000 (Backward 2000 steps)\n",
        b"========================\n",
    ]


def config(timeout=.01, startup_timeout=.01):
    return SerialConveyorConfig(
        "FAKE", "F", 5000, "B", 2000,
        timeout_sec=timeout, startup_timeout_sec=startup_timeout,
    )


class ConveyorSerialControllerTests(unittest.TestCase):
    def test_format_commands(self):
        self.assertEqual(format_move_command("F", 5000), "F5000\n")
        self.assertEqual(format_move_command("b", 2000), "B2000\n")
        self.assertEqual(format_move_ack("F", 5000), ">>> DUAL FORWARD 5000 steps")
        self.assertEqual(format_move_ack("b", 2000), "<<< DUAL BACKWARD 2000 steps")

    def test_forward_backward_and_completion(self):
        fake = FakeSerial(responses=[
            [b">>> DUAL FORWARD 5000 steps\n", b"moving\n", b"=== STOP (Target Reached) ===\n"],
            [b"<<< DUAL BACKWARD 2000 steps\n", b"noise\n", b"=== STOP (Target Reached) ===\n"],
        ])
        fake.lines = startup_banner()
        controller = SerialConveyorController(config(), serial_factory=lambda **kwargs: fake)
        controller.connect()
        controller.move_to_inspection()
        controller.wait_until_stopped()
        controller.move_out()
        controller.wait_until_stopped()
        self.assertEqual(fake.writes, [b"F5000\n", b"B2000\n"])
        self.assertEqual(fake.reset_count, 6)

    def test_malformed_output_times_out(self):
        fake = FakeSerial(lines=startup_banner(), responses=[[b"STOP maybe\n", b"bad bytes \xff\n"]])
        controller = SerialConveyorController(config(.001), serial_factory=lambda **kwargs: fake)
        controller.connect()
        controller.move_to_inspection()
        with self.assertRaises(ConveyorTimeoutError):
            controller.wait_until_stopped()

    def test_stale_completion_is_discarded_before_current_command(self):
        fake = FakeSerial(
            lines=startup_banner() + [b"=== STOP (Target Reached) ===\n"],
            responses=[[
                # A late stale completion arriving after the host reset must
                # still not complete the new move before its command echo.
                b"=== STOP (Target Reached) ===\n",
                b">>> DUAL FORWARD 5000 steps\n",
                b"=== STOP (Target Reached) ===\n",
            ]],
        )
        controller = SerialConveyorController(config(), serial_factory=lambda **kwargs: fake)
        controller.connect()
        controller.move_to_inspection()
        controller.wait_until_stopped()
        self.assertEqual(fake.writes, [b"F5000\n"])
        self.assertEqual(fake.reset_count, 4)

    def test_completion_without_current_command_echo_times_out(self):
        fake = FakeSerial(lines=startup_banner(), responses=[[b"=== STOP (Target Reached) ===\n"]])
        controller = SerialConveyorController(config(.001), serial_factory=lambda **kwargs: fake)
        controller.connect()
        controller.move_to_inspection()
        with self.assertRaises(ConveyorTimeoutError):
            controller.wait_until_stopped()

    def test_startup_banner_is_consumed_before_command(self):
        fake = FakeSerial(lines=startup_banner(), responses=[[
            b">>> DUAL FORWARD 5000 steps\n",
            b"=== STOP (Target Reached) ===\n",
        ]])
        controller = SerialConveyorController(config(), serial_factory=lambda **kwargs: fake)
        controller.connect()
        self.assertEqual(fake.lines, [])
        controller.move_to_inspection()
        controller.wait_until_stopped()

    def test_startup_timeout_does_not_write_command(self):
        fake = FakeSerial(lines=[])
        controller = SerialConveyorController(
            config(startup_timeout=.001), serial_factory=lambda **kwargs: fake,
        )
        with self.assertRaises(ConveyorTimeoutError):
            controller.connect()
        self.assertEqual(fake.writes, [])


if __name__ == "__main__":
    unittest.main()
