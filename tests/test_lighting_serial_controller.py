from __future__ import annotations

import unittest

from src.lighting.serial_controller import (
    LightingConfigurationError,
    LightingTimeoutError,
    SerialLightingConfig,
    SerialLightingController,
)


class FakeSerial:
    def __init__(self, lines=(), responses=(), **kwargs):
        self.lines = list(lines)
        self.responses = [list(items) for items in responses]
        self.writes = []
        self.is_open = True
        self.reset_count = 0

    def readline(self):
        return self.lines.pop(0) if self.lines else b""

    def write(self, data):
        self.writes.append(data)
        if self.responses:
            self.lines.extend(self.responses.pop(0))

    def flush(self):
        pass

    def reset_input_buffer(self):
        self.reset_count += 1
        self.lines.clear()

    def close(self):
        self.is_open = False


def startup():
    return [b"NeoPixel controller started.\n", b"LEDs are initially OFF.\n"]


def servo_startup():
    return [b"Controller started.\n", b"LEDs are initially OFF.\n"]


class LightingSerialControllerTests(unittest.TestCase):
    def controller(self, fake, timeout=.01):
        return SerialLightingController(
            SerialLightingConfig("FAKE", startup_timeout_sec=timeout, command_timeout_sec=timeout),
            serial_factory=lambda **kwargs: fake,
        )

    def test_startup_then_off_on_ack(self):
        fake = FakeSerial(
            lines=startup(),
            responses=[[b"Mode: Neutral White\n"], [b"Mode: All LEDs OFF\n"]],
        )
        controller = self.controller(fake)
        controller.connect()
        controller.inspection_on()
        controller.inspection_off()
        self.assertEqual(fake.writes, [b"2", b"0"])

    def test_led_servo_firmware_startup_banner_is_accepted(self):
        fake = FakeSerial(lines=servo_startup())
        controller = self.controller(fake)
        controller.connect()
        self.assertEqual(fake.writes, [])

    def test_leds_initially_off_line_remains_required_for_both_banners(self):
        for banner in (b"NeoPixel controller started.\n", b"Controller started.\n"):
            fake = FakeSerial(lines=[banner])
            controller = self.controller(fake, timeout=.001)
            with self.subTest(banner=banner), self.assertRaises(LightingTimeoutError):
                controller.connect()
            self.assertEqual(fake.writes, [])

    def test_servo_api_uses_same_owned_serial_and_exact_ack(self):
        fake = FakeSerial(
            lines=servo_startup(),
            responses=[[b"Servo angle: 0\n"], [b"Servo angle: 90\n"]],
        )
        controller = self.controller(fake)
        controller.connect()
        controller.servo_zero()
        controller.servo_ninety()
        self.assertEqual(fake.writes, [b"[", b"]"])

    def test_semantic_cover_mapping_uses_one_explicit_config(self):
        fake = FakeSerial(
            lines=servo_startup(),
            responses=[[b"Servo angle: 90\n"], [b"Servo angle: 0\n"]],
        )
        controller = SerialLightingController(
            SerialLightingConfig(
                "FAKE", startup_timeout_sec=.01, command_timeout_sec=.01,
                projector_cover_open_angle_deg=90,
                projector_cover_close_angle_deg=0,
            ),
            serial_factory=lambda **kwargs: fake,
        )
        controller.connect()
        controller.projector_cover_open()
        controller.projector_cover_close()
        self.assertEqual(fake.writes, [b"]", b"["])

    def test_semantic_cover_refuses_unresolved_hardware_mapping(self):
        fake = FakeSerial(lines=servo_startup())
        controller = self.controller(fake)
        controller.connect()
        with self.assertRaisesRegex(LightingConfigurationError, "mapping is unresolved"):
            controller.projector_cover_open()

    def test_stale_ack_is_ignored_after_command_boundary(self):
        fake = FakeSerial(
            lines=startup(),
            responses=[[b"Mode: All LEDs OFF\n", b"Mode: Neutral White\n"]],
        )
        controller = self.controller(fake)
        controller.connect()
        controller.inspection_on()
        self.assertEqual(fake.writes, [b"2"])

    def test_startup_timeout_writes_nothing(self):
        fake = FakeSerial()
        controller = self.controller(fake, timeout=.001)
        with self.assertRaises(LightingTimeoutError):
            controller.connect()
        self.assertEqual(fake.writes, [])
        self.assertFalse(fake.is_open)

    def test_close_closes_serial(self):
        fake = FakeSerial(lines=startup())
        controller = self.controller(fake)
        controller.connect()
        controller.close()
        self.assertFalse(fake.is_open)


if __name__ == "__main__":
    unittest.main()
