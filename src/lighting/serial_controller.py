from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass
from typing import Any, Callable


class LightingError(RuntimeError):
    pass


class LightingConfigurationError(LightingError):
    pass


class LightingTimeoutError(LightingError):
    pass


@dataclass(frozen=True)
class SerialLightingConfig:
    port: str
    baudrate: int = 115200 
    startup_timeout_sec: float = 5.0
    command_timeout_sec: float = 2.0
    projector_cover_open_angle_deg: int | None = None
    projector_cover_close_angle_deg: int | None = None
    projector_cover_cleanup_state: str = "CLOSE"

    def validate(self) -> None:
        if not self.port:
            raise LightingConfigurationError("lighting port is required")
        if self.baudrate <= 0:
            raise LightingConfigurationError("baudrate must be positive")
        for name, value in (("startup_timeout_sec", self.startup_timeout_sec),
                            ("command_timeout_sec", self.command_timeout_sec)):
            if not math.isfinite(value) or value <= 0:
                raise LightingConfigurationError(f"{name} must be positive and finite")
        angles = (self.projector_cover_open_angle_deg, self.projector_cover_close_angle_deg)
        if any(value is not None and value not in {0, 90} for value in angles):
            raise LightingConfigurationError("projector cover angles must be 0 or 90")
        if (angles[0] is None) != (angles[1] is None):
            raise LightingConfigurationError("both projector cover OPEN/CLOSE angles are required")
        if angles[0] is not None and angles[0] == angles[1]:
            raise LightingConfigurationError("projector cover OPEN/CLOSE angles must differ")
        if self.projector_cover_cleanup_state not in {"OPEN", "CLOSE", "NONE"}:
            raise LightingConfigurationError("projector_cover_cleanup_state must be OPEN, CLOSE, or NONE")


class SerialLightingController:
    """Single-owner serial boundary for the Uno LED + optional Servo firmware."""

    startup_banner_options = (
        "NeoPixel controller started.",
        "Controller started.",
    )
    startup_ready_line = "LEDs are initially OFF."
    off_command = b"0"
    off_ack = "Mode: All LEDs OFF"
    on_command = b"2"
    on_ack = "Mode: Neutral White"
    servo_zero_command = b"["
    servo_zero_ack = "Servo angle: 0"
    servo_ninety_command = b"]"
    servo_ninety_ack = "Servo angle: 90"

    def __init__(self, config: SerialLightingConfig,
                 serial_factory: Callable[..., Any] | None = None) -> None:
        config.validate()
        self.config = config
        self._serial_factory = serial_factory
        self._serial: Any | None = None
        self._startup_ready = False
        self.logger = logging.getLogger(__name__)

    def _port(self) -> Any:
        if self._serial is None or not getattr(self._serial, "is_open", True):
            raise LightingError("lighting serial port is not connected")
        return self._serial

    def connect(self) -> None:
        if self._serial is not None and getattr(self._serial, "is_open", True):
            return
        factory = self._serial_factory
        if factory is None:
            import serial
            factory = serial.Serial
        self._serial = factory(
            port=self.config.port, baudrate=self.config.baudrate,
            bytesize=8, parity="N", stopbits=1, timeout=0.1, write_timeout=1.0,
        )
        try:
            self._wait_for_startup()
            self._reset_input()
            self.logger.info("[LIGHTING] connected port=%s baud=%d", self.config.port, self.config.baudrate)
        except Exception:
            self.close()
            raise

    def close(self) -> None:
        if self._serial is not None:
            self._serial.close()
        self._serial = None
        self._startup_ready = False

    def _reset_input(self) -> None:
        reset = getattr(self._port(), "reset_input_buffer", None)
        if reset is None:
            raise LightingError("lighting transport cannot safely discard stale input")
        reset()

    def _read_line(self) -> str:
        raw = self._port().readline()
        if isinstance(raw, bytes):
            return raw.decode("utf-8", errors="replace").strip()
        return str(raw).strip()

    def _wait_for_startup(self) -> None:
        deadline = time.monotonic() + self.config.startup_timeout_sec
        seen: set[str] = set()
        while time.monotonic() < deadline:
            raw = self._port().readline()
            if not raw:
                continue
            line = raw.decode("utf-8", errors="replace").strip() if isinstance(raw, bytes) else str(raw).strip()
            self.logger.info("[LIGHTING RX] %s", line)
            seen.add(line)
            banner_seen = any(item in seen for item in self.startup_banner_options)
            ready_seen = self.startup_ready_line in seen
            if banner_seen and ready_seen:
                self._startup_ready = True
                return
        missing = []
        if not any(item in seen for item in self.startup_banner_options):
            missing.append(f"one of {self.startup_banner_options!r}")
        if self.startup_ready_line not in seen:
            missing.append(self.startup_ready_line)
        raise LightingTimeoutError(f"lighting startup banner/readiness timeout (missing={missing!r})")

    def _command(self, command: bytes, ack: str) -> None:
        if not self._startup_ready:
            raise LightingError("lighting serial startup is not ready")
        port = self._port()
        self._reset_input()
        port.write(command)
        if hasattr(port, "flush"):
            port.flush()
        deadline = time.monotonic() + self.config.command_timeout_sec
        while time.monotonic() < deadline:
            line = self._read_line()
            if not line:
                continue
            self.logger.info("[LIGHTING RX] %s", line)
            if line == ack:
                return
        raise LightingTimeoutError(f"lighting ACK timeout (expected={ack!r})")

    def inspection_on(self) -> None:
        self._command(self.on_command, self.on_ack)

    def inspection_off(self) -> None:
        self._command(self.off_command, self.off_ack)

    def servo_zero(self) -> None:
        """Explicit diagnostic API only; production cycles do not call it."""
        self._command(self.servo_zero_command, self.servo_zero_ack)

    def servo_ninety(self) -> None:
        """Explicit diagnostic API only; production cycles do not call it."""
        self._command(self.servo_ninety_command, self.servo_ninety_ack)

    def _projector_cover_angle(self, angle_deg: int | None, semantic: str) -> None:
        if angle_deg is None:
            raise LightingConfigurationError(
                f"projector cover {semantic} angle is not configured; hardware mapping is unresolved"
            )
        if angle_deg == 0:
            self.servo_zero()
        elif angle_deg == 90:
            self.servo_ninety()
        else:  # defensive even after config validation
            raise LightingConfigurationError(f"unsupported projector cover angle: {angle_deg}")

    def projector_cover_open(self) -> None:
        self._projector_cover_angle(
            self.config.projector_cover_open_angle_deg, "OPEN",
        )

    def projector_cover_close(self) -> None:
        self._projector_cover_angle(
            self.config.projector_cover_close_angle_deg, "CLOSE",
        )

    def projector_cover_cleanup(self) -> None:
        if self.config.projector_cover_cleanup_state == "OPEN":
            self.projector_cover_open()
        elif self.config.projector_cover_cleanup_state == "CLOSE":
            self.projector_cover_close()
