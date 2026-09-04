from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass
from typing import Any, Callable

from .protocol import (
    MalformedTelemetryError, format_orientation_command, format_pose_command,
    format_z_command, parse_telemetry,
)
from .types import PlatformLimits, PlatformPoseCommand, PlatformTelemetry


class PlatformSerialError(RuntimeError):
    pass


class PlatformTelemetryTimeout(PlatformSerialError):
    pass


@dataclass(frozen=True)
class SerialPlatformConfig:
    port: str
    baudrate: int = 115200
    read_timeout_s: float = 1.0
    write_timeout_s: float = 1.0

    def validate(self) -> None:
        if not self.port:
            raise ValueError("platform serial port is required")
        if self.baudrate <= 0:
            raise ValueError("baudrate must be positive")
        for name, value in (
            ("read_timeout_s", self.read_timeout_s),
            ("write_timeout_s", self.write_timeout_s),
        ):
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be positive and finite")


class SerialPlatformController:
    """STM32 USART2 transport. Construction never opens or writes the port."""

    def __init__(self, config: SerialPlatformConfig, limits: PlatformLimits | None = None,
                 serial_factory: Callable[..., Any] | None = None) -> None:
        config.validate()
        self.config = config
        self.limits = limits or PlatformLimits()
        self._serial_factory = serial_factory
        self._serial: Any | None = None
        self.logger = logging.getLogger(__name__)

    def connect(self) -> None:
        if self._serial is not None and getattr(self._serial, "is_open", True):
            return
        factory = self._serial_factory
        if factory is None:
            import serial
            factory = serial.Serial
        self._serial = factory(
            port=self.config.port, baudrate=self.config.baudrate,
            bytesize=8, parity="N", stopbits=1,
            timeout=self.config.read_timeout_s,
            write_timeout=self.config.write_timeout_s,
        )
        self.logger.info("[PLATFORM] connected port=%s baud=%d", self.config.port, self.config.baudrate)

    def close(self) -> None:
        if self._serial is not None:
            self._serial.close()
        self._serial = None

    def _port(self) -> Any:
        if self._serial is None or not getattr(self._serial, "is_open", True):
            raise PlatformSerialError("platform serial port is not connected")
        return self._serial

    def discard_stale_input(self) -> None:
        """Discard bytes received before the next motion command.

        PySerial's input-buffer reset is required here; reading until empty with
        a blocking timeout would introduce another race with the 50 Hz stream.
        """
        port = self._port()
        reset = getattr(port, "reset_input_buffer", None)
        if reset is None:
            raise PlatformSerialError("serial transport cannot safely discard stale input")
        reset()

    def read_fresh_telemetry(self, timeout: float | None = None, *,
                             settle_s: float = 0.0) -> PlatformTelemetry:
        """Return valid telemetry received after a host-side RX boundary.

        This is a best-effort USB/CDC drain: reset, wait for the configurable
        settle interval, then reset again. It cannot prove the firmware packet's
        creation time without a device sequence/timestamp or command ACK.
        Malformed telemetry after the final boundary is skipped while the
        underlying parser remains strict. ``timeout`` applies after the drain,
        so total call time can be up to ``settle_s + timeout``.
        """
        settle = float(settle_s)
        if not math.isfinite(settle) or settle < 0:
            raise ValueError("fresh telemetry settle_s must be finite and non-negative")
        self.discard_stale_input()
        if settle:
            time.sleep(settle)
        self.discard_stale_input()
        return self._read_telemetry(timeout, skip_malformed=True)

    def move_to(self, command: PlatformPoseCommand) -> None:
        self.limits.validate(command)
        self._write_packet(format_pose_command(command))

    def move_z(self, z_cm: float) -> None:
        """Send only the firmware's verified absolute Z command."""
        self._validate_axis("z_cm", z_cm, self.limits.z_min_cm, self.limits.z_max_cm)
        self._write_packet(format_z_command(z_cm))

    def move_orientation(self, roll_deg: float, pitch_deg: float) -> None:
        """Send only the firmware's verified absolute R/P command."""
        self._validate_axis("roll_deg", roll_deg, self.limits.roll_min_deg, self.limits.roll_max_deg)
        self._validate_axis("pitch_deg", pitch_deg, self.limits.pitch_min_deg, self.limits.pitch_max_deg)
        self._write_packet(format_orientation_command(roll_deg, pitch_deg))

    @staticmethod
    def _validate_axis(name: str, value: float, minimum: float | None, maximum: float | None) -> None:
        if minimum is not None and value < minimum:
            raise ValueError(f"{name}={value} is below configured minimum {minimum}")
        if maximum is not None and value > maximum:
            raise ValueError(f"{name}={value} is above configured maximum {maximum}")

    def _write_packet(self, command: str) -> None:
        packet = command.encode("ascii")
        port = self._port()
        port.write(packet)
        if hasattr(port, "flush"):
            port.flush()
        self.logger.info("[PLATFORM TX] %s", packet.decode("ascii").strip())

    def read_telemetry(self, timeout: float | None = None) -> PlatformTelemetry:
        return self._read_telemetry(timeout, skip_malformed=False)

    def _read_telemetry(self, timeout: float | None, *,
                        skip_malformed: bool) -> PlatformTelemetry:
        port = self._port()
        wait = self.config.read_timeout_s if timeout is None else float(timeout)
        if not math.isfinite(wait) or wait <= 0:
            raise ValueError("timeout must be positive and finite")
        deadline = time.monotonic() + wait
        last_nonempty = ""
        while time.monotonic() < deadline:
            raw = port.readline()
            if not raw:
                continue
            line = raw.decode("utf-8", errors="replace").strip() if isinstance(raw, bytes) else str(raw).strip()
            last_nonempty = line
            if not line.startswith("TLM:"):
                self.logger.debug("[PLATFORM RX] ignored non-telemetry: %s", line)
                continue
            try:
                return parse_telemetry(line)
            except MalformedTelemetryError:
                if not skip_malformed:
                    raise
                self.logger.warning("[PLATFORM RX] skipped malformed telemetry after fresh boundary: %s", line)
                continue
        suffix = f"; last line={last_nonempty!r}" if last_nonempty else ""
        raise PlatformTelemetryTimeout(f"no valid telemetry before timeout{suffix}")

    def get_telemetry(self) -> PlatformTelemetry:
        return self.read_telemetry()

    def wait_until_stable(self, timeout: float) -> PlatformTelemetry:
        """Wait on an already-current stream; this creates no command boundary.

        Hardware command diagnostics must use ``PlatformMotionDiagnostic``,
        which establishes a post-command fresh boundary and requires multiple
        stable samples. This compatibility method alone is not command-safe.
        """
        if not math.isfinite(timeout) or timeout <= 0:
            raise ValueError("timeout must be positive and finite")
        deadline = time.monotonic() + timeout
        last: PlatformTelemetry | None = None
        while time.monotonic() < deadline:
            remaining = deadline - time.monotonic()
            last = self.read_telemetry(timeout=min(self.config.read_timeout_s, remaining))
            if last.stable:
                return last
        detail = f" last telemetry={last}" if last is not None else ""
        raise PlatformTelemetryTimeout(f"platform did not report stable before timeout;{detail}")
