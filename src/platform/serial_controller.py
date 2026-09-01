from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any, Callable

from .protocol import format_pose_command, parse_telemetry
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
        if self.baudrate <= 0 or self.read_timeout_s <= 0 or self.write_timeout_s <= 0:
            raise ValueError("baudrate and serial timeouts must be positive")


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

    def move_to(self, command: PlatformPoseCommand) -> None:
        self.limits.validate(command)
        packet = format_pose_command(command).encode("ascii")
        port = self._port()
        port.write(packet)
        if hasattr(port, "flush"):
            port.flush()
        self.logger.info("[PLATFORM TX] %s", packet.decode("ascii").strip())

    def read_telemetry(self, timeout: float | None = None) -> PlatformTelemetry:
        port = self._port()
        wait = self.config.read_timeout_s if timeout is None else float(timeout)
        if wait <= 0:
            raise ValueError("timeout must be positive")
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
            return parse_telemetry(line)
        suffix = f"; last line={last_nonempty!r}" if last_nonempty else ""
        raise PlatformTelemetryTimeout(f"no valid telemetry before timeout{suffix}")

    def get_telemetry(self) -> PlatformTelemetry:
        return self.read_telemetry()

    def wait_until_stable(self, timeout: float) -> PlatformTelemetry:
        if timeout <= 0:
            raise ValueError("timeout must be positive")
        deadline = time.monotonic() + timeout
        last: PlatformTelemetry | None = None
        while time.monotonic() < deadline:
            remaining = deadline - time.monotonic()
            last = self.read_telemetry(timeout=min(self.config.read_timeout_s, remaining))
            if last.stable:
                return last
        detail = f" last telemetry={last}" if last is not None else ""
        raise PlatformTelemetryTimeout(f"platform did not report stable before timeout;{detail}")
