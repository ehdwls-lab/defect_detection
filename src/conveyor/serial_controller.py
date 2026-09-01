from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any, Callable


class ConveyorError(RuntimeError):
    pass


class ConveyorConfigurationError(ConveyorError):
    pass


class ConveyorTimeoutError(ConveyorError):
    pass


@dataclass(frozen=True)
class SerialConveyorConfig:
    port: str
    inspection_direction: str
    inspection_steps: int
    exit_direction: str
    exit_steps: int
    baudrate: int = 115200
    timeout_sec: float = 30.0

    def validate(self) -> None:
        if not self.port:
            raise ConveyorConfigurationError("conveyor port is required")
        for name, direction in (("inspection_direction", self.inspection_direction),
                                ("exit_direction", self.exit_direction)):
            if direction not in ("F", "B"):
                raise ConveyorConfigurationError(f"{name} must be F or B")
        for name, steps in (("inspection_steps", self.inspection_steps), ("exit_steps", self.exit_steps)):
            if isinstance(steps, bool) or not isinstance(steps, int) or steps <= 0:
                raise ConveyorConfigurationError(f"{name} must be a positive integer")
        if self.baudrate <= 0 or self.timeout_sec <= 0:
            raise ConveyorConfigurationError("baudrate and timeout_sec must be positive")


def format_move_command(direction: str, steps: int) -> str:
    direction = direction.upper()
    if direction not in ("F", "B"):
        raise ConveyorConfigurationError("direction must be F or B")
    if isinstance(steps, bool) or not isinstance(steps, int) or steps <= 0:
        raise ConveyorConfigurationError("steps must be a positive integer")
    return f"{direction}{steps}\n"


class SerialConveyorController:
    """Adapter for the legacy blocking Arduino F/B step protocol."""

    completion_marker = "Target Reached"

    def __init__(self, config: SerialConveyorConfig,
                 serial_factory: Callable[..., Any] | None = None) -> None:
        config.validate()
        self.config = config
        self._serial_factory = serial_factory
        self._serial: Any | None = None
        self._moving = False
        self.logger = logging.getLogger(__name__)

    def connect(self) -> None:
        if self._serial is not None and getattr(self._serial, "is_open", True):
            return
        factory = self._serial_factory
        if factory is None:
            import serial
            factory = serial.Serial
        self._serial = factory(
            port=self.config.port,
            baudrate=self.config.baudrate,
            bytesize=8,
            parity="N",
            stopbits=1,
            timeout=0.1,
            write_timeout=1.0,
        )
        self.logger.info("[CONVEYOR] connected port=%s baud=%d", self.config.port, self.config.baudrate)

    def close(self) -> None:
        if self._serial is not None:
            self._serial.close()
        self._serial = None
        self._moving = False

    def _require_connected(self) -> Any:
        if self._serial is None or not getattr(self._serial, "is_open", True):
            raise ConveyorError("conveyor serial port is not connected")
        return self._serial

    def move_steps(self, direction: str, steps: int) -> None:
        port = self._require_connected()
        command = format_move_command(direction, steps)
        port.write(command.encode("ascii"))
        if hasattr(port, "flush"):
            port.flush()
        self._moving = True
        self.logger.info("[CONVEYOR] command=%s (open-loop steps; not distance)", command.strip())

    def move_to_inspection(self) -> None:
        self.move_steps(self.config.inspection_direction, self.config.inspection_steps)

    def move_out(self) -> None:
        self.move_steps(self.config.exit_direction, self.config.exit_steps)

    def wait_until_stopped(self, timeout: float | None = None) -> None:
        if not self._moving:
            raise ConveyorError("no conveyor movement is pending")
        port = self._require_connected()
        deadline = time.monotonic() + (self.config.timeout_sec if timeout is None else timeout)
        if deadline <= time.monotonic():
            raise ConveyorConfigurationError("timeout must be positive")
        while time.monotonic() < deadline:
            raw = port.readline()
            if not raw:
                continue
            line = raw.decode("utf-8", errors="replace").strip() if isinstance(raw, bytes) else str(raw).strip()
            self.logger.info("[CONVEYOR RX] %s", line)
            if self.completion_marker in line:
                self._moving = False
                return
        raise ConveyorTimeoutError(
            f"conveyor completion marker {self.completion_marker!r} not received before timeout"
        )
