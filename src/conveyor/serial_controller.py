from __future__ import annotations

import logging
import math
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
    rx_settle_sec: float = 0.0
    startup_timeout_sec: float = 5.0

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
        if self.baudrate <= 0:
            raise ConveyorConfigurationError("baudrate must be positive")
        if not math.isfinite(self.timeout_sec) or self.timeout_sec <= 0:
            raise ConveyorConfigurationError("timeout_sec must be positive and finite")
        if not math.isfinite(self.rx_settle_sec) or self.rx_settle_sec < 0:
            raise ConveyorConfigurationError("rx_settle_sec must be finite and non-negative")
        if not math.isfinite(self.startup_timeout_sec) or self.startup_timeout_sec <= 0:
            raise ConveyorConfigurationError("startup_timeout_sec must be positive and finite")


def format_move_command(direction: str, steps: int) -> str:
    direction = direction.upper()
    if direction not in ("F", "B"):
        raise ConveyorConfigurationError("direction must be F or B")
    if isinstance(steps, bool) or not isinstance(steps, int) or steps <= 0:
        raise ConveyorConfigurationError("steps must be a positive integer")
    return f"{direction}{steps}\n"


def format_move_ack(direction: str, steps: int) -> str:
    """Return the exact command echo emitted by the checked-in Mega firmware."""
    direction = direction.upper()
    format_move_command(direction, steps)  # reuse strict validation
    word = "FORWARD" if direction == "F" else "BACKWARD"
    prefix = ">>>" if direction == "F" else "<<<"
    return f"{prefix} DUAL {word} {steps} steps"


class SerialConveyorController:
    """Adapter for the legacy blocking Arduino F/B step protocol."""

    completion_marker = "Target Reached"
    startup_banner_lines = (
        "DUAL STEP MOTOR TEST MODE",
        "Usage: F[steps] or B[steps]",
        "Example: F5000 (Forward 5000 steps)",
        "Example: B2000 (Backward 2000 steps)",
    )

    def __init__(self, config: SerialConveyorConfig,
                 serial_factory: Callable[..., Any] | None = None) -> None:
        config.validate()
        self.config = config
        self._serial_factory = serial_factory
        self._serial: Any | None = None
        self._moving = False
        self._pending_ack: str | None = None
        self._startup_ready = False
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
        try:
            self._wait_for_startup()
            self.discard_stale_input()
        except Exception:
            self.close()
            raise

    def close(self) -> None:
        if self._serial is not None:
            self._serial.close()
        self._serial = None
        self._moving = False
        self._pending_ack = None
        self._startup_ready = False

    def _require_connected(self) -> Any:
        if self._serial is None or not getattr(self._serial, "is_open", True):
            raise ConveyorError("conveyor serial port is not connected")
        return self._serial

    def discard_stale_input(self) -> None:
        """Create a best-effort host RX boundary before a conveyor command."""
        port = self._require_connected()
        reset = getattr(port, "reset_input_buffer", None)
        if reset is None:
            raise ConveyorError("serial transport cannot safely discard stale conveyor input")
        reset()
        if self.config.rx_settle_sec:
            time.sleep(self.config.rx_settle_sec)
        reset()

    def _wait_for_startup(self) -> None:
        port = self._require_connected()
        deadline = time.monotonic() + self.config.startup_timeout_sec
        required = set(self.startup_banner_lines)
        seen: set[str] = set()
        while time.monotonic() < deadline:
            raw = port.readline()
            if not raw:
                continue
            line = raw.decode("utf-8", errors="replace").strip() if isinstance(raw, bytes) else str(raw).strip()
            self.logger.info("[CONVEYOR RX] %s", line)
            seen.add(line)
            if line == "========================" and required.issubset(seen):
                self._startup_ready = True
                return
        missing = sorted(required - seen)
        raise ConveyorTimeoutError(
            "conveyor startup banner/readiness was not received before timeout"
            f" (missing={missing!r})"
        )

    def move_steps(self, direction: str, steps: int) -> None:
        port = self._require_connected()
        if not self._startup_ready:
            raise ConveyorError("conveyor serial startup is not ready")
        command = format_move_command(direction, steps)
        self.discard_stale_input()
        port.write(command.encode("ascii"))
        if hasattr(port, "flush"):
            port.flush()
        self._moving = True
        self._pending_ack = format_move_ack(direction, steps)
        self.logger.info(
            "[CONVEYOR] command=%s (pulse-count positioning; no external conveyor position sensor)",
            command.strip(),
        )

    def move_to_inspection(self) -> None:
        self.move_steps(self.config.inspection_direction, self.config.inspection_steps)

    def move_out(self) -> None:
        self.move_steps(self.config.exit_direction, self.config.exit_steps)

    def wait_until_stopped(self, timeout: float | None = None) -> None:
        if not self._moving:
            raise ConveyorError("no conveyor movement is pending")
        port = self._require_connected()
        wait = self.config.timeout_sec if timeout is None else float(timeout)
        if not math.isfinite(wait) or wait <= 0:
            raise ConveyorConfigurationError("timeout must be positive and finite")
        deadline = time.monotonic() + wait
        acknowledged = False
        while time.monotonic() < deadline:
            raw = port.readline()
            if not raw:
                continue
            line = raw.decode("utf-8", errors="replace").strip() if isinstance(raw, bytes) else str(raw).strip()
            self.logger.info("[CONVEYOR RX] %s", line)
            if line == self._pending_ack:
                acknowledged = True
                continue
            if acknowledged and self.completion_marker in line:
                self._moving = False
                self._pending_ack = None
                return
        raise ConveyorTimeoutError(
            "conveyor command echo and post-echo completion were not both received "
            f"before timeout (expected echo={self._pending_ack!r}, "
            f"completion={self.completion_marker!r})"
        )
