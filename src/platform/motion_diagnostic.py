from __future__ import annotations

import csv
import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable

from .serial_controller import PlatformTelemetryTimeout
from .types import PlatformTelemetry


class MotionDiagnosticError(RuntimeError):
    pass


Z_TARGET_REACHED_TOLERANCE_CM = 0.25
ORIENTATION_TARGET_REACHED_TOLERANCE_DEG = 0.25


@dataclass(frozen=True)
class MotionWaitConfig:
    """Diagnostic timing knobs, not calibrated production safety limits."""

    post_command_guard_s: float = 0.05
    stable_sample_count: int = 3
    deadband_observation_s: float = 0.20
    fresh_read_settle_s: float = 0.10
    z_target_tolerance_cm: float = Z_TARGET_REACHED_TOLERANCE_CM
    orientation_target_tolerance_deg: float = ORIENTATION_TARGET_REACHED_TOLERANCE_DEG

    def validate(self) -> None:
        for name, value in (
            ("post_command_guard_s", self.post_command_guard_s),
            ("deadband_observation_s", self.deadband_observation_s),
            ("fresh_read_settle_s", self.fresh_read_settle_s),
            ("z_target_tolerance_cm", self.z_target_tolerance_cm),
            ("orientation_target_tolerance_deg", self.orientation_target_tolerance_deg),
        ):
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and non-negative")
        if (isinstance(self.stable_sample_count, bool)
                or not isinstance(self.stable_sample_count, int)
                or self.stable_sample_count <= 0):
            raise ValueError("stable_sample_count must be a positive integer")


class TelemetryLog:
    def __init__(self) -> None:
        self.records: list[dict[str, Any]] = []

    def add(self, stage: str, telemetry: PlatformTelemetry) -> None:
        item = asdict(telemetry)
        item["roll_rate"] = item.pop("roll_rate_deg_s")
        item["pitch_rate"] = item.pop("pitch_rate_deg_s")
        item["stage"] = stage
        item["command"] = None
        item["timestamp"] = telemetry.timestamp if telemetry.timestamp is not None else time.time()
        self.records.append(item)

    def add_command(self, command: str) -> None:
        if not self.records:
            raise MotionDiagnosticError("cannot record a command before initial telemetry")
        item = dict(self.records[-1])
        item["stage"] = "command"
        item["command"] = command
        item["timestamp"] = time.time()
        self.records.append(item)

    def save(self, path: str | Path) -> None:
        target = Path(path).expanduser()
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.suffix.lower() == ".json":
            target.write_text(json.dumps(self.records, indent=2), encoding="utf-8")
            return
        if target.suffix.lower() == ".csv":
            with target.open("w", newline="", encoding="utf-8") as handle:
                fields = list(self.records[0]) if self.records else ["stage", "timestamp"]
                writer = csv.DictWriter(handle, fieldnames=fields)
                writer.writeheader()
                writer.writerows(self.records)
            return
        raise MotionDiagnosticError("telemetry log path must end in .json or .csv")


def finite_optional(value: float | None, name: str) -> float | None:
    if value is None:
        return None
    if not math.isfinite(value):
        raise MotionDiagnosticError(f"{name} must be finite")
    return value


class PlatformMotionDiagnostic:
    """Explicitly gated STM motion diagnostic, separate from SystemController."""

    def __init__(self, controller: Any, *, timeout_s: float = 10.0,
                 wait_config: MotionWaitConfig | None = None,
                 confirm: Callable[[str], bool] | None = None) -> None:
        if not math.isfinite(timeout_s) or timeout_s <= 0:
            raise ValueError("timeout_s must be positive and finite")
        self.controller = controller
        self.timeout_s = timeout_s
        self.wait_config = wait_config or MotionWaitConfig()
        self.wait_config.validate()
        if self.wait_config.fresh_read_settle_s >= self.timeout_s:
            raise ValueError("fresh_read_settle_s must be less than timeout_s")
        self.confirm = confirm or (lambda prompt: input(prompt).strip() == "EXECUTE")
        self.log = TelemetryLog()

    def read_before(self) -> PlatformTelemetry:
        telemetry = self.controller.read_fresh_telemetry(
            self.timeout_s, settle_s=self.wait_config.fresh_read_settle_s,
        )
        self.log.add("before", telemetry)
        return telemetry

    def wait_stable(
        self, stage: str, *, command_sent_at: float,
        target_z_cm: float | None = None,
    ) -> PlatformTelemetry:
        deadline = command_sent_at + self.timeout_s
        guard_until = command_sent_at + self.wait_config.post_command_guard_s
        deadband_fallback_at = guard_until + self.wait_config.deadband_observation_s
        last: PlatformTelemetry | None = None
        completion_transition_seen = False
        consecutive_stable = 0
        # A reset immediately after write is not enough for USB/CDC packets that
        # are still in flight. Reuse the same reset -> settle -> reset boundary
        # as a fresh read before any packet may contribute to completion.
        boundary_read_timeout = deadline - time.monotonic() - self.wait_config.fresh_read_settle_s
        if boundary_read_timeout <= 0:
            raise PlatformTelemetryTimeout(
                "fresh telemetry settle interval leaves no post-command read time"
            )
        try:
            pending: PlatformTelemetry | None = self.controller.read_fresh_telemetry(
                boundary_read_timeout,
                settle_s=self.wait_config.fresh_read_settle_s,
            )
        except PlatformTelemetryTimeout as exc:
            raise PlatformTelemetryTimeout(
                "no fresh telemetry after post-command RX boundary"
            ) from exc
        while time.monotonic() < deadline:
            if pending is not None:
                last = pending
                pending = None
            else:
                remaining = deadline - time.monotonic()
                try:
                    last = self.controller.read_telemetry(min(1.0, remaining))
                except PlatformTelemetryTimeout:
                    continue
            received_at = time.monotonic()
            if received_at < guard_until:
                self.log.add("post_command_guard", last)
                consecutive_stable = 0
                continue
            self.log.add("during", last)
            if not last.stable:
                completion_transition_seen = True
                consecutive_stable = 0
                continue
            if (
                target_z_cm is not None
                and not math.isclose(
                    float(last.z_cm), target_z_cm, rel_tol=0.0,
                    abs_tol=self.wait_config.z_target_tolerance_cm,
                )
            ):
                # Firmware S means stable, not that the commanded absolute Z
                # has been reached.  A stale/pre-motion S=1 sample must not
                # complete a Z command.
                completion_transition_seen = True
                consecutive_stable = 0
                continue
            consecutive_stable += 1
            enough_stable = consecutive_stable >= self.wait_config.stable_sample_count
            motion_observed_or_deadband = (
                completion_transition_seen or received_at >= deadband_fallback_at
            )
            if enough_stable and motion_observed_or_deadband:
                self.log.add(stage, last)
                return last
        detail = f"; last telemetry={last}" if last is not None else ""
        raise PlatformTelemetryTimeout(f"platform did not report stable before timeout{detail}")

    def _approved(self, description: str) -> None:
        if not self.confirm(
            f"Requested absolute target: {description}\n"
            "Type EXECUTE to send this command: "
        ):
            raise MotionDiagnosticError("motion cancelled; no command was sent")

    def execute_z(self, z_cm: float) -> PlatformTelemetry:
        z_cm = finite_optional(z_cm, "z_cm")
        assert z_cm is not None
        self._approved(f"Z={z_cm:.2f} cm")
        self.controller.discard_stale_input()
        self.log.add_command(f"Z:{z_cm:.2f}")
        self.controller.move_z(z_cm)
        command_sent_at = time.monotonic()
        return self.wait_stable(
            "after_z", command_sent_at=command_sent_at, target_z_cm=z_cm,
        )

    def execute_orientation(self, *, roll_deg: float | None, pitch_deg: float | None,
                            before: PlatformTelemetry, ack_safe_height: bool) -> PlatformTelemetry:
        if not ack_safe_height:
            raise MotionDiagnosticError("orientation requires --ack-safe-height")
        self.log.add("before_orientation", before)
        roll = before.roll_deg if roll_deg is None else finite_optional(roll_deg, "roll_deg")
        pitch = before.pitch_deg if pitch_deg is None else finite_optional(pitch_deg, "pitch_deg")
        assert roll is not None and pitch is not None
        self._approved(f"R={roll:.2f} deg P={pitch:.2f} deg; safe height acknowledged")
        self.controller.discard_stale_input()
        self.log.add_command(f"R:{roll:.2f} P:{pitch:.2f}")
        self.controller.move_orientation(roll, pitch)
        command_sent_at = time.monotonic()
        return self.wait_stable("after_orientation", command_sent_at=command_sent_at)

    def execute_pose(self, *, safe_z_cm: float, roll_deg: float, pitch_deg: float,
                     ack_safe_height: bool) -> PlatformTelemetry:
        if not ack_safe_height:
            raise MotionDiagnosticError("pose execution requires --ack-safe-height")
        # Canonical host order: explicit user-provided safe Z, stable, then R/P.
        before_orientation = self.execute_z(safe_z_cm)
        return self.execute_orientation(
            roll_deg=roll_deg, pitch_deg=pitch_deg,
            before=before_orientation, ack_safe_height=True,
        )


class DiagnosticZMover:
    """Expose an initialized motion diagnostic through the Automatic-Z mover API."""

    def __init__(self, diagnostic: PlatformMotionDiagnostic) -> None:
        self.diagnostic = diagnostic

    def move_and_wait(self, z_cm: float, timeout_s: float) -> PlatformTelemetry:
        # timeout_s is already represented by the shared diagnostic's validated
        # timeout. Keeping one diagnostic preserves one telemetry log and the
        # same fresh-boundary/stable policy for every candidate and best return.
        if not math.isfinite(timeout_s) or timeout_s <= 0:
            raise ValueError("timeout_s must be positive and finite")
        if not math.isclose(
            timeout_s, self.diagnostic.timeout_s, rel_tol=0.0, abs_tol=1e-12,
        ):
            raise ValueError("Automatic Z timeout must match the shared motion diagnostic timeout")
        return self.diagnostic.execute_z(z_cm)
