from __future__ import annotations

import math
import time

from .types import PlatformPoseCommand, PlatformTelemetry


class PlatformProtocolError(ValueError):
    pass


class MalformedTelemetryError(PlatformProtocolError):
    pass


def _finite(value: float, name: str) -> float:
    value = float(value)
    if not math.isfinite(value):
        raise PlatformProtocolError(f"{name} must be finite")
    return value


def format_pose_command(command: PlatformPoseCommand | None = None, *, z_cm: float | None = None,
                        roll_deg: float | None = None, pitch_deg: float | None = None) -> str:
    if command is not None:
        if any(value is not None for value in (z_cm, roll_deg, pitch_deg)):
            raise PlatformProtocolError("pass a command or numeric fields, not both")
        z_cm, roll_deg, pitch_deg = command.z_cm, command.roll_deg, command.pitch_deg
    if z_cm is None or roll_deg is None or pitch_deg is None:
        raise PlatformProtocolError("z_cm, roll_deg and pitch_deg are required")
    return f"Z:{_finite(z_cm, 'z_cm'):.2f} R:{_finite(roll_deg, 'roll_deg'):.2f} P:{_finite(pitch_deg, 'pitch_deg'):.2f}\r\n"


def format_z_command(z_cm: float) -> str:
    return f"Z:{_finite(z_cm, 'z_cm'):.2f}\r\n"


def format_orientation_command(roll_deg: float, pitch_deg: float) -> str:
    return f"R:{_finite(roll_deg, 'roll_deg'):.2f} P:{_finite(pitch_deg, 'pitch_deg'):.2f}\r\n"


def format_reset_command() -> str:
    return "RST\r\n"


def format_mode_command(mode: int) -> str:
    if isinstance(mode, bool) or int(mode) != mode or int(mode) not in (0, 1, 2):
        raise PlatformProtocolError("mode must be 0 (P), 1 (PD), or 2 (PID)")
    return f"MODE:{int(mode)}\r\n"


def parse_telemetry(line: str, *, timestamp: float | None = None) -> PlatformTelemetry:
    raw = line.strip()
    if not raw.startswith("TLM:"):
        raise MalformedTelemetryError("telemetry must start with 'TLM:'")
    fields: dict[str, str] = {}
    for item in raw[4:].split(","):
        if "=" not in item:
            raise MalformedTelemetryError(f"malformed telemetry field: {item!r}")
        key, value = item.split("=", 1)
        if not key or key in fields:
            raise MalformedTelemetryError(f"invalid or duplicate telemetry key: {key!r}")
        fields[key] = value
    required = {"Z", "R", "P", "S", "M1", "M2", "M3", "H", "G", "C", "VR", "VP"}
    missing = required - fields.keys()
    if missing:
        raise MalformedTelemetryError(f"missing telemetry fields: {sorted(missing)}")
    try:
        ints = {key: int(fields[key]) for key in ("S", "M1", "M2", "M3", "H", "G", "C")}
        floats = {key: float(fields[key]) for key in ("Z", "R", "P", "VR", "VP")}
    except ValueError as exc:
        raise MalformedTelemetryError("telemetry contains a non-numeric value") from exc
    if ints["S"] not in (0, 1) or ints["H"] not in (0, 1):
        raise MalformedTelemetryError("S and H must be 0 or 1")
    if not all(math.isfinite(value) for value in floats.values()):
        raise MalformedTelemetryError("telemetry float values must be finite")
    return PlatformTelemetry(
        z_cm=floats["Z"], roll_deg=floats["R"], pitch_deg=floats["P"],
        stable=bool(ints["S"]), homing=bool(ints["H"]),
        motor1=ints["M1"], motor2=ints["M2"], motor3=ints["M3"],
        imu_mode=ints["G"], control_mode=ints["C"],
        roll_rate_deg_s=floats["VR"], pitch_rate_deg_s=floats["VP"],
        timestamp=time.time() if timestamp is None else timestamp,
    )
