from __future__ import annotations

import logging
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol


@dataclass(frozen=True)
class StructuredLightRunInfo:
    run_id: str
    result_directory: Path
    return_code: int = 0
    stdout: str = ""
    stderr: str = ""
    mock: bool = False


class StructuredLightRunner(Protocol):
    def run_scan(self) -> StructuredLightRunInfo: ...


class StructuredLightPreflightError(RuntimeError):
    pass


@dataclass(frozen=True)
class ShellStructuredLightConfig:
    subsystem_root: Path
    result_root: Path
    script_name: str = "물체검사.sh"
    timeout_sec: float = 900.0


class ShellStructuredLightRunner:
    """Explicit shell boundary for the unmodified external subsystem."""

    def __init__(self, config: ShellStructuredLightConfig) -> None:
        self.config = config

    def preflight(self) -> list[str]:
        issues: list[str] = []
        root = self.config.subsystem_root.expanduser().resolve()
        script = root / self.config.script_name
        if not root.is_dir():
            issues.append(f"subsystem root does not exist: {root}")
        if not script.is_file():
            issues.append(f"scan script does not exist: {script}")
        if not self.config.result_root.expanduser().is_dir():
            issues.append(f"result root does not exist: {self.config.result_root.expanduser()}")
        if script.is_file():
            text = script.read_text(encoding="utf-8", errors="replace")
            for line in text.splitlines():
                if line.strip().startswith("BASE="):
                    configured = line.split("=", 1)[1].strip().strip('"').strip("'")
                    if configured and not Path(configured).is_dir():
                        issues.append(f"script BASE path does not exist on this PC: {configured}")
                    break
        if self.config.timeout_sec <= 0:
            issues.append("timeout_sec must be positive")
        return issues

    def run_scan(self) -> StructuredLightRunInfo:
        issues = self.preflight()
        if issues:
            raise StructuredLightPreflightError("; ".join(issues))
        root = self.config.subsystem_root.expanduser().resolve()
        completed = subprocess.run(
            ["bash", str(root / self.config.script_name)],
            cwd=root,
            text=True,
            capture_output=True,
            timeout=self.config.timeout_sec,
            check=False,
        )
        if completed.returncode != 0:
            raise RuntimeError(
                f"structured-light scan failed rc={completed.returncode}: {completed.stderr.strip()}"
            )
        run_dirs = sorted(
            (path for path in self.config.result_root.expanduser().glob("촬영_*") if path.is_dir()),
            key=lambda path: path.stat().st_mtime,
        )
        if not run_dirs:
            raise RuntimeError("structured-light scan completed but no 촬영_* result directory was found")
        directory = run_dirs[-1]
        return StructuredLightRunInfo(
            run_id=directory.name, result_directory=directory,
            return_code=completed.returncode, stdout=completed.stdout,
            stderr=completed.stderr, mock=False,
        )


class MockStructuredLightRunner:
    def __init__(self, result_directory: Path | None = None, *, fail: bool = False) -> None:
        self.result_directory = result_directory
        self.fail = fail

    def run_scan(self) -> StructuredLightRunInfo:
        if self.fail:
            raise RuntimeError("Mock structured-light scan failure")
        directory = self.result_directory or Path(tempfile.mkdtemp(prefix="mock_structured_light_"))
        directory.mkdir(parents=True, exist_ok=True)
        ply = directory / "03_v2_현재프레임기준_최종_물체만.ply"
        if not ply.exists():
            ply.write_text(
                "ply\nformat ascii 1.0\nelement vertex 3\n"
                "property float x\nproperty float y\nproperty float z\n"
                "property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n"
                "0 0 0 255 0 0\n1 0 0 255 0 0\n0 1 0 255 0 0\n",
                encoding="ascii",
            )
        logging.getLogger(__name__).info("[STRUCTURED] mock result ready: %s", directory)
        return StructuredLightRunInfo("mock_scan", directory, stdout="mock scan complete", mock=True)
