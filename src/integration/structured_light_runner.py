from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
import tempfile
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Protocol


class StructuredLightStatus(str, Enum):
    READY = "READY"
    CALIBRATION_REQUIRED = "CALIBRATION_REQUIRED"
    SOURCE_ERROR = "SOURCE_ERROR"
    ENVIRONMENT_ERROR = "ENVIRONMENT_ERROR"


@dataclass(frozen=True)
class StructuredLightPreflightReport:
    overall_status: StructuredLightStatus
    source_ready: bool
    environment_ready: bool
    calibration_ready: bool
    issues: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()
    python_path: Path | None = None

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["overall_status"] = self.overall_status.value
        value["python_path"] = str(self.python_path) if self.python_path else None
        return value


@dataclass(frozen=True)
class StructuredLightRunInfo:
    run_id: str
    result_directory: Path
    return_code: int = 0
    stdout: str = ""
    stderr: str = ""
    mock: bool = False
    manifest_path: Path | None = None


class StructuredLightRunner(Protocol):
    def run_scan(self) -> StructuredLightRunInfo: ...


class StructuredLightPreflightError(RuntimeError):
    pass


@dataclass(frozen=True)
class ShellStructuredLightConfig:
    subsystem_root: Path
    result_root: Path
    script_name: str = "물체검사.sh"
    python_path: Path | None = None
    timeout_sec: float = 900.0
    non_interactive: bool = True
    visualize: bool = False


class ShellStructuredLightRunner:
    REQUIRED_SOURCES = (
        "물체검사.sh", "structured_light_paths.py", "structured_light_projector.py",
        "구조광_전처리_최종_v2_현재프레임플랫폼기준_Depth홀위상보강_경로수정_0822.py",
        "구조광_전처리_최종_v2_현재프레임플랫폼기준_Depth홀위상보강_경로수정_0822 (1).py",
        "make_dc_grabcut_object_mask_latest.py",
        "make_phase_relative_reunwrap_holefilled_0822.py",
        "add_platform_floor_latest.py", "0823_test.py",
        "초기세팅.sh",
        "프로젝터_XY범위_수동2점_설정_경로수정_0822.py",
        "현재배치_빈플랫폼_Depth_E1999_G64_촬영_경로수정_0822.py",
    )

    def __init__(self, config: ShellStructuredLightConfig) -> None:
        self.config = config

    def _root(self) -> Path:
        return self.config.subsystem_root.expanduser().resolve()

    def _python(self) -> Path | None:
        if self.config.python_path is not None:
            return self.config.python_path.expanduser().absolute()
        configured = os.environ.get("STRUCTURED_LIGHT_PYTHON")
        if configured:
            return Path(configured).expanduser().absolute()
        repository_venv = self._root().parent / ".venv" / "bin" / "python"
        if repository_venv.is_file():
            return repository_venv.absolute()
        executable = shutil.which("python3")
        return Path(executable).resolve() if executable else None

    def preflight_report(self) -> StructuredLightPreflightReport:
        root = self._root()
        source_issues: list[str] = []
        environment_issues: list[str] = []
        warnings: list[str] = []
        if not root.is_dir():
            source_issues.append(f"subsystem root does not exist: {root}")
        else:
            for name in self.REQUIRED_SOURCES:
                if not (root / name).is_file():
                    source_issues.append(f"required source does not exist: {root / name}")
            script = root / self.config.script_name
            if script.is_file():
                for line in script.read_text(encoding="utf-8", errors="replace").splitlines():
                    if line.strip().startswith("BASE="):
                        configured = line.split("=", 1)[1].strip().strip('"').strip("'")
                        if configured.startswith("/") and not Path(configured).is_dir():
                            source_issues.append(f"script BASE path does not exist on this PC: {configured}")
                        break
        python_path = self._python()
        if python_path is None or not python_path.is_file() or not os.access(python_path, os.X_OK):
            environment_issues.append(f"Python interpreter is not executable: {python_path}")
        if self.config.timeout_sec <= 0:
            environment_issues.append("timeout_sec must be positive")
        result_root = self.config.result_root.expanduser().resolve()
        output_parent = result_root
        while not output_parent.exists() and output_parent != output_parent.parent:
            output_parent = output_parent.parent
        if not output_parent.exists() or not os.access(output_parent, os.W_OK):
            environment_issues.append(f"result root is not writable: {result_root}")
        if not os.environ.get("DISPLAY"):
            warnings.append("DISPLAY is not set; real capture/projector GUI requires a desktop session")
        if not (shutil.which("CloudCompare") or shutil.which("cloudcompare")):
            warnings.append("CloudCompare not found (optional visualization only)")
        calibration_files = (
            root / "프로젝터 수동 범위 확인" / "프로젝터_세로범위.json",
            root / "플랫폼 바닥 따기" / "현재배치_기준데이터" / "active" / "E1999_G64" / "플랫폼_바닥_depth.npy",
        )
        missing = [str(path) for path in calibration_files if not path.is_file()]
        calibration_ready = not missing
        warnings.extend(f"calibration required: {path}" for path in missing)
        source_ready = not source_issues
        environment_ready = not environment_issues
        if not source_ready:
            status = StructuredLightStatus.SOURCE_ERROR
        elif not environment_ready:
            status = StructuredLightStatus.ENVIRONMENT_ERROR
        elif not calibration_ready:
            status = StructuredLightStatus.CALIBRATION_REQUIRED
        else:
            status = StructuredLightStatus.READY
        return StructuredLightPreflightReport(
            status, source_ready, environment_ready, calibration_ready,
            tuple(source_issues + environment_issues), tuple(warnings), python_path,
        )

    def preflight(self) -> list[str]:
        """Backward-compatible fatal issue list; calibration absence is nonfatal here."""
        return list(self.preflight_report().issues)

    @staticmethod
    def _latest(directory: Path, pattern: str) -> Path | None:
        candidates = list(directory.rglob(pattern))
        return max(candidates, key=lambda path: path.stat().st_mtime) if candidates else None

    def _write_manifest(self, directory: Path, started_at: str, finished_at: str, return_code: int) -> Path:
        artifact_patterns = {
            "integrated_object_only_ply": "03_v2_현재프레임기준_최종_물체만.ply",
            "phase_object_only_ply": "FINAL_DC_MASK_PHASE*.ply",
            "with_floor_ply": "*WITH_FLOOR.ply",
            "segmented_ply": "*_dominant_plane_segmented.ply",
        }
        artifacts: dict[str, str | None] = {}
        for key, pattern in artifact_patterns.items():
            candidates = list(directory.rglob(pattern))
            if key == "phase_object_only_ply":
                candidates = [path for path in candidates if "WITH_FLOOR" not in path.name]
            selected = max(candidates, key=lambda p: p.stat().st_mtime) if candidates else None
            artifacts[key] = str(selected.resolve()) if selected else None
        root = self._root()
        manifest = {
            "schema_version": "structured_light_run_v1", "run_id": directory.name,
            "started_at": started_at, "finished_at": finished_at, "root": str(root),
            "result_directory": str(directory.resolve()), "return_code": return_code,
            "calibration": {
                "projector_range": str(root / "프로젝터 수동 범위 확인" / "프로젝터_세로범위.json"),
                "platform_depth": str(root / "플랫폼 바닥 따기" / "현재배치_기준데이터" / "active" / "E1999_G64" / "플랫폼_바닥_depth.npy"),
            },
            "artifacts": artifacts,
            "ply_metadata": {
                "integrated_object_only_ply": {"coordinate_contract": "image_centered_phase_relative", "z_sign": -1, "z_scale": 40, "metric_z": False},
                "phase_object_only_ply": {"coordinate_contract": "image_centered_phase_relative", "z_sign": 1, "z_scale": 30, "metric_z": False},
            },
        }
        path = directory / "structured_light_manifest.json"
        path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        return path

    def run_scan(self) -> StructuredLightRunInfo:
        report = self.preflight_report()
        if report.overall_status is not StructuredLightStatus.READY:
            details = "; ".join((*report.issues, *report.warnings))
            raise StructuredLightPreflightError(f"{report.overall_status.value}: {details}")
        root = self._root()
        result_root = self.config.result_root.expanduser().resolve()
        result_root.mkdir(parents=True, exist_ok=True)
        started_at = datetime.now(timezone.utc).isoformat()
        env = os.environ.copy()
        env.update({
            "STRUCTURED_LIGHT_ROOT": str(root), "STRUCTURED_LIGHT_PYTHON": str(report.python_path),
            "STRUCTURED_LIGHT_NON_INTERACTIVE": "1" if self.config.non_interactive else "0",
            "STRUCTURED_LIGHT_VISUALIZE": "1" if self.config.visualize else "0",
        })
        completed = subprocess.run(
            ["bash", str(root / self.config.script_name)], cwd=root, env=env,
            text=True, capture_output=True, timeout=self.config.timeout_sec, check=False,
        )
        finished_at = datetime.now(timezone.utc).isoformat()
        if completed.returncode != 0:
            raise RuntimeError(f"structured-light scan failed rc={completed.returncode}: {completed.stderr.strip()}")
        runs = [path for path in result_root.glob("촬영_*") if path.is_dir()]
        if not runs:
            raise RuntimeError("structured-light scan completed but no 촬영_* result directory was found")
        directory = max(runs, key=lambda path: path.stat().st_mtime)
        manifest_path = self._write_manifest(directory, started_at, finished_at, completed.returncode)
        return StructuredLightRunInfo(
            directory.name, directory, completed.returncode, completed.stdout,
            completed.stderr, False, manifest_path,
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
                "ply\nformat ascii 1.0\nelement vertex 3\nproperty float x\nproperty float y\n"
                "property float z\nproperty uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n"
                "0 0 0 255 0 0\n1 0 0 255 0 0\n0 1 0 255 0 0\n", encoding="ascii",
            )
        logging.getLogger(__name__).info("[STRUCTURED] mock result ready: %s", directory)
        return StructuredLightRunInfo("mock_scan", directory, stdout="mock scan complete", mock=True)
