"""Pure-data adapter from production artifacts to dashboard view models."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class PoseView:
    index: int
    name: str
    status: str
    judgement: str
    classification: str
    score: float | None
    threshold: float | None
    roll: float | None
    pitch: float | None
    z: float | None
    rgb: Path | None
    roi: Path | None
    mask: Path | None
    heatmap: Path | None
    overlay: Path | None
    patch_overlay: Path | None
    board_overlay: Path | None
    depth: Path | None
    failure: str | None


@dataclass(frozen=True)
class InspectionView:
    run_dir: Path
    run_name: str
    product: str
    status: str
    judgement: str
    stage: str
    started_at: str | None
    finished_at: str | None
    stages: tuple[str, ...]
    poses: tuple[PoseView, ...]
    transport_complete: bool
    ply: Path | None


def _path(value: Any, run_dir: Path) -> Path | None:
    if not value:
        return None
    candidate = Path(str(value)).expanduser()
    return candidate if candidate.is_absolute() else run_dir / candidate


def _first(mapping: dict[str, Any], *names: str) -> Any:
    for name in names:
        value = mapping.get(name)
        if value is not None:
            return value
    return None


def load_inspection_view(run: str | Path) -> InspectionView:
    run_dir = Path(run).expanduser().resolve()
    result_path = run_dir / "cycle_result.json"
    if not result_path.is_file():
        raise FileNotFoundError(f"missing production result: {result_path}")
    try:
        data = json.loads(result_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"malformed production result: {result_path}: {exc}") from exc
    if not isinstance(data, dict):
        raise ValueError(f"production result must be a JSON object: {result_path}")

    poses = []
    for index, plane in enumerate(data.get("inspection_planes") or []):
        anomaly = plane.get("anomaly_result") or {}
        metadata = anomaly.get("metadata") or {}
        plane_root = run_dir / f"plane_{index:02d}"
        def artifact(value: Any, fallback: Path) -> Path | None:
            selected = _path(value, run_dir)
            return selected if selected is not None else (fallback if fallback.is_file() else None)
        failure = plane.get("failure_reason")
        poses.append(PoseView(
            index=index,
            name=str(_first(plane, "plane_name", "pose_id") or f"Pose {index + 1}"),
            status=str(plane.get("status") or "UNKNOWN"),
            judgement=str(_first(plane, "inspection_judgement", "quality_judgement") or "RECHECK"),
            classification=str(_first(plane, "classification") or anomaly.get("classification") or "N/A"),
            score=_first(plane, "anomaly_score") if plane.get("anomaly_score") is not None else anomaly.get("score"),
            threshold=_first(plane, "anomaly_threshold") if plane.get("anomaly_threshold") is not None else anomaly.get("threshold"),
            roll=_first(plane, "actual_platform_roll_deg", "selected_roll", "commanded_roll_deg"),
            pitch=_first(plane, "actual_platform_pitch_deg", "selected_pitch", "commanded_pitch_deg"),
            z=_first(plane, "actual_platform_z_cm", "best_z", "selected_z"),
            rgb=artifact(_first(plane, "final_rgb_path") or metadata.get("rgb_path"), plane_root / "final_capture/final_rgb.png"),
            roi=artifact(plane.get("inspection_mask_overlay_path"), plane_root / "final_capture/inspection_mask_overlay.png"),
            mask=artifact(plane.get("inspection_mask_path") or metadata.get("inspection_mask_path"), plane_root / "final_capture/inspection_mask.png"),
            heatmap=artifact(_first(plane, "anomaly_heatmap_path") or anomaly.get("heatmap_path"), plane_root / "anomaly/anomaly_heatmap.png"),
            overlay=artifact(metadata.get("overlay_path"), plane_root / "anomaly/anomaly_overlay.png"),
            patch_overlay=artifact(_first(plane, "surface_patch_overlay_path") or metadata.get("surface_patch_overlay_path"), plane_root / "anomaly/surface_patch_overlay.png"),
            board_overlay=artifact(None, plane_root / "final_capture/board_plane_overlay.png"),
            depth=artifact(_first(plane, "final_depth_path") or metadata.get("depth_path"), plane_root / "final_capture/final_depth.npy"),
            failure=str(failure) if failure else None,
        ))
    status = str(_first(data, "inspection_status", "execution_status", "overall_status") or "UNKNOWN")
    judgement = str(_first(data, "final_judgement", "quality_judgement") or "RECHECK")
    encoded = json.dumps(data, ensure_ascii=False).lower()
    product = str(_first(data, "product", "material", "profile") or
                  ("GRAY" if "gray_" in encoded else "BLUE" if "blue_" in encoded else "--")).upper()
    ply_candidates = sorted(run_dir.glob("structured_light/**/*.ply"))
    preferred_ply = next((path for path in ply_candidates if "dominant_plane_segmented" in path.name),
                         ply_candidates[0] if ply_candidates else None)
    return InspectionView(
        run_dir=run_dir, run_name=run_dir.name, product=product, status=status, judgement=judgement,
        stage=str(data.get("stage") or status), started_at=data.get("started_at"),
        finished_at=data.get("finished_at"), stages=tuple(data.get("stage_history") or ()),
        poses=tuple(poses), transport_complete=bool(
            data.get("cycle_transport_complete", data.get("conveyor_out_executed", False))
        ), ply=preferred_ply,
    )


def display_judgement(judgement: str) -> str:
    return "NORMAL" if judgement in ("OK", "NORMAL") else "DEFECT" if judgement in ("NG", "DEFECT") else "RECHECK"
