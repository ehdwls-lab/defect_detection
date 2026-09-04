from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys
from typing import Iterable

import cv2
import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.core.patch_extractor import measure_patchability
from src.infer_anomaly import select_patch_positions


FIELDNAMES = (
    "path", "split", "label", "session", "mask", "source_run",
    "plane", "material", "view", "notes",
)


def _run_name(run: Path) -> str:
    return run.expanduser().resolve().name


def _plane_rows(run: Path, split: str, session: str, material: str):
    run = run.expanduser().resolve()
    planes = sorted(run.glob("plane_*/final_capture/final_rgb.png"))
    if not planes:
        raise FileNotFoundError(f"no final RGB planes found under {run}")
    for rgb_path in planes:
        plane_dir = rgb_path.parent.parent
        mask_path = plane_dir / "anomaly" / "inspection_mask.png"
        if not mask_path.is_file():
            raise FileNotFoundError(f"missing final anomaly mask: {mask_path}")
        image = cv2.imread(str(rgb_path), cv2.IMREAD_COLOR)
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if image is None or mask is None or image.shape[:2] != mask.shape:
            raise ValueError(f"RGB/mask mismatch: {rgb_path} / {mask_path}")
        positions = select_patch_positions(
            image.shape, 64, 32, surface_mask=mask, min_surface_coverage=1.0,
        )
        _, _, patchable_ratio = measure_patchability(mask, 64, 32, 1.0)
        plane_name = plane_dir.name
        print(
            f"{_run_name(run)} {plane_name} "
            f"inspection_mask_area={int(np.count_nonzero(mask))} "
            f"patches={len(positions)} coverage=1.0 "
            f"patchable_ratio={patchable_ratio:.6f}"
        )
        if not positions:
            raise ValueError(f"zero production patches: {run} {plane_name}")
        yield {
            "path": str(rgb_path), "split": split, "label": "normal",
            "session": session, "mask": str(mask_path),
            "source_run": _run_name(run), "plane": plane_name,
            "material": material, "view": "production_pose",
            "notes": f"patches={len(positions)};coverage=1.0",
        }, image, mask, positions


def build_manifests(
    train_runs: Iterable[str | Path],
    val_runs: Iterable[str | Path],
    output_dir: str | Path,
    *,
    session: str,
    material: str,
) -> tuple[Path, Path]:
    train_paths = {Path(value).expanduser().resolve() for value in train_runs}
    val_paths = {Path(value).expanduser().resolve() for value in val_runs}
    overlap = train_paths & val_paths
    if overlap:
        raise ValueError("the same run cannot be present in train and val: " + ", ".join(map(str, sorted(overlap))))
    output = Path(output_dir).expanduser().resolve()
    debug = output / "debug"
    output.mkdir(parents=True, exist_ok=True)
    debug.mkdir(parents=True, exist_ok=True)
    rows: dict[str, list[dict[str, str]]] = {"train": [], "val": []}
    for split, runs in (("train", sorted(train_paths)), ("val", sorted(val_paths))):
        for run in runs:
            for row, image, mask, positions in _plane_rows(run, split, session, material):
                rows[split].append(row)
                overlay = image.copy()
                contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(overlay, contours, -1, (0, 255, 0), 2)
                for x, y in positions:
                    cv2.rectangle(overlay, (x, y), (x + 63, y + 63), (255, 255, 0), 1)
                target = debug / f"{row['source_run']}_{row['plane']}_overlay.png"
                if not cv2.imwrite(str(target), overlay):
                    raise RuntimeError(f"failed to save debug overlay: {target}")
    for split, split_rows in rows.items():
        target = output / f"{split}.csv"
        with target.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
            writer.writeheader()
            writer.writerows(split_rows)
        print(f"{split}: {len(split_rows)} source planes -> {target}")
    if not rows["train"] or not rows["val"]:
        raise ValueError("both train and val must contain at least one production plane")
    return output / "train.csv", output / "val.csv"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build normal manifests from production final RGB and anomaly masks")
    parser.add_argument("--session", required=True)
    parser.add_argument("--material", required=True, choices=("gray",))
    parser.add_argument("--train-runs", nargs="+", required=True)
    parser.add_argument("--val-runs", nargs="+", required=True)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args(argv)
    build_manifests(
        args.train_runs, args.val_runs, args.output_dir,
        session=args.session, material=args.material,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
