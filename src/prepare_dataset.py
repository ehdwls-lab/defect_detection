from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
FIELDNAMES = ["path", "split", "label", "session", "regions", "material", "view", "notes"]


def resolve_path(path: Path) -> Path:
    if path.is_absolute():
        return path
    working_directory_path = Path.cwd() / path
    if working_directory_path.exists():
        return working_directory_path.resolve()
    return (PROJECT_ROOT / path).resolve()


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="명시적으로 선택한 원본 이미지로 anomaly detection manifest 생성"
    )
    parser.add_argument("--train-normal", nargs="+", type=Path)
    parser.add_argument("--val-normal", nargs="+", type=Path)
    parser.add_argument("--normal-session", type=Path)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--test-normal", nargs="*", type=Path, default=[])
    parser.add_argument("--test-defect", nargs="*", type=Path, default=[])
    parser.add_argument(
        "--manifest-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "manifests",
    )
    parser.add_argument("--regions-root", type=Path, default=PROJECT_ROOT / "data" / "regions")
    parser.add_argument("--material", default="")
    parser.add_argument("--view", default="")
    parser.add_argument("--notes", default="")
    return parser.parse_args()


def expand_inputs(inputs: list[Path]) -> list[Path]:
    paths: list[Path] = []
    for input_path in inputs:
        path = resolve_path(input_path)
        if path.is_dir():
            paths.extend(
                candidate
                for candidate in sorted(path.rglob("*"))
                if candidate.is_file() and candidate.suffix.lower() in IMAGE_EXTENSIONS
            )
        elif path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            paths.append(path)
        else:
            raise FileNotFoundError(f"이미지 파일 또는 폴더가 없습니다: {input_path}")
    unique_paths = list(dict.fromkeys(path.resolve() for path in paths))
    if not unique_paths:
        raise RuntimeError("선택된 이미지가 없습니다.")
    return unique_paths


def make_rows(
    paths: list[Path],
    split: str,
    label: str,
    material: str,
    view: str,
    notes: str,
    regions_root: Path,
) -> list[dict[str, str]]:
    return [
        {
            "path": path.relative_to(PROJECT_ROOT).as_posix(),
            "split": split,
            "label": label,
            "session": path.parent.name,
            "regions": (
                region_path(regions_root, path).relative_to(PROJECT_ROOT).as_posix()
                if split in {"train", "val"} and region_path(regions_root, path).is_relative_to(PROJECT_ROOT)
                and region_path(regions_root, path).exists()
                else str(region_path(regions_root, path))
                if split in {"train", "val"} and region_path(regions_root, path).exists()
                else ""
            ),
            "material": material,
            "view": view,
            "notes": notes,
        }
        for path in paths
    ]


def region_path(regions_root: Path, image_path: Path) -> Path:
    return regions_root / image_path.parent.name / f"{image_path.stem}.json"


def validate_rows(rows: list[dict[str, str]]) -> None:
    paths_by_split = {
        split: {row["path"] for row in rows if row["split"] == split}
        for split in {row["split"] for row in rows}
    }
    if paths_by_split.get("train", set()) & paths_by_split.get("val", set()):
        raise ValueError("train과 val source image가 중복됩니다.")
    for row in rows:
        if row["split"] in {"train", "val"} and row["label"] != "normal":
            raise ValueError("train/val에는 normal label만 허용됩니다.")
        if row["split"] == "test" and row["label"] not in {"normal", "defect"}:
            raise ValueError("test label은 normal 또는 defect여야 합니다.")


def write_manifest(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_arguments()
    regions_root = resolve_path(args.regions_root)
    if args.normal_session and (args.train_normal or args.val_normal):
        raise ValueError("normal-session과 train-normal/val-normal은 함께 사용할 수 없습니다.")
    if not args.normal_session and (not args.train_normal or not args.val_normal):
        raise ValueError("normal-session 또는 train-normal과 val-normal을 지정해야 합니다.")

    if args.normal_session:
        if not 0.0 < args.val_ratio < 1.0:
            raise ValueError("session 모드의 val-ratio는 0과 1 사이여야 합니다.")
        session_path = resolve_path(args.normal_session)
        session_images = expand_inputs([session_path])
        candidates = [
            path for path in session_images
            if (regions_root / path.parent.name / f"{path.stem}.json").exists()
        ]
        if len(candidates) < 2:
            raise RuntimeError("region JSON이 있는 session 이미지가 2장 이상 필요합니다.")
        random_generator = random.Random(args.seed)
        random_generator.shuffle(candidates)
        val_count = max(1, int(len(candidates) * args.val_ratio))
        val_count = min(val_count, len(candidates) - 1)
        train_paths = candidates[val_count:]
        val_paths = candidates[:val_count]
        print(f"session candidates: {len(candidates)} images (region JSON 있는 이미지)")
        print(f"split seed: {args.seed}, val_ratio: {args.val_ratio}")
        for split, paths in (("train", train_paths), ("val", val_paths)):
            print(f"[{split}]")
            for path in paths:
                print(f"  {path}")
    else:
        train_paths = expand_inputs(args.train_normal)
        val_paths = expand_inputs(args.val_normal)

    grouped = {
        "train": (train_paths, "normal"),
        "val": (val_paths, "normal"),
        "test_normal": (expand_inputs(args.test_normal), "normal") if args.test_normal else ([], "normal"),
        "test_defect": (expand_inputs(args.test_defect), "defect") if args.test_defect else ([], "defect"),
    }
    rows: list[dict[str, str]] = []
    for group, (paths, label) in grouped.items():
        split = "test" if group.startswith("test_") else group
        rows.extend(make_rows(paths, split, label, args.material, args.view, args.notes, regions_root))

    validate_rows(rows)
    for split in ("train", "val", "test"):
        write_manifest(
            args.manifest_dir / f"{split}.csv",
            [row for row in rows if row["split"] == split],
        )

    print(f"manifest 저장: {args.manifest_dir}")
    for split in ("train", "val", "test"):
        split_rows = [row for row in rows if row["split"] == split]
        print(f"{split}: {len(split_rows)} images")


if __name__ == "__main__":
    main()