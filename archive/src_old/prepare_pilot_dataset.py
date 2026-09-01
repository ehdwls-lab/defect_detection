from __future__ import annotations

import argparse
import csv
import random
import shutil
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp"}


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="파일럿 데이터셋 원본 단위 분할"
    )

    parser.add_argument(
        "--normal-dir",
        type=Path,
        required=True,
        help="정상 원본 이미지가 들어 있는 세션 폴더",
    )

    parser.add_argument(
        "--defect-dir",
        type=Path,
        required=True,
        help="결함 원본 이미지가 들어 있는 세션 폴더",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "pilot_v1",
        help="분할 결과 저장 폴더",
    )

    parser.add_argument(
        "--exclude-name",
        type=str,
        default="normal_0018",
        help="정상 학습에서 제외할 파일명 일부",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    return parser.parse_args()


def find_images(directory: Path) -> list[Path]:
    if not directory.exists():
        raise FileNotFoundError(
            f"폴더가 존재하지 않습니다: {directory}"
        )

    image_paths = [
        path
        for path in directory.rglob("*")
        if (
            path.is_file()
            and path.suffix.lower() in IMAGE_EXTENSIONS
        )
    ]

    return sorted(image_paths)


def copy_group(
    image_paths: list[Path],
    output_dir: Path,
    split_name: str,
    manifest_rows: list[dict[str, str]],
) -> None:
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    for source_path in image_paths:
        destination_path = (
            output_dir
            / source_path.name
        )

        shutil.copy2(
            source_path,
            destination_path,
        )

        manifest_rows.append(
            {
                "split": split_name,
                "source": str(source_path),
                "destination": str(destination_path),
            }
        )


def main() -> None:
    args = parse_arguments()

    normal_paths = find_images(
        args.normal_dir
    )

    defect_paths = find_images(
        args.defect_dir
    )

    excluded_paths = [
        path
        for path in normal_paths
        if args.exclude_name in path.name
    ]

    usable_normal_paths = [
        path
        for path in normal_paths
        if args.exclude_name not in path.name
    ]

    print("=" * 70)
    print("파일럿 데이터셋 준비")
    print(f"정상 원본: {len(normal_paths)}장")
    print(f"제외 정상: {len(excluded_paths)}장")
    print(f"사용 정상: {len(usable_normal_paths)}장")
    print(f"결함 원본: {len(defect_paths)}장")
    print("=" * 70)

    if len(usable_normal_paths) != 19:
        raise RuntimeError(
            "사용 가능한 정상 이미지가 "
            f"19장이 아닙니다: {len(usable_normal_paths)}장\n"
            f"(전체 {len(normal_paths)}장, "
            f"제외 {len(excluded_paths)}장)"
        )

    if len(defect_paths) != 5:
        raise RuntimeError(
            "선택한 결함 폴더의 이미지가 "
            f"5장이 아닙니다: {len(defect_paths)}장"
        )

    if (
        args.output_dir.exists()
        and any(args.output_dir.rglob("*"))
    ):
        raise RuntimeError(
            "출력 폴더가 비어 있지 않습니다: "
            f"{args.output_dir}\n"
            "기존 파일을 덮어쓰지 않도록 중단했습니다."
        )

    shuffled_normal_paths = (
        usable_normal_paths.copy()
    )

    random_generator = random.Random(
        args.seed
    )

    random_generator.shuffle(
        shuffled_normal_paths
    )

    train_paths = shuffled_normal_paths[:13]
    val_paths = shuffled_normal_paths[13:16]
    test_normal_paths = shuffled_normal_paths[16:19]

    original_root = (
        args.output_dir
        / "original"
    )

    manifest_rows: list[dict[str, str]] = []

    copy_group(
        train_paths,
        original_root / "train" / "normal",
        "train_normal",
        manifest_rows,
    )

    copy_group(
        val_paths,
        original_root / "val" / "normal",
        "val_normal",
        manifest_rows,
    )

    copy_group(
        test_normal_paths,
        original_root / "test" / "normal",
        "test_normal",
        manifest_rows,
    )

    copy_group(
        defect_paths,
        original_root / "test" / "defect",
        "test_defect",
        manifest_rows,
    )

    copy_group(
        excluded_paths,
        original_root / "excluded",
        "excluded_normal",
        manifest_rows,
    )

    manifest_path = (
        args.output_dir
        / "split_manifest.csv"
    )

    with manifest_path.open(
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=[
                "split",
                "source",
                "destination",
            ],
        )

        writer.writeheader()
        writer.writerows(manifest_rows)

    print()
    print("분할 완료")
    print(f"Train 정상      : {len(train_paths)}장")
    print(f"Validation 정상 : {len(val_paths)}장")
    print(f"Test 정상       : {len(test_normal_paths)}장")
    print(f"Test 결함       : {len(defect_paths)}장")
    print(f"제외 정상       : {len(excluded_paths)}장")
    print(f"저장 위치       : {args.output_dir}")
    print(f"분할 기록       : {manifest_path}")


if __name__ == "__main__":
    main()
