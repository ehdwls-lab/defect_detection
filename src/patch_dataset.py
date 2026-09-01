from __future__ import annotations

import csv
import json
import random
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset

try:
    from src.preprocessing import preprocess_anomaly
except ModuleNotFoundError:
    from preprocessing import preprocess_anomaly


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SUPPORTED_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".tif",
    ".tiff",
}


def find_image_files(root: str | Path) -> list[Path]:
    """
    지정된 폴더 아래의 이미지 파일을 재귀적으로 검색함.
    """
    root = Path(root)

    if not root.exists():
        raise FileNotFoundError(f"데이터 폴더가 존재하지 않습니다: {root}")

    image_paths = sorted(
        path
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
    )

    if not image_paths:
        raise RuntimeError(f"이미지를 찾지 못했습니다: {root}")

    return image_paths


def split_image_paths(
    image_paths: Sequence[Path],
    val_ratio: float = 0.2,
    seed: int = 42,
) -> tuple[list[Path], list[Path]]:
    """
    이미지 단위로 학습/검증 데이터를 분리함.

    Patch를 먼저 만든 뒤 무작위로 분리하면 같은 이미지에서 잘린 Patch가
    학습과 검증 데이터에 동시에 들어갈 수 있으므로 이미지 단위로 분리함.
    """
    paths = list(image_paths)

    if not 0.0 <= val_ratio < 1.0:
        raise ValueError("val_ratio는 0 이상 1 미만이어야 합니다.")

    random_generator = random.Random(seed)
    random_generator.shuffle(paths)

    if len(paths) < 2 or val_ratio == 0:
        return paths, []

    val_count = max(1, int(len(paths) * val_ratio))
    val_count = min(val_count, len(paths) - 1)

    val_paths = paths[:val_count]
    train_paths = paths[val_count:]

    return train_paths, val_paths


def read_manifest_paths(
    manifest_path: str | Path,
    split: str,
) -> list[Path]:
    manifest_path = Path(manifest_path)
    if not manifest_path.exists():
        raise FileNotFoundError(f"manifest가 없습니다: {manifest_path}")

    paths: list[Path] = []
    with manifest_path.open("r", newline="", encoding="utf-8-sig") as csv_file:
        for row in csv.DictReader(csv_file):
            if row.get("split") != split:
                continue
            if row.get("label") != "normal" and split in {"train", "val"}:
                raise ValueError(f"{split} manifest에는 normal만 허용됩니다: {row}")
            path = Path(row["path"])
            if not path.is_absolute():
                path = PROJECT_ROOT / path
            paths.append(path)

    if not paths:
        raise RuntimeError(f"manifest에서 split={split} 이미지를 찾지 못했습니다: {manifest_path}")
    return paths


def read_manifest_entries(
    manifest_path: str | Path,
    split: str,
) -> list[dict[str, str]]:
    manifest_path = Path(manifest_path)
    if not manifest_path.exists():
        raise FileNotFoundError(f"manifest가 없습니다: {manifest_path}")
    with manifest_path.open("r", newline="", encoding="utf-8-sig") as csv_file:
        entries = [row for row in csv.DictReader(csv_file) if row.get("split") == split]
    if not entries:
        raise RuntimeError(f"manifest에서 split={split} 이미지를 찾지 못했습니다: {manifest_path}")
    return entries


def calculate_positions(length: int, patch_size: int, stride: int) -> list[int]:
    if length < patch_size:
        return []
    return list(range(0, length - patch_size + 1, stride))


class PatchDataset(Dataset[Tensor]):
    """
    이미지를 고정 크기로 변환한 뒤 grid 형태로 Patch를 추출하는 Dataset.

    OpenCV로 읽은 BGR 원본을 전처리한 뒤 Tensor 직전에 RGB로 변환함.
    추후 RGB + Depth 또는 RGB + IR + Depth 구조로 확장할 예정임.
    """

    def __init__(
        self,
        image_paths: Sequence[Path] | None = None,
        patch_size: int = 64,
        stride: int = 32,
        return_metadata: bool = False,
        manifest_path: str | Path | None = None,
        split: str | None = None,
        gamma: float = 0.82,
        clahe_clip: float = 1.5,
        unsharp_amount: float = 0.30,
        allow_full_image: bool = False,
    ) -> None:
        if patch_size <= 0:
            raise ValueError("patch_size는 양수여야 합니다.")

        if stride <= 0:
            raise ValueError("stride는 양수여야 합니다.")

        if manifest_path is not None:
            if split is None:
                raise ValueError("manifest_path를 사용하면 split이 필요합니다.")
            if image_paths is not None:
                raise ValueError("image_paths와 manifest_path를 동시에 사용할 수 없습니다.")
            manifest_entries = read_manifest_entries(manifest_path, split)
            image_paths = []
            for entry in manifest_entries:
                if split in {"train", "val"} and entry.get("label") != "normal":
                    raise ValueError(f"{split} manifest에는 normal만 허용됩니다: {entry}")
                path = Path(entry["path"])
                image_paths.append(path if path.is_absolute() else PROJECT_ROOT / path)
        elif image_paths is None:
            raise ValueError("image_paths 또는 manifest_path가 필요합니다.")
        else:
            manifest_entries = [
                {"path": str(path), "regions": ""}
                for path in image_paths
            ]

        self.image_paths = list(image_paths)
        self.patch_size = patch_size
        self.stride = stride
        self.return_metadata = return_metadata
        self.preprocessing_params = {
            "gamma": gamma,
            "clahe_clip": clahe_clip,
            "unsharp_amount": unsharp_amount,
        }

        # 각 Patch가 어느 이미지의 어느 좌표인지 저장
        self.samples: list[tuple[Path, int, int, int]] = []

        for image_index, image_path in enumerate(self.image_paths):
            image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            if image is None:
                raise RuntimeError(f"이미지를 읽지 못했습니다: {image_path}")
            height, width = image.shape[:2]
            entry = manifest_entries[image_index]
            region_value = entry.get("regions", "").strip()
            if region_value:
                region_path = Path(region_value)
                if not region_path.is_absolute():
                    region_path = PROJECT_ROOT / region_path
                if not region_path.exists():
                    raise FileNotFoundError(f"region JSON이 없습니다: {region_path}")
                regions = json.loads(region_path.read_text(encoding="utf-8")).get("regions", [])
            elif allow_full_image:
                regions = [{"x1": 0, "y1": 0, "x2": width, "y2": height}]
            else:
                raise ValueError(
                    f"train/val 이미지에 region JSON이 없습니다: {image_path}. "
                    "명시적으로 allow_full_image=True를 사용해야 전체 이미지를 허용합니다."
                )

            coordinates: set[tuple[int, int]] = set()
            for region_index, region in enumerate(regions):
                x1 = max(0, min(width, int(region["x1"])))
                y1 = max(0, min(height, int(region["y1"])))
                x2 = max(0, min(width, int(region["x2"])))
                y2 = max(0, min(height, int(region["y2"])))
                for top in range(y1, y2 - patch_size + 1, stride):
                    for left in range(x1, x2 - patch_size + 1, stride):
                        if (left, top) in coordinates:
                            continue
                        coordinates.add((left, top))
                        self.samples.append((image_path, region_index, left, top))

        if not self.samples:
            raise RuntimeError("생성된 Patch가 없습니다. 크기 설정을 확인하세요.")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        image_path, region_index, left, top = self.samples[index]

        try:
            image_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            if image_bgr is None:
                raise RuntimeError("이미지를 읽지 못했습니다.")
            processed_bgr = preprocess_anomaly(
                image_bgr,
                **self.preprocessing_params,
            )
            patch_bgr = processed_bgr[
                top:top + self.patch_size,
                left:left + self.patch_size,
            ]
            patch_rgb = cv2.cvtColor(patch_bgr, cv2.COLOR_BGR2RGB)
            patch_array = np.ascontiguousarray(patch_rgb.transpose(2, 0, 1))
            patch_tensor = torch.from_numpy(patch_array).float() / 255.0

        except Exception as error:
            raise RuntimeError(
                f"이미지 처리 중 오류가 발생했습니다: {image_path}"
            ) from error

        if not self.return_metadata:
            return patch_tensor

        metadata = {
            "source_path": str(image_path),
            "region_index": region_index,
            "x": left,
            "y": top,
        }
        return patch_tensor, metadata