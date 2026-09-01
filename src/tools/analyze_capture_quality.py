from __future__ import annotations

import csv
from pathlib import Path

import cv2
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "camera_test"
)

OUTPUT_CSV = (
    PROJECT_ROOT
    / "results"
    / "capture_quality.csv"
)

PREVIEW_DIR = (
    PROJECT_ROOT
    / "results"
    / "capture_quality_preview"
)

# 현재 촬영 영상에서 은색판이 위치한 대략적인 영역
# 형식: x1, y1, x2, y2
ROI = (430, 170, 920, 660)

SUPPORTED_EXTENSIONS = {
    ".png",
    ".jpg",
    ".jpeg",
    ".bmp",
}


def calculate_metrics(
    image: np.ndarray,
) -> dict[str, float]:
    """
    주어진 영상 영역의 밝기, 포화, 선명도,
    색상 편차를 계산한다.
    """

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY,
    )

    mean_gray = float(np.mean(gray))
    std_gray = float(np.std(gray))

    dark_ratio = float(
        np.mean(gray <= 20) * 100.0
    )

    bright_ratio = float(
        np.mean(gray >= 240) * 100.0
    )

    # 한 채널이라도 거의 최대값이면 포화로 판단
    channel_clipped = np.any(
        image >= 250,
        axis=2,
    )

    clipped_ratio = float(
        np.mean(channel_clipped) * 100.0
    )

    # 값이 높을수록 경계가 비교적 선명함
    laplacian = cv2.Laplacian(
        gray,
        cv2.CV_64F,
    )

    sharpness = float(
        laplacian.var()
    )

    blue_mean = float(
        np.mean(image[:, :, 0])
    )
    green_mean = float(
        np.mean(image[:, :, 1])
    )
    red_mean = float(
        np.mean(image[:, :, 2])
    )

    # 어두운 영역에서 나타나는 컬러 입자 노이즈의 대리 지표
    dark_mask = gray <= 50

    if np.count_nonzero(dark_mask) >= 100:
        image_float = image.astype(
            np.float32
        )

        blue_green = (
            image_float[:, :, 0]
            - image_float[:, :, 1]
        )

        red_green = (
            image_float[:, :, 2]
            - image_float[:, :, 1]
        )

        dark_chroma_noise = float(
            (
                np.std(blue_green[dark_mask])
                + np.std(red_green[dark_mask])
            )
            / 2.0
        )
    else:
        dark_chroma_noise = 0.0

    return {
        "mean_gray": mean_gray,
        "std_gray": std_gray,
        "dark_ratio_pct": dark_ratio,
        "bright_ratio_pct": bright_ratio,
        "clipped_ratio_pct": clipped_ratio,
        "sharpness": sharpness,
        "dark_chroma_noise": dark_chroma_noise,
        "blue_mean": blue_mean,
        "green_mean": green_mean,
        "red_mean": red_mean,
    }


def crop_roi(
    image: np.ndarray,
) -> np.ndarray:
    height, width = image.shape[:2]

    x1, y1, x2, y2 = ROI

    x1 = max(0, min(x1, width - 1))
    x2 = max(x1 + 1, min(x2, width))

    y1 = max(0, min(y1, height - 1))
    y2 = max(y1 + 1, min(y2, height))

    return image[y1:y2, x1:x2]


def save_preview(
    image: np.ndarray,
    filename: str,
) -> None:
    preview = image.copy()

    x1, y1, x2, y2 = ROI

    cv2.rectangle(
        preview,
        (x1, y1),
        (x2, y2),
        (0, 255, 0),
        2,
    )

    cv2.putText(
        preview,
        "Analysis ROI",
        (x1, max(30, y1 - 10)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        2,
        cv2.LINE_AA,
    )

    output_path = (
        PREVIEW_DIR
        / filename
    )

    cv2.imwrite(
        str(output_path),
        preview,
    )


def main() -> None:
    if not INPUT_DIR.exists():
        raise FileNotFoundError(
            f"입력 폴더가 없습니다: {INPUT_DIR}"
        )

    OUTPUT_CSV.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    PREVIEW_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    image_paths = sorted(
        path
        for path in INPUT_DIR.iterdir()
        if path.suffix.lower()
        in SUPPORTED_EXTENSIONS
    )

    if not image_paths:
        raise RuntimeError(
            f"분석할 이미지가 없습니다: {INPUT_DIR}"
        )

    rows: list[dict[str, object]] = []

    for image_path in image_paths:
        image = cv2.imread(
            str(image_path),
            cv2.IMREAD_COLOR,
        )

        if image is None:
            print(
                f"읽기 실패: {image_path.name}"
            )
            continue

        height, width = image.shape[:2]

        full_metrics = calculate_metrics(
            image
        )

        roi_image = crop_roi(
            image
        )

        roi_metrics = calculate_metrics(
            roi_image
        )

        row: dict[str, object] = {
            "filename": image_path.name,
            "width": width,
            "height": height,
        }

        for key, value in full_metrics.items():
            row[f"full_{key}"] = round(
                value,
                4,
            )

        for key, value in roi_metrics.items():
            row[f"roi_{key}"] = round(
                value,
                4,
            )

        rows.append(row)

        save_preview(
            image,
            image_path.name,
        )

    if not rows:
        raise RuntimeError(
            "정상적으로 읽은 이미지가 없습니다."
        )

    fieldnames = list(
        rows[0].keys()
    )

    with OUTPUT_CSV.open(
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=fieldnames,
        )

        writer.writeheader()
        writer.writerows(rows)

    print("=" * 80)
    print("촬영 이미지 품질 분석 완료")
    print(f"분석 이미지 수: {len(rows)}")
    print(f"CSV 저장: {OUTPUT_CSV}")
    print(f"ROI 확인 이미지: {PREVIEW_DIR}")
    print("=" * 80)

    print(
        f"{'파일명':45s}"
        f"{'ROI밝기':>10s}"
        f"{'과노출%':>10s}"
        f"{'선명도':>12s}"
        f"{'색노이즈':>12s}"
    )

    print("-" * 90)

    for row in rows:
        print(
            f"{str(row['filename']):45s}"
            f"{float(row['roi_mean_gray']):10.1f}"
            f"{float(row['roi_clipped_ratio_pct']):10.2f}"
            f"{float(row['roi_sharpness']):12.1f}"
            f"{float(row['roi_dark_chroma_noise']):12.2f}"
        )


if __name__ == "__main__":
    main()