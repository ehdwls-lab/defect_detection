from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]

IMAGE_EXTENSIONS = {
    ".png",
    ".jpg",
    ".jpeg",
    ".bmp",
}

DATA_GROUPS = (
    Path("train") / "normal",
    Path("val") / "normal",
    Path("test") / "normal",
    Path("test") / "defect",
)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "파일럿 원본 영상에서 "
            "고정 크기 정사각형 ROI 선택"
        )
    )

    parser.add_argument(
        "--input-root",
        type=Path,
        default=(
            PROJECT_ROOT
            / "data"
            / "pilot_v1"
            / "original"
        ),
        help="분할된 원본 데이터 폴더",
    )

    parser.add_argument(
        "--output-root",
        type=Path,
        default=(
            PROJECT_ROOT
            / "data"
            / "pilot_v1"
            / "roi"
        ),
        help="ROI 저장 폴더",
    )

    parser.add_argument(
        "--roi-size",
        type=int,
        default=512,
        help="저장할 정사각형 ROI 크기",
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="이미 존재하는 ROI를 다시 선택",
    )

    return parser.parse_args()


def find_images(
    directory: Path,
) -> list[Path]:
    if not directory.exists():
        raise FileNotFoundError(
            f"입력 폴더가 없습니다: {directory}"
        )

    return sorted(
        path
        for path in directory.iterdir()
        if (
            path.is_file()
            and path.suffix.lower()
            in IMAGE_EXTENSIONS
        )
    )


def crop_fixed_square(
    image: np.ndarray,
    center_x: float,
    center_y: float,
    crop_size: int,
) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    image_height, image_width = image.shape[:2]

    if (
        crop_size > image_width
        or crop_size > image_height
    ):
        raise ValueError(
            "ROI 크기가 원본 영상보다 큽니다. "
            f"영상={image_width}x{image_height}, "
            f"ROI={crop_size}x{crop_size}"
        )

    x1 = int(
        round(
            center_x
            - crop_size / 2
        )
    )

    y1 = int(
        round(
            center_y
            - crop_size / 2
        )
    )

    x1 = max(
        0,
        min(
            x1,
            image_width - crop_size,
        ),
    )

    y1 = max(
        0,
        min(
            y1,
            image_height - crop_size,
        ),
    )

    x2 = x1 + crop_size
    y2 = y1 + crop_size

    cropped = image[
        y1:y2,
        x1:x2,
    ].copy()

    return cropped, (
        x1,
        y1,
        x2,
        y2,
    )


def select_fixed_roi(
    image: np.ndarray,
    image_name: str,
    crop_size: int,
) -> tuple[
    np.ndarray,
    tuple[int, int, int, int],
] | None:
    while True:
        window_name = (
            "Select approximate ROI - "
            f"{image_name}"
        )

        print()
        print(f"현재 영상: {image_name}")
        print(
            "마우스로 원하는 영역을 대략 선택한 뒤 "
            "Enter 또는 Space를 누르세요."
        )
        print(
            "선택 영역의 중심을 기준으로 "
            f"{crop_size}x{crop_size} ROI가 만들어집니다."
        )

        x, y, width, height = cv2.selectROI(
            window_name,
            image,
            showCrosshair=True,
            fromCenter=False,
        )

        cv2.destroyWindow(
            window_name
        )

        if width == 0 or height == 0:
            print("ROI 선택이 취소되었습니다.")
            return None

        center_x = x + width / 2
        center_y = y + height / 2

        cropped, coordinates = crop_fixed_square(
            image=image,
            center_x=center_x,
            center_y=center_y,
            crop_size=crop_size,
        )

        preview = cropped.copy()

        cv2.putText(
            preview,
            "S/ENTER: save | R: reselect | Q: stop",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 0, 255),
            1,
            cv2.LINE_AA,
        )

        preview_window = (
            "ROI Preview - "
            f"{image_name}"
        )

        cv2.imshow(
            preview_window,
            preview,
        )

        key = (
            cv2.waitKey(0)
            & 0xFF
        )

        cv2.destroyWindow(
            preview_window
        )

        if key in (
            ord("s"),
            ord("S"),
            13,
            32,
        ):
            return cropped, coordinates

        if key in (
            ord("q"),
            ord("Q"),
            27,
        ):
            raise KeyboardInterrupt

        print("ROI를 다시 선택합니다.")


def main() -> None:
    args = parse_arguments()

    if args.roi_size <= 0:
        raise ValueError(
            "roi-size는 양수여야 합니다."
        )

    total_count = 0
    saved_count = 0
    skipped_count = 0

    try:
        for relative_group in DATA_GROUPS:
            source_directory = (
                args.input_root
                / relative_group
            )

            output_directory = (
                args.output_root
                / relative_group
            )

            output_directory.mkdir(
                parents=True,
                exist_ok=True,
            )

            image_paths = find_images(
                source_directory
            )

            print()
            print("=" * 70)
            print(f"그룹: {relative_group}")
            print(f"영상 수: {len(image_paths)}장")
            print("=" * 70)

            for image_path in image_paths:
                total_count += 1

                output_path = (
                    output_directory
                    / image_path.name
                )

                if (
                    output_path.exists()
                    and not args.overwrite
                ):
                    print(
                        f"이미 존재하여 건너뜀: "
                        f"{output_path}"
                    )

                    skipped_count += 1
                    continue

                image = cv2.imread(
                    str(image_path)
                )

                if image is None:
                    raise RuntimeError(
                        "영상을 읽을 수 없습니다: "
                        f"{image_path}"
                    )

                result = select_fixed_roi(
                    image=image,
                    image_name=image_path.name,
                    crop_size=args.roi_size,
                )

                if result is None:
                    print(
                        "현재 영상을 저장하지 않고 "
                        "다음 영상으로 넘어갑니다."
                    )
                    continue

                cropped, coordinates = result

                success = cv2.imwrite(
                    str(output_path),
                    cropped,
                )

                if not success:
                    raise RuntimeError(
                        "ROI 저장 실패: "
                        f"{output_path}"
                    )

                x1, y1, x2, y2 = coordinates

                print(
                    f"저장 완료: {output_path}"
                )
                print(
                    f"ROI 좌표: "
                    f"x1={x1}, y1={y1}, "
                    f"x2={x2}, y2={y2}"
                )

                saved_count += 1

    except KeyboardInterrupt:
        print()
        print("사용자가 ROI 선택을 중단했습니다.")

    finally:
        cv2.destroyAllWindows()

    print()
    print("=" * 70)
    print("ROI 선택 작업 종료")
    print(f"확인한 영상: {total_count}장")
    print(f"새로 저장: {saved_count}장")
    print(f"기존 파일 건너뜀: {skipped_count}장")
    print(f"ROI 저장 폴더: {args.output_root}")
    print("=" * 70)


if __name__ == "__main__":
    main()
