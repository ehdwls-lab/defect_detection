from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import cv2


PROJECT_ROOT = Path(__file__).resolve().parents[1]
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
WINDOW_NAME = "Select normal training regions"


def resolve_path(path: Path) -> Path:
    if path.is_absolute():
        return path
    working_directory_path = Path.cwd() / path
    if working_directory_path.exists():
        return working_directory_path.resolve()
    return (PROJECT_ROOT / path).resolve()


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="이미지별 정상 학습 rectangular region 선택"
    )
    parser.add_argument("--images", nargs="+", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=PROJECT_ROOT / "data" / "regions")
    parser.add_argument("--session", default=None, help="모든 JSON에 사용할 session 이름")
    return parser.parse_args()


def resolve_images(inputs: list[Path]) -> list[Path]:
    images: list[Path] = []
    for input_path in inputs:
        path = resolve_path(input_path)
        if path.is_dir():
            images.extend(
                candidate for candidate in sorted(path.rglob("*"))
                if candidate.is_file() and candidate.suffix.lower() in IMAGE_EXTENSIONS
            )
        elif path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            images.append(path)
        else:
            raise FileNotFoundError(f"이미지가 없습니다: {input_path}")
    return list(dict.fromkeys(path.resolve() for path in images))


def normalize_rectangle(start: tuple[int, int], end: tuple[int, int]) -> dict[str, int] | None:
    x1, x2 = sorted((start[0], end[0]))
    y1, y2 = sorted((start[1], end[1]))
    if x2 - x1 < 2 or y2 - y1 < 2:
        return None
    return {"x1": x1, "y1": y1, "x2": x2, "y2": y2}


def region_patch_positions(
    region: dict[str, int],
    patch_size: int = 64,
    stride: int = 32,
) -> list[tuple[int, int]]:
    return [
        (x, y)
        for y in range(region["y1"], region["y2"] - patch_size + 1, stride)
        for x in range(region["x1"], region["x2"] - patch_size + 1, stride)
    ]


def print_region_summary(regions: list[dict[str, int]]) -> None:
    unique_positions: set[tuple[int, int]] = set()
    for index, region in enumerate(regions, start=1):
        width = region["x2"] - region["x1"]
        height = region["y2"] - region["y1"]
        positions = region_patch_positions(region)
        unique_positions.update(positions)
        print(
            f"  region {index}: width={width}, height={height}, "
            f"patches={len(positions)}"
        )
        if not positions:
            print("    WARNING: 이 region에서는 patch가 생성되지 않습니다.")
    print(f"  total unique patches: {len(unique_positions)}")


def select_regions(image_path: Path) -> list[dict[str, int]] | None:
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"이미지를 읽지 못했습니다: {image_path}")

    regions: list[dict[str, int]] = []
    state: dict[str, object] = {"start": None, "current": None, "drawing": False}

    def on_mouse(event: int, x: int, y: int, _flags: int, _param: object) -> None:
        if event == cv2.EVENT_LBUTTONDOWN:
            state["start"] = (x, y)
            state["current"] = (x, y)
            state["drawing"] = True
        elif event == cv2.EVENT_MOUSEMOVE and state["drawing"]:
            state["current"] = (x, y)
        elif event == cv2.EVENT_LBUTTONUP and state["drawing"]:
            state["current"] = (x, y)
            region = normalize_rectangle(state["start"], state["current"])
            if region is not None:
                regions.append(region)
            state["drawing"] = False

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(WINDOW_NAME, on_mouse)
    while True:
        display = image.copy()
        for index, region in enumerate(regions, start=1):
            cv2.rectangle(display, (region["x1"], region["y1"]), (region["x2"], region["y2"]), (0, 255, 0), 2)
            cv2.putText(display, str(index), (region["x1"] + 5, region["y1"] + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        if state["drawing"]:
            start = state["start"]
            current = state["current"]
            cv2.rectangle(display, start, current, (0, 255, 255), 2)
        cv2.putText(display, "Drag=add  U=undo  S=save/next  K=skip  ESC=exit", (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
        cv2.imshow(WINDOW_NAME, display)
        key = cv2.waitKey(20) & 0xFF
        if key == 27:
            cv2.destroyAllWindows()
            return None
        if key in (ord("u"), ord("U")) and regions:
            regions.pop()
        elif key in (ord("s"), ord("S")):
            cv2.destroyWindow(WINDOW_NAME)
            return regions
        elif key in (ord("k"), ord("K")):
            cv2.destroyWindow(WINDOW_NAME)
            return []


def main() -> None:
    args = parse_arguments()
    output_root = resolve_path(args.output_root)
    images = resolve_images(args.images)
    for image_path in images:
        regions = select_regions(image_path)
        if regions is None:
            break
        print(f"{image_path}: selected regions={len(regions)}")
        print_region_summary(regions)
        session = args.session or image_path.parent.name
        output_path = output_root / session / f"{image_path.stem}.json"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            stored_path = image_path.relative_to(PROJECT_ROOT).as_posix()
        except ValueError:
            stored_path = str(image_path)
        output_path.write_text(
            json.dumps({"image_path": stored_path, "regions": regions, "created_at": datetime.now().isoformat(timespec="seconds")}, indent=4),
            encoding="utf-8",
        )
        print(f"saved: {output_path} ({len(regions)} regions)")


if __name__ == "__main__":
    main()