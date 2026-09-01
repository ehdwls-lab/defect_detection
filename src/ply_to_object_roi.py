from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "팀원 구조광 PLY의 X/Y 좌표를 1280x800 영상 좌표로 되돌려 "
            "물체 외곽 ROI와 결함검사용 inspection mask를 생성한다."
        )
    )

    parser.add_argument("ply", type=Path)
    parser.add_argument("--image", type=Path, default=None)

    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=800)

    parser.add_argument(
        "--mode",
        choices=("auto", "color", "z", "all"),
        default="auto",
        help=(
            "auto: 플랫폼 회색(220,220,220)이 있으면 color 방식 우선, "
            "없으면 Z 기반, 그것도 애매하면 모든 PLY point 사용"
        ),
    )

    # 팀원 현재프레임 PLY에서 플랫폼을 (220,220,220)으로 저장
    parser.add_argument("--platform-gray", type=int, default=220)
    parser.add_argument("--platform-color-tol", type=int, default=12)

    # Z 기반 fallback.
    # Z는 실제 mm가 아니라 상대 위상 기반 값이므로 절대 mm 의미가 아니다.
    parser.add_argument(
        "--z-threshold",
        type=float,
        default=None,
        help=(
            "플랫폼 기준 Z와의 절대 차이 threshold. "
            "미지정 시 robust MAD 기반 자동 추정"
        ),
    )
    parser.add_argument(
        "--z-sigma",
        type=float,
        default=5.0,
        help="자동 Z threshold = max(3*MAD sigma floor, z_sigma * sigma)",
    )

    parser.add_argument(
        "--point-radius",
        type=int,
        default=0,
        help=(
            "PLY point를 2D mask에 찍을 반경. "
            "0이면 X/Y spacing으로 자동 추정"
        ),
    )
    parser.add_argument("--close-size", type=int, default=19)
    parser.add_argument("--close-iterations", type=int, default=2)
    parser.add_argument("--open-size", type=int, default=3)
    parser.add_argument("--min-area", type=int, default=1500)

    parser.add_argument(
        "--epsilon-ratio",
        type=float,
        default=0.0015,
        help="최종 외곽 contour smoothing 정도",
    )
    parser.add_argument(
        "--inspection-erode",
        type=int,
        default=10,
        help="결함검사용 ROI를 외곽에서 안쪽으로 줄일 px",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/ply_roi"),
    )

    return parser.parse_args()


def read_ascii_ply(path: Path):
    """
    x y z [red green blue] ASCII PLY를 읽는다.
    property 순서를 header에서 찾아서 유연하게 처리한다.
    """
    path = Path(path)

    properties = []
    vertex_count = None
    header_lines = []

    with path.open("r", encoding="ascii", errors="strict") as f:
        while True:
            line = f.readline()

            if not line:
                raise RuntimeError("PLY header 끝을 찾지 못했습니다.")

            line = line.strip()
            header_lines.append(line)

            if line.startswith("format ") and "ascii" not in line:
                raise RuntimeError(
                    "현재 코드는 ASCII PLY 전용입니다. "
                    f"format line: {line}"
                )

            if line.startswith("element vertex "):
                vertex_count = int(line.split()[-1])

            elif line.startswith("property "):
                parts = line.split()

                # vertex property라고 가정.
                # list property는 현재 팀원 PLY에 없으므로 제외.
                if len(parts) >= 3 and parts[1] != "list":
                    properties.append(parts[-1])

            elif line == "end_header":
                break

        if vertex_count is None:
            raise RuntimeError("element vertex를 찾지 못했습니다.")

        required = ("x", "y", "z")

        for name in required:
            if name not in properties:
                raise RuntimeError(
                    f"PLY에 {name} property가 없습니다. properties={properties}"
                )

        rows = []

        for _ in range(vertex_count):
            line = f.readline()

            if not line:
                break

            parts = line.split()

            if len(parts) < len(properties):
                continue

            rows.append(parts[:len(properties)])

    if not rows:
        raise RuntimeError("PLY vertex data가 없습니다.")

    index = {
        name: i
        for i, name in enumerate(properties)
    }

    x = np.asarray(
        [float(row[index["x"]]) for row in rows],
        dtype=np.float32,
    )
    y = np.asarray(
        [float(row[index["y"]]) for row in rows],
        dtype=np.float32,
    )
    z = np.asarray(
        [float(row[index["z"]]) for row in rows],
        dtype=np.float32,
    )

    rgb = None

    if all(
        name in index
        for name in ("red", "green", "blue")
    ):
        rgb = np.asarray(
            [
                [
                    int(float(row[index["red"]])),
                    int(float(row[index["green"]])),
                    int(float(row[index["blue"]])),
                ]
                for row in rows
            ],
            dtype=np.uint8,
        )

    return {
        "x": x,
        "y": y,
        "z": z,
        "rgb": rgb,
        "properties": properties,
        "declared_vertex_count": vertex_count,
        "read_vertex_count": len(rows),
        "header": header_lines,
    }


def ply_xy_to_pixel(x, y, width, height):
    """
    팀원 PLY 생성식:
      X = x_pixel - width/2
      Y = height/2 - y_pixel

    따라서 역변환:
      x_pixel = X + width/2
      y_pixel = height/2 - Y
    """
    px = np.rint(
        x + width / 2.0
    ).astype(np.int32)

    py = np.rint(
        height / 2.0 - y
    ).astype(np.int32)

    return px, py


def estimate_spacing(values):
    unique = np.unique(
        np.rint(values).astype(np.int32)
    )

    if unique.size < 2:
        return None

    diff = np.diff(
        np.sort(unique)
    )

    diff = diff[
        diff > 0
    ]

    if diff.size == 0:
        return None

    # 큰 gap보다 반복 sampling 간격을 잡기 위해 낮은 percentile 사용
    return float(
        np.percentile(
            diff,
            25,
        )
    )


def choose_object_points(data, args):
    x = data["x"]
    y = data["y"]
    z = data["z"]
    rgb = data["rgb"]

    info = {
        "requested_mode": args.mode,
    }

    # -------------------------------------------------------------
    # 1. Color 방식
    # 현재프레임 PLY: platform=(220,220,220), object=TURBO colormap
    # -------------------------------------------------------------
    platform_by_color = None

    if rgb is not None:
        gray = float(
            args.platform_gray
        )

        color_distance = np.max(
            np.abs(
                rgb.astype(np.int16)
                - int(round(gray))
            ),
            axis=1,
        )

        platform_by_color = (
            color_distance
            <= int(args.platform_color_tol)
        )

        platform_color_ratio = float(
            np.mean(
                platform_by_color
            )
        )

        info["platform_color_ratio"] = platform_color_ratio
    else:
        platform_color_ratio = 0.0

    if args.mode == "color":
        if platform_by_color is None:
            raise RuntimeError(
                "--mode color인데 PLY에 RGB property가 없습니다."
            )

        object_mask = ~platform_by_color
        info["used_mode"] = "color"

        return object_mask, info

    if (
        args.mode == "auto"
        and platform_by_color is not None
        and platform_color_ratio >= 0.05
    ):
        object_mask = ~platform_by_color
        info["used_mode"] = "color"
        return object_mask, info

    # -------------------------------------------------------------
    # 2. Z 방식
    # 현재프레임 PLY는 플랫폼이 상대높이 0 근처여야 함.
    # 실제 mm가 아니라 상대 위상 단위.
    # -------------------------------------------------------------
    finite = np.isfinite(z)

    if np.count_nonzero(finite) < 20:
        if args.mode == "z":
            raise RuntimeError("Z 유효 point가 너무 적습니다.")

        info["used_mode"] = "all"
        return finite, info

    z_valid = z[finite]

    # 0 부근 플랫폼이 충분히 있으면 center=0을 우선 사용.
    # 없으면 가장 밀집된 robust median 사용.
    z_abs_p20 = float(
        np.percentile(
            np.abs(z_valid),
            20,
        )
    )

    if z_abs_p20 < max(
        1e-6,
        0.10
        * float(
            np.percentile(
                np.abs(z_valid),
                90,
            )
            + 1e-9
        ),
    ):
        center = 0.0
        center_source = "zero"
    else:
        center = float(
            np.median(
                z_valid
            )
        )
        center_source = "median"

    residual = np.abs(
        z - center
    )

    med_res = float(
        np.median(
            residual[finite]
        )
    )

    mad = float(
        np.median(
            np.abs(
                residual[finite]
                - med_res
            )
        )
    )

    sigma = max(
        1e-6,
        1.4826 * mad,
    )

    if args.z_threshold is not None:
        threshold = float(
            args.z_threshold
        )
    else:
        # 최소 floor를 전체 Z 폭의 1%로 설정
        z_range = float(
            np.percentile(
                z_valid,
                95,
            )
            - np.percentile(
                z_valid,
                5,
            )
        )

        floor = max(
            1e-6,
            0.01 * abs(z_range),
        )

        threshold = max(
            floor,
            args.z_sigma * sigma,
        )

    object_by_z = (
        finite
        & (
            residual >= threshold
        )
    )

    info.update(
        {
            "z_center": center,
            "z_center_source": center_source,
            "z_threshold": threshold,
            "z_sigma_est": sigma,
            "z_object_ratio": float(
                np.mean(
                    object_by_z
                )
            ),
        }
    )

    if args.mode == "z":
        info["used_mode"] = "z"
        return object_by_z, info

    if (
        args.mode == "auto"
        and np.count_nonzero(
            object_by_z
        )
        >= 50
        and 0.005
        <= np.mean(
            object_by_z
        )
        <= 0.80
    ):
        info["used_mode"] = "z"
        return object_by_z, info

    # -------------------------------------------------------------
    # 3. PLY 자체가 이미 object-only일 가능성
    # -------------------------------------------------------------
    info["used_mode"] = "all"
    return finite, info


def points_to_sparse_mask(
    px,
    py,
    selected,
    width,
    height,
    radius,
):
    mask = np.zeros(
        (
            height,
            width,
        ),
        dtype=np.uint8,
    )

    inside = (
        selected
        & (px >= 0)
        & (px < width)
        & (py >= 0)
        & (py < height)
    )

    pts = np.column_stack(
        [
            px[inside],
            py[inside],
        ]
    )

    for x, y in pts:
        cv2.circle(
            mask,
            (
                int(x),
                int(y),
            ),
            int(radius),
            255,
            thickness=-1,
        )

    return mask, pts


def largest_external_silhouette(
    sparse_mask,
    args,
):
    mask = sparse_mask.copy()

    if args.open_size > 1:
        k = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (
                args.open_size
                if args.open_size % 2 == 1
                else args.open_size + 1,
            )
            * 2,
        )

        mask = cv2.morphologyEx(
            mask,
            cv2.MORPH_OPEN,
            k,
            iterations=1,
        )

    close_size = (
        args.close_size
        if args.close_size % 2 == 1
        else args.close_size + 1
    )

    k_close = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (
            close_size,
            close_size,
        ),
    )

    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_CLOSE,
        k_close,
        iterations=max(
            1,
            int(args.close_iterations),
        ),
    )

    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_NONE,
    )

    contours = [
        contour
        for contour in contours
        if cv2.contourArea(
            contour
        )
        >= args.min_area
    ]

    if not contours:
        return None, mask, None

    contour = max(
        contours,
        key=cv2.contourArea,
    )

    perimeter = cv2.arcLength(
        contour,
        True,
    )

    epsilon = (
        float(args.epsilon_ratio)
        * perimeter
    )

    if epsilon > 0:
        contour = cv2.approxPolyDP(
            contour,
            epsilon,
            True,
        )

    final_mask = np.zeros_like(
        mask
    )

    # 내부 hole은 ROI 목적상 전부 무시하고 외곽 내부 fill
    cv2.drawContours(
        final_mask,
        [contour],
        -1,
        255,
        thickness=cv2.FILLED,
    )

    inspection_mask = final_mask.copy()

    erode_px = max(
        0,
        int(args.inspection_erode),
    )

    if erode_px > 0:
        k_erode = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (
                2 * erode_px + 1,
                2 * erode_px + 1,
            ),
        )

        inspection_mask = cv2.erode(
            inspection_mask,
            k_erode,
            iterations=1,
        )

    return (
        final_mask,
        mask,
        inspection_mask,
    )


def render_ply_points(
    px,
    py,
    rgb,
    width,
    height,
):
    canvas = np.zeros(
        (
            height,
            width,
            3,
        ),
        dtype=np.uint8,
    )

    inside = (
        (px >= 0)
        & (px < width)
        & (py >= 0)
        & (py < height)
    )

    if rgb is None:
        canvas[
            py[inside],
            px[inside],
        ] = (
            220,
            220,
            220,
        )
    else:
        # OpenCV canvas는 BGR
        canvas[
            py[inside],
            px[inside],
        ] = rgb[
            inside,
            ::-1,
        ]

    return canvas


def main():
    args = parse_args()

    data = read_ascii_ply(
        args.ply
    )

    x = data["x"]
    y = data["y"]
    z = data["z"]

    px, py = ply_xy_to_pixel(
        x,
        y,
        args.width,
        args.height,
    )

    selected, selection_info = (
        choose_object_points(
            data,
            args,
        )
    )

    dx = estimate_spacing(
        x
    )
    dy = estimate_spacing(
        y
    )

    spacing = max(
        value
        for value in (
            dx or 1.0,
            dy or 1.0,
        )
    )

    if args.point_radius > 0:
        radius = int(
            args.point_radius
        )
    else:
        radius = max(
            1,
            int(
                np.ceil(
                    spacing
                    * 0.55
                )
            ),
        )

    sparse_mask, object_points = points_to_sparse_mask(
        px,
        py,
        selected,
        args.width,
        args.height,
        radius,
    )

    (
        final_mask,
        connected_mask,
        inspection_mask,
    ) = largest_external_silhouette(
        sparse_mask,
        args,
    )

    if final_mask is None:
        raise RuntimeError(
            "최종 물체 contour를 찾지 못했습니다. "
            "--mode, --z-threshold, --close-size를 조정해보세요."
        )

    # background
    if (
        args.image is not None
        and args.image.exists()
    ):
        background = cv2.imread(
            str(
                args.image
            ),
            cv2.IMREAD_COLOR,
        )

        if background is None:
            raise RuntimeError(
                f"이미지를 읽지 못했습니다: {args.image}"
            )

        if (
            background.shape[1] != args.width
            or background.shape[0] != args.height
        ):
            background = cv2.resize(
                background,
                (
                    args.width,
                    args.height,
                ),
            )
    else:
        background = render_ply_points(
            px,
            py,
            data["rgb"],
            args.width,
            args.height,
        )

    overlay = background.copy()

    contours, _ = cv2.findContours(
        final_mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    cv2.drawContours(
        overlay,
        contours,
        -1,
        (
            255,
            0,
            255,
        ),
        3,
    )

    inspection_contours, _ = cv2.findContours(
        inspection_mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    cv2.drawContours(
        overlay,
        inspection_contours,
        -1,
        (
            0,
            255,
            255,
        ),
        2,
    )

    ys, xs = np.where(
        final_mask > 0
    )

    bbox = {
        "x": int(xs.min()),
        "y": int(ys.min()),
        "w": int(
            xs.max()
            - xs.min()
            + 1
        ),
        "h": int(
            ys.max()
            - ys.min()
            + 1
        ),
    }

    output_dir = Path(
        args.output_dir
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    cv2.imwrite(
        str(
            output_dir
            / "01_ply_sparse_object_points.png"
        ),
        sparse_mask,
    )

    cv2.imwrite(
        str(
            output_dir
            / "02_connected_points.png"
        ),
        connected_mask,
    )

    cv2.imwrite(
        str(
            output_dir
            / "03_object_roi_mask.png"
        ),
        final_mask,
    )

    cv2.imwrite(
        str(
            output_dir
            / "04_inspection_mask.png"
        ),
        inspection_mask,
    )

    cv2.imwrite(
        str(
            output_dir
            / "05_roi_overlay.png"
        ),
        overlay,
    )

    summary = {
        "ply": str(
            args.ply.resolve()
        ),
        "image_size": [
            args.width,
            args.height,
        ],
        "properties": data[
            "properties"
        ],
        "declared_vertex_count": data[
            "declared_vertex_count"
        ],
        "read_vertex_count": data[
            "read_vertex_count"
        ],
        "x_min": float(
            np.min(x)
        ),
        "x_max": float(
            np.max(x)
        ),
        "y_min": float(
            np.min(y)
        ),
        "y_max": float(
            np.max(y)
        ),
        "z_min": float(
            np.min(z)
        ),
        "z_max": float(
            np.max(z)
        ),
        "z_median": float(
            np.median(z)
        ),
        "estimated_x_spacing": dx,
        "estimated_y_spacing": dy,
        "point_radius": radius,
        "selected_point_count": int(
            np.count_nonzero(
                selected
            )
        ),
        "projected_object_point_count": int(
            len(
                object_points
            )
        ),
        "selection": selection_info,
        "bbox": bbox,
        "object_mask_area_px": int(
            np.count_nonzero(
                final_mask
            )
        ),
        "inspection_mask_area_px": int(
            np.count_nonzero(
                inspection_mask
            )
        ),
    }

    (
        output_dir
        / "summary.json"
    ).write_text(
        json.dumps(
            summary,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print("=" * 78)
    print("PLY -> ROI 완료")
    print(
        f"PLY points: {data['read_vertex_count']}"
    )
    print(
        f"X range: {summary['x_min']:.1f} ~ {summary['x_max']:.1f}"
    )
    print(
        f"Y range: {summary['y_min']:.1f} ~ {summary['y_max']:.1f}"
    )
    print(
        f"Z range: {summary['z_min']:.6f} ~ {summary['z_max']:.6f}"
    )
    print(
        f"ROI selection mode: {selection_info['used_mode']}"
    )
    print(
        f"Selected object points: {summary['selected_point_count']}"
    )
    print(
        f"Object bbox: {bbox}"
    )
    print(
        f"Output: {output_dir.resolve()}"
    )
    print("")
    print("PURPLE = object outer ROI")
    print("YELLOW = inspection ROI")
    print("=" * 78)


if __name__ == "__main__":
    main()
