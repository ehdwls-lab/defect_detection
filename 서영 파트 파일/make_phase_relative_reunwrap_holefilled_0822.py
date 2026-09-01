#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
import argparse
from collections import deque

import cv2
import numpy as np


# ============================================================
# 기본 함수
# ============================================================

def wrap_to_pi(x):
    return (x + np.pi) % (2.0 * np.pi) - np.pi


def largest_component(mask):
    """
    여러 개의 분리된 물체가 있을 경우 가장 큰 connected component만 유지.
    """
    u8 = mask.astype(np.uint8)

    n, labels, stats, _ = cv2.connectedComponentsWithStats(
        u8,
        connectivity=8
    )

    if n <= 1:
        return mask

    best = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])

    return labels == best


# ============================================================
# ⭐ 물체 마스크 내부 hole 채우기
# ============================================================

def fill_internal_holes(mask):
    """
    물체 외부와 연결되지 않은 내부의 검은 영역만 hole로 판단하여 채운다.

    중요한 점:
    - 물체 외곽을 팽창시키지 않음
    - 외부 배경은 그대로 둠
    - 내부에 갇힌 hole만 True로 변경
    """

    mask = mask.astype(bool)

    h, w = mask.shape

    # background=True
    background = (~mask).astype(np.uint8)

    n, labels = cv2.connectedComponents(
        background,
        connectivity=8
    )

    # 이미지 테두리에 닿아 있는 background label
    # = 진짜 외부 배경
    border_labels = np.unique(
        np.concatenate([
            labels[0, :],
            labels[-1, :],
            labels[:, 0],
            labels[:, -1]
        ])
    )

    # 외부와 연결된 background
    external_background = (
        np.isin(labels, border_labels)
        &
        (background > 0)
    )

    # background인데 외부와 연결되지 않았다
    # = 물체 내부 hole
    holes = (
        (background > 0)
        &
        (~external_background)
    )

    filled = mask | holes

    return filled, holes


# ============================================================
# BFS Phase Unwrap
# ============================================================

def bfs_unwrap(wrapped, mask):

    wrapped = wrapped.astype(np.float32)

    # Phase 값이 실제 존재하면서
    # 최종 물체 마스크 안에 있는 부분
    valid = mask & np.isfinite(wrapped)

    # 가장 큰 물체만
    valid = largest_component(valid)

    h, w = wrapped.shape

    unwrapped = np.full(
        (h, w),
        np.nan,
        dtype=np.float32
    )

    visited = np.zeros(
        (h, w),
        dtype=bool
    )

    ys, xs = np.where(valid)

    if len(xs) == 0:
        raise RuntimeError(
            "unwrap할 유효 phase가 없습니다."
        )

    # --------------------------------------------------------
    # seed 위치
    # --------------------------------------------------------

    cy = int(np.median(ys))
    cx = int(np.median(xs))

    dist = (
        (ys - cy) ** 2
        +
        (xs - cx) ** 2
    )

    idx = int(np.argmin(dist))

    sy = int(ys[idx])
    sx = int(xs[idx])

    unwrapped[sy, sx] = wrapped[sy, sx]
    visited[sy, sx] = True

    q = deque()
    q.append((sy, sx))

    nbrs = [
        (-1, 0),
        (1, 0),
        (0, -1),
        (0, 1)
    ]

    # --------------------------------------------------------
    # BFS unwrap
    # --------------------------------------------------------

    while q:

        y, x = q.popleft()

        for dy, dx in nbrs:

            ny = y + dy
            nx = x + dx

            if ny < 0 or ny >= h:
                continue

            if nx < 0 or nx >= w:
                continue

            if visited[ny, nx]:
                continue

            if not valid[ny, nx]:
                continue

            diff = wrap_to_pi(
                wrapped[ny, nx]
                -
                wrapped[y, x]
            )

            unwrapped[ny, nx] = (
                unwrapped[y, x]
                +
                diff
            )

            visited[ny, nx] = True

            q.append((ny, nx))

    return unwrapped, visited


# ============================================================
# Masked Gaussian
# ============================================================

def masked_gaussian(
    data,
    valid,
    ksize=7,
    min_weight=0.15
):

    if ksize <= 1:

        out = data.copy()
        out[~valid] = np.nan

        return out

    if ksize % 2 == 0:
        ksize += 1

    valid = (
        valid
        &
        np.isfinite(data)
    )

    src = np.where(
        valid,
        data,
        0.0
    ).astype(np.float32)

    weight = valid.astype(np.float32)

    num = cv2.GaussianBlur(
        src,
        (ksize, ksize),
        0
    )

    den = cv2.GaussianBlur(
        weight,
        (ksize, ksize),
        0
    )

    out = num / (den + 1e-9)

    out[den < min_weight] = np.nan

    # 실제 valid 영역 밖은 다시 제거
    out[~valid] = np.nan

    return out.astype(np.float32)


# ============================================================
# Plane 제거
# ============================================================

def remove_plane(surface, valid):

    h, w = surface.shape

    yy, xx = np.mgrid[
        0:h,
        0:w
    ]

    valid = (
        valid
        &
        np.isfinite(surface)
    )

    if np.count_nonzero(valid) < 300:
        return surface

    x = xx[valid].astype(np.float64)
    y = yy[valid].astype(np.float64)
    z = surface[valid].astype(np.float64)

    lo = np.percentile(z, 5)
    hi = np.percentile(z, 95)

    fit = (
        (z >= lo)
        &
        (z <= hi)
    )

    A = np.column_stack([
        x[fit],
        y[fit],
        np.ones(
            np.count_nonzero(fit)
        )
    ])

    coeff, _, _, _ = np.linalg.lstsq(
        A,
        z[fit],
        rcond=None
    )

    a, b, c = coeff

    plane = (
        a * xx
        +
        b * yy
        +
        c
    )

    out = surface - plane

    out[~valid] = np.nan

    return out.astype(np.float32)


# ============================================================
# Color Preview
# ============================================================

def surface_to_rgb(
    surface,
    valid,
    low_p=5,
    high_p=95
):

    valid = (
        valid
        &
        np.isfinite(surface)
    )

    gray = np.zeros(
        surface.shape,
        dtype=np.uint8
    )

    if np.count_nonzero(valid) > 0:

        vals = surface[valid]

        lo = float(
            np.percentile(
                vals,
                low_p
            )
        )

        hi = float(
            np.percentile(
                vals,
                high_p
            )
        )

        if hi <= lo:
            hi = lo + 1e-6

        norm = np.clip(
            (surface - lo)
            /
            (hi - lo),
            0.0,
            1.0
        )

        gray[valid] = np.rint(
            norm[valid] * 255
        ).astype(np.uint8)

    bgr = cv2.applyColorMap(
        gray,
        cv2.COLORMAP_TURBO
    )

    rgb = cv2.cvtColor(
        bgr,
        cv2.COLOR_BGR2RGB
    )

    rgb[~valid] = (
        0,
        0,
        0
    )

    return rgb


# ============================================================
# PLY 저장
# ============================================================

def save_ply(
    path,
    surface,
    valid,
    color,
    z_scale,
    z_sign,
    step
):

    path = Path(path)

    path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    h, w = surface.shape

    ys, xs = np.where(
        valid
        &
        np.isfinite(surface)
    )

    step = max(
        1,
        int(step)
    )

    keep = (
        (ys % step == 0)
        &
        (xs % step == 0)
    )

    ys = ys[keep]
    xs = xs[keep]

    with path.open(
        "w",
        encoding="ascii"
    ) as f:

        f.write("ply\n")
        f.write("format ascii 1.0\n")

        f.write(
            f"element vertex {len(xs)}\n"
        )

        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")

        f.write("property uchar red\n")
        f.write("property uchar green\n")
        f.write("property uchar blue\n")

        f.write("end_header\n")

        for y, x in zip(ys, xs):

            X = float(
                x - w / 2.0
            )

            Y = float(
                h / 2.0 - y
            )

            Z = float(
                z_sign
                *
                z_scale
                *
                surface[y, x]
            )

            r, g, b = [
                int(v)
                for v
                in color[y, x]
            ]

            f.write(
                f"{X:.6f} "
                f"{Y:.6f} "
                f"{Z:.6f} "
                f"{r} {g} {b}\n"
            )

    print()
    print("PLY 저장:", path)
    print("points:", len(xs))


# ============================================================
# MAIN
# ============================================================

def main():

    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--wrapped",
        required=True
    )

    ap.add_argument(
        "--mask",
        required=True
    )

    ap.add_argument(
        "--out",
        required=True
    )

    ap.add_argument(
        "--z_scale",
        type=float,
        default=60.0
    )

    ap.add_argument(
        "--z_sign",
        type=float,
        default=-1.0
    )

    ap.add_argument(
        "--step",
        type=int,
        default=2
    )

    ap.add_argument(
        "--smooth_ksize",
        type=int,
        default=7
    )

    ap.add_argument(
        "--remove_plane",
        action="store_true"
    )

    args = ap.parse_args()


    # ========================================================
    # Load
    # ========================================================

    wrapped = np.load(
        args.wrapped
    ).astype(np.float32)

    mask_raw = np.load(
        args.mask
    ).astype(bool)


    # ========================================================
    # Shape 맞추기
    # ========================================================

    if mask_raw.shape != wrapped.shape:

        mask_raw = cv2.resize(
            mask_raw.astype(np.float32),
            (
                wrapped.shape[1],
                wrapped.shape[0]
            ),
            interpolation=cv2.INTER_NEAREST
        ).astype(bool)


    # ========================================================
    # ⭐⭐⭐ 핵심
    #
    # Depth 기반 마스크의 내부 hole만 채움
    #
    # Phase 값을 만드는 것이 아님.
    # 단순히 "이 영역도 물체"라고 허용하는 것.
    # 실제 Z는 Phase 값을 그대로 사용.
    # ========================================================

    mask_filled, filled_holes = (
        fill_internal_holes(mask_raw)
    )


    print()
    print("==============================================")
    print("MASK HOLE FILL")
    print("==============================================")

    print(
        "원본 mask 픽셀:",
        np.count_nonzero(mask_raw)
    )

    print(
        "채운 내부 hole 픽셀:",
        np.count_nonzero(filled_holes)
    )

    print(
        "최종 mask 픽셀:",
        np.count_nonzero(mask_filled)
    )


    # ========================================================
    # hole 위치에 Phase 값이 실제 존재하는지 확인
    # ========================================================

    phase_available_in_holes = (
        filled_holes
        &
        np.isfinite(wrapped)
    )

    phase_missing_in_holes = (
        filled_holes
        &
        (~np.isfinite(wrapped))
    )

    print()
    print(
        "hole 중 실제 Phase 존재:",
        np.count_nonzero(
            phase_available_in_holes
        )
    )

    print(
        "hole 중 Phase 없음:",
        np.count_nonzero(
            phase_missing_in_holes
        )
    )

    print("==============================================")


    # ========================================================
    # Phase unwrap
    # ========================================================

    unwrapped, visited = bfs_unwrap(
        wrapped,
        mask_filled
    )

    valid = (
        visited
        &
        np.isfinite(unwrapped)
    )


    # ========================================================
    # 상대 Phase
    # ========================================================

    center = float(
        np.nanmedian(
            unwrapped[valid]
        )
    )

    surface = (
        unwrapped
        -
        center
    )

    surface[~valid] = np.nan


    # ========================================================
    # Plane 제거
    # ========================================================

    if args.remove_plane:

        surface = remove_plane(
            surface,
            valid
        )

        valid = (
            valid
            &
            np.isfinite(surface)
        )


    # ========================================================
    # Smooth
    # ========================================================

    if args.smooth_ksize > 1:

        surface = masked_gaussian(
            surface,
            valid,
            args.smooth_ksize
        )

        valid = (
            valid
            &
            np.isfinite(surface)
        )


    # ========================================================
    # Color
    # ========================================================

    color = surface_to_rgb(
        surface,
        valid
    )


    # ========================================================
    # PLY
    # ========================================================

    save_ply(
        args.out,
        surface,
        valid,
        color,
        args.z_scale,
        args.z_sign,
        args.step
    )


    # ========================================================
    # DEBUG
    # ========================================================

    debug_base = Path(
        args.out
    ).with_suffix("")

    debug_dir = (
        debug_base.parent
        /
        f"{debug_base.name}_debug"
    )

    debug_dir.mkdir(
        parents=True,
        exist_ok=True
    )


    cv2.imwrite(
        str(
            debug_dir
            /
            "01_input_mask_RAW.png"
        ),
        mask_raw.astype(
            np.uint8
        ) * 255
    )


    cv2.imwrite(
        str(
            debug_dir
            /
            "01B_input_mask_HOLE_FILLED.png"
        ),
        mask_filled.astype(
            np.uint8
        ) * 255
    )


    cv2.imwrite(
        str(
            debug_dir
            /
            "01C_새로채운_내부hole.png"
        ),
        filled_holes.astype(
            np.uint8
        ) * 255
    )


    cv2.imwrite(
        str(
            debug_dir
            /
            "01D_hole중_Phase존재.png"
        ),
        phase_available_in_holes.astype(
            np.uint8
        ) * 255
    )


    cv2.imwrite(
        str(
            debug_dir
            /
            "01E_hole중_Phase없음.png"
        ),
        phase_missing_in_holes.astype(
            np.uint8
        ) * 255
    )


    cv2.imwrite(
        str(
            debug_dir
            /
            "02_unwrap_visited_mask.png"
        ),
        visited.astype(
            np.uint8
        ) * 255
    )


    cv2.imwrite(
        str(
            debug_dir
            /
            "03_final_valid_mask.png"
        ),
        valid.astype(
            np.uint8
        ) * 255
    )


    cv2.imwrite(
        str(
            debug_dir
            /
            "04_color_preview.png"
        ),
        cv2.cvtColor(
            color,
            cv2.COLOR_RGB2BGR
        )
    )


    print()
    print("debug:", debug_dir)

    print()
    print(
        "※ 내부 hole은 마스크만 복구."
    )

    print(
        "※ 해당 위치의 Z는 실제 Phase 값 사용."
    )

    print(
        "※ Depth Z값으로 채우지 않음."
    )

    print(
        "※ 주변 값 interpolation도 사용하지 않음."
    )

    print()
    print(
        "주의: phase-only relative surface임."
        " 실제 mm 3D 아님."
    )


if __name__ == "__main__":
    main()
