#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
import numpy as np
import cv2


# ============================================================
# 설정
# ============================================================

BASE = Path(
    "/home/seoyeong/졸업작품/전처리와구조광_통합"
)

SAMPLE = (
    BASE
    / "플랫폼 바닥 따기"
    / "구조광_전처리"
    / "샘플"
)

# 바닥을 얼마나 듬성듬성 표시할지
# 클수록 바닥 점이 적음
FLOOR_STEP = 6

# 물체 최저부보다 살짝 아래에 바닥을 둠
FLOOR_GAP = 8.0

# 물체 주변 몇 pixel은 바닥을 비워둘지
OBJECT_MARGIN = 5


# ============================================================
# 최신 촬영 자동 선택
# ============================================================

runs = sorted([
    p for p in SAMPLE.iterdir()
    if p.is_dir()
    and p.name.startswith("촬영_")
])

if not runs:
    raise RuntimeError("촬영 폴더가 없습니다.")

RUN = runs[-1]

ROOT = (
    RUN
    / "전처리_결과"
    / "구조광_형상복원"
)

V2 = (
    ROOT
    / "최종자동통합"
    / "04_현재프레임플랫폼기준_최종_v2_Depth홀위상보강"
)


# ============================================================
# 가장 최근 최종 Point Cloud 자동 선택
# ============================================================

ply_candidates = list(
    V2.glob("FINAL_DC_MASK_PHASE*.ply")
)

# 이미 바닥을 추가한 파일은 제외
ply_candidates = [
    p for p in ply_candidates
    if "WITH_FLOOR" not in p.name
]

if not ply_candidates:
    raise RuntimeError(
        "FINAL_DC_MASK_PHASE*.ply 파일이 없습니다."
    )

OBJECT_PLY = max(
    ply_candidates,
    key=lambda p: p.stat().st_mtime
)


# ============================================================
# 플랫폼 마스크
# ============================================================

platform_path = (
    V2
    / "03_platform_all_mask.npy"
)

if not platform_path.exists():

    platform_path = (
        V2
        / "04_platform_fit_mask.npy"
    )

if not platform_path.exists():
    raise RuntimeError(
        "플랫폼 마스크를 찾지 못했습니다."
    )

platform_mask = np.load(
    platform_path
).astype(bool)


# ============================================================
# 최종 물체 마스크
# ============================================================

object_mask_path = (
    ROOT
    / "DC_GRABCUT_OBJECT_MASK"
    / "dc_grabcut_object_mask.npy"
)

if not object_mask_path.exists():
    raise RuntimeError(
        "DC GrabCut 물체 마스크가 없습니다."
    )

object_mask = np.load(
    object_mask_path
).astype(bool)


# shape 맞추기
if object_mask.shape != platform_mask.shape:

    object_mask = cv2.resize(
        object_mask.astype(np.uint8),
        (
            platform_mask.shape[1],
            platform_mask.shape[0]
        ),
        interpolation=cv2.INTER_NEAREST
    ).astype(bool)


# ============================================================
# 물체 주변은 바닥 제거
# ============================================================

ksize = OBJECT_MARGIN * 2 + 1

kernel = cv2.getStructuringElement(
    cv2.MORPH_ELLIPSE,
    (ksize, ksize)
)

object_dilated = cv2.dilate(
    object_mask.astype(np.uint8),
    kernel,
    iterations=1
).astype(bool)

floor_mask = (
    platform_mask
    &
    (~object_dilated)
)


# ============================================================
# 기존 PLY 읽기
# ============================================================

with OBJECT_PLY.open(
    "r",
    encoding="ascii"
) as f:

    lines = f.readlines()


end_header_idx = None
vertex_count = None

for i, line in enumerate(lines):

    if line.startswith("element vertex"):
        vertex_count = int(
            line.strip().split()[-1]
        )

    if line.strip() == "end_header":
        end_header_idx = i
        break


if end_header_idx is None:
    raise RuntimeError(
        "PLY header를 읽지 못했습니다."
    )

if vertex_count is None:
    raise RuntimeError(
        "PLY vertex 개수를 읽지 못했습니다."
    )


vertex_lines = lines[
    end_header_idx + 1:
    end_header_idx + 1 + vertex_count
]


# ============================================================
# 기존 물체의 Z 범위 확인
# ============================================================

z_values = []

for line in vertex_lines:

    parts = line.strip().split()

    if len(parts) < 3:
        continue

    z_values.append(
        float(parts[2])
    )


if not z_values:
    raise RuntimeError(
        "PLY에 물체 점이 없습니다."
    )

z_values = np.asarray(
    z_values,
    dtype=np.float64
)


# outlier 한두 점 때문에 바닥이 너무 내려가지 않도록
# 최소값 대신 하위 5% 사용
object_bottom = float(
    np.percentile(
        z_values,
        5
    )
)

floor_z = (
    object_bottom
    -
    FLOOR_GAP
)


# ============================================================
# 바닥 Point 생성
# ============================================================

h, w = platform_mask.shape

ys, xs = np.where(
    floor_mask
)

keep = (
    (ys % FLOOR_STEP == 0)
    &
    (xs % FLOOR_STEP == 0)
)

ys = ys[keep]
xs = xs[keep]


floor_lines = []

for y, x in zip(ys, xs):

    X = float(
        x - w / 2.0
    )

    Y = float(
        h / 2.0 - y
    )

    Z = floor_z

    # 중립적인 회색 바닥
    r = 150
    g = 150
    b = 150

    floor_lines.append(
        f"{X:.6f} "
        f"{Y:.6f} "
        f"{Z:.6f} "
        f"{r} {g} {b}\n"
    )


# ============================================================
# 새 PLY 저장
# ============================================================

OUT = (
    V2
    / f"{OBJECT_PLY.stem}_WITH_FLOOR.ply"
)

total_vertices = (
    vertex_count
    +
    len(floor_lines)
)


new_header = []

for line in lines[:end_header_idx + 1]:

    if line.startswith(
        "element vertex"
    ):

        new_header.append(
            f"element vertex {total_vertices}\n"
        )

    else:

        new_header.append(
            line
        )


with OUT.open(
    "w",
    encoding="ascii"
) as f:

    f.writelines(
        new_header
    )

    f.writelines(
        vertex_lines
    )

    f.writelines(
        floor_lines
    )


# ============================================================
# Debug
# ============================================================

debug_path = (
    V2
    / "DEBUG_floor_mask.png"
)

cv2.imwrite(
    str(debug_path),
    floor_mask.astype(
        np.uint8
    ) * 255
)


print()
print("==============================================")
print("✅ 바닥면 추가 완료")
print("==============================================")
print()
print("최신 촬영:")
print(RUN.name)
print()
print("사용한 물체 PLY:")
print(OBJECT_PLY)
print()
print("플랫폼 마스크:")
print(platform_path)
print()
print("물체 point:", vertex_count)
print("바닥 point:", len(floor_lines))
print()
print("물체 하위 5% Z:")
print(f"{object_bottom:.3f}")
print()
print("바닥 Z:")
print(f"{floor_z:.3f}")
print()
print("최종:")
print(OUT)
print("==============================================")
