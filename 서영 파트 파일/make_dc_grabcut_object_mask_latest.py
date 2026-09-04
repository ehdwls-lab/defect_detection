#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path

import cv2
import numpy as np

from structured_light_paths import RESULT_ROOT, ROOT


# ============================================================
# 기본 경로
# ============================================================

BASE = ROOT

SAMPLE = RESULT_ROOT


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
# 4-step 최종 세트 찾기
# ============================================================

phase_dirs = []

for p0 in RUN.rglob("phase_000.png"):

    d = p0.parent

    req = [
        d / "phase_000.png",
        d / "phase_090.png",
        d / "phase_180.png",
        d / "phase_270.png",
    ]

    if all(p.exists() for p in req):
        phase_dirs.append(d)


if not phase_dirs:
    raise RuntimeError(
        "phase_000/090/180/270 세트를 찾지 못했습니다."
    )


PHASE_DIR = max(
    phase_dirs,
    key=lambda d:
        (d / "phase_000.png").stat().st_mtime
)


# ============================================================
# 4-step LOAD
# ============================================================

def load_gray(name):

    path = PHASE_DIR / name

    img = cv2.imread(
        str(path),
        cv2.IMREAD_GRAYSCALE
    )

    if img is None:
        raise RuntimeError(
            f"읽기 실패: {path}"
        )

    return img.astype(np.float32)


I0 = load_gray("phase_000.png")
I90 = load_gray("phase_090.png")
I180 = load_gray("phase_180.png")
I270 = load_gray("phase_270.png")


# ============================================================
# DC 영상
#
# 사인파 4위상 평균
# ============================================================

dc = (
    I0
    + I90
    + I180
    + I270
) / 4.0

dc_u8 = np.clip(
    dc,
    0,
    255
).astype(np.uint8)


# 대비 개선
clahe = cv2.createCLAHE(
    clipLimit=2.0,
    tileGridSize=(8, 8)
)

dc_clahe = clahe.apply(
    dc_u8
)

dc_bgr = cv2.cvtColor(
    dc_clahe,
    cv2.COLOR_GRAY2BGR
)


# ============================================================
# Depth seed
# ============================================================

seed_path = (
    V2
    / "02_depth_object_seed_mask.npy"
)

if not seed_path.exists():

    seed_path = (
        V2
        / "02_depth_object_mask.npy"
    )


if not seed_path.exists():
    raise RuntimeError(
        "Depth 물체 seed mask를 찾지 못했습니다."
    )


seed = np.load(
    seed_path
).astype(bool)


if seed.shape != dc_u8.shape:

    seed = cv2.resize(
        seed.astype(np.uint8),
        (
            dc_u8.shape[1],
            dc_u8.shape[0]
        ),
        interpolation=cv2.INTER_NEAREST
    ).astype(bool)


# ============================================================
# 구조광 ROI
# ============================================================

roi_path = (
    ROOT
    / "object_area_mask.npy"
)


if roi_path.exists():

    roi = np.load(
        roi_path
    ).astype(bool)

    if roi.shape != dc_u8.shape:

        roi = cv2.resize(
            roi.astype(np.uint8),
            (
                dc_u8.shape[1],
                dc_u8.shape[0]
            ),
            interpolation=cv2.INTER_NEAREST
        ).astype(bool)

else:

    roi = np.ones(
        dc_u8.shape,
        dtype=bool
    )


# ============================================================
# GrabCut 초기화
# ============================================================

# 기본:
# ROI 밖       = 확실한 배경
# ROI 안       = 배경 후보
# Depth 주변   = 물체 후보
# Depth 내부   = 확실한 물체

gc_mask = np.full(
    dc_u8.shape,
    cv2.GC_BGD,
    dtype=np.uint8
)

gc_mask[roi] = cv2.GC_PR_BGD


# ------------------------------------------------------------
# Depth seed 주변까지 "물체일 가능성 있음"으로 확장
# ------------------------------------------------------------

seed_u8 = (
    seed.astype(np.uint8)
    * 255
)

kernel_expand = cv2.getStructuringElement(
    cv2.MORPH_ELLIPSE,
    (61, 61)
)

probable_fg = cv2.dilate(
    seed_u8,
    kernel_expand,
    iterations=1
) > 0

probable_fg &= roi

gc_mask[probable_fg] = cv2.GC_PR_FGD


# ------------------------------------------------------------
# 확실한 물체 영역
#
# Depth seed에서 경계 약간 제외
# ------------------------------------------------------------

kernel_erode = cv2.getStructuringElement(
    cv2.MORPH_ELLIPSE,
    (5, 5)
)

sure_fg = cv2.erode(
    seed_u8,
    kernel_erode,
    iterations=1
) > 0

sure_fg &= roi

gc_mask[sure_fg] = cv2.GC_FGD


# ============================================================
# GrabCut
# ============================================================

bg_model = np.zeros(
    (1, 65),
    np.float64
)

fg_model = np.zeros(
    (1, 65),
    np.float64
)


cv2.grabCut(
    dc_bgr,
    gc_mask,
    None,
    bg_model,
    fg_model,
    8,
    cv2.GC_INIT_WITH_MASK
)


grab = (
    (gc_mask == cv2.GC_FGD)
    |
    (gc_mask == cv2.GC_PR_FGD)
)

grab &= roi


# ============================================================
# Depth seed와 실제로 연결되는 component만 유지
# ============================================================

num, labels, stats, _ = (
    cv2.connectedComponentsWithStats(
        grab.astype(np.uint8),
        connectivity=8
    )
)

best_label = None
best_overlap = 0

for label in range(1, num):

    comp = labels == label

    overlap = np.count_nonzero(
        comp & seed
    )

    if overlap > best_overlap:

        best_overlap = overlap
        best_label = label


if best_label is None:
    raise RuntimeError(
        "Depth seed와 연결되는 DC 물체 영역을 찾지 못했습니다."
    )


final_mask = (
    labels == best_label
)


# ============================================================
# 내부 hole만 채우기
# 외곽은 건드리지 않음
# ============================================================

background = (
    ~final_mask
).astype(np.uint8)

n_bg, labels_bg = cv2.connectedComponents(
    background,
    connectivity=8
)

border_labels = np.unique(
    np.concatenate([
        labels_bg[0, :],
        labels_bg[-1, :],
        labels_bg[:, 0],
        labels_bg[:, -1],
    ])
)

outside = (
    np.isin(
        labels_bg,
        border_labels
    )
    &
    (background > 0)
)

holes = (
    (background > 0)
    &
    (~outside)
)

final_mask |= holes


# ============================================================
# 저장
# ============================================================

OUT = (
    ROOT
    / "DC_GRABCUT_OBJECT_MASK"
)

OUT.mkdir(
    parents=True,
    exist_ok=True
)


np.save(
    OUT
    / "dc_grabcut_object_mask.npy",
    final_mask
)


cv2.imwrite(
    str(
        OUT
        / "01_DC.png"
    ),
    dc_u8
)

cv2.imwrite(
    str(
        OUT
        / "02_DC_CLAHE.png"
    ),
    dc_clahe
)

cv2.imwrite(
    str(
        OUT
        / "03_Depth_seed.png"
    ),
    seed.astype(np.uint8) * 255
)

cv2.imwrite(
    str(
        OUT
        / "04_DC_GrabCut_RAW.png"
    ),
    grab.astype(np.uint8) * 255
)

cv2.imwrite(
    str(
        OUT
        / "05_FINAL_DC_OBJECT_MASK.png"
    ),
    final_mask.astype(np.uint8) * 255
)


# ============================================================
# DC 위에 외곽 표시
# ============================================================

overlay = dc_bgr.copy()

contours, _ = cv2.findContours(
    final_mask.astype(np.uint8),
    cv2.RETR_EXTERNAL,
    cv2.CHAIN_APPROX_SIMPLE
)

cv2.drawContours(
    overlay,
    contours,
    -1,
    (0, 0, 255),
    2
)

cv2.imwrite(
    str(
        OUT
        / "06_FINAL_MASK_OVERLAY.png"
    ),
    overlay
)


print()
print("==============================================")
print("✅ DC 기반 물체 외곽 추출 완료")
print("==============================================")
print()
print("촬영:")
print(RUN.name)
print()
print("Depth seed:")
print(seed_path)
print()
print("최종 mask:")
print(
    OUT
    / "dc_grabcut_object_mask.npy"
)
print()
print("확인:")
print("05_FINAL_DC_OBJECT_MASK.png")
print("06_FINAL_MASK_OVERLAY.png")
print("==============================================")
