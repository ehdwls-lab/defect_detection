import os
import sys
import shutil
import subprocess

import numpy as np
import open3d as o3d


# ==============================================================================
# 0. 파라미터 설정
# ==============================================================================

# ------------------------------------------------------------------------------
# 3축 플랫폼 물리 제약
# ------------------------------------------------------------------------------

PLATFORM_MAX_TILT_DEG = 22.5

MIN_OPTICAL_DIST_CM = 20.0

PLATFORM_MAX_Z_LIFT_CM = 20.0

# 실제 카메라-바닥면 기준 거리
CAMERA_STANDOFF_MM = 455.0


# ------------------------------------------------------------------------------
# RANSAC 평면 검출
# ------------------------------------------------------------------------------

MAX_PLANES = 8

MIN_POINTS_PER_PLANE = 300

RANSAC_DISTANCE_THRESHOLD = 2.5

RANSAC_N = 3

RANSAC_ITERATIONS = 1000


# ------------------------------------------------------------------------------
# ★ 1차 평면 병합 조건
#
# 같은 도장면인데 노이즈 때문에 RANSAC이
# 여러 개의 평면으로 쪼개버린 경우 병합
# ------------------------------------------------------------------------------

MERGE_ANGLE_THRESH_DEG = 10.0

MERGE_PLANE_DIST_MM = 6.0


# ------------------------------------------------------------------------------
# ★ 대표 평면(Dominant Plane) 2차 통합 조건
#
# 가장 많은 점을 가진 평면을 대표 도장면으로 선정.
#
# 대표면과 조금 차이나더라도
# 실제로 같은 도장면이라고 볼 수 있으면 RED로 통합.
# ------------------------------------------------------------------------------

DOMINANT_MERGE_ANGLE_DEG = 15.0

DOMINANT_MERGE_DIST_MM = 10.0


# ------------------------------------------------------------------------------
# ★ RANSAC에서 빠진 잔여점도 대표 평면에 흡수할지 여부
# ------------------------------------------------------------------------------

ABSORB_REMAINING_POINTS = True


# 대표 평면에서 이 거리 이내면 흡수 후보
REMAINING_POINT_DIST_MM = 4.0


# 잔여점의 Normal도 대표 평면과 이 각도 이내여야 함
REMAINING_NORMAL_ANGLE_DEG = 18.0


# ------------------------------------------------------------------------------
# Open3D 전처리
# ------------------------------------------------------------------------------

VOXEL_SIZE_MM = 1.0

NORMAL_RADIUS_MM = 10.0

NORMAL_MAX_NN = 30


# ==============================================================================
# 1. 입력 PLY 파일 결정
# ==============================================================================

# 사용법 1:
#
# python3 plane_test_filtered.py
#
# → 아래 DEFAULT_PLY_FILE 사용
#
#
# 사용법 2:
#
# python3 plane_test_filtered.py 6_0822_filtered_1st.ply
#
# → 입력한 PLY 사용


DEFAULT_PLY_FILE = "3_0822_filtered_1st.ply"


if len(sys.argv) >= 2:

    ply_file_path = sys.argv[1]

else:

    ply_file_path = DEFAULT_PLY_FILE


print("\n" + "=" * 70)

print("입력 PLY 파일")

print(f">> {ply_file_path}")

print("=" * 70)


# ==============================================================================
# 2. PLY 파일 로드
# ==============================================================================

if not os.path.exists(ply_file_path):

    print(
        f"\nError: 파일을 찾을 수 없습니다:\n"
        f"{os.path.abspath(ply_file_path)}"
    )

    print(
        "\n사용 예시:"
    )

    print(
        "python3 plane_test_filtered.py "
        "6_0822_filtered_1st.ply"
    )

    sys.exit(1)


pcd = o3d.io.read_point_cloud(
    ply_file_path
)


if not pcd.has_points():

    print(
        "Error: 포인트 클라우드 데이터가 비어 있습니다."
    )

    sys.exit(1)


print("\n" + "=" * 70)

print("PLY 로드 완료")

print(
    f"원본 Point 수 : "
    f"{len(pcd.points)}"
)

print("=" * 70)


# ==============================================================================
# 3. Down Sampling + Normal 계산
# ==============================================================================

pcd = pcd.voxel_down_sample(
    voxel_size=VOXEL_SIZE_MM
)


pcd.estimate_normals(

    search_param=o3d.geometry.KDTreeSearchParamHybrid(

        radius=NORMAL_RADIUS_MM,

        max_nn=NORMAL_MAX_NN

    )

)


print(
    f"다운샘플링 후 Point 수 : "
    f"{len(pcd.points)}"
)


# ==============================================================================
# 4. 보조 함수
# ==============================================================================

def normalize_plane_model(model):

    """
    Plane Model:

        ax + by + cz + d = 0

    법선을 단위벡터로 정규화하고
    가능한 경우 +Z 방향으로 통일
    """

    a, b, c, d = model


    normal = np.array(
        [a, b, c],
        dtype=np.float64
    )


    norm = np.linalg.norm(
        normal
    )


    if norm < 1e-12:

        return (
            np.array(
                [0.0, 0.0, 1.0],
                dtype=np.float64
            ),
            0.0
        )


    normal /= norm

    d /= norm


    # 법선 방향을 +Z 방향으로 통일
    if normal[2] < 0:

        normal = -normal

        d = -d


    return normal, d



def fit_plane_pca(point_cloud):

    """
    병합된 Point Cloud 전체를 이용해서
    PCA 방식으로 대표 평면을 다시 계산

    병합 후 기존 RANSAC 평면식을 그대로 쓰는 것보다
    전체 점군을 반영할 수 있음.
    """

    points = np.asarray(
        point_cloud.points
    )


    if len(points) < 3:

        return np.array(
            [
                0.0,
                0.0,
                1.0,
                0.0
            ],
            dtype=np.float64
        )


    center = np.mean(
        points,
        axis=0
    )


    centered = (
        points
        -
        center
    )


    covariance = np.cov(
        centered,
        rowvar=False
    )


    eigenvalues, eigenvectors = np.linalg.eigh(
        covariance
    )


    # 가장 작은 eigenvalue 방향
    # = 평면의 Normal 방향
    normal = eigenvectors[
        :,
        np.argmin(eigenvalues)
    ]


    normal /= np.linalg.norm(
        normal
    )


    if normal[2] < 0:

        normal = -normal


    d = -np.dot(
        normal,
        center
    )


    return np.array(

        [
            normal[0],
            normal[1],
            normal[2],
            d
        ],

        dtype=np.float64

    )



def plane_angle_deg(
    model1,
    model2
):

    """
    두 평면 Normal 사이 각도 계산
    """

    n1, _ = normalize_plane_model(
        model1
    )

    n2, _ = normalize_plane_model(
        model2
    )


    cosine = np.clip(

        np.dot(
            n1,
            n2
        ),

        -1.0,

        1.0

    )


    angle = np.degrees(
        np.arccos(
            cosine
        )
    )


    return float(angle)



def plane_distance_mm(
    reference_model,
    target_pcd
):

    """
    target Point Cloud 중심점이
    reference 평면에서 얼마나 떨어져 있는지 계산
    """

    points = np.asarray(
        target_pcd.points
    )


    if len(points) == 0:

        return np.inf


    center = np.mean(
        points,
        axis=0
    )


    normal, d = normalize_plane_model(
        reference_model
    )


    distance = abs(

        np.dot(
            normal,
            center
        )

        +

        d

    )


    return float(distance)



def merge_point_cloud(
    group,
    new_pcd
):

    """
    기존 Plane Group에
    새로운 Point Cloud 병합
    """

    group["pcd"] += new_pcd


    # 병합된 전체 점 기준으로
    # 대표 Plane 다시 계산
    group["model"] = fit_plane_pca(
        group["pcd"]
    )


    group["points_count"] = len(
        group["pcd"].points
    )


# ==============================================================================
# 5. RANSAC 다중 평면 검출 + 1차 자동 병합
# ==============================================================================

plane_groups = []


current_pcd = pcd


raw_plane_count = 0


print("\n" + "=" * 70)

print("RANSAC 다중 평면 검출 시작")

print("=" * 70)


for i in range(MAX_PLANES):


    if len(current_pcd.points) < MIN_POINTS_PER_PLANE:

        break


    plane_model, inliers = current_pcd.segment_plane(

        distance_threshold=RANSAC_DISTANCE_THRESHOLD,

        ransac_n=RANSAC_N,

        num_iterations=RANSAC_ITERATIONS

    )


    if len(inliers) < MIN_POINTS_PER_PLANE:

        break


    raw_plane_count += 1


    candidate_pcd = current_pcd.select_by_index(
        inliers
    )


    # ----------------------------------------------------------
    # 검출된 점은 현재 Point Cloud에서 제거
    #
    # 단,
    #
    # 기존 코드처럼 중복이라고 버리는 것이 아니라
    #
    # 1. 기존 Plane Group에 병합
    # 또는
    # 2. 새 Plane Group 생성
    #
    # 둘 중 하나로 반드시 보존
    # ----------------------------------------------------------

    current_pcd = current_pcd.select_by_index(

        inliers,

        invert=True

    )


    candidate_model = fit_plane_pca(
        candidate_pcd
    )


    merged = False


    # ------------------------------------------------------------------
    # 기존 Plane Group들과 비교
    # ------------------------------------------------------------------

    for group_index, group in enumerate(
        plane_groups
    ):


        angle = plane_angle_deg(

            candidate_model,

            group["model"]

        )


        distance = plane_distance_mm(

            group["model"],

            candidate_pcd

        )


        # --------------------------------------------------------------
        # Normal 각도도 비슷하고
        # 실제 평면 위치도 가까우면
        #
        # 같은 평면으로 판단
        # --------------------------------------------------------------

        if (

            angle <= MERGE_ANGLE_THRESH_DEG

            and

            distance <= MERGE_PLANE_DIST_MM

        ):


            print(

                f"[Plane {raw_plane_count}] "

                f"Group {group_index + 1}에 병합 "

                f"(angle={angle:.2f}°, "

                f"distance={distance:.2f}mm)"

            )


            merge_point_cloud(

                group,

                candidate_pcd

            )


            merged = True


            break


    # ------------------------------------------------------------------
    # 기존 Group과 다르면 새 Plane 생성
    # ------------------------------------------------------------------

    if not merged:


        plane_groups.append(

            {

                "pcd":
                    candidate_pcd,

                "model":
                    candidate_model,

                "points_count":
                    len(candidate_pcd.points)

            }

        )


        print(

            f"[Plane {raw_plane_count}] "

            f"새 평면 Group 생성 "

            f"({len(candidate_pcd.points)} points)"

        )


# ==============================================================================
# 6. 가장 큰 평면 = 대표 도장면 선정
# ==============================================================================

if len(plane_groups) == 0:


    print(
        "\n검출된 유효 평면이 없습니다."
    )


    sys.exit(1)



plane_groups.sort(

    key=lambda x: x["points_count"],

    reverse=True

)


dominant_group = plane_groups[0]


print("\n" + "=" * 70)

print("1차 평면 Group 결과")

print("=" * 70)


for i, group in enumerate(
    plane_groups
):


    print(

        f"Group {i + 1} : "

        f"{group['points_count']} points"

    )


print("\n★ 대표 평면 선정")


print(

    f"Dominant Plane : "

    f"{dominant_group['points_count']} points"

)


# ==============================================================================
# 7. 대표 평면 기준 2차 병합
# ==============================================================================

remaining_groups = []


for i in range(
    1,
    len(plane_groups)
):


    group = plane_groups[i]


    angle = plane_angle_deg(

        dominant_group["model"],

        group["model"]

    )


    distance = plane_distance_mm(

        dominant_group["model"],

        group["pcd"]

    )


    print(

        f"\nDominant Plane vs Group {i + 1}"

    )


    print(

        f"  angle    = "

        f"{angle:.2f}°"

    )


    print(

        f"  distance = "

        f"{distance:.2f} mm"

    )


    # ------------------------------------------------------------------
    # 대표 평면과 충분히 비슷하면
    # 동일 도장면이라고 판단
    # ------------------------------------------------------------------

    if (

        angle <= DOMINANT_MERGE_ANGLE_DEG

        and

        distance <= DOMINANT_MERGE_DIST_MM

    ):


        print(

            "  >> 같은 도장면으로 판단"

            " → 대표 RED 영역으로 통합"

        )


        merge_point_cloud(

            dominant_group,

            group["pcd"]

        )


    else:


        print(

            "  >> 실제 다른 방향의 평면으로 유지"

        )


        remaining_groups.append(
            group
        )


# 대표평면 + 실제 다른 평면만 유지
plane_groups = [

    dominant_group

] + remaining_groups


# ==============================================================================
# 8. RANSAC에서 빠진 잔여점 중
#    대표 평면과 가까운 점 RED로 흡수
# ==============================================================================

if (

    ABSORB_REMAINING_POINTS

    and

    len(current_pcd.points) > 0

):


    remaining_points = np.asarray(
        current_pcd.points
    )


    dominant_normal, dominant_d = normalize_plane_model(
        dominant_group["model"]
    )


    # ------------------------------------------------------------------
    # Point → 대표 평면 거리
    # ------------------------------------------------------------------

    distances = np.abs(

        remaining_points

        @

        dominant_normal

        +

        dominant_d

    )


    distance_mask = (

        distances

        <=

        REMAINING_POINT_DIST_MM

    )


    # ------------------------------------------------------------------
    # Normal도 대표 평면과 비슷한지 확인
    # ------------------------------------------------------------------

    if current_pcd.has_normals():


        normals = np.asarray(
            current_pcd.normals
        )


        normal_length = np.linalg.norm(

            normals,

            axis=1

        )


        valid_normal = (

            normal_length

            >

            1e-6

        )


        normalized_normals = np.zeros_like(
            normals
        )


        normalized_normals[
            valid_normal
        ] = (

            normals[
                valid_normal
            ]

            /

            normal_length[
                valid_normal,
                None
            ]

        )


        # Normal 방향 +/-는 같은 평면이므로 abs 사용
        cosine = np.abs(

            normalized_normals

            @

            dominant_normal

        )


        cosine = np.clip(

            cosine,

            -1.0,

            1.0

        )


        angles = np.degrees(

            np.arccos(
                cosine
            )

        )


        normal_mask = (

            angles

            <=

            REMAINING_NORMAL_ANGLE_DEG

        )


        absorb_mask = (

            distance_mask

            &

            normal_mask

        )


    else:


        absorb_mask = distance_mask


    absorb_indices = np.where(
        absorb_mask
    )[0]


    remain_indices = np.where(
        ~absorb_mask
    )[0]


    if len(absorb_indices) > 0:


        absorb_pcd = current_pcd.select_by_index(

            absorb_indices.tolist()

        )


        print("\n" + "=" * 70)


        print(

            f"대표 RED 평면 주변 잔여점 "

            f"{len(absorb_indices)}개 추가 흡수"

        )


        merge_point_cloud(

            dominant_group,

            absorb_pcd

        )


    current_pcd = current_pcd.select_by_index(

        remain_indices.tolist()

    )


# ==============================================================================
# 9. 최종 Plane Group 재정렬
# ==============================================================================

plane_groups.sort(

    key=lambda x: x["points_count"],

    reverse=True

)


for idx, group in enumerate(
    plane_groups
):


    group["name"] = (

        f"Shot {idx + 1}"

    )


# ==============================================================================
# 10. 각 영역 목표 Pose 계산
# ==============================================================================

def calculate_shot_pose(
    plane_info
):


    points = np.asarray(
        plane_info["pcd"].points
    )


    # 병합 후 최종 Point Cloud로
    # Plane Model 다시 계산
    model = fit_plane_pca(
        plane_info["pcd"]
    )


    plane_info["model"] = model


    normal, d = normalize_plane_model(
        model
    )


    nx, ny, nz = normal


    # ------------------------------------------------------------------
    # 1) 평면 자체 기울기 (위쪽/오른쪽 상승 기준 부호 반영)[cite: 27]
    # ------------------------------------------------------------------

    pitch_deg = -np.degrees(

        np.arctan2(

            ny,

            nz

        )

    )


    roll_deg = -np.degrees(

        np.arctan2(

            nx,

            np.sqrt(

                ny ** 2

                +

                nz ** 2

            )

        )

    )


    center = np.mean(

        points,

        axis=0

    )


    # ------------------------------------------------------------------
    # 2) Working Distance 계산
    # ------------------------------------------------------------------

    if center[2] < 200.0:


        current_wd_mm = (

            CAMERA_STANDOFF_MM

            +

            center[2]

        )


    else:


        current_wd_mm = center[2]


    current_wd_cm = (

        current_wd_mm

        /

        10.0

    )


    # ------------------------------------------------------------------
    # 3) 카메라 시선 Parallax 보정
    # ------------------------------------------------------------------

    view_pitch_offset = np.degrees(

        np.arctan2(

            center[1],

            current_wd_mm

        )

    )


    view_roll_offset = np.degrees(

        np.arctan2(

            center[0],

            current_wd_mm

        )

    )


    raw_target_roll = (

        roll_deg

        -

        view_roll_offset

    )


    raw_target_pitch = (

        pitch_deg

        +

        view_pitch_offset

    )


    # ------------------------------------------------------------------
    # 4) 플랫폼 물리 각도 제한
    # ------------------------------------------------------------------

    target_roll = np.clip(

        raw_target_roll,

        -PLATFORM_MAX_TILT_DEG,

        PLATFORM_MAX_TILT_DEG

    )


    target_pitch = np.clip(

        raw_target_pitch,

        -PLATFORM_MAX_TILT_DEG,

        PLATFORM_MAX_TILT_DEG

    )


    # ------------------------------------------------------------------
    # 5) Z 상승량 계산
    # ------------------------------------------------------------------

    raw_z_lift_cm = (

        current_wd_cm

        -

        MIN_OPTICAL_DIST_CM

    )


    target_z_lift_cm = np.clip(

        raw_z_lift_cm,

        0.0,

        PLATFORM_MAX_Z_LIFT_CM

    )


    # ------------------------------------------------------------------
    # 검사 영역 물리 크기
    # ------------------------------------------------------------------

    p_min = np.min(

        points,

        axis=0

    )


    p_max = np.max(

        points,

        axis=0

    )


    size_w = (

        p_max[0]

        -

        p_min[0]

    )


    size_h = (

        p_max[1]

        -

        p_min[1]

    )


    return {


        "roll":
            target_roll,


        "pitch":
            target_pitch,


        "raw_roll":
            raw_target_roll,


        "raw_pitch":
            raw_target_pitch,


        "z_lift":
            target_z_lift_cm,


        "current_wd":
            current_wd_cm,


        "final_wd":

            current_wd_cm

            -

            target_z_lift_cm,


        "size_w":
            size_w,


        "size_h":
            size_h,


        "center":
            center

    }


# ==============================================================================
# 11. 색상 설정
#
# 가장 큰 대표 평면은 무조건 RED
# ==============================================================================

colors = [

    [1.0, 0.0, 0.0],      # RED

    [0.0, 1.0, 0.0],      # GREEN

    [0.0, 0.0, 1.0],      # BLUE

    [1.0, 0.7, 0.0],      # ORANGE

    [0.8, 0.0, 1.0],      # PURPLE

]


combined_pcd = o3d.geometry.PointCloud()


# ==============================================================================
# 12. 결과 출력 + 색칠
# ==============================================================================

print("\n")

print("=" * 70)


print(

    " 3축 플랫폼 대표 평면 통합 검사 결과"

)


print(

    f" 최종 검출 영역 : "

    f"{len(plane_groups)}개"

)


print("=" * 70)


total_plane_points = sum(

    g["points_count"]

    for g in plane_groups

)


for idx, p in enumerate(
    plane_groups
):


    res = calculate_shot_pose(
        p
    )


    plane_color = colors[

        idx

        %

        len(colors)

    ]


    # ------------------------------------------------------------------
    # 대표 평면 = RED
    #
    # 병합된 같은 평면도
    # 전부 동일한 RED
    # ------------------------------------------------------------------

    p["pcd"].paint_uniform_color(
        plane_color
    )


    combined_pcd += p["pcd"]


    ratio = (

        p["points_count"]

        /

        total_plane_points

        *

        100.0

    )


    print(

        f"\n[{p['name']}]"

    )


    print(

        f" • 유효 점군 : "

        f"{p['points_count']}개 "

        f"({ratio:.1f}%)"

    )


    if idx == 0:


        print(

            " • ★ 대표 도장면 "

            "(Dominant Plane / RED)"

        )


    print(

        " • 실제 측정 형상 각도 : "

        f"Roll = {res['raw_roll']:.2f}°, "

        f"Pitch = {res['raw_pitch']:.2f}°"

    )


    print(

        " • 3축 제어 명령 : "

        f"Roll = {res['roll']:+.2f}°, "

        f"Pitch = {res['pitch']:+.2f}°"

    )


    print(

        " • Z 상승 이동량 : "

        f"{res['z_lift']:.2f} cm"

    )


    print(

        " • 작업 거리 : "

        f"{res['current_wd']:.1f} cm "

        f"→ "

        f"{res['final_wd']:.1f} cm"

    )


    print(

        " • 검사 영역 크기 : "

        f"{res['size_w']:.1f} "

        f"x "

        f"{res['size_h']:.1f} mm"

    )


    print(

        " • 영역 중심 : "

        f"X={res['center'][0]:.1f}, "

        f"Y={res['center'][1]:.1f}, "

        f"Z={res['center'][2]:.1f} mm"

    )


    print(

        " >> STM32 Packet : "

        f"Z:{res['z_lift']:.2f} "

        f"R:{res['roll']:.2f} "

        f"P:{res['pitch']:.2f}"

    )


# ==============================================================================
# 13. 남은 비평면 영역 = 회색
# ==============================================================================

if len(current_pcd.points) > 0:


    current_pcd.paint_uniform_color(

        [0.45, 0.45, 0.45]

    )


    combined_pcd += current_pcd


    print(

        f"\n회색 비평면 Point : "

        f"{len(current_pcd.points)}개"

    )


# ==============================================================================
# 14. 결과 PLY 저장
# ==============================================================================

base_name = os.path.splitext(

    os.path.basename(
        ply_file_path
    )

)[0]


input_dir = os.path.dirname(

    os.path.abspath(
        ply_file_path
    )

)


output_vis_path = os.path.join(

    input_dir,

    f"{base_name}_dominant_plane_segmented.ply"

)


save_success = o3d.io.write_point_cloud(

    output_vis_path,

    combined_pcd

)


print("\n" + "=" * 70)


if save_success:


    print(

        f">> 검출 평면 색상 분할 결과 저장 완료:\n"

        f"{output_vis_path}"

    )


else:


    print(

        "Error: 결과 PLY 저장에 실패했습니다."

    )


print()

print(

    "색상 의미"

)

print(

    " RED    = 가장 많이 차지하는 대표 도장면"

)

print(

    " GREEN  = 대표면과 실제로 다른 두 번째 평면"

)

print(

    " BLUE   = 실제로 다른 세 번째 평면"

)

print(

    " ORANGE = 실제로 다른 네 번째 평면"

)

print(

    " PURPLE = 실제로 다른 다섯 번째 평면"

)

print(

    " GRAY   = 평면으로 판단되지 않은 영역"

)


print("=" * 70)


# ==============================================================================
# 15. ★ 저장된 결과를 CloudCompare에서 자동으로 열기
# ==============================================================================

if save_success:


    print("\n>> CloudCompare 실행 준비...")


    # ------------------------------------------------------------------
    # Ubuntu 환경에서 실행 가능한 CloudCompare 명령 자동 검색
    # ------------------------------------------------------------------

    cloudcompare_candidates = [

        "CloudCompare",

        "cloudcompare"

    ]


    cloudcompare_path = None


    for candidate in cloudcompare_candidates:


        found = shutil.which(
            candidate
        )


        if found is not None:


            cloudcompare_path = found


            break


    # ------------------------------------------------------------------
    # CloudCompare 발견
    # ------------------------------------------------------------------

    if cloudcompare_path is not None:


        print(

            f">> CloudCompare 발견: "

            f"{cloudcompare_path}"

        )


        try:


            subprocess.Popen(

                [

                    cloudcompare_path,

                    os.path.abspath(
                        output_vis_path
                    )

                ]

            )


            print(

                "\n★ CloudCompare 자동 실행 완료"

            )

            print(

                f">> 열린 파일:\n"

                f"{output_vis_path}"

            )


        except Exception as e:


            print(

                "\nCloudCompare 실행 중 오류 발생:"

            )


            print(
                e
            )


    # ------------------------------------------------------------------
    # CloudCompare 명령을 못 찾은 경우
    # ------------------------------------------------------------------

    else:


        print(

            "\n⚠ CloudCompare 실행 파일을 "

            "자동으로 찾지 못했습니다."

        )

        print(

            "\n터미널에서 아래 명령으로 확인해보세요."

        )

        print(

            "which CloudCompare"

        )

        print(

            "which cloudcompare"

        )

        print(

            "\n결과 PLY 자체는 정상적으로 저장되어 있습니다."

        )

        print(

            f"\n{output_vis_path}"

        )