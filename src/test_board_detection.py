import cv2
import numpy as np

from pyorbbecsdk import (
    Pipeline,
    Config,
    OBSensorType,
    OBFormat,
)


WIDTH = 1280
HEIGHT = 800
FPS = 10

# 우선 시작값.
# 판이 안 잡히면 이후 이것만 조절하면 됨.
DARK_THRESHOLD = 100

# 기준판으로 인정할 최소 면적
MIN_BOARD_AREA_RATIO = 0.20

# Perspective 변환 후 기준판 크기
WARP_SIZE = 800


def color_frame_to_bgr(color_frame):
    """Orbbec MJPG frame -> OpenCV BGR"""

    data = np.frombuffer(
        color_frame.get_data(),
        dtype=np.uint8,
    )

    image = cv2.imdecode(
        data,
        cv2.IMREAD_COLOR,
    )

    return image


def order_points(pts):
    """
    4개 점을
    좌상 / 우상 / 우하 / 좌하
    순서로 정렬
    """
    pts = np.array(
        pts,
        dtype=np.float32,
    )

    ordered = np.zeros(
        (4, 2),
        dtype=np.float32,
    )

    s = pts.sum(axis=1)
    diff = np.diff(
        pts,
        axis=1,
    ).reshape(-1)

    ordered[0] = pts[np.argmin(s)]       # 좌상
    ordered[2] = pts[np.argmax(s)]       # 우하

    ordered[1] = pts[np.argmin(diff)]    # 우상
    ordered[3] = pts[np.argmax(diff)]    # 좌하

    return ordered


def find_board(image):
    """
    검은 기준판 후보를 찾는다.

    반환:
        board_contour
        board_points
        binary_debug
    """

    h, w = image.shape[:2]

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY,
    )

    # 작은 노이즈 제거
    blur = cv2.GaussianBlur(
        gray,
        (9, 9),
        0,
    )

    # 어두운 영역 = 흰색
    _, dark_mask = cv2.threshold(
        blur,
        DARK_THRESHOLD,
        255,
        cv2.THRESH_BINARY_INV,
    )

    # 가장자리의 작은 노이즈를 줄이기 위해
    # 아주 조금만 morphology 적용
    kernel_close = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (15, 15),
    )

    dark_mask = cv2.morphologyEx(
        dark_mask,
        cv2.MORPH_CLOSE,
        kernel_close,
        iterations=2,
    )

    kernel_open = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (5, 5),
    )

    dark_mask = cv2.morphologyEx(
        dark_mask,
        cv2.MORPH_OPEN,
        kernel_open,
        iterations=1,
    )

    contours, _ = cv2.findContours(
        dark_mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    if not contours:
        return None, None, dark_mask

    image_area = w * h

    candidates = []

    for contour in contours:
        area = cv2.contourArea(contour)

        area_ratio = area / image_area

        if area_ratio < MIN_BOARD_AREA_RATIO:
            continue

        candidates.append(
            (area, contour)
        )

    if not candidates:
        return None, None, dark_mask

    # 가장 큰 어두운 영역
    candidates.sort(
        key=lambda x: x[0],
        reverse=True,
    )

    contour = candidates[0][1]

    # Convex hull로 내부 물체/홈 등의 영향 감소
    hull = cv2.convexHull(contour)

    perimeter = cv2.arcLength(
        hull,
        True,
    )

    approx = cv2.approxPolyDP(
        hull,
        0.02 * perimeter,
        True,
    )

    if len(approx) == 4:
        points = approx.reshape(4, 2)

    else:
        # 4점 approximation 실패 시 fallback.
        # 완전한 perspective 보정은 아니지만
        # 기준판 후보 확인용으로 충분함.
        rect = cv2.minAreaRect(hull)

        points = cv2.boxPoints(
            rect
        )

    points = order_points(points)

    return contour, points, dark_mask


def warp_board(image, points):
    """검출된 판을 800x800 정규화 영상으로 변환."""

    src = order_points(points)

    dst = np.array(
        [
            [0, 0],
            [WARP_SIZE - 1, 0],
            [WARP_SIZE - 1, WARP_SIZE - 1],
            [0, WARP_SIZE - 1],
        ],
        dtype=np.float32,
    )

    matrix = cv2.getPerspectiveTransform(
        src,
        dst,
    )

    warped = cv2.warpPerspective(
        image,
        matrix,
        (WARP_SIZE, WARP_SIZE),
    )

    return warped


def main():

    pipeline = Pipeline()
    config = Config()

    profile_list = pipeline.get_stream_profile_list(
        OBSensorType.COLOR_SENSOR
    )

    color_profile = profile_list.get_video_stream_profile(
        WIDTH,
        HEIGHT,
        OBFormat.MJPG,
        FPS,
    )

    config.enable_stream(
        color_profile
    )

    print("=" * 70)
    print("기준판 자동 검출 테스트")
    print(
        f"Color: {WIDTH}x{HEIGHT} "
        f"@{FPS}fps MJPG"
    )
    print(
        f"DARK_THRESHOLD = "
        f"{DARK_THRESHOLD}"
    )
    print("")
    print("SPACE : 현재 결과 저장")
    print("Q/ESC : 종료")
    print("=" * 70)

    pipeline.start(config)

    try:

        while True:

            frames = pipeline.wait_for_frames(
                100
            )

            if frames is None:
                continue

            color_frame = (
                frames.get_color_frame()
            )

            if color_frame is None:
                continue

            image = color_frame_to_bgr(
                color_frame
            )

            if image is None:
                continue

            contour, points, mask = (
                find_board(image)
            )

            overlay = image.copy()

            warped = np.zeros(
                (
                    WARP_SIZE,
                    WARP_SIZE,
                    3,
                ),
                dtype=np.uint8,
            )

            if points is not None:

                pts_int = points.astype(
                    np.int32
                )

                cv2.polylines(
                    overlay,
                    [pts_int],
                    True,
                    (0, 255, 0),
                    4,
                )

                # 꼭짓점 표시
                for i, point in enumerate(
                    pts_int
                ):

                    x, y = point

                    cv2.circle(
                        overlay,
                        (x, y),
                        8,
                        (0, 0, 255),
                        -1,
                    )

                    cv2.putText(
                        overlay,
                        str(i),
                        (x + 10, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (0, 255, 255),
                        2,
                    )

                warped = warp_board(
                    image,
                    points,
                )

                cv2.putText(
                    overlay,
                    "BOARD DETECTED",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.0,
                    (0, 255, 0),
                    2,
                )

            else:

                cv2.putText(
                    overlay,
                    "BOARD NOT FOUND",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.0,
                    (0, 0, 255),
                    2,
                )

            cv2.imshow(
                "1. Original + Board",
                overlay,
            )

            cv2.imshow(
                "2. Dark Mask",
                mask,
            )

            cv2.imshow(
                "3. Warped Board",
                warped,
            )

            key = cv2.waitKey(1) & 0xFF

            if key in (
                ord("q"),
                27,
            ):
                break

            if key == 32:

                cv2.imwrite(
                    "board_detection_overlay.png",
                    overlay,
                )

                cv2.imwrite(
                    "board_detection_mask.png",
                    mask,
                )

                cv2.imwrite(
                    "board_detection_warped.png",
                    warped,
                )

                print(
                    "현재 결과 저장 완료:"
                )

                print(
                    "  board_detection_overlay.png"
                )

                print(
                    "  board_detection_mask.png"
                )

                print(
                    "  board_detection_warped.png"
                )

    finally:

        pipeline.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
