"""YOLO 형식 라벨(txt)을 jpg 이미지 위에 bbox로 그려서 저장하는 스크립트.

라벨 한 줄 형식: class_id x_center y_center width height  (모두 0~1로 정규화된 값)

사용법:
    pip install opencv-python numpy
    python draw_bbox.py
(아래 설정 부분의 경로만 수정하면 됩니다.)
"""

from pathlib import Path

import cv2
import numpy as np

# ===== 설정 =====
IMAGE_PATH = Path("images/250424_142806480763.jpg")        # 원본 이미지
LABEL_PATH = Path("images/250424_142806480763.txt")        # bbox 좌표 txt
OUTPUT_PATH = Path("images/250424_142806480763_bbox.jpg")  # 결과 저장 경로
CLASS_NAMES = {}                      # 예: {0: "car", 1: "person", 5: "sign"} (비워두면 번호만 표시)
THICKNESS = 2
SHOW_WINDOW = True                    # True면 결과를 창으로도 확인
# ================


def imread_unicode(path: Path):
    """한글/공백 경로에서도 읽히도록 numpy로 디코딩."""
    data = np.fromfile(str(path), dtype=np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def imwrite_unicode(path: Path, img):
    ok, buf = cv2.imencode(path.suffix or ".jpg", img)
    if not ok:
        raise IOError(f"저장 실패: {path}")
    buf.tofile(str(path))


def class_color(class_id: int):
    """클래스별로 항상 같은 색(BGR)을 반환."""
    rng = np.random.RandomState(class_id * 7 + 3)
    return tuple(int(c) for c in rng.randint(50, 255, 3))


def main():
    img = imread_unicode(IMAGE_PATH)
    if img is None:
        raise FileNotFoundError(f"이미지를 읽을 수 없습니다: {IMAGE_PATH}")
    h, w = img.shape[:2]

    for line in LABEL_PATH.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) < 5:
            continue
        cls = int(float(parts[0]))
        xc, yc, bw, bh = map(float, parts[1:5])

        # 정규화 좌표 -> 픽셀 좌표
        x1 = int((xc - bw / 2) * w)
        y1 = int((yc - bh / 2) * h)
        x2 = int((xc + bw / 2) * w)
        y2 = int((yc + bh / 2) * h)

        color = class_color(cls)
        cv2.rectangle(img, (x1, y1), (x2, y2), color, THICKNESS)

        label = str(CLASS_NAMES.get(cls, cls))
        (tw, th), base = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        ty = max(y1, th + base)  # 이미지 위로 벗어나지 않게
        cv2.rectangle(img, (x1, ty - th - base), (x1 + tw, ty), color, -1)
        cv2.putText(img, label, (x1, ty - base), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (255, 255, 255), 1, cv2.LINE_AA)

    imwrite_unicode(OUTPUT_PATH, img)
    print(f"저장 완료: {OUTPUT_PATH}")

    if SHOW_WINDOW:
        cv2.imshow("bbox", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()