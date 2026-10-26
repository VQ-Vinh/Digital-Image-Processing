"""Ứng dụng: nắn chỉnh ảnh tài liệu chụp nghiêng bằng biến đổi phối cảnh (4 góc → hình chữ nhật).

Quy trình: tìm 4 góc tờ giấy (tự động hoặc do người dùng click) → sắp xếp góc
TL, TR, BR, BL → ước lượng kích thước trang → getPerspectiveTransform → warpPerspective.
"""
from __future__ import annotations

import cv2
import numpy as np

BLUR_KSIZE: tuple[int, int] = (5, 5)
CLOSE_KSIZE: tuple[int, int] = (7, 7)  # lấp các lỗ do chữ tối trong vùng giấy
CANNY_LOW: int = 50
CANNY_HIGH: int = 150
DILATE_ITER: int = 2
APPROX_EPS_RATIO: float = 0.02      # epsilon của approxPolyDP theo chu vi
MIN_AREA_RATIO: float = 0.10        # tờ giấy phải chiếm ≥ 10% diện tích ảnh
N_CORNERS: int = 4


def order_corners(pts: np.ndarray) -> np.ndarray:
    """Sắp xếp 4 điểm theo thứ tự trên-trái, trên-phải, dưới-phải, dưới-trái.

    TL có x + y nhỏ nhất, BR có x + y lớn nhất; TR có y - x nhỏ nhất, BL có y - x lớn nhất.
    """
    pts = np.asarray(pts, dtype=np.float64).reshape(N_CORNERS, 2)
    s = pts.sum(axis=1)
    d = pts[:, 1] - pts[:, 0]
    return np.array([pts[np.argmin(s)], pts[np.argmin(d)], pts[np.argmax(s)], pts[np.argmax(d)]])


def _quad_from_binary(binary: np.ndarray) -> np.ndarray | None:
    """Tìm contour ngoài lớn nhất xấp xỉ được bằng tứ giác; trả về 4 góc đã sắp xếp."""
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    min_area = MIN_AREA_RATIO * binary.shape[0] * binary.shape[1]
    for cnt in sorted(contours, key=cv2.contourArea, reverse=True):
        if cv2.contourArea(cnt) < min_area:
            break
        approx = cv2.approxPolyDP(cnt, APPROX_EPS_RATIO * cv2.arcLength(cnt, True), True)
        if len(approx) == N_CORNERS:
            return order_corners(approx.reshape(N_CORNERS, 2))
    return None


def detect_corners_auto(img: np.ndarray) -> np.ndarray | None:
    """Tự tìm 4 góc tờ giấy (giấy sáng trên nền tối hơn).

    Cách 1: ngưỡng Otsu → đóng hình thái → contour lớn nhất → tứ giác.
    Cách 2 (dự phòng): biên Canny + giãn nở → contour → tứ giác.
    Trả về mảng (4, 2) đã sắp xếp, hoặc None nếu không tìm được.
    """
    gray = img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    blurred = cv2.GaussianBlur(gray, BLUR_KSIZE, 0)
    _, mask = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones(CLOSE_KSIZE, np.uint8))
    quad = _quad_from_binary(mask)
    if quad is not None:
        return quad
    edges = cv2.dilate(cv2.Canny(blurred, CANNY_LOW, CANNY_HIGH), None, iterations=DILATE_ITER)
    return _quad_from_binary(edges)


def target_size(corners: np.ndarray) -> tuple[int, int]:
    """Kích thước (rộng, cao) ảnh ra = cạnh dài nhất theo mỗi chiều của tứ giác."""
    tl, tr, br, bl = corners
    width = max(np.linalg.norm(tr - tl), np.linalg.norm(br - bl))
    height = max(np.linalg.norm(bl - tl), np.linalg.norm(br - tr))
    return int(round(width)), int(round(height))


def rectify(
    img: np.ndarray,
    corners: np.ndarray,
    dsize: tuple[int, int] | None = None,
    interp: int = cv2.INTER_CUBIC,
) -> tuple[np.ndarray, np.ndarray]:
    """Nắn tứ giác ``corners`` (TL, TR, BR, BL) thành hình chữ nhật dsize. Trả về (ảnh, H)."""
    corners = order_corners(corners)
    w, h = target_size(corners) if dsize is None else dsize
    dst = np.float32([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]])
    hmat = cv2.getPerspectiveTransform(np.float32(corners), dst)
    out = cv2.warpPerspective(img, hmat, (w, h), flags=interp, borderMode=cv2.BORDER_REPLICATE)
    return out, hmat


def corner_error(found: np.ndarray, truth: np.ndarray) -> float:
    """Sai số trung bình (px) giữa 4 góc tìm được và 4 góc thật (đã cùng thứ tự)."""
    return float(np.mean(np.linalg.norm(order_corners(found) - order_corners(truth), axis=1)))
