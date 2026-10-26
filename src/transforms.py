"""Hàm bọc gọn các phép biến đổi hình học của OpenCV.

Quy ước:
- Ảnh là mảng NumPy (H, W) hoặc (H, W, C), kiểu uint8.
- Kích thước ``dsize`` theo thứ tự (rộng, cao) như OpenCV.
- Tọa độ pixel: tâm pixel nằm ở số nguyên, gốc (0, 0) ở góc trên-trái;
  tâm hình học của ảnh rộng w, cao h là ((w - 1) / 2, (h - 1) / 2).
"""
from __future__ import annotations

import math

import cv2
import numpy as np

INTERP: dict[str, int] = {
    "nearest": cv2.INTER_NEAREST,
    "linear": cv2.INTER_LINEAR,
    "cubic": cv2.INTER_CUBIC,
    "area": cv2.INTER_AREA,
    "lanczos4": cv2.INTER_LANCZOS4,
}

BORDER: dict[str, int] = {
    "constant": cv2.BORDER_CONSTANT,
    "replicate": cv2.BORDER_REPLICATE,
    "reflect": cv2.BORDER_REFLECT,
    "reflect101": cv2.BORDER_REFLECT_101,
    "wrap": cv2.BORDER_WRAP,
}

INTERP_NAME: dict[int, str] = {v: f"INTER_{k.upper()}" for k, v in INTERP.items()}
BORDER_NAME: dict[int, str] = {
    cv2.BORDER_CONSTANT: "BORDER_CONSTANT",
    cv2.BORDER_REPLICATE: "BORDER_REPLICATE",
    cv2.BORDER_REFLECT: "BORDER_REFLECT",
    cv2.BORDER_REFLECT_101: "BORDER_REFLECT_101",
    cv2.BORDER_WRAP: "BORDER_WRAP",
}

# Dung sai khi làm tròn lên kích thước khung mở rộng (tránh 400.00000000000004 → 401).
SIZE_EPS: float = 1e-6


def image_size(img: np.ndarray) -> tuple[int, int]:
    """Trả về kích thước (rộng, cao) của ảnh."""
    return img.shape[1], img.shape[0]


def image_center(size: tuple[int, int]) -> tuple[float, float]:
    """Tâm hình học của ảnh có kích thước (rộng, cao) theo quy ước tâm pixel."""
    w, h = size
    return (w - 1) / 2.0, (h - 1) / 2.0


def scale(
    img: np.ndarray,
    fx: float,
    fy: float | None = None,
    interp: int = cv2.INTER_LINEAR,
) -> np.ndarray:
    """Co giãn ảnh theo hệ số fx, fy (fy mặc định bằng fx) bằng cv2.resize."""
    fy = fx if fy is None else fy
    return cv2.resize(img, None, fx=fx, fy=fy, interpolation=interp)


def resize_to(img: np.ndarray, dsize: tuple[int, int], interp: int = cv2.INTER_LINEAR) -> np.ndarray:
    """Co giãn ảnh về kích thước dsize = (rộng, cao)."""
    return cv2.resize(img, dsize, interpolation=interp)


def translation_matrix(tx: float, ty: float) -> np.ndarray:
    """Ma trận affine 2x3 của phép tịnh tiến (tx, ty)."""
    return np.array([[1.0, 0.0, tx], [0.0, 1.0, ty]])


def translate(
    img: np.ndarray,
    tx: float,
    ty: float,
    interp: int = cv2.INTER_LINEAR,
    border: int = cv2.BORDER_CONSTANT,
    border_value: int = 0,
) -> np.ndarray:
    """Tịnh tiến ảnh (tx sang phải, ty xuống dưới), giữ nguyên khung."""
    m = translation_matrix(tx, ty)
    return cv2.warpAffine(img, m, image_size(img), flags=interp,
                          borderMode=border, borderValue=border_value)


def expanded_size(size: tuple[int, int], angle_deg: float, scale_factor: float = 1.0) -> tuple[int, int]:
    """Kích thước khung (rộng, cao) vừa đủ chứa trọn ảnh sau khi xoay.

    w' = |s|(w|cosθ| + h|sinθ|),  h' = |s|(w|sinθ| + h|cosθ|).
    """
    w, h = size
    theta = math.radians(angle_deg)
    c, s = abs(math.cos(theta)) * abs(scale_factor), abs(math.sin(theta)) * abs(scale_factor)
    new_w = math.ceil(w * c + h * s - SIZE_EPS)
    new_h = math.ceil(w * s + h * c - SIZE_EPS)
    return new_w, new_h


def rotation_matrix(
    size: tuple[int, int],
    angle_deg: float,
    scale_factor: float = 1.0,
    expand: bool = False,
) -> tuple[np.ndarray, tuple[int, int]]:
    """Ma trận xoay 2x3 quanh tâm ảnh và kích thước khung ra.

    Góc dương = ngược chiều kim đồng hồ (như cv2.getRotationMatrix2D).
    Nếu ``expand`` thì tính lại kích thước khung và cộng bù tịnh tiến để
    tâm ảnh cũ trùng tâm khung mới, nhờ đó không bị cắt góc.
    """
    center = image_center(size)
    m = cv2.getRotationMatrix2D(center, angle_deg, scale_factor)
    if not expand:
        return m, size
    out_size = expanded_size(size, angle_deg, scale_factor)
    new_cx, new_cy = image_center(out_size)
    m[0, 2] += new_cx - center[0]
    m[1, 2] += new_cy - center[1]
    return m, out_size


def rotate(
    img: np.ndarray,
    angle_deg: float,
    scale_factor: float = 1.0,
    expand: bool = False,
    interp: int = cv2.INTER_LINEAR,
    border: int = cv2.BORDER_CONSTANT,
    border_value: int = 0,
) -> np.ndarray:
    """Xoay ảnh quanh tâm; ``expand=True`` để mở rộng khung, giữ trọn ảnh."""
    m, out_size = rotation_matrix(image_size(img), angle_deg, scale_factor, expand)
    return cv2.warpAffine(img, m, out_size, flags=interp,
                          borderMode=border, borderValue=border_value)


def affine_3pts(
    img: np.ndarray,
    src_pts: np.ndarray,
    dst_pts: np.ndarray,
    dsize: tuple[int, int] | None = None,
    interp: int = cv2.INTER_LINEAR,
    border: int = cv2.BORDER_CONSTANT,
    border_value: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Biến đổi affine xác định bởi 3 cặp điểm. Trả về (ảnh ra, ma trận 2x3)."""
    m = cv2.getAffineTransform(np.float32(src_pts), np.float32(dst_pts))
    dsize = image_size(img) if dsize is None else dsize
    out = cv2.warpAffine(img, m, dsize, flags=interp,
                         borderMode=border, borderValue=border_value)
    return out, m


def perspective_4pts(
    img: np.ndarray,
    src_pts: np.ndarray,
    dst_pts: np.ndarray,
    dsize: tuple[int, int] | None = None,
    interp: int = cv2.INTER_LINEAR,
    border: int = cv2.BORDER_CONSTANT,
    border_value: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Biến đổi phối cảnh xác định bởi 4 cặp điểm. Trả về (ảnh ra, homography 3x3)."""
    h = cv2.getPerspectiveTransform(np.float32(src_pts), np.float32(dst_pts))
    dsize = image_size(img) if dsize is None else dsize
    out = cv2.warpPerspective(img, h, dsize, flags=interp,
                              borderMode=border, borderValue=border_value)
    return out, h


def apply_affine_to_points(m: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """Áp ma trận affine 2x3 lên các điểm (N, 2)."""
    return cv2.transform(np.float64(pts).reshape(-1, 1, 2), m).reshape(-1, 2)


def apply_homography_to_points(h: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """Áp homography 3x3 lên các điểm (N, 2)."""
    return cv2.perspectiveTransform(np.float64(pts).reshape(-1, 1, 2), h).reshape(-1, 2)
