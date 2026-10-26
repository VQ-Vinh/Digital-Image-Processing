"""Tự tính các ma trận biến đổi hình học bằng NumPy (KHÔNG dùng cv2).

Chỉ dùng để đối chứng với cv2.getRotationMatrix2D, cv2.getAffineTransform,
cv2.getPerspectiveTransform. Tọa độ: tâm pixel ở số nguyên, trục y hướng xuống.
"""
from __future__ import annotations

import math

import numpy as np

N_AFFINE_PTS: int = 3
N_HOMOGRAPHY_PTS: int = 4
SIZE_EPS: float = 1e-6     # dung sai làm tròn lên kích thước khung mở rộng
NORM_DIST: float = math.sqrt(2.0)  # khoảng cách TB tới gốc sau chuẩn hóa Hartley


def to_homogeneous(m: np.ndarray) -> np.ndarray:
    """Mở rộng ma trận affine 2x3 thành 3x3 (thêm hàng [0, 0, 1])."""
    if m.shape == (3, 3):
        return m.astype(np.float64)
    return np.vstack([m.astype(np.float64), [0.0, 0.0, 1.0]])


def translation_matrix(tx: float, ty: float) -> np.ndarray:
    """Ma trận 2x3 của phép tịnh tiến."""
    return np.array([[1.0, 0.0, tx], [0.0, 1.0, ty]])


def rotation_matrix(center: tuple[float, float], angle_deg: float, scale: float = 1.0) -> np.ndarray:
    """Ma trận xoay 2x3 quanh ``center`` (góc dương = ngược chiều kim đồng hồ khi y hướng xuống).

    M = [[ α, β, (1−α)cx − β cy],
         [−β, α, β cx + (1−α) cy]],  α = s·cosθ, β = s·sinθ.
    """
    theta = math.radians(angle_deg)
    alpha, beta = scale * math.cos(theta), scale * math.sin(theta)
    cx, cy = center
    return np.array([[alpha, beta, (1 - alpha) * cx - beta * cy],
                     [-beta, alpha, beta * cx + (1 - alpha) * cy]])


def rotation_matrix_expand(size: tuple[int, int], angle_deg: float,
                           scale: float = 1.0) -> tuple[np.ndarray, tuple[int, int]]:
    """Ma trận xoay quanh tâm ảnh kèm khung mở rộng vừa đủ chứa ảnh; trả về (M, (w', h'))."""
    w, h = size
    center = ((w - 1) / 2, (h - 1) / 2)
    theta = math.radians(angle_deg)
    c, s = abs(scale * math.cos(theta)), abs(scale * math.sin(theta))
    new_w, new_h = math.ceil(w * c + h * s - SIZE_EPS), math.ceil(w * s + h * c - SIZE_EPS)
    m = rotation_matrix(center, angle_deg, scale)
    m[0, 2] += (new_w - 1) / 2 - center[0]
    m[1, 2] += (new_h - 1) / 2 - center[1]
    return m, (new_w, new_h)


def affine_from_3pts(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """Ma trận affine 2x3 từ 3 cặp điểm: giải hệ tuyến tính 6 ẩn.

    Với mỗi cặp (x, y) → (u, v): u = a·x + b·y + c,  v = d·x + e·y + f.
    """
    src = np.asarray(src, np.float64).reshape(N_AFFINE_PTS, 2)
    dst = np.asarray(dst, np.float64).reshape(N_AFFINE_PTS, 2)
    a = np.zeros((2 * N_AFFINE_PTS, 6))
    a[0::2, 0:2], a[0::2, 2] = src, 1.0
    a[1::2, 3:5], a[1::2, 5] = src, 1.0
    params = np.linalg.solve(a, dst.reshape(-1))
    return params.reshape(2, 3)


def _hartley_normalization(pts: np.ndarray) -> np.ndarray:
    """Ma trận 3x3 đưa trọng tâm các điểm về gốc, khoảng cách TB tới gốc = √2 [3]."""
    centroid = pts.mean(axis=0)
    mean_dist = np.mean(np.linalg.norm(pts - centroid, axis=1))
    s = NORM_DIST / mean_dist
    return np.array([[s, 0, -s * centroid[0]], [0, s, -s * centroid[1]], [0, 0, 1]])


def homography_from_4pts(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """Homography 3x3 từ 4 cặp điểm bằng DLT có chuẩn hóa (Hartley & Zisserman [3]).

    Mỗi cặp (x, y) → (u, v) cho 2 hàng của A trong A·h = 0; h là vector kỳ dị phải
    ứng với giá trị kỳ dị nhỏ nhất. Kết quả chuẩn hóa h33 = 1 như OpenCV.
    """
    src = np.asarray(src, np.float64).reshape(N_HOMOGRAPHY_PTS, 2)
    dst = np.asarray(dst, np.float64).reshape(N_HOMOGRAPHY_PTS, 2)
    t_src, t_dst = _hartley_normalization(src), _hartley_normalization(dst)
    ps = (t_src @ np.c_[src, np.ones(N_HOMOGRAPHY_PTS)].T).T
    pd = (t_dst @ np.c_[dst, np.ones(N_HOMOGRAPHY_PTS)].T).T
    a = np.zeros((2 * N_HOMOGRAPHY_PTS, 9))
    for i, ((x, y, _), (u, v, _)) in enumerate(zip(ps, pd)):
        a[2 * i] = [-x, -y, -1, 0, 0, 0, u * x, u * y, u]
        a[2 * i + 1] = [0, 0, 0, -x, -y, -1, v * x, v * y, v]
    _, _, vt = np.linalg.svd(a)
    h_norm = vt[-1].reshape(3, 3)
    h = np.linalg.inv(t_dst) @ h_norm @ t_src
    return h / h[2, 2]
