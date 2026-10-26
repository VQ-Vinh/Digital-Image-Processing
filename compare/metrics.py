"""Độ đo so sánh hai ảnh: MSE, PSNR, tỉ lệ pixel lệch > 1 mức xám, ảnh hiệu."""
from __future__ import annotations

import math

import numpy as np

PIXEL_MAX: float = 255.0
DIFF_TOLERANCE: int = 1      # lệch "đáng kể" khi |a - b| > 1 mức xám
PERCENT: float = 100.0


def _check_same_shape(a: np.ndarray, b: np.ndarray) -> None:
    """Báo lỗi nếu hai ảnh khác kích thước."""
    if a.shape != b.shape:
        raise ValueError(f"Hai ảnh khác kích thước: {a.shape} và {b.shape}")


def mse(a: np.ndarray, b: np.ndarray) -> float:
    """Sai số bình phương trung bình."""
    _check_same_shape(a, b)
    d = a.astype(np.float64) - b.astype(np.float64)
    return float(np.mean(d * d))


def psnr(a: np.ndarray, b: np.ndarray, peak: float = PIXEL_MAX) -> float:
    """PSNR (dB) = 10 log10(peak^2 / MSE); trả về +inf nếu hai ảnh trùng nhau."""
    m = mse(a, b)
    return math.inf if m == 0 else 10.0 * math.log10(peak * peak / m)


def abs_diff(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """|a - b| theo từng pixel (float64); ảnh màu lấy giá trị lớn nhất qua các kênh."""
    _check_same_shape(a, b)
    d = np.abs(a.astype(np.float64) - b.astype(np.float64))
    return d.max(axis=2) if d.ndim == 3 else d


def pct_diff_gt(a: np.ndarray, b: np.ndarray, tol: int = DIFF_TOLERANCE) -> float:
    """Tỉ lệ (%) pixel có |a - b| > tol."""
    return float(np.mean(abs_diff(a, b) > tol) * PERCENT)


def max_abs_diff(a: np.ndarray, b: np.ndarray) -> float:
    """Độ lệch tuyệt đối lớn nhất."""
    return float(abs_diff(a, b).max())


def diff_image(a: np.ndarray, b: np.ndarray, gain: float) -> np.ndarray:
    """Ảnh hiệu phóng đại tương phản: min(|a - b| * gain, 255), kiểu uint8."""
    return np.clip(abs_diff(a, b) * gain, 0, PIXEL_MAX).astype(np.uint8)


def format_psnr(value: float) -> str:
    """Định dạng PSNR cho bảng/tiêu đề (∞ khi hai ảnh trùng nhau)."""
    return "∞" if math.isinf(value) else f"{value:.2f}"
