"""Cài đặt tham chiếu (NumPy, KHÔNG dùng cv2): ánh xạ ngược + nội suy.

Với mỗi pixel đích (x, y) tìm tọa độ nguồn (x_s, y_s) = T⁻¹(x, y) rồi nội suy từ
các pixel lân cận: nearest (1 điểm), bilinear (2x2), bicubic Keys (4x4, a = −0.75 như OpenCV).
Mục đích duy nhất là so sánh với OpenCV và giải thích cơ chế; không tối ưu.
"""
from __future__ import annotations

import numpy as np

from reference.matrices import to_homogeneous

BICUBIC_A: float = -0.75           # hệ số a của nhân Keys [5], OpenCV dùng −0.75
PIXEL_MAX: int = 255
INTERPS: tuple[str, ...] = ("nearest", "linear", "cubic")
BORDERS: tuple[str, ...] = ("constant", "replicate", "reflect", "reflect101", "wrap")


def _border_index(idx: np.ndarray, n: int, mode: str) -> tuple[np.ndarray, np.ndarray]:
    """Đưa chỉ số ngoài [0, n) về trong ảnh theo chế độ biên; trả về (chỉ số, mặt nạ hợp lệ)."""
    inside = (idx >= 0) & (idx < n)
    if mode == "constant":
        return np.clip(idx, 0, n - 1), inside
    if mode == "replicate":
        return np.clip(idx, 0, n - 1), np.ones_like(inside)
    if mode == "wrap":
        return np.mod(idx, n), np.ones_like(inside)
    if mode == "reflect":                       # fedcba|abcdef|fedcba, chu kỳ 2n
        m = np.mod(idx, 2 * n)
        return np.where(m >= n, 2 * n - 1 - m, m), np.ones_like(inside)
    if mode == "reflect101":                    # fedcb|abcdef|edcba, chu kỳ 2n − 2
        period = max(2 * n - 2, 1)
        m = np.mod(idx, period)
        return np.where(m >= n, period - m, m), np.ones_like(inside)
    raise ValueError(f"Chế độ biên không hỗ trợ: {mode}")


def keys_cubic(t: np.ndarray, a: float = BICUBIC_A) -> np.ndarray:
    """Nhân nội suy bậc ba Keys [5] tại khoảng cách t (mảng bất kỳ)."""
    t = np.abs(t)
    t2, t3 = t * t, t * t * t
    near = (a + 2) * t3 - (a + 3) * t2 + 1
    far = a * t3 - 5 * a * t2 + 8 * a * t - 4 * a
    return np.where(t <= 1, near, np.where(t < 2, far, 0.0))


def _taps(coord: np.ndarray, interp: str) -> tuple[np.ndarray, list[int], list[np.ndarray]]:
    """Chỉ số gốc, các độ lệch lân cận và trọng số tương ứng theo một trục."""
    if interp == "nearest":
        return np.floor(coord + 0.5).astype(np.int64), [0], [np.ones_like(coord)]
    base = np.floor(coord)
    t = coord - base
    if interp == "linear":
        return base.astype(np.int64), [0, 1], [1 - t, t]
    if interp == "cubic":
        offsets = [-1, 0, 1, 2]
        return base.astype(np.int64), offsets, [keys_cubic(t - k) for k in offsets]
    raise ValueError(f"Phương pháp nội suy không hỗ trợ: {interp}")


def _round_to_dtype(values: np.ndarray, dtype: np.dtype) -> np.ndarray:
    """Làm tròn nửa lên (như phép dịch bit có cộng 0.5 của OpenCV) và bão hòa về uint8."""
    if dtype != np.uint8:
        return values.astype(dtype)
    return np.clip(np.floor(values + 0.5), 0, PIXEL_MAX).astype(np.uint8)


def remap_ref(img: np.ndarray, map_x: np.ndarray, map_y: np.ndarray, interp: str = "linear",
              border: str = "constant", border_value: float = 0.0,
              coord_bits: int | None = None) -> np.ndarray:
    """Lấy mẫu ảnh tại tọa độ thực (map_x, map_y) — lõi của mọi phép ánh xạ ngược.

    ``coord_bits``: nếu khác None, lượng tử hóa tọa độ về lưới 1/2^bits pixel
    (OpenCV dùng INTER_BITS = 5, tức 1/32 pixel) — chỉ để kiểm chứng nguồn sai khác.
    """
    if coord_bits is not None:
        q = float(1 << coord_bits)
        map_x, map_y = np.round(map_x * q) / q, np.round(map_y * q) / q
    src = img.astype(np.float64)
    h, w = img.shape[:2]
    x0, ox, wx = _taps(map_x, interp)
    y0, oy, wy = _taps(map_y, interp)
    out = np.zeros(map_x.shape + img.shape[2:], np.float64)
    for dy, wgt_y in zip(oy, wy):
        iy, my = _border_index(y0 + dy, h, border)
        for dx, wgt_x in zip(ox, wx):
            ix, mx = _border_index(x0 + dx, w, border)
            vals = src[iy, ix]
            valid = my & mx
            if img.ndim == 3:
                valid, wgt = valid[..., None], (wgt_y * wgt_x)[..., None]
            else:
                wgt = wgt_y * wgt_x
            out += wgt * np.where(valid, vals, border_value)
    return _round_to_dtype(out, img.dtype)


def _grid(dsize: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    """Lưới tọa độ pixel đích (x, y) kích thước dsize = (rộng, cao)."""
    w, h = dsize
    return np.meshgrid(np.arange(w, dtype=np.float64), np.arange(h, dtype=np.float64))


def warp_affine_ref(img: np.ndarray, m: np.ndarray, dsize: tuple[int, int], interp: str = "linear",
                    border: str = "constant", border_value: float = 0.0,
                    coord_bits: int | None = None) -> np.ndarray:
    """Tương đương cv2.warpAffine: dst(x, y) = src(M⁻¹ · [x, y, 1]ᵀ)."""
    inv = np.linalg.inv(to_homogeneous(m))
    xs, ys = _grid(dsize)
    map_x = inv[0, 0] * xs + inv[0, 1] * ys + inv[0, 2]
    map_y = inv[1, 0] * xs + inv[1, 1] * ys + inv[1, 2]
    return remap_ref(img, map_x, map_y, interp, border, border_value, coord_bits)


def warp_perspective_ref(img: np.ndarray, h: np.ndarray, dsize: tuple[int, int], interp: str = "linear",
                         border: str = "constant", border_value: float = 0.0,
                         coord_bits: int | None = None) -> np.ndarray:
    """Tương đương cv2.warpPerspective: [x_s·w, y_s·w, w]ᵀ = H⁻¹ · [x, y, 1]ᵀ."""
    inv = np.linalg.inv(h)
    xs, ys = _grid(dsize)
    den = inv[2, 0] * xs + inv[2, 1] * ys + inv[2, 2]
    den = np.where(den == 0, np.inf, den)       # điểm ở vô cực → rơi ra ngoài ảnh
    map_x = (inv[0, 0] * xs + inv[0, 1] * ys + inv[0, 2]) / den
    map_y = (inv[1, 0] * xs + inv[1, 1] * ys + inv[1, 2]) / den
    return remap_ref(img, map_x, map_y, interp, border, border_value, coord_bits)


def resize_ref(img: np.ndarray, fx: float, fy: float | None = None, interp: str = "linear") -> np.ndarray:
    """Tương đương cv2.resize(fx, fy): căn tâm pixel x_s = (x + 0.5)/fx − 0.5, biên lặp pixel."""
    fy = fx if fy is None else fy
    h, w = img.shape[:2]
    dsize = (int(round(w * fx)), int(round(h * fy)))
    xs, ys = _grid(dsize)
    map_x, map_y = (xs + 0.5) / fx - 0.5, (ys + 0.5) / fy - 0.5
    return remap_ref(img, map_x, map_y, interp, border="replicate")
