"""Đối chứng reference/ với OpenCV: ma trận (sai số < 1e-6) và ảnh (PSNR > ngưỡng)."""
from __future__ import annotations

import cv2
import numpy as np
import pytest

from compare.metrics import pct_diff_gt, psnr
from reference import matrices as RM
from reference import warp_ref as RW
from src import images

MATRIX_TOL: float = 1e-6
MIN_PSNR_DB: float = 50.0        # > 50 dB: hai ảnh gần như đồng nhất (MSE < 0.65)
MAX_PCT_DIFF_GT1: float = 1.0    # < 1% pixel lệch quá 1 mức xám
INTERP_CV = {"nearest": cv2.INTER_NEAREST, "linear": cv2.INTER_LINEAR, "cubic": cv2.INTER_CUBIC}
SRC4 = np.array([[40, 40], [470, 40], [470, 470], [40, 470]], np.float32)
DST4 = np.array([[150, 70], [380, 30], [490, 490], [30, 420]], np.float32)


@pytest.fixture(scope="module")
def cam() -> np.ndarray:
    """Ảnh xám 512x512."""
    return images.load_camera()


@pytest.mark.parametrize("center, angle, scale", [((255.5, 255.5), 45, 1.0), ((100, 300), -30, 0.7),
                                                  ((0, 0), 90, 2.0), ((319.5, 239.5), 12.5, 1.3)])
def test_rotation_matrix(center: tuple[float, float], angle: float, scale: float) -> None:
    """Ma trận xoay tự tính trùng cv2.getRotationMatrix2D."""
    err = np.abs(RM.rotation_matrix(center, angle, scale) - cv2.getRotationMatrix2D(center, angle, scale)).max()
    assert err < MATRIX_TOL


def test_affine_matrix() -> None:
    """Affine 3 điểm tự tính trùng cv2.getAffineTransform."""
    src, dst = SRC4[:3], DST4[:3]
    assert np.abs(RM.affine_from_3pts(src, dst) - cv2.getAffineTransform(src, dst)).max() < MATRIX_TOL


def test_homography_matrix() -> None:
    """Homography DLT tự tính trùng cv2.getPerspectiveTransform."""
    err = np.abs(RM.homography_from_4pts(SRC4, DST4) - cv2.getPerspectiveTransform(SRC4, DST4)).max()
    assert err < MATRIX_TOL


def test_rotation_expand_matches_src() -> None:
    """Ma trận xoay mở rộng khung tự tính trùng bản dùng OpenCV trong src/transforms.py."""
    from src.transforms import rotation_matrix
    m_ref, size_ref = RM.rotation_matrix_expand((600, 400), 30)
    m_cv, size_cv = rotation_matrix((600, 400), 30, expand=True)
    assert size_ref == size_cv
    assert np.abs(m_ref - m_cv).max() < MATRIX_TOL


def _check(a: np.ndarray, b: np.ndarray) -> None:
    """Hai ảnh gần như đồng nhất."""
    assert a.shape == b.shape and a.dtype == b.dtype
    assert psnr(a, b) > MIN_PSNR_DB
    assert pct_diff_gt(a, b) < MAX_PCT_DIFF_GT1


@pytest.mark.parametrize("interp", list(INTERP_CV))
def test_resize(cam: np.ndarray, interp: str) -> None:
    """resize ×4 tự cài đặt ≈ cv2.resize."""
    small = cv2.resize(cam, (128, 128), interpolation=cv2.INTER_AREA)
    _check(RW.resize_ref(small, 4, interp=interp), cv2.resize(small, None, fx=4, fy=4,
                                                              interpolation=INTERP_CV[interp]))


@pytest.mark.parametrize("interp", list(INTERP_CV))
def test_warp_affine(cam: np.ndarray, interp: str) -> None:
    """Xoay 45° tự cài đặt ≈ cv2.warpAffine."""
    m = RM.rotation_matrix((255.5, 255.5), 45)
    _check(RW.warp_affine_ref(cam, m, (512, 512), interp), cv2.warpAffine(cam, m, (512, 512),
                                                                         flags=INTERP_CV[interp]))


@pytest.mark.parametrize("interp", list(INTERP_CV))
def test_warp_perspective(cam: np.ndarray, interp: str) -> None:
    """Phối cảnh tự cài đặt ≈ cv2.warpPerspective."""
    h = RM.homography_from_4pts(SRC4, DST4)
    _check(RW.warp_perspective_ref(cam, h, (512, 512), interp),
           cv2.warpPerspective(cam, h, (512, 512), flags=INTERP_CV[interp]))


@pytest.mark.parametrize("border, cv_border", [("replicate", cv2.BORDER_REPLICATE), ("reflect", cv2.BORDER_REFLECT),
                                               ("reflect101", cv2.BORDER_REFLECT_101), ("wrap", cv2.BORDER_WRAP)])
def test_border_modes(cam: np.ndarray, border: str, cv_border: int) -> None:
    """Các chế độ biên tự cài đặt ≈ borderMode của OpenCV (xoay 30°, thu nhỏ 0.6)."""
    m = RM.rotation_matrix((255.5, 255.5), 30, 0.6)
    _check(RW.warp_affine_ref(cam, m, (512, 512), "linear", border),
           cv2.warpAffine(cam, m, (512, 512), flags=cv2.INTER_LINEAR, borderMode=cv_border))


def test_color_image() -> None:
    """Ảnh màu (H, W, 3) cũng được xử lý đúng."""
    img = images.load_astronaut()
    m = RM.rotation_matrix((255.5, 255.5), 20)
    _check(RW.warp_affine_ref(img, m, (512, 512)), cv2.warpAffine(img, m, (512, 512)))


def test_keys_kernel_partition_of_unity() -> None:
    """Tổng 4 trọng số bicubic luôn bằng 1 và nhân bằng 1 tại 0, bằng 0 tại các số nguyên khác."""
    t = np.linspace(0, 1, 33)
    total = sum(RW.keys_cubic(t - k) for k in (-1, 0, 1, 2))
    np.testing.assert_allclose(total, 1.0, atol=1e-12)
    np.testing.assert_allclose(RW.keys_cubic(np.array([0.0, 1.0, 2.0, -1.0])), [1, 0, 0, 0], atol=1e-12)
