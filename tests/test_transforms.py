"""Kiểm thử các hàm bọc OpenCV trong src/transforms.py và ứng dụng src/app_document.py."""
from __future__ import annotations

import cv2
import numpy as np
import pytest

from compare.metrics import psnr
from src import app_document as doc
from src import images
from src import transforms as T

POINT_TOL: float = 1e-6
CORNER_TOL_PX: float = 2.0
RECTIFY_MIN_PSNR_DB: float = 15.0


@pytest.fixture(scope="module")
def coffee() -> np.ndarray:
    """Ảnh màu không vuông 600x400."""
    return images.load_coffee()


def test_scale_shape(coffee: np.ndarray) -> None:
    """resize ×0.5 và ×2 cho đúng kích thước."""
    assert T.scale(coffee, 0.5).shape == (200, 300, 3)
    assert T.scale(coffee, 2, interp=cv2.INTER_CUBIC).shape == (800, 1200, 3)


def test_translate_moves_pixels() -> None:
    """Tịnh tiến nguyên pixel dời đúng nội dung, phần trống bằng borderValue."""
    img = np.arange(100, dtype=np.uint8).reshape(10, 10)
    out = T.translate(img, 3, 2, interp=cv2.INTER_NEAREST)
    np.testing.assert_array_equal(out[2:, 3:], img[:-2, :-3])
    assert out[:2].max() == 0 and out[:, :3].max() == 0


def test_rotate_keep_frame_size(coffee: np.ndarray) -> None:
    """Xoay giữ khung không đổi kích thước."""
    assert T.rotate(coffee, 30).shape == coffee.shape


@pytest.mark.parametrize("angle, size", [(90, (400, 600)), (45, (708, 708)), (30, (720, 647)), (0, (600, 400))])
def test_expanded_size(angle: float, size: tuple[int, int]) -> None:
    """Kích thước khung mở rộng theo công thức w|cos| + h|sin|."""
    assert T.expanded_size((600, 400), angle) == size


def test_rotate_90_expand_equals_rot90(coffee: np.ndarray) -> None:
    """Xoay 90° mở rộng khung (nearest) phải trùng np.rot90 — kiểm tra cộng bù tịnh tiến."""
    out = T.rotate(coffee, 90, expand=True, interp=cv2.INTER_NEAREST)
    np.testing.assert_array_equal(out, np.rot90(coffee))


def test_affine_3pts_maps_points() -> None:
    """Ma trận affine đưa đúng 3 điểm nguồn tới 3 điểm đích."""
    src = np.array([[10, 10], [200, 30], [40, 220]], float)
    dst = np.array([[50, 60], [180, 20], [70, 250]], float)
    _, m = T.affine_3pts(np.zeros((256, 256), np.uint8), src, dst)
    assert np.abs(T.apply_affine_to_points(m, src) - dst).max() < POINT_TOL


def test_perspective_4pts_maps_points() -> None:
    """Homography đưa đúng 4 điểm nguồn tới 4 điểm đích."""
    src = np.array([[0, 0], [255, 0], [255, 255], [0, 255]], float)
    dst = np.array([[30, 20], [220, 5], [250, 250], [10, 230]], float)
    _, h = T.perspective_4pts(np.zeros((256, 256), np.uint8), src, dst)
    assert np.abs(T.apply_homography_to_points(h, src) - dst).max() < POINT_TOL


def test_order_corners_any_permutation() -> None:
    """Sắp xếp góc TL, TR, BR, BL không phụ thuộc thứ tự vào."""
    corners = np.array([[10, 10], [100, 15], [95, 120], [5, 110]], float)
    rng = np.random.default_rng(0)
    for _ in range(5):
        np.testing.assert_array_equal(doc.order_corners(rng.permutation(corners)), corners)


def test_document_rectification() -> None:
    """Tự tìm góc trên ảnh tổng hợp sai số < 2 px và ảnh nắn chỉnh gần trang gốc."""
    sd = images.make_synthetic_document()
    corners = doc.detect_corners_auto(sd.photo)
    assert corners is not None
    assert doc.corner_error(corners, sd.corners) < CORNER_TOL_PX
    out, _ = doc.rectify(sd.photo, corners, dsize=T.image_size(sd.flat))
    assert psnr(out, sd.flat) > RECTIFY_MIN_PSNR_DB
