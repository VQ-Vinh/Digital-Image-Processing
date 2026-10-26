"""Tạo và nạp ảnh thử: ảnh có sẵn của skimage.data và ảnh tổng hợp (không tải từ Internet)."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from skimage import data as skdata

from src.config import DATA_DIR, SEED, SIZE_LARGE, SIZE_SMALL

CHECKER_SQUARE: int = 32           # cạnh ô bàn cờ (px)
ZONEPLATE_KMAX: float = np.pi      # tần số cục bộ lớn nhất ở góc ảnh (rad/px), = Nyquist
TEXT_FONT: int = cv2.FONT_HERSHEY_SIMPLEX
TEXT_LINES: tuple[str, ...] = ("EE3035", "Geometric", "Transform", "OpenCV 4.x", "x' = Mx")

# Ảnh tài liệu tổng hợp (tỉ lệ A4 ≈ 1 : 1.414).
DOC_SIZE: tuple[int, int] = (600, 848)
DOC_CANVAS: tuple[int, int] = (1280, 960)
# Góc TL, TR, BR, BL: mép trên xa máy ảnh nên ngắn hơn mép dưới, trang hơi xoay.
DOC_CORNERS: np.ndarray = np.array([[420, 90], [900, 140], [980, 880], [400, 850]], dtype=np.float64)
DOC_NOISE_SIGMA: float = 3.0
DOC_JPEG_QUALITY: int = 92
DOC_WORDS: tuple[str, ...] = ("anh", "so", "bien", "doi", "hinh", "hoc", "ma", "tran", "noi",
                              "suy", "pixel", "affine", "phoi", "canh", "xoay", "tinh", "tien")


def to_gray(img: np.ndarray) -> np.ndarray:
    """Chuyển ảnh RGB sang ảnh xám uint8 (giữ nguyên nếu đã là ảnh xám)."""
    return img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)


def load_camera() -> np.ndarray:
    """Ảnh 'camera' (xám, 512x512) của skimage."""
    return skdata.camera()


def load_astronaut() -> np.ndarray:
    """Ảnh 'astronaut' (RGB, 512x512) của skimage."""
    return skdata.astronaut()


def load_coffee() -> np.ndarray:
    """Ảnh 'coffee' (RGB, 600x400) của skimage."""
    return skdata.coffee()


def make_large(img: np.ndarray, size: tuple[int, int] = SIZE_LARGE) -> np.ndarray:
    """Phóng ảnh về kích thước lớn (mặc định 1920x1080) bằng INTER_LANCZOS4."""
    return cv2.resize(img, size, interpolation=cv2.INTER_LANCZOS4)


def make_checkerboard(size: tuple[int, int] = SIZE_SMALL, square: int = CHECKER_SQUARE) -> np.ndarray:
    """Ảnh bàn cờ đen trắng, kích thước (rộng, cao)."""
    w, h = size
    yy, xx = np.mgrid[0:h, 0:w]
    board = ((xx // square + yy // square) % 2).astype(np.uint8)
    return board * np.uint8(255)


def make_zoneplate(size: tuple[int, int] = SIZE_SMALL, kmax: float = ZONEPLATE_KMAX) -> np.ndarray:
    """Ảnh 'zone plate' cos(k r^2): tần số tăng dần ra mép, dễ thấy moiré khi thu nhỏ."""
    w, h = size
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float64)
    r2 = (xx - (w - 1) / 2) ** 2 + (yy - (h - 1) / 2) ** 2
    r_max = np.hypot(w - 1, h - 1) / 2
    k = kmax / (2 * r_max)  # tần số cục bộ d(k r^2)/dr = 2kr = kmax tại góc ảnh
    return np.uint8(np.round(127.5 * (1 + np.cos(k * r2))))


def make_text_image(size: tuple[int, int] = SIZE_SMALL) -> np.ndarray:
    """Ảnh chữ đen trên nền trắng, vẽ không khử răng cưa (LINE_8) để dễ thấy răng cưa."""
    w, h = size
    img = np.full((h, w), 255, np.uint8)
    line_h = h // (len(TEXT_LINES) + 1)
    for i, text in enumerate(TEXT_LINES):
        font_scale = line_h / 40
        cv2.putText(img, text, (w // 16, line_h * (i + 1)), TEXT_FONT, font_scale, 0,
                    thickness=max(1, line_h // 30), lineType=cv2.LINE_8)
    return img


def _make_flat_document(size: tuple[int, int], rng: np.random.Generator) -> np.ndarray:
    """Trang tài liệu phẳng (RGB): tiêu đề, các dòng chữ, một bảng kẻ ô."""
    w, h = size
    doc = np.full((h, w, 3), 250, np.uint8)
    margin = w // 12
    cv2.putText(doc, "BAO CAO BAI TAP LON", (margin, h // 12), TEXT_FONT, 1.0, (20, 20, 20), 2, cv2.LINE_AA)
    cv2.putText(doc, "EE3035 - Xu ly anh so", (margin, h // 12 + 40), TEXT_FONT, 0.7, (60, 60, 160), 2,
                cv2.LINE_AA)
    y = h // 12 + 90
    line_gap = 26
    for _ in range(12):
        words = rng.choice(DOC_WORDS, size=7)
        cv2.putText(doc, " ".join(words), (margin, y), TEXT_FONT, 0.6, (30, 30, 30), 1, cv2.LINE_AA)
        y += line_gap
    # Bảng 4 x 3 ô.
    top, rows, cols = y + 10, 4, 3
    cell_w, cell_h = (w - 2 * margin) // cols, 40
    for r in range(rows + 1):
        cv2.line(doc, (margin, top + r * cell_h), (margin + cols * cell_w, top + r * cell_h), (0, 0, 0), 2)
    for c in range(cols + 1):
        cv2.line(doc, (margin + c * cell_w, top), (margin + c * cell_w, top + rows * cell_h), (0, 0, 0), 2)
    for r in range(rows):
        for c in range(cols):
            cv2.putText(doc, f"{rng.integers(10, 99)}.{rng.integers(0, 9)}",
                        (margin + c * cell_w + 12, top + r * cell_h + 28), TEXT_FONT, 0.6, (0, 0, 0), 1,
                        cv2.LINE_AA)
    y = top + rows * cell_h + 40
    while y < h - margin:
        words = rng.choice(DOC_WORDS, size=7)
        cv2.putText(doc, " ".join(words), (margin, y), TEXT_FONT, 0.6, (30, 30, 30), 1, cv2.LINE_AA)
        y += line_gap
    return doc


def _make_background(size: tuple[int, int], rng: np.random.Generator) -> np.ndarray:
    """Nền mặt bàn màu nâu, có vân nhiễu tần số thấp và độ sáng giảm dần."""
    w, h = size
    noise = rng.normal(0, 1, (h // 16, w // 16)).astype(np.float32)
    noise = cv2.resize(noise, (w, h), interpolation=cv2.INTER_CUBIC)
    base = np.array([110, 75, 45], np.float32)  # nâu gỗ (RGB)
    bg = base[None, None, :] * (1 + 0.12 * noise[..., None])
    xx = np.linspace(1.0, 0.75, w, dtype=np.float32)
    return np.clip(bg * xx[None, :, None], 0, 255).astype(np.uint8)


@dataclass
class SyntheticDocument:
    """Ảnh tài liệu chụp nghiêng tổng hợp cùng dữ liệu đúng (ground truth)."""

    photo: np.ndarray        # ảnh "chụp nghiêng" (RGB)
    flat: np.ndarray         # trang tài liệu phẳng ban đầu (RGB)
    corners: np.ndarray      # 4 góc thật trong ảnh chụp: TL, TR, BR, BL


def make_synthetic_document(seed: int = SEED) -> SyntheticDocument:
    """Tạo ảnh 'tài liệu chụp nghiêng' bằng warpPerspective trang phẳng lên nền bàn."""
    rng = np.random.default_rng(seed)
    flat = _make_flat_document(DOC_SIZE, rng)
    w, h = DOC_SIZE
    src = np.float32([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]])
    hmat = cv2.getPerspectiveTransform(src, np.float32(DOC_CORNERS))
    photo = _make_background(DOC_CANVAS, rng)
    cv2.warpPerspective(flat, hmat, DOC_CANVAS, dst=photo, flags=cv2.INTER_LINEAR,
                        borderMode=cv2.BORDER_TRANSPARENT)
    shade = np.linspace(1.0, 0.85, DOC_CANVAS[1], dtype=np.float32)[:, None, None]
    noisy = photo.astype(np.float32) * shade + rng.normal(0, DOC_NOISE_SIGMA, photo.shape)
    photo = np.clip(np.round(noisy), 0, 255).astype(np.uint8)
    return SyntheticDocument(photo=photo, flat=flat, corners=DOC_CORNERS.copy())


def save_rgb(path: Path, img: np.ndarray, jpeg_quality: int = DOC_JPEG_QUALITY) -> None:
    """Ghi ảnh RGB/xám ra file (cv2.imwrite cần thứ tự BGR)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    out = img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(path), out, [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality])


def read_rgb(path: str | Path) -> np.ndarray:
    """Đọc ảnh từ file, trả về RGB (hoặc xám nếu file là ảnh xám)."""
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(f"Không đọc được ảnh: {path}")
    if img.ndim == 2:
        return img
    if img.shape[2] == 4:
        img = img[..., :3]
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def write_data_files(data_dir: Path = DATA_DIR) -> list[Path]:
    """Ghi toàn bộ ảnh thử vào data/ để demo dùng lại. Trả về danh sách file."""
    files = {
        "camera.png": load_camera(),
        "astronaut.png": load_astronaut(),
        "coffee.png": load_coffee(),
        "coffee_1920x1080.png": make_large(load_coffee()),
        "checkerboard.png": make_checkerboard(),
        "zoneplate.png": make_zoneplate(),
        "text.png": make_text_image(),
        "doc.jpg": make_synthetic_document().photo,
    }
    paths = []
    for name, img in files.items():
        path = data_dir / name
        save_rgb(path, img)
        paths.append(path)
    return paths
