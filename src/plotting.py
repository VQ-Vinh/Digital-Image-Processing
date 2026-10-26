"""Tiện ích vẽ hình: nền trắng, DPI 200, nhãn tiếng Việt có dấu, tiêu đề từng ô."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from src.config import FIG_DPI, FIG_FONT, TITLE_FONTSIZE

MARKER_COLORS: tuple[str, ...] = ("#e41a1c", "#377eb8", "#4daf4a", "#ff7f00")

# Hình ghép 4–6 ô được vẽ đúng kích thước khi in để chữ trong hình đọc được.
FIG_W_FULL: float = 6.3        # lưới 2×3 rộng bằng khổ chữ A4 (16 cm)
FIG_W_2X2: float = 5.0         # lưới 2×2 (12,7 cm)
TITLE_SPACE_IN: float = 0.42   # chỗ cho tiêu đề 2 dòng phía trên mỗi ô
PANEL_TITLE_SIZE: int = 8


def setup_style() -> None:
    """Đặt cấu hình matplotlib chung cho mọi hình."""
    plt.rcParams.update({
        "font.family": FIG_FONT,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
        "savefig.dpi": FIG_DPI,
        "axes.titlesize": TITLE_FONTSIZE,
        "figure.titlesize": TITLE_FONTSIZE + 3,
    })


def panel_grid(nrow: int, ncol: int, width: float, panel_aspect: float) -> tuple[plt.Figure, np.ndarray]:
    """Lưới nrow×ncol ô ảnh, rộng ``width`` inch; panel_aspect = cao/rộng của mỗi ô ảnh."""
    panel_w = width / ncol
    height = nrow * (panel_w * panel_aspect + TITLE_SPACE_IN)
    fig, axes = plt.subplots(nrow, ncol, figsize=(width, height), squeeze=False)
    return fig, axes


def save_panels(fig: plt.Figure, out_dir: Path, name: str) -> Path:
    """Căn lề gọn rồi lưu hình ghép ra ``<name>.png``."""
    fig.tight_layout(pad=0.3, h_pad=0.6, w_pad=0.4)
    return save_figure(fig, out_dir / f"{name}.png")


def show_image(ax: plt.Axes, img: np.ndarray, title: str, vmin: float = 0, vmax: float = 255,
               smooth: bool = False) -> None:
    """Vẽ ảnh lên một ô (ảnh xám dùng thang xám cố định 0–255), tắt trục, đặt tiêu đề.

    Mặc định hiển thị bằng 'nearest' để không làm nhòe thêm kết quả nội suy.
    """
    cmap = "gray" if img.ndim == 2 else None
    ax.imshow(img, cmap=cmap, vmin=vmin, vmax=vmax, interpolation="antialiased" if smooth else "nearest")
    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])


def draw_points(ax: plt.Axes, pts: np.ndarray, prefix: str, marker: str = "o",
                fontsize: int = TITLE_FONTSIZE) -> None:
    """Đánh dấu các điểm (N, 2) kèm nhãn prefix1, prefix2, ..."""
    for i, (x, y) in enumerate(pts):
        color = MARKER_COLORS[i % len(MARKER_COLORS)]
        ax.plot(x, y, marker, ms=fontsize - 1, mfc=color, mec="white", mew=1.2)
        ax.annotate(f"{prefix}{i + 1}", (x, y), xytext=(5, -7), textcoords="offset points",
                    color="white", fontsize=fontsize, fontweight="bold",
                    bbox={"boxstyle": "round,pad=0.15", "fc": color, "ec": "none"})


def draw_box(ax: plt.Axes, x0: int, y0: int, w: int, h: int, color: str = "red") -> None:
    """Vẽ khung chữ nhật (đánh dấu vùng cắt phóng lớn)."""
    ax.add_patch(plt.Rectangle((x0 - 0.5, y0 - 0.5), w, h, fill=False, ec=color, lw=1.5))


DOC_MAX_WIDTH_PX: int = 1800    # ≈ 290 dpi khi in rộng 15,9 cm
DOC_JPEG_QUALITY: int = 90


def make_doc_copies(fig_dir: Path, out_dir: Path) -> list[Path]:
    """Tạo bản JPEG nhẹ của các hình để chèn vào báo cáo Google Docs (giới hạn xuất 10 MB)."""
    from PIL import Image

    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for src in sorted(fig_dir.glob("Hinh*.png")):
        img = Image.open(src).convert("RGB")
        if img.width > DOC_MAX_WIDTH_PX:
            img = img.resize((DOC_MAX_WIDTH_PX, round(img.height * DOC_MAX_WIDTH_PX / img.width)), Image.LANCZOS)
        dst = out_dir / f"{src.stem}.jpg"
        img.save(dst, quality=DOC_JPEG_QUALITY, optimize=True)
        paths.append(dst)
    return paths


def save_figure(fig: plt.Figure, path: Path) -> Path:
    """Lưu hình ra PNG (DPI 200, nền trắng) rồi đóng hình."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=FIG_DPI, facecolor="white", bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)
    return path
