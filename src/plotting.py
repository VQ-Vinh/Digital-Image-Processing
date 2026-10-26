"""Tiện ích vẽ hình: nền trắng, DPI 200, nhãn tiếng Việt có dấu, tiêu đề từng ô."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from src.config import FIG_DPI, FIG_FONT, TITLE_FONTSIZE

MARKER_COLORS: tuple[str, ...] = ("#e41a1c", "#377eb8", "#4daf4a", "#ff7f00")


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


def draw_points(ax: plt.Axes, pts: np.ndarray, prefix: str, marker: str = "o") -> None:
    """Đánh dấu các điểm (N, 2) kèm nhãn prefix1, prefix2, ..."""
    for i, (x, y) in enumerate(pts):
        color = MARKER_COLORS[i % len(MARKER_COLORS)]
        ax.plot(x, y, marker, ms=9, mfc=color, mec="white", mew=1.5)
        ax.annotate(f"{prefix}{i + 1}", (x, y), xytext=(6, -8), textcoords="offset points",
                    color="white", fontsize=TITLE_FONTSIZE, fontweight="bold",
                    bbox={"boxstyle": "round,pad=0.15", "fc": color, "ec": "none"})


def draw_box(ax: plt.Axes, x0: int, y0: int, w: int, h: int, color: str = "red") -> None:
    """Vẽ khung chữ nhật (đánh dấu vùng cắt phóng lớn)."""
    ax.add_patch(plt.Rectangle((x0 - 0.5, y0 - 0.5), w, h, fill=False, ec=color, lw=1.5))


def save_figure(fig: plt.Figure, path: Path) -> Path:
    """Lưu hình ra PNG (DPI 200, nền trắng) rồi đóng hình."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=FIG_DPI, facecolor="white", bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)
    return path
