"""Các thí nghiệm khảo sát tham số OpenCV, sinh Hinh1–Hinh6 và Bang1–Bang2.

Mỗi hàm ``make_*`` ghi file kết quả và trả về dict số liệu để viết report_numbers.md.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from compare.benchmark import benchmark
from compare.metrics import abs_diff, format_psnr, psnr
from src import app_document as doc
from src import images
from src import transforms as T
from src.config import BENCH_REPEAT, BENCH_WARMUP, FIG_DIR, SIZE_LARGE, SIZE_SMALL, TITLE_FONTSIZE
from src.plotting import (FIG_W_2X2, FIG_W_FULL, PANEL_TITLE_SIZE, draw_points, panel_grid, save_figure,
                          save_panels, setup_style, show_image)
from src.tables import write_table

# ---------------------------------------------------------------- Hình 2: co giãn
UP_FACTOR: int = 4
DOWN_FACTOR: float = 1 / UP_FACTOR
ZOOM_BOX: tuple[int, int, int, int] = (200, 100, 128, 128)   # (x, y, rộng, cao) trên ảnh 512x512
UPSCALE_FLAGS: tuple[int, ...] = (cv2.INTER_NEAREST, cv2.INTER_LINEAR, cv2.INTER_CUBIC,
                                  cv2.INTER_LANCZOS4, cv2.INTER_AREA)
DOWNSCALE_FLAGS: tuple[int, ...] = (cv2.INTER_NEAREST, cv2.INTER_LINEAR, cv2.INTER_CUBIC,
                                    cv2.INTER_LANCZOS4, cv2.INTER_AREA)

# ---------------------------------------------------------------- Hình 3: tịnh tiến, xoay
SHIFT: tuple[int, int] = (100, 60)
ANGLES: tuple[float, ...] = (30.0, 45.0, 90.0)
BORDER_DEMO_ANGLE: float = 30.0
BORDER_DEMO_SCALE: float = 0.6
BORDER_FLAGS: tuple[int, ...] = (cv2.BORDER_CONSTANT, cv2.BORDER_REPLICATE, cv2.BORDER_REFLECT,
                                 cv2.BORDER_WRAP, cv2.BORDER_REFLECT_101)

# ---------------------------------------------------------------- Hình 4, 5: affine, phối cảnh
AFFINE_SRC: np.ndarray = np.array([[60, 60], [450, 70], [70, 450]], dtype=np.float64)
AFFINE_DST: np.ndarray = np.array([[110, 150], [440, 40], [170, 470]], dtype=np.float64)
PERSP_SRC: np.ndarray = np.array([[40, 40], [470, 40], [470, 470], [40, 470]], dtype=np.float64)
PERSP_DST: np.ndarray = np.array([[150, 70], [380, 30], [490, 490], [30, 420]], dtype=np.float64)

# ---------------------------------------------------------------- Bảng 2
BANG2_FLAGS: tuple[tuple[str, int, int], ...] = (
    ("INTER_NEAREST", cv2.INTER_NEAREST, cv2.INTER_NEAREST),
    ("INTER_LINEAR", cv2.INTER_LINEAR, cv2.INTER_LINEAR),
    ("INTER_CUBIC", cv2.INTER_CUBIC, cv2.INTER_CUBIC),
    ("INTER_AREA", cv2.INTER_AREA, cv2.INTER_AREA),
    ("INTER_LANCZOS4", cv2.INTER_LANCZOS4, cv2.INTER_LANCZOS4),
    ("INTER_AREA ↓ + INTER_CUBIC ↑", cv2.INTER_AREA, cv2.INTER_CUBIC),
)

CELL_IN: float = 2.9   # cạnh mỗi ô hình (inch), dùng cho Hình 1

COFFEE_PANEL_ASPECT: float = 0.72   # ảnh coffee 600×400 (cao/rộng 0,67) cộng lề
DOC_PANEL_ASPECT: float = 0.95
DOC_DIFF_VMAX: int = 64        # thang ảnh hiệu của ứng dụng tài liệu (mức xám)
MAX_REAL_DOCS: int = 3


def _flag(f: int) -> str:
    """Tên cờ nội suy ngắn gọn để đặt tiêu đề."""
    return T.INTERP_NAME[f]


# ============================================================== Hình 1
def _box(ax: plt.Axes, cx: float, cy: float, w: float, h: float, text: str,
         fc: str = "#e8f0fe", ec: str = "#1a56db", ls: str = "-") -> None:
    """Vẽ một khối chữ nhật bo góc có chữ ở giữa."""
    ax.add_patch(FancyBboxPatch((cx - w / 2, cy - h / 2), w, h, boxstyle="round,pad=0.08",
                                fc=fc, ec=ec, lw=1.6, ls=ls))
    ax.text(cx, cy, text, ha="center", va="center", fontsize=TITLE_FONTSIZE, linespacing=1.5)


def _arrow(ax: plt.Axes, p0: tuple[float, float], p1: tuple[float, float], dashed: bool = False,
           rad: float = 0.0, color: str = "#333333") -> None:
    """Vẽ mũi tên từ p0 đến p1 (nét đứt nếu dashed, cong nếu rad ≠ 0)."""
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle="-|>", mutation_scale=16, lw=1.6, color=color,
                                 ls="--" if dashed else "-", connectionstyle=f"arc3,rad={rad}"))


def make_hinh1(out_dir: Path = FIG_DIR) -> dict[str, Any]:
    """Hình 1: sơ đồ khối quy trình biến đổi hình học với OpenCV và nhánh đối chứng."""
    fig, ax = plt.subplots(figsize=(16, 7.2))
    ax.set_xlim(0, 17)
    ax.set_ylim(0, 8)
    ax.axis("off")
    y_main, h_main = 5.0, 2.1
    _box(ax, 1.5, y_main, 2.4, h_main, "Đọc ảnh\ncv2.imread\n(uint8, H×W×C)")
    _box(ax, 5.6, y_main, 4.4, h_main,
         "Tính ma trận biến đổi\n• getRotationMatrix2D → M 2×3\n• getAffineTransform (3 điểm) → M 2×3"
         "\n• getPerspectiveTransform (4 điểm) → H 3×3")
    _box(ax, 10.85, y_main, 4.5, h_main,
         "Biến đổi ảnh (ánh xạ ngược + nội suy)\n• warpAffine(src, M, dsize)"
         "\n• warpPerspective(src, H, dsize)\n• resize(src, dsize, fx, fy)")
    _box(ax, 15.5, y_main, 2.2, h_main, "Ảnh ra\ncv2.imwrite\n/ hiển thị", fc="#e6f4ea", ec="#1e8e3e")
    _box(ax, 10.85, 2.5, 4.5, 1.35,
         "Tham số: interpolation (NEAREST, LINEAR,\nCUBIC, AREA, LANCZOS4), borderMode,\nborderValue, dsize",
         fc="#fef7e0", ec="#b06000")
    _arrow(ax, (2.75, y_main), (3.35, y_main))
    _arrow(ax, (7.85, y_main), (8.55, y_main))
    _arrow(ax, (13.15, y_main), (14.35, y_main))
    _arrow(ax, (10.85, 3.2), (10.85, y_main - h_main / 2 - 0.1))
    _arrow(ax, (1.5, y_main + h_main / 2 + 0.1), (10.2, y_main + h_main / 2 + 0.1), rad=-0.18)
    ax.text(5.6, 7.55, "resize: không cần ma trận, chỉ cần dsize hoặc fx, fy",
            ha="center", fontsize=TITLE_FONTSIZE, style="italic")
    ref_color, ref_ec = "#f3e8fd", "#7627bb"
    _box(ax, 5.6, 0.85, 4.4, 1.25, "Cài đặt tham chiếu (NumPy, không dùng cv2)\n"
         "ma trận tự tính + ánh xạ ngược + nội suy", fc=ref_color, ec=ref_ec, ls="--")
    _box(ax, 15.0, 0.85, 3.2, 1.25, "So sánh: PSNR, MSE,\n% pixel lệch, ảnh hiệu, thời gian",
         fc=ref_color, ec=ref_ec, ls="--")
    _arrow(ax, (5.6, y_main - h_main / 2 - 0.1), (5.6, 1.6), dashed=True, color=ref_ec)
    _arrow(ax, (7.85, 0.85), (13.3, 0.85), dashed=True, color=ref_ec)
    _arrow(ax, (15.5, y_main - h_main / 2 - 0.1), (15.5, 1.6), dashed=True, color=ref_ec)
    ax.text(10.4, 1.05, "nhánh phụ: đối chứng", ha="center", fontsize=TITLE_FONTSIZE - 1, color=ref_ec,
            style="italic")
    path = save_figure(fig, out_dir / "Hinh1.png")
    return {"paths": [path]}


# ============================================================== Hình 2
def _zoom(img: np.ndarray, box: tuple[int, int, int, int] = ZOOM_BOX) -> np.ndarray:
    """Cắt vùng phóng lớn (x, y, rộng, cao)."""
    x, y, w, h = box
    return img[y:y + h, x:x + w]


def _alias_std(img: np.ndarray, factor: float) -> float:
    """Độ lệch chuẩn mức xám trong vùng zone plate có tần số vượt Nyquist sau khi thu nhỏ.

    Lý tưởng (lọc thông thấp hoàn hảo) vùng này phải xám đều (std ≈ 0); std càng lớn
    nghĩa là càng nhiều vân moiré do chồng phổ.
    """
    h, w = img.shape
    yy, xx = np.mgrid[0:h, 0:w]
    r = np.hypot(xx - (w - 1) / 2, yy - (h - 1) / 2)
    # Tần số cục bộ tỉ lệ với bán kính, bằng ZONEPLATE_KMAX ở góc ảnh; Nyquist mới = factor·π
    # (tính theo pixel ảnh gốc) ⇒ chồng phổ khi r > factor·π/KMAX · r_góc.
    r_corner = np.hypot(w - 1, h - 1) / 2
    r_nyq = factor * np.pi / images.ZONEPLATE_KMAX * r_corner
    return float(img[r > r_nyq].std())


def make_hinh2(out_dir: Path = FIG_DIR) -> dict[str, Any]:
    """Hình 2a–2c: phóng to ×4 (vùng cắt phóng lớn), thu nhỏ ×0.25 trên zone plate (moiré) và ảnh chữ."""
    cam = images.load_camera()
    small = T.resize_to(cam, (cam.shape[1] // UP_FACTOR, cam.shape[0] // UP_FACTOR), cv2.INTER_AREA)
    ups = {f: T.scale(small, UP_FACTOR, interp=f) for f in UPSCALE_FLAGS}
    up_psnr = {_flag(f): psnr(u, cam) for f, u in ups.items()}

    zp, txt = images.make_zoneplate(), images.make_text_image()
    downs_zp = {f: T.scale(zp, DOWN_FACTOR, interp=f) for f in DOWNSCALE_FLAGS}
    downs_txt = {f: T.scale(txt, DOWN_FACTOR, interp=f) for f in DOWNSCALE_FLAGS}
    alias = {_flag(f): _alias_std(d, DOWN_FACTOR) for f, d in downs_zp.items()}

    paths = []
    with plt.rc_context({"axes.titlesize": PANEL_TITLE_SIZE}):
        fig, axes = panel_grid(2, 3, FIG_W_FULL, 1.0)
        show_image(axes.flat[0], _zoom(cam), "Ảnh gốc 512×512 (tham chiếu)")
        for ax, (f, u) in zip(axes.flat[1:], ups.items()):
            show_image(ax, _zoom(u), f"{_flag(f)}, PSNR = {format_psnr(up_psnr[_flag(f)])} dB")
        paths.append(save_panels(fig, out_dir, "Hinh2a"))

        fig, axes = panel_grid(2, 3, FIG_W_FULL, 1.0)
        show_image(axes.flat[0], zp, f"Zone plate gốc {zp.shape[1]}×{zp.shape[0]}")
        for ax, f in zip(axes.flat[1:], DOWNSCALE_FLAGS):
            show_image(ax, downs_zp[f], f"{_flag(f)}\nđộ lệch chuẩn > Nyquist = {alias[_flag(f)]:.1f}")
        paths.append(save_panels(fig, out_dir, "Hinh2b"))

        fig, axes = panel_grid(2, 3, FIG_W_FULL, 1.0)
        show_image(axes.flat[0], txt, f"Ảnh chữ gốc {txt.shape[1]}×{txt.shape[0]}")
        for ax, f in zip(axes.flat[1:], DOWNSCALE_FLAGS):
            show_image(ax, downs_txt[f], f"{_flag(f)}, ×{DOWN_FACTOR:g}")
        paths.append(save_panels(fig, out_dir, "Hinh2c"))
    return {"paths": paths, "up_psnr": up_psnr, "alias_std": alias,
            "small_size": small.shape[::-1], "zoom_box": ZOOM_BOX}


# ============================================================== Hình 3
def _kept_ratio(size: tuple[int, int], m: np.ndarray, out_size: tuple[int, int]) -> float:
    """Tỉ lệ (%) pixel ảnh gốc còn nằm trong khung ra sau phép affine m."""
    w, h = size
    ones = np.ones((h, w), np.uint8)
    warped = cv2.warpAffine(ones, m, out_size, flags=cv2.INTER_NEAREST)
    scale = abs(np.linalg.det(m[:, :2]))
    return float(warped.sum() / (w * h * scale) * 100)


def make_hinh3(out_dir: Path = FIG_DIR) -> dict[str, Any]:
    """Hình 3a–3c: xoay giữ / mở rộng khung; tịnh tiến và lỗi quên bù tịnh tiến; các borderMode."""
    img = images.load_coffee()
    size = T.image_size(img)
    kept_keep, sizes_expand = {}, {}
    paths = []
    with plt.rc_context({"axes.titlesize": PANEL_TITLE_SIZE}):
        fig, axes = panel_grid(2, 3, FIG_W_FULL, COFFEE_PANEL_ASPECT)
        for i, a in enumerate(ANGLES):
            m, out_size = T.rotation_matrix(size, a, expand=False)
            kept_keep[a] = _kept_ratio(size, m, out_size)
            show_image(axes[0, i], T.rotate(img, a),
                       f"Xoay {a:g}°, giữ khung {out_size[0]}×{out_size[1]}\ncòn {kept_keep[a]:.1f}% ảnh gốc")
            m, out_size = T.rotation_matrix(size, a, expand=True)
            sizes_expand[a] = out_size
            show_image(axes[1, i], T.rotate(img, a, expand=True),
                       f"Xoay {a:g}°, mở rộng khung {out_size[0]}×{out_size[1]}\n"
                       f"còn {_kept_ratio(size, m, out_size):.1f}% ảnh gốc")
        paths.append(save_panels(fig, out_dir, "Hinh3a"))

        fig, axes = panel_grid(2, 2, FIG_W_2X2, COFFEE_PANEL_ASPECT)
        kept_shift = _kept_ratio(size, T.translation_matrix(*SHIFT), size)
        show_image(axes[0, 0], img, f"Ảnh gốc {size[0]}×{size[1]}")
        show_image(axes[0, 1], T.translate(img, *SHIFT),
                   f"Tịnh tiến tx = {SHIFT[0]}, ty = {SHIFT[1]}\ncòn {kept_shift:.1f}% ảnh gốc")
        # Lỗi thường gặp: tính khung mới nhưng quên cộng bù tịnh tiến.
        a = ANGLES[1]
        m_wrong, _ = T.rotation_matrix(size, a, expand=False)
        out_size = T.expanded_size(size, a)
        show_image(axes[1, 0], cv2.warpAffine(img, m_wrong, out_size),
                   f"Xoay {a:g}°, khung {out_size[0]}×{out_size[1]}\nKHÔNG cộng bù tịnh tiến")
        show_image(axes[1, 1], T.rotate(img, a, expand=True),
                   f"Xoay {a:g}°, khung {out_size[0]}×{out_size[1]}\ncó cộng bù tịnh tiến")
        paths.append(save_panels(fig, out_dir, "Hinh3b"))

        fig, axes = panel_grid(2, 3, FIG_W_FULL, COFFEE_PANEL_ASPECT)
        show_image(axes.flat[0], img, f"Ảnh gốc {size[0]}×{size[1]}")
        for ax, b in zip(axes.flat[1:], BORDER_FLAGS):
            show_image(ax, T.rotate(img, BORDER_DEMO_ANGLE, BORDER_DEMO_SCALE, border=b),
                       f"{T.BORDER_NAME[b]}\nxoay {BORDER_DEMO_ANGLE:g}°, scale = {BORDER_DEMO_SCALE}")
        paths.append(save_panels(fig, out_dir, "Hinh3c"))
    return {"paths": paths, "size": size, "kept_shift": kept_shift, "kept_keep": kept_keep,
            "sizes_expand": sizes_expand}


# ============================================================== Hình 4, 5
def _four_panels(img: np.ndarray, out: np.ndarray, board: np.ndarray, board_out: np.ndarray,
                 src: np.ndarray, dst: np.ndarray, titles: tuple[str, str, str, str], name: str,
                 out_dir: Path) -> Path:
    """Lưới 2×2: ảnh gốc / ảnh sau biến đổi (hàng 1), bàn cờ gốc / bàn cờ sau biến đổi (hàng 2)."""
    with plt.rc_context({"axes.titlesize": PANEL_TITLE_SIZE}):
        fig, axes = panel_grid(2, 2, FIG_W_2X2, 1.0)
        for ax, im, pts, prefix, title in zip(axes.flat, (img, out, board, board_out), (src, dst, src, dst),
                                              "PQPQ", titles):
            show_image(ax, im, title)
            draw_points(ax, pts, prefix, fontsize=PANEL_TITLE_SIZE)
        return save_panels(fig, out_dir, name)


def make_hinh4(out_dir: Path = FIG_DIR) -> dict[str, Any]:
    """Hình 4: affine xác định bởi 3 cặp điểm (ảnh màu và bàn cờ), lưới 2×2."""
    img, board = images.load_astronaut(), images.make_checkerboard()
    out, m = T.affine_3pts(img, AFFINE_SRC, AFFINE_DST)
    board_out, _ = T.affine_3pts(board, AFFINE_SRC, AFFINE_DST)
    err = float(np.abs(T.apply_affine_to_points(m, AFFINE_SRC) - AFFINE_DST).max())
    path = _four_panels(img, out, board, board_out, AFFINE_SRC, AFFINE_DST,
                        ("Ảnh gốc, 3 điểm nguồn P1–P3", "warpAffine, INTER_LINEAR, điểm đích Q1–Q3",
                         "Bàn cờ gốc", "Bàn cờ sau affine: song song vẫn song song"), "Hinh4", out_dir)
    return {"paths": [path], "matrix": m, "point_err": err, "det": float(np.linalg.det(m[:, :2]))}


def make_hinh5(out_dir: Path = FIG_DIR) -> dict[str, Any]:
    """Hình 5: phối cảnh xác định bởi 4 cặp điểm (ảnh màu và bàn cờ), lưới 2×2."""
    img, board = images.load_astronaut(), images.make_checkerboard()
    out, h = T.perspective_4pts(img, PERSP_SRC, PERSP_DST)
    board_out, _ = T.perspective_4pts(board, PERSP_SRC, PERSP_DST)
    err = float(np.abs(T.apply_homography_to_points(h, PERSP_SRC) - PERSP_DST).max())
    top_len = np.linalg.norm(PERSP_DST[1] - PERSP_DST[0])
    bottom_len = np.linalg.norm(PERSP_DST[2] - PERSP_DST[3])
    path = _four_panels(img, out, board, board_out, PERSP_SRC, PERSP_DST,
                        ("Ảnh gốc, 4 điểm nguồn P1–P4", "warpPerspective, INTER_LINEAR, điểm đích Q1–Q4",
                         "Bàn cờ gốc", "Bàn cờ sau phối cảnh: song song không còn"), "Hinh5", out_dir)
    return {"paths": [path], "matrix": h, "point_err": err, "edge_ratio": float(bottom_len / top_len)}


# ============================================================== Hình 6
def _real_document_files() -> list[Path]:
    """Ảnh tài liệu thật do nhóm tự thêm: data/doc_real*.jpg|png (nếu có)."""
    return sorted(p for ext in ("jpg", "jpeg", "png") for p in images.DATA_DIR.glob(f"doc_real*.{ext}"))


def _make_hinh6_real(real: list[Path], out_dir: Path) -> tuple[Path | None, list[dict[str, Any]]]:
    """Hình 6b (chỉ khi có ảnh thật): mỗi hàng là ảnh chụp và ảnh sau nắn chỉnh, tối đa 3 hàng."""
    if not real:
        return None, []
    real = real[:MAX_REAL_DOCS]
    info = []
    with plt.rc_context({"axes.titlesize": PANEL_TITLE_SIZE}):
        fig, axes = panel_grid(len(real), 2, FIG_W_2X2, DOC_PANEL_ASPECT)
        for row, path in zip(axes, real):
            photo = images.read_rgb(path)
            c = doc.detect_corners_auto(photo)
            show_image(row[0], photo, f"Ảnh thật: {path.name}", smooth=True)
            if c is None:
                row[1].axis("off")
                row[1].text(0.5, 0.5, "Không tự tìm được 4 góc\n(dùng demo_document.py để click)", ha="center")
            else:
                draw_points(row[0], c, "C", fontsize=PANEL_TITLE_SIZE)
                out, _ = doc.rectify(photo, c)
                show_image(row[1], out, f"Sau nắn chỉnh {out.shape[1]}×{out.shape[0]}", smooth=True)
            info.append({"file": path.name, "found": c is not None})
        return save_panels(fig, out_dir, "Hinh6b"), info


def make_hinh6(out_dir: Path = FIG_DIR) -> dict[str, Any]:
    """Hình 6: nắn chỉnh ảnh tài liệu chụp nghiêng, lưới 2×2 (ảnh chụp, sau nắn, tham chiếu, ảnh hiệu)."""
    sd = images.make_synthetic_document()
    flat_size = T.image_size(sd.flat)
    corners = doc.detect_corners_auto(sd.photo)
    if corners is None:
        raise RuntimeError("Không tìm được 4 góc trên ảnh tài liệu tổng hợp")
    rect_auto, _ = doc.rectify(sd.photo, corners, dsize=flat_size)
    rect_true, _ = doc.rectify(sd.photo, sd.corners, dsize=flat_size)
    t = benchmark(doc.rectify, sd.photo, corners, flat_size)
    t_detect = benchmark(doc.detect_corners_auto, sd.photo)
    res = {"corner_err": doc.corner_error(corners, sd.corners), "psnr_auto": psnr(rect_auto, sd.flat),
           "psnr_true": psnr(rect_true, sd.flat), "t_rectify_ms": t.mean_ms,
           "t_detect_ms": t_detect.mean_ms, "photo_size": T.image_size(sd.photo), "out_size": flat_size}

    with plt.rc_context({"axes.titlesize": PANEL_TITLE_SIZE}):
        fig, axes = panel_grid(2, 2, FIG_W_2X2, DOC_PANEL_ASPECT)
        show_image(axes[0, 0], sd.photo, f"Ảnh chụp nghiêng {res['photo_size'][0]}×{res['photo_size'][1]}\n"
                   f"4 góc tự tìm, sai số TB {res['corner_err']:.2f} px", smooth=True)
        draw_points(axes[0, 0], corners, "C", fontsize=PANEL_TITLE_SIZE)
        show_image(axes[0, 1], rect_auto, f"Sau nắn chỉnh {flat_size[0]}×{flat_size[1]}\n"
                   "warpPerspective, INTER_CUBIC", smooth=True)
        show_image(axes[1, 0], sd.flat, "Trang tài liệu gốc (tham chiếu)", smooth=True)
        diff = abs_diff(rect_auto, sd.flat)
        axes[1, 1].imshow(diff, cmap="magma", vmin=0, vmax=DOC_DIFF_VMAX)
        axes[1, 1].set_xticks([])
        axes[1, 1].set_yticks([])
        axes[1, 1].set_title(f"|Sau nắn − tham chiếu| (thang 0–{DOC_DIFF_VMAX})\n"
                             f"PSNR = {res['psnr_auto']:.2f} dB")
        res_paths = [save_panels(fig, out_dir, "Hinh6")]
    real_path, res["real"] = _make_hinh6_real(_real_document_files(), out_dir)
    if real_path is not None:
        res_paths.append(real_path)
    res["paths"] = res_paths
    return res


# ============================================================== Bảng 1
BANG1_HEADER: list[str] = ["Hàm OpenCV", "Chức năng", "Tham số chính", "Ý nghĩa / lưu ý"]
BANG1_ROWS: list[list[str]] = [
    ["cv2.resize", "Co giãn ảnh", "src, dsize=(w, h), fx, fy, interpolation",
     "dsize ưu tiên hơn fx, fy; nên dùng INTER_AREA khi thu nhỏ, INTER_CUBIC/LANCZOS4 khi phóng to"],
    ["cv2.getRotationMatrix2D", "Tạo ma trận xoay 2×3", "center=(x, y), angle (độ), scale",
     "Góc dương = ngược chiều kim đồng hồ; không tự mở rộng khung, phải tự cộng bù tịnh tiến"],
    ["cv2.getAffineTransform", "Ma trận affine 2×3 từ 3 cặp điểm", "src, dst: mảng float32 (3, 2)",
     "Giải hệ 6 phương trình; 3 điểm không được thẳng hàng"],
    ["cv2.warpAffine", "Áp phép affine lên ảnh", "src, M (2×3), dsize, flags, borderMode, borderValue",
     "Ánh xạ ngược dst(x, y) = src(M⁻¹[x, y, 1]ᵀ); flags = cờ nội suy | WARP_INVERSE_MAP"],
    ["cv2.getPerspectiveTransform", "Homography 3×3 từ 4 cặp điểm", "src, dst: mảng float32 (4, 2)",
     "Giải hệ 8 phương trình với h33 = 1; không có 3 điểm nào thẳng hàng"],
    ["cv2.warpPerspective", "Áp phép phối cảnh lên ảnh", "src, H (3×3), dsize, flags, borderMode, borderValue",
     "Tọa độ nguồn chia cho thành phần thứ ba w; đường thẳng giữ thẳng, song song không còn"],
    ["cv2.transform / cv2.perspectiveTransform", "Biến đổi tọa độ các điểm", "pts (N, 1, 2), M hoặc H",
     "Kiểm tra điểm đích, vẽ điểm đánh dấu; không nội suy ảnh"],
    ["Cờ interpolation", "Chọn cách nội suy", "INTER_NEAREST, INTER_LINEAR, INTER_CUBIC, INTER_AREA, INTER_LANCZOS4",
     "Lân cận 1, 2×2, 4×4 (a = −0.75), lấy trung bình vùng, 8×8 (Lanczos)"],
    ["Cờ borderMode", "Ngoại suy pixel ngoài ảnh", "BORDER_CONSTANT, REPLICATE, REFLECT, REFLECT_101, WRAP",
     "Hằng số borderValue | lặp pixel biên aaa|abc | phản xạ cba|abc | phản xạ bỏ biên cb|abc | cuộn vòng xyz|abc"],
    ["cv2.findContours + cv2.approxPolyDP", "Tìm 4 góc tài liệu (ứng dụng)", "ảnh nhị phân, epsilon = 2% chu vi",
     "Lấy contour lớn nhất xấp xỉ được bằng tứ giác"],
]


def make_bang1() -> dict[str, Any]:
    """Bảng 1: các hàm OpenCV sử dụng, chức năng, tham số chính, ý nghĩa."""
    paths = write_table("Bang1", "Bảng 1. Các hàm OpenCV dùng cho biến đổi hình học", BANG1_HEADER, BANG1_ROWS)
    return {"paths": paths}


# ============================================================== Bảng 2
def _roundtrip(img: np.ndarray, down: int, up: int) -> np.ndarray:
    """Thu nhỏ ×DOWN_FACTOR bằng cờ down rồi phóng về kích thước gốc bằng cờ up."""
    small = T.scale(img, DOWN_FACTOR, interp=down)
    return T.resize_to(small, T.image_size(img), up)


def make_bang2() -> dict[str, Any]:
    """Bảng 2: PSNR khi thu nhỏ ×0.25 rồi phóng về kích thước gốc, thời gian ở 2 kích thước."""
    test_imgs = {
        f"{SIZE_SMALL[0]}×{SIZE_SMALL[1]}": images.load_camera(),
        f"{SIZE_LARGE[0]}×{SIZE_LARGE[1]}": images.make_large(images.to_gray(images.load_coffee())),
    }
    results: list[dict[str, Any]] = []
    for name, down, up in BANG2_FLAGS:
        row: dict[str, Any] = {"flag": name}
        for size_name, img in test_imgs.items():
            small = T.scale(img, DOWN_FACTOR, interp=down)
            row[f"psnr_{size_name}"] = psnr(_roundtrip(img, down, up), img)
            row[f"t_down_{size_name}"] = benchmark(T.scale, img, DOWN_FACTOR, interp=down).mean_ms
            row[f"t_up_{size_name}"] = benchmark(T.resize_to, small, T.image_size(img), up).mean_ms
        results.append(row)
    sizes = list(test_imgs)
    header = ["Cờ nội suy"] + [f"PSNR {s} (dB)" for s in sizes]
    for s in sizes:
        header += [f"t thu nhỏ {s} (ms)", f"t phóng to {s} (ms)"]
    rows = []
    for r in results:
        cells = [r["flag"]] + [format_psnr(r[f"psnr_{s}"]) for s in sizes]
        for s in sizes:
            cells += [f"{r[f't_down_{s}']:.3f}", f"{r[f't_up_{s}']:.3f}"]
        rows.append(cells)
    notes = [
        f"Ảnh {sizes[0]}: 'camera' (skimage, xám). Ảnh {sizes[1]}: 'coffee' (skimage) chuyển xám rồi phóng "
        "bằng INTER_LANCZOS4 lên 1920×1080 (ảnh đã mịn hơn ảnh chụp thật nên PSNR cao hơn).",
        f"PSNR: thu nhỏ ×{DOWN_FACTOR:g} rồi phóng ×{UP_FACTOR} về kích thước gốc bằng cùng một cờ "
        "(dòng cuối: INTER_AREA khi thu nhỏ, INTER_CUBIC khi phóng to), so với ảnh gốc.",
        f"Thời gian: trung bình {BENCH_REPEAT} lần gọi cv2.resize, bỏ {BENCH_WARMUP} lần warm-up; "
        f"OpenCV dùng {cv2.getNumThreads()} luồng.",
    ]
    paths = write_table("Bang2", f"Bảng 2. Khảo sát cờ nội suy của cv2.resize (thu nhỏ ×{DOWN_FACTOR:g} "
                        f"rồi phóng ×{UP_FACTOR})", header, rows, notes)
    return {"paths": paths, "rows": results, "sizes": sizes}


def run_all_experiments() -> dict[str, Any]:
    """Chạy mọi thí nghiệm phần chính (OpenCV). Trả về dict số liệu theo tên hình/bảng."""
    setup_style()
    return {
        "Hinh1": make_hinh1(), "Hinh2": make_hinh2(), "Hinh3": make_hinh3(), "Hinh4": make_hinh4(),
        "Hinh5": make_hinh5(), "Hinh6": make_hinh6(), "Bang1": make_bang1(), "Bang2": make_bang2(),
    }


if __name__ == "__main__":
    run_all_experiments()
