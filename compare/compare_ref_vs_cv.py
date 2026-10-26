"""So sánh cài đặt tham chiếu (reference/, NumPy) với OpenCV: sinh Hinh7 và Bang3."""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import matplotlib.pyplot as plt
import numpy as np

from compare.benchmark import benchmark
from compare.metrics import abs_diff, format_psnr, max_abs_diff, pct_diff_gt, psnr
from reference import matrices as RM
from reference import warp_ref as RW
from src import images
from src.config import BENCH_REPEAT, BENCH_WARMUP, FIG_DIR, SEED
from src.experiments import AFFINE_DST, AFFINE_SRC, PERSP_DST, PERSP_SRC
from src.plotting import FIG_W_2X2, FIG_W_FULL, PANEL_TITLE_SIZE, panel_grid, save_panels, setup_style, show_image
from src.tables import write_table

INTERP_CV: dict[str, int] = {"nearest": cv2.INTER_NEAREST, "linear": cv2.INTER_LINEAR,
                             "cubic": cv2.INTER_CUBIC}
INTERP_LABEL: dict[str, str] = {"nearest": "nearest", "linear": "bilinear", "cubic": "bicubic (a = −0.75)"}
OPENCV_INTER_BITS: int = 5          # OpenCV lượng tử hóa tọa độ về 1/2^5 = 1/32 pixel
RESIZE_FACTOR: int = 4
RESIZE_INPUT: tuple[int, int] = (256, 256)
ROTATE_ANGLE: float = 45.0
N_RANDOM_MATRICES: int = 200        # số ma trận ngẫu nhiên để đo sai số ma trận lớn nhất
RANDOM_COORD_MAX: float = 1000.0
QUAD_SIDE_MIN, QUAD_SIDE_MAX = 100.0, 600.0
QUAD_JITTER: float = 0.15
DIFF_VMAX: float = 3.0              # thang màu ảnh hiệu: 0 → 3 mức xám (phóng đại tương phản)
FIG7_INTERP: str = "linear"


@dataclass
class Case:
    """Một phép biến đổi cần so sánh: hàm tự cài đặt và hàm OpenCV cùng đầu vào."""

    name: str
    io_size: str
    ref: Callable[[str], np.ndarray]
    cv: Callable[[int], np.ndarray]
    matrix_err: float | None = None
    results: dict[str, dict[str, Any]] = field(default_factory=dict)


# ------------------------------------------------------------------ sai số ma trận
@dataclass
class MatrixError:
    """Sai số giữa ma trận tự tính và ma trận OpenCV trên N_RANDOM_MATRICES trường hợp ngẫu nhiên."""

    abs_max: float        # max |M_tự tính − M_OpenCV|
    rel_max: float        # max (|ΔM|max / |M_OpenCV|max)
    reproj_ref: float     # sai số chiếu lại điểm lớn nhất (px) của ma trận tự tính
    reproj_cv: float      # sai số chiếu lại điểm lớn nhất (px) của ma trận OpenCV


def _project(m: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """Chiếu các điểm (N, 2) qua ma trận 2x3 hoặc 3x3."""
    ph = RM.to_homogeneous(m) @ np.c_[pts, np.ones(len(pts))].T
    return (ph[:2] / ph[2]).T


def _collect(pairs: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]) -> MatrixError:
    """Tổng hợp sai số từ các bộ (M tự tính, M OpenCV, điểm nguồn, điểm đích)."""
    abs_e = [np.abs(a - b).max() for a, b, _, _ in pairs]
    rel_e = [np.abs(a - b).max() / np.abs(b).max() for a, b, _, _ in pairs]
    rp_ref = [np.abs(_project(a, s) - d).max() for a, _, s, d in pairs]
    rp_cv = [np.abs(_project(b, s) - d).max() for _, b, s, d in pairs]
    return MatrixError(float(max(abs_e)), float(max(rel_e)), float(max(rp_ref)), float(max(rp_cv)))


def _random_quad(rng: np.random.Generator) -> np.ndarray:
    """Tứ giác lồi ngẫu nhiên (float32): hình vuông cạnh s, mỗi đỉnh lệch ≤ QUAD_JITTER·s.

    Tránh cấu hình gần suy biến (3 điểm gần thẳng hàng), nơi mọi cách giải đều kém ổn định.
    """
    square = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=np.float64)
    side = rng.uniform(QUAD_SIDE_MIN, QUAD_SIDE_MAX)
    offset = rng.uniform(0, RANDOM_COORD_MAX - QUAD_SIDE_MAX * (1 + QUAD_JITTER), 2)
    jitter = rng.uniform(-QUAD_JITTER, QUAD_JITTER, (4, 2)) * side
    return (square * side + jitter + offset).astype(np.float32)


def rotation_matrix_error(rng: np.random.Generator) -> MatrixError:
    """Ma trận xoay tự tính so với cv2.getRotationMatrix2D.

    OpenCV nhận tâm kiểu Point2f (float32) nên tâm được làm tròn về float32 trước
    để hai bên dùng đúng cùng đầu vào. Điểm kiểm tra chiếu lại: 4 đỉnh tứ giác ngẫu nhiên.
    """
    pairs = []
    for _ in range(N_RANDOM_MATRICES):
        center = tuple(float(v) for v in rng.uniform(0, RANDOM_COORD_MAX, 2).astype(np.float32))
        angle, scale = rng.uniform(-360, 360), rng.uniform(0.1, 4)
        a = RM.rotation_matrix(center, angle, scale)
        b = cv2.getRotationMatrix2D(center, angle, scale)
        pts = np.float64(_random_quad(rng))
        pairs.append((a, b, pts, _project(a, pts)))
    return _collect(pairs)


def affine_matrix_error(rng: np.random.Generator) -> MatrixError:
    """Affine 3 điểm tự tính so với cv2.getAffineTransform."""
    pairs = []
    for _ in range(N_RANDOM_MATRICES):
        src, dst = _random_quad(rng)[:3], _random_quad(rng)[:3]
        pairs.append((RM.affine_from_3pts(src, dst), cv2.getAffineTransform(src, dst),
                      np.float64(src), np.float64(dst)))
    return _collect(pairs)


def homography_error(rng: np.random.Generator) -> MatrixError:
    """Homography DLT tự tính so với cv2.getPerspectiveTransform."""
    pairs = []
    for _ in range(N_RANDOM_MATRICES):
        src, dst = _random_quad(rng), _random_quad(rng)
        pairs.append((RM.homography_from_4pts(src, dst), cv2.getPerspectiveTransform(src, dst),
                      np.float64(src), np.float64(dst)))
    return _collect(pairs)


# ------------------------------------------------------------------ các phép so sánh
def build_cases() -> list[Case]:
    """Tạo 4 phép so sánh: resize ×4, xoay 45°, affine 3 điểm, phối cảnh 4 điểm (ảnh 'camera')."""
    cam = images.load_camera()
    size = (cam.shape[1], cam.shape[0])
    small = cv2.resize(cam, RESIZE_INPUT, interpolation=cv2.INTER_AREA)
    center = ((size[0] - 1) / 2, (size[1] - 1) / 2)
    m_rot_ref, m_rot_cv = RM.rotation_matrix(center, ROTATE_ANGLE), cv2.getRotationMatrix2D(center, ROTATE_ANGLE, 1)
    m_aff_ref = RM.affine_from_3pts(AFFINE_SRC, AFFINE_DST)
    m_aff_cv = cv2.getAffineTransform(np.float32(AFFINE_SRC), np.float32(AFFINE_DST))
    h_ref = RM.homography_from_4pts(PERSP_SRC, PERSP_DST)
    h_cv = cv2.getPerspectiveTransform(np.float32(PERSP_SRC), np.float32(PERSP_DST))
    out_small = f"{RESIZE_INPUT[0]}×{RESIZE_INPUT[1]} → {RESIZE_INPUT[0] * RESIZE_FACTOR}×" \
                f"{RESIZE_INPUT[1] * RESIZE_FACTOR}"
    same = f"{size[0]}×{size[1]} → {size[0]}×{size[1]}"
    return [
        Case(f"resize ×{RESIZE_FACTOR}", out_small,
             lambda it: RW.resize_ref(small, RESIZE_FACTOR, interp=it),
             lambda f: cv2.resize(small, None, fx=RESIZE_FACTOR, fy=RESIZE_FACTOR, interpolation=f)),
        Case(f"xoay {ROTATE_ANGLE:g}° (giữ khung)", same,
             lambda it: RW.warp_affine_ref(cam, m_rot_ref, size, it),
             lambda f: cv2.warpAffine(cam, m_rot_cv, size, flags=f),
             matrix_err=float(np.abs(m_rot_ref - m_rot_cv).max())),
        Case("affine 3 điểm", same,
             lambda it: RW.warp_affine_ref(cam, m_aff_ref, size, it),
             lambda f: cv2.warpAffine(cam, m_aff_cv, size, flags=f),
             matrix_err=float(np.abs(m_aff_ref - m_aff_cv).max())),
        Case("phối cảnh 4 điểm", same,
             lambda it: RW.warp_perspective_ref(cam, h_ref, size, it),
             lambda f: cv2.warpPerspective(cam, h_cv, size, flags=f),
             matrix_err=float(np.abs(h_ref - h_cv).max())),
    ]


def quantized_psnr() -> dict[str, dict[str, float]]:
    """Kiểm chứng: lượng tử hóa tọa độ của bản tự cài đặt về 1/32 px làm PSNR so với OpenCV tăng lên."""
    cam = images.load_camera()
    size = (cam.shape[1], cam.shape[0])
    center = ((size[0] - 1) / 2, (size[1] - 1) / 2)
    m = cv2.getRotationMatrix2D(center, ROTATE_ANGLE, 1)
    h = cv2.getPerspectiveTransform(np.float32(PERSP_SRC), np.float32(PERSP_DST))
    out: dict[str, dict[str, float]] = {}
    for it in ("linear", "cubic"):
        cv_rot = cv2.warpAffine(cam, m, size, flags=INTERP_CV[it])
        cv_per = cv2.warpPerspective(cam, h, size, flags=INTERP_CV[it])
        out[f"xoay {ROTATE_ANGLE:g}°, {INTERP_LABEL[it]}"] = {
            "float": psnr(RW.warp_affine_ref(cam, m, size, it), cv_rot),
            "q32": psnr(RW.warp_affine_ref(cam, m, size, it, coord_bits=OPENCV_INTER_BITS), cv_rot)}
        out[f"phối cảnh, {INTERP_LABEL[it]}"] = {
            "float": psnr(RW.warp_perspective_ref(cam, h, size, it), cv_per),
            "q32": psnr(RW.warp_perspective_ref(cam, h, size, it, coord_bits=OPENCV_INTER_BITS), cv_per)}
    return out


def run_cases(cases: list[Case]) -> None:
    """Chạy mọi phép × mọi cách nội suy: đo PSNR, % pixel lệch, thời gian."""
    for case in cases:
        for it, flag in INTERP_CV.items():
            a, b = case.ref(it), case.cv(flag)
            t_ref, t_cv = benchmark(case.ref, it), benchmark(case.cv, flag)
            case.results[it] = {"ref": a, "cv": b, "psnr": psnr(a, b), "pct": pct_diff_gt(a, b),
                                "max": max_abs_diff(a, b), "t_ref": t_ref.mean_ms, "t_cv": t_cv.mean_ms,
                                "slower": t_ref.mean_ms / t_cv.mean_ms}


# ------------------------------------------------------------------ Hình 7, Bảng 3
def _diff_panel(fig: plt.Figure, ax: plt.Axes, a: np.ndarray, b: np.ndarray, title: str) -> None:
    """Ô ảnh hiệu |a − b| phóng đại tương phản (thang 0–DIFF_VMAX mức xám) kèm thanh màu."""
    im = ax.imshow(abs_diff(a, b), cmap="inferno", vmin=0, vmax=DIFF_VMAX, interpolation="nearest")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title)
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
    cb.ax.tick_params(labelsize=PANEL_TITLE_SIZE - 1)


def make_hinh7(cases: list[Case], out_dir: Path = FIG_DIR) -> list[Path]:
    """Hình 7a (2×3): resize ×4 và xoay 45°; Hình 7b (2×2): phối cảnh và kiểm chứng lượng tử 1/32 px."""
    rows_7a = [c for c in cases if c.name.startswith(("resize", "xoay"))]
    persp = next(c for c in cases if c.name.startswith("phối cảnh"))
    paths = []
    with plt.rc_context({"axes.titlesize": PANEL_TITLE_SIZE}):
        fig, axes = panel_grid(2, 3, FIG_W_FULL, 1.0)
        for row, case in zip(axes, rows_7a):
            r = case.results[FIG7_INTERP]
            show_image(row[0], r["ref"], f"Tự cài đặt: {case.name}\n{INTERP_LABEL[FIG7_INTERP]}, {case.io_size}")
            show_image(row[1], r["cv"], f"OpenCV: {case.name}\nINTER_LINEAR")
            _diff_panel(fig, row[2], r["ref"], r["cv"], f"|Hiệu| (0–{DIFF_VMAX:g}), PSNR {format_psnr(r['psnr'])} dB\n"
                        f"lệch >1: {r['pct']:.3f}%, max = {r['max']:.0f}")
        paths.append(save_panels(fig, out_dir, "Hinh7a"))

        cam = images.load_camera()
        size = (cam.shape[1], cam.shape[0])
        h = cv2.getPerspectiveTransform(np.float32(PERSP_SRC), np.float32(PERSP_DST))
        q32 = RW.warp_perspective_ref(cam, h, size, FIG7_INTERP, coord_bits=OPENCV_INTER_BITS)
        r = persp.results[FIG7_INTERP]
        fig, axes = panel_grid(2, 2, FIG_W_2X2, 1.0)
        show_image(axes[0, 0], r["ref"], f"Tự cài đặt: {persp.name}\n{INTERP_LABEL[FIG7_INTERP]}")
        show_image(axes[0, 1], r["cv"], f"OpenCV: {persp.name}\nINTER_LINEAR")
        _diff_panel(fig, axes[1, 0], r["ref"], r["cv"],
                    f"|Hiệu|, tọa độ float64\nPSNR {format_psnr(r['psnr'])} dB")
        _diff_panel(fig, axes[1, 1], q32, r["cv"],
                    f"|Hiệu|, tọa độ lượng tử 1/32 px\nPSNR {format_psnr(psnr(q32, r['cv']))} dB")
        paths.append(save_panels(fig, out_dir, "Hinh7b"))
    return paths


BANG3_HEADER: list[str] = ["Phép biến đổi", "Kích thước", "Nội suy", "PSNR (dB)", "Pixel lệch >1 (%)",
                           "t tự cài đặt (ms)", "t OpenCV (ms)", "Chậm hơn (lần)", "Sai số ma trận max"]


def make_bang3(cases: list[Case], matrix_errs: dict[str, MatrixError]) -> tuple[Path, Path]:
    """Bảng 3: so sánh định lượng + dòng sai số lớn nhất giữa ma trận tự tính và OpenCV."""
    rows = []
    for case in cases:
        for it, r in case.results.items():
            m_err = "—" if case.matrix_err is None else f"{case.matrix_err:.2e}"
            rows.append([case.name, case.io_size, INTERP_LABEL[it], format_psnr(r["psnr"]), f"{r['pct']:.3f}",
                         f"{r['t_ref']:.2f}", f"{r['t_cv']:.3f}", f"{r['slower']:.0f}", m_err])
    worst = max(abs(c.matrix_err) for c in cases if c.matrix_err is not None)
    rows.append(["Sai số lớn nhất |M tự tính − M OpenCV| (các ma trận dùng ở trên)",
                 "—", "—", "—", "—", "—", "—", "—", f"{worst:.2e}"])
    for k, e in matrix_errs.items():
        rows.append([f"Ma trận {k}: {N_RANDOM_MATRICES} trường hợp ngẫu nhiên, sai số tương đối lớn nhất",
                     "—", "—", "—", "—", "—", "—", "—", f"{e.rel_max:.2e}"])
    notes = [
        "Ảnh vào: 'camera' (skimage, xám 512×512); resize dùng ảnh camera thu nhỏ về 256×256 bằng INTER_AREA.",
        "Biên: BORDER_CONSTANT (giá trị 0) cho warp, lặp pixel biên cho resize — giống mặc định OpenCV.",
        "Sai số tương đối = max|M tự tính − M OpenCV| / max|M OpenCV| (homography ngẫu nhiên có phần tử "
        "tới ~10⁴ nên dùng sai số tương đối).",
        f"Thời gian: trung bình {BENCH_REPEAT} lần, bỏ {BENCH_WARMUP} lần warm-up; "
        f"OpenCV dùng {cv2.getNumThreads()} luồng, bản tự cài đặt chạy NumPy đơn luồng.",
    ]
    return write_table("Bang3", "Bảng 3. Cài đặt tham chiếu (NumPy) so với OpenCV", BANG3_HEADER, rows, notes)


def run_comparison() -> dict[str, Any]:
    """Chạy toàn bộ phần đối chứng; trả về số liệu cho report_numbers.md."""
    setup_style()
    rng = np.random.default_rng(SEED)
    matrix_errs = {"xoay": rotation_matrix_error(rng), "affine": affine_matrix_error(rng),
                   "homography": homography_error(rng)}
    cases = build_cases()
    run_cases(cases)
    fig_paths = make_hinh7(cases)
    table_paths = make_bang3(cases, matrix_errs)
    summary = {c.name: {it: {k: v for k, v in r.items() if k not in ("ref", "cv")}
                        for it, r in c.results.items()} for c in cases}
    return {"Hinh7": {"paths": fig_paths}, "Bang3": {"paths": table_paths, "cases": summary,
                                                    "matrix_errs": matrix_errs,
                                                    "case_matrix_errs": {c.name: c.matrix_err for c in cases}},
            "quantization": quantized_psnr()}


if __name__ == "__main__":
    run_comparison()
