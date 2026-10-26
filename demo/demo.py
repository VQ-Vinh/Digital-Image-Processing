"""Demo dòng lệnh từng phép biến đổi hình học bằng OpenCV, tùy chọn so sánh với bản tự cài đặt.

Ví dụ:
    python demo/demo.py --op rotate --angle 45 --interp linear --border reflect --expand --image data/astronaut.png
    python demo/demo.py --op rotate --angle 45 --expand --compare
    python demo/demo.py --op scale --fx 4 --interp cubic --image data/text.png --compare
    python demo/demo.py --op perspective --compare --save out.png --no-show
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from compare.benchmark import benchmark  # noqa: E402
from compare.metrics import abs_diff, format_psnr, max_abs_diff, pct_diff_gt, psnr  # noqa: E402
from reference import matrices as RM  # noqa: E402
from reference import warp_ref as RW  # noqa: E402
from src import transforms as T  # noqa: E402
from src.images import read_rgb  # noqa: E402
from src.plotting import setup_style, show_image  # noqa: E402

OPS: tuple[str, ...] = ("scale", "translate", "rotate", "affine", "perspective")
REF_INTERPS: tuple[str, ...] = RW.INTERPS            # bản tự cài đặt chỉ có nearest / linear / cubic
DIFF_VMAX: float = 3.0
# Điểm mặc định cho affine / phối cảnh, tính theo tỉ lệ kích thước ảnh.
AFFINE_SRC_FRAC = np.array([[0.1, 0.1], [0.9, 0.1], [0.1, 0.9]])
AFFINE_DST_FRAC = np.array([[0.2, 0.3], [0.85, 0.08], [0.3, 0.92]])
PERSP_SRC_FRAC = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
PERSP_DST_FRAC = np.array([[0.3, 0.12], [0.75, 0.05], [0.96, 0.96], [0.05, 0.82]])


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Đọc tham số dòng lệnh."""
    p = argparse.ArgumentParser(description="Demo biến đổi hình học bằng OpenCV")
    p.add_argument("--op", choices=OPS, default="rotate", help="phép biến đổi")
    p.add_argument("--image", default=str(ROOT / "data" / "astronaut.png"), help="ảnh vào")
    p.add_argument("--interp", choices=list(T.INTERP), default="linear", help="cờ nội suy")
    p.add_argument("--border", choices=list(T.BORDER), default="constant", help="chế độ biên")
    p.add_argument("--border-value", type=int, default=0, help="giá trị biên khi --border constant")
    p.add_argument("--angle", type=float, default=45.0, help="góc xoay (độ, dương = ngược chiều kim đồng hồ)")
    p.add_argument("--scale", type=float, default=1.0, help="hệ số co giãn đi kèm phép xoay")
    p.add_argument("--expand", action="store_true", help="xoay mở rộng khung (không bị cắt góc)")
    p.add_argument("--fx", type=float, default=2.0, help="hệ số co giãn theo x (op scale)")
    p.add_argument("--fy", type=float, default=None, help="hệ số co giãn theo y (mặc định = fx)")
    p.add_argument("--tx", type=float, default=80.0, help="tịnh tiến theo x (px)")
    p.add_argument("--ty", type=float, default=50.0, help="tịnh tiến theo y (px)")
    p.add_argument("--compare", action="store_true", help="so sánh với bản tự cài đặt (reference/)")
    p.add_argument("--repeat", type=int, default=10, help="số lần đo thời gian (bỏ 1 lần warm-up)")
    p.add_argument("--save", default=None, help="lưu hình kết quả ra file PNG")
    p.add_argument("--no-show", action="store_true", help="không mở cửa sổ hiển thị")
    return p.parse_args(argv)


def _points(img: np.ndarray, frac: np.ndarray) -> np.ndarray:
    """Đổi tọa độ tỉ lệ (0–1) thành pixel."""
    h, w = img.shape[:2]
    return frac * np.array([w - 1, h - 1])


def run_opencv(args: argparse.Namespace, img: np.ndarray) -> tuple[np.ndarray, str]:
    """Thực hiện phép biến đổi bằng OpenCV; trả về (ảnh ra, mô tả tham số)."""
    interp, border = T.INTERP[args.interp], T.BORDER[args.border]
    kw = {"interp": interp, "border": border, "border_value": args.border_value}
    if args.op == "scale":
        return T.scale(img, args.fx, args.fy, interp), f"resize fx={args.fx}, fy={args.fy or args.fx}"
    if args.op == "translate":
        return T.translate(img, args.tx, args.ty, **kw), f"tịnh tiến ({args.tx:g}, {args.ty:g})"
    if args.op == "rotate":
        out = T.rotate(img, args.angle, args.scale, args.expand, **kw)
        return out, f"xoay {args.angle:g}°, scale={args.scale:g}, {'mở rộng' if args.expand else 'giữ'} khung"
    if args.op == "affine":
        out, _ = T.affine_3pts(img, _points(img, AFFINE_SRC_FRAC), _points(img, AFFINE_DST_FRAC), **kw)
        return out, "affine 3 điểm"
    out, _ = T.perspective_4pts(img, _points(img, PERSP_SRC_FRAC), _points(img, PERSP_DST_FRAC), **kw)
    return out, "phối cảnh 4 điểm"


def run_reference(args: argparse.Namespace, img: np.ndarray) -> np.ndarray:
    """Cùng phép biến đổi bằng bản tự cài đặt (ma trận tự tính, ánh xạ ngược NumPy)."""
    h, w = img.shape[:2]
    kw = {"interp": args.interp, "border": args.border, "border_value": args.border_value}
    if args.op == "scale":
        return RW.resize_ref(img, args.fx, args.fy, args.interp)
    if args.op == "translate":
        return RW.warp_affine_ref(img, RM.translation_matrix(args.tx, args.ty), (w, h), **kw)
    if args.op == "rotate":
        if args.expand:
            m, size = RM.rotation_matrix_expand((w, h), args.angle, args.scale)
        else:
            m, size = RM.rotation_matrix(((w - 1) / 2, (h - 1) / 2), args.angle, args.scale), (w, h)
        return RW.warp_affine_ref(img, m, size, **kw)
    if args.op == "affine":
        m = RM.affine_from_3pts(_points(img, AFFINE_SRC_FRAC), _points(img, AFFINE_DST_FRAC))
        return RW.warp_affine_ref(img, m, (w, h), **kw)
    hm = RM.homography_from_4pts(_points(img, PERSP_SRC_FRAC), _points(img, PERSP_DST_FRAC))
    return RW.warp_perspective_ref(img, hm, (w, h), **kw)


def main(argv: list[str] | None = None) -> int:
    """Chạy demo; trả về mã thoát."""
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    args = parse_args(argv)
    if args.no_show:
        plt.switch_backend("Agg")
    setup_style()
    img = read_rgb(args.image)
    out, desc = run_opencv(args, img)
    flag = T.INTERP_NAME[T.INTERP[args.interp]]
    border = T.BORDER_NAME[T.BORDER[args.border]]
    print(f"Ảnh vào {img.shape[1]}×{img.shape[0]} → ảnh ra {out.shape[1]}×{out.shape[0]}; {desc}, {flag}, {border}")
    t_cv = benchmark(run_opencv, args, img, repeat=args.repeat)
    print(f"Thời gian OpenCV: {t_cv.mean_ms:.3f} ms (trung bình {args.repeat} lần)")

    compare = args.compare and args.interp in REF_INTERPS
    if args.compare and not compare:
        print(f"Bỏ qua --compare: bản tự cài đặt chỉ hỗ trợ {', '.join(REF_INTERPS)}.")
    ncol = 4 if compare else 2
    fig, axes = plt.subplots(1, ncol, figsize=(4.5 * ncol, 5))
    show_image(axes[0], img, f"Ảnh gốc {img.shape[1]}×{img.shape[0]}")
    show_image(axes[1], out, f"OpenCV: {desc}\n{flag}, {border}")
    if compare:
        ref = run_reference(args, img)
        t_ref = benchmark(run_reference, args, img, repeat=args.repeat)
        p, pct = psnr(ref, out), pct_diff_gt(ref, out)
        print(f"Tự cài đặt: {t_ref.mean_ms:.2f} ms (chậm hơn {t_ref.mean_ms / t_cv.mean_ms:.0f} lần)")
        print(f"PSNR(tự cài đặt, OpenCV) = {format_psnr(p)} dB; pixel lệch >1: {pct:.3f}%; "
              f"lệch lớn nhất: {max_abs_diff(ref, out):.0f}")
        show_image(axes[2], ref, f"Tự cài đặt (NumPy)\n{args.interp}, {args.border}")
        im = axes[3].imshow(abs_diff(ref, out), cmap="inferno", vmin=0, vmax=DIFF_VMAX)
        axes[3].set_xticks([])
        axes[3].set_yticks([])
        axes[3].set_title(f"Ảnh hiệu (thang 0–{DIFF_VMAX:g})\nPSNR = {format_psnr(p)} dB")
        fig.colorbar(im, ax=axes[3], fraction=0.046, pad=0.03)
    fig.tight_layout()
    if args.save:
        fig.savefig(args.save, dpi=200, facecolor="white", bbox_inches="tight")
        print(f"Đã lưu {args.save}")
    if not args.no_show:
        plt.show()
    plt.close(fig)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
