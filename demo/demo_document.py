"""Demo nắn chỉnh ảnh tài liệu chụp nghiêng: click 4 góc (hoặc --auto) → ảnh thẳng.

Ví dụ:
    python demo/demo_document.py --image data/doc.jpg            # click 4 góc bằng chuột
    python demo/demo_document.py --image data/doc.jpg --auto     # tự tìm 4 góc
    python demo/demo_document.py --auto --save out.png --no-show

Cửa sổ chọn góc: click trái 4 góc tờ giấy (thứ tự bất kỳ), Enter/Space để xác nhận,
r để chọn lại, Esc để thoát.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import cv2  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from src import app_document as doc  # noqa: E402
from src.images import read_rgb  # noqa: E402
from src.plotting import draw_points, setup_style, show_image  # noqa: E402

WINDOW: str = "Click 4 goc tai lieu - Enter: xong, r: chon lai, Esc: thoat"
MAX_VIEW_SIDE: int = 1000           # thu nhỏ ảnh khi hiển thị để vừa màn hình
KEY_ENTER, KEY_SPACE, KEY_ESC, KEY_RESET = 13, 32, 27, ord("r")
MARK_RADIUS: int = 7
MARK_COLOR: tuple[int, int, int] = (0, 0, 255)   # BGR đỏ
MS_PER_S: float = 1000.0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Đọc tham số dòng lệnh."""
    p = argparse.ArgumentParser(description="Nắn chỉnh ảnh tài liệu chụp nghiêng")
    p.add_argument("--image", default=str(ROOT / "data" / "doc.jpg"), help="ảnh tài liệu chụp nghiêng")
    p.add_argument("--auto", action="store_true", help="tự tìm 4 góc thay vì click")
    p.add_argument("--save", default=None, help="lưu hình trước/sau ra file PNG")
    p.add_argument("--no-show", action="store_true", help="không mở cửa sổ hiển thị kết quả")
    return p.parse_args(argv)


def pick_corners(img_rgb: np.ndarray) -> np.ndarray | None:
    """Cho người dùng click 4 góc trên cửa sổ OpenCV; trả về (4, 2) theo tọa độ ảnh gốc."""
    h, w = img_rgb.shape[:2]
    ratio = min(1.0, MAX_VIEW_SIDE / max(h, w))
    view = cv2.resize(cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR), None, fx=ratio, fy=ratio,
                      interpolation=cv2.INTER_AREA)
    pts: list[tuple[int, int]] = []

    def on_mouse(event: int, x: int, y: int, *_: object) -> None:
        if event == cv2.EVENT_LBUTTONDOWN and len(pts) < doc.N_CORNERS:
            pts.append((x, y))

    cv2.namedWindow(WINDOW, cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(WINDOW, on_mouse)
    while True:
        canvas = view.copy()
        for p in pts:
            cv2.circle(canvas, p, MARK_RADIUS, MARK_COLOR, -1)
        if len(pts) > 1:
            cv2.polylines(canvas, [np.int32(pts)], len(pts) == doc.N_CORNERS, MARK_COLOR, 2)
        cv2.imshow(WINDOW, canvas)
        key = cv2.waitKey(20) & 0xFF
        if key == KEY_ESC:
            cv2.destroyAllWindows()
            return None
        if key == KEY_RESET:
            pts.clear()
        if key in (KEY_ENTER, KEY_SPACE) and len(pts) == doc.N_CORNERS:
            cv2.destroyAllWindows()
            return np.array(pts, dtype=np.float64) / ratio


def main(argv: list[str] | None = None) -> int:
    """Chạy demo; trả về mã thoát."""
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    args = parse_args(argv)
    if args.no_show:
        plt.switch_backend("Agg")
    setup_style()
    img = read_rgb(args.image)
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    corners = None
    if args.auto:
        t0 = time.perf_counter()
        corners = doc.detect_corners_auto(img)
        print(f"Tự tìm góc: {'thành công' if corners is not None else 'thất bại'} "
              f"({(time.perf_counter() - t0) * MS_PER_S:.1f} ms)")
        if corners is None and args.no_show:
            print("Không tìm được 4 góc và không có màn hình để click — dừng.")
            return 1
    if corners is None:
        corners = pick_corners(img)
        if corners is None:
            print("Đã hủy.")
            return 1
    corners = doc.order_corners(corners)
    t0 = time.perf_counter()
    out, hmat = doc.rectify(img, corners)
    print(f"4 góc (TL, TR, BR, BL): {np.round(corners, 1).tolist()}")
    print(f"Ảnh ra {out.shape[1]}×{out.shape[0]}, nắn chỉnh mất {(time.perf_counter() - t0) * MS_PER_S:.1f} ms")
    print("Homography H =\n" + np.array2string(hmat, precision=4, suppress_small=True))

    fig, axes = plt.subplots(1, 2, figsize=(13, 6.5))
    show_image(axes[0], img, f"Trước: {Path(args.image).name} {img.shape[1]}×{img.shape[0]}", smooth=True)
    draw_points(axes[0], corners, "C")
    show_image(axes[1], out, f"Sau nắn chỉnh {out.shape[1]}×{out.shape[0]}\nwarpPerspective, INTER_CUBIC",
               smooth=True)
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
