"""Sinh lại toàn bộ results/ (Hinh1–7, Bang1–3, report_numbers.md) bằng một lệnh.

Cách dùng:  python run_all.py
"""
from __future__ import annotations

import sys
import time

import matplotlib

matplotlib.use("Agg")  # chỉ ghi file, không mở cửa sổ

from compare.compare_ref_vs_cv import run_comparison  # noqa: E402
from compare.metrics import format_psnr  # noqa: E402
from src.config import DATA_DIR, FIG_DIR, TABLE_DIR  # noqa: E402
from src.experiments import run_all_experiments  # noqa: E402
from src.images import write_data_files  # noqa: E402
from src.plotting import make_doc_copies  # noqa: E402
from src.report import write_report  # noqa: E402


def print_summary(res: dict, cmp: dict) -> None:
    """In tóm tắt Bảng 2, Bảng 3 ra màn hình."""
    b2 = res["Bang2"]
    print("\n=== Bảng 2: PSNR thu nhỏ ×0.25 rồi phóng ×4 (dB) / thời gian thu nhỏ (ms) ===")
    for r in b2["rows"]:
        cells = "  ".join(f"{s}: {format_psnr(r[f'psnr_{s}']):>6} dB, {r[f't_down_{s}']:7.3f} ms" for s in b2["sizes"])
        print(f"  {r['flag']:<30} {cells}")
    print("\n=== Bảng 3: tự cài đặt so với OpenCV ===")
    for name, c in cmp["Bang3"]["cases"].items():
        for it, r in c.items():
            print(f"  {name:<22} {it:<8} PSNR {format_psnr(r['psnr']):>7} dB  lệch>1 {r['pct']:6.3f}%  "
                  f"chậm hơn {r['slower']:6.0f}×")
    for k, e in cmp["Bang3"]["matrix_errs"].items():
        print(f"  Ma trận {k:<11} sai số tương đối max {e.rel_max:.2e}")


def main() -> None:
    """Chạy toàn bộ: dữ liệu → thí nghiệm OpenCV → đối chứng → report_numbers.md."""
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")  # console Windows mặc định cp1252
    t0 = time.perf_counter()
    files = write_data_files()
    print(f"[1/4] Ghi {len(files)} ảnh thử vào {DATA_DIR}")
    res = run_all_experiments()
    print(f"[2/4] Phần chính (OpenCV): Hinh1–Hinh6, Bang1–Bang2 → {FIG_DIR}, {TABLE_DIR}")
    cmp = run_comparison()
    print("[3/4] Phần đối chứng: Hinh7, Bang3")
    path = write_report(res, cmp)
    make_doc_copies(FIG_DIR, FIG_DIR / "doc")
    print(f"[4/4] Ghi {path} và bản JPEG nhẹ của các hình vào {FIG_DIR / 'doc'}")
    print_summary(res, cmp)
    print(f"\nHoàn tất sau {time.perf_counter() - t0:.1f} s.")


if __name__ == "__main__":
    main()
