"""Sinh results/report_numbers.md: cấu hình máy, số liệu chính và nhận xét gợi ý.

Mọi con số trong file đều lấy từ dict kết quả do experiments.py và compare_ref_vs_cv.py
trả về; các câu nhận xét được chọn theo số liệu (so sánh lớn/nhỏ) chứ không viết cứng.
"""
from __future__ import annotations

import platform
import subprocess
import sys
from pathlib import Path
from typing import Any

import cv2
import matplotlib
import numpy as np
import psutil
import skimage

from compare.metrics import format_psnr
from src.config import BENCH_REPEAT, BENCH_WARMUP, REPORT_NUMBERS_PATH

BYTES_PER_GB: float = 1024 ** 3
PSNR_EQUAL_TOL: float = 1e-9
REALTIME_FRAME_MS: float = 1000 / 30   # ngân sách thời gian mỗi khung ở 30 khung/s


def cpu_name() -> str:
    """Tên CPU theo hệ điều hành (registry Windows, /proc/cpuinfo Linux, sysctl macOS)."""
    system = platform.system()
    try:
        if system == "Windows":
            import winreg
            key = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0")
            return str(winreg.QueryValueEx(key, "ProcessorNameString")[0]).strip()
        if system == "Linux":
            for line in Path("/proc/cpuinfo").read_text().splitlines():
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
        if system == "Darwin":
            return subprocess.check_output(["sysctl", "-n", "machdep.cpu.brand_string"], text=True).strip()
    except (OSError, subprocess.SubprocessError):
        pass
    return platform.processor() or "không xác định"


def system_info() -> dict[str, str]:
    """Cấu hình máy và phiên bản thư viện."""
    return {
        "CPU": f"{cpu_name()} ({psutil.cpu_count(logical=False)} nhân / {psutil.cpu_count()} luồng)",
        "RAM": f"{psutil.virtual_memory().total / BYTES_PER_GB:.1f} GB",
        "Hệ điều hành": platform.platform(),
        "Python": sys.version.split()[0],
        "NumPy": np.__version__,
        "OpenCV": cv2.__version__,
        "Matplotlib": matplotlib.__version__,
        "scikit-image": skimage.__version__,
        "Số luồng OpenCV (cv2.getNumThreads)": str(cv2.getNumThreads()),
    }


def _argmax(d: dict[str, float]) -> str:
    """Khóa có giá trị lớn nhất."""
    return max(d, key=d.get)


def _argmin(d: dict[str, float]) -> str:
    """Khóa có giá trị nhỏ nhất."""
    return min(d, key=d.get)


# ------------------------------------------------------------------ nhận xét từng hình
def comments_hinh1() -> list[str]:
    """Hình 1 là sơ đồ, nhận xét mô tả quy trình."""
    return [
        "Mọi phép biến đổi hình học trong OpenCV đều theo hai bước: tính ma trận (2×3 cho affine, 3×3 cho "
        "phối cảnh) rồi gọi hàm warp để ánh xạ ngược từng pixel đích về ảnh nguồn và nội suy.",
        "Riêng cv2.resize không cần ma trận tường minh: ma trận co giãn được suy ra từ dsize hoặc fx, fy.",
        "Chất lượng và tốc độ được quyết định bởi các tham số interpolation, borderMode và dsize, độc lập với "
        "cách tính ma trận.",
        "Nhánh nét đứt là phần đối chứng: cài đặt lại cùng quy trình bằng NumPy để kiểm chứng cơ chế bên trong.",
    ]


def comments_hinh2(r: dict[str, Any]) -> list[str]:
    """Nhận xét phóng to / thu nhỏ dựa trên PSNR và độ lệch chuẩn vùng chồng phổ."""
    up, alias = r["up_psnr"], r["alias_std"]
    best, worst = _argmax(up), _argmin(up)
    out = [f"Khi phóng to ×4 ảnh {r['small_size'][0]}×{r['small_size'][1]}, {best} cho PSNR cao nhất "
           f"({up[best]:.2f} dB), {worst} thấp nhất ({up[worst]:.2f} dB); chênh lệch "
           f"{up[best] - up[worst]:.2f} dB cho thấy nội suy bậc cao khôi phục chi tiết tốt hơn nhưng không "
           "thể tạo lại chi tiết đã mất khi thu nhỏ."]
    if abs(up["INTER_AREA"] - up["INTER_NEAREST"]) < PSNR_EQUAL_TOL:
        out.append("Khi phóng to với hệ số nguyên, INTER_AREA cho kết quả trùng INTER_NEAREST (cùng PSNR "
                   f"{up['INTER_AREA']:.2f} dB) — đúng như tài liệu OpenCV mô tả, nên không dùng INTER_AREA để phóng to.")
    out.append("Vùng phóng lớn cho thấy INTER_NEAREST tạo khối vuông (răng cưa), INTER_LINEAR làm nhòe biên, "
               "INTER_CUBIC và INTER_LANCZOS4 giữ biên sắc hơn nhưng có thể xuất hiện viền sáng/tối (ringing).")
    lo, hi = _argmin(alias), _argmax(alias)
    out.append(f"Khi thu nhỏ ×0.25 zone plate, độ lệch chuẩn ở vùng có tần số vượt Nyquist (lý tưởng ≈ 0) "
               f"nhỏ nhất với {lo} ({alias[lo]:.1f}) và lớn nhất với {hi} ({alias[hi]:.1f}), tức {lo} "
               f"giảm vân moiré khoảng {alias[hi] / alias[lo]:.1f} lần nhờ lấy trung bình cả vùng pixel.")
    out.append("Các cờ còn lại chỉ lấy mẫu vài pixel lân cận nên bỏ sót phần lớn pixel nguồn: chữ mảnh bị đứt "
               "nét và vân tròn giả xuất hiện ở vùng tần số cao.")
    return out


def comments_hinh3(r: dict[str, Any]) -> list[str]:
    """Nhận xét tịnh tiến, xoay giữ / mở rộng khung, chế độ biên."""
    kk, se = r["kept_keep"], r["sizes_expand"]
    w, h = r["size"]
    a_min = min(kk, key=kk.get)
    big = max(se, key=lambda a: se[a][0] * se[a][1])
    return [
        f"Tịnh tiến giữ khung {w}×{h} làm mất phần ảnh trượt ra ngoài: chỉ còn {r['kept_shift']:.1f}% ảnh gốc, "
        "phần trống được lấp bằng borderValue.",
        "Xoay giữ khung làm mất góc ảnh: còn " + ", ".join(f"{kk[a]:.1f}% ở {a:g}°" for a in kk)
        + f"; với ảnh không vuông, góc {a_min:g}° mất nhiều nhất.",
        f"Xoay mở rộng khung tính lại kích thước (ví dụ {big:g}° → {se[big][0]}×{se[big][1]}, diện tích gấp "
        f"{se[big][0] * se[big][1] / (w * h):.2f} lần) và cộng bù tịnh tiến vào cột cuối của ma trận; nếu quên "
        "bù tịnh tiến, ảnh bị lệch và vẫn bị cắt như ô đầu hàng thứ hai.",
        "Với borderMode: CONSTANT tô màu cố định, REPLICATE kéo dài pixel biên thành vệt sọc, REFLECT/"
        "REFLECT_101 tạo ảnh phản chiếu liền mạch, WRAP lặp ảnh tuần hoàn nên có đường nối đột ngột.",
    ]


def comments_hinh4(r: dict[str, Any]) -> list[str]:
    """Nhận xét affine 3 điểm."""
    m = r["matrix"]
    return [
        "Ba cặp điểm cho 6 phương trình, vừa đủ xác định 6 tham số của ma trận affine 2×3.",
        f"Sai số lớn nhất khi chiếu P1–P3 qua M so với Q1–Q3 là {r['point_err']:.1e} px, chỉ còn sai số làm tròn.",
        f"det của phần tuyến tính M[:, :2] = {r['det']:.3f}, tức diện tích ảnh sau biến đổi bằng "
        f"{r['det'] * 100:.1f}% diện tích ban đầu; tịnh tiến ({m[0, 2]:.1f}, {m[1, 2]:.1f}) px.",
        "Ảnh bàn cờ cho thấy affine giữ đường thẳng và tính song song nhưng làm các ô vuông thành hình bình hành.",
    ]


def comments_hinh5(r: dict[str, Any]) -> list[str]:
    """Nhận xét phối cảnh 4 điểm."""
    h = r["matrix"]
    return [
        "Bốn cặp điểm cho 8 phương trình, đủ xác định 8 bậc tự do của homography 3×3 (chuẩn hóa h33 = 1).",
        f"Hàng cuối H có h31 = {h[2, 0]:.2e}, h32 = {h[2, 1]:.2e} khác 0 nên mẫu số w thay đổi theo vị trí — "
        "đây là khác biệt cốt lõi so với affine.",
        f"Sai số chiếu lại 4 điểm là {r['point_err']:.1e} px.",
        f"Cạnh trên và cạnh dưới vốn song song, sau biến đổi có độ dài tỉ lệ {r['edge_ratio']:.2f} : 1 và không còn "
        "song song; bàn cờ cho thấy đường thẳng vẫn thẳng nhưng các ô co dần về phía xa.",
    ]


def comments_hinh6(r: dict[str, Any]) -> list[str]:
    """Nhận xét ứng dụng nắn chỉnh tài liệu."""
    out = [
        f"Trên ảnh tổng hợp {r['photo_size'][0]}×{r['photo_size'][1]}, thuật toán tự tìm 4 góc (Otsu + contour + "
        f"approxPolyDP) với sai số trung bình {r['corner_err']:.2f} px so với góc thật.",
        f"Ảnh sau nắn chỉnh {r['out_size'][0]}×{r['out_size'][1]} đạt PSNR {r['psnr_auto']:.2f} dB so với trang "
        f"gốc; dùng 4 góc thật thì đạt {r['psnr_true']:.2f} dB — phần chênh do sai số góc, phần còn lại do bóng "
        "đổ, nhiễu và nội suy khi tạo ảnh.",
        f"Thời gian tìm góc {r['t_detect_ms']:.2f} ms, nắn chỉnh {r['t_rectify_ms']:.2f} ms"
        + (f", tổng dưới {REALTIME_FRAME_MS:.0f} ms/khung nên đủ nhanh cho video thời gian thực (30 khung/s)."
           if r["t_detect_ms"] + r["t_rectify_ms"] < REALTIME_FRAME_MS else "."),
        "Chữ và bảng kẻ ô sau nắn chỉnh thẳng hàng trở lại, chứng tỏ homography khử được méo phối cảnh.",
    ]
    for item in r.get("real", []):
        out.append(f"Ảnh thật {item['file']}: " + ("tự tìm được 4 góc." if item["found"]
                                                    else "không tự tìm được 4 góc, cần chọn góc bằng tay."))
    return out


def comments_hinh7(cmp: dict[str, Any]) -> list[str]:
    """Nhận xét so sánh tự cài đặt với OpenCV."""
    cases = cmp["Bang3"]["cases"]
    lin = {name: c["linear"] for name, c in cases.items()}
    psnrs = [c["psnr"] for c in lin.values()]
    pcts = [c["pct"] for c in lin.values()]
    slower = [r["slower"] for c in cases.values() for r in c.values()]
    q = cmp["quantization"]
    q_gain = [v["q32"] - v["float"] for v in q.values() if np.isfinite(v["q32"])]
    q_exact = [k for k, v in q.items() if not np.isfinite(v["q32"])]
    out = [
        f"Với nội suy bilinear, ảnh tự cài đặt và ảnh OpenCV gần như trùng nhau: PSNR {min(psnrs):.2f}–"
        f"{max(psnrs):.2f} dB, tỉ lệ pixel lệch >1 mức xám chỉ {min(pcts):.3f}–{max(pcts):.3f}%.",
        "Ảnh hiệu tập trung ở biên vật thể và mép khung ảnh — nơi gradient lớn nên lệch tọa độ rất nhỏ cũng "
        "đổi giá trị pixel; vùng phẳng gần như bằng 0.",
        f"Bản NumPy chậm hơn OpenCV {min(slower):.0f}–{max(slower):.0f} lần (OpenCV dùng SIMD, đa luồng, "
        "bảng hệ số tính sẵn và số nguyên dấu phẩy cố định).",
    ]
    if q_gain:
        out.append(f"Khi lượng tử hóa tọa độ của bản tự cài đặt về 1/32 pixel như OpenCV, PSNR tăng thêm "
                   f"{min(q_gain):.1f}–{max(q_gain):.1f} dB" + (f" và trùng khớp hoàn toàn ở: {', '.join(q_exact)}"
                                                               if q_exact else "")
                   + " — xác nhận lượng tử hóa tọa độ là nguồn sai khác chính.")
    return out


EXPLANATION: str = (
    "Bản tự cài đặt và OpenCV dùng cùng một công thức ánh xạ ngược, nên sai khác chỉ đến từ chi tiết số học. "
    "(1) Quy ước tâm pixel: cả hai coi tâm pixel nằm ở tọa độ nguyên; với cv2.resize, tọa độ nguồn là "
    "x_s = (x + 0.5)/f − 0.5 (căn tâm pixel), nếu dùng x_s = x/f thì ảnh lệch nửa pixel. Tâm xoay của ảnh "
    "rộng w là (w − 1)/2 chứ không phải w/2. "
    "(2) Lượng tử hóa tọa độ: trong warpAffine/warpPerspective, OpenCV tính tọa độ nguồn bằng số nguyên dấu "
    "phẩy cố định rồi làm tròn về lưới 1/32 pixel (INTER_BITS = 5, INTER_TAB_SIZE = 32); bản tự cài đặt dùng "
    "tọa độ float64 nên lệch tới 1/64 pixel. "
    "(3) Hệ số nội suy dấu phẩy cố định: với ảnh uint8, OpenCV tra bảng hệ số tính sẵn cho 32 vị trí lẻ, "
    "nhân với 2^15 (INTER_REMAP_COEF_BITS) trong warp hoặc 2^11 (INTER_RESIZE_COEF_BITS) trong resize, cộng "
    "dồn số nguyên rồi dịch bit và bão hòa về [0, 255]; bản tự cài đặt cộng dồn số thực rồi mới làm tròn. "
    "(4) Nội suy bậc ba dùng nhân Keys với a = −0.75 ở cả hai bên (Keys [5] đề xuất a = −0.5), nên khác biệt "
    "chỉ do làm tròn hệ số."
)


def explanation(cmp: dict[str, Any]) -> str:
    """Đoạn giải thích nguyên nhân sai khác, kèm câu tổng kết lấy từ Bảng 3."""
    cases = cmp["Bang3"]["cases"].values()
    smooth = [r for c in cases for it, r in c.items() if it != "nearest"]
    nearest = [c["nearest"] for c in cases]
    text = (EXPLANATION + f" Theo Bảng 3, với bilinear và bicubic tỉ lệ pixel lệch quá 1 mức xám lớn nhất là "
            f"{max(r['pct'] for r in smooth):.3f}% và độ lệch lớn nhất chỉ {max(r['max'] for r in smooth):.0f} "
            "mức xám, tập trung ở biên có gradient lớn.")
    worst_nn = max(nearest, key=lambda r: r["max"])
    if worst_nn["max"] > 1:
        text += (f" Với nearest, nhiều nhất {worst_nn['pct']:.3f}% pixel lệch nhưng mỗi pixel lệch có thể khác tới "
                 f"{worst_nn['max']:.0f} mức xám: khi tọa độ nguồn rơi gần đúng giữa hai pixel, hai cách làm tròn "
                 "chọn hai pixel kề nhau khác nhau.")
    return text


# ------------------------------------------------------------------ số liệu chính
def _bang2_lines(b2: dict[str, Any]) -> list[str]:
    """Tóm tắt Bảng 2: cờ tốt nhất theo PSNR và nhanh/chậm nhất ở từng kích thước."""
    lines = []
    for s in b2["sizes"]:
        ps = {r["flag"]: r[f"psnr_{s}"] for r in b2["rows"]}
        td = {r["flag"]: r[f"t_down_{s}"] for r in b2["rows"]}
        best, worst = _argmax(ps), _argmin(ps)
        lines.append(f"- Ảnh {s}: PSNR cao nhất {best} ({ps[best]:.2f} dB), thấp nhất {worst} "
                     f"({ps[worst]:.2f} dB); thu nhỏ nhanh nhất {_argmin(td)} ({td[_argmin(td)]:.3f} ms), "
                     f"chậm nhất {_argmax(td)} ({td[_argmax(td)]:.3f} ms).")
    return lines


def _bang3_lines(cmp: dict[str, Any]) -> list[str]:
    """Tóm tắt Bảng 3 theo từng phép biến đổi và sai số ma trận."""
    lines = []
    for name, c in cmp["Bang3"]["cases"].items():
        parts = ", ".join(f"{it}: PSNR {format_psnr(r['psnr'])} dB, lệch >1 {r['pct']:.3f}%, "
                          f"chậm {r['slower']:.0f}×" for it, r in c.items())
        lines.append(f"- {name} — {parts}.")
    for k, e in cmp["Bang3"]["matrix_errs"].items():
        lines.append(f"- Ma trận {k} (ngẫu nhiên): sai số tuyệt đối max {e.abs_max:.2e}, tương đối max "
                     f"{e.rel_max:.2e}; sai số chiếu lại max: tự tính {e.reproj_ref:.2e} px, OpenCV "
                     f"{e.reproj_cv:.2e} px.")
    for name, v in cmp["quantization"].items():
        lines.append(f"- Lượng tử 1/32 px, {name}: PSNR {format_psnr(v['float'])} dB → {format_psnr(v['q32'])} dB.")
    return lines


def build_report(res: dict[str, Any], cmp: dict[str, Any]) -> str:
    """Ghép toàn bộ nội dung report_numbers.md."""
    info = system_info()
    lines = ["# Số liệu cho báo cáo (sinh tự động bởi run_all.py)", "",
             "## Cấu hình máy và phần mềm", ""]
    lines += [f"- {k}: {v}" for k, v in info.items()]
    lines += [f"- Đo thời gian: trung bình {BENCH_REPEAT} lần, bỏ {BENCH_WARMUP} lần warm-up "
              "(time.perf_counter).", "", "## Số liệu chính", "", "### Bảng 2 — cờ nội suy cv2.resize", ""]
    lines += _bang2_lines(res["Bang2"])
    lines += ["", "### Bảng 3 — tự cài đặt so với OpenCV", ""]
    lines += _bang3_lines(cmp)
    lines += ["", "## Nhận xét gợi ý theo từng hình", ""]
    comments = {
        "Hình 1": comments_hinh1(), "Hình 2": comments_hinh2(res["Hinh2"]), "Hình 3": comments_hinh3(res["Hinh3"]),
        "Hình 4": comments_hinh4(res["Hinh4"]), "Hình 5": comments_hinh5(res["Hinh5"]),
        "Hình 6": comments_hinh6(res["Hinh6"]), "Hình 7": comments_hinh7(cmp),
    }
    for name, sentences in comments.items():
        lines += [f"### {name}", "", *(f"- {s}" for s in sentences), ""]
    lines += ["## Nguyên nhân sai khác giữa tự cài đặt và OpenCV", "", explanation(cmp), ""]
    return "\n".join(lines)


def write_report(res: dict[str, Any], cmp: dict[str, Any], path: Path = REPORT_NUMBERS_PATH) -> Path:
    """Ghi report_numbers.md và trả về đường dẫn."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(build_report(res, cmp), encoding="utf-8")
    return path
