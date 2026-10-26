# Biến đổi hình học trên ảnh — BTL Xử lý ảnh số (EE3035)

Mã nguồn cho bài tập lớn "Biến đổi hình học trên ảnh", gồm hai phần:

- **Phần chính (`src/`, OpenCV):** thực hiện các phép co giãn, tịnh tiến, xoay (giữ khung / mở rộng khung), affine 3 điểm, phối cảnh 4 điểm; khảo sát ảnh hưởng của phương pháp nội suy, chế độ biên, kích thước khung; ứng dụng nắn chỉnh ảnh tài liệu chụp nghiêng.
- **Phần đối chứng (`reference/`, chỉ NumPy, không dùng `cv2`):** bản cài đặt tham chiếu đơn giản (ma trận tự tính + ánh xạ ngược + nội suy) chỉ để so sánh với OpenCV và giải thích cơ chế bên trong. Phần này không tối ưu.

Mọi hình, bảng và số liệu trong `results/` đều do code sinh ra bằng một lệnh `python run_all.py`; không có số nào nhập tay.

## 1. Ánh xạ file/hàm ↔ mục báo cáo ↔ hình/bảng

| File / hàm | Mục báo cáo | Hình / Bảng |
|---|---|---|
| `src/experiments.py::make_hinh1`, `make_bang1` | 1. Quy trình biến đổi hình học với OpenCV | Hinh1.png, Bang1 |
| `src/transforms.py::scale`; `src/experiments.py::make_hinh2`, `make_bang2` | 2. Co giãn ảnh, khảo sát cờ nội suy | Hinh2.png, Bang2 |
| `src/transforms.py::translate`, `rotation_matrix`, `rotate`, `expanded_size`; `make_hinh3` | 3. Tịnh tiến, xoay giữ / mở rộng khung, chế độ biên | Hinh3.png |
| `src/transforms.py::affine_3pts`; `make_hinh4` | 4. Biến đổi affine 3 điểm | Hinh4.png |
| `src/transforms.py::perspective_4pts`; `make_hinh5` | 5. Biến đổi phối cảnh 4 điểm | Hinh5.png |
| `src/app_document.py` (`detect_corners_auto`, `order_corners`, `rectify`); `make_hinh6` | 6. Ứng dụng: nắn chỉnh ảnh tài liệu | Hinh6.png |
| `reference/matrices.py`, `reference/warp_ref.py`; `compare/compare_ref_vs_cv.py` | 7. Đối chứng: tự cài đặt so với OpenCV | Hinh7.png, Bang3 |
| `compare/metrics.py`, `compare/benchmark.py` | Phương pháp đánh giá (PSNR, MSE, % pixel lệch, thời gian) | dùng cho Bang2, Bang3 |
| `src/report.py` | Số liệu và nhận xét tổng hợp | `results/report_numbers.md` |
| `src/images.py` | Dữ liệu thử | `data/*.png`, `data/doc.jpg` |

Cấu trúc thư mục (thư mục gốc của repo chính là `geotrans-project/`):

```
├── README.md, requirements.txt, pytest.ini, run_all.py
├── data/            ảnh thử (sinh lại bởi run_all.py; nhóm có thể thêm ảnh thật doc_real*.jpg)
├── src/             PHẦN CHÍNH (OpenCV): transforms, experiments, app_document
│                    + config (hằng số), images (dữ liệu), plotting, tables, report
├── reference/       PHẦN ĐỐI CHỨNG (NumPy, không cv2): matrices, warp_ref
├── compare/         metrics, benchmark, compare_ref_vs_cv
├── demo/            demo.py, demo_document.py, demo.ipynb
├── tests/           test_transforms, test_reference, test_no_cv2_in_reference
└── results/         figures/Hinh1–7.png, tables/Bang1–3.{md,csv}, report_numbers.md
```

Dữ liệu: không tải từ Internet. Dùng `skimage.data` (camera 512×512, astronaut 512×512, coffee 600×400) và ảnh tự tạo: bàn cờ, zone plate (thấy moiré), ảnh chữ vẽ không khử răng cưa, ảnh "tài liệu chụp nghiêng" tổng hợp bằng `warpPerspective` có kèm 4 góc thật. Hai kích thước thử: 512×512 và 1920×1080 (coffee phóng lên). Seed cố định `SEED = 2025` trong `src/config.py`.

## 2. Cài đặt và chạy test

Yêu cầu Python 3.10+ (đã kiểm tra với Python 3.13, OpenCV 4.10).

```bash
python -m venv .venv
```

Kích hoạt môi trường: Windows `.venv\Scripts\activate`, Linux/macOS `source .venv/bin/activate`. Sau đó:

```bash
pip install -r requirements.txt
```

```bash
python -m pytest
```

Kết quả mong đợi: `36 passed`. Bộ test kiểm tra:

- `test_transforms.py`: kích thước sau co giãn, tịnh tiến đúng pixel, công thức khung mở rộng, xoay 90° mở rộng khung trùng `np.rot90`, ma trận affine/phối cảnh đưa đúng điểm, ứng dụng tài liệu (sai số góc < 2 px).
- `test_reference.py`: ma trận tự tính so với `cv2` (sai số < 1e-6); ảnh tự cài đặt so với `cv2` cho resize, xoay, phối cảnh, 4 chế độ biên, ảnh màu (PSNR > 50 dB và < 1% pixel lệch quá 1 mức xám).
- `test_no_cv2_in_reference.py`: phân tích cú pháp mọi file trong `reference/` và chạy thử `reference/` trong tiến trình con đã chặn `cv2`.

## 3. Hướng dẫn demo

### Bước 1 — Sinh lại toàn bộ kết quả

```bash
python run_all.py
```

Chạy khoảng 25 s. In ra 4 bước, rồi tóm tắt Bảng 2 (PSNR, thời gian từng cờ nội suy) và Bảng 3 (PSNR, % pixel lệch, số lần chậm hơn của bản tự cài đặt). Kết quả ghi vào `results/figures/Hinh1–7.png` (DPI 200, nền trắng), `results/tables/Bang1–3.md/.csv` và `results/report_numbers.md`. Thời gian đo phụ thuộc máy và dao động nhẹ giữa các lần chạy; PSNR thì cố định.

### Bước 2 — Demo từng phép biến đổi

```bash
python demo/demo.py --op rotate --angle 45 --interp linear --border reflect --expand --image data/astronaut.png
```

Mở cửa sổ gồm ảnh gốc và ảnh OpenCV xoay 45°, khung mở rộng 725×725, biên phản xạ. Thêm `--compare` để hiện thêm ảnh tự cài đặt và ảnh hiệu, đồng thời in PSNR và thời gian:

```bash
python demo/demo.py --op rotate --angle 45 --interp linear --border reflect --expand --image data/astronaut.png --compare
```

Trên máy thử, lệnh này in PSNR khoảng 58.9 dB, khoảng 0.4% pixel lệch quá 1 mức xám, bản tự cài đặt chậm hơn khoảng 100 lần.

Tham số chính: `--op {scale,translate,rotate,affine,perspective}`, `--interp {nearest,linear,cubic,area,lanczos4}`, `--border {constant,replicate,reflect,reflect101,wrap}`, `--angle`, `--scale`, `--expand`, `--fx/--fy`, `--tx/--ty`, `--image`, `--save out.png`, `--no-show`. Bản tự cài đặt chỉ có nearest/linear/cubic; với `area` hoặc `lanczos4` thì `--compare` được bỏ qua và chương trình báo rõ. Ví dụ khác:

```bash
python demo/demo.py --op scale --fx 4 --interp cubic --image data/text.png --compare
```

```bash
python demo/demo.py --op perspective --interp linear --compare
```

### Bước 3 — Ứng dụng nắn chỉnh tài liệu

```bash
python demo/demo_document.py --image data/doc.jpg --auto
```

Tự tìm 4 góc (ngưỡng Otsu → contour lớn nhất → `approxPolyDP`; dự phòng bằng Canny), in 4 góc và ma trận H, hiện ảnh trước/sau. Với `data/doc.jpg` kết quả là trang thẳng 580×760. Bỏ `--auto` để tự click 4 góc: click trái 4 góc tờ giấy (thứ tự bất kỳ), Enter để xác nhận, `r` để chọn lại, Esc để thoát. Có thể dùng ảnh tài liệu thật, ví dụ đặt `data/doc_real1.jpg` (ảnh tên `doc_real*` cũng được thêm vào Hình 6 khi chạy `run_all.py`). Kích thước ảnh ra được ước lượng từ cạnh dài nhất của tứ giác nên tỉ lệ khung chỉ gần đúng; muốn đúng tỉ lệ A4 thì truyền `dsize` vào `rectify`.

### Bước 4 — Notebook

Mở `demo/demo.ipynb` (Jupyter/VS Code) và chạy lần lượt từ trên xuống dưới. Mỗi ô ứng với một mục báo cáo (Mục 1–8 như bảng ở phần 1) và gọi đúng các hàm mà `run_all.py` dùng. Ô Mục 7 mất khoảng 20–30 s.

## 4. Ghi chú kỹ thuật

- Quy ước tọa độ: tâm pixel nằm ở số nguyên, gốc ở góc trên-trái, tâm ảnh rộng w là (w − 1)/2. Góc dương = ngược chiều kim đồng hồ (giống `cv2.getRotationMatrix2D`).
- Xoay mở rộng khung: w' = ⌈w|cosθ| + h|sinθ|⌉, h' = ⌈w|sinθ| + h|cosθ|⌉, rồi cộng (w'−1)/2 − (w−1)/2 và (h'−1)/2 − (h−1)/2 vào cột tịnh tiến của ma trận.
- Bản tự cài đặt dùng bicubic Keys với a = −0.75 như OpenCV. Có tùy chọn `coord_bits` trong `remap_ref` để lượng tử hóa tọa độ về 1/32 pixel; chỉ dùng để kiểm chứng nguồn sai khác (xem `report_numbers.md`).
- Homography tự tính bằng DLT có chuẩn hóa Hartley [3]. Với tứ giác ngẫu nhiên, H có phần tử tới ~10⁴ nên Bảng 3 dùng sai số tương đối.

## 5. Tài liệu tham khảo

Đánh số trùng với báo cáo.

- [1] R. C. Gonzalez, R. E. Woods, *Digital Image Processing*, 3rd ed., Prentice Hall, 2008.
- [3] R. Hartley, A. Zisserman, *Multiple View Geometry in Computer Vision*, 2nd ed., Cambridge University Press, 2004.
- [5] R. Keys, "Cubic convolution interpolation for digital image processing," *IEEE Trans. Acoustics, Speech, and Signal Processing*, vol. 29, no. 6, pp. 1153–1160, 1981.
- [9] OpenCV documentation, "Geometric Image Transformations" (module imgproc), https://docs.opencv.org/4.10.0/da/d54/group__imgproc__transform.html
- [11] OpenCV source code, `modules/imgproc/src/imgwarp.cpp` và `resize.cpp`, https://github.com/opencv/opencv (tag 4.10.0).
