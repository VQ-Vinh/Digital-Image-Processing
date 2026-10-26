# Số liệu cho báo cáo (sinh tự động bởi run_all.py)

## Cấu hình máy và phần mềm

- CPU: AMD Ryzen 7 6800H with Radeon Graphics (8 nhân / 16 luồng)
- RAM: 15.2 GB
- Hệ điều hành: Windows-11-10.0.26300-SP0
- Python: 3.13.15
- NumPy: 2.5.4
- OpenCV: 4.10.0
- Matplotlib: 3.11.2
- scikit-image: 0.26.0
- Số luồng OpenCV (cv2.getNumThreads): 16
- Đo thời gian: trung bình 10 lần, bỏ 1 lần warm-up (time.perf_counter).

## Số liệu chính

### Bảng 2 — cờ nội suy cv2.resize

- Ảnh 512×512: PSNR cao nhất INTER_AREA ↓ + INTER_CUBIC ↑ (26.38 dB), thấp nhất INTER_NEAREST (21.36 dB); thu nhỏ nhanh nhất INTER_NEAREST (0.009 ms), chậm nhất INTER_LANCZOS4 (0.272 ms).
- Ảnh 1920×1080: PSNR cao nhất INTER_LANCZOS4 (34.40 dB), thấp nhất INTER_NEAREST (25.42 dB); thu nhỏ nhanh nhất INTER_NEAREST (0.064 ms), chậm nhất INTER_LANCZOS4 (1.211 ms).

### Bảng 3 — tự cài đặt so với OpenCV

- resize ×4 — nearest: PSNR ∞ dB, lệch >1 0.000%, chậm 146×, linear: PSNR 58.62 dB, lệch >1 0.000%, chậm 361×, cubic: PSNR 101.35 dB, lệch >1 0.000%, chậm 1412×.
- xoay 45° (giữ khung) — nearest: PSNR ∞ dB, lệch >1 0.000%, chậm 70×, linear: PSNR 59.61 dB, lệch >1 0.133%, chậm 93×, cubic: PSNR 58.95 dB, lệch >1 0.200%, chậm 175×.
- affine 3 điểm — nearest: PSNR 59.80 dB, lệch >1 0.008%, chậm 69×, linear: PSNR 60.03 dB, lệch >1 0.243%, chậm 77×, cubic: PSNR 59.44 dB, lệch >1 0.285%, chậm 209×.
- phối cảnh 4 điểm — nearest: PSNR ∞ dB, lệch >1 0.000%, chậm 64×, linear: PSNR 58.57 dB, lệch >1 0.213%, chậm 81×, cubic: PSNR 57.90 dB, lệch >1 0.282%, chậm 152×.
- Ma trận xoay (ngẫu nhiên): sai số tuyệt đối max 0.00e+00, tương đối max 0.00e+00; sai số chiếu lại max: tự tính 0.00e+00 px, OpenCV 0.00e+00 px.
- Ma trận affine (ngẫu nhiên): sai số tuyệt đối max 1.14e-13, tương đối max 3.27e-16; sai số chiếu lại max: tự tính 3.41e-13 px, OpenCV 2.27e-13 px.
- Ma trận homography (ngẫu nhiên): sai số tuyệt đối max 1.81e-01, tương đối max 6.93e-06; sai số chiếu lại max: tự tính 5.68e-13 px, OpenCV 4.05e-05 px.
- Lượng tử 1/32 px, xoay 45°, bilinear: PSNR 59.61 dB → 69.73 dB.
- Lượng tử 1/32 px, phối cảnh, bilinear: PSNR 58.57 dB → ∞ dB.
- Lượng tử 1/32 px, xoay 45°, bicubic (a = −0.75): PSNR 58.95 dB → 68.60 dB.
- Lượng tử 1/32 px, phối cảnh, bicubic (a = −0.75): PSNR 57.90 dB → 82.40 dB.

## Nhận xét gợi ý theo từng hình

### Hình 1

- Mọi phép biến đổi hình học trong OpenCV đều theo hai bước: tính ma trận (2×3 cho affine, 3×3 cho phối cảnh) rồi gọi hàm warp để ánh xạ ngược từng pixel đích về ảnh nguồn và nội suy.
- Riêng cv2.resize không cần ma trận tường minh: ma trận co giãn được suy ra từ dsize hoặc fx, fy.
- Chất lượng và tốc độ được quyết định bởi các tham số interpolation, borderMode và dsize, độc lập với cách tính ma trận.
- Nhánh nét đứt là phần đối chứng: cài đặt lại cùng quy trình bằng NumPy để kiểm chứng cơ chế bên trong.

### Hình 2

- Khi phóng to ×4 ảnh 128×128, INTER_LANCZOS4 cho PSNR cao nhất (26.45 dB), INTER_NEAREST thấp nhất (25.17 dB); chênh lệch 1.28 dB cho thấy nội suy bậc cao khôi phục chi tiết tốt hơn nhưng không thể tạo lại chi tiết đã mất khi thu nhỏ.
- Khi phóng to với hệ số nguyên, INTER_AREA cho kết quả trùng INTER_NEAREST (cùng PSNR 25.17 dB) — đúng như tài liệu OpenCV mô tả, nên không dùng INTER_AREA để phóng to.
- Vùng phóng lớn cho thấy INTER_NEAREST tạo khối vuông (răng cưa), INTER_LINEAR làm nhòe biên, INTER_CUBIC và INTER_LANCZOS4 giữ biên sắc hơn nhưng có thể xuất hiện viền sáng/tối (ringing).
- Khi thu nhỏ ×0.25 zone plate, độ lệch chuẩn ở vùng có tần số vượt Nyquist (lý tưởng ≈ 0) nhỏ nhất với INTER_AREA (20.1) và lớn nhất với INTER_NEAREST (90.2), tức INTER_AREA giảm vân moiré khoảng 4.5 lần nhờ lấy trung bình cả vùng pixel.
- Các cờ còn lại chỉ lấy mẫu vài pixel lân cận nên bỏ sót phần lớn pixel nguồn: chữ mảnh bị đứt nét và vân tròn giả xuất hiện ở vùng tần số cao.

### Hình 3

- Tịnh tiến giữ khung 600×400 làm mất phần ảnh trượt ra ngoài: chỉ còn 70.8% ảnh gốc, phần trống được lấp bằng borderValue.
- Xoay giữ khung làm mất góc ảnh: còn 82.0% ở 30°, 77.9% ở 45°, 66.7% ở 90°; với ảnh không vuông, góc 90° mất nhiều nhất.
- Xoay mở rộng khung tính lại kích thước (ví dụ 45° → 708×708, diện tích gấp 2.09 lần) và cộng bù tịnh tiến vào cột cuối của ma trận; nếu quên bù tịnh tiến, ảnh bị lệch và vẫn bị cắt như ô đầu hàng thứ hai.
- Với borderMode: CONSTANT tô màu cố định, REPLICATE kéo dài pixel biên thành vệt sọc, REFLECT/REFLECT_101 tạo ảnh phản chiếu liền mạch, WRAP lặp ảnh tuần hoàn nên có đường nối đột ngột.

### Hình 4

- Ba cặp điểm cho 6 phương trình, vừa đủ xác định 6 tham số của ma trận affine 2×3.
- Sai số lớn nhất khi chiếu P1–P3 qua M so với Q1–Q3 là 2.8e-14 px, chỉ còn sai số làm tròn.
- det của phần tuyến tính M[:, :2] = 0.738, tức diện tích ảnh sau biến đổi bằng 73.8% diện tích ban đầu; tịnh tiến (51.5, 118.5) px.
- Ảnh bàn cờ cho thấy affine giữ đường thẳng và tính song song nhưng làm các ô vuông thành hình bình hành.

### Hình 5

- Bốn cặp điểm cho 8 phương trình, đủ xác định 8 bậc tự do của homography 3×3 (chuẩn hóa h33 = 1).
- Hàng cuối H có h31 = -3.72e-04, h32 = -1.01e-03 khác 0 nên mẫu số w thay đổi theo vị trí — đây là khác biệt cốt lõi so với affine.
- Sai số chiếu lại 4 điểm là 1.1e-13 px.
- Cạnh trên và cạnh dưới vốn song song, sau biến đổi có độ dài tỉ lệ 1.99 : 1 và không còn song song; bàn cờ cho thấy đường thẳng vẫn thẳng nhưng các ô co dần về phía xa.

### Hình 6

- Trên ảnh tổng hợp 1280×960, thuật toán tự tìm 4 góc (Otsu + contour + approxPolyDP) với sai số trung bình 0.60 px so với góc thật.
- Ảnh sau nắn chỉnh 600×848 đạt PSNR 17.38 dB so với trang gốc; dùng 4 góc thật thì đạt 20.82 dB — phần chênh do sai số góc, phần còn lại do bóng đổ, nhiễu và nội suy khi tạo ảnh.
- Thời gian tìm góc 2.84 ms, nắn chỉnh 2.09 ms, tổng dưới 33 ms/khung nên đủ nhanh cho video thời gian thực (30 khung/s).
- Chữ và bảng kẻ ô sau nắn chỉnh thẳng hàng trở lại, chứng tỏ homography khử được méo phối cảnh.

### Hình 7

- Với nội suy bilinear, ảnh tự cài đặt và ảnh OpenCV gần như trùng nhau: PSNR 58.57–60.03 dB, tỉ lệ pixel lệch >1 mức xám chỉ 0.000–0.243%.
- Ảnh hiệu tập trung ở biên vật thể và mép khung ảnh — nơi gradient lớn nên lệch tọa độ rất nhỏ cũng đổi giá trị pixel; vùng phẳng gần như bằng 0.
- Bản NumPy chậm hơn OpenCV 64–1412 lần (OpenCV dùng SIMD, đa luồng, bảng hệ số tính sẵn và số nguyên dấu phẩy cố định).
- Khi lượng tử hóa tọa độ của bản tự cài đặt về 1/32 pixel như OpenCV, PSNR tăng thêm 9.7–24.5 dB và trùng khớp hoàn toàn ở: phối cảnh, bilinear — xác nhận lượng tử hóa tọa độ là nguồn sai khác chính.

## Nguyên nhân sai khác giữa tự cài đặt và OpenCV

Bản tự cài đặt và OpenCV dùng cùng một công thức ánh xạ ngược, nên sai khác chỉ đến từ chi tiết số học. (1) Quy ước tâm pixel: cả hai coi tâm pixel nằm ở tọa độ nguyên; với cv2.resize, tọa độ nguồn là x_s = (x + 0.5)/f − 0.5 (căn tâm pixel), nếu dùng x_s = x/f thì ảnh lệch nửa pixel. Tâm xoay của ảnh rộng w là (w − 1)/2 chứ không phải w/2. (2) Lượng tử hóa tọa độ: trong warpAffine/warpPerspective, OpenCV tính tọa độ nguồn bằng số nguyên dấu phẩy cố định rồi làm tròn về lưới 1/32 pixel (INTER_BITS = 5, INTER_TAB_SIZE = 32); bản tự cài đặt dùng tọa độ float64 nên lệch tới 1/64 pixel. (3) Hệ số nội suy dấu phẩy cố định: với ảnh uint8, OpenCV tra bảng hệ số tính sẵn cho 32 vị trí lẻ, nhân với 2^15 (INTER_REMAP_COEF_BITS) trong warp hoặc 2^11 (INTER_RESIZE_COEF_BITS) trong resize, cộng dồn số nguyên rồi dịch bit và bão hòa về [0, 255]; bản tự cài đặt cộng dồn số thực rồi mới làm tròn. (4) Nội suy bậc ba dùng nhân Keys với a = −0.75 ở cả hai bên (Keys [5] đề xuất a = −0.5), nên khác biệt chỉ do làm tròn hệ số. Theo Bảng 3, với bilinear và bicubic tỉ lệ pixel lệch quá 1 mức xám lớn nhất là 0.285% và độ lệch lớn nhất chỉ 4 mức xám, tập trung ở biên có gradient lớn. Với nearest, nhiều nhất 0.008% pixel lệch nhưng mỗi pixel lệch có thể khác tới 117 mức xám: khi tọa độ nguồn rơi gần đúng giữa hai pixel, hai cách làm tròn chọn hai pixel kề nhau khác nhau.
