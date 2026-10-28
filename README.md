# Lab – Nhận diện cảm xúc khuôn mặt realtime bằng Deep Learning

**Sinh viên:** Trần Thanh Gia Khang – CE200292 – lớp AI2001 – FPT University Cần Thơ

Pipeline: IP camera điện thoại → phát hiện mặt (Haar Cascade) → tiền xử lý → phân loại cảm xúc (M1 hoặc M2) → vẽ bounding box + nhãn + confidence + FPS.

| Mô hình | Mô tả | Ảnh vào |
|---|---|---|
| M1 `cnn` | CNN tự thiết kế: 3 block (Conv–BN–ReLU ×2 → MaxPool → Dropout) + FC | 1×48×48, chuẩn hoá [-1, 1] |
| M2 `transfer` | MobileNetV2 pretrained ImageNet, fine-tune toàn mạng | 3×96×96, chuẩn hoá mean/std ImageNet |

## 1. Cài đặt

```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## 2. Huấn luyện (Google Colab)

1. Upload cả thư mục project lên Google Drive (hoặc Colab), mở `notebooks/train.ipynb`, chọn **Runtime → GPU**.
2. Chạy từ thư mục gốc project (cell đầu tự nhận cả khi cwd là `notebooks/`). Dataset FER-2013 (`msambare/fer2013`) được tải tự động bằng `kagglehub`; nếu lỗi, tải tay từ Kaggle rồi gán `DATA_DIR` (thư mục chứa `train/` và `test/`).
3. **Run all** (~20–40 phút). Notebook tạo `weights/cnn.pt`, `weights/transfer.pt` và `report_assets/` (biểu đồ, `results.json`).
4. Tải về `weights/`, `report_assets/` và notebook đã chạy (File → Download .ipynb, giữ nguyên output).

## 3. Chạy ứng dụng realtime

Dùng IP Webcam (Android) hoặc DroidCam: bấm *Start server*, ghi lại địa chỉ (vd. `http://192.168.1.15:8080`), mở thử trên trình duyệt laptop. Laptop và điện thoại phải cùng Wi-Fi (hoặc laptop bắt hotspot của điện thoại).

```bash
python src/app.py --source http://<ip>:8080/video --model cnn       # camera điện thoại, M1
python src/app.py --source test.jpg --model transfer                 # ảnh tĩnh, M2
python src/app.py --source video.mp4 --model cnn                     # file video
python src/app.py --source 0 --model cnn                             # webcam laptop
```

Tuỳ chọn: `--save out.jpg|out.mp4` (lưu kết quả), `--no-show` (không mở cửa sổ), `--config`, `--weights`. Nhấn **q** để thoát.
Nếu không truyền `--source`, chương trình lấy `camera.source` trong `config.yaml` (URL không hard-code trong code).

Xử lý mất kết nối: luồng camera đọc trong thread riêng; khi mất kết nối chương trình in *"Mất kết nối camera, đang thử lại..."*, hiện khung thông báo và tự kết nối lại, không crash.

## 4. Cấu trúc

```
├── README.md  requirements.txt  config.yaml
├── notebooks/train.ipynb     # huấn luyện + đánh giá M1, M2 (M1–M5)
├── src/model.py              # M1, M2
├── src/detector.py           # Haar Cascade
├── src/predict.py            # preprocess() dùng chung train/app + EmotionPredictor
├── src/app.py                # ứng dụng realtime
├── weights/                  # cnn.pt (M1), transfer.pt (M2)
├── report_assets/            # hình/biểu đồ cho báo cáo
└── report.pdf                # báo cáo (4 trang)
```

## 5. Kết quả (tập test FER-2013, 7.178 ảnh)

| Mô hình | Accuracy | Macro F1 | Tham số | FPS suy luận CPU (Colab, batch=1) |
|---|---|---|---|---|
| M1 CNN tự thiết kế | **64,50%** (≥60% ✓) | 0,608 | 3.507.015 | 95,9 |
| M2 MobileNetV2 | **66,68%** (≥65% ✓) | 0,652 | 2.232.839 | 108,5 |

Ablation M1 không dùng class weight: 67,36% / macro-F1 0,632. Chi tiết, đồ thị, confusion matrix và phân tích lỗi: `report.pdf`, `report_assets/`, `notebooks/train.ipynb` (đã chạy, còn output).

FPS của cả ứng dụng (đọc luồng + detect + dự đoán + hiển thị) đo trên MacBook Pro 13-inch (chip Apple M1, RAM 16 GB, CPU), camera iPhone qua Wi-Fi bằng app IP Camera Lite for iOS: **M1 ≈ 28,9 FPS**, **M2 ≈ 20,2 FPS** (yêu cầu ≥10). Ảnh demo: `report_assets/demo_*.jpg`.

## 6. Xử lý sự cố

- Không xem được video trên trình duyệt → kiểm tra cùng Wi-Fi, tắt VPN/firewall chặn cổng 8080/4747.
- Trễ hình → giảm độ phân giải trong app IP camera xuống 640×480.
- Thiếu `weights/*.pt` → chạy notebook huấn luyện trước.

## 7. Ghi chú về nguồn video

- URL luồng phụ thuộc app camera (Android: IP Webcam thường là `http://<ip>:8080/video`; iPhone: IP Camera Lite thường là `http://<ip>:8081/video`). Hãy dùng đúng địa chỉ app hiển thị.
- Nếu app yêu cầu đăng nhập: `--source "http://user:pass@<ip>:8081/video"` (không ghi mật khẩu vào `config.yaml`).
