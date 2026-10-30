# Lab – Nhận diện cảm xúc khuôn mặt realtime bằng Deep Learning

**Sinh viên:** Trần Thanh Gia Khang – CE200292 – lớp AI2001 – FPT University Cần Thơ

Pipeline: IP camera điện thoại → phát hiện mặt (Haar Cascade) → tiền xử lý → phân loại cảm xúc (M1 hoặc M2) → vẽ bounding box + nhãn + confidence + FPS.

| Mô hình | Mô tả | Ảnh vào | Trọng số |
|---|---|---|---|
| M1 `cnn` | CNN tự thiết kế: 3 block (Conv–BN–ReLU ×2 → MaxPool → Dropout) + FC | 1×48×48, chuẩn hoá [-1, 1] | `weights/cnn.pt` (FER-2013) |
| M2 `transfer` | MobileNetV2 pretrained ImageNet, fine-tune toàn mạng | 3×96×96, chuẩn hoá mean/std ImageNet | `weights/transfer.pt` (FER-2013) |
| M1-R | M1 fine-tune thêm RAF-DB | như M1 | `weights/cnn_raf.pt` |
| M2-R | M2 fine-tune thêm RAF-DB (**dùng khi demo**) | như M2 | `weights/transfer_raf.pt` |

`config.yaml` đang trỏ `transfer` → `weights/transfer_raf.pt` (M2-R) và `cnn` → `weights/cnn.pt`. Muốn dùng bản khác: sửa `config.yaml` hoặc truyền `--weights`.

## 1. Cài đặt

```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## 2. Huấn luyện (Google Colab)

### 2.1. M1, M2 trên FER-2013
1. Upload cả thư mục project lên Google Drive (hoặc Colab), mở `notebooks/train.ipynb`, chọn **Runtime → GPU**.
2. Chạy từ thư mục gốc project (cell đầu tự nhận cả khi cwd là `notebooks/`). Dataset FER-2013 (`msambare/fer2013`) được tải tự động bằng `kagglehub`; nếu lỗi, tải tay từ Kaggle rồi gán `DATA_DIR` (thư mục chứa `train/` và `test/`).
3. **Run all** (~20–40 phút). Notebook tạo `weights/cnn.pt`, `weights/transfer.pt` và `report_assets/` (biểu đồ, `results.json`).
4. Tải về `weights/`, `report_assets/` và notebook đã chạy (File → Download .ipynb, giữ nguyên output).

### 2.2. Fine-tune thêm RAF-DB (M1-R, M2-R)
Dữ liệu ngoài: **RAF-DB** (Li, Deng, Du, CVPR 2017), bản *basic* với ảnh mặt căn chỉnh 100×100, 7 lớp, tải từ Kaggle ([shuvoalok/raf-db-dataset](https://www.kaggle.com/datasets/shuvoalok/raf-db-dataset)): 12.271 ảnh train + 3.068 ảnh test. **Dataset không nằm trong repo** (giấy phép); cần tải riêng, cấu trúc sau khi giải nén: `DATASET/{train,test}/1..7/*.jpg` (1=surprise, 2=fear, 3=disgust, 4=happy, 5=sad, 6=angry, 7=neutral).

Mở `notebooks/finetune_raf.ipynb` trên Colab (GPU) hoặc chạy trực tiếp:

```bash
unzip -q RAF-DB.zip -d /content/raf
python tools/finetune_raf.py --check-labels --raf-dir /content/raf/DATASET        # xem raf_check.png để kiểm tra map nhãn
python tools/finetune_raf.py --fer-dir <thư mục FER-2013 chứa train/ test/> --raf-dir /content/raf/DATASET \
       --init weights/transfer.pt --out weights/transfer_raf.pt                   # M2-R
python tools/finetune_raf.py --fer-dir <...> --raf-dir /content/raf/DATASET \
       --init weights/cnn.pt --out weights/cnn_raf.pt --min-fer-acc 0.60           # M1-R
```
Script in bảng số ảnh theo lớp, accuracy/macro-F1 trước và sau fine-tune trên cả hai tập test, và báo đạt/không đạt ngưỡng của đề. Chi tiết siêu tham số: `report.pdf`, mục 6.

Đánh giá một hoặc nhiều trọng số trên test RAF-DB (không cần detector):

```bash
python tools/eval_webcam.py --raf --data <DATASET>/test --weights weights/transfer.pt weights/transfer_raf.pt
```
(Không có `--raf`, script đọc thư mục `<lớp>/ảnh|video` tự thu và chạy cả Haar detector.)

## 3. Chạy ứng dụng realtime

Dùng IP Webcam (Android) hoặc DroidCam: bấm *Start server*, ghi lại địa chỉ (vd. `http://192.168.1.15:8080`), mở thử trên trình duyệt laptop. Laptop và điện thoại phải cùng Wi-Fi (hoặc laptop bắt hotspot của điện thoại).

```bash
python src/app.py --source http://<ip>:8080/video --model cnn       # camera điện thoại, M1
python src/app.py --source test.jpg --model transfer                 # ảnh tĩnh, M2 (trọng số trong config.yaml)
python src/app.py --source video.mp4 --model cnn                     # file video
python src/app.py --source 0 --model cnn                             # webcam laptop
python src/app.py --source http://<ip>:8081/video --model transfer --weights weights/transfer.pt   # dùng bản FER-2013 gốc
```

Tuỳ chọn: `--save out.jpg|out.mp4` (lưu kết quả), `--no-show` (không mở cửa sổ), `--config`, `--weights`. Nhấn **q** để thoát.

Hai tuỳ chọn giúp dự đoán realtime ổn định hơn (mặc định tắt; đặt sẵn trong `config.yaml`: `smoothing_frames`, `face_crop_scale`):
- `--smooth N`: **làm mượt theo thời gian**, lấy trung bình xác suất softmax của N khung gần nhất cho từng khuôn mặt (ghép theo vị trí giữa các khung). Nhãn bớt nhấp nháy, đổi lại phản ứng chậm hơn ~N khung. Gợi ý N = 5–8. Không áp dụng cho ảnh tĩnh.
- `--crop-scale S`: **thu nhỏ khung cắt mặt** quanh tâm khung Haar (1.0 = nguyên, 0.85 = cắt chặt hơn). Khung Haar thường bao cả trán và cằm, trong khi ảnh huấn luyện là mặt cắt sát; khung vẽ ra chính là vùng mô hình nhìn thấy.

```bash
python src/app.py --source http://<ip>:8081/video --model transfer --smooth 6 --crop-scale 0.85
```
Nếu không truyền `--source`, chương trình lấy `camera.source` trong `config.yaml` (URL không hard-code trong code).

Xử lý mất kết nối: luồng camera đọc trong thread riêng; khi mất kết nối chương trình in *"Mất kết nối camera, đang thử lại..."*, hiện khung thông báo và tự kết nối lại, không crash.

## 4. Cấu trúc

```
├── README.md  requirements.txt  config.yaml  .gitignore
├── notebooks/train.ipynb         # huấn luyện + đánh giá M1, M2 trên FER-2013 (M1–M5)
├── notebooks/finetune_raf.ipynb  # fine-tune M1-R, M2-R với RAF-DB (đã chạy, còn output)
├── src/model.py                  # M1, M2
├── src/detector.py               # Haar Cascade
├── src/predict.py                # preprocess() dùng chung train/app + EmotionPredictor
├── src/app.py                    # ứng dụng realtime
├── tools/finetune_raf.py         # fine-tune trên FER-2013 + RAF-DB
├── tools/eval_webcam.py          # đánh giá trên RAF-DB test hoặc ảnh/clip tự thu
├── weights/                      # cnn.pt, transfer.pt (FER-2013); cnn_raf.pt, transfer_raf.pt (+ .json log) (RAF-DB)
├── report_assets/                # hình/biểu đồ cho báo cáo
└── report.pdf                    # báo cáo (5 trang)
```

## 5. Kết quả

**Tập test FER-2013 (7.178 ảnh) và test RAF-DB (3.068 ảnh):**

| Mô hình | FER Acc | FER Macro-F1 | RAF Acc | RAF Macro-F1 | Tham số |
|---|---|---|---|---|---|
| M1 CNN (FER-2013) | **64,50%** (≥60% ✓) | 0,608 | 63,07% | 0,466 | 3.507.015 |
| M1-R (+RAF-DB) | 62,93% (≥60% ✓) | 0,586 | 71,71% | 0,617 | 3.507.015 |
| M2 MobileNetV2 (FER-2013) | **66,68%** (≥65% ✓) | 0,652 | 58,83% | 0,403 | 2.232.839 |
| **M2-R (+RAF-DB)** | **67,23%** (≥65% ✓) | 0,646 | **76,69%** | **0,683** | 2.232.839 |

Lưu ý: mức tăng trên RAF-DB là kỳ vọng vì train và test cùng phân phối; FER-2013 của M2-R không giảm. FPS suy luận CPU (Colab, batch=1) của kiến trúc M1 / M2: 95,9 / 108,5.

Ablation M1 không dùng class weight: 67,36% / macro-F1 0,632. Chi tiết, đồ thị, confusion matrix và phân tích lỗi: `report.pdf`, `report_assets/`, `notebooks/train.ipynb`, `notebooks/finetune_raf.ipynb` (đã chạy, còn output).

FPS của cả ứng dụng (đọc luồng + detect + dự đoán + hiển thị) đo trên MacBook Pro 13-inch (chip Apple M1, RAM 16 GB, CPU), camera iPhone qua Wi-Fi bằng app IP Camera Lite for iOS, **M1 ≈ 28,9 FPS**, **M2 ≈ 20,2 FPS**, **M2-R ≈ 24,8 FPS** (yêu cầu ≥10). M2 và M2-R cùng kiến trúc nên chênh lệch là dao động của một lần đọc; M1-R chưa đo riêng. Ảnh demo: `report_assets/demo_*.jpg`.

## 6. Xử lý sự cố

- Không xem được video trên trình duyệt → kiểm tra cùng Wi-Fi, tắt VPN/firewall chặn cổng 8080/4747.
- Trễ hình → giảm độ phân giải trong app IP camera xuống 640×480.
- Thiếu `weights/*.pt` → chạy notebook huấn luyện trước (hoặc tải trọng số theo link nộp bài).
- `finetune_raf.py` báo không thấy ảnh → `--raf-dir` phải trỏ tới thư mục `DATASET` (chứa `train/` và `test/`); `--fer-dir` phải trỏ tới thư mục chứa `train/` và `test/` của FER-2013.

## 7. Ghi chú về nguồn video

- URL luồng phụ thuộc app camera (Android: IP Webcam thường là `http://<ip>:8080/video`; iPhone: IP Camera Lite thường là `http://<ip>:8081/video`). Hãy dùng đúng địa chỉ app hiển thị.
- Nếu app yêu cầu đăng nhập: `--source "http://user:pass@<ip>:8081/video"` (không ghi mật khẩu vào `config.yaml`).
