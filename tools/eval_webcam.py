"""Đo độ chính xác trên ảnh/clip tự thu, thử nhiều crop-scale và ensemble.
Cấu trúc dữ liệu:  webcam_eval/<angry|disgust|fear|happy|neutral|sad|surprise>/*.jpg|png|mp4
  python tools/eval_webcam.py --data webcam_eval --weights weights/cnn.pt weights/transfer.pt
  python tools/eval_webcam.py --data webcam_eval --weights weights/transfer.pt weights/transfer_raf.pt --crop-scales 0.85 0.9
Với >= 2 file trọng số cùng danh sách lớp, in thêm dòng 'ensemble' (trung bình softmax).
Cột: tỉ lệ detect được mặt | accuracy trên ảnh có mặt | accuracy trên TẤT CẢ ảnh (không detect = sai) | macro-F1."""
import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import yaml
from sklearn.metrics import f1_score

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from app import scale_box                    # noqa: E402
from detector import HaarFaceDetector        # noqa: E402
from predict import EmotionPredictor         # noqa: E402

RAF_MAP = {"1": "surprise", "2": "fear", "3": "disgust", "4": "happy", "5": "sad", "6": "angry", "7": "neutral"}
IMG = {".jpg", ".jpeg", ".png", ".bmp"}
VID = {".mp4", ".mov", ".avi", ".mkv"}


def iter_frames(path, step):
    if path.suffix.lower() in IMG:
        f = cv2.imread(str(path))
        if f is not None:
            yield f
    elif path.suffix.lower() in VID:
        cap, i = cv2.VideoCapture(str(path)), 0
        while True:
            ok, f = cap.read()
            if not ok:
                break
            if i % step == 0:
                yield f
            i += 1
        cap.release()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--weights", nargs="+", required=True)
    ap.add_argument("--crop-scales", nargs="+", type=float, default=None)
    ap.add_argument("--raf", action="store_true",
                    help="--data là DATASET/test của RAF-DB (thư mục 1..7): coi cả ảnh là khuôn mặt, không chạy detector")
    ap.add_argument("--step", type=int, default=5, help="với video: lấy mỗi N khung")
    ap.add_argument("--config", default=str(ROOT / "config.yaml"))
    a = ap.parse_args()

    d = yaml.safe_load(open(a.config, encoding="utf-8"))["detector"]
    det = HaarFaceDetector(d["scale_factor"], d["min_neighbors"], d["min_size"], d["detect_width"])
    preds = [EmotionPredictor(w) for w in a.weights]
    classes = preds[0].classes

    # detect 1 lần cho mỗi ảnh (lấy mặt lớn nhất); lưu ảnh xám + khung
    scales = a.crop_scales or ([1.0] if a.raf else [0.8, 0.85, 0.9, 1.0])
    folder = {v: k for k, v in RAF_MAP.items()} if a.raf else {c: c for c in classes}
    items = []          # (nhãn thật, ảnh xám, khung hoặc None)
    for c in classes:
        sub = Path(a.data) / folder[c]
        for p in sorted(sub.glob("*")) if sub.is_dir() else []:
            if a.raf:   # ảnh RAF-DB đã là mặt căn chỉnh: không detect, khung = cả ảnh
                g = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
                if g is not None:
                    items.append((classes.index(c), g, (0, 0, g.shape[1], g.shape[0])))
                continue
            for f in iter_frames(p, a.step):
                boxes, gray = det.detect(f)
                items.append((classes.index(c), gray, max(boxes, key=lambda b: b[2] * b[3]) if boxes else None))
    if not items:
        sys.exit(f"Không có ảnh trong {a.data}/<tên lớp>/")
    y_all = np.array([t for t, _, _ in items])
    print("Số ảnh mỗi lớp:", {c: int((y_all == i).sum()) for i, c in enumerate(classes)})

    print(f"\n{'cấu hình':38s} {'detect':>7s} {'acc(có mặt)':>12s} {'acc(tất cả)':>12s} {'macro-F1':>9s}")
    for s in scales:
        idx, faces = [], []
        for i, (t, gray, box) in enumerate(items):
            if box is None:
                continue
            x, y, w, h = scale_box(box, s, gray.shape)
            if w > 0 and h > 0:
                idx.append(i)
                faces.append(gray[y:y + h, x:x + w])
        probs = [p.proba_batch(faces) for p in preds]
        runs = list(zip([Path(w).stem for w in a.weights], probs))
        if len(preds) > 1:
            runs.append(("ensemble", np.mean(probs, axis=0)))
        for name, pr in runs:
            yp, yt = pr.argmax(1), y_all[idx]
            correct = int((yp == yt).sum())
            print(f"{name + ' | crop ' + str(s):38s} {len(idx) / len(items):7.1%} {correct / max(len(idx), 1):12.1%} "
                  f"{correct / len(items):12.1%} {f1_score(yt, yp, average='macro'):9.3f}")


if __name__ == "__main__":
    main()
