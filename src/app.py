"""Ứng dụng nhận diện cảm xúc realtime.
Ví dụ:
  python src/app.py --source http://192.168.1.15:8080/video --model cnn
  python src/app.py --source test.jpg --model transfer
  python src/app.py --source video.mp4 --model cnn
  python src/app.py --source 0 --model cnn        (webcam laptop)
Nhấn 'q' để thoát."""
import argparse
import os
import re
import sys
import threading
import time
from pathlib import Path

os.environ.setdefault("OPENCV_LOG_LEVEL", "SILENT")      # tắt log lặp lại của OpenCV khi mất kết nối
os.environ.setdefault("OPENCV_FFMPEG_LOGLEVEL", "-8")
import cv2
import numpy as np
import yaml

from detector import HaarFaceDetector
from predict import EmotionPredictor

ROOT = Path(__file__).resolve().parent.parent
IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
COLORS = {  # BGR
    "angry": (0, 0, 255), "disgust": (0, 128, 0), "fear": (128, 0, 128), "happy": (0, 200, 255),
    "neutral": (200, 200, 200), "sad": (255, 100, 0), "surprise": (0, 165, 255),
}


def parse_args():
    ap = argparse.ArgumentParser(description="Nhận diện cảm xúc khuôn mặt realtime")
    ap.add_argument("--source", default=None, help="URL IP camera | chỉ số webcam | file ảnh | file video "
                                                    "(mặc định: camera.source trong config.yaml)")
    ap.add_argument("--model", choices=["cnn", "transfer"], default="cnn", help="cnn = M1, transfer = M2")
    ap.add_argument("--config", default=str(ROOT / "config.yaml"))
    ap.add_argument("--weights", default=None, help="ghi đè đường dẫn trọng số trong config")
    ap.add_argument("--save", default=None, help="lưu kết quả (ảnh: .jpg/.png, video: .mp4)")
    ap.add_argument("--no-show", action="store_true", help="không mở cửa sổ (chạy không có màn hình)")
    return ap.parse_args()


def hide_password(src):
    """http://user:pass@host -> http://***@host (không in mật khẩu ra màn hình)."""
    return re.sub(r"//[^/@]*@", "//***@", str(src))


def kind_is_path(src):
    s = str(src)
    return not s.isdigit() and not s.lower().startswith(("http://", "https://", "rtsp://"))


def source_kind(src):
    s = str(src)
    if s.isdigit():
        return "webcam"
    if s.lower().startswith(("http://", "https://", "rtsp://")):
        return "stream"
    return "image" if Path(s).suffix.lower() in IMG_EXT else "video"


class StreamReader:
    """Đọc IP camera/webcam trong thread riêng: luôn giữ khung hình MỚI NHẤT (giảm độ trễ)
    và tự kết nối lại khi mất kết nối (chương trình chính không bị treo hay crash)."""

    def __init__(self, src, retry_sec=2.0):
        self.src = int(src) if str(src).isdigit() else src
        self.retry_sec = retry_sec
        self.frame, self.frame_id, self.connected = None, 0, False
        self.stopped = False
        self.lock = threading.Lock()
        threading.Thread(target=self._run, daemon=True).start()

    def _set(self, frame, connected):
        with self.lock:
            self.connected = connected
            if frame is not None:
                self.frame, self.frame_id = frame, self.frame_id + 1

    def _run(self):
        cap = None
        while not self.stopped:
            if cap is None:
                cap = cv2.VideoCapture(self.src)
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                if not cap.isOpened():
                    cap.release()
                    cap = None
                    self._set(None, False)
                    time.sleep(self.retry_sec)
                    continue
            ok, frame = cap.read()
            if ok:
                self._set(frame, True)
            else:  # mất kết nối: đóng lại, vòng lặp sau sẽ mở lại
                cap.release()
                cap = None
                self._set(None, False)
                time.sleep(0.5)
        if cap is not None:
            cap.release()

    def read(self):
        """Trả về (trạng thái, frame, id): 'ok' | 'lost'."""
        with self.lock:
            if not self.connected or self.frame is None:
                return "lost", None, self.frame_id
            return "ok", self.frame, self.frame_id

    def release(self):
        self.stopped = True


class FileReader:
    """Đọc file video tuần tự (không bỏ khung hình)."""

    def __init__(self, path):
        self.cap = cv2.VideoCapture(path)
        if not self.cap.isOpened():
            raise IOError(f"Không mở được video: {path}")

    def read(self):
        ok, frame = self.cap.read()
        return ("ok", frame, 0) if ok else ("end", None, 0)

    def release(self):
        self.cap.release()


def annotate(frame, detector, predictor, thr):
    """Phát hiện mặt, dự đoán cảm xúc cho TẤT CẢ mặt trong khung hình, vẽ kết quả. Trả về số mặt."""
    boxes, gray = detector.detect(frame)
    crops = [gray[y:y + h, x:x + w] for x, y, w, h in boxes]
    keep = [i for i, c in enumerate(crops) if c.size > 0]
    results = predictor.predict_batch([crops[i] for i in keep])
    for i, (label, conf) in zip(keep, results):
        x, y, w, h = boxes[i]
        color = COLORS.get(label, (255, 255, 255)) if conf >= thr else (150, 150, 150)  # xám = kém tin cậy
        text = f"{label.capitalize()}{'' if conf >= thr else '?'} {conf * 100:.0f}%"
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        cv2.rectangle(frame, (x, max(0, y - 24)), (x + 12 * len(text) + 6, y), color, -1)
        cv2.putText(frame, text, (x + 3, max(14, y - 7)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
    return len(keep)


def message_frame(text, size=(480, 640)):
    img = np.zeros((*size, 3), np.uint8)
    cv2.putText(img, text, (20, size[0] // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    return img


def main():
    args = parse_args()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    cfg = yaml.safe_load(open(args.config, encoding="utf-8"))

    source = args.source if args.source is not None else cfg["camera"]["source"]
    source = os.path.expanduser(source) if kind_is_path(source) else source
    weights = Path(args.weights or cfg["models"][args.model])
    weights = weights if weights.is_absolute() else ROOT / weights
    if not weights.exists():
        sys.exit(f"Không tìm thấy trọng số: {weights}. Hãy huấn luyện (notebooks/train.ipynb) và chép vào weights/.")

    d = cfg["detector"]
    detector = HaarFaceDetector(d["scale_factor"], d["min_neighbors"], d["min_size"], d["detect_width"])
    predictor = EmotionPredictor(weights)
    thr = cfg["confidence_threshold"]
    kind = source_kind(source)
    print(f"Model: {args.model} | nguồn: {hide_password(source)} ({kind})")

    # ---- Chế độ ảnh tĩnh ----
    if kind == "image":
        if not Path(source).exists():
            sys.exit(f"Không tìm thấy file ảnh: {source}\n"
                     f"  (thư mục hiện tại: {Path.cwd()}). Hãy đặt ảnh vào đây hoặc dùng đường dẫn đầy đủ, "
                     "ví dụ: --source ~/Desktop/anh.jpg")
        frame = cv2.imread(str(source))
        if frame is None:
            sys.exit(f"Không đọc được ảnh (file hỏng hoặc sai định dạng): {source}")
        n = annotate(frame, detector, predictor, thr)
        print(f"Phát hiện {n} khuôn mặt." if n else "Không phát hiện khuôn mặt nào.")
        if args.save:
            cv2.imwrite(args.save, frame)
        if not args.no_show:
            cv2.imshow("Emotion", frame)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        return

    # ---- Chế độ camera / video ----
    try:
        reader = FileReader(source) if kind == "video" else StreamReader(source)
    except IOError as e:
        sys.exit(str(e))
    writer, fps, last_id, was_lost, t_prev = None, 0.0, -1, False, time.time()
    t_lost = time.time()   # thời điểm bắt đầu không có hình (để chỉ báo lỗi sau vài giây)
    ever_connected = False
    try:
        while True:
            status, frame, fid = reader.read()
            if status == "end":
                print("Hết video.")
                break
            if status == "lost":
                if not was_lost and time.time() - t_lost > 4:   # chờ 4 giây rồi mới coi là lỗi
                    print(f"Không kết nối được / mất kết nối camera: {hide_password(source)}\n"
                          "  -> Mở đúng URL này trên trình duyệt của laptop: nếu không thấy hình thì app điện thoại "
                          "chưa chạy, sai địa chỉ/đường dẫn/cổng, khác Wi-Fi, hoặc app yêu cầu đăng nhập "
                          "(dùng http://user:pass@ip:cổng/đường_dẫn).\n"
                          "  -> Đang tự thử lại mỗi 2 giây (nhấn q hoặc Ctrl+C để thoát).")
                    was_lost = True
                shown = message_frame("Dang ket noi camera..." if not was_lost else "Mat ket noi camera - dang thu lai...")
            else:
                if was_lost:
                    print("Đã kết nối lại camera." if ever_connected else "Đã kết nối camera.")
                    was_lost = False
                ever_connected = True
                t_lost = time.time()
                if fid == last_id and kind != "video":   # chưa có khung mới
                    if not args.no_show and cv2.waitKey(1) & 0xFF == ord("q"):
                        break
                    time.sleep(0.001)
                    continue
                last_id = fid
                frame = frame.copy()
                annotate(frame, detector, predictor, thr)
                now = time.time()
                fps = 0.9 * fps + 0.1 / max(now - t_prev, 1e-6) if fps else 1 / max(now - t_prev, 1e-6)
                t_prev = now
                label = f"FPS: {fps:.1f} | {args.model}"          # nền đen ở góc dưới trái cho dễ đọc
                fh = frame.shape[0]
                cv2.rectangle(frame, (0, fh - 32), (14 + 11 * len(label), fh), (0, 0, 0), -1)
                cv2.putText(frame, label, (8, fh - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)
                shown = frame
                if args.save:
                    if writer is None:
                        h, w = shown.shape[:2]
                        writer = cv2.VideoWriter(args.save, cv2.VideoWriter_fourcc(*"mp4v"), 20, (w, h))
                    writer.write(shown)
            if not args.no_show:
                cv2.imshow("Emotion", shown)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    except KeyboardInterrupt:
        pass
    finally:
        reader.release()
        if writer is not None:
            writer.release()
        if not args.no_show:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
