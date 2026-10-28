"""Phát hiện khuôn mặt bằng Haar Cascade (có sẵn trong OpenCV, không cần tải thêm file)."""
import cv2


class HaarFaceDetector:
    def __init__(self, scale_factor=1.1, min_neighbors=5, min_size=40, detect_width=480):
        if not hasattr(cv2, "CascadeClassifier"):   # OpenCV 5.x đã bỏ CascadeClassifier khỏi gói chính
            raise RuntimeError("Cần OpenCV 4.x: pip install \"opencv-python>=4.7,<5\"")
        path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        self.cascade = cv2.CascadeClassifier(path)
        if self.cascade.empty():
            raise RuntimeError(f"Không nạp được Haar Cascade: {path}")
        self.scale_factor = scale_factor
        self.min_neighbors = min_neighbors
        self.min_size = min_size          # tính trên ảnh đã thu nhỏ
        self.detect_width = detect_width  # thu nhỏ ảnh về chiều rộng này để detect nhanh hơn

    def detect(self, frame_bgr):
        """Trả về (danh sách (x, y, w, h) trên ảnh gốc, ảnh xám gốc để cắt mặt)."""
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        s = min(1.0, self.detect_width / gray.shape[1])
        small = cv2.resize(gray, None, fx=s, fy=s) if s < 1.0 else gray
        small = cv2.equalizeHist(small)   # chỉ dùng cho detect; ảnh cắt mặt vẫn là ảnh xám gốc
        rects = self.cascade.detectMultiScale(
            small, scaleFactor=self.scale_factor, minNeighbors=self.min_neighbors,
            minSize=(self.min_size, self.min_size))
        boxes = [(int(x / s), int(y / s), int(w / s), int(h / s)) for x, y, w, h in rects]
        return boxes, gray
