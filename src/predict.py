"""Tiền xử lý + dự đoán cảm xúc. Hàm preprocess() dùng CHUNG cho train, test và app
=> đảm bảo tiền xử lý lúc chạy realtime giống hệt lúc huấn luyện."""
import cv2
import numpy as np
import torch

from model import build_model

INPUT = {"cnn": 48, "transfer": 96}   # kích thước ảnh vào của M1 và M2
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)[:, None, None]
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)[:, None, None]


def preprocess(gray, kind, size=None):
    """gray: ảnh xám HxW uint8 (vùng mặt) -> tensor đã chuẩn hoá.
    cnn:      1xSxS, chuẩn hoá về [-1, 1]
    transfer: 3xSxS (nhân bản kênh xám), chuẩn hoá theo mean/std của ImageNet"""
    size = size or INPUT[kind]
    x = cv2.resize(gray, (size, size), interpolation=cv2.INTER_LINEAR).astype(np.float32) / 255.0
    if kind == "cnn":
        x = ((x - 0.5) / 0.5)[None]
    else:
        x = (np.repeat(x[None], 3, axis=0) - IMAGENET_MEAN) / IMAGENET_STD
    return torch.from_numpy(np.ascontiguousarray(x, dtype=np.float32))


class EmotionPredictor:
    def __init__(self, weights_path, device="cpu"):
        ck = torch.load(weights_path, map_location=device)
        self.kind, self.classes, self.size = ck["kind"], ck["classes"], ck["img_size"]
        self.device = device
        self.model = build_model(self.kind, len(self.classes), pretrained=False)
        self.model.load_state_dict(ck["state_dict"])
        self.model.to(device).eval()

    @torch.inference_mode()
    def proba_batch(self, faces):
        """faces: list ảnh xám vùng mặt -> mảng (N, số_lớp) xác suất softmax. Dự đoán cả batch một lần."""
        if not faces:
            return np.zeros((0, len(self.classes)), dtype=np.float32)
        x = torch.stack([preprocess(f, self.kind, self.size) for f in faces]).to(self.device)
        return torch.softmax(self.model(x), dim=1).cpu().numpy()

    def predict_batch(self, faces):
        """faces: list ảnh xám vùng mặt -> list (nhãn, confidence)."""
        out = []
        for p in self.proba_batch(faces):
            i = int(p.argmax())
            out.append((self.classes[i], float(p[i])))
        return out

    def predict(self, face_gray):
        """Dự đoán cho 1 ảnh khuôn mặt (xám) -> (nhãn, confidence)."""
        return self.predict_batch([face_gray])[0]
