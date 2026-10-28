"""Định nghĩa 2 mô hình: M1 (CNN tự thiết kế) và M2 (transfer learning MobileNetV2)."""
import torch.nn as nn
from torchvision import models


def conv_block(cin, cout, p_drop):
    """Conv-BN-ReLU x2 -> MaxPool -> Dropout (pooling giảm 1/2 kích thước ảnh)."""
    return nn.Sequential(
        nn.Conv2d(cin, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
        nn.Conv2d(cout, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
        nn.MaxPool2d(2),
        nn.Dropout(p_drop),
    )


class EmotionCNN(nn.Module):
    """M1: ảnh xám 1x48x48 -> 3 block (48->24->12->6) -> FC. ~3.5M tham số."""

    def __init__(self, num_classes=7):
        super().__init__()
        self.features = nn.Sequential(
            conv_block(1, 64, 0.25),
            conv_block(64, 128, 0.25),
            conv_block(128, 256, 0.30),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 6 * 6, 256), nn.BatchNorm1d(256), nn.ReLU(inplace=True), nn.Dropout(0.5),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


def build_transfer(num_classes=7, pretrained=True):
    """M2: MobileNetV2 pretrained ImageNet, thay lớp phân loại cuối bằng 7 lớp."""
    weights = models.MobileNet_V2_Weights.IMAGENET1K_V1 if pretrained else None
    m = models.mobilenet_v2(weights=weights)
    m.classifier = nn.Sequential(nn.Dropout(0.3), nn.Linear(m.last_channel, num_classes))
    return m


def build_model(kind, num_classes=7, pretrained=True):
    """kind: 'cnn' (M1) hoặc 'transfer' (M2)."""
    if kind == "cnn":
        return EmotionCNN(num_classes)
    if kind == "transfer":
        return build_transfer(num_classes, pretrained)
    raise ValueError(f"kind phải là 'cnn' hoặc 'transfer', nhận được: {kind}")


def count_params(model):
    return sum(p.numel() for p in model.parameters())
