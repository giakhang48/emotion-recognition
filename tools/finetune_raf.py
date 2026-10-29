"""Fine-tune mô hình đã train (mặc định M2 transfer) trên FER-2013 + RAF-DB.

Colab:
  !unzip -q RAF-DB.zip -d /content/raf            # -> /content/raf/DATASET/{train,test}/1..7
  !python tools/finetune_raf.py --check-labels --raf-dir /content/raf/DATASET     # xem ảnh để kiểm tra map nhãn
  !python tools/finetune_raf.py --fer-dir <thư mục chứa train/ và test/ của FER-2013> \
        --raf-dir /content/raf/DATASET --init weights/transfer.pt --out weights/transfer_raf.pt

Ý tưởng: tiền xử lý dùng ĐÚNG hàm preprocess() của app => không phải sửa app.py.
Chọn checkpoint theo val accuracy (tách 10% từ tập train gộp). Tập test KHÔNG dùng để chọn, chỉ để báo cáo.
"""
import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from model import build_model      # noqa: E402
from predict import preprocess     # noqa: E402

# RAF-DB (basic): thư mục 1..7 = Surprise, Fear, Disgust, Happiness, Sadness, Anger, Neutral
RAF_MAP = {"1": "surprise", "2": "fear", "3": "disgust", "4": "happy", "5": "sad", "6": "angry", "7": "neutral"}
EXT = {".jpg", ".jpeg", ".png", ".bmp"}


def load_split(root, classes, name_map=None):
    """root/<lớp>/*.jpg -> (list ảnh xám uint8, mảng nhãn theo thứ tự `classes`)."""
    imgs, labels = [], []
    for d in sorted(p for p in Path(root).iterdir() if p.is_dir()):
        name = (name_map or {}).get(d.name, d.name).lower()
        if name not in classes:
            print(f"  bỏ qua thư mục lạ: {d}")
            continue
        for p in sorted(d.iterdir()):
            if p.suffix.lower() in EXT:
                g = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
                if g is not None:
                    imgs.append(g)
                    labels.append(classes.index(name))
    return imgs, np.array(labels, dtype=np.int64)


def augment(g, rng):
    """Augmentation chỉ cho tập train: lật ngang, xoay ±10°, dịch 10%, co giãn 0.9–1.1,
    độ sáng/tương phản ±0.3, thêm làm mờ nhẹ (mô phỏng ảnh mặt lớn bị thu nhỏ) và random erasing."""
    h, w = g.shape
    if rng.random() < 0.5:
        g = g[:, ::-1]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), rng.uniform(-10, 10), rng.uniform(0.9, 1.1))
    M[:, 2] += (rng.uniform(-0.1, 0.1) * w, rng.uniform(-0.1, 0.1) * h)
    g = cv2.warpAffine(np.ascontiguousarray(g), M, (w, h), borderMode=cv2.BORDER_REFLECT)
    g = np.clip(g.astype(np.float32) * rng.uniform(0.7, 1.3) + rng.uniform(-0.3, 0.3) * 255, 0, 255).astype(np.uint8)
    if rng.random() < 0.3:
        g = cv2.GaussianBlur(g, (3, 3), rng.uniform(0.3, 1.2))
    if rng.random() < 0.25:
        eh, ew = int(h * rng.uniform(0.1, 0.3)), int(w * rng.uniform(0.1, 0.3))
        y0, x0 = rng.integers(0, h - eh + 1), rng.integers(0, w - ew + 1)
        g = g.copy()
        g[y0:y0 + eh, x0:x0 + ew] = int(g.mean())
    return g


class Faces(Dataset):
    def __init__(self, imgs, labels, kind, size, train):
        self.imgs, self.labels, self.kind, self.size, self.train = imgs, labels, kind, size, train

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, i):
        g = self.imgs[i]
        if self.train:
            g = augment(g, np.random.default_rng())
        return preprocess(g, self.kind, self.size), int(self.labels[i])


@torch.inference_mode()
def evaluate(model, loader, device):
    model.eval()
    preds, tgts = [], []
    for x, y in loader:
        preds.append(model(x.to(device)).argmax(1).cpu())
        tgts.append(y)
    p, t = torch.cat(preds).numpy(), torch.cat(tgts).numpy()
    return accuracy_score(t, p), f1_score(t, p, average="macro")


def check_labels(raf_dir, out="raf_check.png"):
    """Mỗi hàng = 1 thư mục RAF-DB (1..7), 8 ảnh đầu: nhìn mắt thường để chắc chắn map nhãn đúng."""
    rows = []
    for k, name in RAF_MAP.items():
        files = sorted((Path(raf_dir) / "train" / k).glob("*.jpg"))[:8]
        row = [cv2.resize(cv2.imread(str(f)), (100, 100)) for f in files]
        row = np.hstack(row) if row else np.zeros((100, 800, 3), np.uint8)
        cv2.putText(row, f"{k} -> {name}", (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        rows.append(row)
    cv2.imwrite(out, np.vstack(rows))
    print(f"Đã lưu {out}. Mở ra xem: hàng '4 -> happy' phải toàn mặt cười, '1 -> surprise' phải ngạc nhiên...")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fer-dir")
    ap.add_argument("--raf-dir", required=True)
    ap.add_argument("--init", default=str(ROOT / "weights/transfer.pt"))
    ap.add_argument("--out", default=str(ROOT / "weights/transfer_raf.pt"))
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--bs", type=int, default=128)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--min-fer-acc", type=float, default=0.65, help="ngưỡng accuracy FER-2013 của đề cho mô hình này")
    ap.add_argument("--check-labels", action="store_true")
    a = ap.parse_args()

    if a.check_labels:
        return check_labels(a.raf_dir)
    if not a.fer_dir:
        sys.exit("Thiếu --fer-dir")

    torch.manual_seed(42)
    np.random.seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ck = torch.load(a.init, map_location="cpu")
    kind, classes, size = ck["kind"], ck["classes"], ck["img_size"]
    model = build_model(kind, len(classes), pretrained=False)
    model.load_state_dict(ck["state_dict"])
    model.to(device)

    print("Đọc dữ liệu...")
    fer_tr = load_split(Path(a.fer_dir) / "train", classes)
    fer_te = load_split(Path(a.fer_dir) / "test", classes)
    raf_tr = load_split(Path(a.raf_dir) / "train", classes, RAF_MAP)
    raf_te = load_split(Path(a.raf_dir) / "test", classes, RAF_MAP)

    print(f"\n{'lớp':10s} {'FER train':>10s} {'RAF train':>10s} {'FER test':>9s} {'RAF test':>9s}   (ghi vào report)")
    for i, c in enumerate(classes):
        print(f"{c:10s} {(fer_tr[1] == i).sum():10d} {(raf_tr[1] == i).sum():10d} "
              f"{(fer_te[1] == i).sum():9d} {(raf_te[1] == i).sum():9d}")
    print(f"{'tổng':10s} {len(fer_tr[0]):10d} {len(raf_tr[0]):10d} {len(fer_te[0]):9d} {len(raf_te[0]):9d}\n")

    imgs = fer_tr[0] + raf_tr[0]
    y = np.concatenate([fer_tr[1], raf_tr[1]])
    tr_idx, va_idx = train_test_split(np.arange(len(y)), test_size=0.1, stratify=y, random_state=42)
    counts = np.bincount(y[tr_idx], minlength=len(classes))
    weight = torch.tensor(len(tr_idx) / (len(classes) * counts), dtype=torch.float32, device=device)  # w_c = N/(K*n_c)
    print("class weight:", dict(zip(classes, weight.cpu().numpy().round(2))))

    def loader(imgs_, labels_, train):
        return DataLoader(Faces(imgs_, labels_, kind, size, train), batch_size=a.bs, shuffle=train,
                          num_workers=a.workers, pin_memory=device == "cuda")

    tr_ld = loader([imgs[i] for i in tr_idx], y[tr_idx], True)
    va_ld = loader([imgs[i] for i in va_idx], y[va_idx], False)
    fer_ld, raf_ld = loader(*fer_te, False), loader(*raf_te, False)

    def report(tag):
        fa, ff = evaluate(model, fer_ld, device)
        ra, rf = evaluate(model, raf_ld, device)
        print(f"[{tag}] FER test: acc {fa:.4f} F1 {ff:.4f} | RAF test: acc {ra:.4f} F1 {rf:.4f}")
        return {"fer_acc": fa, "fer_f1": ff, "raf_acc": ra, "raf_f1": rf}

    before = report("TRƯỚC fine-tune")

    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=a.epochs)
    crit = nn.CrossEntropyLoss(weight=weight)
    best, hist = -1.0, []
    for ep in range(1, a.epochs + 1):
        model.train()
        tot = n = 0
        for x, t in tr_ld:
            x, t = x.to(device), t.to(device)
            loss = crit(model(x), t)
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            tot += loss.item() * len(t)
            n += len(t)
        sched.step()
        va_acc, va_f1 = evaluate(model, va_ld, device)
        hist.append({"epoch": ep, "train_loss": tot / n, "val_acc": va_acc, "val_f1": va_f1})
        flag = ""
        if va_acc > best:
            best = va_acc
            torch.save({"kind": kind, "classes": classes, "img_size": size,
                        "state_dict": {k: v.cpu() for k, v in model.state_dict().items()}}, a.out)
            flag = "  <- lưu"
        print(f"epoch {ep:2d}  train loss {tot / n:.4f}  val acc {va_acc:.4f}  val F1 {va_f1:.4f}{flag}")

    model.load_state_dict(torch.load(a.out, map_location=device)["state_dict"])
    after = report("SAU fine-tune (checkpoint val tốt nhất)")
    Path(a.out).with_suffix(".json").write_text(json.dumps({"before": before, "after": after, "history": hist}, indent=1))
    ok = after["fer_acc"] >= a.min_fer_acc
    print(f"\nFER-2013 test {after['fer_acc']:.2%} so với ngưỡng {a.min_fer_acc:.0%}: "
          f"{'ĐẠT' if ok else 'KHÔNG ĐẠT -> không dùng bản này làm M2 nộp'}")


if __name__ == "__main__":
    main()
