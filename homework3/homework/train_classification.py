import argparse
import os
from datetime import datetime
from torchvision import datasets, transforms

import torch
import torch.nn as nn
import torch.optim as optim
from homework.models import Classifier
import os, csv
from PIL import Image
from torch.utils.data import Dataset, DataLoader


try:
    from homework.datasets.classification_dataset import load_data, get_transform
except Exception:
    load_data = None


def accuracy(logits, y):
    with torch.no_grad():
        preds = logits.argmax(dim=1)
        correct = (preds == y).sum().item()
        return correct / y.size(0)


def save_model(model, out_dir="logs", prefix="classifier"):
    os.makedirs(out_dir, exist_ok=True)
    ts = datetime.now().strftime("%m%d_%H%M%S")
    path = os.path.join(out_dir, f"{prefix}_{ts}.th")
    torch.save({"model_state": model.state_dict()}, path)
    return path


def train_one_epoch(model, loader, criterion, optimizer, device, scaler=None):
    model.train()
    running_loss, running_acc, n = 0.0, 0.0, 0
    for batch in loader:
        x, y = batch["image"], batch["label"] if isinstance(batch, dict) else batch
        x, y = x.to(device), y.to(device)

        optimizer.zero_grad(set_to_none=True)
        if scaler is not None:
            with torch.autocast(device_type=device.type, dtype=torch.float16):
                logits = model(x)
                loss = criterion(logits, y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()


        bs = y.size(0)
        running_loss += loss.item() * bs
        running_acc += accuracy(logits, y) * bs
        n += bs


    return running_loss / n, running_acc / n

def evaluate(model, loader, criterion, device):
    model.eval()
    loss_sum, acc_sum, n = 0.0, 0.0, 0
    with torch.inference_mode():
        for batch in loader:
            x, y = batch["image"], batch["label"] if isinstance(batch, dict) else batch
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = criterion(logits, y)
            bs = y.size(0)
            loss_sum += loss.item() * bs
            acc_sum += accuracy(logits, y) * bs
            n += bs
    return loss_sum / n, acc_sum / n

class STKFolderOrCSV(Dataset):
    """
    Works with either:
      A) labels.csv in each split dir (filename,label) where label can be int or string
      B) per-class subfolders (ImageFolder-style)
    """
    def __init__(self, root, transform=None):
        self.root = root
        self.transform = transform
        csv_path = os.path.join(root, "labels.csv")
        self.samples = []
        self.class_to_idx = None

        if os.path.isfile(csv_path):
            # --- CSV mode: supports string labels ---
            rows = []
            with open(csv_path, "r") as f:
                import csv as _csv
                reader = _csv.reader(f)
                header = next(reader, None)
                for row in reader:
                    if not row or len(row) < 2:
                        continue
                    rows.append((row[0], row[1]))

            if not rows:
                raise FileNotFoundError(f"No image entries found via {csv_path}.")

            # Build label mapping: if label looks like int, use int; else build a string->index map
            # Collect all label tokens
            label_tokens = [r[1] for r in rows]
            all_intlike = False
            try:
                _ = [int(t) for t in label_tokens]
                all_intlike = True
            except Exception:
                all_intlike = False

            if all_intlike:
                # Use provided ints directly, but still record mapping for reference
                classes = sorted(set(int(t) for t in label_tokens))
                self.class_to_idx = {str(c): c for c in classes}
                def _to_idx(lbl): return int(lbl)
            else:
                # Map strings to indices deterministically
                classes = sorted(set(label_tokens))
                self.class_to_idx = {c: i for i, c in enumerate(classes)}
                def _to_idx(lbl): return self.class_to_idx[lbl]

            # Resolve image paths and build samples
            for fname, lbl in rows:
                img_path = os.path.join(root, "images", fname)
                if not os.path.isfile(img_path):
                    img_path = os.path.join(root, fname)
                if os.path.isfile(img_path):
                    self.samples.append((img_path, _to_idx(lbl)))

            if len(self.samples) == 0:
                raise FileNotFoundError(
                    f"No images found for entries in {csv_path}. "
                    "Checked both <split>/images/<filename> and <split>/<filename>."
                )

            print(f"[STKFolderOrCSV] CSV mode at '{root}': {len(self.samples)} samples, "
                  f"{len(self.class_to_idx)} classes -> {self.class_to_idx}")

        else:
            # --- Fallback: class-subfolders mode ---
            classes = sorted([d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))])
            if not classes:
                raise FileNotFoundError(f"Couldn't find labels.csv or class folders in {root}.")
            self.class_to_idx = {c: i for i, c in enumerate(classes)}
            exts = {".jpg", ".jpeg", ".png", ".bmp"}
            for c in classes:
                cdir = os.path.join(root, c)
                for fn in os.listdir(cdir):
                    if os.path.splitext(fn)[1].lower() in exts:
                        self.samples.append((os.path.join(cdir, fn), self.class_to_idx[c]))
            if len(self.samples) == 0:
                raise FileNotFoundError(f"No images found under class folders in {root}.")
            print(f"[STKFolderOrCSV] Folder mode at '{root}': {len(self.samples)} samples, "
                  f"{len(self.class_to_idx)} classes -> {self.class_to_idx}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label_idx = self.samples[idx]
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return {"image": img, "label": torch.tensor(label_idx, dtype=torch.long)}



def main():
    p = argparse.ArgumentParser()
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--transform", type=str, default="aug")
    p.add_argument("--save_dir", type=str, default="logs")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")


    # === Data ===
    try:
        from homework.datasets.classification_dataset import get_transform as _get_transform
        tr_tf = _get_transform(transform_pipeline=args.transform) if "aug" in args.transform else _get_transform(
            "default")
        va_tf = _get_transform("default")
    except Exception:
        from torchvision import transforms
        MEAN = (0.485, 0.456, 0.406)
        STD = (0.229, 0.224, 0.225)
        if args.transform in ("aug", "basic_aug"):
            tr_tf = transforms.Compose([
                transforms.RandomResizedCrop(64, scale=(0.7, 1.0), ratio=(0.9, 1.1)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(degrees=10),
                transforms.ToTensor(),
                transforms.Normalize(MEAN, STD),
            ])
        else:
            tr_tf = transforms.Compose([
                transforms.Resize(72),
                transforms.CenterCrop(64),
                transforms.ToTensor(),
                transforms.Normalize(MEAN, STD),
            ])
        va_tf = transforms.Compose([
            transforms.Resize(72),
            transforms.CenterCrop(64),
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD),
        ])

    train_root = os.path.join("classification_data", "train")
    val_root = os.path.join("classification_data", "val")
    if not os.path.isdir(train_root) or not os.path.isdir(val_root):
        raise FileNotFoundError("Expected classification_data/train and classification_data/val.")

    train_ds = STKFolderOrCSV(train_root, transform=tr_tf)
    val_ds = STKFolderOrCSV(val_root, transform=va_tf)

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True
    )
    # === Model, loss, optim ===
    model = Classifier().to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20)

    best_acc = 0.0
    for epoch in range(1, 21):
        model.train()
        tr_loss, tr_correct, n = 0.0, 0, 0
        for batch in train_loader:
            x = batch["image"].to(device)
            y = batch["label"].to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            tr_loss += float(loss.item()) * x.size(0)
            tr_correct += (logits.argmax(1) == y).sum().item()
            n += x.size(0)
        train_acc = tr_correct / max(n, 1)

        model.eval()
        val_correct, m = 0, 0
        with torch.inference_mode():
            for batch in val_loader:
                x = batch["image"].to(device)
                y = batch["label"].to(device)
                logits = model(x)
                val_correct += (logits.argmax(1) == y).sum().item()
                m += x.size(0)
        val_acc = val_correct / max(m, 1)
        scheduler.step()

        print(f"Epoch {epoch:02d} | train acc {train_acc:.3f} | val acc {val_acc:.3f}")
        if val_acc > best_acc:
            best_acc = val_acc
            save_model(model)  # whatever your grader expects

    print(f"Best val acc: {best_acc:.4f} ")


if __name__ == "__main__":
    main()
