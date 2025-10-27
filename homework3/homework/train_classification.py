import argparse
import os
from datetime import datetime
from torchvision import datasets, transforms

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from homework.models import Classifier

try:
    from homework.datasets.classification_datasets import load_data
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

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--transform", type=str, default="basic_aug")
    p.add_argument("--save_dir", type=str, default="logs")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # === Data ===
    # === Data ===
if load_data is not None:
    loaders = load_data(
        transform_pipeline=args.transform,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    if isinstance(loaders, dict):
        train_loader = loaders.get("train") or loaders.get("trn")
        val_loader = loaders.get("val") or loaders.get("valid") or loaders.get("test")
    else:
        train_loader, val_loader = loaders
else:
    

    # Try to use your project transform if present; otherwise, use a simple default.
    try:
        from homework.datasets.classification_datasets import get_transform
        tr_tf = get_transform(split="train", transform_pipeline=args.transform)
        va_tf = get_transform(split="val", transform_pipeline=args.transform)
    except Exception:
        MEAN = (0.485, 0.456, 0.406)
        STD  = (0.229, 0.224, 0.225)
        tr_tf = transforms.Compose([
            transforms.RandomResizedCrop(64, scale=(0.7, 1.0), ratio=(0.9, 1.1)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=10),
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
    # Auto-detect a validation folder name
    for cand in ("val", "valid", "validation", "test"):
        val_root = os.path.join("classification_data", cand)
        if os.path.isdir(val_root):
            break
    else:
        raise FileNotFoundError(
            "Could not find a validation folder in classification_data/. "
            "Expected one of: val, valid, validation, test"
        )

    train_ds = datasets.ImageFolder(train_root, transform=tr_tf)
    val_ds   = datasets.ImageFolder(val_root,   transform=va_tf)

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True
    )


    # === Model, loss, optim ===
    model = Classifier(num_classes=6).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2)

    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))

    best_acc, best_path = 0.0, None
    for epoch in range(1, args.epochs + 1):
        tr_loss, tr_acc = train_one_epoch(model, train_loader, criterion, optimizer, device, scaler)
        va_loss, va_acc = evaluate(model, val_loader, criterion, device)
        scheduler.step(va_acc)

        print(f"Epoch {epoch:02d} | train loss {tr_loss:.4f} acc {tr_acc:.4f} | val loss {va_loss:.4f} acc {va_acc:.4f}")

        if va_acc > best_acc:
            best_acc = va_acc
            best_path = save_model(model, out_dir=args.save_dir, prefix="classifier")
            print(f"  ↳ New best val acc: {best_acc:.4f}. Saved to {best_path}")

    print(f"Best val acc: {best_acc:.4f} | Best path: {best_path}")


if __name__ == "__main__":
    main()
