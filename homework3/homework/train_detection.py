import argparse, os
from datetime import datetime
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from homework.models import Detector

try:
    from homework.datasets.drive_dataset import load_data
except Exception:
    load_data = None


def save_model(model, out_dir="logs", prefix="detector"):
    os.makedirs(out_dir, exist_ok=True)
    ts = datetime.now().strftime("%m%d_%H%M%S")
    path = os.path.join(out_dir, f"{prefix}_{ts}.th")
    torch.save({"model_state": model.state_dict()}, path)
    return path


def compute_iou(preds, targets, num_classes=3):
    ious = []
    for cls in range(num_classes):
        pred_cls = (preds == cls)
        tgt_cls = (targets == cls)
        inter = (pred_cls & tgt_cls).sum().item()
        union = (pred_cls | tgt_cls).sum().item()
        if union == 0:
            ious.append(float("nan"))
        else:
            ious.append(inter / union)
    return torch.tensor(ious).nanmean().item()


def evaluate(model, loader, device, num_classes=3):
    model.eval()
    ce = nn.CrossEntropyLoss()
    mae = nn.L1Loss()
    total_iou, total_mae, total_lane_mae, n = 0, 0, 0, 0
    with torch.inference_mode():
        for batch in loader:
            x = batch["image"].to(device)
            y_seg = batch["track"].to(device)
            y_depth = batch["depth"].to(device)

            seg_logits, depth = model(x)
            preds = seg_logits.argmax(1)
            total_iou += compute_iou(preds, y_seg, num_classes) * x.size(0)
            total_mae += mae(depth, y_depth.unsqueeze(1)) * x.size(0)
            # lane pixels only
            mask = (y_seg > 0).unsqueeze(1)
            total_lane_mae += (mae(depth[mask], y_depth.unsqueeze(1)[mask]) if mask.any() else 0)
            n += x.size(0)

    return total_iou / n, total_mae / n, total_lane_mae / n


def train_one_epoch(model, loader, optimizer, device, w_seg=1.0, w_depth=1.0):
    model.train()
    ce = nn.CrossEntropyLoss()
    mae = nn.L1Loss()
    running_loss, n = 0.0, 0
    for batch in loader:
        x = batch["image"].to(device)
        y_seg = batch["track"].to(device)
        y_depth = batch["depth"].to(device)

        optimizer.zero_grad()
        seg_logits, depth = model(x)
        loss_seg = ce(seg_logits, y_seg)
        loss_depth = mae(depth, y_depth.unsqueeze(1))
        loss = w_seg * loss_seg + w_depth * loss_depth
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * x.size(0)
        n += x.size(0)
    return running_loss / n


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--save_dir", type=str, default="logs")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    loaders = load_data(batch_size=args.batch_size, num_workers=args.num_workers)
    train_loader = loaders.get("train")
    val_loader   = loaders.get("val")

    model = Detector().to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_iou, best_path = 0.0, None
    for epoch in range(1, args.epochs + 1):
        tr_loss = train_one_epoch(model, train_loader, optimizer, device)
        val_iou, val_mae, val_lane_mae = evaluate(model, val_loader, device)
        print(f"Epoch {epoch:02d} | train loss {tr_loss:.4f} | "
              f"val IoU {val_iou:.4f} | MAE {val_mae:.4f} | lane MAE {val_lane_mae:.4f}")
        if val_iou > best_iou:
            best_iou = val_iou
            best_path = save_model(model, out_dir=args.save_dir, prefix="detector")
            print(f"  ↳ New best IoU: {best_iou:.4f}. Saved to {best_path}")

    print(f"Best IoU {best_iou:.4f} | Best path {best_path}")


if __name__ == "__main__":
    main()
