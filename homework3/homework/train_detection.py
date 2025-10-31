
import argparse, os
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from homework.models import Detector

try:
    # Provided by the assignment for Part 2
    from homework.datasets.drive_dataset import load_data
except Exception:
    load_data = None


def save_model(model, out_dir="logs", prefix="detector"):
    os.makedirs(out_dir, exist_ok=True)
    ts = datetime.now().strftime("%m%d_%H%M%S")
    path = os.path.join(out_dir, f"{prefix}_{ts}.th")
    torch.save({"model_state": model.state_dict()}, path)
    return path


@torch.no_grad()
def compute_mean_iou(preds, targets, num_classes=3):
    """
    preds: (B,H,W) int
    targets: (B,H,W) int
    """
    iou_sum = 0.0
    count = 0
    for cls in range(num_classes):
        pred_cls = (preds == cls)
        tgt_cls  = (targets == cls)
        inter = (pred_cls & tgt_cls).sum().item()
        union = (pred_cls | tgt_cls).sum().item()
        if union > 0:
            iou_sum += inter / union
            count += 1
    return (iou_sum / count) if count > 0 else 0.0


def train_one_epoch(model, loader, optimizer, device, w_seg=1.0, w_depth=1.0):
    model.train()
    ce = nn.CrossEntropyLoss()
    l1 = nn.L1Loss()

    running_loss = 0.0
    n = 0
    for batch in loader:
        x = batch["image"].to(device)               # (B,3,96,128)
        y_seg = batch["track"].to(device).long()    # (B,96,128) with {0,1,2}
        y_depth = batch["depth"].to(device).float() # (B,96,128) in [0,1]

        optimizer.zero_grad(set_to_none=True)

        seg_logits, depth = model(x)                # seg:(B,3,H,W), depth:(B,1,H,W)
        loss_seg = ce(seg_logits, y_seg)
        loss_depth = l1(depth, y_depth.unsqueeze(1))
        loss = w_seg * loss_seg + w_depth * loss_depth

        loss.backward()
        optimizer.step()

        bs = x.size(0)
        running_loss += loss.item() * bs
        n += bs

    return running_loss / n


@torch.no_grad()
def evaluate(model, loader, device, num_classes=3):
    model.eval()
    l1 = nn.L1Loss(reduction="mean")

    total_iou = 0.0
    total_mae = 0.0
    total_lane_mae = 0.0
    n = 0

    for batch in loader:
        x = batch["image"].to(device)
        y_seg = batch["track"].to(device).long()
        y_depth = batch["depth"].to(device).float()

        seg_logits, depth = model(x)
        preds = seg_logits.argmax(dim=1)  # (B,H,W)

        # IoU (mean over present classes)
        miou = compute_mean_iou(preds, y_seg, num_classes=num_classes)

        # Depth MAE (all pixels)
        mae_all = l1(depth, y_depth.unsqueeze(1)).item()

        # Depth MAE (lane boundary pixels only: classes 1 or 2)
        abs_err = (depth - y_depth.unsqueeze(1)).abs()  # (B,1,H,W)
        lane_mask = (y_seg > 0).unsqueeze(1)            # (B,1,H,W)
        if lane_mask.any():
            lane_mae = abs_err[lane_mask].mean().item()
        else:
            lane_mae = 0.0

        bs = x.size(0)
        total_iou += miou * bs
        total_mae += mae_all * bs
        total_lane_mae += lane_mae * bs
        n += bs

    return total_iou / n, total_mae / n, total_lane_mae / n


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--w_seg", type=float, default=1.0)
    p.add_argument("--w_depth", type=float, default=1.0)
    p.add_argument("--save_dir", type=str, default="logs")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    if load_data is None:
        raise RuntimeError("Could not import load_data from homework.datasets.drive_dataset")

    # Expecting load_data() to return dict with 'train' and 'val' DataLoaders.
    loaders = load_data(batch_size=args.batch_size, num_workers=args.num_workers)
    train_loader = loaders.get("train") or loaders.get("trn")
    val_loader   = loaders.get("val") or loaders.get("valid") or loaders.get("test")
    if train_loader is None or val_loader is None:
        raise RuntimeError("drive_dataset.load_data() did not return expected loaders for train/val")

    model = Detector().to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=3)

    best_iou, best_path = 0.0, None
    for epoch in range(1, args.epochs + 1):
        tr_loss = train_one_epoch(model, train_loader, optimizer, device, w_seg=args.w_seg, w_depth=args.w_depth)
        val_iou, val_mae, val_lane_mae = evaluate(model, val_loader, device)

        print(
            f"Epoch {epoch:02d} | train loss {tr_loss:.4f} | "
            f"val IoU {val_iou:.4f} | MAE {val_mae:.4f} | lane MAE {val_lane_mae:.4f}"
        )

        scheduler.step(val_iou)

        if val_iou > best_iou:
            best_iou = val_iou
            best_path = save_model(model, out_dir=args.save_dir, prefix="detector")
            print(f"  ↳ New best IoU: {best_iou:.4f}. Saved to {best_path}")

    print(f"Best IoU {best_iou:.4f} | Best path {best_path}")


if __name__ == "__main__":
    main()
