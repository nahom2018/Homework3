
import argparse, os, math
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader  # noqa: F401

from homework.models import Detector, save_model as save_for_grader

# --- Use the CORRECT dataset loader (README typo) ---
# Prefer datasets/road_dataset.py; keep fallbacks just in case.
load_road_data = None
try:
    from homework.datasets.road_dataset import load_data as load_road_data
except Exception:
    try:
        from homework.road_dataset import load_data as load_road_data
    except Exception:
        load_road_data = None


def save_checkpoint(model, out_dir="logs", prefix="detector"):
    os.makedirs(out_dir, exist_ok=True)
    ts = datetime.now().strftime("%m%d_%H%M%S")
    path = os.path.join(out_dir, f"{prefix}_{ts}.th")
    torch.save({"model_state": model.state_dict()}, path)
    return path


@torch.no_grad()
def compute_mean_iou(preds, targets, num_classes=3):
    """
    preds:   (B,H,W) int
    targets: (B,H,W) int
    """
    iou_sum, count = 0.0, 0
    for cls in range(num_classes):
        pred_cls = (preds == cls)
        tgt_cls = (targets == cls)
        inter = (pred_cls & tgt_cls).sum().item()
        union = (pred_cls | tgt_cls).sum().item()
        if union > 0:
            iou_sum += inter / union
            count += 1
    return (iou_sum / count) if count > 0 else float("nan")


# ---------- Batch helpers (tolerate different key names) ----------
def get_seg(batch):
    # common names: "track", "seg", "mask", "labels"
    for k in ("track", "seg", "mask", "labels"):
        if k in batch:
            return batch[k]
    return None

def get_depth(batch):
    # common names: "depth", "depths"
    for k in ("depth", "depths"):
        if k in batch:
            return batch[k]
    return None
# ------------------------------------------------------------------


def train_one_epoch(model, loader, optimizer, device, w_seg=1.0, w_depth=1.0, have_masks=True):
    model.train()
    ce = nn.CrossEntropyLoss()
    l1 = nn.L1Loss()

    running_loss, n = 0.0, 0
    for batch in loader:
        x = batch["image"].to(device, non_blocking=True)          # (B,3,H,W)
        y_seg = get_seg(batch)
        y_depth = get_depth(batch).to(device, non_blocking=True).float()  # (B,H,W)

        optimizer.zero_grad(set_to_none=True)
        seg_logits, depth = model(x)                              # seg:(B,3,H,W), depth:(B,1,H,W)

        # Depth loss (required)
        loss_depth = l1(depth, y_depth.unsqueeze(1)) * w_depth

        # Seg loss only if masks exist and weight > 0
        if have_masks and (y_seg is not None) and w_seg > 0.0:
            y_seg = y_seg.to(device, non_blocking=True).long()
            loss_seg = ce(seg_logits, y_seg) * w_seg
        else:
            loss_seg = torch.zeros((), device=device)

        loss = loss_seg + loss_depth
        loss.backward()
        optimizer.step()

        bs = x.size(0)
        running_loss += loss.item() * bs
        n += bs

    return running_loss / n


@torch.no_grad()
def evaluate(model, loader, device, num_classes=3, have_masks=True):
    model.eval()
    l1 = nn.L1Loss(reduction="mean")

    total_iou = 0.0
    total_mae = 0.0
    total_lane_mae = 0.0
    n = 0

    for batch in loader:
        x = batch["image"].to(device, non_blocking=True)
        y_seg = get_seg(batch)
        y_depth = get_depth(batch).to(device, non_blocking=True).float()

        seg_logits, depth = model(x)

        mae_all = l1(depth, y_depth.unsqueeze(1)).item()
        total_mae += mae_all * x.size(0)

        if have_masks and (y_seg is not None):
            y_seg = y_seg.to(device, non_blocking=True).long()
            preds = seg_logits.argmax(dim=1)  # (B,H,W)
            miou = compute_mean_iou(preds, y_seg, num_classes=num_classes)
            if not math.isnan(miou):
                total_iou += miou * x.size(0)

            abs_err = (depth - y_depth.unsqueeze(1)).abs()  # (B,1,H,W)
            lane_mask = (y_seg > 0).unsqueeze(1)            # (B,1,H,W)
            if lane_mask.any():
                lane_mae = abs_err[lane_mask].mean().item()
            else:
                lane_mae = float("nan")
            if not math.isnan(lane_mae):
                total_lane_mae += lane_mae * x.size(0)

        n += x.size(0)

    if n == 0:
        return float("nan"), float("nan"), float("nan")

    avg_mae = total_mae / n
    if have_masks:
        avg_iou = total_iou / n if total_iou > 0 else float("nan")
        avg_lane_mae = total_lane_mae / n if total_lane_mae > 0 else float("nan")
        return avg_iou, avg_mae, avg_lane_mae
    else:
        return float("nan"), avg_mae, float("nan")


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
    p.add_argument("--allow_missing_masks", action="store_true",
                   help="Depth-only mode if loader doesn't supply segmentation labels.")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    if load_road_data is None:
        raise RuntimeError(
            "Could not import load_data from datasets/road_dataset.py. "
            "Make sure you're using the course-provided road_dataset.py."
        )

    # Try calling road_dataset.load_data with minimal parameters first;
    # fall back to richer signatures if supported.
    loaders = None
    tried = []
    for call in (
        lambda: load_road_data(batch_size=args.batch_size, num_workers=args.num_workers),
        lambda: load_road_data(batch_size=args.batch_size, num_workers=args.num_workers, image_size=(96,128)),
        lambda: load_road_data(batch_size=args.batch_size, num_workers=args.num_workers, root="drive_data"),
        lambda: load_road_data(batch_size=args.batch_size, num_workers=args.num_workers, root="drive_data", image_size=(96,128)),
    ):
        try:
            loaders = call()
            break
        except TypeError as e:
            tried.append(str(e))
            continue
    if loaders is None:
        raise RuntimeError("road_dataset.load_data signature mismatch. Tried variants:\n" + "\n".join(tried))

    train_loader = loaders.get("train") or loaders.get("trn")
    val_loader = loaders.get("val") or loaders.get("valid") or loaders.get("test")
    if train_loader is None or val_loader is None:
        raise RuntimeError("road_dataset.load_data() did not return expected loaders for train/val")

    # --- One-batch sanity check so you immediately see labels & depth are present ---
    try:
        dbg = next(iter(train_loader))
        seg_dbg = get_seg(dbg)
        depth_dbg = get_depth(dbg)
        seg_uni = seg_dbg.unique().tolist() if seg_dbg is not None else []
        d_mean = float(depth_dbg.float().mean()) if depth_dbg is not None else float("nan")
        d_std  = float(depth_dbg.float().std())  if depth_dbg is not None else float("nan")
        print(f"[Sanity] seg classes: {seg_uni if seg_uni else 'N/A'} | depth mean {d_mean:.4f} std {d_std:.4f}")
    except Exception as e:
        print(f"[Sanity] Could not sample a debug batch: {e}")

    model = Detector().to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=3)

    # If loader provides no masks, force seg loss to 0 (depth-only)
    have_masks = True
    try:
        # peek again to decide
        have_masks = get_seg(next(iter(train_loader))) is not None and (args.w_seg > 0.0) and (not args.allow_missing_masks)
    except Exception:
        have_masks = not args.allow_missing_masks

    effective_w_seg = args.w_seg if have_masks else 0.0
    if have_masks and effective_w_seg <= 0.0:
        print("[Warn] w_seg <= 0 with masks present; segmentation will not learn.")

    best_iou, best_path = -float("inf"), None
    for epoch in range(1, args.epochs + 1):
        tr_loss = train_one_epoch(model, train_loader, optimizer, device,
                                  w_seg=effective_w_seg, w_depth=args.w_depth, have_masks=have_masks)
        val_iou, val_mae, val_lane_mae = evaluate(model, val_loader, device, have_masks=have_masks)

        iou_str  = "N/A" if (math.isnan(val_iou)) else f"{val_iou:.4f}"
        lane_str = "N/A" if (math.isnan(val_lane_mae)) else f"{val_lane_mae:.4f}"
        print(f"Epoch {epoch:02d} | train loss {tr_loss:.4f} | val IoU {iou_str} | MAE {val_mae:.4f} | lane MAE {lane_str}")

        score_for_sched = (val_iou if not math.isnan(val_iou) else -val_mae)
        scheduler.step(score_for_sched)

        if not math.isnan(val_iou) and val_iou > best_iou:
            best_iou = val_iou
            best_path = save_checkpoint(model, out_dir=args.save_dir, prefix="detector")
            save_for_grader(model)  # writes homework/detector.th for the grader
            print(f"  ↳ New best IoU: {best_iou:.4f}. Saved to {best_path} and homework/detector.th")

    if best_path is None:
        best_path = save_checkpoint(model, out_dir=args.save_dir, prefix="detector_final")
        print(f"Saved final detector checkpoint to {best_path}")
    save_for_grader(model)  # refresh homework/detector.th with final state
    best_str = "N/A" if math.isnan(best_iou) or best_iou == -float("inf") else f"{best_iou:.4f}"
    print(f"Best IoU {best_str} | Best path {best_path}")


if __name__ == "__main__":
    main()
