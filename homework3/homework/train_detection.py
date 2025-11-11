# homework/train_detection.py
import argparse, os, math
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader  # noqa: F401


from homework.models import Detector, save_model as save_for_grader

try:
    from .datasets import drive_dataset as drive_ds
except Exception:
    import datasets.drive_dataset as drive_ds



def _loaders(dataset_path, batch_size, num_workers, transform_pipeline="basic"):
    # Try common function names found in different starter repos
    for fname in ("load_data", "load_data_fn", "get_loaders", "load"):
        fn = getattr(drive_ds, fname, None)
        if fn is None:
            continue
        # Call with the richest signature first; fall back as needed
        try:
            return fn(
                dataset_path,
                split=None,
                batch_size=batch_size,
                num_workers=num_workers,
                return_dataloader=True,
                transform_pipeline=transform_pipeline,
            )
        except TypeError:
            try:
                return fn(dataset_path, batch_size=batch_size, num_workers=num_workers)
            except TypeError:
                return fn(dataset_path)
    raise ImportError(
        "No loader found in drive_dataset.py. Expected one of: "
        "load_data, load_data_fn, get_loaders, load"
    )



def save_checkpoint(model, out_dir="logs", prefix="detector"):
    os.makedirs(out_dir, exist_ok=True)
    ts = datetime.now().strftime("%m%d_%H%M%S")
    path = os.path.join(out_dir, f"{prefix}_{ts}.th")
    torch.save({"model_state": model.state_dict()}, path)
    return path


@torch.no_grad()
def compute_mean_iou(preds, targets, num_classes=3):
    iou_sum, count = 0.0, 0
    for cls in range(num_classes):
        pc = (preds == cls)
        tc = (targets == cls)
        inter = (pc & tc).sum().item()
        union = (pc | tc).sum().item()
        if union > 0:
            iou_sum += inter / union
            count += 1
    return (iou_sum / count) if count > 0 else float("nan")


def _get_seg(batch):
    for k in ("track", "seg", "mask", "labels"):
        if k in batch and batch[k] is not None:
            return batch[k]
    return None


def _get_depth(batch):
    for k in ("depth", "depths"):
        if k in batch and batch[k] is not None:
            return batch[k]
    raise KeyError("Depth tensor not found in batch (looked for keys: depth/depths).")


def train_one_epoch(model, loader, optimizer, device, w_seg=1.0, w_depth=1.0, have_masks=True):
    model.train()
    ce = nn.CrossEntropyLoss()
    l1 = nn.L1Loss()

    running_loss, n = 0.0, 0
    for batch in loader:
        x = batch["image"].to(device, non_blocking=True)             # (B,3,H,W)
        y_depth = _get_depth(batch).to(device, non_blocking=True).float()  # (B,H,W)

        y_seg = _get_seg(batch)
        if have_masks and y_seg is not None:
            y_seg = y_seg.to(device, non_blocking=True).long()

        optimizer.zero_grad(set_to_none=True)
        seg_logits, depth = model(x)                                  # seg:(B,3,H,W), depth:(B,1,H,W)

        loss_depth = l1(depth, y_depth.unsqueeze(1)) * w_depth
        loss_seg = ce(seg_logits, y_seg) * w_seg if (have_masks and y_seg is not None and w_seg > 0.0) \
                   else torch.zeros((), device=device)

        loss = loss_seg + loss_depth
        loss.backward()
        optimizer.step()

        bs = x.size(0)
        running_loss += float(loss.item()) * bs
        n += bs

    return running_loss / max(n, 1)


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
        y_depth = _get_depth(batch).to(device, non_blocking=True).float()

        y_seg = _get_seg(batch)
        if have_masks and y_seg is not None:
            y_seg = y_seg.to(device, non_blocking=True).long()

        seg_logits, depth = model(x)

        mae_all = l1(depth, y_depth.unsqueeze(1)).item()
        total_mae += mae_all * x.size(0)

        if have_masks and y_seg is not None:
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
    p.add_argument("--dataset_path", type=str, default="drive_data", help="Path that contains train/ and val/ dirs")
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--w_seg", type=float, default=1.0)
    p.add_argument("--w_depth", type=float, default=1.0)
    p.add_argument("--save_dir", type=str, default="logs")
    p.add_argument("--allow_missing_masks", action="store_true",
                   help="Force depth-only even if masks are present.")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    if load_data is None:
        raise RuntimeError(
            "Could not import load_data from datasets/road_dataset.py. "
            "Ensure the file exists and is importable."
        )

    # road_dataset.load_data expects the dataset path as the FIRST positional arg
    # and (based on your error) doesn't accept root/image_size/require_masks.
    loaders = load_data(args.dataset_path, batch_size=args.batch_size, num_workers=args.num_workers)

    # Accept both dict or tuple returns
    if isinstance(loaders, dict):
        train_loader = loaders.get("train") or loaders.get("trn")
        val_loader = loaders.get("val") or loaders.get("valid") or loaders.get("test")
    elif isinstance(loaders, (list, tuple)) and len(loaders) >= 2:
        train_loader, val_loader = loaders[0], loaders[1]
    else:
        raise RuntimeError("road_dataset.load_data() returned an unexpected type. Expect dict with train/val or (train_loader, val_loader).")

    if train_loader is None or val_loader is None:
        raise RuntimeError("road_dataset.load_data() did not provide both train and val loaders.")

    # Peek one batch to decide if masks exist; allow override with --allow_missing_masks
    try:
        dbg = next(iter(train_loader))
        seg_dbg = _get_seg(dbg)
        depth_dbg = _get_depth(dbg)
        have_masks_detected = (seg_dbg is not None)
        have_masks = have_masks_detected and (not args.allow_missing_masks)

        seg_uni = seg_dbg.unique().tolist() if seg_dbg is not None else []
        d_mean = float(depth_dbg.float().mean()) if depth_dbg is not None else float("nan")
        d_std  = float(depth_dbg.float().std())  if depth_dbg is not None else float("nan")
        print(f"[Sanity] seg classes: {seg_uni if seg_uni else 'N/A'} | depth mean {d_mean:.4f} std {d_std:.4f}")
        if have_masks and set(seg_uni) == {0}:
            print("[Warn] Masks appear to be all background; IoU will be low. Verify labels.")
    except Exception as e:
        print(f"[Sanity] Could not sample a debug batch: {e}")
        have_masks = not args.allow_missing_masks  # conservative default

    effective_w_seg = args.w_seg if have_masks else 0.0
    if have_masks and effective_w_seg <= 0.0:
        print("[Warn] w_seg <= 0 with masks present; segmentation will not learn.")

    model = Detector().to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=3)

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
