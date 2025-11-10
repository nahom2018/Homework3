import argparse, time
from pathlib import Path
import importlib, importlib.util, sys

import torch
import torch.nn as nn
import torch.optim as optim

from .models import load_model  # if running as module: python -m homework.train_detection

def import_drive_loader():
    here = Path(__file__).resolve()
    sys.path.insert(0, str(here.parent))          # .../homework
    sys.path.insert(0, str(here.parent.parent))   # .../HW3

    candidates = [
        "homework.datasets.drive_dataset",
        "homework.datasets.road_dataset",
        "drive_dataset",
        "road_dataset",
    ]
    last_err = None
    for modname in candidates:
        try:
            m = importlib.import_module(modname)
            if hasattr(m, "load_data"):
                return getattr(m, "load_data")
        except Exception as e:
            last_err = e

    # fallback: direct file load
    search_places = [
        here.parent,                                  # homework/
        here.parent / "datasets",                     # homework/datasets/
        here.parent.parent,                           # HW3/
        here.parent.parent / "homework" / "datasets", # HW3/homework/datasets/
    ]
    for fname in ("drive_dataset.py", "road_dataset.py"):
        for base in search_places:
            p = base / fname
            if p.exists():
                spec = importlib.util.spec_from_file_location("dyn_ds", p)
                mod = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(mod)  # type: ignore
                if hasattr(mod, "load_data"):
                    return mod.load_data

    raise ImportError(f"Could not import drive dataset loader. Last error: {last_err}")

def iou_from_logits(logits, y_true, num_classes=3):
    preds = logits.argmax(1)
    ious = []
    for c in range(num_classes):
        pred_c = (preds == c)
        true_c = (y_true == c)
        inter = (pred_c & true_c).sum().float()
        union = (pred_c | true_c).sum().float().clamp(min=1)
        ious.append((inter / union).item())
    return sum(ious) / num_classes

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=str, default="drive_data")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--lambda_depth", type=float, default=0.0, help="set 0.0 if no depth available")
    ap.add_argument("--save", type=str, default="detector.pt")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    load_data_fn = import_drive_loader()

    # Adaptive loader resolution: split API or (train,val) API
    def _resolve_loaders():
        try:
            tr = load_data_fn(args.data, split="train", batch_size=args.bs, return_dataloader=True)
            va = load_data_fn(args.data, split="val",   batch_size=args.bs, return_dataloader=True, shuffle=False)
            return tr, va
        except TypeError:
            loaders = load_data_fn(args.data, batch_size=args.bs, num_workers=2)
            if isinstance(loaders, tuple) and len(loaders) == 2:
                return loaders[0], loaders[1]
            if isinstance(loaders, dict):
                tr = loaders.get("train") or loaders.get("train_loader")
                va = loaders.get("val") or loaders.get("val_loader")
                if tr is None or va is None:
                    raise RuntimeError("Could not find 'train'/'val' loaders in dict from load_data().")
                return tr, va
            raise RuntimeError("Unsupported return type from load_data().")
    train_loader, val_loader = _resolve_loaders()

    model = load_model("detector", in_channels=3, num_seg_classes=3).to(device)
    ce = nn.CrossEntropyLoss()
    l1 = nn.L1Loss()
    opt = optim.Adam(model.parameters(), lr=args.lr)

    best_miou = 0.0
    for epoch in range(1, args.epochs + 1):
        model.train()
        t0 = time.time()
        tr_loss = 0.0
        tr_miou = 0.0

        for batch in train_loader:
            x   = batch["image"].to(device)         # (B,3,H,W)
            seg = batch["track"].long().to(device)  # (B,H,W)
            dep = batch["depth"].float().to(device) # (B,H,W)
            has_depth = batch.get("has_depth", None)

            opt.zero_grad()
            seg_logits, dep_pred = model(x)         # (B,3,H,W), (B,H,W)
            loss_seg = ce(seg_logits, seg)

            # depth loss masked by availability
            loss_dep = torch.tensor(0.0, device=device)
            if has_depth is not None:
                mask = has_depth.to(device).bool()
                if mask.any():
                    loss_dep = l1(dep_pred[mask], dep[mask])
            else:
                # No flag -> assume depth exists (legacy datasets)
                loss_dep = l1(dep_pred, dep)

            loss = loss_seg + args.lambda_depth * loss_dep
            loss.backward()
            opt.step()

            tr_loss += loss.item()
            tr_miou += iou_from_logits(seg_logits, seg)

        ntr = max(1, len(train_loader))
        tr_loss /= ntr
        tr_miou /= ntr

        # eval
        model.eval()
        val_miou = 0.0
        val_l1_sum = 0.0
        val_l1_count = 0
        with torch.inference_mode():
            for batch in val_loader:
                x   = batch["image"].to(device)
                seg = batch["track"].long().to(device)
                dep = batch["depth"].float().to(device)
                has_depth = batch.get("has_depth", None)

                seg_logits, dep_pred = model(x)
                val_miou += iou_from_logits(seg_logits, seg)

                if has_depth is not None:
                    mask = has_depth.to(device).bool()
                    if mask.any():
                        val_l1_sum += l1(dep_pred[mask], dep[mask]).item()
                        val_l1_count += 1
                else:
                    val_l1_sum += l1(dep_pred, dep).item()
                    val_l1_count += 1

        val_miou /= max(1, len(val_loader))
        val_l1 = (val_l1_sum / val_l1_count) if val_l1_count > 0 else None

        if val_l1 is None:
            print(f"Epoch {epoch:02d} | loss {tr_loss:.4f} | train_mIoU {tr_miou:.3f} | val_mIoU {val_miou:.3f} | val_L1 N/A | {time.time()-t0:.1f}s")
        else:
            print(f"Epoch {epoch:02d} | loss {tr_loss:.4f} | train_mIoU {tr_miou:.3f} | val_mIoU {val_miou:.3f} | val_L1 {val_l1:.4f} | {time.time()-t0:.1f}s")

        if val_miou > best_miou:
            best_miou = val_miou
            torch.save(model.state_dict(), args.save)
            print(f"  Saved best model to {args.save} (val_mIoU={val_miou:.3f})")

    print(f"Best val_mIoU: {best_miou:.3f}")

if __name__ == "__main__":
    main()
