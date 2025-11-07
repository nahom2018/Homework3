
import os
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms.functional as TF
from torchvision.transforms import InterpolationMode


def _find_mask_and_depth(img_path: Path):
    """
    Try to find matching mask/depth files for an image.
    Supports same-folder names and common subfolders like 'track/', 'seg/', 'masks/', 'labels/', 'lane/', 'depth/'.
    Returns (mask_path or None, depth_path or None).
    """
    base = img_path.name
    # strip common image suffix patterns like *_im.jpg, *_im.png
    if base.endswith("_im.jpg"):
        base = base[:-7]  # remove "_im.jpg"
    elif base.endswith("_im.jpeg"):
        base = base[:-8]
    elif base.endswith("_im.png"):
        base = base[:-7]
    else:
        base = img_path.stem  # generic fallback

    root = img_path.parent

    # Candidates in same folder (most common)
    mask_candidates = [
        root / f"{base}_track.png",
        root / f"{base}_seg.png",
        root / f"{base}_mask.png",
        root / f"{base}_labels.png",
        root / f"{base}_lane.png",
        root / f"{base}_lane_mask.png",
        root / f"{base}.png",  # sometimes just base.png
    ]
    depth_candidates = [
        root / f"{base}_depth.png",
        root / f"{base}_depth.jpg",
        root / f"{base}.png",
        root / f"{base}.jpg",
    ]

    # Candidates in subfolders
    for sd in ["track", "seg", "masks", "labels", "lane"]:
        mask_candidates += [
            root / sd / f"{base}.png",
            root / sd / f"{base}_track.png",
            root / sd / f"{base}_seg.png",
        ]
    for sd in ["depth"]:
        depth_candidates += [
            root / sd / f"{base}.png",
            root / sd / f"{base}.jpg",
            root / sd / f"{base}_depth.png",
            root / sd / f"{base}_depth.jpg",
        ]

    mask_path = next((p for p in mask_candidates if p.exists()), None)
    depth_path = next((p for p in depth_candidates if p.exists()), None)
    return mask_path, depth_path


class DriveDataset(Dataset):
    """
    Expects episodes like:
      drive_data/train/cornfield_crossing_00/00000_im.jpg
      drive_data/train/cornfield_crossing_00/00000_depth.png (or in depth/)
      drive_data/train/cornfield_crossing_00/00000_track.png (or in track/)

    If require_masks=True, samples without masks are skipped.
    """

    def __init__(self, root, image_size=(96, 128), require_masks=True):
        self.root = Path(root)
        self.image_size = image_size
        self.require_masks = require_masks
        self.samples = []

        if not self.root.is_dir():
            raise FileNotFoundError(f"Root not found: {root}")

        # Find image files: *_im.jpg (primary) + fallbacks *_im.png
        im_paths = sorted(list(self.root.glob("*/*_im.jpg"))) + sorted(list(self.root.glob("*/*_im.png")))
        if not im_paths:
            raise FileNotFoundError(f"No '*_im.jpg' (or png) files found under {root}")

        for imf in im_paths:
            maskf, depthf = _find_mask_and_depth(imf)
            # Depth is required for this assignment; skip if missing
            if depthf is None:
                continue
            # If masks are required, skip when missing
            if self.require_masks and maskf is None:
                continue

            self.samples.append({"img": imf, "mask": maskf, "depth": depthf})

        if not self.samples:
            if self.require_masks:
                raise FileNotFoundError(
                    f"No samples with masks found under {root}. "
                    f"Ensure mask files exist (e.g., *_track.png or under track/)."
                )
            else:
                raise FileNotFoundError(f"No (image, depth) pairs found under {root}")

        print(f"[DriveDataset] Loaded {len(self.samples)} samples from {root}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        rec = self.samples[idx]

        # --- Image ---
        img = Image.open(rec["img"]).convert("RGB")
        # to tensor [0,1], then resize bilinear on tensor
        img = TF.to_tensor(img)  # (3,H,W) float in [0,1]
        if self.image_size is not None:
            img = TF.resize(img, size=[self.image_size[0], self.image_size[1]],
                            interpolation=InterpolationMode.BILINEAR)

        # --- Depth: float in [0,1] ---
        d = Image.open(rec["depth"]).convert("L")
        d = TF.to_tensor(d)  # (1,H,W) in [0,1]
        if self.image_size is not None:
            d = TF.resize(d, size=[self.image_size[0], self.image_size[1]],
                          interpolation=InterpolationMode.BILINEAR)
        depth = d.squeeze(0)  # (H,W) float32

        # --- Segmentation mask: integer labels (0/1/2), resize NEAREST, no normalization ---
        mask = None
        if rec["mask"] is not None:
            m = Image.open(rec["mask"]).convert("L")
            m = np.array(m, dtype=np.uint8)           # (H,W) uint8
            mask = torch.from_numpy(m).long()         # (H,W) long in {0,1,2}
            if self.image_size is not None:
                mask = mask.unsqueeze(0).float()      # (1,H,W)
                mask = TF.resize(mask, size=[self.image_size[0], self.image_size[1]],
                                 interpolation=InterpolationMode.NEAREST)
                mask = mask.squeeze(0).to(dtype=torch.long)

        out = {"image": img, "depth": depth}
        # For the trainer/evaluator convenience, include 'track' even if missing (depth-only mode)
        if mask is not None:
            out["track"] = mask
        else:
            # Provide a background-only tensor so shapes are consistent if caller expects it
            H, W = depth.shape
            out["track"] = torch.zeros((H, W), dtype=torch.long)

        return out


def load_data(batch_size=16, num_workers=2, root="drive_data",
              image_size=(96, 128), require_masks=True):
    train_root = Path(root) / "train"
    val_root = Path(root) / "val"
    if not train_root.is_dir() or not val_root.is_dir():
        raise FileNotFoundError(f"Expected {root}/train and {root}/val")

    train_ds = DriveDataset(train_root, image_size=image_size, require_masks=require_masks)
    val_ds = DriveDataset(val_root, image_size=image_size, require_masks=require_masks)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=True)

    return {"train": train_loader, "val": val_loader}
