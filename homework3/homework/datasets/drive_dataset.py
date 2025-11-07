# homework/datasets/drive_dataset.py
import os
from pathlib import Path
from typing import Optional, Tuple, List

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms.functional as TF
from torchvision.transforms import InterpolationMode


# ---------- Helpers ----------

def _strip_known_suffixes(stem: str) -> str:
    """
    Remove common trailing tokens used in filenames, e.g. *_im, *_rgb, *_image.
    """
    for suf in ["_im", "_rgb", "_image"]:
        if stem.endswith(suf):
            return stem[: -len(suf)]
    return stem


def _base_id_from_image_path(img_path: Path) -> str:
    """
    Derive a base id usable to find matching mask/depth.
    Examples:
      00000_im.jpg   -> 00000
      frame_001_rgb.png -> frame_001
      00042.png      -> 00042
    """
    stem = img_path.stem  # filename without extension
    stem = _strip_known_suffixes(stem)
    return stem


def _first_existing(candidates: List[Path]) -> Optional[Path]:
    for p in candidates:
        if p.exists():
            return p
    return None


def _find_mask_and_depth(img_path: Path) -> Tuple[Optional[Path], Optional[Path]]:
    """
    Try to find matching mask/depth files near an image.
    Supports same-folder names and common subfolders like
    'track/', 'seg/', 'masks/', 'labels/', 'lane/', 'depth/'.
    Returns (mask_path or None, depth_path or None).
    """
    base = _base_id_from_image_path(img_path)
    root = img_path.parent

    # Same-folder candidates
    mask_candidates = [
        root / f"{base}_track.png",
        root / f"{base}_seg.png",
        root / f"{base}_mask.png",
        root / f"{base}_labels.png",
        root / f"{base}_lane.png",
        root / f"{base}_lane_mask.png",
        root / f"{base}.png",  # plain base.png
    ]
    depth_candidates = [
        root / f"{base}_depth.png",
        root / f"{base}_depth.jpg",
        root / f"{base}.png",   # sometimes depth is just base.png
        root / f"{base}.jpg",
    ]

    # Subfolder candidates
    for sd in ["track", "seg", "masks", "labels", "lane"]:
        mask_candidates += [
            root / sd / f"{base}.png",
            root / sd / f"{base}_track.png",
            root / sd / f"{base}_seg.png",
            root / sd / f"{base}_mask.png",
        ]
    for sd in ["depth", "depths"]:
        depth_candidates += [
            root / sd / f"{base}.png",
            root / sd / f"{base}.jpg",
            root / sd / f"{base}_depth.png",
            root / sd / f"{base}_depth.jpg",
        ]

    return _first_existing(mask_candidates), _first_existing(depth_candidates)


# ---------- Dataset ----------

class DriveDataset(Dataset):
    """
    Expects episodes like:
      drive_data/train/<episode>/<image files>
    where matching mask/depth files are in the same folder or in common subfolders.

    If require_masks=True, samples without masks are skipped.
    """

    def __init__(self, root, image_size=(96, 128), require_masks=True):
        self.root = Path(root)
        self.image_size = image_size
        self.require_masks = require_masks
        self.samples = []

        if not self.root.is_dir():
            raise FileNotFoundError(f"Root not found: {root}")

        # Find image files: prefer *_im.{jpg,png}, but also accept plain {jpg,png}
        im_paths = sorted(list(self.root.glob("*/*_im.jpg"))) + \
                   sorted(list(self.root.glob("*/*_im.png")))
        if not im_paths:
            # Fall back to any jpg/png under episode folders
            im_paths = sorted(list(self.root.glob("*/*.jpg"))) + \
                       sorted(list(self.root.glob("*/*.png")))
        if not im_paths:
            raise FileNotFoundError(f"No image files found under {root}")

        n_total, n_with_mask, n_with_depth = 0, 0, 0

        for imf in im_paths:
            n_total += 1
            maskf, depthf = _find_mask_and_depth(imf)

            if depthf is not None:
                n_with_depth += 1

            if self.require_masks:
                if maskf is None:
                    # skip when masks are required
                    continue
                else:
                    n_with_mask += 1

            # If depth missing, skip (depth is required for assignment)
            if depthf is None:
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

        print(f"[DriveDataset] Scanned {n_total} images, found depth for {n_with_depth}, masks for {n_with_mask} (require_masks={self.require_masks})")
        print(f"[DriveDataset] Loaded {len(self.samples)} samples from {root}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        rec = self.samples[idx]

        # --- Image ---
        img = Image.open(rec["img"]).convert("RGB")
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
        # Provide 'track' always for trainer; background-only if mask missing
        if mask is not None:
            out["track"] = mask
        else:
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
