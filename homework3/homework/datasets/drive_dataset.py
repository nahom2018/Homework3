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


MASK_DIRS = ["track", "tracks", "seg", "segs", "masks", "labels", "lane", "annotation", "annotations"]
DEPTH_DIRS = ["depth", "depths"]
MASK_KEYWORDS = ["track", "seg", "mask", "label", "lane", "annot"]
DEPTH_KEYWORDS = ["depth"]
IMG_EXTS = {".jpg", ".jpeg", ".png"}
MASK_EXTS = {".png", ".jpg", ".jpeg", ".npy"}
DEPTH_EXTS = {".png", ".jpg", ".jpeg"}

DRIVE_DEBUG = os.environ.get("DRIVE_DEBUG", "0") == "1"


# ---------- Helpers ----------

def _strip_known_suffixes(stem: str) -> str:
    """Remove common trailing tokens used in image filenames (e.g., *_im, *_rgb, *_image)."""
    for suf in ["_im", "_rgb", "_image"]:
        if stem.endswith(suf):
            return stem[: -len(suf)]
    return stem


def _base_id_from_image_path(img_path: Path) -> str:
    """Derive a base id usable to find matching mask/depth."""
    stem = img_path.stem  # filename without extension
    return _strip_known_suffixes(stem)


def _first_existing(cands: List[Path]) -> Optional[Path]:
    for p in cands:
        if p.exists():
            return p
    return None


def _find_files_with_keywords(root: Path, base: str, dirs: List[str], keywords: List[str], exts: set) -> List[Path]:
    """Search in root and provided subdirs for files whose name contains any keyword and base id."""
    cands: List[Path] = []

    # same-folder candidates
    for kw in keywords + [""]:
        for ext in exts:
            cands.append(root / f"{base}_{kw}{ext}" if kw else root / f"{base}{ext}")

    # subfolder candidates
    for sd in dirs:
        for kw in keywords + [""]:
            for ext in exts:
                cands.append(root / sd / f"{base}_{kw}{ext}" if kw else root / sd / f"{base}{ext}")

    # de-duplicate while preserving order
    seen = set()
    uniq = []
    for p in cands:
        if p not in seen:
            uniq.append(p)
            seen.add(p)
    return [p for p in uniq if p.exists()]


def _find_mask_and_depth(img_path: Path) -> Tuple[Optional[Path], Optional[Path]]:
    """
    Try to find matching mask/depth files near an image using many filename patterns and folders.
    Returns (mask_path or None, depth_path or None).
    """
    base = _base_id_from_image_path(img_path)
    root = img_path.parent

    # Gather candidates
    mask_cands = _find_files_with_keywords(root, base, MASK_DIRS, MASK_KEYWORDS, MASK_EXTS)
    depth_cands = _find_files_with_keywords(root, base, DEPTH_DIRS, DEPTH_KEYWORDS, DEPTH_EXTS)

    # Filter out obvious mis-matches: don't let 'depth' slip into mask list, and vice versa
    mask_cands = [p for p in mask_cands if "depth" not in p.name.lower()]
    depth_cands = [p for p in depth_cands if any(k in p.name.lower() for k in DEPTH_KEYWORDS)]

    # Heuristic: if nothing matched by keywords, try any file with the same base but correct ext
    if not mask_cands:
        for ext in MASK_EXTS:
            p = root / f"{base}{ext}"
            if p.exists() and "depth" not in p.name.lower():
                mask_cands.append(p)
                break
        for sd in MASK_DIRS:
            for ext in MASK_EXTS:
                p = root / sd / f"{base}{ext}"
                if p.exists() and "depth" not in p.name.lower():
                    mask_cands.append(p)
                    break
            if mask_cands:
                break

    if not depth_cands:
        # look for any file with 'depth' in name regardless of ext or directory
        for ext in DEPTH_EXTS:
            p = root / f"{base}_depth{ext}"
            if p.exists():
                depth_cands.append(p)
                break
        if not depth_cands:
            for sd in DEPTH_DIRS:
                for ext in DEPTH_EXTS:
                    p = root / sd / f"{base}_depth{ext}"
                    if p.exists():
                        depth_cands.append(p)
                        break
                if depth_cands:
                    break
        # final fallback: base.ext but only if nothing else exists and name includes 'depth'
        if not depth_cands:
            for ext in DEPTH_EXTS:
                p = root / f"{base}{ext}"
                if p.exists() and "depth" in p.name.lower():
                    depth_cands.append(p)
                    break

    mask_path = mask_cands[0] if mask_cands else None
    depth_path = depth_cands[0] if depth_cands else None

    if DRIVE_DEBUG and (mask_path is None or depth_path is None):
        print(f"[DEBUG] For image {img_path.name}: mask={mask_path}, depth={depth_path}")

    return mask_path, depth_path


# ---------- Dataset ----------

class DriveDataset(Dataset):
    """
    Expects episodes like:
      drive_data/train/<episode>/<image files>
    Matching mask/depth files may be in the same folder or common subfolders.

    If require_masks=True, samples without masks are skipped.
    """

    def __init__(self, root, image_size=(96, 128), require_masks=True):
        self.root = Path(root)
        self.image_size = image_size
        self.require_masks = require_masks
        self.samples: List[dict] = []

        if not self.root.is_dir():
            raise FileNotFoundError(f"Root not found: {root}")

        # Prefer *_im.(jpg|png) but also accept any (jpg|png)
        im_paths = sorted(list(self.root.glob("*/*_im.jpg"))) + \
                   sorted(list(self.root.glob("*/*_im.png")))
        if not im_paths:
            im_paths = sorted(list(self.root.glob("*/*.jpg"))) + \
                       sorted(list(self.root.glob("*/*.png")))
        if not im_paths:
            raise FileNotFoundError(f"No image files found under {root}")

        n_total = len(im_paths)
        n_with_mask = 0
        n_with_depth = 0

        for imf in im_paths:
            maskf, depthf = _find_mask_and_depth(imf)

            if depthf is not None:
                n_with_depth += 1
            if maskf is not None:
                n_with_mask += 1

            # Decide whether to keep
            if depthf is None:
                # Depth required for assignment; skip
                continue
            if self.require_masks and (maskf is None):
                # Skip when masks are required
                continue

            self.samples.append({"img": imf, "mask": maskf, "depth": depthf})

        if not self.samples:
            raise FileNotFoundError(
                f"No samples with masks found under {root}. "
                f"Ensure mask files exist (e.g., *_track.png or under track/)."
            )

        print(f"[DriveDataset] Scanned {n_total} images | found depth for {n_with_depth} | masks for {n_with_mask} (require_masks={self.require_masks})")
        print(f"[DriveDataset] Loaded {len(self.samples)} samples from {root}")

        if DRIVE_DEBUG:
            # Print a few examples so you can verify mapping
            for rec in self.samples[:5]:
                print(f"[DEBUG] img={rec['img'].name} | mask={rec['mask'].name if rec['mask'] else None} | depth={rec['depth'].name}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        rec = self.samples[idx]

        # --- Image: float [0,1] ---
        img = Image.open(rec["img"]).convert("RGB")
        img = TF.to_tensor(img)
        if self.image_size is not None:
            img = TF.resize(img, size=[self.image_size[0], self.image_size[1]],
                            interpolation=InterpolationMode.BILINEAR)

        # --- Depth: float [0,1] ---
        dpath = rec["depth"]
        if dpath.suffix.lower() == ".npy":
            d = np.load(dpath)
            if d.ndim == 3:  # HxWxC -> take first channel
                d = d[..., 0]
            d = np.clip(d, 0.0, 1.0).astype(np.float32)
            d = torch.from_numpy(d).unsqueeze(0)  # 1xH xW
        else:
            dimg = Image.open(dpath).convert("L")
            d = TF.to_tensor(dimg)  # 1xH xW in [0,1]
        if self.image_size is not None:
            d = TF.resize(d, size=[self.image_size[0], self.image_size[1]],
                          interpolation=InterpolationMode.BILINEAR)
        depth = d.squeeze(0)  # HxW

        # --- Mask: integer labels (0/1/2), resize NEAREST ---
        mask = None
        mpath = rec["mask"]
        if mpath is not None:
            if mpath.suffix.lower() == ".npy":
                m = np.load(mpath)
                if m.ndim == 3:
                    m = m[..., 0]
                m = m.astype(np.uint8)
            else:
                mimg = Image.open(mpath).convert("L")
                m = np.array(mimg, dtype=np.uint8)
            mask = torch.from_numpy(m).long()
            if self.image_size is not None:
                mask = mask.unsqueeze(0).float()
                mask = TF.resize(mask, size=[self.image_size[0], self.image_size[1]],
                                 interpolation=InterpolationMode.NEAREST)
                mask = mask.squeeze(0).to(dtype=torch.long)

        out = {"image": img, "depth": depth}
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
