"""
Tolerant SuperTuxKart Drive Dataset loader.

Goals:
- No hard requirement for an 'images/' subfolder.
- Accept frames as filenames or numeric indices.
- If track in info.npz is (H,W) for the episode, tile it to frame count.
- If depth.npz missing, provide zeros and set has_depth=0.
- Output per item:
  {
    "image": (3,96,128) float32 normalized,
    "depth": (96,128) float32 in [0,1] (zeros if missing),
    "track": (96,128) int64 {0,1,2},
    "has_depth": uint8 scalar {0,1}
  }
"""

from __future__ import annotations
import os, glob
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.functional as TF
from torchvision import transforms

# Normalization from README
INPUT_MEAN = [0.2788, 0.2657, 0.2629]
INPUT_STD  = [0.2064, 0.1944, 0.2252]

TARGET_HW = (96, 128)  # (H, W)

def _read_npz(path: Path) -> Dict[str, np.ndarray]:
    with np.load(path, allow_pickle=True) as z:
        return {k: z[k] for k in z.files}

def _ensure_3d(arr: np.ndarray) -> np.ndarray:
    """If arr is (H,W), make it (1,H,W)."""
    arr = np.asarray(arr)
    if arr.ndim == 2:
        arr = arr[None, ...]
    return arr

def _is_str_array(a: np.ndarray) -> bool:
    return isinstance(a, np.ndarray) and a.dtype.kind in ("U", "S", "O")

def _list_images_recursive(root: Path) -> List[Path]:
    pats = ["**/*.png", "**/*.jpg", "**/*.jpeg"]
    res: List[Path] = []
    for pat in pats:
        res.extend(sorted(root.glob(pat)))
    return res

def _index_to_filename(idx: int, episode_dir: Path) -> Optional[Path]:
    """Common STK naming patterns anywhere under episode_dir."""
    candidates = [
        f"{idx:06d}.png", f"{idx:05d}.png", f"{idx:04d}.png",
        f"frame_{idx:06d}.png", f"frame_{idx:05d}.png", f"frame_{idx:04d}.png",
        f"img_{idx:06d}.png",   f"img_{idx:05d}.png",   f"img_{idx:04d}.png",
    ]
    for c in candidates:
        found = list(episode_dir.rglob(c))
        if found:
            return found[0]
    return None

class DriveEpisode:
    """
    A single episode directory that contains:
      - info.npz (must exist; keys: frames, track)
      - optional depth.npz
      - image files somewhere inside the episode folder
    """
    def __init__(self, episode_dir: Path):
        self.episode_dir = episode_dir
        info_p = episode_dir / "info.npz"
        if not info_p.exists():
            raise FileNotFoundError(f"Missing info.npz in {episode_dir}")

        info = _read_npz(info_p)
        frames = info.get("frames", None)
        if frames is None:
            raise KeyError(f"'frames' not found in {info_p}")

        track = info.get("track", None)
        if track is None:
            raise KeyError(f"'track' not found in {info_p}")
        track = np.asarray(track)
        track = _ensure_3d(track).astype(np.uint8, copy=False)  # (N_t,H,W) or (1,H,W)

        # Resolve image paths
        self.image_paths: List[Path] = []
        if _is_str_array(frames):
            # Treat as paths; resolve relative to episode_dir
            for f in frames:
                rel = str(f)
                p = (episode_dir / rel) if not rel.startswith("/") else Path(rel)
                if not p.exists():
                    # try to find by filename anywhere under episode_dir
                    cand = list(episode_dir.rglob(Path(rel).name))
                    if not cand:
                        raise FileNotFoundError(f"Image path from frames not found: {rel} in {episode_dir}")
                    p = cand[0]
                self.image_paths.append(p)
        else:
            # Assume integer indices -> try to map indices to real files
            # First, scan all images in the episode recursively
            all_imgs = _list_images_recursive(episode_dir)
            if not all_imgs:
                # try common subfolder names if recursive somehow blocked
                for sub in ("images", "rgb", "frames"):
                    all_imgs = _list_images_recursive(episode_dir / sub)
                    if all_imgs:
                        break
            # If still none, resolve per-index by name patterns
            if not all_imgs:
                # Fallback to direct patterns for first 20000 indices (safe upper bound)
                # but we cannot guess max; require at least some index match
                # Try to resolve the first 2000 frames
                for i in range(2000):
                    p = _index_to_filename(i, episode_dir)
                    if p is not None:
                        self.image_paths.append(p)
                if not self.image_paths:
                    raise FileNotFoundError(f"No images found under episode {episode_dir}")
            else:
                # If we have a full scan, keep it sorted as the sequence
                self.image_paths = all_imgs

        # Depth — optional
        depth_path = episode_dir / "depth.npz"
        has_depth = depth_path.exists()
        if has_depth:
            depth_npz = _read_npz(depth_path)
            depth = depth_npz.get("depth") or depth_npz.get("depths")
            if depth is None:
                has_depth = False
        if has_depth:
            depth = _ensure_3d(np.asarray(depth)).astype(np.float32, copy=False)  # (N_d,H,W)
        else:
            # dummy zeros; we will set has_depth flag to 0 so trainer can skip the loss
            # We'll shape-align after we know frame count.
            depth = None

        # Frame count is number of images we actually found
        N_frames = len(self.image_paths)
        if N_frames == 0:
            raise FileNotFoundError(f"No image frames resolved in {episode_dir}")

        # Align track to N_frames: if only 1 mask given, tile it
        if track.shape[0] == 1 and N_frames > 1:
            track = np.repeat(track, N_frames, axis=0)
        else:
            # if track shorter/longer than frames, clip
            track = track[:N_frames]

        # Align depth similarly
        if has_depth:
            if depth.shape[0] == 1 and N_frames > 1:
                depth = np.repeat(depth, N_frames, axis=0)
            else:
                depth = depth[:N_frames]
        else:
            depth = np.zeros_like(track, dtype=np.float32)

        self.N = min(N_frames, track.shape[0], depth.shape[0])
        self.image_paths = self.image_paths[: self.N]
        self.track = track[: self.N]
        self.depth = depth[: self.N]
        self.has_depth = bool(has_depth)

    def __len__(self) -> int:
        return self.N

class DriveDataset(Dataset):
    def __init__(self, root_dir: str | Path, split: str = "train", transform_pipeline: Optional[str] = None):
        self.root_dir = Path(root_dir)
        self.split = split
        self.split_dir = self.root_dir / split
        if not self.split_dir.exists():
            raise FileNotFoundError(f"Split directory not found: {self.split_dir}")

        # Find episodes that contain info.npz somewhere inside
        episode_dirs = sorted({p.parent for p in self.split_dir.rglob("info.npz")})
        if not episode_dirs:
            raise FileNotFoundError(f"No episodes (info.npz) found under {self.split_dir}")

        self.episodes: List[DriveEpisode] = []
        for ep in episode_dirs:
            try:
                self.episodes.append(DriveEpisode(ep))
            except Exception as e:
                print(f"[DriveDataset] Warning: skipping episode {ep}: {e}")

        if not self.episodes:
            raise RuntimeError(f"All episodes under {self.split_dir} failed to load.")

        # Global index
        self.index: List[Tuple[int, int]] = []
        for ei, ep in enumerate(self.episodes):
            for fi in range(len(ep)):
                self.index.append((ei, fi))

        self.do_hflip = (split == "train" and transform_pipeline in ("hflip", "flip", "strong"))
        self.normalize = transforms.Normalize(mean=INPUT_MEAN, std=INPUT_STD)

    def __len__(self) -> int:
        return len(self.index)

    def _load_triplet(self, ep: DriveEpisode, fidx: int):
        img_path = ep.image_paths[fidx]
        with Image.open(img_path) as im:
            im = im.convert("RGB")
        depth = ep.depth[fidx]
        track = ep.track[fidx]
        return im, depth, track, ep.has_depth

    def _apply_transforms(self, im: Image.Image, depth_np: np.ndarray, track_np: np.ndarray):
        dep_img = Image.fromarray(depth_np.astype(np.float32), mode="F")
        trk_img = Image.fromarray(track_np.astype(np.uint8), mode="L")

        if self.do_hflip and np.random.rand() < 0.5:
            im = TF.hflip(im); dep_img = TF.hflip(dep_img); trk_img = TF.hflip(trk_img)

        H, W = TARGET_HW
        im = TF.resize(im, [H, W], interpolation=transforms.InterpolationMode.BILINEAR)
        dep_img = TF.resize(dep_img, [H, W], interpolation=transforms.InterpolationMode.BILINEAR)
        trk_img = TF.resize(trk_img, [H, W], interpolation=transforms.InterpolationMode.NEAREST)

        img_t = TF.to_tensor(im)
        img_t = self.normalize(img_t)

        depth_t = torch.from_numpy(np.array(dep_img, dtype=np.float32)).clamp_(0.0, 1.0)
        track_t = torch.from_numpy(np.array(trk_img, dtype=np.uint8)).long()
        return img_t, depth_t, track_t

    def __getitem__(self, idx: int):
        ep_idx, fidx = self.index[idx]
        ep = self.episodes[ep_idx]
        im, depth_np, track_np, has_depth = self._load_triplet(ep, fidx)
        img_t, depth_t, track_t = self._apply_transforms(im, depth_np, track_np)
        return {
            "image": img_t,
            "depth": depth_t,
            "track": track_t,
            "has_depth": torch.tensor(1 if has_depth else 0, dtype=torch.uint8),
        }

def load_data(
    dataset_path: str | Path,
    split: str = "train",
    transform_pipeline: Optional[str] = None,
    batch_size: int = 16,
    num_workers: int = 2,
    shuffle: bool = True,
    return_dataloader: bool = True,
):
    ds = DriveDataset(dataset_path, split=split, transform_pipeline=transform_pipeline)

    if not return_dataloader:
        return ds

    do_shuffle = (shuffle and split == "train")
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=do_shuffle,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=(split == "train"),
    )
