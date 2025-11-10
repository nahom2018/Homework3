"""
SuperTuxKart Drive Dataset loader for segmentation (track) + depth (optional).

If depth files are missing, we:
  - return a dummy zeros depth map aligned to track
  - expose has_depth=0 so the trainer can skip the depth loss

Per-sample output:
  {
    "image": (3,96,128) float32 normalized,
    "depth": (96,128) float32 in [0,1] (zeros if missing),
    "track": (96,128) int64 {0,1,2},
    "has_depth": uint8 tensor scalar in {0,1}
  }
"""

from __future__ import annotations
import os
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

# Target size
TARGET_HW = (96, 128)  # (H, W)


def _read_npz(path: Path) -> Dict[str, np.ndarray]:
    with np.load(path, allow_pickle=True) as z:
        return {k: z[k] for k in z.files}


def _list_images(images_dir: Path) -> List[Path]:
    if not images_dir.exists():
        return []
    return sorted([p for p in images_dir.iterdir() if p.suffix.lower() in [".png", ".jpg", ".jpeg"]])


def _index_to_filename(idx: int, images_dir: Path) -> Optional[Path]:
    candidates = [
        images_dir / f"{idx:06d}.png",
        images_dir / f"{idx:05d}.png",
        images_dir / f"{idx:04d}.png",
        images_dir / f"frame_{idx:06d}.png",
        images_dir / f"frame_{idx:05d}.png",
        images_dir / f"frame_{idx:04d}.png",
        images_dir / f"img_{idx:06d}.png",
        images_dir / f"img_{idx:05d}.png",
        images_dir / f"img_{idx:04d}.png",
    ]
    for c in candidates:
        if c.exists():
            return c
    return None


def _ensure_3d(arr: np.ndarray) -> np.ndarray:
    if arr.ndim == 2:
        return arr[None, ...]
    return arr


class DriveEpisode:
    """
    Container for a single episode:
      - image_paths: List[Path] length N
      - track: (N,H,W) uint8 in {0,1,2}
      - depth: (N,H,W) float32 in [0,1] (zeros if missing)
      - has_depth: bool (True if real depth was loaded)
    """
    def __init__(self, episode_dir: Path):
        self.episode_dir = episode_dir
        info_p = episode_dir / "info.npz"
        images_dir = episode_dir / "images"

        if not info_p.exists():
            raise FileNotFoundError(f"Missing info.npz in {episode_dir}")
        if not images_dir.exists():
            raise FileNotFoundError(f"Missing images/ folder in {episode_dir}")

        info = _read_npz(info_p)

        frames = info.get("frames", None)
        if frames is None:
            raise KeyError(f"'frames' not found in {info_p}")

        track = info.get("track", None)
        if track is None:
            raise KeyError(f"'track' not found in {info_p}")
        track = _ensure_3d(np.asarray(track))  # (N,H,W)
        track = track.astype(np.uint8, copy=False)

        # depth: tolerant load; make dummy if absent
        depth_path = episode_dir / "depth.npz"
        has_depth = depth_path.exists()
        if has_depth:
            depth_npz = _read_npz(depth_path)
            depth = depth_npz.get("depth")
            if depth is None:
                depth = depth_npz.get("depths")  # occasional alt key
            if depth is None:
                has_depth = False

        if has_depth:
            depth = _ensure_3d(np.asarray(depth))  # (N,H,W)
            depth = depth.astype(np.float32, copy=False)
        else:
            depth = np.zeros_like(track, dtype=np.float32)

        # Build image path list
        self.image_paths: List[Path] = []
        N = max(track.shape[0], depth.shape[0])

        if isinstance(frames, np.ndarray) and frames.dtype.kind in ("U", "S", "O"):
            # frames has file names / relpaths
            for i in range(len(frames)):
                rel = str(frames[i])
                p = (episode_dir / rel) if not rel.startswith("images/") else (episode_dir / rel)
                if not p.exists():
                    cand = images_dir / rel
                    if cand.exists():
                        p = cand
                    else:
                        raise FileNotFoundError(f"Image path from frames not found: {p}")
                self.image_paths.append(p)
            N = min(N, len(self.image_paths))
        else:
            # integer indices -> try list or pattern-match
            listed = _list_images(images_dir)
            if listed and len(listed) >= N:
                self.image_paths = listed[:N]
            else:
                for i in range(N):
                    p = _index_to_filename(i, images_dir)
                    if p is None:
                        raise FileNotFoundError(
                            f"Could not resolve image filename for index {i} in {images_dir}"
                        )
                    self.image_paths.append(p)

        # Align lengths strictly
        self.N = min(N, track.shape[0], depth.shape[0], len(self.image_paths))
        self.track = track[: self.N]          # (N,H,W)
        self.depth = depth[: self.N]          # (N,H,W)
        self.has_depth = bool(has_depth)

    def __len__(self) -> int:
        return self.N


class DriveDataset(Dataset):
    """
    Dataset across episodes for a split (train/val/test).
    """
    def __init__(self, root_dir: str | Path, split: str = "train", transform_pipeline: Optional[str] = None):
        self.root_dir = Path(root_dir)
        self.split = split
        self.split_dir = self.root_dir / split

        if not self.split_dir.exists():
            raise FileNotFoundError(f"Split directory not found: {self.split_dir}")

        # Gather episode dirs
        episode_dirs = sorted([p for p in self.split_dir.iterdir() if (p / "info.npz").exists()])
        if not episode_dirs:
            for p in self.split_dir.rglob("info.npz"):
                episode_dirs.append(p.parent)
            episode_dirs = sorted(set(episode_dirs))

        if not episode_dirs:
            raise FileNotFoundError(f"No episodes found under {self.split_dir}")

        self.episodes: List[DriveEpisode] = []
        for ep in episode_dirs:
            try:
                self.episodes.append(DriveEpisode(ep))
            except Exception as e:
                print(f"[DriveDataset] Warning: skipping episode {ep}: {e}")

        if not self.episodes:
            raise RuntimeError(f"All episodes under {self.split_dir} failed to load.")

        # Build global index
        self.index: List[Tuple[int, int]] = []
        for ep_idx, ep in enumerate(self.episodes):
            for fidx in range(len(ep)):
                self.index.append((ep_idx, fidx))

        # transforms
        self.do_hflip = (split == "train" and transform_pipeline in ("hflip", "flip", "strong"))
        self.normalize = transforms.Normalize(mean=INPUT_MEAN, std=INPUT_STD)

    def __len__(self) -> int:
        return len(self.index)

    def _load_triplet(self, ep: DriveEpisode, fidx: int):
        img_path = ep.image_paths[fidx]
        with Image.open(img_path) as im:
            im = im.convert("RGB")
        depth = ep.depth[fidx]  # (H,W) float32
        track = ep.track[fidx]  # (H,W) uint8
        return im, depth, track, ep.has_depth

    def _apply_transforms(self, im: Image.Image, depth_np: np.ndarray, track_np: np.ndarray):
        dep_img = Image.fromarray(depth_np.astype(np.float32), mode="F")  # continuous
        trk_img = Image.fromarray(track_np.astype(np.uint8), mode="L")    # categorical

        if self.do_hflip and np.random.rand() < 0.5:
            im = TF.hflip(im)
            dep_img = TF.hflip(dep_img)
            trk_img = TF.hflip(trk_img)

        H, W = TARGET_HW
        im = TF.resize(im, size=[H, W], interpolation=transforms.InterpolationMode.BILINEAR)
        dep_img = TF.resize(dep_img, size=[H, W], interpolation=transforms.InterpolationMode.BILINEAR)
        trk_img = TF.resize(trk_img, size=[H, W], interpolation=transforms.InterpolationMode.NEAREST)

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
