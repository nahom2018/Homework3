

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

# Normalization used in README / rest of homework
INPUT_MEAN = [0.2788, 0.2657, 0.2629]
INPUT_STD  = [0.2064, 0.1944, 0.2252]

# Target size for the drive dataset per README
TARGET_HW = (96, 128)  # (H, W)


def _read_npz(path: Path) -> Dict[str, np.ndarray]:
    with np.load(path, allow_pickle=True) as z:
        return {k: z[k] for k in z.files}


def _list_images(images_dir: Path) -> List[Path]:
    if not images_dir.exists():
        return []
    return sorted([p for p in images_dir.iterdir() if p.suffix.lower() in [".png", ".jpg", ".jpeg"]])


def _index_to_filename(idx: int, images_dir: Path) -> Optional[Path]:
    """
    Try a few common SuperTuxKart filename patterns.
    Returns a valid path or None if none match.
    """
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
    """
    If arr is (H,W), expand to (1,H,W). If already (N,H,W), return as-is.
    """
    if arr.ndim == 2:
        return arr[None, ...]
    return arr


class DriveEpisode:
    """
    Helper container for a single episode directory.
    Gathers:
      - list of frame image paths (List[Path]) length N
      - track masks: np.ndarray (N, H, W) with {0,1,2}
      - depth maps: np.ndarray (N, H, W) in [0,1]
    """
    def __init__(self, episode_dir: Path):
        self.episode_dir = episode_dir
        info_p = episode_dir / "info.npz"
        depth_p = episode_dir / "depth.npz"
        images_dir = episode_dir / "images"

        if not info_p.exists():
            raise FileNotFoundError(f"Missing info.npz in {episode_dir}")
        if not depth_p.exists():
            raise FileNotFoundError(f"Missing depth.npz in {episode_dir}")
        if not images_dir.exists():
            raise FileNotFoundError(f"Missing images/ folder in {episode_dir}")

        info = _read_npz(info_p)
        depth_npz = _read_npz(depth_p)

        # Extract arrays (be defensive about key names / shapes)
        frames = info.get("frames", None)
        if frames is None:
            raise KeyError(f"'frames' not found in {info_p}")

        track = info.get("track", None)
        if track is None:
            raise KeyError(f"'track' not found in {info_p}")
        track = _ensure_3d(np.asarray(track))  # (N,H,W)

        depth = depth_npz.get("depth", None)
        if depth is None:
            # occasionally seen as 'depths'
            depth = depth_npz.get("depths", None)
        if depth is None:
            raise KeyError(f"'depth' not found in {depth_p}")
        depth = _ensure_3d(np.asarray(depth))  # (N,H,W)

        # Build list of image paths aligned with frames length
        self.image_paths: List[Path] = []
        N = max(track.shape[0], depth.shape[0])

        # frames can be a list of file names or numeric indices
        if isinstance(frames, np.ndarray) and frames.dtype.kind in ("U", "S", "O"):
            # strings/object: treat as relative paths
            for i in range(len(frames)):
                # normalize to Path
                rel = str(frames[i])
                p = (episode_dir / rel) if not rel.startswith("images/") else (episode_dir / rel)
                if not p.exists():
                    # maybe frames hold only the file name under images/
                    candidate = images_dir / rel
                    if candidate.exists():
                        p = candidate
                    else:
                        raise FileNotFoundError(f"Image path from frames not found: {p}")
                self.image_paths.append(p)
            N = min(N, len(self.image_paths))
        else:
            # assume integer frame indices -> try to resolve to filenames under images/
            listed = _list_images(images_dir)
            if listed:
                # assume natural sort already matches frame order
                if len(listed) >= N:
                    self.image_paths = listed[:N]
                else:
                    # fallback: try file-per-index resolution
                    for i in range(N):
                        p = _index_to_filename(i, images_dir)
                        if p is None:
                            raise FileNotFoundError(
                                f"Could not resolve image filename for index {i} in {images_dir}"
                            )
                        self.image_paths.append(p)
            else:
                # no images found by listing; resolve individually
                for i in range(N):
                    p = _index_to_filename(i, images_dir)
                    if p is None:
                        raise FileNotFoundError(
                            f"Could not resolve image filename for index {i} in {images_dir}"
                        )
                    self.image_paths.append(p)

        # Now align lengths strictly
        self.N = min(N, track.shape[0], depth.shape[0], len(self.image_paths))
        self.track = track[: self.N]          # (N,H,W)
        self.depth = depth[: self.N]          # (N,H,W)

    def __len__(self) -> int:
        return self.N


class DriveDataset(Dataset):
    """
    Dataset across many episodes for a given split (train/val/test).
    Each __getitem__ returns a dict with keys: image, depth, track.
    """
    def __init__(self, root_dir: str | Path, split: str = "train", transform_pipeline: Optional[str] = None):
        self.root_dir = Path(root_dir)
        self.split = split
        self.split_dir = self.root_dir / split

        if not self.split_dir.exists():
            raise FileNotFoundError(f"Split directory not found: {self.split_dir}")

        # Gather episode directories (contain info.npz)
        episode_dirs = sorted([p for p in self.split_dir.iterdir() if (p / "info.npz").exists()])

        if not episode_dirs:
            # sometimes episodes are nested one level deeper
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
                # If an episode is malformed, skip with a warning
                print(f"[DriveDataset] Warning: skipping episode {ep}: {e}")

        if not self.episodes:
            raise RuntimeError(f"All episodes under {self.split_dir} failed to load.")

        # Build a global index of (ep_idx, frame_idx)
        self.index: List[Tuple[int, int]] = []
        for ep_idx, ep in enumerate(self.episodes):
            for fidx in range(len(ep)):
                self.index.append((ep_idx, fidx))

        # Build transforms
        # We apply geometric transforms consistently to image/depth/track.
        # For simplicity, we just resize to (96,128). Optionally add train-only hflip.
        self.do_hflip = (split == "train" and transform_pipeline in ("hflip", "flip", "strong"))
        self.normalize = transforms.Normalize(mean=INPUT_MEAN, std=INPUT_STD)

    def __len__(self) -> int:
        return len(self.index)

    def _load_triplet(self, ep: DriveEpisode, fidx: int) -> Tuple[Image.Image, np.ndarray, np.ndarray]:
        # Load RGB image
        img_path = ep.image_paths[fidx]
        with Image.open(img_path) as im:
            im = im.convert("RGB")

        depth = ep.depth[fidx]  # (H,W) float 0..1
        track = ep.track[fidx]  # (H,W) labels {0,1,2}

        return im, depth, track

    def _apply_transforms(self, im: Image.Image, depth_np: np.ndarray, track_np: np.ndarray) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Convert depth/track to PIL for consistent geometry ops
        # Depth is continuous -> use mode "F" (32-bit float)
        dep_img = Image.fromarray(depth_np.astype(np.float32), mode="F")
        # Track is categorical -> keep as L (8-bit) but will use nearest interpolation
        trk_img = Image.fromarray(track_np.astype(np.uint8), mode="L")

        # Optional horizontal flip (train only), applied consistently
        if self.do_hflip:
            if np.random.rand() < 0.5:
                im = TF.hflip(im)
                dep_img = TF.hflip(dep_img)
                trk_img = TF.hflip(trk_img)

        # Resize to target resolution
        H, W = TARGET_HW
        im = TF.resize(im, size=[H, W], interpolation=transforms.InterpolationMode.BILINEAR)
        dep_img = TF.resize(dep_img, size=[H, W], interpolation=transforms.InterpolationMode.BILINEAR)
        trk_img = TF.resize(trk_img, size=[H, W], interpolation=transforms.InterpolationMode.NEAREST)

        # To tensor
        img_t = TF.to_tensor(im)  # (3,H,W) float32 [0,1]
        img_t = self.normalize(img_t)

        # Depth back to numpy -> tensor
        depth_t = torch.from_numpy(np.array(dep_img, dtype=np.float32))  # (H,W)
        # Ensure depth is within [0,1] after resizing
        depth_t.clamp_(0.0, 1.0)

        # Track to tensor (long)
        track_t = torch.from_numpy(np.array(trk_img, dtype=np.uint8)).long()  # (H,W)

        return img_t, depth_t, track_t

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        ep_idx, fidx = self.index[idx]
        ep = self.episodes[ep_idx]
        im, depth_np, track_np = self._load_triplet(ep, fidx)
        img_t, depth_t, track_t = self._apply_transforms(im, depth_np, track_np)
        return {"image": img_t, "depth": depth_t, "track": track_t}


def load_data(
    dataset_path: str | Path,
    split: str = "train",
    transform_pipeline: Optional[str] = None,
    batch_size: int = 16,
    num_workers: int = 2,
    shuffle: bool = True,
    return_dataloader: bool = True,
):
    """
    Create a Dataset or DataLoader for the drive task.
    Returns DataLoader by default; set return_dataloader=False to get the Dataset.
    """
    ds = DriveDataset(dataset_path, split=split, transform_pipeline=transform_pipeline)

    if not return_dataloader:
        return ds

    # Shuffle only for train by default
    do_shuffle = (shuffle and split == "train")
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=do_shuffle,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=(split == "train"),
    )
