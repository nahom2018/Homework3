"""
Ultra-tolerant SuperTuxKart Drive Dataset loader.

Key behaviors:
- If depth.npz is missing -> provide zeros depth and has_depth=0.
- frames can be filenames or indices; filenames may be relative to episode,
  split, or root. We search all three by basename if needed.
- No hard requirement for an 'images/' folder.
- If track is (H,W), tile to number of frames.
- Per item returns:
  {
    "image": (3,96,128) float32 normalized,
    "depth": (96,128) float32 in [0,1] (zeros if missing),
    "track": (96,128) int64 {0,1,2},
    "has_depth": uint8 scalar {0,1}
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
TARGET_HW = (96, 128)  # (H, W)

def _read_npz(path: Path) -> Dict[str, np.ndarray]:
    with np.load(path, allow_pickle=True) as z:
        return {k: z[k] for k in z.files}

def _ensure_3d(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr)
    if arr.ndim == 2:
        arr = arr[None, ...]
    return arr

def _is_str_array(a: np.ndarray) -> bool:
    return isinstance(a, np.ndarray) and a.dtype.kind in ("U", "S", "O")

def _glob_images(root: Path) -> List[Path]:
    pats = ("**/*.png", "**/*.jpg", "**/*.jpeg")
    out: List[Path] = []
    for pat in pats:
        out.extend(sorted(root.glob(pat)))
    return out

def _index_to_filename_anywhere(idx: int, episode_dir: Path) -> Optional[Path]:
    # Try common patterns anywhere under episode_dir
    names = [
        f"{idx:06d}.png", f"{idx:05d}.png", f"{idx:04d}.png",
        f"frame_{idx:06d}.png", f"frame_{idx:05d}.png", f"frame_{idx:04d}.png",
        f"img_{idx:06d}.png",   f"img_{idx:05d}.png",   f"img_{idx:04d}.png",
    ]
    for n in names:
        hits = list(episode_dir.rglob(n))
        if hits:
            return hits[0]
    return None

def _resolve_frame_path(rel_or_abs: str, episode_dir: Path, split_dir: Path, root_dir: Path) -> Optional[Path]:
    """
    Try to resolve a frames entry to an actual file by testing:
      1) Absolute path as-is
      2) episode_dir / given
      3) search by basename under episode_dir
      4) search by basename under split_dir
      5) search by basename under root_dir
    """
    p = Path(rel_or_abs)
    if p.is_absolute() and p.exists():
        return p

    cand = episode_dir / rel_or_abs
    if cand.exists():
        return cand

    basename = Path(rel_or_abs).name
    for base in (episode_dir, split_dir, root_dir):
        hits = list(base.rglob(basename))
        if hits:
            return hits[0]
    return None


class DriveEpisode:
    """
    A single episode directory that contains:
      - info.npz (must exist; keys at least: frames, track)
      - optional depth.npz
      - images somewhere under the episode folder (or resolvable via frames)
    """
    def __init__(self, episode_dir: Path, split_dir: Path, root_dir: Path):
        self.episode_dir = episode_dir
        self.split_dir = split_dir
        self.root_dir = root_dir

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
        track = _ensure_3d(np.asarray(track, dtype=np.uint8))  # (N_t,H,W) or (1,H,W)

        # depth (optional)
        depth_path = episode_dir / "depth.npz"
        has_depth = depth_path.exists()
        if has_depth:
            depth_npz = _read_npz(depth_path)
            depth = depth_npz.get("depth") or depth_npz.get("depths")
            if depth is None:
                has_depth = False
        if has_depth:
            depth = _ensure_3d(np.asarray(depth, dtype=np.float32))
        else:
            depth = None  # will create zeros later

        # Resolve image paths:
        self.image_paths: List[Path] = []

        if _is_str_array(frames):
            for f in frames:
                p = _resolve_frame_path(str(f), self.episode_dir, self.split_dir, self.root_dir)
                if p is None or not p.exists():
                    raise FileNotFoundError(
                        f"Could not resolve frame path '{f}' for episode {self.episode_dir}"
                    )
                self.image_paths.append(p)
        else:
            # frames likely numeric indices -> resolve via patterns anywhere in episode
            # If episode has lots of images, also accept a simple recursive listing
            all_imgs = _glob_images(self.episode_dir)
            if not all_imgs:
                # try common subfolders if any
                for sub in ("images", "rgb", "frames"):
                    all_imgs = _glob_images(self.episode_dir / sub)
                    if all_imgs:
                        break
            if all_imgs:
                self.image_paths = all_imgs  # assume ordering is sequential
            else:
                # last resort: try mapping first few thousand indices
                for i in range(20000):  # generous upper bound; we slice later
                    p = _index_to_filename_anywhere(i, self.episode_dir)
                    if p:
                        self.image_paths.append(p)
                if not self.image_paths:
                    raise FileNotFoundError(f"No images found for episode {self.episode_dir}")

        N_frames = len(self.image_paths)
        if N_frames == 0:
            raise FileNotFoundError(f"No image frames resolved in {self.episode_dir}")

        # Align track to N_frames
        if track.shape[0] == 1 and N_frames > 1:
            track = np.repeat(track, N_frames, axis=0)
        else:
            track = track[:N_frames]

        # Align depth to N_frames; if missing, zeros shaped like track
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

        # Find episodes by locating info.npz anywhere under the split
        episode_dirs = sorted({p.parent for p in self.split_dir.rglob("info.npz")})
        if not episode_dirs:
            raise FileNotFoundError(f"No episodes (info.npz) found under {self.split_dir}")

        self.episodes: List[DriveEpisode] = []
        for ep in episode_dirs:
            try:
                self.episodes.append(DriveEpisode(ep, self.split_dir, self.root_dir))
            except Exception as e:
                print(f"[DriveDataset] Warning: skipping episode {ep}: {e}")

        if not self.episodes:
            # As a very last resort, try to build a single "episode" directly from all images under split root
            # and a dummy single-mask track if any info.npz exists.
            info_files = list(self.split_dir.rglob("info.npz"))
            if info_files:
                try:
                    # pick one info.npz just to get a track shape
                    info = _read_npz(info_files[0])
                    track = _ensure_3d(np.asarray(info.get("track"), dtype=np.uint8))
                    imgs = _glob_images(self.split_dir)
                    if imgs:
                        N = min(len(imgs), track.shape[0] if track.shape[0] > 1 else len(imgs))
                        if track.shape[0] == 1 and N > 1:
                            track = np.repeat(track, N, axis=0)
                        imgs = imgs[:N]
                        track = track[:N]
                        depth = np.zeros_like(track, dtype=np.float32)

                        # synthesize one episode
                        ep = type("SyntheticEpisode", (), {})()
                        ep.image_paths = imgs
                        ep.track = track
                        ep.depth = depth
                        ep.N = N
                        ep.has_depth = False
                        self.episodes = [ep]  # type: ignore
                        print("[DriveDataset] Fallback: using synthetic episode from split images.")
                    else:
                        raise RuntimeError("No images found for fallback synthetic episode.")
                except Exception as e:
                    raise RuntimeError(f"All episodes under {self.split_dir} failed to load (even fallback): {e}") from e
            else:
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
        img_path = ep.image_paths[fidx]
        with Image.open(img_path) as im:
            im = im.convert("RGB")
        depth_np = ep.depth[fidx]
        track_np = ep.track[fidx]
        img_t, depth_t, track_t = self._apply_transforms(im, depth_np, track_np)
        has_depth = getattr(ep, "has_depth", False)
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
