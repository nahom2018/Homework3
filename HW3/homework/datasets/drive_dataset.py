"""
Ultra-tolerant SuperTuxKart Drive Dataset loader.

- Resolves frames as filenames or indices, searching episode/split/root.
- Accepts many "track" formats:
    * numeric (H,W) or (N,H,W)
    * object arrays / lists of dicts -> extract via keys: 'track','mask','labels','seg','segmentation'
    * dicts with separate 'left' / 'right' boolean masks -> merge to classes 1/2
    * lists of file paths -> load masks from images
- If track is (H,W), tiles to number of frames.
- If depth.npz missing -> use zeros and set has_depth=0.
- Output per item:
    {
      "image": (3,96,128) float32 normalized,
      "depth": (96,128) float32 in [0,1] (zeros if missing),
      "track": (96,128) int64 {0,1,2},
      "has_depth": uint8 scalar {0,1}
    }
"""

from __future__ import annotations
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.functional as TF
from torchvision import transforms

# Normalization from README
INPUT_MEAN = [0.2788, 0.2657, 0.2629]
INPUT_STD  = [0.2064, 0.1944, 0.2252]
TARGET_HW  = (96, 128)  # (H, W)

def _read_npz(path: Path) -> Dict[str, Any]:
    with np.load(path, allow_pickle=True) as z:
        return {k: z[k] for k in z.files}

def _ensure_3d_numeric(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr)
    if arr.ndim == 2:
        arr = arr[None, ...]
    return arr

def _is_str_array(a: Any) -> bool:
    return isinstance(a, (list, tuple)) and len(a) > 0 and isinstance(a[0], (str, bytes)) \
        or (isinstance(a, np.ndarray) and a.dtype.kind in ("U","S","O") and a.size > 0 and isinstance(a.flat[0], (str, bytes)))

def _glob_images(root: Path) -> List[Path]:
    out: List[Path] = []
    for pat in ("**/*.png", "**/*.jpg", "**/*.jpeg"):
        out.extend(sorted(root.glob(pat)))
    return out

def _index_to_filename_anywhere(idx: int, episode_dir: Path) -> Optional[Path]:
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

def _load_mask_image(p: Path) -> np.ndarray:
    with Image.open(p) as im:
        im = im.convert("L")  # grayscale for mask
        m = np.array(im, dtype=np.uint8)
    return m

def _coerce_mask_values_to_012(mask: np.ndarray) -> np.ndarray:
    """Reduce arbitrary integer labels to {0,1,2} if possible."""
    mask = np.asarray(mask)
    if mask.dtype != np.uint8:
        # try safe cast
        mask = mask.astype(np.int64, copy=False)
    uniq = np.unique(mask)
    if set(uniq.tolist()).issubset({0,1,2}):
        return mask.astype(np.uint8, copy=False)
    # map sorted unique values to 0..K-1, then clip to 0/1/2
    mapping = {v:i for i,v in enumerate(sorted(uniq.tolist()))}
    mapped = np.vectorize(mapping.get)(mask)
    mapped = np.clip(mapped, 0, 2).astype(np.uint8)
    return mapped

def _merge_left_right_to_classes(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """Make class map: 0=bg, 1=left, 2=right. If overlap, right wins."""
    h, w = left.shape
    out = np.zeros((h,w), dtype=np.uint8)
    out[left.astype(bool)] = 1
    out[right.astype(bool)] = 2
    return out

def _extract_track_from_object_array(arr: Any, episode_dir: Path, split_dir: Path, root_dir: Path) -> np.ndarray:
    """
    Handle object-like tracks:
      - list/array of dicts (per frame)
      - list/array of file paths (per frame)
      - single dict with 'left'/'right' or a 'mask'
    Returns (N,H,W) uint8 in {0,1,2}.
    """
    # Normalize to Python list for iteration
    if isinstance(arr, np.ndarray) and arr.dtype == object:
        arr_list = list(arr)
    elif isinstance(arr, (list, tuple)):
        arr_list = list(arr)
    elif isinstance(arr, dict):
        arr_list = [arr]
    else:
        raise ValueError("Unsupported 'track' object format")

    # Case: list of strings -> file paths for masks
    if len(arr_list) > 0 and isinstance(arr_list[0], (str, bytes)):
        masks: List[np.ndarray] = []
        for s in arr_list:
            p = _resolve_frame_path(str(s), episode_dir, split_dir, root_dir)
            if p is None or not p.exists():
                raise FileNotFoundError(f"Mask path not found: {s}")
            m = _load_mask_image(p)
            masks.append(_coerce_mask_values_to_012(m))
        return np.stack(masks, axis=0)  # (N,H,W)

    # Case: list of dicts (per frame)
    if len(arr_list) > 0 and isinstance(arr_list[0], dict):
        masks: List[np.ndarray] = []
        for item in arr_list:
            # Try common keys for a categorical mask
            for k in ("track","mask","labels","seg","segmentation"):
                if k in item:
                    m = np.array(item[k])
                    if m.ndim == 3:  # squeeze channel if (H,W,1)
                        if m.shape[-1] == 1:
                            m = m[...,0]
                    if m.ndim != 2:
                        raise ValueError(f"Per-frame mask under key '{k}' must be (H,W), got {m.shape}")
                    masks.append(_coerce_mask_values_to_012(m))
                    break
            else:
                # Try left/right boolean maps
                if "left" in item and "right" in item:
                    left = np.array(item["left"]).astype(bool)
                    right = np.array(item["right"]).astype(bool)
                    if left.ndim == 3 and left.shape[-1] == 1: left = left[...,0]
                    if right.ndim == 3 and right.shape[-1] == 1: right = right[...,0]
                    if left.ndim != 2 or right.ndim != 2:
                        raise ValueError("left/right masks must be (H,W)")
                    masks.append(_merge_left_right_to_classes(left, right))
                else:
                    raise ValueError("Dict track item missing known keys (track/mask/labels/seg/segmentation or left/right).")
        return np.stack(masks, axis=0)

    # Case: single dict (episode-level) -> try same keys
    if len(arr_list) == 1 and isinstance(arr_list[0], dict):
        item = arr_list[0]
        for k in ("track","mask","labels","seg","segmentation"):
            if k in item:
                m = np.array(item[k])
                m = _ensure_3d_numeric(m)
                m = np.squeeze(m) if m.ndim == 3 and m.shape[0] == 1 else m
                if m.ndim == 2:
                    return m[None, ...].astype(np.uint8)
                if m.ndim == 3:
                    return np.array([_coerce_mask_values_to_012(mi) for mi in m], dtype=np.uint8)
        if "left" in item and "right" in item:
            left = np.array(item["left"]).astype(bool)
            right = np.array(item["right"]).astype(bool)
            if left.ndim == 3 and left.shape[-1] == 1: left = left[...,0]
            if right.ndim == 3 and right.shape[-1] == 1: right = right[...,0]
            if left.ndim == 2 and right.ndim == 2:
                return _merge_left_right_to_classes(left, right)[None, ...]
        raise ValueError("Unsupported single-dict 'track' structure.")
    raise ValueError("Unsupported 'track' object structure.")

class DriveEpisode:
    """
    Episode containing:
      - info.npz (keys: frames, track; track can be varied format)
      - optional depth.npz (ignored if missing)
      - images resolvable via frames or recursive search
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

        # --- Parse track robustly ---
        raw_track = info.get("track", None)
        if raw_track is None:
            raise KeyError(f"'track' not found in {info_p}")

        try:
            # Simple numeric case
            track = _ensure_3d_numeric(np.asarray(raw_track, dtype=np.uint8))
        except (TypeError, ValueError):
            # Object/dict/list cases
            track = _extract_track_from_object_array(raw_track, episode_dir, split_dir, root_dir)
        # track now (N_t,H,W) uint8
        # sanitize label values to 0/1/2
        if track.dtype != np.uint8:
            track = track.astype(np.int64, copy=False)
        uniq = np.unique(track)
        if not set(uniq.tolist()).issubset({0,1,2}):
            track = np.array([_coerce_mask_values_to_012(t) for t in track], dtype=np.uint8)

        # --- Resolve image paths ---
        self.image_paths: List[Path] = []
        if _is_str_array(frames):
            for f in (list(frames) if not isinstance(frames, list) else frames):
                p = _resolve_frame_path(str(f), self.episode_dir, self.split_dir, self.root_dir)
                if p is None or not p.exists():
                    raise FileNotFoundError(f"Frame path not found: {f} in {self.episode_dir}")
                self.image_paths.append(p)
        else:
            # numeric indices -> prefer recursive listing; else patterns anywhere
            all_imgs = _glob_images(self.episode_dir)
            if not all_imgs:
                for sub in ("images", "rgb", "frames"):
                    all_imgs = _glob_images(self.episode_dir / sub)
                    if all_imgs:
                        break
            if all_imgs:
                self.image_paths = all_imgs
            else:
                for i in range(20000):
                    p = _index_to_filename_anywhere(i, self.episode_dir)
                    if p: self.image_paths.append(p)
                if not self.image_paths:
                    raise FileNotFoundError(f"No images found in episode {self.episode_dir}")

        N_frames = len(self.image_paths)
        if N_frames == 0:
            raise FileNotFoundError(f"No image frames resolved in {self.episode_dir}")

        # Align track frames
        if track.shape[0] == 1 and N_frames > 1:
            track = np.repeat(track, N_frames, axis=0)
        else:
            track = track[:N_frames]

        # --- Depth (optional) ---
        depth_path = episode_dir / "depth.npz"
        has_depth = depth_path.exists()
        if has_depth:
            depth_npz = _read_npz(depth_path)
            depth = depth_npz.get("depth") or depth_npz.get("depths")
            if depth is None:
                has_depth = False
        if has_depth:
            depth = _ensure_3d_numeric(np.asarray(depth, dtype=np.float32))
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

        # Find episodes by info.npz
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
            # Fallback synthetic: use first info.npz's track (any shape) + all images under split
            info_files = list(self.split_dir.rglob("info.npz"))
            if info_files:
                try:
                    info = _read_npz(info_files[0])
                    raw_track = info.get("track")
                    if raw_track is None:
                        raise RuntimeError("Fallback failed: no 'track' in info.npz")
                    try:
                        track = _ensure_3d_numeric(np.asarray(raw_track, dtype=np.uint8))
                    except (TypeError, ValueError):
                        track = _extract_track_from_object_array(raw_track, info_files[0].parent, self.split_dir, self.root_dir)
                    imgs = _glob_images(self.split_dir)
                    if not imgs:
                        raise RuntimeError("No images found for fallback synthetic episode.")
                    N = len(imgs)
                    if track.shape[0] == 1 and N > 1:
                        track = np.repeat(track, N, axis=0)
                    track = track[:N]
                    depth = np.zeros_like(track, dtype=np.float32)

                    # synthesize episode
                    ep = type("SyntheticEpisode", (), {})()
                    ep.image_paths = imgs[:N]
                    ep.track = track
                    ep.depth = depth
                    ep.N = N
                    ep.has_depth = False
                    self.episodes = [ep]  # type: ignore
                    print("[DriveDataset] Fallback: using synthetic episode from split images.")
                except Exception as e:
                    raise RuntimeError(f"All episodes under {self.split_dir} failed to load (even fallback): {e}") from e
            else:
                raise RuntimeError(f"All episodes under {self.split_dir} failed to load.")

        # Index
        self.index: List[Tuple[int,int]] = []
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
