# ------------------ road_dataset.py ------------------
from pathlib import Path
from typing import Optional, List
import numpy as np
import numpy as _np
import torch
import re

# Use relative imports inside the package
from . import road_transforms  # and add similar for road_utils if you use it

def _extract_frames_meta(frames_obj):
    """
    Unpack 'frames' from info.npz into a plain dict[str, np.ndarray].
    Handles 0-d object arrays, structured arrays, or dicts.
    """


    # 0-d object array that stores a dict
    if isinstance(frames_obj, _np.ndarray) and frames_obj.dtype == object and frames_obj.shape == ():
        inner = frames_obj.item()
        return dict(inner)

    # Already a dict
    if isinstance(frames_obj, dict):
        return dict(frames_obj)

    # Structured/record array
    if isinstance(frames_obj, _np.ndarray) and frames_obj.dtype.names:
        return {name: frames_obj[name] for name in frames_obj.dtype.names}

    # Fallback: wrap as a single array (indexable at least)
    if isinstance(frames_obj, _np.ndarray):
        return {"_array": frames_obj}

    return {}

# ===================== Dataset =====================

class RoadDataset(torch.utils.data.Dataset):
    """
    A single SuperTuxKart driving episode.
    Expects:
      episode_path/
        info.npz
        image/   (frames)
        depth/   (frames)
        ... (whatever your loaders expect)
    """
    def __init__(self, episode_path, transform_pipeline: Optional[str] = None, allow_missing_masks: bool = False):
        self.episode_path = Path(episode_path)
        self.allow_missing_masks = bool(allow_missing_masks)

        info_path = self.episode_path / "info.npz"
        if not info_path.exists():
            raise FileNotFoundError(f"Missing info.npz at {info_path}")

        info = np.load(str(info_path), allow_pickle=True)
        self.info = info
        files = getattr(info, "files", [])

        # --- masks ---
        self.track = info["track"] if ("track" in files) else None
        self.has_masks = self.track is not None

        # --- unpack frames -> dict of arrays (expects keys like 'distance_down_track') ---
        if "frames" in files:
            self.frames_meta = _extract_frames_meta(info["frames"])
        else:
            self.frames_meta = {}

        # ---- choose indices using frames_meta first; fallback to files on disk ----
        def _len_of_any_array(d):
            for v in d.values():
                try:
                    return len(v)
                except Exception:
                    pass
            return None

        N = _len_of_any_array(self.frames_meta)
        if N is not None and N > 0:
            # frames arrays define the timeline; use simple 0..N-1
            self.indices = list(range(N))
        else:
            # Fallback: discover frames by filenames
            import re
            exts = {".png", ".jpg", ".jpeg"}

            # Prefer files like 00000_im.jpg in the episode root
            pattern = re.compile(r"^(\d{1,})_im\.(jpg|png|jpeg)$", re.IGNORECASE)
            root_files = sorted([p for p in self.episode_path.iterdir() if p.is_file()])
            match_files = [p for p in root_files if p.suffix.lower() in exts and pattern.match(p.name)]

            if match_files:
                self.indices = [int(pattern.match(p.name).group(1)) for p in match_files]
            else:
                # Try common subdirs
                candidate_dirs = ["image", "images", "rgb", "imgs", "color"]
                frame_dir = next((self.episode_path / d for d in candidate_dirs
                                  if (self.episode_path / d).exists() and (self.episode_path / d).is_dir()), None)
                if frame_dir is not None:
                    frame_files = sorted([p for p in frame_dir.iterdir()
                                          if p.is_file() and p.suffix.lower() in exts])
                    if not frame_files:
                        raise FileNotFoundError(f"No frame files found in {frame_dir}")
                    # Try numeric stems; otherwise enumerate
                    try:
                        self.indices = [int(p.stem) for p in frame_files]
                    except ValueError:
                        self.indices = list(range(len(frame_files)))
                else:
                    # Last resort, look for length-like keys in info.npz
                    for k in ("length", "n", "num_frames"):
                        if k in files:
                            self.indices = list(range(int(info[k])))
                            break
                    else:
                        present_dirs = [p.name for p in self.episode_path.iterdir() if p.is_dir()]
                        present_files = [p.name for p in self.episode_path.iterdir() if p.is_file()]
                        raise RuntimeError(
                            f"Could not infer frame indices for episode {self.episode_path}.\n"
                            f"Looked for 'frames' arrays, files like 00000_im.jpg, common subdirs {candidate_dirs} "
                            f"(found dirs: {present_dirs}), and 'length'/'n'/'num_frames' in info.npz.\n"
                            f"Found files: {present_files}"
                        )

        self.length = len(self.indices)

    # >>>>>>>>>>> define get_transform AS A METHOD OF THE CLASS <<<<<<<<<<
    def get_transform(self, transform_pipeline: Optional[str]):
        """
        Build a transform pipeline.
          - None, default/eval/val/test/none => default (no aug)
          - aug/train                       => augmentation pipeline (customize if desired)
        """
        alias = (transform_pipeline or "default").lower()

        def _default():
            ops = [
                road_transforms.ImageLoader(self.episode_path),
                road_transforms.DepthLoader(self.episode_path),
            ]
            if self.has_masks:
                ops.append(road_transforms.TrackProcessor(self.track))
            return road_transforms.Compose(ops)

        def _aug():
            ops = [
                road_transforms.ImageLoader(self.episode_path),
                road_transforms.DepthLoader(self.episode_path),
            ]
            if self.has_masks:
                ops.append(road_transforms.TrackProcessor(self.track))
            # Example aug (uncomment if defined in your road_transforms):
            # ops.append(road_transforms.RandomHorizontalFlip(p=0.5))
            return road_transforms.Compose(ops)

        if alias in {"default", "eval", "val", "test", "none"}:
            return _default()
        if alias in {"aug", "train"}:
            return _aug()
        return _default()

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        frame_id = int(self.indices[idx])
        sample = {
            "_idx": frame_id,  # for file-based loaders (e.g., 00000_im.jpg)
            "_frames": self.frames_meta  # dict with arrays like 'distance_down_track'
        }
        return self.transform(sample)


# ===================== Data loading =====================

def _collect_episode_dirs(split_dir: Path):
    """Return episode dirs under split_dir that contain info.npz (or [] if split_dir missing)."""
    if not split_dir.exists():
        return []
    return sorted(
        p for p in split_dir.iterdir()
        if p.is_dir() and (p / "info.npz").exists()
    )

def load_data(dataset_path,
              batch_size: int = 32,
              num_workers: int = 2,
              transform_pipeline: Optional[str] = None,
              allow_missing_masks: bool = False,
              seed: int = 1337):
    """
    Build DataLoaders for train/val/test by concatenating per-episode datasets.
    - If val/ or test/ missing, derive them (test reuses val).
    - allow_missing_masks applies ONLY to train.
    """
    dataset_path = Path(dataset_path)
    train_dir = dataset_path / "train"
    val_dir   = dataset_path / "val"
    test_dir  = dataset_path / "test"

    if not train_dir.exists():
        raise FileNotFoundError(
            f"Missing required directory: {train_dir}\n"
            f"Expected: drive_data/train/<episode>/info.npz"
        )

    train_eps = _collect_episode_dirs(train_dir)
    val_eps   = _collect_episode_dirs(val_dir)
    test_eps  = _collect_episode_dirs(test_dir)

    if not train_eps:
        raise FileNotFoundError(f"No episodes with info.npz under {train_dir}")

    # If val missing, carve 80/20 from train
    if not val_eps:
        rng = np.random.RandomState(seed)
        idx = np.arange(len(train_eps))
        rng.shuffle(idx)
        n_train = max(1, int(0.8 * len(idx)))
        val_eps = [train_eps[i] for i in idx[n_train:] or idx[-1:]]
        train_eps = [train_eps[i] for i in idx[:n_train]]

    # If test missing, reuse val
    if not test_eps:
        test_eps = val_eps

    def _make_loader(episode_dirs, split_name: str):
        per_episode = [
            RoadDataset(
                ep,
                transform_pipeline=(
                    "aug" if (transform_pipeline is None and split_name == "train")
                    else ("default" if transform_pipeline is None else transform_pipeline)
                ),
                allow_missing_masks=(split_name == "train" and allow_missing_masks),
            )
            for ep in episode_dirs
        ]
        concat = torch.utils.data.ConcatDataset(per_episode)
        return torch.utils.data.DataLoader(
            concat,
            batch_size=batch_size,
            shuffle=(split_name == "train"),
            num_workers=num_workers,
            pin_memory=True,
        )

    train_loader = _make_loader(train_eps, "train")
    val_loader   = _make_loader(val_eps,   "val")
    test_loader  = _make_loader(test_eps,  "test")
    return train_loader, val_loader, test_loader
# ------------------ end road_dataset_
