# ------------------ road_dataset.py ------------------
from pathlib import Path
from typing import Optional, List
import numpy as np
import torch

# Use relative imports inside the package
from . import road_transforms  # and add similar for road_utils if you use it


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

        # Extract per-episode track masks if present
        self.track = info["track"] if ("track" in files) else None
        self.has_masks = self.track is not None

        # If masks required (val/test), enforce presence
        if not self.allow_missing_masks and not self.has_masks:
            raise FileNotFoundError(f"No 'track' masks in {info_path} but masks are required for this split.")

        # Build transform (define AFTER has_masks is known)
        self.transform = self.get_transform("default" if transform_pipeline is None else transform_pipeline)

        # ---- Frame index bookkeeping ----
        img_dir = self.episode_path / "image"
        exts = {".png", ".jpg", ".jpeg"}

        if "indices" in files:  # explicit indices in info.npz
            self.indices: List[int] = [int(x) for x in list(self.info["indices"])]
        elif img_dir.exists():
            frame_files = sorted(p for p in img_dir.iterdir() if p.is_file() and p.suffix.lower() in exts)
            if not frame_files:
                raise FileNotFoundError(f"No frames found in {img_dir}")
            try:
                self.indices = [int(p.stem) for p in frame_files]  # numeric stems like 000.png
            except ValueError:
                # fallback to range if stems are not numeric
                self.indices = list(range(len(frame_files)))
        elif "length" in files:
            self.indices = list(range(int(self.info["length"])))
        elif "n" in files:
            self.indices = list(range(int(self.info["n"])))
        else:
            raise RuntimeError(
                f"Could not infer frame indices for episode {self.episode_path}. "
                f"Expected 'image/' folder or 'indices'/'length'/'n' in info.npz."
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
        sample = {"index": frame_id}
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
