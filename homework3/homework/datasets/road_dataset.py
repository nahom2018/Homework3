# --- imports (ensure these are present at top of file) ---
from pathlib import Path
from typing import Optional
import numpy as np
import torch
import road_transforms  # keep your existing import path

# ===================== RoadDataset =====================

class RoadDataset(torch.utils.data.Dataset):
    def __init__(self, episode_path, transform_pipeline: Optional[str] = None, allow_missing_masks: bool = False):
        """
        episode_path: path to a single episode directory that contains info.npz (and images/depth)
        """
        self.episode_path = Path(episode_path)
        self.allow_missing_masks = allow_missing_masks

        # Load per-episode info
        info_path = self.episode_path / "info.npz"
        if not info_path.exists():
            raise FileNotFoundError(f"Missing info.npz at {info_path}")
        info = np.load(str(info_path), allow_pickle=True)
        self.info = info

        # Extract track masks if present
        files = getattr(info, "files", [])
        self.track = info["track"] if ("track" in files) else None
        self.has_masks = self.track is not None

        # If masks are required (val/test) but missing, fail early
        if not self.allow_missing_masks and not self.has_masks:
            raise FileNotFoundError(
                f"No 'track' masks in {info_path} but masks are required for this split."
            )

        # Build transform (defaults to "default" if None)
        self.transform = self.get_transform("default" if transform_pipeline is None else transform_pipeline)

        # ===== index/length bookkeeping =====
        img_dir = self.episode_path / "image"
        exts = {".png", ".jpg", ".jpeg"}

        # 1) prefer an explicit list if present in info.npz
        files = getattr(self.info, "files", [])
        if "indices" in files:
            self.indices = list(self.info["indices"])
        # 2) otherwise, infer from image/ filenames
        elif img_dir.exists():
            frame_files = sorted(
                [p for p in img_dir.iterdir() if p.is_file() and p.suffix.lower() in exts]
            )
            if len(frame_files) == 0:
                raise FileNotFoundError(f"No frames found in {img_dir}")
            # If filenames are numeric like 000.png, 001.png...
            try:
                self.indices = [int(p.stem) for p in frame_files]
            except ValueError:
                # Fallback: just enumerate the files 0..N-1
                self.indices = list(range(len(frame_files)))
        # 3) last resort: look for a length/n field in info.npz
        elif "length" in files:
            self.indices = list(range(int(self.info["length"])))
        elif "n" in files:
            self.indices = list(range(int(self.info["n"])))
        else:
            raise RuntimeError(
                f"Could not infer frame indices for episode {self.episode_path}. "
                f"Expected an 'image/' folder or 'indices'/'length' in info.npz."
            )

        self.length = len(self.indices)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        # frame id used by your loaders/transforms to pick the right files
        frame_id = self.indices[idx]

        # Many road_transforms pipelines expect a dict input and add keys progressively.
        # If YOUR Compose expects just an int, change the next two lines to:
        #   return self.transform(frame_id)
        sample = {"index": int(frame_id)}
        return self.transform(sample)


def get_transform(self, transform_pipeline: Optional[str]):
        """
        Build a transform pipeline.
        - None, "default", "eval", "val", "test", "none" -> default (no augmentation)
        - "aug" or "train"                               -> augmentation pipeline (customize if desired)
        """
        alias = (transform_pipeline or "default").lower()

        def _default():
            ops = [
                road_transforms.ImageLoader(self.episode_path),
                road_transforms.DepthLoader(self.episode_path),
            ]
            # Only include TrackProcessor when masks exist
            if self.has_masks:
                ops.append(road_transforms.TrackProcessor(self.track))
            return road_transforms.Compose(ops)

        def _aug():
            # Start with default ops; add your aug ops here if available in road_transforms
            ops = [
                road_transforms.ImageLoader(self.episode_path),
                road_transforms.DepthLoader(self.episode_path),
            ]
            if self.has_masks:
                ops.append(road_transforms.TrackProcessor(self.track))
            # Example (uncomment if your road_transforms defines them):
            # ops.append(road_transforms.RandomHorizontalFlip(p=0.5))
            return road_transforms.Compose(ops)

        if alias in {"default", "eval", "val", "test", "none"}:
            return _default()
        if alias in {"aug", "train"}:
            return _aug()

        # Fallback to default
        return _default()

    # keep your __len__, __getitem__, etc. unchanged


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

    # If val is missing, carve 80/20 out of train
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
                # If caller didn’t specify, use "aug" for train, "default" for val/test
                transform_pipeline=(
                    "aug" if (transform_pipeline is None and split_name == "train")
                    else ("default" if transform_pipeline is None else transform_pipeline)
                ),
                # ✅ Only allow missing masks for TRAIN
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
