from pathlib import Path
from typing import Optional

import numpy as np
import torch
import numpy as np
from torch.utils.data import ConcatDataset, DataLoader, Dataset

from . import road_transforms
from .road_utils import Track


class RoadDataset(Dataset):
    """
    SuperTux dataset for road detection
    """

    def __init__(self, episode_path, transform_pipeline=None, allow_missing_masks=False):
        self.episode_path = Path(episode_path)  # ensure it's a Path
        self.transform_pipeline = transform_pipeline
        self.allow_missing_masks = allow_missing_masks

        info = np.load(str(self.episode_path / "info.npz"), allow_pickle=True)

        self.track = Track(**info["track"].item())
        self.frames: dict[str, np.ndarray] = {k: np.stack(v) for k, v in info["frames"].item().items()}
        self.transform = self.get_transform("eval" if transform_pipeline is None else transform_pipeline)

    def get_transform(self, transform_pipeline: Optional[str]):
        """
        Build a transform pipeline.
        - None, "default", "eval", "val", "test", "none"  -> default (no augmentation)
        - "aug" (or "train")                              -> augmentation pipeline (currently same as default; customize if desired)
        """

        alias = (transform_pipeline or "default").lower()

        def _default_pipeline():
            return road_transforms.Compose(
                [
                    road_transforms.ImageLoader(self.episode_path),
                    road_transforms.DepthLoader(self.episode_path),
                    road_transforms.TrackProcessor(self.track),
                ]
            )

        def _aug_pipeline():\
            return _default_pipeline()

        if alias in {"default", "eval", "val", "test", "none"}:
            return _default_pipeline()
        if alias in {"aug", "train"}:
            return _aug_pipeline()
        return _default_pipeline()

    def __len__(self):
        return len(self.frames["location"])

    def __getitem__(self, idx: int):
        """
        Returns:
            dict: sample data with keys "image", "depth", "track"
        """
        sample = {"_idx": idx, "_frames": self.frames}
        sample = self.transform(sample)

        # remove private keys
        for key in list(sample.keys()):
            if key.startswith("_"):
                sample.pop(key)

        return sample


def _collect_episode_dirs(split_dir: Path):
    if not split_dir.exists():
        return []
    return sorted(
        p for p in split_dir.iterdir()
        if p.is_dir() and (p / "info.npz").exists()
    )

def load_data(dataset_path, batch_size=32, num_workers=2, transform_pipeline=None,
              allow_missing_masks=False, seed: int = 1337):
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

    # If val is missing, carve 80/20 from train
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
                # If caller didn’t specify, use aug for train, default for others
                transform_pipeline=("aug" if (transform_pipeline is None and split_name == "train") else
                                    ("default" if transform_pipeline is None else transform_pipeline)),
                allow_missing_masks=allow_missing_masks,
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
