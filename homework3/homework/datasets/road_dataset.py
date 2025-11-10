from pathlib import Path

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
        self.transform = self.get_transform(transform_pipeline)

    def get_transform(self, transform_pipeline: str):
        xform = None

        if transform_pipeline == "default":
            xform = road_transforms.Compose(
                [
                    road_transforms.ImageLoader(self.episode_path),
                    road_transforms.DepthLoader(self.episode_path),
                    road_transforms.TrackProcessor(self.track),
                ]
            )
        elif transform_pipeline == "aug":
            pass

        if xform is None:
            raise ValueError(f"Invalid transform {transform_pipeline} specified!")

        return xform

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


def load_data(dataset_path, batch_size=32, num_workers=2, transform_pipeline=None,
              allow_missing_masks=False):
    """
    Build DataLoaders for train/val/test by concatenating per-episode datasets.
    Each episode directory must contain an info.npz file.
    """
    dataset_path = Path(dataset_path)
    loaders = {}

    for split in ["train", "val", "test"]:
        split_dir = dataset_path / split

        # Collect episode directories (those that contain info.npz)
        episode_dirs = sorted(
            p for p in split_dir.iterdir()
            if p.is_dir() and (p / "info.npz").exists()
        )
        if not episode_dirs:
            raise FileNotFoundError(
                f"No episodes with info.npz found under: {split_dir}\n"
                f"Expected structure like: {split_dir}/<episode>/info.npz"
            )

        # Build a RoadDataset per episode, then concat
        per_episode = [
            RoadDataset(
                p,
                transform_pipeline=transform_pipeline,
                allow_missing_masks=allow_missing_masks
            )
            for p in episode_dirs
        ]
        concat = torch.utils.data.ConcatDataset(per_episode)

        loaders[split] = torch.utils.data.DataLoader(
            concat,
            batch_size=batch_size,
            shuffle=(split == "train"),
            num_workers=num_workers,
            pin_memory=True,
        )

    return loaders["train"], loaders["val"], loaders["test"]
