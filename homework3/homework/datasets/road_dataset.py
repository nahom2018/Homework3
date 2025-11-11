from pathlib import Path
from pathlib import Path
from typing import Dict, Union
from torch.utils.data import DataLoader, ConcatDataset, Dataset
import numpy as np
from torch.utils.data import ConcatDataset, DataLoader, Dataset

from . import road_transforms
from .road_utils import Track


class RoadDataset(Dataset):
    """
    SuperTux dataset for road detection
    """

    def __init__(
        self,
        episode_path: str,
        transform_pipeline: str = "default",
    ):
        super().__init__()

        self.episode_path = Path(episode_path)

        info = np.load(self.episode_path / "info.npz", allow_pickle=True)

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


def load_data(
    dataset_path: str,
    transform_pipeline: str = "default",
    return_dataloader: bool = True,
    num_workers: int = 2,
    batch_size: int = 32,
    shuffle: bool = False,
) -> Union[DataLoader, Dataset, Dict[str, DataLoader]]:
    """
    Build datasets/dataloaders for the SuperTuxKart drive_data.

    Accepts:
      - Root path with splits:        drive_data/
        (expects drive_data/train/* and drive_data/val/* episodes)
      - Single split path:            drive_data/train
      - Single episode path:          drive_data/train/cornfield_crossing_00

    If both 'train' and 'val' exist under dataset_path, returns {"train": DL, "val": DL}.
    Otherwise returns a single DataLoader (or Dataset if return_dataloader=False).
    """

    root = Path(dataset_path)

    def _episodes_in_dir(split_dir: Path):
        # Return sorted episode directories inside a split dir
        if not split_dir.exists():
            return []
        return sorted([p for p in split_dir.iterdir() if p.is_dir()])

    # Case A: dataset_path is the root containing train/ and/or val/
    train_dir = root / "train"
    val_dir   = root / "val"
    has_train = train_dir.exists() and train_dir.is_dir()
    has_val   = val_dir.exists() and val_dir.is_dir()

    if has_train or has_val:
        splits = {}
        if has_train:
            train_eps = _episodes_in_dir(train_dir)
            if not train_eps:
                raise FileNotFoundError(f"No episode directories found in {train_dir}")
            train_list = [RoadDataset(ep, transform_pipeline=transform_pipeline) for ep in train_eps]
            train_ds = ConcatDataset(train_list) if len(train_list) > 1 else train_list[0]
            splits["train"] = train_ds

        if has_val:
            val_eps = _episodes_in_dir(val_dir)
            if not val_eps:
                raise FileNotFoundError(f"No episode directories found in {val_dir}")
            val_list = [RoadDataset(ep, transform_pipeline=transform_pipeline) for ep in val_eps]
            val_ds = ConcatDataset(val_list) if len(val_list) > 1 else val_list[0]
            splits["val"] = val_ds

        if not return_dataloader:
            return splits  # dict of datasets

        # Build loaders (shuffle train only)
        loaders: Dict[str, DataLoader] = {}
        for sp_name, ds in splits.items():
            loaders[sp_name] = DataLoader(
                ds,
                batch_size=batch_size,
                shuffle=(sp_name == "train"),
                num_workers=num_workers,
                pin_memory=True,
                persistent_workers=(num_workers > 0),
            )
        print(f"Loaded train={has_train} val={has_val} | "
              f"train_ep={len(_episodes_in_dir(train_dir)) if has_train else 0} "
              f"val_ep={len(_episodes_in_dir(val_dir)) if has_val else 0}")
        return loaders

    # Case B: dataset_path is a split directory (e.g., drive_data/train)
    #         or a single episode directory (e.g., drive_data/train/cornfield_crossing_00)
    if root.is_dir():
        # If it contains episode subfolders, use them; otherwise treat it as a single episode
        episode_dirs = _episodes_in_dir(root)
        if episode_dirs:
            ds_list = [RoadDataset(ep, transform_pipeline=transform_pipeline) for ep in episode_dirs]
            dataset: Dataset = ConcatDataset(ds_list) if len(ds_list) > 1 else ds_list[0]
        else:
            # Single episode folder expected to contain info.npz and frames/
            dataset = RoadDataset(root, transform_pipeline=transform_pipeline)
    else:
        raise FileNotFoundError(f"Path does not exist: {root}")

    if not return_dataloader:
        return dataset

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,                 # caller decides when not using root
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=(num_workers > 0),
    )
