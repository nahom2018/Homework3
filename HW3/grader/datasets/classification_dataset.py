import csv
from pathlib import Path
from typing import Optional, Tuple
from PIL import Image
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

LABEL_NAMES = ["background", "kart", "pickup", "nitro", "bomb", "projectile"]

def get_transform(split: str, pipeline: Optional[str] = None):
    split = split.lower()
    aug = []
    if split == "train":
        if pipeline == "strong":
            aug = [
                transforms.RandomHorizontalFlip(0.5),
                transforms.RandomResizedCrop((64, 64), scale=(0.8, 1.0)),
                transforms.ColorJitter(0.2, 0.2, 0.2),
            ]
        else:
            aug = [transforms.RandomHorizontalFlip(0.5)]
    aug += [
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.2788, 0.2657, 0.2629], std=[0.2064, 0.1944, 0.2252]),
    ]
    return transforms.Compose(aug)

class SuperTuxDataset(Dataset):
    def __init__(self, dataset_root: str | Path, split="train", transform_pipeline=None):
        self.root = Path(dataset_root)
        self.split = split
        self.transform = get_transform(split, transform_pipeline)
        self.samples = []
        for cls_id, cls_name in enumerate(LABEL_NAMES):
            class_dir = self.root / split / cls_name
            if not class_dir.exists():
                continue
            for p in class_dir.rglob("*.png"):
                self.samples.append((p, cls_id))
        if not self.samples:
            raise RuntimeError(f"No samples found in {self.root}/{split}")

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        path, label = self.samples[idx]
        with Image.open(path) as img:
            img = img.convert("RGB")
            return self.transform(img), label

def load_data(dataset_path, split="train", transform_pipeline=None, batch_size=64, num_workers=2, shuffle=True, return_dataloader=True):
    ds = SuperTuxDataset(dataset_path, split, transform_pipeline)
    if not return_dataloader:
        return ds
    return DataLoader(ds, batch_size=batch_size, shuffle=(split=="train" and shuffle), num_workers=num_workers, drop_last=(split=="train"))
