# homework/datasets/drive_dataset.py
import os, glob
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


class DriveDataset(Dataset):
    """
    Reads scenes like:
      drive_data/train/cornfield_crossing_00/00000_im.jpg
      drive_data/train/cornfield_crossing_00/00000_depth.png
      and optionally *_track.png or *_seg.png if available.
    """

    def __init__(self, root, image_size=(96, 128), require_masks=True):
        self.root = root
        self.image_size = image_size
        self.require_masks = require_masks
        self.samples = []

        # --- Find all scene subdirectories ---
        scene_dirs = sorted(
            [os.path.join(root, d) for d in os.listdir(root)
             if os.path.isdir(os.path.join(root, d))]
        )
        if not scene_dirs:
            raise FileNotFoundError(f"No scene folders found under {root}")

        # --- Scan for image/depth/mask triplets ---
        for scene in scene_dirs:
            im_files = sorted(glob.glob(os.path.join(scene, "*_im.jpg")))
            for imf in im_files:
                stem = imf.replace("_im.jpg", "")
                depthf = stem + "_depth.png"

                # candidate mask names
                candidates = [
                    stem + "_track.png",
                    stem + "_seg.png",
                    stem + "_mask.png",
                    stem + "_labels.png",
                    stem + "_lane.png",
                    stem + "_lane_mask.png",
                ]
                trackf = next((p for p in candidates if os.path.isfile(p)), None)

                if not os.path.isfile(depthf):
                    continue

                if self.require_masks and not trackf:
                    raise FileNotFoundError(
                        f"Missing segmentation mask for {imf} (tried {', '.join(os.path.basename(c) for c in candidates)})"
                    )

                self.samples.append((imf, depthf, trackf))

        if not self.samples:
            raise FileNotFoundError(f"No (im, depth) pairs found under {root}")

        print(f"[DriveDataset] Loaded {len(self.samples)} samples from {root}")

        # --- Transforms ---
        self.img_tf = transforms.Compose([
            transforms.Resize(self.image_size[::-1]),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406),
                                 (0.229, 0.224, 0.225)),
        ])
        self.depth_tf = transforms.Resize(
            self.image_size[::-1],
            interpolation=transforms.InterpolationMode.NEAREST
        )
        self.seg_tf = transforms.Resize(
            self.image_size[::-1],
            interpolation=transforms.InterpolationMode.NEAREST
        )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        imf, depthf, trackf = self.samples[idx]

        # Image
        img = Image.open(imf).convert("RGB")
        img = self.img_tf(img)

        # Depth (normalize to [0,1])
        depth = np.array(Image.open(depthf).convert("L"), dtype=np.float32) / 255.0
        depth = self.depth_tf(Image.fromarray((depth * 255).astype(np.uint8)))
        depth = torch.tensor(np.array(depth), dtype=torch.float32) / 255.0

        # Segmentation
        if trackf and os.path.isfile(trackf):
            seg = Image.open(trackf).convert("L")
            seg = self.seg_tf(seg)
            seg = torch.tensor(np.array(seg), dtype=torch.long)
        else:
            seg = torch.zeros_like(depth, dtype=torch.long).squeeze(0)

        return {"image": img, "depth": depth, "track": seg}


def load_data(batch_size=16, num_workers=2, root="drive_data",
              image_size=(96, 128), require_masks=True):
    train_root = os.path.join(root, "train")
    val_root = os.path.join(root, "val")
    if not os.path.isdir(train_root) or not os.path.isdir(val_root):
        raise FileNotFoundError(f"Expected {root}/train and {root}/val")

    train_ds = DriveDataset(train_root, image_size=image_size, require_masks=require_masks)
    val_ds = DriveDataset(val_root, image_size=image_size, require_masks=require_masks)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=True)

    return {"train": train_loader, "val": val_loader}
