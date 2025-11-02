# homework/datasets/drive_dataset.py
import os, glob
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


# homework/datasets/drive_dataset.py
# CHANGES: add require_masks flag in ctor and load_data; only raise if require_masks=True

class DriveDataset(Dataset):
    def __init__(self, root, image_size=(96, 128), require_masks: bool = False):
        self.root = root
        self.image_size = image_size
        self.require_masks = require_masks
        self.samples = []
        ...
        for scene in scene_dirs:
            im_files = sorted(glob.glob(os.path.join(scene, "*_im.jpg")))
            for imf in im_files:
                stem = imf.replace("_im.jpg", "")
                depthf = stem + "_depth.png"
                trackf = stem + "_track.png"
                if not os.path.isfile(depthf):
                    continue
                if not os.path.isfile(trackf):
                    # try common alternatives
                    for cand in (stem + "_seg.png", stem + "_mask.png", stem + "_labels.png",
                                 stem + "_lane.png", stem + "_lane_mask.png"):
                        if os.path.isfile(cand):
                            trackf = cand
                            break
                # if still missing
                if not (trackf and os.path.isfile(trackf)):
                    if self.require_masks:
                        # strict mode -> drop this sample
                        continue
                    else:
                        trackf = None
                self.samples.append((imf, depthf, trackf))

        if not self.samples:
            raise FileNotFoundError(f"No (im, depth){' & mask' if self.require_masks else ''} samples found under {root}")
        ...

    def __getitem__(self, idx):
        imf, depthf, trackf = self.samples[idx]
        ...
        # depth (unchanged)
        ...

        # segmentation
        if trackf and os.path.isfile(trackf):
            seg = Image.open(trackf).convert("L")
            seg = self.seg_tf(seg)
            seg = torch.tensor(np.array(seg), dtype=torch.long)
        else:
            if self.require_masks:
                # strict mode: never reached because we filtered above, but keep for safety
                raise FileNotFoundError(f"Missing segmentation mask for {imf}")
            # dev mode: provide background-only mask (keeps shapes OK)
            seg = torch.zeros_like(depth, dtype=torch.long).squeeze(0)

        return {"image": img, "depth": depth, "track": seg}


def load_data(batch_size=16, num_workers=2, root="drive_data", image_size=(96, 128), require_masks: bool = False):
    train_root = os.path.join(root, "train")
    val_root = os.path.join(root, "val")
    if not os.path.isdir(train_root) or not os.path.isdir(val_root):
        raise FileNotFoundError(f"Expected {root}/train and {root}/val")

    train_ds = DriveDataset(train_root, image_size=image_size, require_masks=require_masks)
    val_ds   = DriveDataset(val_root,   image_size=image_size, require_masks=require_masks)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=True)
    return {"train": train_loader, "val": val_loader}
