
import os, glob
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

class DriveDataset(Dataset):
    def __init__(self, root, image_size=(96, 128)):
        self.root = root
        self.image_size = image_size

        # find files
        img_dir   = os.path.join(root, "images")
        depth_dir = os.path.join(root, "depth")
        seg_dir   = os.path.join(root, "track")

        self.imgs   = sorted(glob.glob(os.path.join(img_dir, "*.png")))
        self.depths_npy = sorted(glob.glob(os.path.join(depth_dir, "*.npy")))
        self.depths_png = sorted(glob.glob(os.path.join(depth_dir, "*.png")))
        self.segs   = sorted(glob.glob(os.path.join(seg_dir, "*.png")))

        if not self.imgs or not self.segs:
            raise FileNotFoundError(f"Could not find images or track masks under {root} "
                                    f"(expected images/*.png and track/*.png)")

        # match by filename stem if possible
        def stem(p): return os.path.splitext(os.path.basename(p))[0]
        seg_map = {stem(p): p for p in self.segs}

        # depth can be npy or png (normalized to [0,1])
        depth_map = {}
        if self.depths_npy:
            depth_map.update({stem(p): p for p in self.depths_npy})
        if self.depths_png:
            # png will be loaded and divided by 255
            for p in self.depths_png:
                depth_map[stem(p)] = p

        # build sample tuples
        samples = []
        for ip in self.imgs:
            s = stem(ip)
            if s not in seg_map or s not in depth_map:
                # skip if any modality missing
                continue
            samples.append((ip, depth_map[s], seg_map[s]))

        if not samples:
            raise FileNotFoundError(f"No aligned (image, depth, track) triplets found under {root}")

        self.samples = samples

        self.img_tf = transforms.Compose([
            transforms.Resize(self.image_size[::-1]),
            transforms.ToTensor(),
            transforms.Normalize((0.485,0.456,0.406), (0.229,0.224,0.225)),
        ])
        # depth & seg are resized with nearest to keep labels intact
        self.seg_tf = transforms.Resize(self.image_size[::-1], interpolation=transforms.InterpolationMode.NEAREST)

    def __len__(self):
        return len(self.samples)

    def _load_depth(self, path):
        if path.endswith(".npy"):
            d = np.load(path).astype("float32")
        else:
            # png -> [0,1]
            d = np.array(Image.open(path).convert("L"), dtype="float32") / 255.0
        return d

    def __getitem__(self, idx):
        ip, dp, sp = self.samples[idx]

        img = Image.open(ip).convert("RGB")
        img = self.img_tf(img)  # (3,H,W)

        depth = self._load_depth(dp)  # (H,W) float32
        depth = Image.fromarray((depth * 255.0).astype("uint8")) if depth.max() <= 1.0 else Image.fromarray(depth.astype("float32"))
        depth = self.seg_tf(depth)  # keep nearest interpolation
        depth = torch.tensor(np.array(depth), dtype=torch.float32) / 255.0
        depth = depth.clamp(0.0, 1.0)  # ensure [0,1]
        # (H,W) -> (H,W) tensor; training script will unsqueeze(1)

        seg = Image.open(sp).convert("L")  # labels {0,1,2}
        seg = self.seg_tf(seg)
        seg = torch.tensor(np.array(seg), dtype=torch.long)

        return {"image": img, "depth": depth, "track": seg}


def load_data(batch_size=16, num_workers=2, root="classification_data", image_size=(96,128)):
    train_root = os.path.join(root, "train")
    val_root   = os.path.join(root, "val")
    if not (os.path.isdir(train_root) and os.path.isdir(val_root)):
        # try alternative split names
        for alt in ("valid", "validation", "test"):
            if os.path.isdir(os.path.join(root, alt)):
                val_root = os.path.join(root, alt)
                break
    if not (os.path.isdir(train_root) and os.path.isdir(val_root)):
        raise FileNotFoundError(f"Expected {root}/train and {root}/val (or valid/validation/test)")

    tr_ds = DriveDataset(train_root, image_size=image_size)
    va_ds = DriveDataset(val_root, image_size=image_size)

    tr = DataLoader(tr_ds, batch_size=batch_size, shuffle=True,  num_workers=num_workers, pin_memory=True)
    va = DataLoader(va_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    return {"train": tr, "val": va}
