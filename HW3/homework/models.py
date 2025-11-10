from pathlib import Path
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

HOMEWORK_DIR = Path(__file__).resolve().parent
INPUT_MEAN = [0.2788, 0.2657, 0.2629]
INPUT_STD = [0.2064, 0.1944, 0.2252]


class ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, k: int = 3, s: int = 1, p: int = 1, use_bn: bool = True):
        super().__init__()
        layers = [nn.Conv2d(in_ch, out_ch, kernel_size=k, stride=s, padding=p, bias=not use_bn)]
        if use_bn:
            layers.append(nn.BatchNorm2d(out_ch))
        layers.append(nn.ReLU(inplace=True))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Classifier(nn.Module):
    """CNN that maps (B, 3, 64, 64) -> (B, num_classes) logits"""
    def __init__(self, in_channels: int = 3, num_classes: int = 6):
        super().__init__()
        self.features = nn.Sequential(
            ConvBlock(in_channels, 32),
            ConvBlock(32, 32),
            nn.MaxPool2d(2),
            ConvBlock(32, 64),
            ConvBlock(64, 64),
            nn.MaxPool2d(2),
            ConvBlock(64, 128),
            nn.Dropout(0.2),
            nn.MaxPool2d(2),
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 8 * 8, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.head(x)
        return x


class Detector(nn.Module):
    """U-Net-like encoder-decoder with segmentation + depth outputs."""
    def __init__(self, in_channels: int = 3, num_seg_classes: int = 3, base_ch: int = 16):
        super().__init__()
        # Encoder
        self.enc1 = nn.Sequential(ConvBlock(in_channels, base_ch), ConvBlock(base_ch, base_ch))
        self.down1 = nn.Conv2d(base_ch, base_ch * 2, 3, 2, 1)
        self.enc2 = nn.Sequential(ConvBlock(base_ch * 2, base_ch * 2), ConvBlock(base_ch * 2, base_ch * 2))
        self.down2 = nn.Conv2d(base_ch * 2, base_ch * 4, 3, 2, 1)
        self.bottleneck = nn.Sequential(ConvBlock(base_ch * 4, base_ch * 4), ConvBlock(base_ch * 4, base_ch * 4))

        # Decoder
        self.up2 = nn.ConvTranspose2d(base_ch * 4, base_ch * 2, 2, 2)
        self.dec2 = nn.Sequential(ConvBlock(base_ch * 4, base_ch * 2), ConvBlock(base_ch * 2, base_ch * 2))
        self.up1 = nn.ConvTranspose2d(base_ch * 2, base_ch, 2, 2)
        self.dec1 = nn.Sequential(ConvBlock(base_ch * 2, base_ch), ConvBlock(base_ch, base_ch))

        # Heads
        self.seg_head = nn.Conv2d(base_ch, num_seg_classes, 1)
        self.depth_head = nn.Conv2d(base_ch, 1, 1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        e1 = self.enc1(x)
        d1 = self.down1(e1)
        e2 = self.enc2(d1)
        d2 = self.down2(e2)
        b = self.bottleneck(d2)
        u2 = self.up2(b)
        if u2.shape[-2:] != e2.shape[-2:]:
            diff_y = e2.size(-2) - u2.size(-2)
            diff_x = e2.size(-1) - u2.size(-1)
            u2 = F.pad(u2, [diff_x // 2, diff_x - diff_x // 2, diff_y // 2, diff_y - diff_y // 2])
        cat2 = torch.cat([u2, e2], dim=1)
        dcd2 = self.dec2(cat2)
        u1 = self.up1(dcd2)
        if u1.shape[-2:] != e1.shape[-2:]:
            diff_y = e1.size(-2) - u1.size(-2)
            diff_x = e1.size(-1) - u1.size(-1)
            u1 = F.pad(u1, [diff_x // 2, diff_x - diff_x // 2, diff_y // 2, diff_y - diff_y // 2])
        cat1 = torch.cat([u1, e1], dim=1)
        dcd1 = self.dec1(cat1)
        seg_logits = self.seg_head(dcd1)
        depth = torch.sigmoid(self.depth_head(dcd1))
        return seg_logits, depth.squeeze(1)


def load_model(kind: str, **kwargs) -> nn.Module:
    if kind in ["classifier", "cls"]:
        return Classifier(**kwargs)
    if kind in ["detector", "segdepth"]:
        return Detector(**kwargs)
    raise ValueError(f"Unknown model kind: {kind}")


def debug_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cls = load_model("classifier").to(device)
    x = torch.randn(2, 3, 64, 64, device=device)
    print("Classifier:", cls(x).shape)
    det = load_model("detector").to(device)
    x2 = torch.randn(2, 3, 96, 128, device=device)
    seg, dep = det(x2)
    print("Detector:", seg.shape, dep.shape)


if __name__ == "__main__":
    debug_model()
