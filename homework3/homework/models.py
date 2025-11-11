from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

HOMEWORK_DIR = Path(__file__).resolve().parent
_MEAN = (0.485, 0.456, 0.406)
_STD  = (0.229, 0.224, 0.225)


class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, k=3, s=1, p=1, use_bn=True):
        super().__init__()
        layers = [nn.Conv2d(in_ch, out_ch, k, s, p, bias=not use_bn)]
        if use_bn:
            layers.append(nn.BatchNorm2d(out_ch))
        layers.append(nn.ReLU(inplace=True))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class UpBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2)
        self.conv = ConvBlock(out_ch * 2, out_ch)

    def forward(self, x, skip):
        x = self.up(x)
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


class Classifier(nn.Module):
    def __init__(self, num_classes=6):
        super().__init__()
        def block(in_ch, out_ch):
            return nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 3, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_ch, out_ch, 3, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
            )
        self.stem = block(3, 32)          # 64x64 -> 64x64
        self.pool1 = nn.MaxPool2d(2)      # -> 32x32
        self.enc2 = block(32, 64)
        self.pool2 = nn.MaxPool2d(2)      # -> 16x16
        self.enc3 = block(64, 128)
        self.pool3 = nn.AdaptiveAvgPool2d(1)  # -> 1x1
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):                 # (B,3,64,64)
        x = self.stem(x)
        x = self.pool1(x)
        x = self.enc2(x)
        x = self.pool2(x)
        x = self.enc3(x)
        x = self.pool3(x)
        return self.head(x)

    def _to_batch(self, x):
        """
        Convert PIL/np/tensor image -> (B,3,H,W) float32 in [0,1].
        """
        if isinstance(x, np.ndarray):
            arr = x
            if arr.ndim == 3 and arr.shape[0] == 3:           # CHW
                t = torch.from_numpy(arr).float()
            elif arr.ndim == 3 and arr.shape[-1] == 3:        # HWC -> CHW
                t = torch.from_numpy(arr.transpose(2, 0, 1)).float()
            else:
                raise ValueError(f"Unexpected numpy image shape: {arr.shape}")
            if t.max() > 1.0:  # likely uint8 range
                t = t / 255.0
            t = t.unsqueeze(0)
            return t
        try:
            # PIL Image?
            from PIL.Image import Image as PILImage
            if isinstance(x, PILImage):
                t = torch.from_numpy(np.array(x)).permute(2,0,1).float() / 255.0
                return t.unsqueeze(0)
        except Exception:
            pass
        if torch.is_tensor(x):
            t = x
            if t.ndim == 3:
                t = t.unsqueeze(0)
            elif t.ndim != 4:
                raise ValueError(f"Unexpected tensor shape: {tuple(t.shape)}")
            t = t.float()
            if t.max() > 1.0:
                t = t / 255.0
            return t
        raise TypeError(f"Unsupported input type for predict: {type(x)}")

    @torch.inference_mode()
    def predict(self, x):
        """
        If x is a single image -> return int class id.
        If x is a batch      -> return 1D tensor of class ids (CPU).
        """
        self.eval()
        batch = self._to_batch(x)

        # normalize exactly like training
        device = next(self.parameters()).device
        mean = torch.tensor(_MEAN, device=device).view(3, 1, 1)
        std = torch.tensor(_STD, device=device).view(3, 1, 1)
        batch = (batch - mean) / std

        logits = self(batch)                 # (B, num_classes)
        preds = logits.argmax(dim=1)
        if preds.numel() == 1:
            return int(preds.item())
        return preds.cpu()

class Detector(nn.Module):
    def __init__(self, num_classes: int = 3, in_channels: int = 3, **kwargs):
        super().__init__()

        self.down1 = nn.Sequential(ConvBlock(in_channels, 32), ConvBlock(32, 32))
        self.down2 = nn.Sequential(ConvBlock(32, 64), ConvBlock(64, 64))
        self.down3 = nn.Sequential(ConvBlock(64, 128), ConvBlock(128, 128))
        self.pool = nn.MaxPool2d(2)

        self.bottleneck = nn.Sequential(ConvBlock(128, 256), ConvBlock(256, 256))

        self.up2 = UpBlock(256, 128)
        self.up1 = UpBlock(128, 64)
        self.up0 = UpBlock(64, 32)

        self.seg_head = nn.Conv2d(32, num_classes, kernel_size=1)
        self.depth_head = nn.Conv2d(32, 1, kernel_size=1)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        # Encoder
        x1 = self.down1(x)                  # (B, 32, 96, 128)
        x2 = self.down2(self.pool(x1))      # (B, 64, 48, 64)
        x3 = self.down3(self.pool(x2))      # (B,128,24,32)

        # Bottleneck
        xb = self.bottleneck(self.pool(x3)) # (B,256,12,16)

        # Decoder with skip connections
        xd2 = self.up2(xb, x3)              # (B,128,24,32)
        xd1 = self.up1(xd2, x2)             # (B,64,48,64)
        xd0 = self.up0(xd1, x1)             # (B,32,96,128)

        seg_logits = self.seg_head(xd0)                 # (B,3,96,128)
        depth = torch.sigmoid(self.depth_head(xd0))     # (B,1,96,128) in [0,1]
        return seg_logits, depth

    def predict(self, x: torch.Tensor):
        self.eval()
        with torch.no_grad():
            seg_logits, depth = self.forward(x)         # (B,3,H,W), (B,1,H,W)
            seg_pred = seg_logits.argmax(dim=1)         # (B,H,W), long
            depth = depth.squeeze(1)                    # (B,H,W), float in [0,1]
            return seg_pred.long(), depth.float()


MODEL_FACTORY = {
    "classifier": Classifier,
    "detector": Detector,
}


def load_model(
    model_name: str,
    with_weights: bool = False,
    **model_kwargs,
) -> torch.nn.Module:
    """
    Called by the grader to load a pre-trained model by name
    """
    m = MODEL_FACTORY[model_name](**model_kwargs)


    if with_weights:
        model_path = HOMEWORK_DIR / f"{model_name}.th"
        assert model_path.exists(), f"{model_path.name} not found"
        try:
            ckpt = torch.load(model_path, map_location="cpu")
            if isinstance(ckpt, dict) and "model_state" in ckpt:
                ckpt = ckpt["model_state"]
            m.load_state_dict(ckpt)
        except RuntimeError as e:
            raise AssertionError(
                f"Failed to load {model_path.name}, make sure the default model arguments are set correctly"
            ) from e

    # limit model sizes since they will be zipped and submitted
    model_size_mb = calculate_model_size_mb(m)
    if model_size_mb > 20:
        raise AssertionError(f"{model_name} is too large: {model_size_mb:.2f} MB")

    return m


def save_model(model: torch.nn.Module) -> str:
    """
    Use this function to save your model in train.py
    """
    model_name = None
    for n, m in MODEL_FACTORY.items():
        if type(model) is m:
            model_name = n

    if model_name is None:
        raise ValueError(f"Model type '{str(type(model))}' not supported")

    output_path = HOMEWORK_DIR / f"{model_name}.th"
    torch.save(model.state_dict(), output_path)
    return output_path


def calculate_model_size_mb(model: torch.nn.Module) -> float:
    """
    Args:
        model: torch.nn.Module
    Returns:
        float, size in megabytes
    """
    return sum(p.numel() for p in model.parameters()) * 4 / 1024 / 1024


def debug_model(batch_size: int = 1):
    """
    Test your model implementation

    Feel free to add additional checks to this function -
    this function is NOT used for grading
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sample_batch = torch.rand(batch_size, 3, 64, 64).to(device)

    print(f"Input shape: {sample_batch.shape}")

    model = load_model("classifier", num_classes=6).to(device)
    output = model(sample_batch)

    # should output logits (b, num_classes)
    print(f"Output shape: {output.shape}")


if __name__ == "__main__":
    debug_model()
