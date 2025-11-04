from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

HOMEWORK_DIR = Path(__file__).resolve().parent
INPUT_MEAN = [0.2788, 0.2657, 0.2629]
INPUT_STD = [0.2064, 0.1944, 0.2252]
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

    def __init__(self, num_classes: int = 6, in_channels: int = 3, **kwargs):  # <-- add in_channels, **kwargs
        super().__init__()
        self.stem = nn.Sequential(
            ConvBlock(in_channels, 32),   # <-- use in_channels here (was 3)
            ConvBlock(32, 32),
            nn.MaxPool2d(2),
            ConvBlock(32, 64),
            ConvBlock(64, 64),
            nn.MaxPool2d(2),
            ConvBlock(64, 128),
            ConvBlock(128, 128),
            nn.MaxPool2d(2),
        )
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(p=0.3)
        self.fc = nn.Linear(128, num_classes)
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
        x_min, x_max = x.amin(dim=(1, 2, 3), keepdim=True), x.amax(dim=(1, 2, 3), keepdim=True)
        x_mean = x.mean()
        if (x_min >= 0).all() and (x_max <= 1).all() and 0.2 < float(x_mean) < 0.8:
            x = (x - self.input_mean) / self.input_std

        x = self.stem(x)
        x = self.gap(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        logits = self.fc(x)
        return logits

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Return class indices (B,), as expected by the grader."""
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)  # (B, 6)
            preds = logits.argmax(dim=1)  # (B,)
            return preds

class Detector(nn.Module):
    def __init__(self, num_classes: int = 6, in_channels: int = 3, **kwargs):  # <-- add in_channels, **kwargs
        super().__init__()
        self.stem = nn.Sequential(
            ConvBlock(in_channels, 32),   # <-- use in_channels here (was 3)
            ConvBlock(32, 32),
            nn.MaxPool2d(2),
            ConvBlock(32, 64),
            ConvBlock(64, 64),
            nn.MaxPool2d(2),
            ConvBlock(64, 128),
            ConvBlock(128, 128),
            nn.MaxPool2d(2),
        )
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(p=0.3)
        self.fc = nn.Linear(128, num_classes)
        self._init_weights()

    def forward(self, x):
        # Encoder
        x1 = self.down1(x)            # (B, 32, 96, 128)
        x2 = self.down2(self.pool(x1))# (B, 64, 48, 64)
        x3 = self.down3(self.pool(x2))# (B,128,24,32)

        # Bottleneck
        xb = self.bottleneck(self.pool(x3))  # (B,256,12,16)

        # Decoder with skip connections
        xd2 = self.up2(xb, x3)        # (B,128,24,32)
        xd1 = self.up1(xd2, x2)       # (B,64,48,64)
        xd0 = self.up0(xd1, x1)       # (B,32,96,128)

        seg_logits = self.seg_head(xd0)
        depth = torch.sigmoid(self.depth_head(xd0))  # depth normalized [0,1]
        return seg_logits, depth

    def predict(self, x: torch.Tensor):
        self.eval()
        with torch.no_grad():
            seg_logits, depth = self.forward(x)  # (B,3,H,W), (B,1,H,W)
            seg_pred = seg_logits.argmax(dim=1)  # (B,H,W)
            return seg_pred, depth



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
        raise AssertionError(
            f"{model_name} is too large: {model_size_mb:.2f} MB")

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
