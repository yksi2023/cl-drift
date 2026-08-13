"""Paper CNN model: ResNet-18 with GroupNorm + Weight Standardization."""
import torch
from torch import nn
from torch.nn import functional as F


class StdConv2d(nn.Conv2d):
    """Conv2d with Weight Standardization (Qiao et al. 2019)."""

    def forward(self, x):
        w = self.weight
        w_mean = w.mean(dim=(1, 2, 3), keepdim=True)
        # Use reshape (not view): channels_last weights may be non-contiguous.
        w_std = w.reshape(w.size(0), -1).std(dim=1, unbiased=False).reshape(-1, 1, 1, 1)
        w = (w - w_mean) / (w_std + 1e-5)
        return F.conv2d(
            x, w, self.bias, self.stride, self.padding, self.dilation, self.groups
        )


def _gn(num_channels: int, max_groups: int = 32) -> nn.GroupNorm:
    """GroupNorm with group count auto-adjusted so it divides channels."""
    g = max_groups
    while num_channels % g != 0 and g > 1:
        g //= 2
    return nn.GroupNorm(g, num_channels)


class BasicBlockGN(nn.Module):
    """ResNet basic block using StdConv2d + GroupNorm (zero-gamma on gn2)."""

    def __init__(self, in_c, out_c, stride=1):
        super().__init__()
        self.conv1 = StdConv2d(in_c, out_c, kernel_size=3, stride=stride, padding=1, bias=False)
        self.gn1 = _gn(out_c)
        self.conv2 = StdConv2d(out_c, out_c, kernel_size=3, stride=1, padding=1, bias=False)
        self.gn2 = _gn(out_c)
        if stride != 1 or in_c != out_c:
            self.shortcut = nn.Sequential(
                StdConv2d(in_c, out_c, kernel_size=1, stride=stride, bias=False),
                _gn(out_c),
            )
        else:
            self.shortcut = nn.Identity()
        self.relu1 = nn.ReLU(inplace=True)
        self.relu2 = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.relu1(self.gn1(self.conv1(x)))
        out = self.gn2(self.conv2(out))
        out = out + self.shortcut(x)
        return self.relu2(out)


class ResNet18GN(nn.Module):
    """ResNet-18 with ImageNet stem + GroupNorm + Weight Standardization.

    Used for the paper ImageNet-1k experiments (224×224 inputs).
    """

    def __init__(self, num_classes: int = 100):
        super().__init__()
        self.stem = nn.Sequential(
            StdConv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
            _gn(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )
        self.layer1 = nn.Sequential(BasicBlockGN(64, 64), BasicBlockGN(64, 64))
        self.layer2 = nn.Sequential(BasicBlockGN(64, 128, 2), BasicBlockGN(128, 128))
        self.layer3 = nn.Sequential(BasicBlockGN(128, 256, 2), BasicBlockGN(256, 256))
        self.layer4 = nn.Sequential(BasicBlockGN(256, 512, 2), BasicBlockGN(512, 512))
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)
        self._init_weights()
        self._zero_init_last_gn()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.GroupNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def _zero_init_last_gn(self):
        for m in self.modules():
            if isinstance(m, BasicBlockGN):
                nn.init.zeros_(m.gn2.weight)

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return self.fc(x)


MODEL_DEFAULTS = {
    "imagenet1k": {"model": "resnet18_gn", "num_classes": 100, "img_size": 224},
}

MODEL_CHOICES = ("resnet18_gn",)

# Older configs may still say resnet18_tiny_gn.
_MODEL_ALIASES = {
    "resnet18_tiny_gn": "resnet18_gn",
}


def build_model(name: str, num_classes: int, **kwargs) -> nn.Module:
    """Instantiate the paper ResNet-18 GN/WS model."""
    del kwargs
    name = _MODEL_ALIASES.get(name.lower(), name.lower())
    if name == "resnet18_gn":
        return ResNet18GN(num_classes=num_classes)
    raise ValueError(f"Unknown model '{name}'. Valid: {MODEL_CHOICES}")
