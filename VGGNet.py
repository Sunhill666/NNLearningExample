from typing import Dict, List, Union, cast, Any

import torch
from torch import nn

cfgs: Dict[str, List[Union[str, int]]] = {
    "VGGNet_11": [64, "M", 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
    "VGGNet_13": [64, 64, "M", 128, 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
    "VGGNet_16": [64, 64, "M", 128, 128, "M", 256, 256, 256, "M", 512, 512, 512, "M", 512, 512, 512, "M"],
    "VGGNet_19": [64, 64, "M", 128, 128, "M", 256, 256, 256, 256, "M", 512, 512, 512, 512, "M", 512, 512, 512, 512, "M"]
}


class VGGNet(nn.Module):
    def __init__(self, feature: nn.Module) -> None:
        super().__init__()
        self.features = feature
        self.avg_pool = nn.AdaptiveAvgPool2d(7)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.avg_pool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


class OriginVGGNet(VGGNet):
    def __init__(self, feature: nn.Module, num_classes: int = 1000, dropout: float = 0.5) -> None:
        super().__init__(feature)
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(p=dropout),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(p=dropout),
            nn.Linear(4096, num_classes),
        )


class SimplifiedVGGNet(VGGNet):
    def __init__(self, feature: nn.Module, num_classes: int = 1000, dropout: float = 0.5) -> None:
        super().__init__(feature)
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(512 * 7 * 7, 2048),
            nn.ReLU(True),
            nn.Dropout(p=dropout),
            nn.Linear(2048, 2048),
            nn.ReLU(True),
            nn.Dropout(p=dropout),
            nn.Linear(2048, num_classes),
        )


def _make_layers(cfg: List[Union[str, int]], batch_norm: bool = False) -> nn.Sequential:
    layers: List[nn.Module] = []
    in_channels = 3
    for v in cfg:
        if v == "M":
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            v = cast(int, v)
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


def vgg(model_type: str, cfg: str, batch_norm: bool, **kwargs: Any) -> VGGNet:
    if model_type == "Origin":
        return OriginVGGNet(_make_layers(cfgs[cfg], batch_norm=batch_norm), **kwargs)
    else:
        return SimplifiedVGGNet(_make_layers(cfgs[cfg], batch_norm=batch_norm), **kwargs)
