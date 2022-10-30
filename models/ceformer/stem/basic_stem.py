import torch
from torch import nn


class BasicStem(nn.Module):
    conv_params = [
        {"kernel_size": (3, 3), "stride": 2, "padding": 1, "in_channels": 3, "out_channels": 24},
        {"kernel_size": (3, 3), "stride": 2, "padding": 1, "in_channels": 24, "out_channels": 48},
        {"kernel_size": (3, 3), "stride": 2, "padding": 1, "in_channels": 48, "out_channels": 96},
        {"kernel_size": (3, 3), "stride": 2, "padding": 1, "in_channels": 96, "out_channels": 192},
        {"kernel_size": (1, 1), "stride": 1, "padding": 0, "in_channels": 192, "out_channels": 128},
    ]

    def __init__(self, in_channels=3, out_channels=128):
        super(BasicStem, self).__init__()
        self.stem = []
        self.conv_params[0]["in_channels"] = in_channels
        self.conv_params[-1]["out_channels"] = out_channels

        for param in self.conv_params:
            self.stem.append(nn.Conv2d(**param))
            self.stem.append(nn.BatchNorm2d(param["out_channels"]))
            self.stem.append(nn.ReLU())
        self.stem = nn.ModuleList(self.stem)

    def forward(self, img: torch.Tensor):
        for model in self.stem:
            img = model.forward(img)
        seq = img.permute(0, 2, 3, 1)
        return seq
