from __future__ import annotations

import torch
from monai.networks.nets import UNet


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_model(in_channels: int = 2, out_channels: int = 3) -> torch.nn.Module:
    return UNet(
        spatial_dims=2,
        in_channels=in_channels,
        out_channels=out_channels,
        channels=(16, 32, 64),
        strides=(2, 2),
        num_res_units=1,
    )