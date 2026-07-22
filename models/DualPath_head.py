import torch
import torch.nn as nn


class MainBranchAnomalyHead(nn.Module):
    """
    Lightweight anomaly head used at test time.

    Input: concatenated [image, reconstruction], shape (B, 2C, H, W)
    Output: anomaly probability map, shape (B, 1, H, W)
    """

    def __init__(self, in_channels=6, hidden_channels=64, out_channels=1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden_channels, out_channels, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.net(x)


def residual_anomaly_map(image, reconstruction, reduce_mode="mean"):
    """
    Build anomaly map only from backbone outputs.

    reduce_mode:
    - mean: channel-wise mean absolute residual
    - max: channel-wise max absolute residual
    """
    residual = torch.abs(image - reconstruction)
    if reduce_mode == "mean":
        return torch.mean(residual, dim=1, keepdim=True)
    if reduce_mode == "max":
        return torch.max(residual, dim=1, keepdim=True).values
    raise ValueError(f"Unknown reduce_mode: {reduce_mode}")
