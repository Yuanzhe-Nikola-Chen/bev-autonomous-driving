"""
Neural feature encoders for image-view and BEV-view representations.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18


class ConvBNReLU(nn.Module):
    """
    A compact convolutional block used throughout the encoders.
    """

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3) -> None:
        super().__init__()
        padding = kernel_size // 2
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class CameraFeatureEncoder(nn.Module):
    """
    Encode image features and predict a depth distribution for BEV lifting.

    The module outputs:
        - a per-pixel categorical depth distribution
        - depth-aware image features lifted into a frustum representation
    """

    def __init__(self, in_channels: int = 3, feat_channels: int = 64, depth_bins: int = 41) -> None:
        super().__init__()
        self.depth_bins = depth_bins
        self.feat_channels = feat_channels

        self.backbone = nn.Sequential(
            ConvBNReLU(in_channels, 32),
            ConvBNReLU(32, 64),
            nn.MaxPool2d(kernel_size=2, stride=2),
            ConvBNReLU(64, 128),
            nn.MaxPool2d(kernel_size=2, stride=2),
            ConvBNReLU(128, 128),
        )

        self.prediction_head = nn.Conv2d(
            in_channels=128,
            out_channels=depth_bins + feat_channels,
            kernel_size=1,
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: [B*N, 3, H, W]

        Returns:
            depth_prob:  [B*N, D, Hf, Wf]
            lifted_feat: [B*N, C, D, Hf, Wf]
        """
        feat = self.backbone(x)
        pred = self.prediction_head(feat)

        depth_logits = pred[:, : self.depth_bins]
        image_features = pred[:, self.depth_bins :]

        depth_prob = F.softmax(depth_logits, dim=1)
        lifted_feat = depth_prob.unsqueeze(1) * image_features.unsqueeze(2)

        return depth_prob, lifted_feat


class BEVFeatureEncoder(nn.Module):
    """
    Refine pooled BEV features for downstream prediction tasks.
    """

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        backbone = resnet18(weights=None)

        self.input_proj = nn.Conv2d(
            in_channels=in_channels,
            out_channels=64,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False,
        )
        self.bn1 = backbone.bn1
        self.relu = backbone.relu
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3

        self.decoder = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            ConvBNReLU(256, 128),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            ConvBNReLU(128, 128),
            nn.Conv2d(128, out_channels, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_proj(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        return self.decoder(x)
