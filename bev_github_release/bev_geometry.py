"""
Geometry utilities for camera-based BEV projection.

The implementation in this file is written from scratch to express the core
ideas behind frustum construction and ego-frame projection in a clean,
research-oriented form.
"""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn


class FrustumGrid(nn.Module):
    """
    Create a fixed frustum template in image coordinates.

    Each frustum element stores:
        (u, v, depth)
    where:
        u, v   are image-plane coordinates
        depth  is a sampled distance along the camera ray
    """

    def __init__(self, grid_conf: dict, downsample: int) -> None:
        super().__init__()
        self.grid_conf = grid_conf
        self.downsample = downsample

    def build(self, image_size: Tuple[int, int]) -> torch.Tensor:
        """
        Build the frustum tensor for one camera.

        Args:
            image_size: (image_height, image_width)

        Returns:
            Tensor with shape [D, Hf, Wf, 3]
        """
        image_height, image_width = image_size
        feat_height = image_height // self.downsample
        feat_width = image_width // self.downsample

        depth_values = torch.arange(
            *self.grid_conf["dbound"],
            dtype=torch.float32,
        )
        depth_bins = depth_values.numel()

        depth_grid = depth_values.view(depth_bins, 1, 1).expand(
            depth_bins, feat_height, feat_width
        )

        x_grid = torch.linspace(
            0.0, image_width - 1.0, feat_width, dtype=torch.float32
        ).view(1, 1, feat_width).expand(depth_bins, feat_height, feat_width)

        y_grid = torch.linspace(
            0.0, image_height - 1.0, feat_height, dtype=torch.float32
        ).view(1, feat_height, 1).expand(depth_bins, feat_height, feat_width)

        return torch.stack((x_grid, y_grid, depth_grid), dim=-1)


class CameraToEgoProjector(nn.Module):
    """
    Project frustum points from augmented image space into the ego frame.
    """

    def __init__(self, frustum: torch.Tensor) -> None:
        super().__init__()
        self.register_buffer("frustum", frustum, persistent=False)

    def forward(
        self,
        rotations: torch.Tensor,
        translations: torch.Tensor,
        intrinsics: torch.Tensor,
        post_rots: torch.Tensor,
        post_trans: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            rotations:   [B, N, 3, 3]
            translations:[B, N, 3]
            intrinsics:  [B, N, 3, 3]
            post_rots:   [B, N, 3, 3]
            post_trans:  [B, N, 3]

        Returns:
            Ego-frame points with shape [B, N, D, H, W, 3]
        """
        batch_size, num_cams, _ = translations.shape
        depth_bins, feat_height, feat_width, _ = self.frustum.shape

        points = self.frustum.view(1, 1, depth_bins, feat_height, feat_width, 3)
        points = points.expand(batch_size, num_cams, depth_bins, feat_height, feat_width, 3)

        # Undo image-space augmentation.
        points = points - post_trans.view(batch_size, num_cams, 1, 1, 1, 3)
        points = torch.matmul(
            torch.inverse(post_rots).view(batch_size, num_cams, 1, 1, 1, 3, 3),
            points.unsqueeze(-1),
        ).squeeze(-1)

        # Convert (u, v, d) to camera-frame scaled coordinates (u*d, v*d, d).
        depth = points[..., 2:3]
        camera_points = torch.cat((points[..., :2] * depth, depth), dim=-1)

        # camera_points_ego = R * K^{-1} * camera_points + t
        transform = torch.matmul(rotations, torch.inverse(intrinsics))
        ego_points = torch.matmul(
            transform.view(batch_size, num_cams, 1, 1, 1, 3, 3),
            camera_points.unsqueeze(-1),
        ).squeeze(-1)

        ego_points = ego_points + translations.view(batch_size, num_cams, 1, 1, 1, 3)
        return ego_points
