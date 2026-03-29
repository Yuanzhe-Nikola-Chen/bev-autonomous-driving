"""
Voxel-based feature aggregation for BEV construction.
"""

from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn


def metric_to_voxel_index(
    points: torch.Tensor,
    voxel_size: torch.Tensor,
    voxel_origin: torch.Tensor,
) -> torch.Tensor:
    """
    Convert continuous coordinates into discrete voxel indices.
    """
    return torch.floor((points - voxel_origin) / voxel_size).long()


class VoxelPooler(nn.Module):
    """
    Aggregate lifted image features into a BEV voxel grid.

    This implementation uses a straightforward accumulation scheme to keep the
    code readable for research and educational use.
    """

    def __init__(
        self,
        voxel_size: Sequence[float],
        voxel_origin: Sequence[float],
        grid_size: Sequence[int],
    ) -> None:
        super().__init__()
        self.register_buffer("voxel_size", torch.tensor(voxel_size, dtype=torch.float32))
        self.register_buffer("voxel_origin", torch.tensor(voxel_origin, dtype=torch.float32))
        self.register_buffer("grid_size", torch.tensor(grid_size, dtype=torch.long))

    def forward(self, geometry: torch.Tensor, features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            geometry: [B, N, D, H, W, 3] ego-frame point coordinates
            features: [B, N, D, H, W, C] lifted image features

        Returns:
            BEV tensor with shape [B, C * Z, X, Y]
        """
        batch_size, num_cams, depth_bins, feat_height, feat_width, channels = features.shape

        flat_geometry = geometry.reshape(batch_size, -1, 3)
        flat_features = features.reshape(batch_size, -1, channels)

        grid_x, grid_y, grid_z = map(int, self.grid_size.tolist())
        bev_volume = torch.zeros(
            batch_size,
            channels,
            grid_z,
            grid_x,
            grid_y,
            device=features.device,
            dtype=features.dtype,
        )

        for batch_idx in range(batch_size):
            voxel_indices = metric_to_voxel_index(
                flat_geometry[batch_idx],
                voxel_size=self.voxel_size,
                voxel_origin=self.voxel_origin,
            )

            valid_mask = (
                (voxel_indices[:, 0] >= 0) & (voxel_indices[:, 0] < grid_x) &
                (voxel_indices[:, 1] >= 0) & (voxel_indices[:, 1] < grid_y) &
                (voxel_indices[:, 2] >= 0) & (voxel_indices[:, 2] < grid_z)
            )

            valid_voxels = voxel_indices[valid_mask]
            valid_features = flat_features[batch_idx][valid_mask]

            if valid_voxels.numel() == 0:
                continue

            x_idx = valid_voxels[:, 0]
            y_idx = valid_voxels[:, 1]
            z_idx = valid_voxels[:, 2]

            bev_volume[batch_idx, :, z_idx, x_idx, y_idx] += valid_features.T

        bev = torch.cat([bev_volume[:, :, z] for z in range(bev_volume.shape[2])], dim=1)
        return bev
