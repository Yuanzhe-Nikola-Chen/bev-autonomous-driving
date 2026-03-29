"""
Minimal example showing how the refactored modules fit together.
"""

from bev_dataset import BEVDataset
from bev_geometry import CameraToEgoProjector, FrustumGrid
from bev_encoders import BEVFeatureEncoder, CameraFeatureEncoder
from bev_pooling import VoxelPooler
from utils.image_ops import image_transform, normalize_image


def build_pipeline(nusc, aug_config, grid_conf):
    dataset = BEVDataset(
        nusc=nusc,
        is_train=True,
        data_aug_conf=aug_config,
    )

    frustum_builder = FrustumGrid(grid_conf=grid_conf, downsample=16)
    frustum = frustum_builder.build(image_size=(aug_config["H"], aug_config["W"]))

    projector = CameraToEgoProjector(frustum=frustum)
    camera_encoder = CameraFeatureEncoder(in_channels=3, feat_channels=64, depth_bins=frustum.shape[0])

    voxel_pooler = VoxelPooler(
        voxel_size=(0.5, 0.5, 0.5),
        voxel_origin=(-50.0, -50.0, -5.0),
        grid_size=(200, 200, 16),
    )

    bev_encoder = BEVFeatureEncoder(in_channels=64 * 16, out_channels=128)

    return {
        "dataset": dataset,
        "projector": projector,
        "camera_encoder": camera_encoder,
        "voxel_pooler": voxel_pooler,
        "bev_encoder": bev_encoder,
        "image_transform": image_transform,
        "normalize_image": normalize_image,
    }
