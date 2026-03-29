"""
Image preprocessing and augmentation helpers for camera-based BEV pipelines.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
import torch
from PIL import Image
from torchvision import transforms


def normalize_image(image: Image.Image) -> torch.Tensor:
    """
    Convert a PIL image into a normalized tensor.
    """
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )
    return transform(image)


def image_transform(
    image: Image.Image,
    post_rot: torch.Tensor,
    post_tran: torch.Tensor,
    resize: float,
    resize_dims: Tuple[int, int],
    crop: Tuple[int, int, int, int],
    flip: bool,
    rotate: float,
) -> tuple[Image.Image, torch.Tensor, torch.Tensor]:
    """
    Apply image-space geometric augmentation and update the associated 2D transform.

    The returned transform can later be used to compensate for augmentation
    during camera-to-BEV projection.
    """
    image = image.resize(resize_dims, resample=Image.BILINEAR)

    post_rot = post_rot.clone()
    post_tran = post_tran.clone()

    if flip:
        image = image.transpose(Image.FLIP_LEFT_RIGHT)
        flip_matrix = torch.tensor([[-1.0, 0.0], [0.0, 1.0]], dtype=torch.float32)
        flip_bias = torch.tensor([resize_dims[0], 0.0], dtype=torch.float32)
        post_rot = flip_matrix @ post_rot
        post_tran = flip_matrix @ post_tran + flip_bias

    image = image.crop(crop)
    crop_left, crop_top, _, _ = crop
    post_tran = post_tran - torch.tensor([crop_left, crop_top], dtype=torch.float32)

    if abs(rotate) > 1e-6:
        image = image.rotate(rotate)
        theta = float(np.deg2rad(rotate))
        rot_matrix = torch.tensor(
            [[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]],
            dtype=torch.float32,
        )

        image_center = torch.tensor(
            [image.size[0] / 2.0, image.size[1] / 2.0],
            dtype=torch.float32,
        )
        post_tran = rot_matrix @ (post_tran - image_center) + image_center
        post_rot = rot_matrix @ post_rot

    return image, post_rot, post_tran
