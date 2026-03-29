"""
Dataset utilities for multi-camera BEV perception research.

This module provides an independent implementation of dataset indexing,
camera selection, image loading, and image-space augmentation bookkeeping
for BEV-oriented autonomous driving experiments.
"""

from __future__ import annotations

import os
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from PIL import Image
from pyquaternion import Quaternion

try:
    from nuscenes.utils.splits import create_splits_scenes
except ImportError:  # pragma: no cover
    create_splits_scenes = None


class BEVDataset:
    """
    Dataset wrapper for camera-based BEV perception.

    The class is intentionally designed as a reusable research abstraction
    rather than a thin file loader. It handles:
    - split-aware scene filtering
    - sample indexing and temporal ordering
    - camera subset selection
    - image loading and normalization
    - augmentation parameter sampling
    - image-plane transform tracking for later geometry projection
    """

    def __init__(
        self,
        nusc,
        is_train: bool,
        data_aug_conf: Dict,
        scene_names: Optional[Sequence[str]] = None,
    ) -> None:
        self.nusc = nusc
        self.is_train = is_train
        self.data_aug_conf = data_aug_conf
        self.scenes = set(scene_names) if scene_names is not None else self._get_default_scenes()
        self.samples = self._build_sample_index()

    def _get_default_scenes(self) -> set:
        """
        Retrieve the default NuScenes split for the current mode.

        Returns:
            set: scene names belonging to the selected split.
        """
        if create_splits_scenes is None:
            raise ImportError(
                "nuscenes-devkit is required to infer default train/val splits. "
                "Either install it or pass scene_names explicitly."
            )

        split_key_map = {
            "v1.0-trainval": {True: "train", False: "val"},
            "v1.0-mini": {True: "mini_train", False: "mini_val"},
        }

        if self.nusc.version not in split_key_map:
            raise ValueError(f"Unsupported NuScenes version: {self.nusc.version}")

        split_name = split_key_map[self.nusc.version][self.is_train]
        return set(create_splits_scenes()[split_name])

    def _build_sample_index(self) -> List[Dict]:
        """
        Build a time-ordered sample index restricted to the selected scenes.
        """
        samples = []
        for sample in self.nusc.sample:
            scene = self.nusc.get("scene", sample["scene_token"])
            if scene["name"] in self.scenes:
                samples.append(sample)

        samples.sort(key=lambda item: (item["scene_token"], item["timestamp"]))
        return samples

    def sample_augmentation(
        self,
    ) -> Tuple[float, Tuple[int, int], Tuple[int, int, int, int], bool, float]:
        """
        Sample geometric augmentation parameters for one image.

        Returns:
            resize_ratio: isotropic resize factor
            resize_dims: resized image size as (width, height)
            crop_box: crop region as (left, top, right, bottom)
            do_flip: whether to apply horizontal flipping
            rotate_deg: in-plane image rotation in degrees
        """
        src_h = self.data_aug_conf["H"]
        src_w = self.data_aug_conf["W"]
        out_h, out_w = self.data_aug_conf["final_dim"]

        if self.is_train:
            resize_ratio = float(np.random.uniform(*self.data_aug_conf["resize_lim"]))
            new_w = int(src_w * resize_ratio)
            new_h = int(src_h * resize_ratio)

            crop_bottom_pct = float(np.random.uniform(*self.data_aug_conf["bot_pct_lim"]))
            crop_top = int((1.0 - crop_bottom_pct) * new_h) - out_h
            crop_left = int(np.random.uniform(0, max(0, new_w - out_w)))

            crop_box = (crop_left, crop_top, crop_left + out_w, crop_top + out_h)
            do_flip = bool(
                self.data_aug_conf.get("rand_flip", False) and np.random.randint(2)
            )
            rotate_deg = float(np.random.uniform(*self.data_aug_conf["rot_lim"]))
        else:
            resize_ratio = max(out_h / src_h, out_w / src_w)
            new_w = int(src_w * resize_ratio)
            new_h = int(src_h * resize_ratio)

            crop_top = int((1.0 - np.mean(self.data_aug_conf["bot_pct_lim"])) * new_h) - out_h
            crop_left = int(max(0, new_w - out_w) / 2)

            crop_box = (crop_left, crop_top, crop_left + out_w, crop_top + out_h)
            do_flip = False
            rotate_deg = 0.0

        return resize_ratio, (new_w, new_h), crop_box, do_flip, rotate_deg

    def choose_cameras(self) -> List[str]:
        """
        Select the active camera set for the current sample.

        During training, a random subset can be chosen to improve efficiency
        and robustness. During evaluation, all configured cameras are used.
        """
        available_cameras = list(self.data_aug_conf["cams"])
        target_count = int(self.data_aug_conf.get("Ncams", len(available_cameras)))

        if self.is_train and target_count < len(available_cameras):
            chosen = np.random.choice(available_cameras, target_count, replace=False)
            return list(chosen)

        return available_cameras

    def load_camera_data(
        self,
        sample_record: Dict,
        camera_names: Sequence[str],
        image_transform: Callable,
        normalize_image: Callable,
    ) -> Tuple[torch.Tensor, ...]:
        """
        Load images and calibration data for a set of cameras.

        Args:
            sample_record: NuScenes sample dictionary.
            camera_names: list of active camera channel names.
            image_transform: callable that applies augmentation and returns
                (image, updated_post_rot, updated_post_tran).
            normalize_image: callable that converts a PIL image into a normalized tensor.

        Returns:
            Tuple containing:
                images       [N, C, H, W]
                rotations    [N, 3, 3]
                translations [N, 3]
                intrinsics   [N, 3, 3]
                post_rots    [N, 3, 3]
                post_trans   [N, 3]
        """
        images: List[torch.Tensor] = []
        rotations: List[torch.Tensor] = []
        translations: List[torch.Tensor] = []
        intrinsics: List[torch.Tensor] = []
        post_rots: List[torch.Tensor] = []
        post_trans: List[torch.Tensor] = []

        for camera_name in camera_names:
            sample_data = self.nusc.get("sample_data", sample_record["data"][camera_name])
            calibration = self.nusc.get("calibrated_sensor", sample_data["calibrated_sensor_token"])

            image_path = os.path.join(self.nusc.dataroot, sample_data["filename"])
            image = Image.open(image_path).convert("RGB")

            intrinsic = torch.tensor(calibration["camera_intrinsic"], dtype=torch.float32)
            rotation = torch.tensor(
                Quaternion(calibration["rotation"]).rotation_matrix,
                dtype=torch.float32,
            )
            translation = torch.tensor(calibration["translation"], dtype=torch.float32)

            resize_ratio, resize_dims, crop_box, do_flip, rotate_deg = self.sample_augmentation()

            post_rot_2d = torch.eye(2, dtype=torch.float32)
            post_tran_2d = torch.zeros(2, dtype=torch.float32)

            image, post_rot_2d, post_tran_2d = image_transform(
                image=image,
                post_rot=post_rot_2d,
                post_tran=post_tran_2d,
                resize=resize_ratio,
                resize_dims=resize_dims,
                crop=crop_box,
                flip=do_flip,
                rotate=rotate_deg,
            )

            post_rot_3d = torch.eye(3, dtype=torch.float32)
            post_tran_3d = torch.zeros(3, dtype=torch.float32)
            post_rot_3d[:2, :2] = post_rot_2d
            post_tran_3d[:2] = post_tran_2d

            images.append(normalize_image(image))
            rotations.append(rotation)
            translations.append(translation)
            intrinsics.append(intrinsic)
            post_rots.append(post_rot_3d)
            post_trans.append(post_tran_3d)

        return (
            torch.stack(images, dim=0),
            torch.stack(rotations, dim=0),
            torch.stack(translations, dim=0),
            torch.stack(intrinsics, dim=0),
            torch.stack(post_rots, dim=0),
            torch.stack(post_trans, dim=0),
        )

    def __getitem__(self, index: int) -> Dict:
        return self.samples[index]

    def __len__(self) -> int:
        return len(self.samples)

    def __repr__(self) -> str:
        split = "train" if self.is_train else "val"
        return (
            f"{self.__class__.__name__}(num_samples={len(self)}, "
            f"split='{split}', cameras={list(self.data_aug_conf['cams'])})"
        )
