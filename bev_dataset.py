import numpy as np
from nuscenes.utils.splits import create_splits_scenes

import os
import torch
from PIL import Image
from pyquaternion import Quaternion

# Part 1: Input Data Collection and Preprocessing

class BEVAutoDriveDataset:
    def __init__(self, nusc, is_train, data_aug_conf):
        """
        Args:
            nusc: NuScenes dataset handle.
            is_train: Boolean flag indicating training or validation mode.
            data_aug_conf: Dictionary containing image augmentation settings.
        """
        self.nusc = nusc
        self.is_train = is_train
        self.data_aug_conf = data_aug_conf

        # Load the list of scene names used in the current split.
        self.scenes = self._load_scene_names()

        # Collect and sort all valid samples for the selected scenes.
        self.samples = self._prepare_samples()

    def _load_scene_names(self):
        """
        Select the correct scene split according to the dataset version
        and whether the dataset is used for training or validation.

        Returns:
            A list of scene names belonging to the chosen split.
        """
        split_lookup = {
            'v1.0-trainval': {True: 'train', False: 'val'},
            'v1.0-mini': {True: 'mini_train', False: 'mini_val'},
        }

        split_name = split_lookup[self.nusc.version][self.is_train]
        scene_names = create_splits_scenes()[split_name]
        return scene_names

    def _prepare_samples(self):
        """
        Filter all samples so that only samples from the selected scenes remain.
        Then sort them first by scene and then by timestamp.

        Returns:
            A sorted list of sample records.
        """
        valid_samples = []

        for sample in self.nusc.sample:
            scene_info = self.nusc.get('scene', sample['scene_token'])
            if scene_info['name'] in self.scenes:
                valid_samples.append(sample)

        valid_samples.sort(key=lambda item: (item['scene_token'], item['timestamp']))
        return valid_samples

    def sample_augmentation(self):
        """
        Generate augmentation parameters for image preprocessing.

        During training:
        - Randomly resize the image
        - Randomly crop the image
        - Optionally apply horizontal flip
        - Randomly rotate the image

        During validation:
        - Use deterministic resize and center-style crop
        - No flip
        - No rotation

        Returns:
            resize (float): Resize scale factor
            resize_dims (tuple): Resized image dimensions as (width, height)
            crop (tuple): Crop box in PIL style (left, top, right, bottom)
            flip (bool): Whether horizontal flip is applied
            rotate (float): Rotation angle in degrees
        """
        orig_h = self.data_aug_conf['H']
        orig_w = self.data_aug_conf['W']
        final_h, final_w = self.data_aug_conf['final_dim']

        if self.is_train:
            # Random scaling for training augmentation.
            resize = np.random.uniform(*self.data_aug_conf['resize_lim'])
            resized_w = int(orig_w * resize)
            resized_h = int(orig_h * resize)
            resize_dims = (resized_w, resized_h)

            # Random crop position, with bias toward keeping the lower image area.
            bottom_ratio = np.random.uniform(*self.data_aug_conf['bot_pct_lim'])
            crop_top = int((1.0 - bottom_ratio) * resized_h) - final_h
            crop_left = int(np.random.uniform(0, max(0, resized_w - final_w)))
            crop = (crop_left, crop_top, crop_left + final_w, crop_top + final_h)

            # Optional random horizontal flip.
            flip = bool(
                self.data_aug_conf['rand_flip'] and np.random.choice([0, 1])
            )

            # Random in-plane rotation.
            rotate = np.random.uniform(*self.data_aug_conf['rot_lim'])

        else:
            # Deterministic resize so that the final crop always fits.
            resize = max(final_h / orig_h, final_w / orig_w)
            resized_w = int(orig_w * resize)
            resized_h = int(orig_h * resize)
            resize_dims = (resized_w, resized_h)

            # Use a stable crop strategy for evaluation.
            mean_bottom_ratio = np.mean(self.data_aug_conf['bot_pct_lim'])
            crop_top = int((1.0 - mean_bottom_ratio) * resized_h) - final_h
            crop_left = int(max(0, resized_w - final_w) / 2)
            crop = (crop_left, crop_top, crop_left + final_w, crop_top + final_h)

            flip = False
            rotate = 0.0

        return resize, resize_dims, crop, flip, rotate
        
# Part 2: Multi-View Image Data Extraction and Camera Configuration

class BEVAutoDriveDataset:
    """
    Dataset loader for multi-camera BEV perception.

    This module handles:
    - Image loading from NuScenes-style dataset
    - Camera parameter extraction (intrinsics & extrinsics)
    - Geometric data augmentation
    - Post-homography transformation tracking

    Designed for research in BEV perception, control, and planning.
    """

    def get_image_data(self, record, camera_list):
        """
        Load multi-camera images and corresponding calibration data.

        Args:
            record (dict): A NuScenes sample record
            camera_list (list): List of camera names (e.g., CAM_FRONT, CAM_BACK, ...)

        Returns:
            tuple:
                images      (Tensor): [N, C, H, W]
                rotations   (Tensor): [N, 3, 3]
                translations(Tensor): [N, 3]
                intrinsics  (Tensor): [N, 3, 3]
                post_rots   (Tensor): [N, 3, 3]
                post_trans  (Tensor): [N, 3]
        """

        images = []
        rotations = []
        translations = []
        intrinsics = []
        post_rots = []
        post_trans = []

        for cam in camera_list:
            # Retrieve sample data for this camera
            sample_data = self.nusc.get('sample_data', record['data'][cam])

            # Load image from disk
            img_path = os.path.join(self.nusc.dataroot, sample_data['filename'])
            img = Image.open(img_path)

            # Initialize post-transformation (image plane augmentation tracking)
            post_rot = torch.eye(3)
            post_tran = torch.zeros(3)

            # Load camera calibration parameters
            calib = self.nusc.get('calibrated_sensor', sample_data['calibrated_sensor_token'])

            intrinsic = torch.tensor(calib['camera_intrinsic'], dtype=torch.float32)

            rotation = torch.tensor(
                Quaternion(calib['rotation']).rotation_matrix,
                dtype=torch.float32
            )

            translation = torch.tensor(calib['translation'], dtype=torch.float32)

            # Sample augmentation parameters
            resize, resize_dims, crop, flip, rotate = self.sample_augmentation()

            # Apply augmentation and update post-transform
            img, rot_aug, tran_aug = img_transform(
                img,
                post_rot[:2, :2],
                post_tran[:2],
                resize=resize,
                resize_dims=resize_dims,
                crop=crop,
                flip=flip,
                rotate=rotate
            )

            # Update post transformation matrices (lift to 3D homogeneous form)
            post_rot[:2, :2] = rot_aug
            post_tran[:2] = tran_aug

            # Normalize and store
            images.append(normalize_img(img))
            intrinsics.append(intrinsic)
            rotations.append(rotation)
            translations.append(translation)
            post_rots.append(post_rot)
            post_trans.append(post_tran)

        return (
            torch.stack(images),
            torch.stack(rotations),
            torch.stack(translations),
            torch.stack(intrinsics),
            torch.stack(post_rots),
            torch.stack(post_trans),
        )

    def choose_cameras(self):
        """
        Randomly select a subset of cameras during training
        to improve robustness and reduce computation.

        Returns:
            list: selected camera names
        """

        available_cams = self.data_aug_conf['cams']

        if self.is_train and self.data_aug_conf['Ncams'] < len(available_cams):
            selected = np.random.choice(
                available_cams,
                self.data_aug_conf['Ncams'],
                replace=False
            )
        else:
            selected = available_cams

        return list(selected)

    def __len__(self):
        """Return dataset size."""
        return len(self.ixes)

    def __str__(self):
        """Readable dataset summary."""
        split = "train" if self.is_train else "val"
        return (
            f"BEVAutoDriveDataset: {len(self)} samples | "
            f"Split: {split} | "
            f"Augmentation: {self.data_aug_conf}"
        )
