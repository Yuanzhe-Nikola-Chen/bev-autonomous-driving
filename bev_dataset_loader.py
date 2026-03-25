import numpy as np
from nuscenes.utils.splits import create_splits_scenes


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
