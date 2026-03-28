import torch
import torch.nn as nn


class BEVProjectionMixin:
    """
    Utility mixin for constructing image-space frustum coordinates
    used in BEV lifting and projection modules.
    """

    def build_frustum_template(self, image_height: int, image_width: int) -> nn.Parameter:
        """
        Construct a fixed frustum grid in image space.

        The frustum is defined over:
        - x coordinates on the image plane
        - y coordinates on the image plane
        - sampled depth values along the viewing direction

        This tensor serves as a geometric template for lifting 2D image features
        into a 3D frustum before projecting them into Bird's Eye View (BEV).

        Args:
            image_height (int): Original input image height.
            image_width (int): Original input image width.

        Returns:
            nn.Parameter:
                A non-trainable tensor of shape [D, H_ds, W_ds, 3], where:
                - D is the number of depth bins
                - H_ds is the downsampled image height
                - W_ds is the downsampled image width
                - the last dimension stores (x, y, depth)
        """

        # Compute feature-map resolution after encoder downsampling
        feat_height = image_height // self.downsample
        feat_width = image_width // self.downsample

        # Depth sampling range: [d_min, d_max) with step d_step
        depth_values = torch.arange(
            *self.grid_conf["dbound"],
            dtype=torch.float32
        ).view(-1, 1, 1).expand(-1, feat_height, feat_width)

        num_depth_bins = depth_values.shape[0]

        # Uniform image-plane coordinates at the feature-map resolution
        x_coords = torch.linspace(
            0,
            image_width - 1,
            feat_width,
            dtype=torch.float32
        ).view(1, 1, feat_width).expand(num_depth_bins, feat_height, feat_width)

        y_coords = torch.linspace(
            0,
            image_height - 1,
            feat_height,
            dtype=torch.float32
        ).view(1, feat_height, 1).expand(num_depth_bins, feat_height, feat_width)

        # Stack into frustum coordinates: (x, y, depth)
        frustum = torch.stack((x_coords, y_coords, depth_values), dim=-1)

        # Register as a fixed tensor rather than a trainable model parameter
        return nn.Parameter(frustum, requires_grad=False)
