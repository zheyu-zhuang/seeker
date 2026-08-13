"""Image randomization utilities for crops, overlays, and simple augmentations."""

import os
import random
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import kornia as K
import torchvision.transforms.functional as ttf
from PIL import Image

from seeker.util.image_ops import normalize_imagenet


def _flatten(x: torch.Tensor, begin_axis: int = 1) -> torch.Tensor:
    fixed = x.size()[:begin_axis]
    return x.reshape(*fixed, -1)


def _reshape_dimensions(
    x: torch.Tensor,
    begin_axis: int,
    end_axis: int,
    target_dims: tuple[int, ...],
) -> torch.Tensor:
    assert begin_axis <= end_axis
    shape = x.shape
    out_shape = []
    for i in range(len(shape)):
        if i == begin_axis:
            out_shape.extend(target_dims)
        elif i < begin_axis or i > end_axis:
            out_shape.append(shape[i])
    return x.reshape(*out_shape)


def _join_dimensions(x: torch.Tensor, begin_axis: int, end_axis: int) -> torch.Tensor:
    return _reshape_dimensions(x, begin_axis, end_axis, (-1,))


def _unsqueeze_expand_at(x: torch.Tensor, size: int, dim: int) -> torch.Tensor:
    x = x.unsqueeze(dim)
    target = list(x.shape)
    target[dim] = size
    return x.expand(*target)


class CropRandomizer(nn.Module):
    """
    Randomly sample crops at input, and then average across crop features at output.
    """

    def __init__(
        self,
        input_shape,
        crop_size,
        num_crops=1,
        pos_enc=False,
    ):
        """
        Args:
            input_shape (tuple, list): shape of input (not including batch dimension)
            crop_height (int): crop height
            crop_width (int): crop width
            num_crops (int): number of random crops to take
            pos_enc (bool): if True, add 2 channels to the output to encode the spatial
                location of the cropped pixels in the source image
        """
        super().__init__()

        assert len(input_shape) == 3  # (C, H, W)
        assert crop_size < input_shape[1]
        assert crop_size < input_shape[2]

        self.input_shape = input_shape
        self.crop_height = crop_size
        self.crop_width = crop_size
        self.num_crops = num_crops
        self.pos_enc = pos_enc

    def forward(self, inputs, center_crop=False):
        """
        Samples N random crops for each input in the batch, and then reshapes
        inputs to [B * N, ...].
        """
        assert len(inputs.shape) >= 3  # must have at least (C, H, W) dimensions
        if self.training and not center_crop:
            # generate random crops
            out, _ = sample_random_image_crops(
                images=inputs,
                crop_height=self.crop_height,
                crop_width=self.crop_width,
                num_crops=self.num_crops,
                pos_enc=self.pos_enc,
            )
            # [B, N, ...] -> [B * N, ...]
            return _join_dimensions(out, 0, 1)
        else:
            # take center crop during eval
            out = ttf.center_crop(
                img=inputs, output_size=(self.crop_height, self.crop_width)
            )
            if self.num_crops > 1:
                B, C, H, W = out.shape
                out = (
                    out.unsqueeze(1)
                    .expand(B, self.num_crops, C, H, W)
                    .reshape(-1, C, H, W)
                )
                # [B * N, ...]
            return out

    def __repr__(self):
        """Pretty print network."""
        header = "{}".format(str(self.__class__.__name__))
        msg = header + "(input_shape={}, crop_size=[{}, {}], num_crops={})".format(
            self.input_shape, self.crop_height, self.crop_width, self.num_crops
        )
        return msg


def crop_image_from_indices(images, crop_indices, crop_height, crop_width):
    """
    Crops images at the locations specified by @crop_indices. Crops will be
    taken across all channels.

    Args:
        images (torch.Tensor): batch of images of shape [..., C, H, W]

        crop_indices (torch.Tensor): batch of indices of shape [..., N, 2] where
            N is the number of crops to take per image and each entry corresponds
            to the pixel height and width of where to take the crop. Note that
            the indices can also be of shape [..., 2] if only 1 crop should
            be taken per image. Leading dimensions must be consistent with
            @images argument. Each index specifies the top left of the crop.
            Values must be in range [0, H - CH - 1] x [0, W - CW - 1] where
            H and W are the height and width of @images and CH and CW are
            @crop_height and @crop_width.

        crop_height (int): height of crop to take

        crop_width (int): width of crop to take

    Returns:
        crops (torch.Tesnor): cropped images of shape [..., C, @crop_height, @crop_width]
    """

    # make sure length of input shapes is consistent
    assert crop_indices.shape[-1] == 2
    ndim_im_shape = len(images.shape)
    ndim_indices_shape = len(crop_indices.shape)
    assert (ndim_im_shape == ndim_indices_shape + 1) or (
        ndim_im_shape == ndim_indices_shape + 2
    )

    # maybe pad so that @crop_indices is shape [..., N, 2]
    is_padded = False
    if ndim_im_shape == ndim_indices_shape + 2:
        crop_indices = crop_indices.unsqueeze(-2)
        is_padded = True

    # make sure leading dimensions between images and indices are consistent
    assert images.shape[:-3] == crop_indices.shape[:-2]

    device = images.device
    image_c, image_h, image_w = images.shape[-3:]
    num_crops = crop_indices.shape[-2]

    # make sure @crop_indices are in valid range
    assert (crop_indices[..., 0] >= 0).all().item()
    assert (crop_indices[..., 0] < (image_h - crop_height)).all().item()
    assert (crop_indices[..., 1] >= 0).all().item()
    assert (crop_indices[..., 1] < (image_w - crop_width)).all().item()

    # convert each crop index (ch, cw) into a list of pixel indices that correspond to the entire window.

    # 2D index array with columns [0, 1, ..., CH - 1] and shape [CH, CW]
    crop_ind_grid_h = torch.arange(crop_height).to(device)
    crop_ind_grid_h = _unsqueeze_expand_at(crop_ind_grid_h, size=crop_width, dim=-1)
    # 2D index array with rows [0, 1, ..., CW - 1] and shape [CH, CW]
    crop_ind_grid_w = torch.arange(crop_width).to(device)
    crop_ind_grid_w = _unsqueeze_expand_at(crop_ind_grid_w, size=crop_height, dim=0)
    # combine into shape [CH, CW, 2]
    crop_in_grid = torch.cat(
        (crop_ind_grid_h.unsqueeze(-1), crop_ind_grid_w.unsqueeze(-1)), dim=-1
    )

    # Add above grid with the offset index of each sampled crop to get 2d indices for each crop.
    # After broadcasting, this will be shape [..., N, CH, CW, 2] and each crop has a [CH, CW, 2]
    # shape array that tells us which pixels from the corresponding source image to grab.
    grid_reshape = [1] * len(crop_indices.shape[:-1]) + [crop_height, crop_width, 2]
    all_crop_inds = crop_indices.unsqueeze(-2).unsqueeze(-2) + crop_in_grid.reshape(
        grid_reshape
    )

    # For using @torch.gather, convert to flat indices from 2D indices, and also
    # repeat across the channel dimension. To get flat index of each pixel to grab for
    # each sampled crop, we just use the mapping: ind = h_ind * @image_w + w_ind
    all_crop_inds = (
        all_crop_inds[..., 0] * image_w + all_crop_inds[..., 1]
    )  # shape [..., N, CH, CW]
    all_crop_inds = _unsqueeze_expand_at(
        all_crop_inds, size=image_c, dim=-3
    )  # shape [..., N, C, CH, CW]
    all_crop_inds = _flatten(all_crop_inds, begin_axis=-2)  # shape [..., N, C, CH * CW]

    # Repeat and flatten the source images -> [..., N, C, H * W] and then use gather to index with crop pixel inds
    images_to_crop = _unsqueeze_expand_at(images, size=num_crops, dim=-4)
    images_to_crop = _flatten(images_to_crop, begin_axis=-2)
    crops = torch.gather(images_to_crop, dim=-1, index=all_crop_inds)
    # [..., N, C, CH * CW] -> [..., N, C, CH, CW]
    reshape_axis = len(crops.shape) - 1
    crops = _reshape_dimensions(
        crops,
        begin_axis=reshape_axis,
        end_axis=reshape_axis,
        target_dims=(crop_height, crop_width),
    )

    if is_padded:
        # undo padding -> [..., C, CH, CW]
        crops = crops.squeeze(-4)
    return crops


def sample_random_image_crops(
    images, crop_height, crop_width, num_crops, pos_enc=False
):
    """
    For each image, randomly sample @num_crops crops of size (@crop_height, @crop_width), from
    @images.

    Args:
        images (torch.Tensor): batch of images of shape [..., C, H, W]

        crop_height (int): height of crop to take

        crop_width (int): width of crop to take

        num_crops (n): number of crops to sample

        pos_enc (bool): if True, also add 2 channels to the outputs that gives a spatial
            encoding of the original source pixel locations. This means that the
            output crops will contain information about where in the source image
            it was sampled from.

    Returns:
        crops (torch.Tensor): crops of shape (..., @num_crops, C, @crop_height, @crop_width)
            if @pos_enc is False, otherwise (..., @num_crops, C + 2, @crop_height, @crop_width)

        crop_inds (torch.Tensor): sampled crop indices of shape (..., N, 2)
    """
    device = images.device

    # maybe add 2 channels of spatial encoding to the source image
    source_im = images
    if pos_enc:
        # spatial encoding [y, x] in [0, 1]
        h, w = source_im.shape[-2:]
        pos_y, pos_x = torch.meshgrid(torch.arange(h), torch.arange(w))
        pos_y = pos_y.float().to(device) / float(h)
        pos_x = pos_x.float().to(device) / float(w)
        position_enc = torch.stack((pos_y, pos_x))  # shape [C, H, W]

        # unsqueeze and expand to match leading dimensions -> shape [..., C, H, W]
        leading_shape = source_im.shape[:-3]
        position_enc = position_enc[(None,) * len(leading_shape)]
        position_enc = position_enc.expand(*leading_shape, -1, -1, -1)

        # concat across channel dimension with input
        source_im = torch.cat((source_im, position_enc), dim=-3)

    # make sure sample boundaries ensure crops are fully within the images
    image_c, image_h, image_w = source_im.shape[-3:]
    max_sample_h = image_h - crop_height
    max_sample_w = image_w - crop_width

    # Sample crop locations for all tensor dimensions up to the last 3, which are [C, H, W].
    # Each gets @num_crops samples - typically this will just be the batch dimension (B), so
    # we will sample [B, N] indices, but this supports having more than one leading dimension,
    # or possibly no leading dimension.
    #
    # Trick: sample in [0, 1) with rand, then re-scale to [0, M) and convert to long to get sampled ints
    crop_inds_h = (
        max_sample_h * torch.rand(*source_im.shape[:-3], num_crops).to(device)
    ).long()
    crop_inds_w = (
        max_sample_w * torch.rand(*source_im.shape[:-3], num_crops).to(device)
    ).long()
    crop_inds = torch.cat(
        (crop_inds_h.unsqueeze(-1), crop_inds_w.unsqueeze(-1)), dim=-1
    )  # shape [..., N, 2]

    crops = crop_image_from_indices(
        images=source_im,
        crop_indices=crop_inds,
        crop_height=crop_height,
        crop_width=crop_width,
    )

    return crops, crop_inds


class BackgroundRandomizer(torch.nn.Module):
    """
    Memory-efficient background randomization for visual augmentation.

    Background images are preloaded as uint8 and optionally kept on GPU.
    Random geometric/photometric transforms and ImageNet normalization
    are applied on-the-fly when sampling.

    - Stored as uint8 for low memory footprint
    - Moved to GPU once via `.to(device)` if desired
    - Returns float32 ImageNet-normalized images

    Args:
        input_shape (Tuple[int, int]): Output image size (H, W).
        background_path (str): Directory of background images.

    Returns:
        torch.Tensor: [B, 3, H, W] ImageNet-normalized float32 tensor.
    """

    def __init__(self, input_shape: Tuple[int, int], background_path: str):
        assert (
            input_shape is not None and len(input_shape) == 2
        ), "input_shape must be (H, W)"
        assert isinstance(background_path, str), "background_path must be a string"
        assert os.path.exists(background_path), "background_path does not exist"

        super().__init__()

        self.H, self.W = int(input_shape[0]), int(input_shape[1])
        self.background_path = background_path

        self.register_buffer(
            "backgrounds_u8",
            self._preload_all_u8(background_path),
            persistent=False,  # not saved in checkpoints
        )

    def get_config(self) -> dict:
        """
        Return config dict for serialization.
        """
        return {
            "input_shape": (self.H, self.W),
            "num_backgrounds": int(self.backgrounds_u8.shape[0]),
            "background_path": self.background_path,
        }

    @torch.no_grad()
    def forward(self, num_samples: int) -> torch.Tensor:
        """
        Sample random backgrounds and return ImageNet-normalized float32 tensor:
            [num_samples, 3, H, W]
        """
        # sanity checks
        assert num_samples > 0, "num_samples must be > 0"
        N_total = int(self.backgrounds_u8.shape[0])
        msg = f"Requested {num_samples} samples exceeds available backgrounds={N_total}"
        if num_samples > N_total:
            raise ValueError(msg)

        idx = torch.tensor(
            random.sample(range(N_total), num_samples),
            dtype=torch.long,
            device=self.backgrounds_u8.device,
        )

        bg_u8 = self.backgrounds_u8.index_select(0, idx)  # [B,3,H,W] uint8
        bg = bg_u8.to(torch.float32).div_(255.0)
        # transforms in float space, then normalize
        bg = self._random_transform(bg)
        bg = normalize_imagenet(bg, source="float01")
        return bg

    # ------------------------------------------------------------------ #
    # Transforms
    # ------------------------------------------------------------------ #
    def _random_transform(self, bg: torch.Tensor) -> torch.Tensor:
        """
        bg: [B,3,H,W] float in [0,1]
        Applies:
          1) random rotation in [0,360)
          2) random brightness in [-0.1, +0.1]
        """
        B, _, H, W = bg.shape
        if (H != self.H) or (W != self.W):
            bg = F.interpolate(
                bg, size=(self.H, self.W), mode="bilinear", align_corners=False
            )

        angles = torch.rand(B, device=bg.device, dtype=bg.dtype) * 360.0
        bg = K.geometry.transform.rotate(
            bg, angles, mode="bilinear", padding_mode="border"
        )

        brightness = (torch.rand(B, device=bg.device, dtype=bg.dtype) * 0.2) - 0.1
        bg = K.enhance.adjust_brightness(bg, brightness)

        return bg.clamp(0.0, 1.0)

    # ------------------------------------------------------------------ #
    # Loading
    # ------------------------------------------------------------------ #
    def _preload_all_u8(self, background_path: str) -> torch.Tensor:
        """
        Preload all images into uint8 tensor [N,3,H,W] on CPU.
        """
        if os.path.isfile(background_path):
            return self._load_background_pack(background_path)

        files = [
            f
            for f in os.listdir(background_path)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        ]
        files.sort()
        if len(files) == 0:
            raise FileNotFoundError(f"No .jpg/.png images found in {background_path}")

        out = torch.empty((len(files), 3, self.H, self.W), dtype=torch.uint8)

        for i, name in enumerate(files):
            path = os.path.join(background_path, name)
            im = Image.open(path).convert("RGB")

            # PIL uses (W,H)
            if im.size != (self.W, self.H):
                im = im.resize((self.W, self.H), resample=Image.BILINEAR)

            arr = np.asarray(im, dtype=np.uint8)  # [H,W,3]
            t = torch.from_numpy(arr.copy()).permute(2, 0, 1)
            out[i] = t

        return out

    def _load_background_pack(self, path: str) -> torch.Tensor:
        payload = torch.load(path, map_location="cpu")
        if isinstance(payload, dict):
            if "backgrounds_u8" in payload:
                payload = payload["backgrounds_u8"]
            elif "images" in payload:
                payload = payload["images"]
            else:
                keys = ", ".join(sorted(str(key) for key in payload.keys()))
                raise KeyError(
                    f"Background pack {path} must contain 'backgrounds_u8' "
                    f"or 'images'; found keys: {keys}"
                )
        if not torch.is_tensor(payload):
            raise TypeError(f"Background pack {path} must contain a tensor")
        if payload.ndim != 4 or int(payload.shape[1]) != 3:
            raise ValueError(
                f"Background pack {path} must have shape [N,3,H,W], "
                f"got {tuple(payload.shape)}"
            )
        if payload.dtype != torch.uint8:
            payload = payload.clamp(0, 255).to(torch.uint8)
        if tuple(payload.shape[-2:]) != (self.H, self.W):
            payload = F.interpolate(
                payload.float(),
                size=(self.H, self.W),
                mode="bilinear",
                align_corners=False,
            ).round().clamp(0, 255).to(torch.uint8)
        return payload.contiguous()
