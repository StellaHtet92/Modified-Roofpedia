"""Joint transforms for RGB (PIL) + structural (float tensor) + mask (PIL).

Add to src/transforms.py or import from here. Every geometric augmentation is applied
IDENTICALLY to all three modalities so they stay pixel-aligned. RGB and mask are handled
as PIL until the final tensor conversion; struct is a (2,H,W) float tensor throughout and
is transformed with torch ops that mirror the PIL flips/rotations.

IMPORTANT ON ROTATIONS:
  PIL Image.ROTATE_90 rotates 90 deg COUNTER-CLOCKWISE. The torch equivalent is
  torch.rot90(x, k=1, dims=(H, W)). We match them exactly so struct and RGB rotate the
  same direction.
"""

import random
import numpy as np
import torch
from PIL import Image
import torchvision.transforms.functional as TF


class MultiModalCompose:
    """Chain of multimodal transforms."""

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, rgb, struct, mask):
        for t in transforms_iter(self.transforms):
            rgb, struct, mask = t(rgb, struct, mask)
        return rgb, struct, mask


def transforms_iter(ts):
    for t in ts:
        yield t


class MMConvertMode:
    """Convert RGB and mask PIL modes; struct untouched."""

    def __init__(self, rgb_mode="RGB", mask_mode="P"):
        self.rgb_mode = rgb_mode
        self.mask_mode = mask_mode

    def __call__(self, rgb, struct, mask):
        return rgb.convert(self.rgb_mode), struct, mask.convert(self.mask_mode)


class MMResize:
    """Resize all three to (size, size). RGB bilinear, mask nearest, struct bilinear."""

    def __init__(self, size):
        self.size = (size, size)

    def __call__(self, rgb, struct, mask):
        rgb = rgb.resize(self.size, Image.BILINEAR)
        mask = mask.resize(self.size, Image.NEAREST)
        # struct: (2,H,W) tensor -> interpolate bilinear
        struct = torch.nn.functional.interpolate(
            struct.unsqueeze(0), size=self.size, mode="bilinear", align_corners=False
        ).squeeze(0)
        return rgb, struct, mask


class MMCenterCrop:
    """Center-crop all three to (size, size)."""

    def __init__(self, size):
        self.size = size

    def __call__(self, rgb, struct, mask):
        rgb = TF.center_crop(rgb, [self.size, self.size])
        mask = TF.center_crop(mask, [self.size, self.size])
        # struct tensor center crop
        _, h, w = struct.shape
        top = (h - self.size) // 2
        left = (w - self.size) // 2
        struct = struct[:, top:top + self.size, left:left + self.size]
        return rgb, struct, mask


class MMRandomHorizontalFlip:
    """Flip left-right (all three) with probability p."""

    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, rgb, struct, mask):
        if random.random() < self.p:
            rgb = rgb.transpose(Image.FLIP_LEFT_RIGHT)
            mask = mask.transpose(Image.FLIP_LEFT_RIGHT)
            struct = torch.flip(struct, dims=[2])   # flip width
        return rgb, struct, mask


class MMRandomVerticalFlip:
    """Flip top-bottom (all three) with probability p."""

    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, rgb, struct, mask):
        if random.random() < self.p:
            rgb = rgb.transpose(Image.FLIP_TOP_BOTTOM)
            mask = mask.transpose(Image.FLIP_TOP_BOTTOM)
            struct = torch.flip(struct, dims=[1])   # flip height
        return rgb, struct, mask


class MMRandomRotation90:
    """Rotate 90 deg counter-clockwise (all three) with probability p.

    Matches PIL Image.ROTATE_90 (CCW) with torch.rot90(k=1, dims=(1,2)) on (C,H,W).
    """

    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, rgb, struct, mask):
        if random.random() < self.p:
            rgb = rgb.transpose(Image.ROTATE_90)
            mask = mask.transpose(Image.ROTATE_90)
            struct = torch.rot90(struct, k=1, dims=(1, 2))
        return rgb, struct, mask


class MMToTensorNormalize:
    """Final conversion: RGB->tensor+ImageNet normalize, mask->long tensor,
    struct->per-channel standardize with provided mean/std.

    Args:
        struct_mean: list/tuple of per-channel means (len = struct channels)
        struct_std:  list/tuple of per-channel stds
    """

    IMAGENET_MEAN = [0.485, 0.456, 0.406]
    IMAGENET_STD = [0.229, 0.224, 0.225]

    def __init__(self, struct_mean, struct_std):
        self.struct_mean = torch.tensor(struct_mean, dtype=torch.float32).view(-1, 1, 1)
        self.struct_std = torch.tensor(struct_std, dtype=torch.float32).view(-1, 1, 1)

    def __call__(self, rgb, struct, mask):
        rgb = TF.to_tensor(rgb)                                   # (3,H,W) in [0,1]
        rgb = TF.normalize(rgb, self.IMAGENET_MEAN, self.IMAGENET_STD)

        mask = torch.from_numpy(np.array(mask, dtype=np.uint8)).long()  # (H,W)

        struct = (struct - self.struct_mean) / (self.struct_std + 1e-6)

        return rgb, struct, mask
