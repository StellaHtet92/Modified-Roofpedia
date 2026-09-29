"""Drop-in replacements for get_dataset_loaders, train, validate in src/train.py
for the dual-encoder multimodal FusionUNet.

Changes vs original:
  - dataset returns (rgb, struct, mask, tiles)
  - loaders build the struct mean/std once (from a sample) or use provided constants
  - net is called as net(rgb, struct)
"""

import os
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.datasets_multimodal import SlippyMapMultiModal            # add the class to datasets.py
from src.transforms_multimodal import (                            # add the classes to transforms.py
    MultiModalCompose, MMConvertMode, MMResize, MMCenterCrop,
    MMRandomHorizontalFlip, MMRandomVerticalFlip, MMRandomRotation90,
    MMToTensorNormalize,
)
from src.metrics import Metrics


def compute_struct_stats(struct_root, max_tiles=200):
    """Estimate per-channel mean/std of struct patches from a sample of .npy files.

    Run once; then you can hard-code the returned values in config for reproducibility.
    """
    paths = []
    for z in os.listdir(struct_root):
        zp = os.path.join(struct_root, z)
        if not os.path.isdir(zp):
            continue
        for x in os.listdir(zp):
            xp = os.path.join(zp, x)
            if not os.path.isdir(xp):
                continue
            for fn in os.listdir(xp):
                if fn.endswith(".npy"):
                    paths.append(os.path.join(xp, fn))
    if not paths:
        raise FileNotFoundError(f"no .npy struct patches under {struct_root}")

    rng = np.random.default_rng(0)
    rng.shuffle(paths)
    paths = paths[:max_tiles]

    # Welford-ish accumulation per channel.
    sums = None
    sqsums = None
    count = 0
    for p in paths:
        arr = np.load(p).astype(np.float64)          # (C,H,W)
        c = arr.shape[0]
        if sums is None:
            sums = np.zeros(c)
            sqsums = np.zeros(c)
        sums += arr.reshape(c, -1).sum(axis=1)
        sqsums += (arr.reshape(c, -1) ** 2).sum(axis=1)
        count += arr.reshape(c, -1).shape[1]

    mean = sums / count
    var = sqsums / count - mean ** 2
    std = np.sqrt(np.maximum(var, 1e-12))
    return mean.tolist(), std.tolist()


def get_dataset_loaders(target_size, batch_size, dataset_path,
                        struct_mean=None, struct_std=None):
    """Build train/val loaders for multimodal data.

    dataset_path/
      training/images  training/struct  training/labels
      validation/images validation/struct validation/labels
    """
    train_struct_root = os.path.join(dataset_path, "training", "struct")

    # If stats not provided, estimate from training struct patches.
    if struct_mean is None or struct_std is None:
        struct_mean, struct_std = compute_struct_stats(train_struct_root)
        print(f"[struct stats] mean={struct_mean} std={struct_std}")

    # NOTE: apply the SAME geometric augmentations to all modalities.
    # Resize/crop/convert are deterministic; flips/rotations are random but shared.
    train_tf = MultiModalCompose([
        MMConvertMode("RGB", "P"),
        MMResize(target_size),
        MMCenterCrop(target_size),
        MMRandomHorizontalFlip(0.5),
        MMRandomVerticalFlip(0.5),
        MMRandomRotation90(0.5),
        MMToTensorNormalize(struct_mean, struct_std),
    ])
    # Validation: no random augmentation.
    val_tf = MultiModalCompose([
        MMConvertMode("RGB", "P"),
        MMResize(target_size),
        MMCenterCrop(target_size),
        MMToTensorNormalize(struct_mean, struct_std),
    ])

    train_dataset = SlippyMapMultiModal(
        image_root=os.path.join(dataset_path, "training", "images"),
        struct_root=os.path.join(dataset_path, "training", "struct"),
        target_root=os.path.join(dataset_path, "training", "labels"),
        joint_transform=train_tf,
    )
    val_dataset = SlippyMapMultiModal(
        image_root=os.path.join(dataset_path, "validation", "images"),
        struct_root=os.path.join(dataset_path, "validation", "struct"),
        target_root=os.path.join(dataset_path, "validation", "labels"),
        joint_transform=val_tf,
    )

    assert len(train_dataset) > 0, "at least one tile in training dataset"
    assert len(val_dataset) > 0, "at least one tile in validation dataset"

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, drop_last=True)
    return train_loader, val_loader


def train(loader, num_classes, device, net, optimizer, criterion):
    """Train one epoch.

    No attention-balance regularizer is needed here (unlike an earlier version of
    this function): SpatialAttentionFusion (see unet_fusion.py) always combines both
    modalities before any gating happens, so there is no "pick one modality"
    decision left for training to collapse into -- the failure mode that regularizer
    existed to fight (the gate driving itself to a near-one-hot modality choice, e.g.
    a_struct_mean -> ~0, which starved a decoder block of gradient and killed it) is
    structurally impossible with this fusion design. `attn_mean` below is tracked
    purely as a diagnostic (how much of the combined signal survives on average),
    not something to regularize toward any particular value.
    """
    num_samples = 0
    running_loss = 0
    running_attn_mean = 0
    num_batches = 0
    metrics = Metrics(range(num_classes))
    net.train()

    for rgb, struct, masks, tiles in tqdm(loader, desc="Train", unit="batch", ascii=True):
        rgb = rgb.to(device)
        struct = struct.to(device)
        masks = masks.to(device)

        assert rgb.size()[2:] == masks.size()[1:], "resolutions for images and masks are in sync"
        assert struct.size()[2:] == masks.size()[1:], "resolutions for struct and masks are in sync"

        num_samples += int(rgb.size(0))
        num_batches += 1
        optimizer.zero_grad()
        outputs, attn = net(rgb, struct, return_attention=True)   # <-- dual input
        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        running_attn_mean += attn.mean().item()

        for mask, output in zip(masks, outputs):
            metrics.add(mask, output.detach())

    return {
        "loss": running_loss / num_samples,
        "miou": metrics.get_miou(),
        "fg_iou": metrics.get_fg_iou(),
        "mcc": metrics.get_mcc(),
        "attn_mean": running_attn_mean / num_batches,
    }


def validate(loader, num_classes, device, net, criterion):
    num_samples = 0
    running_loss = 0
    running_attn_mean = 0
    num_batches = 0
    metrics = Metrics(range(num_classes))

    with torch.no_grad():
        net.eval()
        for rgb, struct, masks, tiles in tqdm(loader, desc="Validate", unit="batch", ascii=True):
            rgb = rgb.to(device)
            struct = struct.to(device)
            masks = masks.to(device)

            assert rgb.size()[2:] == masks.size()[1:], "resolutions for images and masks are in sync"

            num_samples += int(rgb.size(0))
            num_batches += 1
            outputs, attn = net(rgb, struct, return_attention=True)   # <-- dual input
            loss = criterion(outputs, masks)
            running_loss += loss.item()
            running_attn_mean += attn.mean().item()

            for mask, output in zip(masks, outputs):
                metrics.add(mask, output)

    return {
        "loss": running_loss / num_samples,
        "miou": metrics.get_miou(),
        "fg_iou": metrics.get_fg_iou(),
        "mcc": metrics.get_mcc(),
        "attn_mean": running_attn_mean / num_batches,
    }
