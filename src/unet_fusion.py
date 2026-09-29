"""Dual-encoder spatial-attention fusion U-Net for structurally-informed rooftop
classification.

Extends the original Roofpedia AlbuNet (ResNet-50 encoder + U-Net decoder) with a
second lightweight encoder branch for structural modalities (nDSM + slope + aspect),
fused with the spectral (RGB) features at the bottleneck through a spatial attention
gate that decides WHERE to emphasize the combined representation.

The RGB encoder and decoder tensor shapes are IDENTICAL to the original UNet in
unet.py, so the spectral pathway and all decoder shapes are preserved. This keeps
the comparison to the RGB-only baseline fair: only the elevation branch and the
fusion module are new.

Fusion design (see SpatialAttentionFusion below): BOTH modalities are combined
unconditionally into one representation, and a single-channel spatial gate then
decides how much of THAT combined representation to keep at each location -- it
never decides which modality to trust. This replaces an earlier design
(CrossModalAttentionFusion, a per-pixel softmax over {RGB, struct} producing a
convex combination) that let the network pick a modality per pixel. That design
empirically collapsed: the gate drove itself to a near-one-hot split (e.g.
a_rgb_mean -> ~1.0, a_struct_mean -> ~0.0, or the reverse) within the first few
epochs of a 200-epoch run and stayed there, starving the losing modality's gradient
and, downstream, permanently killing a decoder block (dec0 was measured 100% zero,
every channel and pixel, on 1381/1381 validation tiles). Combining both modalities
before any gating removes that failure mode structurally: there's no "pick a
modality" decision left for training to collapse into.

One deliberate deviation from unet.py: ConvRelu below adds a BatchNorm2d between
the conv and the ReLU (the original has plain conv+ReLU, no normalisation). This
was added (independently of the fusion redesign above) after diagnosing dec0 going
permanently dead under the old fusion design -- checked empirically to be 100% zero
on all validation tiles for a checkpoint trained without it. Without BatchNorm, a
decoder ConvRelu block whose input distribution skews heavily one-sided has nothing
to keep its pre-activations centered, and under plain ReLU a block that goes
all-negative for its entire training input distribution dies permanently with no
gradient path back. Kept here as a general safeguard even though the fusion redesign
above already removes the specific sparsity pattern that triggered it originally.
If you need strict parity with the RGB-only baseline (e.g. an ablation table), add
the same BatchNorm to unet.py's ConvRelu too, or retrain both.

References:
- Original AlbuNet: unet.py (this repository)
- https://arxiv.org/abs/1505.04597  U-Net
- https://arxiv.org/abs/1512.03385  ResNet
"""

import torch
import torch.nn as nn

from torchvision.models import resnet50, resnet18


# ---------------------------------------------------------------------------
# Original building blocks (ConvRelu adds BatchNorm vs. unet.py -- see module
# docstring above)
# ---------------------------------------------------------------------------
class ConvRelu(nn.Module):
    """3x3 convolution + BatchNorm + ReLU activation building block."""

    def __init__(self, num_in, num_out):
        super().__init__()
        self.block = nn.Conv2d(num_in, num_out, kernel_size=3, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(num_out)

    def forward(self, x):
        return nn.functional.relu(self.bn(self.block(x)), inplace=True)


class DecoderBlock(nn.Module):
    """Decoder building block upsampling resolution by a factor of two."""

    def __init__(self, num_in, num_out):
        super().__init__()
        self.block = ConvRelu(num_in, num_out)

    def forward(self, x):
        return self.block(nn.functional.interpolate(x, scale_factor=2, mode="nearest"))


# ---------------------------------------------------------------------------
# Spatial attention fusion module (NEW)
# ---------------------------------------------------------------------------
class SpatialAttentionFusion(nn.Module):
    """Fuses spectral and structural bottleneck features by combining BOTH modalities
    unconditionally, then applying a spatial attention gate that decides which
    locations of the combined representation to keep -- not which modality to trust.

        F_combined = ConvRelu( Concat[F_rgb, F_dsm] )      # both modalities, every pixel, always
        gate       = sigmoid( Conv(F_combined) )            # (B, 1, H, W) spatial mask in [0,1]
        F_fused    = refine( F_combined * gate )

    Unlike a per-modality softmax gate, `gate` here operates strictly downstream of
    `combine`, which has already mixed both modalities into every channel -- so there
    is no way for the gate to express "ignore struct entirely" the way the earlier
    per-pixel modality-softmax design could (and did, empirically). It can only say
    "how much of the (always-both) combined signal survives here", e.g. suppressing
    background/non-roof regions.

    Args:
        channels: channel count C of each incoming modality (they must match).
    """

    def __init__(self, channels):
        super().__init__()
        self.channels = channels

        # Combine both modalities into one C-channel representation. A learned 3x3
        # conv over the concatenation (not a fixed average), so the network can
        # learn how to mix the two per-channel -- but every output channel is a
        # function of both inputs, so both are always present downstream.
        self.combine = ConvRelu(channels * 2, channels)

        # Spatial attention gate: single-channel sigmoid mask over the already-
        # combined features.
        self.gate = nn.Conv2d(channels, 1, kernel_size=3, padding=1, bias=True)

        # Post-fusion refinement so the gated features can adapt before the decoder.
        self.refine = ConvRelu(channels, channels)

    def forward(self, f_rgb, f_dsm):
        """Returns (fused, attn). attn: (B, 1, H, W) spatial gate in [0, 1]."""
        # f_rgb, f_dsm: (B, C, H, W) with matching shapes
        combined = self.combine(torch.cat([f_rgb, f_dsm], dim=1))   # (B, C, H, W)
        attn = torch.sigmoid(self.gate(combined))                    # (B, 1, H, W)
        fused = combined * attn
        fused = self.refine(fused)
        return fused, attn


# ---------------------------------------------------------------------------
# Lightweight structural encoder (NEW) - ResNet-18, adapted to 2 input channels
# ---------------------------------------------------------------------------
class ElevationEncoder(nn.Module):
    """A lightweight ResNet-18 encoder for the structural modalities (nDSM + slope+ aspect).

    Mirrors the spatial downsampling of the ResNet-50 RGB encoder so that its deepest
    feature map aligns with `enc4` of the RGB branch. ResNet-18's layer4 outputs 512
    channels; we project these to 2048 to match the RGB bottleneck for fusion.

    Args:
        in_channels: number of structural input channels (4 for nDSM + slope + aspect).
        out_channels: channels to project the bottleneck to (2048 to match ResNet-50).
        pretrained: use ImageNet weights for the RGB-derived layers (input conv is reinit).
    """

    def __init__(self, in_channels=4, out_channels=2048, pretrained=True):
        super().__init__()
        backbone = resnet18(pretrained=pretrained)

        # Replace the first conv to accept `in_channels` instead of 3.
        old_conv = backbone.conv1
        new_conv = nn.Conv2d(
            in_channels, old_conv.out_channels,
            kernel_size=old_conv.kernel_size, stride=old_conv.stride,
            padding=old_conv.padding, bias=(old_conv.bias is not None),
        )
        # Initialise new conv from the mean of the pretrained RGB filters (sensible prior).
        if pretrained:
            with torch.no_grad():
                mean_w = old_conv.weight.mean(dim=1, keepdim=True)          # (64,1,7,7)
                new_conv.weight.copy_(mean_w.repeat(1, in_channels, 1, 1))
        self.conv1 = new_conv
        self.bn1 = backbone.bn1
        self.relu = backbone.relu
        self.maxpool = backbone.maxpool
        self.layer1 = backbone.layer1   # 64
        self.layer2 = backbone.layer2   # 128
        self.layer3 = backbone.layer3   # 256
        self.layer4 = backbone.layer4   # 512

        # Project 512 -> out_channels (2048) to match RGB bottleneck.
        self.project = nn.Conv2d(512, out_channels, kernel_size=1, bias=False)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)              # (B, 512, H/32, W/32)
        return self.project(x)          # (B, 2048, H/32, W/32)


# ---------------------------------------------------------------------------
# Dual-encoder spatial-attention fusion U-Net (NEW top-level model)
# ---------------------------------------------------------------------------
class FusionUNet(nn.Module):
    """Dual-encoder AlbuNet with spatial-attention fusion at the bottleneck.

    The RGB branch and decoder are identical to the original UNet. The structural
    branch encodes nDSM + slope + aspect, and fusion happens on the deepest RGB
    feature (enc4) before the `center` decoder block. Decoder skip connections still
    come from the RGB encoder, so every decoder shape is unchanged.

    Args:
        num_classes:    number of output classes (4).
        struct_channels: number of structural input channels (4: nDSM + slope + aspect).
        num_filters:    decoder width, identical to the original (32).
        pretrained:     ImageNet weights for both encoders.
    """

    def __init__(self, num_classes, struct_channels=4, num_filters=32, pretrained=True):
        super().__init__()

        # ---- RGB branch: IDENTICAL to original UNet ----
        self.resnet = resnet50(pretrained=pretrained)

        # ---- Structural branch (new) ----
        self.elev_encoder = ElevationEncoder(
            in_channels=struct_channels, out_channels=2048, pretrained=pretrained
        )

        # ---- Spatial-attention fusion at the bottleneck (new) ----
        self.fusion = SpatialAttentionFusion(channels=2048)

        # ---- Decoder: IDENTICAL to original UNet ----
        self.center = DecoderBlock(2048, num_filters * 8)
        self.dec0 = DecoderBlock(2048 + num_filters * 8, num_filters * 8)
        self.dec1 = DecoderBlock(1024 + num_filters * 8, num_filters * 8)
        self.dec2 = DecoderBlock(512 + num_filters * 8, num_filters * 2)
        self.dec3 = DecoderBlock(256 + num_filters * 2, num_filters * 2 * 2)
        self.dec4 = DecoderBlock(num_filters * 2 * 2, num_filters)
        self.dec5 = ConvRelu(num_filters, num_filters)
        self.final = nn.Conv2d(num_filters, num_classes, kernel_size=1)

    def forward(self, x_rgb, x_struct, return_attention=False):
        """
        Args:
            x_rgb:    (B, 3, H, W) RGB input.
            x_struct: (B, C, H, W) structural input (nDSM + slope + aspect), same H, W as RGB.
            return_attention: if True, also returns the spatial gate, shape (B, 1, H/32, W/32).
        """
        size = x_rgb.size()
        assert size[-1] % 32 == 0 and size[-2] % 32 == 0, \
            "image resolution has to be divisible by 32 for resnet"

        # ---- RGB encoder (identical) ----
        enc0 = self.resnet.conv1(x_rgb)
        enc0 = self.resnet.bn1(enc0)
        enc0 = self.resnet.relu(enc0)
        enc0 = self.resnet.maxpool(enc0)

        enc1 = self.resnet.layer1(enc0)
        enc2 = self.resnet.layer2(enc1)
        enc3 = self.resnet.layer3(enc2)
        enc4 = self.resnet.layer4(enc3)          # (B, 2048, H/32, W/32)

        # ---- Structural encoder ----
        struct_feat = self.elev_encoder(x_struct)  # (B, 2048, H/32, W/32)

        # ---- Spatial-attention fusion at the bottleneck ----
        fused_enc4, attn = self.fusion(enc4, struct_feat)   # (B, 2048, H/32, W/32)

        # ---- Decoder (identical; consumes the fused bottleneck) ----
        center = self.center(nn.functional.max_pool2d(fused_enc4, kernel_size=2, stride=2))

        dec0 = self.dec0(torch.cat([fused_enc4, center], dim=1))
        dec1 = self.dec1(torch.cat([enc3, dec0], dim=1))
        dec2 = self.dec2(torch.cat([enc2, dec1], dim=1))
        dec3 = self.dec3(torch.cat([enc1, dec2], dim=1))
        dec4 = self.dec4(dec3)
        dec5 = self.dec5(dec4)

        out = self.final(dec5)

        if return_attention:
            return out, attn
        return out


# ---------------------------------------------------------------------------
# Quick shape self-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    model = FusionUNet(num_classes=4, struct_channels=4, pretrained=False)
    rgb = torch.randn(2, 3, 256, 256)
    struct = torch.randn(2, 4, 256, 256)
    y = model(rgb, struct)
    print("output:", y.shape)                       # expect (2, 4, 256, 256)
    y2, a = model(rgb, struct, return_attention=True)
    print("attention:", a.shape)                    # expect (2, 1, 8, 8)
    print("attention in [0,1]:", bool((a.min() >= 0) and (a.max() <= 1)))
