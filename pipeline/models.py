"""Image-only U-Net and structure-aware FusionUNet sharing one decoder definition.

Both networks are the Roofpedia AlbuNet (ResNet-50 encoder + U-Net decoder). The decoder uses
Conv-BN-ReLU blocks in BOTH networks, so the only difference between them is the structural
encoder and the bottleneck fusion block. A 4-band (R, G, B, NIR) input is supported by widening
the first convolution; the NIR filters are initialised with the mean of the pretrained RGB filters.
"""
import torch
import torch.nn as nn
from torchvision.models import resnet50, resnet18, ResNet50_Weights, ResNet18_Weights


class ConvRelu(nn.Module):
    def __init__(self, num_in, num_out):
        super().__init__()
        self.block = nn.Conv2d(num_in, num_out, kernel_size=3, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(num_out)

    def forward(self, x):
        return nn.functional.relu(self.bn(self.block(x)), inplace=True)


class DecoderBlock(nn.Module):
    def __init__(self, num_in, num_out):
        super().__init__()
        self.block = ConvRelu(num_in, num_out)

    def forward(self, x):
        return self.block(nn.functional.interpolate(x, scale_factor=2, mode='nearest'))


def widen_conv1(conv, in_channels):
    if in_channels == 3:
        return conv
    new = nn.Conv2d(in_channels, conv.out_channels, kernel_size=conv.kernel_size, stride=conv.stride,
                    padding=conv.padding, bias=conv.bias is not None)
    with torch.no_grad():
        new.weight[:, :3] = conv.weight
        new.weight[:, 3:] = conv.weight.mean(1, keepdim=True)
    return new


class StructEncoder(nn.Module):
    """ResNet-18 encoder for nDSM, slope, sin(aspect), cos(aspect); 512 -> 2048 projection."""

    def __init__(self, in_channels=4, out_channels=2048):
        super().__init__()
        b = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        conv = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        with torch.no_grad():
            conv.weight.copy_(b.conv1.weight.mean(1, keepdim=True).repeat(1, in_channels, 1, 1))
        self.stem = nn.Sequential(conv, b.bn1, b.relu, b.maxpool)
        self.layers = nn.Sequential(b.layer1, b.layer2, b.layer3, b.layer4)
        self.project = nn.Conv2d(512, out_channels, kernel_size=1, bias=False)

    def forward(self, x):
        return self.project(self.layers(self.stem(x)))


class ConcatFusion(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.combine = ConvRelu(channels * 2, channels)
        self.refine = ConvRelu(channels, channels)

    def forward(self, a, b):
        return self.refine(self.combine(torch.cat([a, b], dim=1)))


class SegNet(nn.Module):
    """fusion=False: image-only baseline; fusion=True: structure-aware dual-encoder network."""

    def __init__(self, num_classes=5, in_channels=3, fusion=False, struct_channels=4, num_filters=32):
        super().__init__()
        self.resnet = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
        self.resnet.conv1 = widen_conv1(self.resnet.conv1, in_channels)
        self.fusion = fusion
        if fusion:
            self.struct_encoder = StructEncoder(struct_channels, 2048)
            self.fuse = ConcatFusion(2048)
        f = num_filters
        self.center = DecoderBlock(2048, f * 8)
        self.dec0 = DecoderBlock(2048 + f * 8, f * 8)
        self.dec1 = DecoderBlock(1024 + f * 8, f * 8)
        self.dec2 = DecoderBlock(512 + f * 8, f * 2)
        self.dec3 = DecoderBlock(256 + f * 2, f * 2 * 2)
        self.dec4 = DecoderBlock(f * 2 * 2, f)
        self.dec5 = ConvRelu(f, f)
        self.final = nn.Conv2d(f, num_classes, kernel_size=1)

    def forward(self, x, struct=None):
        r = self.resnet
        enc0 = r.maxpool(r.relu(r.bn1(r.conv1(x))))
        enc1 = r.layer1(enc0); enc2 = r.layer2(enc1); enc3 = r.layer3(enc2); enc4 = r.layer4(enc3)
        if self.fusion:
            enc4 = self.fuse(enc4, self.struct_encoder(struct))
        center = self.center(nn.functional.max_pool2d(enc4, kernel_size=2, stride=2))
        dec0 = self.dec0(torch.cat([enc4, center], 1))
        dec1 = self.dec1(torch.cat([enc3, dec0], 1))
        dec2 = self.dec2(torch.cat([enc2, dec1], 1))
        dec3 = self.dec3(torch.cat([enc1, dec2], 1))
        return self.final(self.dec5(self.dec4(dec3)))


class LovaszLoss2d(nn.Module):
    """Lovász hinge on one-hot logits, identical to the Roofpedia implementation."""

    def forward(self, inputs, targets):
        N, C, H, W = inputs.size()
        masks = torch.zeros(N, C, H, W, device=targets.device).scatter_(1, targets.view(N, 1, H, W), 1)
        loss = 0.
        for mask, inp in zip(masks.view(N, -1), inputs.reshape(N, -1)):
            err = 1. - ((mask * 2 - 1) * inp)
            err_sorted, idx = torch.sort(err, descending=True)
            lab = mask[idx]
            inter = lab.sum() - lab.cumsum(0)
            union = lab.sum() + (1. - lab).cumsum(0)
            iou = 1. - inter / union
            iou = torch.cat([iou[:1], iou[1:] - iou[:-1]])
            loss = loss + torch.dot(nn.functional.relu(err_sorted), iou)
        return loss / N
