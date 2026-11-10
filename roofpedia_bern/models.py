"""Network definitions for the Modified-Roofpedia Bern models.

SegNet(fusion=False): image-only baseline (Roofpedia AlbuNet: ResNet-50 encoder + U-Net decoder with
                      Conv-BN-ReLU blocks), 3 input bands (RGB) or 4 (RGB + near-infrared).
SegNet(fusion=True):  structure-aware dual-encoder network; a ResNet-18 encoder for
                      [nDSM, slope, sin(aspect), cos(aspect)] is fused with the image features at the
                      bottleneck by concatenation and two Conv-BN-ReLU blocks.
Output: 5 logits per pixel (0 background, 1 Green, 2 Potential Green, 3 Solar, 4 Not Suitable).
"""
import torch
import torch.nn as nn
from torchvision.models import resnet50, resnet18


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


class StructEncoder(nn.Module):
    def __init__(self, in_channels=4, out_channels=2048):
        super().__init__()
        b = resnet18(weights=None)
        self.stem = nn.Sequential(nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False), b.bn1, b.relu, b.maxpool)
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
    def __init__(self, num_classes=5, in_channels=3, fusion=False, struct_channels=4, num_filters=32):
        super().__init__()
        self.resnet = resnet50(weights=None)
        if in_channels != 3:
            c = self.resnet.conv1
            self.resnet.conv1 = nn.Conv2d(in_channels, c.out_channels, kernel_size=c.kernel_size, stride=c.stride, padding=c.padding, bias=False)
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


def load_model(weights_path, device='cpu'):
    """Load a released weight file; returns (model in eval mode, metadata dict)."""
    ck = torch.load(weights_path, map_location=device, weights_only=True)
    meta = ck['meta']
    net = SegNet(5, meta['in_channels'], meta['fusion'])
    net.load_state_dict(ck['state_dict'])
    return net.to(device).eval(), meta
