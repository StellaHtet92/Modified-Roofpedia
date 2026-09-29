"""Prediction for the dual-encoder FusionUNet (RGB + structural channels).

Drop-in counterpart to src/predict.py: same buffering/unbuffering strategy and mask
output format, but reads BOTH the RGB tile and its matching struct/<z>/<x>/<y>.npy
patch and feeds them to FusionUNet as net(rgb, struct).

Directory layout expected:
    tiles_dir/<z>/<x>/<y>.png     RGB tiles (as produced for src/predict.py)
    struct_dir/<z>/<x>/<y>.npy    (4, H, W) float32 struct patches, channel order
                                   [nDSM, slope, aspect_sin, aspect_cos] -- see
                                   built_stuct_for_splits_four.py / src/struct_extract.py
"""

import os

import mercantile
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, Normalize

from tqdm import tqdm
from PIL import Image

from src.tiles import tiles_from_slippy_map, buffer_tile_image
from src.unet_fusion import FusionUNet
from src.transforms import ConvertImageMode, ImageToTensor
from src.colors import make_palette
from src.train_multimodal import compute_struct_stats


def _discover_struct_tiles(struct_root):
    """Return {mercantile.Tile: path} for struct_root/<z>/<x>/<y>.npy.

    Mirrors SlippyMapMultiModal._discover_npy in src/datasets_multimodal.py, since
    tiles_from_slippy_map only globs image formats PIL can open.
    """
    out = {}
    if not os.path.isdir(struct_root):
        raise FileNotFoundError(f"struct_dir not found: {struct_root}")
    for z in os.listdir(struct_root):
        zp = os.path.join(struct_root, z)
        if not (z.isdigit() and os.path.isdir(zp)):
            continue
        for x in os.listdir(zp):
            xp = os.path.join(zp, x)
            if not (x.isdigit() and os.path.isdir(xp)):
                continue
            for fn in os.listdir(xp):
                if not fn.endswith(".npy"):
                    continue
                y = os.path.splitext(fn)[0]
                if not y.isdigit():
                    continue
                out[mercantile.Tile(x=int(x), y=int(y), z=int(z))] = os.path.join(xp, fn)
    return out


def _buffer_tile_struct(tile, struct_tiles, overlap, tile_size, channels):
    """Struct-array equivalent of src.tiles.buffer_tile_image.

    Pads with zeros where a neighbouring tile's struct patch is unavailable (matches
    buffer_tile_image's `nodata` behaviour for missing neighbouring RGB tiles).
    """
    x, y, z = map(int, [tile.x, tile.y, tile.z])
    composite_size = tile_size + 2 * overlap
    composite = np.zeros((channels, composite_size, composite_size), dtype=np.float32)

    def load(dx, dy):
        other = mercantile.Tile(x=x + dx, y=y + dy, z=z)
        path = struct_tiles.get(other)
        return np.load(path).astype(np.float32) if path is not None else None

    center = load(0, 0)
    if center is None:
        raise KeyError(f"no struct patch for tile {tile}")
    composite[:, overlap:overlap + tile_size, overlap:overlap + tile_size] = center

    def paste(arr, dst_box, src_box):
        if arr is None:
            return
        dx0, dy0, dx1, dy1 = dst_box
        sx0, sy0, sx1, sy1 = src_box
        composite[:, dy0:dy1, dx0:dx1] = arr[:, sy0:sy1, sx0:sx1]

    o, t = overlap, tile_size
    paste(load(-1, -1), (0, 0, o, o), (t - o, t - o, t, t))
    paste(load(+1, -1), (t + o, 0, composite_size, o), (0, t - o, o, t))
    paste(load(-1, +1), (0, t + o, o, composite_size), (t - o, 0, t, o))
    paste(load(+1, +1), (t + o, t + o, composite_size, composite_size), (0, 0, o, o))
    paste(load(0, -1), (o, 0, t + o, o), (0, t - o, t, t))
    paste(load(-1, 0), (0, o, o, t + o), (t - o, 0, t, t))
    paste(load(0, +1), (o, t + o, t + o, composite_size), (0, 0, t, o))
    paste(load(+1, 0), (t + o, o, composite_size, t + o), (0, 0, o, t))

    return composite


class BufferedSlippyMapMultiModalDirectory(torch.utils.data.Dataset):
    """RGB + structural dual-modality buffered slippy-map dataset for prediction.

    Same buffer/unbuffer strategy as src.datasets.BufferedSlippyMapDirectory, applied
    identically to both modalities so they stay pixel-aligned inside the network.
    """

    def __init__(self, image_root, struct_root, rgb_transform, struct_mean, struct_std,
                 size=256, overlap=32, struct_channels=4):
        super().__init__()

        assert overlap >= 0
        assert size >= 256

        self.rgb_transform = rgb_transform
        self.size = size
        self.overlap = overlap
        self.struct_channels = struct_channels
        self.struct_mean = np.asarray(struct_mean, dtype=np.float32).reshape(-1, 1, 1)
        self.struct_std = np.asarray(struct_std, dtype=np.float32).reshape(-1, 1, 1)

        self.rgb_tiles = list(tiles_from_slippy_map(image_root))
        struct_tiles = _discover_struct_tiles(struct_root)

        self.tiles = [(tile, path) for tile, path in self.rgb_tiles if tile in struct_tiles]
        missing = len(self.rgb_tiles) - len(self.tiles)
        if missing:
            print(f"[predict_multimodal] warning: {missing} RGB tiles have no struct "
                  f"patch and will be skipped. Run struct extraction for this city first.")

        self.struct_tiles = struct_tiles

    def __len__(self):
        return len(self.tiles)

    def __getitem__(self, i):
        tile, _path = self.tiles[i]

        rgb = buffer_tile_image(tile, self.rgb_tiles, overlap=self.overlap, tile_size=self.size)
        rgb = self.rgb_transform(rgb)

        struct = _buffer_tile_struct(tile, self.struct_tiles, self.overlap, self.size, self.struct_channels)
        struct = (struct - self.struct_mean) / (self.struct_std + 1e-6)
        struct = torch.from_numpy(struct)

        return rgb, struct, torch.IntTensor([tile.x, tile.y, tile.z])

    def unbuffer(self, probs):
        """Removes the overlap borders added around the original tile. See
        src.datasets.BufferedSlippyMapDirectory.unbuffer."""
        o = self.overlap
        _, x, y = probs.shape
        return probs[:, o:x - o, o:y - o]


def predict(tiles_dir, struct_dir, mask_dir, tile_size, device, chkpt,
            num_classes=5, struct_channels=4, struct_mean=None, struct_std=None,
            training_struct_root=None, overlap=32):
    """Run the dual-encoder FusionUNet over a city's RGB + struct tiles and save masks.

    Args:
        tiles_dir:  RGB tile root, results/02Images/<city>.
        struct_dir: struct tile root, results/05Struct/<city> (see src.struct_extract).
        mask_dir:   where predicted masks are written, results/03Masks/<type>/<city>.
        struct_mean/struct_std: per-channel normalisation stats. If omitted, they are
            recomputed from training_struct_root with the same procedure (and RNG seed)
            used at training time in src.train_multimodal.compute_struct_stats, so they
            match what the checkpoint was actually trained on.
    """
    net = FusionUNet(num_classes=num_classes, struct_channels=struct_channels).to(device)
    net = nn.DataParallel(net)
    net.load_state_dict(chkpt["state_dict"])
    net.eval()

    if struct_mean is None or struct_std is None:
        if not training_struct_root:
            raise ValueError(
                "predict_multimodal.predict: pass struct_mean/struct_std, or "
                "training_struct_root to derive them (must match training)."
            )
        struct_mean, struct_std = compute_struct_stats(training_struct_root)
        print(f"[struct stats] mean={struct_mean} std={struct_std}")

    mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    rgb_transform = Compose([ConvertImageMode(mode="RGB"), ImageToTensor(), Normalize(mean=mean, std=std)])

    directory = BufferedSlippyMapMultiModalDirectory(
        tiles_dir, struct_dir, rgb_transform, struct_mean, struct_std,
        size=tile_size, overlap=overlap, struct_channels=struct_channels,
    )
    assert len(directory) > 0, "at least one tile with both RGB and struct data"

    loader = DataLoader(directory, batch_size=1)

    with torch.no_grad():
        for rgb, struct, tiles in tqdm(loader, desc="Eval", unit="batch", ascii=True):
            rgb = rgb.to(device)
            struct = struct.to(device)
            outputs = net(rgb, struct)

            probs = nn.functional.softmax(outputs, dim=1).data.cpu().numpy()

            for tile, prob in zip(tiles, probs):
                x, y, z = list(map(int, tile))

                prob = directory.unbuffer(prob)
                mask = np.argmax(prob, axis=0)  # class indices: 0..num_classes-1
                mask = mask.astype(np.uint8)

                palette = make_palette("light", "green", "yellow", "dark", "red")
                out = Image.fromarray(mask, mode="P")
                out.putpalette(palette)

                os.makedirs(os.path.join(mask_dir, str(z), str(x)), exist_ok=True)
                path = os.path.join(mask_dir, str(z), str(x), str(y) + ".png")
                out.save(path, optimize=True)

    print("Prediction Done, saved masks to " + mask_dir)
    print(f"Masks contain class indices: 0 (background), 1-{num_classes - 1} (classes)")
