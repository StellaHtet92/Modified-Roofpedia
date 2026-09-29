"""Multimodal slippy-map dataset: RGB (PIL) + structural (.npy float) + mask (PIL).

Add this class to src/datasets.py (or import it from here). It returns the RGB image,
the 2-channel structural patch (nDSM + slope), and the label mask as THREE separate
tensors, all spatially aligned and augmented in sync.

Directory layout expected (standard slippy map, matching filenames across all three):

    <root>/training/images/<z>/<x>/<y>.png       RGB
    <root>/training/struct/<z>/<x>/<y>.npy        (2, H, W) float32  (nDSM, slope)
    <root>/training/labels/<z>/<x>/<y>.png        label (mode "P")

The .npy struct patches are produced by extract_struct_patches.py.
"""

import os
import numpy as np
import torch
import torch.utils.data
from PIL import Image

from src.tiles import tiles_from_slippy_map


class SlippyMapMultiModal(torch.utils.data.Dataset):
    """RGB + structural + mask dataset in slippy-map format.

    Args:
        image_root:  directory of RGB tiles   (<z>/<x>/<y>.png)
        struct_root: directory of struct tiles (<z>/<x>/<y>.npy, shape (2,H,W) float32)
        target_root: directory of label tiles  (<z>/<x>/<y>.png)
        joint_transform: a callable(rgb_pil, struct_tensor, mask_pil) ->
                         (rgb_tensor, struct_tensor, mask_tensor). See transforms_multimodal.
    """

    def __init__(self, image_root, struct_root, target_root, joint_transform=None):
        super().__init__()
        self.joint_transform = joint_transform

        # Build tile -> path lookups for each modality, then intersect by tile key.
        self.rgb = {tile: path for tile, path in tiles_from_slippy_map(image_root)}
        self.target = {tile: path for tile, path in tiles_from_slippy_map(target_root)}

        # struct tiles are .npy, so tiles_from_slippy_map (which globs images) may not
        # find them. Discover them directly.
        self.struct = self._discover_npy(struct_root)

        # Keep only tiles present in all three modalities.
        common = set(self.rgb) & set(self.struct) & set(self.target)
        assert len(common) > 0, "no tiles common to images, struct, and labels"

        self.tiles = sorted(common)

        # Warn (not fail) if some tiles are missing a modality.
        missing_struct = set(self.rgb) - set(self.struct)
        if missing_struct:
            print(f"[SlippyMapMultiModal] warning: {len(missing_struct)} RGB tiles have no struct patch and are skipped.")

    @staticmethod
    def _discover_npy(root):
        """Return {tile_key: path} for <root>/<z>/<x>/<y>.npy.

        tile_key matches the (x, y, z) namedtuple produced by tiles_from_slippy_map,
        so we reuse the same Tile type by importing it lazily.
        """
        from src.tiles import tiles_from_slippy_map  # noqa
        # tiles_from_slippy_map globs *.png; replicate its Tile object for .npy files.
        # We reconstruct tiles from the folder structure z/x/y.npy.
        import mercantile  # Tile namedtuple compatible with mercantile.Tile
        out = {}
        if not os.path.isdir(root):
            raise FileNotFoundError(f"struct_root not found: {root}")
        for z in os.listdir(root):
            zp = os.path.join(root, z)
            if not os.path.isdir(zp):
                continue
            for x in os.listdir(zp):
                xp = os.path.join(zp, x)
                if not os.path.isdir(xp):
                    continue
                for fn in os.listdir(xp):
                    if not fn.endswith(".npy"):
                        continue
                    y = os.path.splitext(fn)[0]
                    tile = mercantile.Tile(x=int(x), y=int(y), z=int(z))
                    out[tile] = os.path.join(xp, fn)
        return out

    def __len__(self):
        return len(self.tiles)

    def __getitem__(self, i):
        tile = self.tiles[i]

        rgb = Image.open(self.rgb[tile])                       # PIL, RGB
        mask = Image.open(self.target[tile])                  # PIL, mode P
        struct = np.load(self.struct[tile]).astype(np.float32)  # (2, H, W)
        struct = torch.from_numpy(struct)                      # tensor (2, H, W)

        if self.joint_transform is not None:
            rgb, struct, mask = self.joint_transform(rgb, struct, mask)

        # Return tile as a plain tuple of ints for batching safety.
        tile_id = torch.IntTensor([tile.x, tile.y, tile.z])
        return rgb, struct, mask, tile_id
