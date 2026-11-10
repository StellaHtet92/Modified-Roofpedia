"""Render SWISSIMAGE RS (Bern_SOP GeoTIFFs, EPSG:2056, 0.1 m, 4 x uint8) to z19 XYZ tiles.

The GeoTIFFs tag band 4 as *alpha*, so GIS tools (QGIS XYZ export) treat it as transparency.
Here every band is read as plain data from numpy windows, mosaicked on the native 0.1 m grid
and warped to the Web-Mercator tile grid, so no band is ever interpreted as a mask.
"""
import glob, os, numpy as np, rasterio, mercantile
from rasterio.windows import from_bounds as win_from_bounds
from rasterio.warp import reproject, Resampling, transform_bounds
from rasterio.transform import from_bounds, Affine

import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config as C
SOP = C.RS_DIR


class RSRenderer:
    def __init__(self, folder=SOP):
        self.files = sorted(glob.glob(os.path.join(folder, '*.tif')))
        self.ds = [rasterio.open(f) for f in self.files]
        self.bounds = np.array([list(d.bounds) for d in self.ds])

    def tile(self, x, y, z=19, size=256, margin=2.0):
        """Return (4, size, size) uint8 array in source band order and a validity fraction."""
        b = mercantile.xy_bounds(mercantile.Tile(x, y, z))
        l, bo, r, t = transform_bounds('EPSG:3857', C.RS_CRS, b.left, b.bottom, b.right, b.top, densify_pts=21)
        l, bo, r, t = l - margin, bo - margin, r + margin, t + margin
        res = 0.1
        l, t = np.floor(l / res) * res, np.ceil(t / res) * res
        W, H = int(np.ceil((r - l) / res)), int(np.ceil((t - bo) / res))
        mos = np.zeros((4, H, W), np.uint8); valid = np.zeros((H, W), bool)
        hit = np.where((self.bounds[:, 0] < r) & (self.bounds[:, 2] > l) & (self.bounds[:, 1] < t) & (self.bounds[:, 3] > bo))[0]
        for i in hit:
            d = self.ds[i]
            il, ib, ir, it = max(l, d.bounds.left), max(bo, d.bounds.bottom), min(r, d.bounds.right), min(t, d.bounds.top)
            if ir <= il or it <= ib: continue
            w = win_from_bounds(il, ib, ir, it, d.transform).round_offsets().round_lengths()
            a = d.read(window=w, boundless=False)
            wt = d.window_transform(w)
            c0 = int(round((wt.c - l) / res)); r0 = int(round((t - wt.f) / res))
            hh, ww = a.shape[1:]
            c1, r1 = min(c0 + ww, W), min(r0 + hh, H)
            c0c, r0c = max(c0, 0), max(r0, 0)
            sub = a[:, r0c - r0:r1 - r0, c0c - c0:c1 - c0]
            # a pixel is data if any band is non-zero (no-data margins in the delivery are all-zero)
            m = sub.any(0)
            for k in range(4):
                mos[k, r0c:r1, c0c:c1][m] = sub[k][m]
            valid[r0c:r1, c0c:c1] |= m
        src_t = Affine(res, 0, l, 0, -res, t)
        dst_t = from_bounds(b.left, b.bottom, b.right, b.top, size, size)
        out = np.zeros((4, size, size), np.float32)
        for k in range(4):
            reproject(mos[k].astype(np.float32), out[k], src_transform=src_t, src_crs=C.RS_CRS,
                      dst_transform=dst_t, dst_crs='EPSG:3857', resampling=Resampling.bilinear)
        v = np.zeros((size, size), np.float32)
        reproject(valid.astype(np.float32), v, src_transform=src_t, src_crs=C.RS_CRS,
                  dst_transform=dst_t, dst_crs='EPSG:3857', resampling=Resampling.nearest)
        return np.clip(np.round(out), 0, 255).astype(np.uint8), float((v > 0.5).mean())
