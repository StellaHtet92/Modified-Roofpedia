"""Build structural input tiles (nDSM, slope, sin/cos aspect) for every image tile, as used to train the fusion models.

python -m roofpedia_bern.make_struct --tiles my_tiles --ndsm ndsm.tif --slope slope.tif --aspect aspect.tif --out my_struct

Inputs are single-band rasters in any projected CRS that cover the tiles:
  ndsm   : normalized DSM = DSM - DTM, in metres (height of objects above ground)
  slope  : slope of the DSM in degrees      (gdaldem slope  dsm.tif slope.tif  -alg Horn -compute_edges)
  aspect : aspect of the DSM in degrees     (gdaldem aspect dsm.tif aspect.tif -alg Horn -zero_for_flat -compute_edges)
For each tile <z>/<x>/<y>, the three rasters are resampled bilinearly to the tile grid (Web Mercator, same size as the
image tile) and written to <out>/<z>/<x>/<y>.npy as float32 (4, H, W): [nDSM, slope, sin(aspect), cos(aspect)].
Aspect is set to the zero vector where slope < 6.5 deg (flat surfaces have no orientation), exactly as in training.
No-data is written as 0. Normalization is applied later by predict.py.
"""
import argparse, os, numpy as np, rasterio, mercantile
from rasterio.windows import from_bounds as win_from_bounds
from rasterio.warp import reproject, Resampling, transform_bounds
from rasterio.transform import from_bounds
from .predict import list_tiles

FLAT_SLOPE_DEG = 6.5


def warp(ds, z, x, y, h, w, margin_px=4):
    b = mercantile.xy_bounds(mercantile.Tile(x, y, z))
    l, bo, r, t = transform_bounds('EPSG:3857', ds.crs, b.left, b.bottom, b.right, b.top, densify_pts=21)
    m = margin_px * max(abs(ds.res[0]), abs(ds.res[1]))
    win = win_from_bounds(l - m, bo - m, r + m, t + m, ds.transform).round_offsets().round_lengths()
    src = ds.read(1, window=win, boundless=True, fill_value=ds.nodata if ds.nodata is not None else 0).astype(np.float32)
    out = np.full((h, w), np.nan, np.float32)
    reproject(src, out, src_transform=ds.window_transform(win), src_crs=ds.crs, src_nodata=ds.nodata,
              dst_transform=from_bounds(b.left, b.bottom, b.right, b.top, w, h), dst_crs='EPSG:3857', dst_nodata=np.nan,
              resampling=Resampling.bilinear)
    return np.nan_to_num(out, nan=0.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tiles', required=True); ap.add_argument('--ndsm', required=True); ap.add_argument('--slope', required=True)
    ap.add_argument('--aspect', required=True); ap.add_argument('--out', required=True)
    a = ap.parse_args()
    tiles = list_tiles(a.tiles)
    with rasterio.open(a.ndsm) as dn, rasterio.open(a.slope) as dsl, rasterio.open(a.aspect) as das:
        for k, (z, x, y, f) in enumerate(tiles):
            o = os.path.join(a.out, str(z), str(x), f'{y}.npy')
            if os.path.exists(o): continue
            if f.endswith('.npy'):
                h, w = np.load(f, mmap_mode='r').shape[1:]
            else:
                from PIL import Image
                w, h = Image.open(f).size
            nd, sl, asp = warp(dn, z, x, y, h, w), warp(dsl, z, x, y, h, w), warp(das, z, x, y, h, w)
            rad = np.deg2rad(asp); s_, c_ = np.sin(rad), np.cos(rad)
            s_[sl < FLAT_SLOPE_DEG] = 0; c_[sl < FLAT_SLOPE_DEG] = 0
            os.makedirs(os.path.dirname(o), exist_ok=True)
            np.save(o, np.stack([nd, sl, s_, c_]).astype(np.float32))
            if (k + 1) % 500 == 0: print(k + 1, '/', len(tiles))
    print('done', len(tiles))


if __name__ == '__main__':
    main()
