"""Build one consistent z19 tile dataset for all imagery sources and models.

For every z19 tile of the study area (the 67,519 SWISSIMAGE/Mapbox city tiles):
  labels/19/x/y.png   uint8 class codes rendered directly from the reference raster
                      (0 = background, 1 = Green, 2 = Potential Green, 3 = Solar, 4 = Not Suitable)
Kept tiles (labelled AND fully covered by all three sources) additionally get
  rs/19/x/y.npy       uint8 (4, 256, 256): R, G, B, NIR from the Bern_SOP GeoTIFFs
  struct/19/x/y.npy   float32 (4, 256, 256): nDSM, slope, sin(aspect), cos(aspect) (aspect = 0 where slope < 6.5 deg)
  swissimage/19/x/y.png, mapbox/19/x/y.png   RGB copies of the existing tiles
and are listed in tiles.csv.
"""
import os, sys, csv, glob
os.environ['GDAL_CACHEMAX'] = '64'
import numpy as np, rasterio, mercantile
from multiprocessing import Pool
from PIL import Image
from rasterio.windows import from_bounds as win_from_bounds
from rasterio.warp import reproject, Resampling, transform_bounds
from rasterio.transform import from_bounds, Affine
sys.path.insert(0, os.path.dirname(__file__))
from rs_render import RSRenderer

import config as C
OUT = C.DATA
LABEL = C.LABEL_RASTER
NDSM, SLOPE, ASPECT = C.NDSM, C.SLOPE, C.ASPECT
SWISS = C.SWISS_TILES
MAPBOX = C.MAPBOX_TILES
Z, SIZE, FLAT = 19, 256, 6.5

G = {}
def init():
    G['lab'] = rasterio.open(LABEL); G['ndsm'] = rasterio.open(NDSM); G['slope'] = rasterio.open(SLOPE); G['aspect'] = rasterio.open(ASPECT)
    G['rs'] = RSRenderer() if C.RS_DIR else None


class Strip:
    """A full-width band of raster rows held in memory (the rasters are stored as 1-row strips)."""
    def __init__(self, ds, l, bo, r, t, margin=3.0):
        w = win_from_bounds(max(l - margin, ds.bounds.left), max(bo - margin, ds.bounds.bottom),
                            min(r + margin, ds.bounds.right), min(t + margin, ds.bounds.top), ds.transform).round_offsets().round_lengths()
        self.a = ds.read(1, window=w); self.tr = ds.window_transform(w); self.crs = ds.crs; self.nodata = ds.nodata

    def warp(self, x, y, resampling, fill):
        b = mercantile.xy_bounds(mercantile.Tile(x, y, Z))
        out = np.full((SIZE, SIZE), fill, np.float32 if self.a.dtype.kind == 'f' else self.a.dtype)
        reproject(self.a, out, src_transform=self.tr, src_crs=self.crs, src_nodata=self.nodata,
                  dst_transform=from_bounds(b.left, b.bottom, b.right, b.top, SIZE, SIZE), dst_crs='EPSG:3857',
                  dst_nodata=fill, resampling=resampling)
        return out


def row_bounds(y, xs):
    b0 = mercantile.xy_bounds(mercantile.Tile(min(xs), y, Z)); b1 = mercantile.xy_bounds(mercantile.Tile(max(xs), y, Z))
    # label, nDSM, slope and aspect rasters must share one projected CRS (that of the label raster)
    return transform_bounds('EPSG:3857', G['lab'].crs, b0.left, b1.bottom, b1.right, b0.top, densify_pts=51)


def work_row(arg):
    y, xs = arg
    rb = row_bounds(y, xs)
    lab_s = Strip(G['lab'], *rb)
    rows, keep_xs = [], []
    for x in xs:
        lab = lab_s.warp(x, y, Resampling.nearest, 0).astype(np.uint8); lab[lab > 4] = 0
        if not lab.any(): continue
        p = f'{OUT}/labels/{Z}/{x}'; os.makedirs(p, exist_ok=True); Image.fromarray(lab).save(f'{p}/{y}.png')
        counts = np.bincount(lab.ravel(), minlength=5).tolist()
        rgb_sources = [('swissimage', SWISS)] + ([('mapbox', MAPBOX)] if MAPBOX else [])
        have = all(os.path.exists(f'{src}/{x}/{y}.png') for _, src in rgb_sources)
        if have:   # reject tiles with black no-data areas (e.g. at the edge of an orthophoto mosaic)
            have = all((np.asarray(Image.open(f'{src}/{x}/{y}.png').convert('RGB')).max(-1) <= 3).mean() <= 0.01 for _, src in rgb_sources)
        if G['rs'] is not None:
            rs, valid = G['rs'].tile(x, y)
        else:
            rs, valid = None, 1.0
        # a tile is kept only if it is completely covered by every configured imagery source
        keep = have and valid >= 0.999
        if keep:
            if rs is not None:
                p = f'{OUT}/rs/{Z}/{x}'; os.makedirs(p, exist_ok=True)
                np.save(f'{p}/{y}.npy', np.stack([rs[1], rs[2], rs[3], rs[0]]))   # NIR,R,G,B -> R,G,B,NIR
            for name, src in rgb_sources:
                p = f'{OUT}/{name}/{Z}/{x}'; os.makedirs(p, exist_ok=True)
                Image.open(f'{src}/{x}/{y}.png').convert('RGB').save(f'{p}/{y}.png')
            keep_xs.append(x)
        rows.append([x, y, int(keep), round(valid, 4), int(have), *counts])
    del lab_s
    if keep_xs:
        st = {k: Strip(G[k], *row_bounds(y, keep_xs)) for k in ['ndsm', 'slope', 'aspect']}
        for x in keep_xs:
            nd, sl, asp = [np.nan_to_num(st[k].warp(x, y, Resampling.bilinear, np.nan), nan=0.0) for k in ['ndsm', 'slope', 'aspect']]
            rad = np.deg2rad(asp); s_, c_ = np.sin(rad), np.cos(rad); s_[sl < FLAT] = 0; c_[sl < FLAT] = 0
            p = f'{OUT}/struct/{Z}/{x}'; os.makedirs(p, exist_ok=True)
            np.save(f'{p}/{y}.npy', np.stack([nd, sl, s_, c_]).astype(np.float32))
    return rows

if __name__ == '__main__':
    tiles = sorted((int(os.path.basename(os.path.dirname(f))), int(os.path.splitext(os.path.basename(f))[0]))
                   for f in glob.glob(SWISS + '/*/*.png'))
    print('city tiles', len(tiles), flush=True)
    from collections import defaultdict
    byrow = defaultdict(list)
    for x, y in tiles: byrow[y].append(x)
    jobs = sorted(byrow.items())
    rows = []
    with Pool(6, initializer=init) as pool:
        for i, r in enumerate(pool.imap_unordered(work_row, jobs, chunksize=1)):
            rows += r
            if (i + 1) % 20 == 0: print(i + 1, '/', len(jobs), 'tile rows,', len(rows), 'labelled', flush=True)
    rows.sort()
    with open(OUT + '/tiles_all_labelled.csv', 'w', newline='') as f:
        w = csv.writer(f); w.writerow(['x', 'y', 'keep', 'rs_valid', 'rgb_sources', 'n_bg', 'n_g', 'n_pg', 'n_s', 'n_ns']); w.writerows(rows)
    k = [r for r in rows if r[2]]
    print('labelled', len(rows), 'kept', len(k), flush=True)
