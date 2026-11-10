"""City-scale per-footprint class counts from the Fusion SWISSIMAGE predicted mask tiles (all footprints)."""
import glob, os, numpy as np, shapefile, sys
from PIL import Image
from rasterio.features import rasterize
from rasterio.transform import from_bounds
from rasterio.warp import transform_geom
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config as C
S = C.EVAL + os.sep
RUN, KEY = sys.argv[1], sys.argv[2]
PMAP = np.arange(5)
Z = 19; T = 2 * 20037508.342789244 / 2 ** Z; O = 20037508.342789244
sf = shapefile.Reader(C.FOOTPRINTS)
shapes = sf.shapes(); n = len(shapes)
geoms = []; tile_index = {}
for i, s in enumerate(shapes):
    g = transform_geom(C.FOOTPRINT_CRS, 'EPSG:3857', s.__geo_interface__)
    geoms.append(g)
    xs = []; ys = []
    def walk(c):
        if isinstance(c[0], (float, int)): xs.append(c[0]); ys.append(c[1])
        else:
            for cc in c: walk(cc)
    walk(g['coordinates'])
    for tx in range(int((min(xs) + O) // T), int((max(xs) + O) // T) + 1):
        for ty in range(int((O - max(ys)) // T), int((O - min(ys)) // T) + 1):
            tile_index.setdefault((tx, ty), []).append(i)
print('footprints', n, 'tiles touched', len(tile_index), flush=True)
cnt = np.zeros((n, 5), np.int64); covered = np.zeros((n, 1), np.int64)
files = glob.glob(f'{C.RUNS}/{RUN}/pred_city/19/*/*.png')
for j, f in enumerate(files):
    p = f.replace(os.sep, '/').split('/'); t = (int(p[-2]), int(p[-1][:-4]))
    if t not in tile_index: continue
    pred = np.asarray(Image.open(f)).astype(np.int64); pred[pred > 4] = 0
    pred = PMAP[pred]
    x0 = t[0] * T - O; y1 = O - t[1] * T
    ids = tile_index[t]
    ras = rasterize(((geoms[i], i + 1) for i in ids), out_shape=(256, 256), transform=from_bounds(x0, y1 - T, x0 + T, y1, 256, 256), fill=0, dtype='int32')
    m = ras > 0
    if m.any():
        np.add.at(cnt, (ras[m] - 1, pred[m]), 1)
    if j % 10000 == 0: print(j, len(files), flush=True)
np.save(S + f'city_{KEY}.npy', cnt)
print('done')

