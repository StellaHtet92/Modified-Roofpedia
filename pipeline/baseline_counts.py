import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config as C
"""Per-footprint pixel counts of class codes 0-4 for the reference raster and each configured baseline raster.
python baseline_counts.py          -> eval/counts_reference.npy, eval/counts_<baseline>.npy  (n_footprints x 6)
Rasters must share the grid of the reference label raster (as for the Bern 0.1 m products)."""
import numpy as np, shapefile, rasterio
from rasterio.features import rasterize
from rasterio.windows import Window
BLOCK = 512
sf = shapefile.Reader(C.FOOTPRINTS); shapes = sf.shapes(); n = len(shapes)
geoms = [s.__geo_interface__ for s in shapes]; bboxes = np.array([s.bbox for s in shapes])
with rasterio.open(C.LABEL_RASTER) as ref:
    H, W, T = ref.height, ref.width, ref.transform
for name, ras in [('reference', C.LABEL_RASTER)] + list(C.BASELINES.items()):
    counts = np.zeros((n + 1) * 6, dtype=np.int64)
    with rasterio.open(ras) as src:
        for r0 in range(0, H, BLOCK):
            rb = min(r0 + BLOCK, H); ytop = T.f + T.e * r0; ybot = T.f + T.e * rb
            sel = np.where((bboxes[:, 3] >= ybot) & (bboxes[:, 1] <= ytop))[0]
            if len(sel) == 0: continue
            win = Window(0, r0, W, rb - r0)
            lab = rasterize(((geoms[i], int(i) + 1) for i in sel), out_shape=(rb - r0, W), transform=rasterio.windows.transform(win, T), fill=0, dtype='int32')
            m = lab > 0
            if not m.any(): continue
            v = src.read(1, window=win)[m].astype(np.int64); v[v > 4] = 5
            counts += np.bincount(lab[m].astype(np.int64) * 6 + v, minlength=(n + 1) * 6)
    np.save(os.path.join(C.EVAL, f'counts_{name}.npy'), counts.reshape(n + 1, 6)[1:]); print('done', name)
