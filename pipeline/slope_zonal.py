"""Per-building roof-slope statistics for all swissTLM3D footprints.

Outputs CSV with: idx, UUID, n_px, mean_all (plain zonal mean), n_core, mean_core
(after removing the outer ring of pixels), and fractions of core roof pixels with
slope below 5/6.5/8/10/15 degrees.
"""
import os, sys, numpy as np, pandas as pd, shapefile, rasterio
from rasterio.features import rasterize
from rasterio.windows import Window

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import os, config as C
SHP = C.FOOTPRINTS
RAS = C.SLOPE
OUT = sys.argv[1] if len(sys.argv) > 1 else os.path.join(C.EVAL, 'slope_stats.csv')
THR = [5, 6.5, 8, 10, 15]
BLOCK = 2000

sf = shapefile.Reader(SHP)
n = len(sf)
uuids = [r[0] for r in sf.records()]
geoms = [s.__geo_interface__ for s in sf.shapes()]
bboxes = np.array([s.bbox for s in sf.shapes()])  # xmin ymin xmax ymax
print('buildings', n)

sum_all = np.zeros(n + 1); cnt_all = np.zeros(n + 1)
sum_core = np.zeros(n + 1); cnt_core = np.zeros(n + 1)
below = {t: np.zeros(n + 1) for t in THR}
below_all = {t: np.zeros(n + 1) for t in THR}

with rasterio.open(RAS) as src:
    T = src.transform; H, W = src.height, src.width; nod = src.nodata
    print('raster', W, H, T)
    for r0 in range(0, H, BLOCK):
        ra = max(r0 - 1, 0); rb = min(r0 + BLOCK + 1, H)
        win = Window(0, ra, W, rb - ra)
        wt = src.window_transform(win)
        ytop = T.f + T.e * ra; ybot = T.f + T.e * rb
        sel = np.where((bboxes[:, 3] >= ybot) & (bboxes[:, 1] <= ytop))[0]
        if len(sel) == 0:
            continue
        lab = rasterize(((geoms[i], int(i) + 1) for i in sel), out_shape=(rb - ra, W), transform=wt,
                        fill=0, dtype='int32', all_touched=False)
        sl = src.read(1, window=win)
        valid = np.isfinite(sl) & (sl != nod) & (sl >= 0)
        # core = pixel whose 4-neighbours carry the same label
        core = lab.copy()
        core[1:, :][lab[1:, :] != lab[:-1, :]] = 0
        core[:-1, :][lab[:-1, :] != lab[1:, :]] = 0
        core[:, 1:][lab[:, 1:] != lab[:, :-1]] = 0
        core[:, :-1][lab[:, :-1] != lab[:, 1:]] = 0
        # only count the interior rows of this block (exclude the 1-row margins)
        i0 = r0 - ra; i1 = i0 + min(BLOCK, H - r0)
        L = lab[i0:i1]; C = core[i0:i1]; S = sl[i0:i1]; V = valid[i0:i1]
        m = (L > 0) & V
        sum_all += np.bincount(L[m], weights=S[m], minlength=n + 1)
        cnt_all += np.bincount(L[m], minlength=n + 1)
        for t in THR:
            mm = m & (S < t)
            below_all[t] += np.bincount(L[mm], minlength=n + 1)
        mc = (C > 0) & V
        sum_core += np.bincount(C[mc], weights=S[mc], minlength=n + 1)
        cnt_core += np.bincount(C[mc], minlength=n + 1)
        for t in THR:
            mm = mc & (S < t)
            below[t] += np.bincount(C[mm], minlength=n + 1)
        print(f'rows {r0}/{H}  buildings in block {len(sel)}', flush=True)

with np.errstate(invalid='ignore', divide='ignore'):
    df = pd.DataFrame({'idx': np.arange(n), 'UUID': uuids,
                       'n_px': cnt_all[1:], 'mean_all': sum_all[1:] / cnt_all[1:],
                       'n_core': cnt_core[1:], 'mean_core': sum_core[1:] / cnt_core[1:]})
    for t in THR:
        df[f'fall_lt{t}'] = below_all[t][1:] / cnt_all[1:]
        df[f'fcore_lt{t}'] = below[t][1:] / cnt_core[1:]
df.to_csv(OUT, index=False)
print('done', OUT)
