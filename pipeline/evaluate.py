"""Pixel confusion matrices and per-building class counts for the six retrained configurations.

All labels and predictions share one coding (0 bg, 1 G, 2 PG, 3 S, 4 NS) and one tile set, so no
code harmonisation is needed. Outputs (eval/): tile_eval.json and tilebld_<key>.npz with
pred/lab per-building pixel counts, held (building touches evaluation tiles only), any (touches any tile).
"""
import os, sys, json, csv, numpy as np, pandas as pd, shapefile
from PIL import Image
from rasterio.features import rasterize
from rasterio.transform import from_bounds
from rasterio.warp import transform_geom
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config as C
EV = C.EVAL
RUNS = {'mapbox': 'mapbox_rgb_s1', 'f_mapbox': 'mapbox_fusion_s1', 'swissimage': 'swissimage_rgb_s1',
        'f_swissimage': 'swissimage_fusion_s1', 'rs': 'rs_rgb_s1', 'f_rs': 'rs_fusion_s1'}
if len(sys.argv) > 1: RUNS = {k: v.replace('_s1', '_' + sys.argv[1]) for k, v in RUNS.items()}
SPLITS = ['training', 'validation', 'evaluation']
Z = 19; T = 2 * 20037508.342789244 / 2 ** Z; O = 20037508.342789244

split = {(int(r['x']), int(r['y'])): r['split'] for r in csv.DictReader(open(os.path.join(C.DATA, 'split.csv')))}
sf = shapefile.Reader(C.FOOTPRINTS)
r = shapefile.Reader(dbf=open(C.REFERENCE_TABLE, 'rb'))
L = pd.DataFrame([list(x) for x in r.records()], columns=[f[0] for f in r.fields[1:]])
ref_map = dict(zip(L[C.ID_FIELD], pd.to_numeric(L[C.REF_FIELD], errors='coerce').fillna(0).astype(int)))
recs, shapes = sf.records(), sf.shapes()
lab_idx = [i for i, rec in enumerate(recs) if ref_map.get(rec[0], 0) > 0]
geoms, tile_index = {}, {}
for i in lab_idx:
    g = transform_geom(C.FOOTPRINT_CRS, 'EPSG:3857', shapes[i].__geo_interface__); geoms[i] = g
    xs, ys = [], []
    def walk(c):
        if isinstance(c[0], (float, int)): xs.append(c[0]); ys.append(c[1])
        else:
            for cc in c: walk(cc)
    walk(g['coordinates'])
    for tx in range(int((min(xs) + O) // T), int((max(xs) + O) // T) + 1):
        for ty in range(int((O - max(ys)) // T), int((O - min(ys)) // T) + 1):
            tile_index.setdefault((tx, ty), []).append(i)
bid = {i: k for k, i in enumerate(lab_idx)}
# rasterised footprints per tile are model-independent: cache them once
fp = {}
for t in split:
    if t in tile_index:
        x0 = t[0] * T - O; y1 = O - t[1] * T
        fp[t] = rasterize(((geoms[i], bid[i] + 1) for i in tile_index[t]), out_shape=(256, 256),
                          transform=from_bounds(x0, y1 - T, x0 + T, y1, 256, 256), fill=0, dtype='int32')
print('labelled buildings', len(lab_idx), 'tiles', len(split), 'tiles with footprints', len(fp), flush=True)
labels = {t: np.asarray(Image.open(f'{C.DATA}/labels/19/{t[0]}/{t[1]}.png')).astype(np.int64) for t in split}
res = {'runs': RUNS, 'models': {}}
for key, run in RUNS.items():
    cms = {sp: np.zeros((5, 5), np.int64) for sp in SPLITS}
    pc = np.zeros((len(lab_idx), 5), np.int64); lc = np.zeros((len(lab_idx), 5), np.int64)
    bt = {sp: np.zeros(len(lab_idx), bool) for sp in SPLITS}
    for t, sp in split.items():
        pred = np.asarray(Image.open(f'{C.RUNS}/{run}/pred_{sp}/19/{t[0]}/{t[1]}.png')).astype(np.int64)
        lab = labels[t]
        cms[sp] += np.bincount((lab * 5 + pred).ravel(), minlength=25).reshape(5, 5)
        if t in fp:
            ras = fp[t]; m = ras > 0
            if m.any():
                b = ras[m] - 1
                np.add.at(pc, (b, pred[m]), 1); np.add.at(lc, (b, lab[m]), 1)
                bt[sp][np.unique(b)] = True
    held = bt['evaluation'] & ~bt['training'] & ~bt['validation']
    anyt = bt['training'] | bt['validation'] | bt['evaluation']
    np.savez(f'{EV}/tilebld_{key}.npz', pred=pc, lab=lc, held=held, any=anyt, idx=np.array(lab_idx))
    res['models'][key] = {'cm': {sp: cms[sp].tolist() for sp in SPLITS}}
    e = cms['evaluation']; print(f'{key:13s} eval roof OA {np.trace(e[1:, 1:]) / e[1:, :].sum():.3f} held-out buildings {held.sum()}', flush=True)
json.dump(res, open(f'{EV}/tile_eval.json', 'w'))
print('done')
