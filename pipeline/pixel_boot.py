"""Spatial block bootstrap of pixel-level differences (evaluation split, blocks of 8x8 tiles)."""
import csv, json, numpy as np
from PIL import Image
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config as C
CR = C.WORK
RUNS = {'mapbox': 'mapbox_rgb_s1', 'f_mapbox': 'mapbox_fusion_s1', 'swissimage': 'swissimage_rgb_s1',
        'f_swissimage': 'swissimage_fusion_s1', 'rs': 'rs_rgb_s1', 'f_rs': 'rs_fusion_s1'}
rows = [r for r in csv.DictReader(open(f'{CR}/data/split.csv')) if r['split'] == 'evaluation']
blocks = sorted(set(r['block'] for r in rows)); bidx = {b: i for i, b in enumerate(blocks)}
CM = {k: np.zeros((len(blocks), 5, 5), np.int64) for k in RUNS}
for r in rows:
    x, y = r['x'], r['y']; b = bidx[r['block']]
    lab = np.asarray(Image.open(f'{CR}/data/labels/19/{x}/{y}.png')).astype(np.int64)
    for k, run in RUNS.items():
        p = np.asarray(Image.open(f'{CR}/runs/{run}/pred_evaluation/19/{x}/{y}.png')).astype(np.int64)
        CM[k][b] += np.bincount((lab * 5 + p).ravel(), minlength=25).reshape(5, 5)

def met(cm):
    TP = np.diag(cm).astype(float); iou = TP / np.maximum(cm.sum(0) + cm.sum(1) - TP, 1)
    tot = cm.sum(); po = TP.sum() / tot; pe = (cm.sum(1).astype(float) * cm.sum(0)).sum() / float(tot) ** 2
    return {'kappa': (po - pe) / (1 - pe), 'miou4': iou[1:].mean(), 'oa': TP[1:].sum() / cm[1:].sum(), **{f'iou{c}': iou[c] for c in range(1, 5)}}

rng = np.random.default_rng(0); B = 2000; nb = len(blocks)
samp = [rng.integers(0, nb, nb) for _ in range(B)]
pairs = [('mapbox', 'f_mapbox'), ('swissimage', 'f_swissimage'), ('rs', 'f_rs'), ('swissimage', 'rs'), ('f_swissimage', 'f_rs'), ('swissimage', 'mapbox'), ('f_swissimage', 'f_mapbox')]
out = {'n_blocks': nb, 'n_tiles': len(rows)}
for a, c in pairs:
    base = {m: met(CM[c].sum(0))[m] - met(CM[a].sum(0))[m] for m in ['kappa', 'miou4', 'oa', 'iou1', 'iou2', 'iou3', 'iou4']}
    bs = {m: [] for m in base}
    for s in samp:
        ma, mc = met(CM[a][s].sum(0)), met(CM[c][s].sum(0))
        for m in base: bs[m].append(mc[m] - ma[m])
    out[f'{c}-{a}'] = {m: dict(diff=base[m], lo=float(np.percentile(bs[m], 2.5)), hi=float(np.percentile(bs[m], 97.5))) for m in base}
    print(f'{c} - {a}: ' + '  '.join(f'{m} {base[m]:+.3f} [{np.percentile(bs[m], 2.5):+.3f},{np.percentile(bs[m], 97.5):+.3f}]' for m in ['kappa', 'miou4', 'oa']))
json.dump(out, open(f'{CR}/eval/pixel_boot.json', 'w'), indent=1, default=float)
print('blocks', nb)
