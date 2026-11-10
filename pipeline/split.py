"""Spatially blocked 70/15/15 split shared by all configurations, plus normalisation statistics."""
import csv, json, os, numpy as np
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config as C
DATA = C.DATA
B = 8                                                       # block = 8 x 8 z19 tiles (~420 m)
rows = [r for r in csv.DictReader(open(f'{DATA}/tiles_all_labelled.csv')) if r['keep'] == '1']
xy = np.array([[int(r['x']), int(r['y'])] for r in rows]); cnt = np.array([[int(r[k]) for k in ['n_g', 'n_pg', 'n_s', 'n_ns']] for r in rows], float)
blk = (xy[:, 0] // B) * 100000 + xy[:, 1] // B
ub, inv = np.unique(blk, return_inverse=True)
tot = cnt.sum(0) / cnt.sum()
best = None
for seed in range(500):
    rng = np.random.default_rng(seed); order = rng.permutation(len(ub))
    ntile = np.bincount(inv, minlength=len(ub)); cum = np.cumsum(ntile[order]) / len(xy)
    lab = np.empty(len(ub), int); lab[order] = np.where(cum <= 0.70, 0, np.where(cum <= 0.85, 1, 2))
    s = lab[inv]
    frac = np.array([(s == k).mean() for k in range(3)])
    dev = max(np.abs(cnt[s == k].sum(0) / cnt[s == k].sum() - tot).max() for k in range(3))
    if np.all(np.abs(frac - [0.7, 0.15, 0.15]) < 0.015) and (best is None or dev < best[0]):
        best = (dev, seed, s)
dev, seed, s = best
names = ['training', 'validation', 'evaluation']
with open(f'{DATA}/split.csv', 'w', newline='') as f:
    w = csv.writer(f); w.writerow(['x', 'y', 'split', 'block'])
    for (x, y), k, b in zip(xy, s, blk): w.writerow([x, y, names[k], b])
print('tiles', len(xy), 'blocks', len(ub), 'seed', seed, 'max class-share deviation', round(dev, 4))
for k in range(3):
    c = cnt[s == k].sum(0)
    print(names[k], (s == k).sum(), 'tiles', round((s == k).mean(), 3), 'class shares G/PG/S/NS', (c / c.sum()).round(3))
print('overall shares', tot.round(3))

# normalisation statistics from training tiles only
rng = np.random.default_rng(0); tr = xy[s == 0]; samp = tr[rng.choice(len(tr), min(1500, len(tr)), replace=False)]
st = np.stack([np.load(f'{DATA}/struct/19/{x}/{y}.npy').reshape(4, -1)[:, ::7] for x, y in samp], 1).reshape(4, -1).astype(float)
stats = {'struct_mean': st.mean(1).tolist(), 'struct_std': st.std(1).tolist(), 'block_tiles': B, 'split_seed': int(seed)}
if os.path.isdir(f'{DATA}/rs'):   # near-infrared statistics only when SWISSIMAGE RS tiles were prepared
    nir = np.concatenate([np.load(f'{DATA}/rs/19/{x}/{y}.npy')[3].ravel()[::7] for x, y in samp]).astype(float) / 255.
    stats.update(nir_mean=float(nir.mean()), nir_std=float(nir.std()))
json.dump(stats, open(f'{DATA}/norm_stats.json', 'w'), indent=1); print(stats)
