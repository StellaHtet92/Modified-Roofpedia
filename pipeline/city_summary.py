import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config as C
import numpy as np, pandas as pd, shapefile
"""City inventory from city_counts.py output (all footprints), with the reference set taken from the reference table."""
KEY = sys.argv[1] if len(sys.argv) > 1 else 'f_swissimage'
c = np.load(os.path.join(C.EVAL, f'city_{KEY}.npy')).astype(float)
ids = [r[0] for r in shapefile.Reader(C.FOOTPRINTS).records()]
r = shapefile.Reader(dbf=open(C.REFERENCE_TABLE, 'rb'))
L = pd.DataFrame([list(x) for x in r.records()], columns=[q[0] for q in r.fields[1:]])
ref = dict(zip(L[C.ID_FIELD], pd.to_numeric(L[C.REF_FIELD], errors='coerce').fillna(0).astype(int)))
labelled = np.array([ref.get(i, 0) > 0 for i in ids])
tot = c.sum(1); A = (2 * 20037508.342789244 / 2 ** 19 / 256 * np.cos(np.radians(46.95))) ** 2   # m2 per tile pixel at Bern
f = c[:, 1:5] / np.maximum(tot, 1)[:, None]; det = f.sum(1) > 0.05
lab = np.where(det, f.argmax(1) + 1, 0)
print('footprints with tile coverage', int((tot > 0).sum()), ' detected flat roofs', int(det.sum()), ' in reference set', int((det & labelled).sum()), ' not in reference set', int((det & ~labelled).sum()))
names = ['G', 'PG', 'S', 'NS']
for nm, m in [('all detected', det), ('not in reference set', det & ~labelled)]:
    print(nm, 'buildings', {names[k]: int(np.sum(lab[m] == k + 1)) for k in range(4)}, ' shares %', {names[k]: round(100 * np.mean(lab[m] == k + 1), 1) for k in range(4)})
    fa = c[m, 1:5].sum(0) * A
    print('   fractional roof area m2', {names[k]: round(fa[k]) for k in range(4)}, ' total', round(fa.sum()))
