"""All manuscript metrics for the retrained configurations. Writes CSV/JSON to eval/ and prints a summary."""
import json, os, numpy as np, pandas as pd
from math import comb
pd.set_option('display.width', 250); pd.set_option('display.max_columns', 40)
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config as C
EV = C.EVAL + os.sep
NAME = {'mapbox': 'Image-only, Mapbox', 'f_mapbox': 'Fusion, Mapbox', 'swissimage': 'Image-only, SWISSIMAGE', 'f_swissimage': 'Fusion, SWISSIMAGE',
        'rs': 'Image-only, SWISSIMAGE RS', 'f_rs': 'Fusion, SWISSIMAGE RS'}
CN = ['G', 'PG', 'S', 'NS']; TAU = 0.05; A = 0.0416
J = json.load(open(EV + 'tile_eval.json'))
OUT = {}

# ---------------------------------------------------------------- pixel level
rows = []
for k in NAME:
    for subset in ['evaluation', 'all']:
        cm = np.array(J['models'][k]['cm']['evaluation']) if subset == 'evaluation' else sum(np.array(J['models'][k]['cm'][s]) for s in ['training', 'validation', 'evaluation'])
        TP = np.diag(cm).astype(float); FP = cm.sum(0) - TP; FN = cm.sum(1) - TP
        iou = TP / np.maximum(TP + FP + FN, 1)
        pr = TP / np.maximum(TP + FP, 1); rc = TP / np.maximum(TP + FN, 1); f1 = 2 * pr * rc / np.maximum(pr + rc, 1e-12)
        tot = cm.sum(); po = TP.sum() / tot; pe = (cm.sum(1).astype(float) * cm.sum(0)).sum() / float(tot) ** 2
        both = cm[1:, 1:].sum()
        rows.append({'key': k, 'model': NAME[k], 'subset': subset, 'OA': TP[1:].sum() / cm[1:, :].sum(), 'mIoU4': iou[1:].mean(), 'mF1': f1[1:].mean(),
                     'kappa': (po - pe) / (1 - pe), **{f'IoU_{CN[i]}': iou[i + 1] for i in range(4)},
                     'roof_det': both / cm[1:, :].sum(), 'roof_IoU': both / (cm[1:, :].sum() + cm[:, 1:].sum() - both), 'px_M': cm[1:, :].sum() / 1e6})
P = pd.DataFrame(rows); P.to_csv(EV + 'pixel.csv', index=False)
print('=== PIXEL'); print(P.round(3).to_string(index=False))


def labels(pc, tot, tau=TAU, mode='area'):
    f = pc[:, 1:5] / np.maximum(tot, 1)[:, None]; has = f.sum(1) > tau
    if mode == 'area': return np.where(has, f.argmax(1) + 1, 0)
    out = np.zeros(len(f), int)
    for c in [4, 1, 2, 3]: out = np.where(f[:, c - 1] > 0, c, out)
    return np.where(has, out, 0)


def bscores(y, p):
    cm = np.zeros((4, 5), int)
    for a, b in zip(y, p): cm[a - 1, b] += 1
    n = cm.sum(); oa = sum(cm[i, i + 1] for i in range(4)) / n
    rec = [cm[i, i + 1] / max(cm[i].sum(), 1) for i in range(4)]; prec = [cm[i, i + 1] / max(cm[:, i + 1].sum(), 1) for i in range(4)]
    f1 = [2 * a * b / (a + b) if a + b else 0 for a, b in zip(prec, rec)]
    full = np.zeros((5, 5)); full[1:, :] = cm; po = np.trace(full) / n; pe = (full.sum(1) * full.sum(0)).sum() / n ** 2
    return dict(n=int(n), OA=oa, mF1=np.mean(f1), kappa=(po - pe) / (1 - pe), missed=int(cm[:, 0].sum()),
                **{f'PA_{CN[i]}': rec[i] for i in range(4)}, **{f'UA_{CN[i]}': prec[i] for i in range(4)}), cm


# ---------------------------------------------------------------- building level
store = {}; brows = []
for k in NAME:
    z = np.load(EV + f'tilebld_{k}.npz'); pc, lc, held, anyt, idx = z['pred'], z['lab'], z['held'], z['any'], z['idx']
    tot = lc.sum(1); y = np.where(lc[:, 1:5].sum(1) > 0, lc[:, 1:5].argmax(1) + 1, 0); ok = anyt & (y > 0)
    store[k] = dict(pc=pc, lc=lc, held=held, ok=ok, y=y, tot=tot, idx=idx)
    for subset, m in [('held-out', ok & held), ('all', ok)]:
        for mode in ['area', 'priority']:
            s, cm = bscores(y[m], labels(pc[m], tot[m], TAU, mode))
            brows.append({'key': k, 'model': NAME[k], 'subset': subset, 'rule': mode, **s})
            if subset == 'held-out' and mode == 'area': OUT[f'cm_{k}'] = cm.tolist()
Bd = pd.DataFrame(brows); Bd.to_csv(EV + 'building.csv', index=False)
print('\n=== BUILDING (tau=0.05)'); print(Bd.round(3).to_string(index=False))
print('\n=== tau sensitivity (held-out, area OA)')
OUT['tau'] = {}
for k in NAME:
    s = store[k]; m = s['ok'] & s['held']
    OUT['tau'][k] = [round(bscores(s['y'][m], labels(s['pc'][m], s['tot'][m], t))[0]['OA'], 3) for t in [0, 0.01, 0.05, 0.1, 0.2]]
    print(NAME[k], OUT['tau'][k])

# ---------------------------------------------------------------- paired tests (image-only vs fusion)
rng = np.random.default_rng(0)
def mcnemar(b, c):
    n = b + c; kk = min(b, c); return min(2 * sum(comb(n, i) for i in range(kk + 1)) / 2 ** n, 1.0) if n else 1.0
OUT['paired'] = {}
print('\n=== paired (area-based building OA)')
for a, f in [('mapbox', 'f_mapbox'), ('swissimage', 'f_swissimage'), ('rs', 'f_rs')]:
    sa, sf_ = store[a], store[f]
    pa = labels(sa['pc'], sa['tot']); pf = labels(sf_['pc'], sf_['tot']); y = sa['y']
    for nm, m in [('held-out', sa['ok'] & sa['held']), ('all', sa['ok'])]:
        ca = pa[m] == y[m]; cf = pf[m] == y[m]; b = int(np.sum(ca & ~cf)); c = int(np.sum(~ca & cf)); ii = np.arange(m.sum())
        d = [cf[s].mean() - ca[s].mean() for s in (rng.choice(ii, len(ii)) for _ in range(2000))]
        cia = np.percentile([ca[s].mean() for s in (rng.choice(ii, len(ii)) for _ in range(2000))], [2.5, 97.5])
        cif = np.percentile([cf[s].mean() for s in (rng.choice(ii, len(ii)) for _ in range(2000))], [2.5, 97.5])
        lo, hi = np.percentile(d, [2.5, 97.5]); p = mcnemar(b, c)
        OUT['paired'][f'{a}|{nm}'] = dict(n=int(m.sum()), oa_img=ca.mean(), ci_img=cia.tolist(), oa_fus=cf.mean(), ci_fus=cif.tolist(), diff=cf.mean() - ca.mean(), ci=[lo, hi], b=b, c=c, p=p)
        print(f'{a:10s} [{nm}] n={m.sum()} img={ca.mean():.3f} fus={cf.mean():.3f} diff={cf.mean() - ca.mean():+.3f} [{lo:+.3f},{hi:+.3f}] b={b} c={c} p={p:.4f}')

# ---------------------------------------------------------------- propagation, label rules, areas (fusion SWISSIMAGE + others)
OUT['prop'] = {}
for k in NAME:
    s = store[k]; pc, lc, y, tot = s['pc'], s['lc'], s['y'], s['tot']
    for nm, m in [('all', s['ok']), ('held-out', s['ok'] & s['held'])]:
        f = pc[:, 1:5] / np.maximum(pc[:, 1:5].sum(1), 1)[:, None]; srt = np.sort(f, 1); margin = srt[:, -1] - srt[:, -2]
        p = labels(pc, tot); pp = labels(pc, tot, mode='priority'); det = m & (p > 0)
        pix = np.array([pc[i, y[i]] / max(pc[i, 1:5].sum(), 1) if y[i] else np.nan for i in range(len(y))])
        bins = []
        for lo, hi in [(0, .2), (.2, .5), (.5, .8), (.8, 1.01)]:
            mm = det & (margin >= lo) & (margin < hi); bins.append(dict(lo=lo, n=int(mm.sum()), share=mm.sum() / max(det.sum(), 1), acc=float(np.mean(p[mm] == y[mm])) if mm.any() else None))
        area = []
        for c in range(1, 5):
            refa = lc[m, c].sum(); frac = pc[m, c].sum()
            hard = (lc[m, 1:5].sum(1) * (p[m] == c)).sum(); hardp = (lc[m, 1:5].sum(1) * (pp[m] == c)).sum()
            area.append(dict(cls=CN[c - 1], ref_m2=refa * A, frac_err=100 * (frac / refa - 1), hard_err=100 * (hard / refa - 1), prio_err=100 * (hardp / refa - 1)))
        sh = lambda v: [round(100 * np.mean(v[v > 0] == c), 1) for c in range(1, 5)]
        OUT['prop'][f'{k}|{nm}'] = dict(n=int(m.sum()), detected=int(det.sum()), missed=int((m & (p == 0)).sum()),
                                       pix_acc_det=float(np.nanmean(pix[det])), bld_acc_det=float(np.mean(p[det] == y[det])), margin=bins, area=area,
                                       share_ref=sh(y[m]), share_area=sh(p[m]), share_prio=sh(pp[m]),
                                       area_share_ref=[round(100 * lc[m, c].sum() / lc[m, 1:5].sum(), 1) for c in range(1, 5)],
                                       area_share_pred=[round(100 * pc[m, c].sum() / pc[m, 1:5].sum(), 1) for c in range(1, 5)])
for k in ['f_swissimage', 'swissimage']:
    print(f'\n=== propagation/areas {NAME[k]} (all)'); print(json.dumps(OUT['prop'][f'{k}|all'], indent=0, default=float)[:1800])

# ---------------------------------------------------------------- binary green-roof detection vs existing pipelines (optional)
def binm(y, p):
    tp = np.sum(p & y); fp = np.sum(p & ~y); fn = np.sum(~p & y); tn = np.sum(~p & ~y); n = len(y)
    Pp = tp / max(tp + fp, 1); R = tp / max(tp + fn, 1); F = 2 * Pp * R / max(Pp + R, 1e-9); oa = (tp + tn) / n
    pe = ((tp + fp) * (tp + fn) + (fn + tn) * (fp + tn)) / n ** 2
    return dict(n=int(n), P=Pp, R=R, F1=F, OA=oa, kappa=(oa - pe) / (1 - pe))
base = store['swissimage']; sel_all = np.array(sorted(base['idx'][base['ok']])); sel_held = np.array(sorted(base['idx'][base['ok'] & base['held']]))
ref_f = EV + 'counts_reference.npy'
brs = []
for nm, sel in [('all', sel_all), ('held-out', sel_held)]:
    if os.path.exists(ref_f):
        cref = np.load(ref_f).astype(float); tr = cref.sum(1); yb = cref[sel, 1:5].argmax(1) + 1 == 1
        for bname in C.BASELINES:
            cb = np.load(EV + f'counts_{bname}.npy').astype(float)
            if 'proj' in bname:   # proj-vegroofs-DL binary output: 1 vegetated, 2 bare
                pb = (cb[sel, 1] >= cb[sel, 2]) & ((cb[sel, 1] + cb[sel, 2]) / np.maximum(tr[sel], 1) > TAU)
            else:                 # Roofpedia green-roof output: 1 green roof
                pb = cb[sel, 1] / np.maximum(tr[sel], 1) > TAU
            brs.append({'subset': nm, 'model': f'{bname} (binary)', **binm(yb, pb)})
    for k in NAME:
        s = store[k]; m = np.isin(s['idx'], sel) & s['ok']
        brs.append({'subset': nm, 'model': NAME[k], **binm(s['y'][m] == 1, labels(s['pc'][m], s['tot'][m]) == 1)})
Bn = pd.DataFrame(brs); Bn.to_csv(EV + 'binary.csv', index=False)
print('\n=== BINARY green roof'); print(Bn.round(3).to_string(index=False))
json.dump(OUT, open(EV + 'analysis.json', 'w'), default=float, indent=1)
print('saved')
