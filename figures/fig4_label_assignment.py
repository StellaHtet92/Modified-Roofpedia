"""Fig. 4: label assignment strategies, illustrated with a real building (Fusion SWISSIMAGE predictions)."""
import sys, os, numpy as np, shapefile, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import FancyBboxPatch, Rectangle, Patch
from PIL import Image
from rasterio.warp import transform_geom
plt.rcParams.update({'font.family': 'Arial', 'font.size': 8, 'axes.linewidth': 0.6, 'xtick.major.width': 0.6, 'ytick.major.width': 0.6,
                     'pdf.fonttype': 42, 'ps.fonttype': 42})
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'pipeline'))
import config as C
J = int(sys.argv[1]) if len(sys.argv) > 1 else 4562
OUT = sys.argv[2] if len(sys.argv) > 2 else os.path.join(C.EVAL, 'fig4_label_assignment')
Z = 19; T = 2 * 20037508.342789244 / 2 ** Z; O = 20037508.342789244
PIX_M = T / 256 * np.cos(np.radians(46.95))            # ground pixel size at Bern (~0.204 m)
COL = ['#FFFFFF', '#A4E5D3', '#C8B455', '#2064AE', '#7E1700']   # bg, G, PG, S, NS (Okabe-Ito)
NAMES = ['Green', 'Potential Green', 'Solar', 'Not Suitable']
PM = np.arange(5); LM = np.arange(5)
CR = C.WORK
import csv
SPLIT = {(int(r['x']), int(r['y'])): r['split'] for r in csv.DictReader(open(CR + '/data/split.csv'))}

z = np.load(CR + '/eval/tilebld_f_swissimage.npz'); idx = z['idx']
shapes = shapefile.Reader(C.FOOTPRINTS).shapes()
g = transform_geom(C.FOOTPRINT_CRS, 'EPSG:3857', shapes[idx[J]].__geo_interface__)
ring = np.array(g['coordinates'][0] if g['type'] == 'Polygon' else g['coordinates'][0][0])
tx0, tx1 = int((ring[:, 0].min() + O) // T), int((ring[:, 0].max() + O) // T)
ty0, ty1 = int((O - ring[:, 1].max()) // T), int((O - ring[:, 1].min()) // T)

def find(t, kind):
    for sp in ['training', 'validation', 'evaluation']:
        p = f'{MD}/dataset/{sp}/{kind}/19/{t[0]}/{t[1]}.png'
        if os.path.exists(p): return p
nx, ny = tx1 - tx0 + 1, ty1 - ty0 + 1
img = np.zeros((256 * ny, 256 * nx, 3), np.uint8); pred = np.zeros((256 * ny, 256 * nx), int); lab = np.zeros_like(pred)
for a in range(nx):
    for b in range(ny):
        t = (tx0 + a, ty0 + b); sl = (slice(b * 256, (b + 1) * 256), slice(a * 256, (a + 1) * 256))
        img[sl] = np.asarray(Image.open(f'{CR}/data/swissimage/19/{t[0]}/{t[1]}.png').convert('RGB'))
        lab[sl] = np.asarray(Image.open(f'{CR}/data/labels/19/{t[0]}/{t[1]}.png'))
        pred[sl] = np.asarray(Image.open(f'{CR}/runs/swissimage_fusion_s1/pred_{SPLIT[t]}/19/{t[0]}/{t[1]}.png'))
X0 = tx0 * T - O; Y1 = O - ty0 * T
px = (ring[:, 0] - X0) / T * 256; py = (Y1 - ring[:, 1]) / T * 256
from matplotlib.path import Path
yy, xx = np.mgrid[0:pred.shape[0], 0:pred.shape[1]]
inside = Path(np.c_[px, py]).contains_points(np.c_[xx.ravel() + .5, yy.ravel() + .5]).reshape(pred.shape)
pad = 18
c0, c1 = int(max(px.min() - pad, 0)), int(min(px.max() + pad, pred.shape[1])); r0, r1 = int(max(py.min() - pad, 0)), int(min(py.max() + pad, pred.shape[0]))
side = max(c1 - c0, r1 - r0); c1 = min(c0 + side, pred.shape[1]); r1 = min(r0 + side, pred.shape[0])
crop = (slice(r0, r1), slice(c0, c1))
n_fp = inside.sum()
pf = np.array([(pred[inside] == k).sum() / n_fp for k in range(1, 5)])
rf = np.array([(lab[inside] == k).sum() / n_fp for k in range(1, 5)])
area_m2 = n_fp * PIX_M ** 2
area_lab = int(np.argmax(pf)) + 1
prio_lab = next(k for k in [3, 2, 1, 4] if pf[k - 1] > 0)
ref_lab = int(np.argmax(rf)) + 1
print('fractions pred', pf.round(3), 'ref', rf.round(3), 'area m2', round(area_m2), 'area', area_lab, 'prio', prio_lab, 'ref', ref_lab)

fig = plt.figure(figsize=(7.16, 2.75))
gs = fig.add_gridspec(1, 4, width_ratios=[1, 1, 1.15, 1.45], wspace=0.28, left=0.01, right=0.995, top=0.97, bottom=0.29)
ax = [fig.add_subplot(gs[0, i]) for i in range(4)]
# (a) orthophoto
ax[0].imshow(img[crop], interpolation='lanczos')
for a in ax[:2]:
    a.plot(px - c0, py - r0, color='white', lw=1.6); a.plot(px - c0, py - r0, color='black', lw=0.7)
    a.set_xticks([]); a.set_yticks([]); a.set_xlim(-0.5, c1 - c0 - 0.5); a.set_ylim(r1 - r0 - 0.5, -0.5)
sb = 10 / PIX_M; w = c1 - c0; h = r1 - r0
ax[0].add_patch(Rectangle((w * 0.94 - sb, h * 0.93), sb, h * 0.022, fc='white', ec='black', lw=0.5))
ax[0].text(w * 0.94 - sb / 2, h * 0.895, '10 m', ha='center', va='bottom', fontsize=7, color='white',
           bbox=dict(boxstyle='round,pad=0.12', fc='black', ec='none', alpha=0.55))
ax[0].annotate('', xy=(w * 0.92, h * 0.07), xytext=(w * 0.92, h * 0.2), arrowprops=dict(arrowstyle='-|>', color='white', lw=1.2, mutation_scale=8))
ax[0].text(w * 0.92, h * 0.23, 'N', ha='center', va='top', fontsize=7, color='white', fontweight='bold')
# (b) prediction inside footprint, faded outside
rgb = np.array([matplotlib.colors.to_rgb(c) for c in COL])[pred[crop]]
outside = ~inside[crop]
rgb[outside] = rgb[outside] * 0.35 + 0.65 * np.array([0.93, 0.93, 0.93])
ax[1].imshow(rgb, interpolation='nearest')
# (c) fractions
y = np.arange(4)[::-1]
ax[2].barh(y + 0.19, pf * 100, height=0.36, color=COL[1:], edgecolor='black', lw=0.5, label='Predicted')
ax[2].barh(y - 0.19, rf * 100, height=0.36, color='white', edgecolor=COL[1:], hatch='////', lw=0.9, label='Reference')
for yi, p, r in zip(y, pf, rf):
    ax[2].text(p * 100 + 2, yi + 0.19, f'{100*p:.0f}', va='center', fontsize=7)
    ax[2].text(r * 100 + 2, yi - 0.19, f'{100*r:.0f}', va='center', fontsize=7, color='#444444')
ax[2].set_yticks(y); ax[2].set_yticklabels(['G', 'PG', 'S', 'NS'])
ax[2].set_xlim(0, 100); ax[2].set_xlabel(r'Coverage fraction $p_{b,k}$ (%)')
ax[2].spines[['top', 'right']].set_visible(False)
leg = [Patch(fc='#bdbdbd', ec='black', lw=0.5, label='Predicted'), Patch(fc='white', ec='#555555', hatch='////', lw=0.6, label='Reference')]
ax[2].legend(handles=leg, fontsize=7, frameon=False, loc='lower right', handlelength=1.2)
# (d) rule outcomes
d = ax[3]; d.set_xlim(0, 10); d.set_ylim(0, 10); d.axis('off')
def chip(x, yv, k, wdt=3.9):
    d.add_patch(FancyBboxPatch((x, yv - 0.55), wdt, 1.1, boxstyle='round,pad=0.05,rounding_size=0.25', fc=COL[k], ec='black', lw=0.5))
    d.text(x + wdt / 2, yv, NAMES[k - 1], ha='center', va='center', fontsize=7, color='white' if k in (3, 4) else 'black')
def panel(ytop, title, rule, k, ok, note):
    d.add_patch(FancyBboxPatch((0.1, ytop - 3.05), 9.8, 3.0, boxstyle='round,pad=0.02,rounding_size=0.3', fc='#F5F5F5', ec='#9E9E9E', lw=0.5))
    d.text(0.4, ytop - 0.45, title, fontsize=8, fontweight='bold', va='center')
    d.text(0.4, ytop - 1.25, rule, fontsize=7, va='center')
    chip(0.4, ytop - 2.25, k)
    d.text(4.6, ytop - 2.25, '\u2713' if ok else '\u2717', fontsize=9, va='center', fontfamily='DejaVu Sans', color='#1B5E20' if ok else '#B71C1C'); d.text(5.25, ytop - 2.25, note, fontsize=7, va='center', color='#1B5E20' if ok else '#B71C1C')
panel(10, 'Area-based rule', r'$\hat{y}_b=\mathrm{arg\,max}_{k}\;p_{b,k}$', area_lab, area_lab == ref_lab, 'matches reference')
panel(6.6, 'Priority-based rule', 'first class present: S > PG > G > NS', prio_lab, prio_lab == ref_lab,
      f'entire {area_m2:,.0f} m² roof\ncounted as {NAMES[prio_lab-1]}')
d.text(0.4, 2.65, 'Reference (dominant class)', fontsize=8, fontweight='bold', va='center')
chip(0.4, 1.6, ref_lab)
d.text(4.6, 1.6, f'{100*rf[ref_lab-1]:.0f}% of footprint', fontsize=7, va='center')
for a, l in zip(ax, 'abcd'):
    a.text(0.5, -0.1 if a in ax[:2] else -0.3, f'({l})', transform=a.transAxes, ha='center', va='top', fontsize=8)

handles = [Patch(fc=COL[k], ec='black', lw=0.4, label=NAMES[k - 1]) for k in range(1, 5)]
fig.legend(handles=handles, loc='lower left', bbox_to_anchor=(0.01, -0.005), ncol=4, fontsize=7, frameon=False, handlelength=1.1, columnspacing=1.2)
fig.savefig(OUT + '.pdf'); fig.savefig(OUT + '.png', dpi=600); fig.savefig(OUT + '.tif', dpi=600, pil_kwargs={'compression': 'tiff_lzw'})
print('saved', OUT)



