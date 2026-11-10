"""Side-by-side preview of image, reference label and prediction for tiles of a run.
python preview.py <run_name> <x> <y> [<x> <y> ...]
Writes consistent_retrain/previews/<run>_<x>_<y>.png
"""
import os, sys, json, numpy as np, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
from PIL import Image
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config as C
CR = C.WORK
COLORS = ['#FFFFFF', '#A4E5D3', '#C8B455', '#2064AE', '#7E1700']      # roma10: bg, G, PG, S, NS
NAMES = ['Background', 'Green', 'Potential Green', 'Solar', 'Not Suitable']
CMAP = ListedColormap(COLORS)

run = sys.argv[1]; cfg = json.load(open(f'{CR}/runs/{run}/config.json')); src = cfg['source']
split = {}
import csv
for r in csv.DictReader(open(f'{CR}/data/split.csv')): split[(int(r['x']), int(r['y']))] = r['split']
os.makedirs(f'{CR}/previews', exist_ok=True)
plt.rcParams.update({'font.family': 'Arial', 'font.size': 8})
xy = list(map(int, sys.argv[2:]))
for x, y in zip(xy[0::2], xy[1::2]):
    sp = split.get((x, y))
    if src == 'rs':
        img = np.load(f'{CR}/data/rs/19/{x}/{y}.npy')[:3].transpose(1, 2, 0)
    else:
        img = np.asarray(Image.open(f'{CR}/data/{src}/19/{x}/{y}.png').convert('RGB'))
    lab = np.asarray(Image.open(f'{CR}/data/labels/19/{x}/{y}.png'))
    pred = np.asarray(Image.open(f'{CR}/runs/{run}/pred_{sp}/19/{x}/{y}.png'))
    roof = lab > 0
    acc = (pred[roof] == lab[roof]).mean() if roof.any() else float('nan')
    fig, ax = plt.subplots(1, 3, figsize=(9, 3.4))
    ax[0].imshow(img); ax[0].set_title(f'{src} image')
    ax[1].imshow(lab, cmap=CMAP, vmin=0, vmax=4, interpolation='nearest'); ax[1].set_title('Reference label')
    ax[2].imshow(pred, cmap=CMAP, vmin=0, vmax=4, interpolation='nearest'); ax[2].set_title(f'Prediction ({run})')
    for a in ax:
        a.set_xticks([]); a.set_yticks([])
        for s in a.spines.values(): s.set_linewidth(0.5)
    fig.suptitle(f'Tile z19 / {x} / {y}   split: {sp}   roof-pixel accuracy: {acc:.1%}', fontsize=9)
    fig.legend(handles=[Patch(fc=c, ec='#555555', lw=0.5, label=n) for c, n in zip(COLORS, NAMES)],
               loc='lower center', ncol=5, frameon=False, fontsize=8)
    plt.subplots_adjust(left=0.01, right=0.99, top=0.86, bottom=0.12, wspace=0.03)
    out = f'{CR}/previews/{run}_{x}_{y}.png'; fig.savefig(out, dpi=200); plt.close(fig)
    print('saved', out)
