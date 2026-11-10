"""Fig. 5: learning curves of the six retrained configurations (four-class validation mIoU, loss, gap)."""
import os, pandas as pd, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'pipeline'))
import config as C
R = C.RUNS
OUT = os.path.join(C.EVAL, 'fig_training')
plt.rcParams.update({'font.family': 'Arial', 'font.size': 7, 'pdf.fonttype': 42})
SRC = [('mapbox', 'Mapbox', '#2064AE'), ('swissimage', 'SWISSIMAGE', '#B0802B'), ('rs', 'SWISSIMAGE RS', '#7E1700')]
fig, ax = plt.subplots(1, 3, figsize=(7.16, 2.35))
for key, name, col in SRC:
    for kind, ls, lw, lab in [('rgb', '--', 0.9, 'Image-only'), ('fusion', '-', 1.2, 'Fusion')]:
        f = f'{R}/{key}_{kind}_s1/history.csv'
        if not os.path.exists(f): continue
        h = pd.read_csv(f); sm = lambda s: s.rolling(5, min_periods=1, center=True).mean()
        ax[0].plot(h.epoch, sm(h.val_miou4), color=col, ls=ls, lw=lw, label=f'{lab}, {name}')
        ax[1].plot(h.epoch, sm(h.val_loss), color=col, ls=ls, lw=lw)
        ax[2].plot(h.epoch, sm(h.train_miou4 - h.val_miou4), color=col, ls=ls, lw=lw)
for a, t, yl in zip(ax, ['(a) Validation mIoU (four roof classes)', '(b) Validation Lovász loss', '(c) Train − validation mIoU'],
                    ['mIoU', 'Loss', 'Gap']):
    a.set_title(t, fontsize=7.5); a.set_xlabel('Epoch'); a.set_ylabel(yl); a.grid(alpha=.3, lw=0.4); a.set_xlim(0, 200)
ax[0].legend(fontsize=5.6, loc='lower right', frameon=False, ncol=1)
plt.tight_layout(pad=0.4)
for ext, kw in [('pdf', {}), ('png', {'dpi': 600})]: fig.savefig(f'{OUT}.{ext}', **kw)
print('ok')
