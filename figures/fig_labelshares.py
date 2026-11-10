"""Fig. 6: class shares of the SWISSIMAGE fusion network (all labelled buildings in the tiles)."""
import json, numpy as np, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'pipeline'))
import config as C
EV = C.EVAL + os.sep
P = json.load(open(EV + 'analysis.json'))['prop']['f_swissimage|all']
data = {'Reference\n(dominant class)': P['share_ref'], 'Area-based\nlabels': P['share_area'], 'Priority-based\nlabels': P['share_prio'],
        'Reference\narea': P['area_share_ref'], 'Predicted\nfractions': P['area_share_pred']}
plt.rcParams.update({'font.family': 'Arial', 'pdf.fonttype': 42})
cols = ['#A4E5D3', '#C8B455', '#2064AE', '#7E1700']; names = ['Green', 'Potential Green', 'Solar', 'Not Suitable']
fig, ax = plt.subplots(figsize=(3.45, 2.6))
xs = np.arange(len(data)); bottom = np.zeros(len(data))
for c in range(4):
    vals = np.array([v[c] for v in data.values()])
    ax.bar(xs, vals, bottom=bottom, color=cols[c], label=names[c], width=0.62, edgecolor='white', lw=0.5)
    for x, b, v in zip(xs, bottom, vals):
        if v > 5: ax.text(x, b + v / 2, f'{v:.1f}', ha='center', va='center', fontsize=5.5, color='white' if c >= 2 else 'black')
    bottom += vals
ax.set_xticks(xs); ax.set_xticklabels(list(data.keys()), fontsize=5.6); ax.set_ylabel('Share (%)', fontsize=7); ax.tick_params(axis='y', labelsize=6)
ax.axvline(2.5, color='#757575', lw=0.6, ls='--')
ax.text(1, 103, 'buildings', ha='center', fontsize=6); ax.text(3.5, 103, 'roof area', ha='center', fontsize=6)
ax.set_ylim(0, 108); ax.legend(fontsize=5.5, ncol=4, loc='upper center', bbox_to_anchor=(0.5, -0.2), frameon=False, handlelength=1)
plt.tight_layout()
for ext, kw in [('pdf', {}), ('png', {'dpi': 600})]: plt.savefig(EV + 'fig_labelshares.' + ext, **kw)
print({k: v for k, v in data.items()})
