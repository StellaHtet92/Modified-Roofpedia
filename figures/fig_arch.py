import matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
plt.rcParams['font.family'] = 'DejaVu Sans'
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'pipeline'))
import config as C
OUT = os.path.join(C.EVAL, 'fig_arch.png')
fig, ax = plt.subplots(figsize=(7.16, 2.55)); ax.set_xlim(0, 100); ax.set_ylim(0, 36); ax.axis('off')
plt.subplots_adjust(0, 0, 1, 1)
FS = 6.3
def box(x, y, w, h, t, fc, fs=FS, title=None):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.2,rounding_size=0.8', fc=fc, ec='#37474f', lw=0.7))
    if title:
        ax.text(x + w / 2, y + h - 1.6, title, ha='center', va='top', fontsize=fs + 0.4, fontweight='bold')
        ax.text(x + w / 2, y + (h - 3.2) / 2, t, ha='center', va='center', fontsize=fs, linespacing=1.25)
    else:
        ax.text(x + w / 2, y + h / 2, t, ha='center', va='center', fontsize=fs, linespacing=1.25)
def arr(x0, y0, x1, y1):
    ax.annotate('', xy=(x1, y1), xytext=(x0, y0), arrowprops=dict(arrowstyle='-|>', lw=0.8, color='#37474f', mutation_scale=7))
C_IN, C_RGB, C_ST, C_F, C_D, C_P = '#ECF4DA', '#C2DEED', '#E7D9BF', '#DBF5EE', '#BEE6ED', '#F2F2F2'
box(0.5, 19.5, 16.5, 12, 'Orthophoto tile, 0.2 m\nRGB: 3 × 256 × 256\nRGB+NIR: 4 × 256 × 256', C_IN, title='Spectral input')
box(0.5, 2.5, 16.5, 12, 'nDSM, slope,\nsin(aspect), cos(aspect)\n4 × 256 × 256', C_IN, title='Structural input')
box(20.5, 19.5, 17, 12, 'ResNet-50, ImageNet\nenc1 … enc4\n2048 × 8 × 8', C_RGB, title='Image encoder')
box(20.5, 2.5, 17, 12, 'ResNet-18, 4-ch conv1\n1×1 conv 512→2048\n2048 × 8 × 8', C_ST, title='Structural encoder')
box(41, 11, 16, 12, 'Concat (4096 ch)\n→ Conv-BN-ReLU\n→ Conv-BN-ReLU', C_F, title='Bottleneck fusion')
box(60.5, 11, 17, 12, 'U-Net decoder, BN\nskip connections\nfrom image encoder', C_D, title='Decoder')
box(81, 11, 18.5, 12, 'background, Green,\nPotential Green,\nSolar, Not Suitable', C_P, title='5-class softmax')
arr(17.2, 25.5, 20.4, 25.5); arr(17.2, 8.5, 20.4, 8.5)
arr(37.7, 25.5, 40.9, 20); arr(37.7, 8.5, 40.9, 14)
arr(57.2, 17, 60.4, 17); arr(77.7, 17, 80.9, 17)
ax.annotate('', xy=(66, 23.3), xytext=(37.7, 29), arrowprops=dict(arrowstyle='-|>', lw=0.7, color='#033198', ls=(0, (2, 2)), mutation_scale=7, connectionstyle='arc3,rad=-0.15'))
ax.text(52, 31.2, 'skip connections enc1–enc3', fontsize=FS - 0.3, color='#033198', ha='center')
ax.text(29, 0.2, 'removed in the image-only baseline', fontsize=FS - 0.5, color='#995215', ha='center', style='italic')
ax.text(49, 9.6, 'removed in the image-only baseline', fontsize=FS - 0.5, color='#995215', ha='center', style='italic', va='top')
plt.savefig(OUT, dpi=400); print('ok')
