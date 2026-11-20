"""Layer-level diagram of SegNet (roofpedia_bern/models.py): image-only U-Net and structure-aware fusion.
Tensor shapes are for a 256 x 256 tile (verified with forward hooks). Output: docs/architecture.png / .pdf"""
import os, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'docs', 'architecture')
plt.rcParams.update({'font.family': 'DejaVu Sans', 'pdf.fonttype': 42})
C_IN, C_ENC, C_ST, C_F, C_DEC, C_OUT = '#ECF4DA', '#C2DEED', '#E7D9BF', '#DBF5EE', '#BEE6ED', '#F2F2F2'
EDGE, SKIP, NOTE = '#37474F', '#033198', '#995215'
CLS = [('#FFFFFF', 'background'), ('#A4E5D3', 'Green'), ('#C8B455', 'Potential Green'), ('#2064AE', 'Solar'), ('#7E1700', 'Not Suitable')]

fig, ax = plt.subplots(figsize=(7.16, 4.9)); ax.set_xlim(0, 100); ax.set_ylim(0, 70); ax.axis('off')
plt.subplots_adjust(0, 0, 1, 1)
W, H = 14.5, 6.2
ROW = {0: 62, 1: 53, 2: 44, 3: 35, 4: 26, 5: 17, 6: 6}        # vertical position (box centre) per stage row
XS, XE, XF, XD, XO = 9, 28, 50, 72, 91                            # column centres: struct, encoder, fusion, decoder, output


def box(xc, yc, title, op, shape, fc, w=W, h=H):
    ax.add_patch(FancyBboxPatch((xc - w / 2, yc - h / 2), w, h, boxstyle='round,pad=0.15,rounding_size=0.7', fc=fc, ec=EDGE, lw=0.6))
    ax.text(xc, yc + h / 2 - 1.15, title, ha='center', va='top', fontsize=5.6, fontweight='bold')
    ax.text(xc, yc - 0.15, op, ha='center', va='center', fontsize=4.6, color='#263238')
    ax.text(xc, yc - h / 2 + 1.0, shape, ha='center', va='bottom', fontsize=5.0, family='DejaVu Sans Mono')


def arrow(x0, y0, x1, y1, color=EDGE, ls='-', lw=0.7, rad=0.0):
    ax.annotate('', xy=(x1, y1), xytext=(x0, y0), arrowprops=dict(arrowstyle='-|>', color=color, lw=lw, ls=ls, mutation_scale=6,
                                                              shrinkA=0, shrinkB=0, connectionstyle=f'arc3,rad={rad}'))


# ---------------------------------------------------------------- elevation branch (fusion only)
ax.add_patch(Rectangle((XS - W / 2 - 1.3, 1.2), W + 2.6, 66.6, fill=False, ec=NOTE, lw=0.6, ls=(0, (3, 2))))
ax.add_patch(Rectangle((XF - W / 2 - 1.3, 1.6), W + 2.6, 19.9, fill=False, ec=NOTE, lw=0.6, ls=(0, (3, 2))))
ax.text(XS, 68.6, 'elevation branch', ha='center', fontsize=5.4, color=NOTE, style='italic')
ax.text(XF, 0.35, 'fusion block', ha='center', fontsize=5.4, color=NOTE, style='italic')
box(XS, ROW[0], 'Elevation input', 'nDSM, slope, sin/cos aspect', '4 × 256 × 256', C_IN)
box(XS, ROW[1], 'Stem', 'conv 7×7/2, BN, ReLU, pool', '64 × 64 × 64', C_ST)
box(XS, (ROW[2] + ROW[4]) / 2, 'ResNet-18 layers 1–4', 'basic residual blocks', '512 × 8 × 8', C_ST, h=H + 3)
box(XS, ROW[5], 'Projection', 'conv 1×1', '2048 × 8 × 8', C_ST)
arrow(XS, ROW[0] - H / 2, XS, ROW[1] + H / 2); arrow(XS, ROW[1] - H / 2, XS, (ROW[2] + ROW[4]) / 2 + (H + 3) / 2)
arrow(XS, (ROW[2] + ROW[4]) / 2 - (H + 3) / 2, XS, ROW[5] + H / 2)

# ---------------------------------------------------------------- image encoder (ResNet-50)
box(XE, ROW[0], 'Image input', 'RGB or RGB + NIR tile, 0.2 m', '3|4 × 256 × 256', C_IN)
enc = [('Stem', 'conv 7×7/2, BN, ReLU, pool', '64 × 64 × 64'), ('enc1 = layer1', '3 bottleneck blocks', '256 × 64 × 64'),
       ('enc2 = layer2', '4 bottleneck blocks, /2', '512 × 32 × 32'), ('enc3 = layer3', '6 bottleneck blocks, /2', '1024 × 16 × 16'),
       ('enc4 = layer4', '3 bottleneck blocks, /2', '2048 × 8 × 8')]
for r, (t, o, s) in enumerate(enc, start=1):
    box(XE, ROW[r], t, o, s, C_ENC)
for r in range(0, 5):
    arrow(XE, ROW[r] - H / 2, XE, ROW[r + 1] + H / 2)
ax.text(XE, 68.6, 'image encoder (ResNet-50, ImageNet)', ha='center', fontsize=5.4, color=EDGE, style='italic')

# ---------------------------------------------------------------- fusion
box(XF, ROW[5], 'Concat', 'enc4 ‖ projection', '4096 × 8 × 8', C_F)
box(XF, ROW[6], 'Conv-BN-ReLU ×2', '3×3: 4096→2048, 2048→2048', '2048 × 8 × 8', C_F)
arrow(XE + W / 2, ROW[5], XF - W / 2, ROW[5])
yb, xv = ROW[5] - H / 2 - 2.2, XF - W / 2 - 0.6           # route under enc4: down, right, up, into Concat
ax.plot([XS, XS, xv, xv], [ROW[5] - H / 2, yb, yb, ROW[5] - 1.6], color=EDGE, lw=0.7, solid_capstyle='butt')
arrow(xv, ROW[5] - 1.6, XF - W / 2, ROW[5] - 1.6)
arrow(XF, ROW[5] - H / 2, XF, ROW[6] + H / 2)
ax.text(XE, ROW[6] + 0.5, 'image-only network:\nno elevation branch and no fusion block;\nenc4 goes straight to center and dec0', ha='center', va='center', fontsize=4.4, color=NOTE, style='italic')

# ---------------------------------------------------------------- decoder
dec = {6: ('center', 'max-pool /2, up ×2, Conv-BN-ReLU', '256 × 8 × 8'),
       5: ('dec0', '[fused ‖ center] 2304 → up ×2 → conv', '256 × 16 × 16'),
       4: ('dec1', '[enc3 ‖ dec0] 1280 → up ×2 → conv', '256 × 32 × 32'),
       3: ('dec2', '[enc2 ‖ dec1] 768 → up ×2 → conv', '64 × 64 × 64'),
       2: ('dec3', '[enc1 ‖ dec2] 320 → up ×2 → conv', '128 × 128 × 128'),
       1: ('dec4, dec5', 'up ×2 → conv; Conv-BN-ReLU', '32 × 256 × 256')}
for r, (t, o, s) in dec.items():
    box(XD, ROW[r], t, o, s, C_DEC, w=W + 3.5)
for r in range(6, 1, -1):
    arrow(XD, ROW[r] + H / 2, XD, ROW[r - 1] - H / 2)
ax.text(XD, 68.6, 'U-Net decoder (Conv-BN-ReLU blocks)', ha='center', fontsize=5.4, color=EDGE, style='italic')
arrow(XF + W / 2, ROW[6], XD - (W + 3.5) / 2, ROW[6])                                   # fused bottleneck -> center
arrow(XF + W / 2, ROW[6] + 1.6, XD - (W + 3.5) / 2, ROW[5] - 1.0, color=SKIP, ls=(0, (3, 2)), rad=-0.15)  # fused -> dec0 concat
for r in (2, 3, 4):                                                                       # skip connections enc1-enc3
    arrow(XE + W / 2, ROW[r], XD - (W + 3.5) / 2, ROW[r], color=SKIP, ls=(0, (3, 2)))
ax.text((XE + XD) / 2, ROW[2] + 1.2, 'skip connection (concatenation)', ha='center', fontsize=4.6, color=SKIP)

# ---------------------------------------------------------------- output
box(XO, ROW[1], 'Classifier', 'conv 1×1', '5 × 256 × 256', C_OUT, w=W - 1.5)
arrow(XD + (W + 3.5) / 2, ROW[1], XO - (W - 1.5) / 2, ROW[1])
box(XO, ROW[0], 'Class map', 'argmax over 5 classes', '256 × 256', C_OUT, w=W - 1.5)
arrow(XO, ROW[1] + H / 2, XO, ROW[0] - H / 2)
for k, (c, n) in enumerate(CLS):
    yy = ROW[3] + 2.5 - k * 2.6
    ax.add_patch(Rectangle((XO - 6.2, yy - 0.9), 1.8, 1.8, fc=c, ec=EDGE, lw=0.4))
    ax.text(XO - 3.8, yy, f'{k} {n}', va='center', fontsize=4.8)
ax.text(XO, ROW[3] + 5.2, 'output classes', ha='center', fontsize=5.0, fontweight='bold')
ax.text(XO, ROW[5] + 1.0, 'parameters\nimage-only  39.4 M\nfusion      164.9 M', ha='center', va='center', fontsize=4.8, family='DejaVu Sans Mono',
        bbox=dict(fc='white', ec='#B0BEC5', lw=0.5, boxstyle='round,pad=0.4'))

os.makedirs(os.path.dirname(OUT), exist_ok=True)
fig.savefig(OUT + '.png', dpi=400); fig.savefig(OUT + '.pdf')
print('saved', OUT + '.png')
