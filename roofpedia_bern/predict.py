"""Predict roof-class masks for a folder of slippy-map tiles.

python -m roofpedia_bern.predict --weights weights/swissimage_fusion.pth --tiles my_tiles --out masks [--struct my_struct] [--color masks_color]

--tiles   folder with <z>/<x>/<y>.png|.jpg|.tif (RGB, 8-bit) or <z>/<x>/<y>.npy (uint8, shape (4, H, W): R, G, B, NIR)
--struct  (fusion models only) folder with <z>/<x>/<y>.npy, float32 (4, H, W): nDSM [m], slope [deg],
          sin(aspect), cos(aspect), as written by make_struct.py; NOT normalized (normalization is applied here)
--out     masks are written as paletted PNG: the pixel values are the class codes 0 background, 1 Green,
          2 Potential Green, 3 Solar, 4 Not Suitable, and image viewers display them in the class colours
--color   optional folder for colored copies of the masks
"""
import argparse, glob, os, numpy as np, torch
from PIL import Image
from .models import load_model

PALETTE = np.array([[255, 255, 255], [164, 229, 211], [200, 180, 85], [32, 100, 174], [126, 23, 0]], np.uint8)


def save_mask(codes, path):
    """Save class codes 0-4 as a paletted PNG: pixel values stay 0-4, but image viewers show the class colours."""
    im = Image.fromarray(codes.astype(np.uint8), mode='P')
    im.putpalette(PALETTE.ravel().tolist() + [0] * (768 - PALETTE.size))
    im.save(path)


def list_tiles(root):
    out = []
    for f in glob.glob(os.path.join(root, '*', '*', '*')):
        stem, ext = os.path.splitext(f)
        if ext.lower() in ('.png', '.jpg', '.jpeg', '.tif', '.tiff', '.npy'):
            z, x, y = f.replace('\\', '/').split('/')[-3:]
            out.append((int(z), int(x), int(os.path.splitext(y)[0]), f))
    return sorted(out)


def read_image(path, n_bands):
    if path.endswith('.npy'):
        a = np.load(path)
    else:
        im = Image.open(path)
        a = np.asarray(im.convert('RGB') if n_bands == 3 else im).transpose(2, 0, 1)
    if a.shape[0] != n_bands:
        raise ValueError(f'{path}: expected {n_bands} bands, got {a.shape[0]}')
    return a.astype(np.float32) / 255.


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--weights', required=True); ap.add_argument('--tiles', required=True); ap.add_argument('--out', required=True)
    ap.add_argument('--struct'); ap.add_argument('--color'); ap.add_argument('--batch', type=int, default=16)
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    a = ap.parse_args()
    net, meta = load_model(a.weights, a.device)
    nb = meta['in_channels']
    if meta['fusion'] and not a.struct:
        ap.error('this is a fusion model: --struct is required (see make_struct.py)')
    m = np.array(meta['image_mean'], np.float32)[:, None, None]; s = np.array(meta['image_std'], np.float32)[:, None, None]
    sm = np.array(meta['struct_mean'], np.float32)[:, None, None]; ss = np.array(meta['struct_std'], np.float32)[:, None, None]
    tiles = list_tiles(a.tiles)
    print(f'{meta["name"]}: {len(tiles)} tiles, {nb} bands, fusion={meta["fusion"]}, device={a.device}')
    n = 0
    for i in range(0, len(tiles), a.batch):
        chunk = tiles[i:i + a.batch]; imgs, sts, keep = [], [], []
        for z, x, y, f in chunk:
            img = read_image(f, nb)
            if img.shape[1] % 32 or img.shape[2] % 32:
                print('skip (size not divisible by 32):', f); continue
            if meta['fusion']:
                sp = os.path.join(a.struct, str(z), str(x), f'{y}.npy')
                if not os.path.exists(sp):
                    print('skip (no struct tile):', f); continue
                sts.append((np.load(sp).astype(np.float32) - sm) / ss)
            imgs.append((img - m) / s); keep.append((z, x, y))
        if not imgs: continue
        with torch.no_grad():
            o = net(torch.from_numpy(np.stack(imgs)).to(a.device), torch.from_numpy(np.stack(sts)).to(a.device) if meta['fusion'] else None)
        for (z, x, y), p in zip(keep, o.argmax(1).byte().cpu().numpy()):
            d = os.path.join(a.out, str(z), str(x)); os.makedirs(d, exist_ok=True); save_mask(p, os.path.join(d, f'{y}.png'))
            if a.color:
                d = os.path.join(a.color, str(z), str(x)); os.makedirs(d, exist_ok=True); Image.fromarray(PALETTE[p]).save(os.path.join(d, f'{y}.png'))
            n += 1
    print('masks written:', n)


if __name__ == '__main__':
    main()
