"""City-wide prediction for one run over all z19 study-area tiles: python predict_city.py <run_name>
Structural tiles are rendered per tile row exactly as in prep_data.py (same rasters, resampling, 6.5 deg rule)."""
import os, sys, glob, json, numpy as np, torch
os.environ['GDAL_CACHEMAX'] = '64'
from collections import defaultdict
from PIL import Image
from rasterio.warp import Resampling
sys.path.insert(0, os.path.dirname(__file__))
import prep_data as P
from models import SegNet
from data import norm_stats, SOURCES, IMNET_MEAN, IMNET_STD
import config as C
RUNS = C.RUNS
run = sys.argv[1]; cfg = json.load(open(f'{RUNS}/{run}/config.json')); fusion = cfg['kind'] == 'fusion'
assert cfg['source'] in ('swissimage', 'mapbox'), 'city-wide RS tiles are rendered on the fly only for the training set'
SRC = P.SWISS if cfg['source'] == 'swissimage' else P.MAPBOX
out = f'{RUNS}/{run}/pred_city'
def struct_row(arg):
    y, xs = arg
    strips = {k: P.Strip(P.G[k], *P.row_bounds(y, xs)) for k in ['ndsm', 'slope', 'aspect']}
    out = []
    for x in xs:
        nd, sl, asp = [np.nan_to_num(strips[k].warp(x, y, Resampling.bilinear, np.nan), nan=0.0) for k in ['ndsm', 'slope', 'aspect']]
        rad = np.deg2rad(asp); s_, c_ = np.sin(rad), np.cos(rad); s_[sl < P.FLAT] = 0; c_[sl < P.FLAT] = 0
        out.append(np.stack([nd, sl, s_, c_]).astype(np.float32))
    return y, xs, out


if __name__ == '__main__':
    from multiprocessing import Pool
    dev = torch.device('cuda')
    net = SegNet(5, SOURCES[cfg['source']], fusion).to(dev).to(memory_format=torch.channels_last)
    net.load_state_dict(torch.load(f'{RUNS}/{run}/best.pth', map_location=dev)['state_dict']); net.eval()
    st = norm_stats(); m = np.array(IMNET_MEAN, np.float32)[:, None, None]; s = np.array(IMNET_STD, np.float32)[:, None, None]
    sm = np.array(st['struct_mean'], np.float32)[:, None, None]; ss = np.array(st['struct_std'], np.float32)[:, None, None]
    tiles = sorted((int(os.path.basename(os.path.dirname(f))), int(os.path.splitext(os.path.basename(f))[0])) for f in glob.glob(SRC + '/*/*.png'))
    byrow = defaultdict(list)
    for x, y in tiles:
        if not os.path.exists(f'{out}/19/{x}/{y}.png'): byrow[y].append(x)
    jobs = sorted(byrow.items()); print('rows to do', len(jobs), 'tiles', sum(len(v) for v in byrow.values()), flush=True)
    n = 0
    with Pool(6, initializer=P.init) as pool:
        for j, (y, xs, sts_all) in enumerate(pool.imap(struct_row, jobs)):
            for i in range(0, len(xs), 32):
                bx = xs[i:i + 32]; imgs = []
                for x in bx:
                    a = np.asarray(Image.open(f'{SRC}/{x}/{y}.png').convert('RGB')).transpose(2, 0, 1).astype(np.float32) / 255.
                    imgs.append((a - m) / s)
                sts = [(st_.astype(np.float32) - sm) / ss for st_ in sts_all[i:i + 32]]
                with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
                    o = net(torch.from_numpy(np.stack(imgs)).to(dev).to(memory_format=torch.channels_last),
                            torch.from_numpy(np.stack(sts)).to(dev) if fusion else None)
                for x, pr in zip(bx, o.argmax(1).byte().cpu().numpy()):
                    d = f'{out}/19/{x}'; os.makedirs(d, exist_ok=True); Image.fromarray(pr).save(f'{d}/{y}.png'); n += 1
            if j % 20 == 0: print(j, '/', len(jobs), 'rows', n, 'tiles', flush=True)
    print('done', n)