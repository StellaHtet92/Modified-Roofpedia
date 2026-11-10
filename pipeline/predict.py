"""Predict class masks for one run on a split: python predict.py <run_name> [split=evaluation] [ckpt=best]"""
import os, sys, json, numpy as np, torch
from PIL import Image
sys.path.insert(0, os.path.dirname(__file__))
from models import SegNet
from data import TileDataset, read_split, SOURCES
import config as C
RUNS = C.RUNS

run = sys.argv[1]; split = sys.argv[2] if len(sys.argv) > 2 else 'evaluation'; ck = sys.argv[3] if len(sys.argv) > 3 else 'best'
cfg = json.load(open(f'{RUNS}/{run}/config.json')); fusion = cfg['kind'] == 'fusion'
dev = torch.device('cuda')
net = SegNet(5, SOURCES[cfg['source']], fusion).to(dev).to(memory_format=torch.channels_last)
net.load_state_dict(torch.load(f'{RUNS}/{run}/{ck}.pth', map_location=dev)['state_dict']); net.eval()
ds = TileDataset(read_split(split), cfg['source'], fusion, augment=False)
dl = torch.utils.data.DataLoader(ds, batch_size=16, num_workers=0)
out = f'{RUNS}/{run}/pred_{split}'
n = 0
with torch.no_grad():
    for img, st, _, xy in dl:
        with torch.autocast('cuda', dtype=torch.bfloat16):
            o = net(img.to(dev).to(memory_format=torch.channels_last), st.to(dev) if fusion else None)
        p = o.argmax(1).byte().cpu().numpy()
        for (x, y), m in zip(xy.tolist(), p):
            d = f'{out}/19/{x}'; os.makedirs(d, exist_ok=True); Image.fromarray(m).save(f'{d}/{y}.png'); n += 1
print(run, split, ck, 'predicted', n)
