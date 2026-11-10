"""Save coloured copies of class-code masks (0 bg, 1 G, 2 PG, 3 S, 4 NS) in the roma10 class colours.
python colorize.py <run_name>      -> consistent_retrain/colored/<run>/<split>/19/x/y.png
python colorize.py labels          -> consistent_retrain/colored/labels/19/x/y.png
"""
import glob, os, sys, numpy as np
from PIL import Image
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config as C
CR = C.WORK
PAL = np.array([[255, 255, 255], [164, 229, 211], [200, 180, 85], [32, 100, 174], [126, 23, 0]], np.uint8)
name = sys.argv[1]
if name == 'labels':
    jobs = [(f'{CR}/data/labels', f'{CR}/colored/labels')]
else:
    jobs = [(d, f'{CR}/colored/{name}/' + os.path.basename(d).replace('pred_', '')) for d in glob.glob(f'{CR}/runs/{name}/pred_*')]
n = 0
for src, dst in jobs:
    for f in glob.glob(f'{src}/19/*/*.png'):
        a = np.asarray(Image.open(f)); a = np.where(a > 4, 0, a)
        o = f.replace(src, dst); os.makedirs(os.path.dirname(o), exist_ok=True)
        Image.fromarray(PAL[a]).save(o); n += 1
print(name, 'coloured tiles:', n)
