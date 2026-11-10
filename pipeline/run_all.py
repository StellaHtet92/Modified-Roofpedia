"""Train all configurations (max N concurrently on the GPU), then predict every split with best.pth.
python run_all.py <epochs> <concurrency> <seed> [config ...]   e.g. python run_all.py 200 3 1
"""
import os, sys, subprocess, time
CODE = os.path.dirname(os.path.abspath(__file__))
PY = sys.executable
sys.path.insert(0, CODE)
import config as C
RUNS = C.RUNS
epochs, conc, seed = sys.argv[1], int(sys.argv[2]), sys.argv[3]
cfgs = [c.split(':') for c in sys.argv[4:]] or [('swissimage', 'fusion'), ('swissimage', 'rgb'), ('mapbox', 'fusion'),
                                              ('mapbox', 'rgb'), ('rs', 'fusion'), ('rs', 'rgb')]
queue = list(cfgs); active = []
os.makedirs(RUNS, exist_ok=True)

def finish(src, kind):
    run = f'{src}_{kind}_s{seed}'
    with open(f'{RUNS}/{run}/predict.log', 'w') as lg:
        for sp in ['evaluation', 'validation', 'training']:
            subprocess.run([PY, f'{CODE}/predict.py', run, sp, 'best'], stdout=lg, stderr=subprocess.STDOUT)

while queue or active:
    while queue and len(active) < conc:
        src, kind = queue.pop(0); run = f'{src}_{kind}_s{seed}'
        if os.path.exists(f'{RUNS}/{run}/DONE'):
            print('skip (done)', run, flush=True); finish(src, kind); continue
        os.makedirs(f'{RUNS}/{run}', exist_ok=True)
        lg = open(f'{RUNS}/{run}/train.log', 'w')
        p = subprocess.Popen([PY, f'{CODE}/train.py', src, kind, epochs, seed], stdout=lg, stderr=subprocess.STDOUT)
        active.append((p, src, kind, lg)); print(time.strftime('%H:%M'), 'started', run, flush=True)
    time.sleep(30)
    for a in list(active):
        p, src, kind, lg = a
        if p.poll() is not None:
            lg.close(); active.remove(a)
            print(time.strftime('%H:%M'), 'finished', src, kind, 'exit', p.returncode, flush=True)
            if p.returncode == 0: finish(src, kind)
print('ALL DONE', flush=True)
