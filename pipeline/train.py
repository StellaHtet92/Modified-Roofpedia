"""Train one configuration: python train.py <source: mapbox|swissimage|rs> <rgb|fusion> [epochs] [seed]

Identical for all configurations: tiles, labels, split, augmentation (random dihedral transform),
Lovász loss, Adam (lr 5e-4), batch size 8, number of epochs, decoder, and model selection
(checkpoint with the highest validation mIoU over the four roof classes).
"""
import os, sys, time, json, random, csv, numpy as np, torch
sys.path.insert(0, os.path.dirname(__file__))
from models import SegNet, LovaszLoss2d
from data import TileDataset, read_split, SOURCES

import config as C
RUNS = C.RUNS


def _winit(w):
    np.random.seed(torch.initial_seed() % 2 ** 32)


def confusion(pred, lab, n=5):
    return torch.bincount((lab.view(-1) * n + pred.view(-1)), minlength=n * n).view(n, n)


def scores(cm):
    cm = cm.double()
    iou = cm.diag() / (cm.sum(0) + cm.sum(1) - cm.diag()).clamp(min=1)
    roof = cm[1:, 1:].sum() / cm[1:, :].sum().clamp(min=1)           # OA over reference roof pixels
    rb = cm[1:, 1:].sum() / (cm[1:, :].sum() + cm[0, 1:].sum())     # roof-vs-background IoU
    return {'miou4': iou[1:].mean().item(), 'iou': [round(v, 4) for v in iou.tolist()], 'roof_oa': roof.item(), 'roof_iou': rb.item()}


def main():
    source, kind = sys.argv[1], sys.argv[2]
    epochs = int(sys.argv[3]) if len(sys.argv) > 3 else 200
    seed = int(sys.argv[4]) if len(sys.argv) > 4 else 1
    fusion = kind == 'fusion'
    out = os.path.join(RUNS, f'{source}_{kind}_s{seed}'); os.makedirs(out, exist_ok=True)
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    dev = torch.device('cuda')
    torch.backends.cudnn.benchmark = True

    tr = TileDataset(read_split('training'), source, fusion, augment=True)
    va = TileDataset(read_split('validation'), source, fusion, augment=False)
    g = torch.Generator(); g.manual_seed(seed)
    tl = torch.utils.data.DataLoader(tr, batch_size=8, shuffle=True, drop_last=True, num_workers=2, persistent_workers=True,
                                     pin_memory=True, generator=g, worker_init_fn=_winit)
    vl = torch.utils.data.DataLoader(va, batch_size=16, shuffle=False, num_workers=0)

    net = SegNet(5, SOURCES[source], fusion).to(dev).to(memory_format=torch.channels_last)
    opt = torch.optim.Adam(net.parameters(), lr=5e-4)
    crit = LovaszLoss2d()
    hist, best = [], -1
    json.dump({'source': source, 'kind': kind, 'epochs': epochs, 'seed': seed, 'n_train': len(tr), 'n_val': len(va)}, open(f'{out}/config.json', 'w'))
    for ep in range(1, epochs + 1):
        t0 = time.time(); net.train(); tl_sum = 0; cm = torch.zeros(5, 5, dtype=torch.long, device=dev)
        for img, st, lab, _ in tl:
            img = img.to(dev, non_blocking=True).to(memory_format=torch.channels_last); lab = lab.to(dev, non_blocking=True)
            st = st.to(dev, non_blocking=True) if fusion else None
            with torch.autocast('cuda', dtype=torch.bfloat16):
                o = net(img, st)
            loss = crit(o.float(), lab)
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
            tl_sum += loss.item() * img.size(0)
            cm += confusion(o.detach().argmax(1), lab)
            nb = getattr(main, 'nb', 0) + 1; main.nb = nb
            if ep == 1 and nb % 100 == 0: print(f'  batch {nb} {time.time() - t0:.0f}s', flush=True)
        trs = scores(cm)
        net.eval(); vl_sum = 0; cm = torch.zeros(5, 5, dtype=torch.long, device=dev)
        with torch.no_grad():
            for img, st, lab, _ in vl:
                img = img.to(dev).to(memory_format=torch.channels_last); lab = lab.to(dev); st = st.to(dev) if fusion else None
                with torch.autocast('cuda', dtype=torch.bfloat16):
                    o = net(img, st)
                vl_sum += crit(o.float(), lab).item() * img.size(0)
                cm += confusion(o.argmax(1), lab)
        vs = scores(cm)
        h = {'epoch': ep, 'train_loss': tl_sum / len(tr), 'val_loss': vl_sum / len(va), 'train_miou4': trs['miou4'], 'val_miou4': vs['miou4'],
             'train_roof_iou': trs['roof_iou'], 'val_roof_iou': vs['roof_iou'], 'val_roof_oa': vs['roof_oa'],
             **{f'val_iou{c}': vs['iou'][c] for c in range(5)}, 'sec': time.time() - t0}
        hist.append(h)
        if vs['miou4'] > best:
            best = vs['miou4']; torch.save({'epoch': ep, 'state_dict': net.state_dict(), 'val': vs}, f'{out}/best.pth')
        with open(f'{out}/history.csv', 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=list(h)); w.writeheader(); w.writerows(hist)
        print(f'{source} {kind} ep {ep}/{epochs} loss {h["train_loss"]:.4f}/{h["val_loss"]:.4f} val mIoU4 {vs["miou4"]:.3f} roofIoU {vs["roof_iou"]:.3f} best {best:.3f} {h["sec"]:.0f}s', flush=True)
    torch.save({'epoch': epochs, 'state_dict': net.state_dict(), 'val': vs}, f'{out}/final.pth')
    open(f'{out}/DONE', 'w').write('ok')


if __name__ == '__main__':
    main()
