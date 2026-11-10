"""Tile dataset shared by all six configurations (same tiles, labels, split and augmentation)."""
import csv, json, os, numpy as np, torch
from PIL import Image

import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config as C
DATA = C.DATA
IMNET_MEAN, IMNET_STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
SOURCES = {'mapbox': 3, 'swissimage': 3, 'rs': 4}


def read_split(split=None):
    rows = list(csv.DictReader(open(os.path.join(DATA, 'split.csv'))))
    return [(int(r['x']), int(r['y'])) for r in rows if split is None or r['split'] == split]


def load_image(source, x, y, root=DATA):
    if source == 'rs':
        return np.load(f'{root}/rs/19/{x}/{y}.npy')                                   # (4,H,W) uint8 R,G,B,NIR
    return np.asarray(Image.open(f'{root}/{source}/19/{x}/{y}.png').convert('RGB')).transpose(2, 0, 1)


def norm_stats():
    return json.load(open(os.path.join(DATA, 'norm_stats.json')))


class TileDataset(torch.utils.data.Dataset):
    def __init__(self, tiles, source, fusion, augment, with_label=True, root=DATA):
        self.tiles, self.source, self.fusion, self.augment, self.with_label, self.root = tiles, source, fusion, augment, with_label, root
        st = norm_stats()
        mean = IMNET_MEAN + ([st['nir_mean']] if source == 'rs' else [])
        std = IMNET_STD + ([st['nir_std']] if source == 'rs' else [])
        self.mean = np.array(mean, np.float32)[:, None, None]; self.std = np.array(std, np.float32)[:, None, None]
        self.smean = np.array(st['struct_mean'], np.float32)[:, None, None]; self.sstd = np.array(st['struct_std'], np.float32)[:, None, None]

    def __len__(self):
        return len(self.tiles)

    def __getitem__(self, i):
        x, y = self.tiles[i]
        img = (load_image(self.source, x, y, self.root).astype(np.float32) / 255. - self.mean) / self.std
        arrs = [img]
        if self.fusion:
            arrs.append((np.load(f'{self.root}/struct/19/{x}/{y}.npy') - self.smean) / self.sstd)
        lab = np.asarray(Image.open(f'{self.root}/labels/19/{x}/{y}.png')).astype(np.int64) if self.with_label else np.zeros(img.shape[1:], np.int64)
        if self.augment:
            k = np.random.randint(4); hf = np.random.rand() < 0.5
            arrs = [np.rot90(a, k, axes=(1, 2)) for a in arrs]; lab = np.rot90(lab, k)
            if hf:
                arrs = [a[:, :, ::-1] for a in arrs]; lab = lab[:, ::-1]
        out = [torch.from_numpy(np.ascontiguousarray(a, dtype=np.float32)) for a in arrs]
        s = out[1] if self.fusion else torch.zeros(1)
        return out[0], s, torch.from_numpy(np.ascontiguousarray(lab)), torch.tensor([x, y])
