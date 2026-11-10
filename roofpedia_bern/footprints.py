"""Aggregate predicted masks to building footprints: coverage fractions and area-based labels.

python -m roofpedia_bern.footprints --masks masks --footprints buildings.gpkg --out roofs.gpkg [--tau 0.05]

For every footprint b, the coverage fraction of class k is p_b,k = (# mask pixels of class k inside b) / (# mask pixels inside b).
A footprint is a detected flat roof if p_G + p_PG + p_S + p_NS > tau (default 0.05); its area-based label is the class with the
largest fraction. Output columns: px (pixels inside the footprint), p_G, p_PG, p_S, p_NS, roof_frac, label (0 = not detected,
1 Green, 2 Potential Green, 3 Solar, 4 Not Suitable), label_name, margin (difference between the two largest fractions;
small margins indicate mixed or uncertain roofs).
"""
import argparse, os, numpy as np, geopandas as gpd, mercantile
from PIL import Image
from rasterio.features import rasterize
from rasterio.transform import from_bounds
from .predict import list_tiles

NAMES = ['not detected', 'Green', 'Potential Green', 'Solar', 'Not Suitable']


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--masks', required=True); ap.add_argument('--footprints', required=True); ap.add_argument('--out', required=True)
    ap.add_argument('--tau', type=float, default=0.05)
    a = ap.parse_args()
    fp = gpd.read_file(a.footprints).reset_index(drop=True)
    geo = fp.to_crs('EPSG:3857')
    sindex = geo.sindex
    cnt = np.zeros((len(fp), 5), np.int64)
    tiles = list_tiles(a.masks)
    for z, x, y, f in tiles:
        b = mercantile.xy_bounds(mercantile.Tile(x, y, z))
        ids = list(sindex.intersection((b.left, b.bottom, b.right, b.top)))
        if not ids: continue
        m = np.asarray(Image.open(f)).astype(np.int64); m[m > 4] = 0
        h, w = m.shape
        ras = rasterize(((geo.geometry.iloc[i], i + 1) for i in ids), out_shape=(h, w),
                        transform=from_bounds(b.left, b.bottom, b.right, b.top, w, h), fill=0, dtype='int32')
        inside = ras > 0
        if inside.any(): np.add.at(cnt, (ras[inside] - 1, m[inside]), 1)
    px = cnt.sum(1); frac = cnt[:, 1:5] / np.maximum(px, 1)[:, None]
    roof = frac.sum(1); det = roof > a.tau
    lab = np.where(det, frac.argmax(1) + 1, 0)
    srt = np.sort(frac, 1); margin = srt[:, -1] - srt[:, -2]
    fp['px'] = px
    for k, c in enumerate(['p_G', 'p_PG', 'p_S', 'p_NS']): fp[c] = frac[:, k].round(4)
    fp['roof_frac'] = roof.round(4); fp['label'] = lab; fp['label_name'] = [NAMES[v] for v in lab]; fp['margin'] = margin.round(4)
    fp.to_file(a.out)
    print(f'{len(fp)} footprints, {int((px > 0).sum())} covered by tiles, {int(det.sum())} detected flat roofs:',
          {NAMES[k]: int((lab == k).sum()) for k in range(1, 5)})


if __name__ == '__main__':
    main()
