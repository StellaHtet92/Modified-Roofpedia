"""
Extract 4-channel structural patches (nDSM, slope, aspect_sin, aspect_cos) that
spatially match a set of RGB slippy-map tiles. Use this for PREDICTION tiles, where
there is no train/val/eval split, just a folder of RGB tiles for a city.

Struct channels (must match training):
    0 = nDSM        (height above ground, metres)
    1 = slope       (degrees)
    2 = aspect_sin  (sin of roof aspect; 0 on flat roofs)
    3 = aspect_cos  (cos of roof aspect; 0 on flat roofs)

RGB tiles are PNGs named by slippy Z/X/Y (Web Mercator, EPSG:3857):
    <RGB_TILE_ROOT>/19/<x>/<y>.png
Output:
    <STRUCT_OUT_ROOT>/19/<x>/<y>.npy      shape (4, 256, 256) float32

REQUIREMENTS
------------
    pip install rasterio numpy mercantile
"""

import os
import glob
import numpy as np
import rasterio
from rasterio.warp import reproject, Resampling
from rasterio.transform import from_bounds
import mercantile


# ------------------------------------------------------------------ CONFIG ----
RGB_TILE_ROOT   = r"D:\DeepLearning\Roofpedia_IEEE_Paper\Modified_Roofpedia_fusion_with_SwissImage\results\02Images\Bern_Swisstopo"
STRUCT_OUT_ROOT = r"D:\DeepLearning\Roofpedia_IEEE_Paper\Modified_Roofpedia_fusion_with_SwissImage\results\05Struct\Bern_Swisstopo"

NDSM_RASTER   = r"D:\DeepLearning\Roofpedia_IEEE_Paper\dataset\nDSM\nDSM_Clip.tif"
SLOPE_RASTER  = r"D:\DeepLearning\Roofpedia_IEEE_Paper\dataset\Slope_Swisssurface3d_2024_clip.tif"
ASPECT_RASTER = r"D:\DeepLearning\Roofpedia_IEEE_Paper\dataset\Aspect_Swisssurface3d_2024_clip.tif"

TILE_SIZE      = 256
ZOOM           = 19
FLAT_SLOPE_DEG = 6.5     # match training (reality it is <6.5)
# -----------------------------------------------------------------------------


def tile_bounds_3857(x, y, z):
    b = mercantile.xy_bounds(mercantile.Tile(x=x, y=y, z=z))
    return b.left, b.bottom, b.right, b.top


def sample_raster_to_tile(src_ds, bounds_3857, size):
    xmin, ymin, xmax, ymax = bounds_3857
    dst = np.full((size, size), np.nan, dtype=np.float32)
    dst_transform = from_bounds(xmin, ymin, xmax, ymax, size, size)
    reproject(
        source=rasterio.band(src_ds, 1),
        destination=dst,
        src_transform=src_ds.transform,
        src_crs=src_ds.crs,
        dst_transform=dst_transform,
        dst_crs="EPSG:3857",
        resampling=Resampling.bilinear,
        src_nodata=src_ds.nodata,
        dst_nodata=np.nan,
    )
    return dst


def find_rgb_tiles(root, zoom):
    pattern = os.path.join(root, str(zoom), "*", "*.png")
    for path in glob.glob(pattern):
        y = int(os.path.splitext(os.path.basename(path))[0])
        x = int(os.path.basename(os.path.dirname(path)))
        yield x, y, path


def main():
    ndsm_ds   = rasterio.open(NDSM_RASTER)
    slope_ds  = rasterio.open(SLOPE_RASTER)
    aspect_ds = rasterio.open(ASPECT_RASTER)

    tiles = list(find_rgb_tiles(RGB_TILE_ROOT, ZOOM))
    print(f"Found {len(tiles)} RGB tiles at zoom {ZOOM}.")
    if not tiles:
        print(f"WARNING: no tiles under {RGB_TILE_ROOT}\\{ZOOM}\\*\\*.png -- check RGB_TILE_ROOT.")

    written = 0
    try:
        for i, (x, y, _) in enumerate(tiles, 1):
            bounds = tile_bounds_3857(x, y, ZOOM)

            ndsm   = sample_raster_to_tile(ndsm_ds,   bounds, TILE_SIZE)
            slope  = sample_raster_to_tile(slope_ds,  bounds, TILE_SIZE)
            aspect = sample_raster_to_tile(aspect_ds, bounds, TILE_SIZE)

            ndsm   = np.nan_to_num(ndsm,   nan=0.0)
            slope  = np.nan_to_num(slope,  nan=0.0)
            aspect = np.nan_to_num(aspect, nan=0.0)

            rad = np.deg2rad(aspect)
            aspect_sin = np.sin(rad).astype(np.float32)
            aspect_cos = np.cos(rad).astype(np.float32)
            flat = slope < FLAT_SLOPE_DEG
            aspect_sin[flat] = 0.0
            aspect_cos[flat] = 0.0

            struct = np.stack([ndsm, slope, aspect_sin, aspect_cos], axis=0).astype(np.float32)

            out_dir = os.path.join(STRUCT_OUT_ROOT, str(ZOOM), str(x))
            os.makedirs(out_dir, exist_ok=True)
            np.save(os.path.join(out_dir, f"{y}.npy"), struct)
            written += 1

            if i % 200 == 0:
                print(f"  {i}/{len(tiles)}...")
    finally:
        ndsm_ds.close()
        slope_ds.close()
        aspect_ds.close()

    print(f"Done. Wrote {written} struct patches to {STRUCT_OUT_ROOT}")
    print("Struct channels: [nDSM, slope, aspect_sin, aspect_cos]")


if __name__ == "__main__":
    main()