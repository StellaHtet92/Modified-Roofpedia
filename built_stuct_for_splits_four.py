"""
Build the struct/ folder for EACH existing split (training, validation, evaluation)
using the RGB tiles already present in each split's images/ folder.

Struct channels (4):
    0 = nDSM        (height above ground, metres)
    1 = slope       (degrees)
    2 = aspect_sin  (sin of roof aspect; 0 on flat roofs)
    3 = aspect_cos  (cos of roof aspect; 0 on flat roofs)

Aspect is encoded as sin/cos because it is circular (0 deg == 360 deg). Flat roofs
have no meaningful orientation, so both aspect channels are set to 0 there (a zero
vector), which is distinct from any real facing direction.

You already have:
    dataset/<split>/images/<z>/<x>/<y>.png
    dataset/<split>/labels/<z>/<x>/<y>.png

This script adds, for each split:
    dataset/<split>/struct/<z>/<x>/<y>.npy      shape (4, 256, 256) float32

It does NOT re-split anything. It reads the Z/X/Y of each existing image tile,
samples the rasters at that exact tile footprint, and writes a matching .npy.

REQUIREMENTS
------------
    pip install rasterio numpy mercantile

Provide three full-coverage rasters over the study area (any georeferenced CRS):
    NDSM_RASTER   : nDSM (DSM - DTM), metres
    SLOPE_RASTER  : slope in degrees      (gdaldem slope  dsm.tif slope.tif  -alg Horn -compute_edges)
    ASPECT_RASTER : aspect in degrees     (gdaldem aspect dsm.tif aspect.tif -zero_for_flat -alg Horn -compute_edges)

FLAT_SLOPE_DEG below controls which pixels are treated as flat (aspect zeroed).
"""

import os
import glob
import numpy as np
import rasterio
from rasterio.warp import reproject, Resampling
from rasterio.transform import from_bounds
import mercantile


# ------------------------------------------------------------------ CONFIG ----
DATASET_ROOT = r"D:\DeepLearning\Roofpedia_IEEE_Paper\Modified_Roofpedia_fusion_with_mapbox\dataset"
SPLITS       = ["training", "validation", "evaluation"]   # existing split folders

NDSM_RASTER  = r"D:\DeepLearning\Roofpedia_IEEE_Paper\dataset\nDSM\nDSM_Clip.tif"
SLOPE_RASTER = r"D:\DeepLearning\Roofpedia_IEEE_Paper\dataset\Slope_Swisssurface3d_2024_clip.tif"
ASPECT_RASTER = r"D:\DeepLearning\Roofpedia_IEEE_Paper\dataset\Aspect_Swisssurface3d_2024_clip.tif"

TILE_SIZE     = 256
ZOOM          = 19
FLAT_SLOPE_DEG = 3.0     # roofs below this slope are treated as flat (aspect -> 0 vector)
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


def find_image_tiles(images_root, zoom):
    """Yield (x, y, path) for each RGB tile under images_root/zoom/x/y.png."""
    pattern = os.path.join(images_root, str(zoom), "*", "*.png")
    for path in glob.glob(pattern):
        y = int(os.path.splitext(os.path.basename(path))[0])
        x = int(os.path.basename(os.path.dirname(path)))
        yield x, y, path


def process_split(split, ndsm_ds, slope_ds, aspect_ds):
    images_root = os.path.join(DATASET_ROOT, split, "images")
    struct_root = os.path.join(DATASET_ROOT, split, "struct")

    if not os.path.isdir(images_root):
        print(f"[{split}] images folder not found, skipping: {images_root}")
        return

    tiles = list(find_image_tiles(images_root, ZOOM))
    print(f"[{split}] {len(tiles)} image tiles found.")

    written = 0
    for i, (x, y, _) in enumerate(tiles, 1):
        bounds = tile_bounds_3857(x, y, ZOOM)

        ndsm   = sample_raster_to_tile(ndsm_ds,   bounds, TILE_SIZE)
        slope  = sample_raster_to_tile(slope_ds,  bounds, TILE_SIZE)
        aspect = sample_raster_to_tile(aspect_ds, bounds, TILE_SIZE)

        ndsm  = np.nan_to_num(ndsm,  nan=0.0)
        slope = np.nan_to_num(slope, nan=0.0)
        aspect = np.nan_to_num(aspect, nan=0.0)

        # Encode circular aspect as sin/cos; zero out flat roofs (no orientation).
        rad = np.deg2rad(aspect)
        aspect_sin = np.sin(rad).astype(np.float32)
        aspect_cos = np.cos(rad).astype(np.float32)
        flat = slope < FLAT_SLOPE_DEG
        aspect_sin[flat] = 0.0
        aspect_cos[flat] = 0.0

        struct = np.stack([ndsm, slope, aspect_sin, aspect_cos], axis=0).astype(np.float32)

        out_dir = os.path.join(struct_root, str(ZOOM), str(x))
        os.makedirs(out_dir, exist_ok=True)
        np.save(os.path.join(out_dir, f"{y}.npy"), struct)
        written += 1

        if i % 200 == 0:
            print(f"  [{split}] {i}/{len(tiles)}...")

    print(f"[{split}] done. wrote {written} struct patches to {struct_root}")


def main():
    ndsm_ds   = rasterio.open(NDSM_RASTER)
    slope_ds  = rasterio.open(SLOPE_RASTER)
    aspect_ds = rasterio.open(ASPECT_RASTER)
    try:
        for split in SPLITS:
            process_split(split, ndsm_ds, slope_ds, aspect_ds)
    finally:
        ndsm_ds.close()
        slope_ds.close()
        aspect_ds.close()
    print("All splits processed. Struct channels: [nDSM, slope, aspect_sin, aspect_cos]")


if __name__ == "__main__":
    main()