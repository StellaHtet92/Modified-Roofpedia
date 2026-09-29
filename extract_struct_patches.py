"""
Extract 2-channel structural patches (nDSM + slope) that spatially match the
existing RGB / label slippy-map tiles used by Roofpedia.

The RGB and label patches are PNGs named by slippy-map Z/X/Y (Web Mercator,
EPSG:3857), e.g.:
    tiles/rgb/19/272843/184394.png
    tiles/label/19/272843/184394.png

This script, for every RGB tile, computes that tile's ground bounding box in
EPSG:3857, samples the source DSM and slope rasters (in Swiss CRS EPSG:2056)
into that exact box at 256x256 with bilinear resampling, stacks them into a
2-channel array, and writes:
    tiles/struct/19/272843/184394.npy      (shape: 2 x 256 x 256, float32)

Channel 0 = nDSM (height above ground, metres)
Channel 1 = slope (degrees)

REQUIREMENTS
------------
    pip install rasterio numpy mercantile

You must provide, as full-coverage rasters over the study area:
    - nDSM raster  (DSM - DTM, i.e. building height above ground), OR a DSM if
      you want to treat raw DSM as channel 0 (not recommended, see note below).
    - slope raster (degrees), computed from the DSM.

Both may be in any CRS with a valid .tif georeference; the script reprojects on
the fly. If they are already in EPSG:3857 it still works.

NOTE ON nDSM
------------
Raw DSM elevation is dominated by terrain and is a poor model input. Prefer
nDSM = DSM - DTM (needs swissALTI3D). If you only have the DSM, you can still
run this by pointing NDSM_RASTER at the DSM, but expect weaker results. See the
companion preprocessing script to build nDSM and slope first.
"""

import os
import glob
import numpy as np
import rasterio
from rasterio.warp import reproject, Resampling
from rasterio.transform import from_bounds
import mercantile


# ------------------------------------------------------------------ CONFIG ----
RGB_TILE_ROOT = r"D:/DeepLearning/Roofpedia_IEEE_Paper/Modified_Roofpedia_fusion_with_mapbox/dataset/images"     # has 19/X/Y.png
STRUCT_OUT_ROOT = r"D:/DeepLearning/Roofpedia_IEEE_Paper/Modified_Roofpedia_fusion_with_mapbox/dataset/ndsm_slope"  # will hold 19/X/Y.npy

NDSM_RASTER  = r"D:/DeepLearning/Roofpedia_IEEE_Paper/dataset/nDSM/nDSM_Clip.tif"   # metres above ground
SLOPE_RASTER = r"D:/DeepLearning/Roofpedia_IEEE_Paper/dataset/Slope_Swisssurface3d_2024_clip.tif"  # degrees

TILE_SIZE = 256          # output patch size in pixels (match RGB)
ZOOM      = 19           # slippy zoom level (from your filenames)
# -----------------------------------------------------------------------------


def tile_bounds_3857(x, y, z):
    """Return (xmin, ymin, xmax, ymax) of a slippy tile in EPSG:3857 (metres)."""
    b = mercantile.xy_bounds(mercantile.Tile(x=x, y=y, z=z))   # left,bottom,right,top
    return b.left, b.bottom, b.right, b.top


def sample_raster_to_tile(src_ds, bounds_3857, size):
    """Read `src_ds` into a `size`x`size` array covering `bounds_3857` (EPSG:3857).

    Reprojects/resamples on the fly with bilinear interpolation. Returns a float32
    array with NaN where there is no source data.
    """
    xmin, ymin, xmax, ymax = bounds_3857

    # Destination grid: size x size, aligned to the tile bounds, in EPSG:3857.
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
    """Yield (x, y, path) for every RGB tile png under root/zoom/x/y.png."""
    pattern = os.path.join(root, str(zoom), "*", "*.png")
    for path in glob.glob(pattern):
        # .../<zoom>/<x>/<y>.png
        y = int(os.path.splitext(os.path.basename(path))[0])
        x = int(os.path.basename(os.path.dirname(path)))
        yield x, y, path


def main():
    os.makedirs(STRUCT_OUT_ROOT, exist_ok=True)

    ndsm_ds  = rasterio.open(NDSM_RASTER)
    slope_ds = rasterio.open(SLOPE_RASTER)

    tiles = list(find_rgb_tiles(RGB_TILE_ROOT, ZOOM))
    print(f"Found {len(tiles)} RGB tiles at zoom {ZOOM}.")

    written, skipped = 0, 0
    for i, (x, y, rgb_path) in enumerate(tiles, 1):
        bounds = tile_bounds_3857(x, y, ZOOM)

        ndsm  = sample_raster_to_tile(ndsm_ds,  bounds, TILE_SIZE)
        slope = sample_raster_to_tile(slope_ds, bounds, TILE_SIZE)

        # Replace NaN (outside coverage) with 0 so the model gets finite input.
        # (Alternatively, keep a validity mask if you want to handle edges.)
        ndsm  = np.nan_to_num(ndsm,  nan=0.0)
        slope = np.nan_to_num(slope, nan=0.0)

        struct = np.stack([ndsm, slope], axis=0).astype(np.float32)  # (2, 256, 256)

        out_dir = os.path.join(STRUCT_OUT_ROOT, str(ZOOM), str(x))
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, f"{y}.npy")
        np.save(out_path, struct)
        written += 1

        if i % 200 == 0:
            print(f"  {i}/{len(tiles)} processed...")

    ndsm_ds.close()
    slope_ds.close()
    print(f"Done. Wrote {written} struct patches to {STRUCT_OUT_ROOT}. Skipped {skipped}.")


if __name__ == "__main__":
    main()
