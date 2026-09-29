"""Predict + extract for the dual-encoder FusionUNet (RGB + structural channels).

Counterpart to predict_and_extract.py, which only feeds RGB into the (original,
single-encoder) UNet. The "Green" checkpoint in checkpoint/Green-final.pth was
trained by train_fusionResUNet.py, i.e. it IS a FusionUNet expecting a matching
structural (nDSM/slope/aspect) patch alongside every RGB tile -- this script builds
those patches (if missing) and runs prediction accordingly.

Usage:
    python predict_and_extract_fusion.py <city> <type>

e.g.:
    python predict_and_extract_fusion.py Bern Green
"""

import os
import torch
import toml
import argparse

from src.predict_multimodal import predict
from src.struct_extract import build_struct_patches
from src.extract import intersection

parser = argparse.ArgumentParser()
parser.add_argument("city", help="City to be predicted, must be the same as the name of the dataset")
parser.add_argument("type", help="Roof Typology, Green for Greenroof, Solar for PV Roof")
args = parser.parse_args()

# Anchored to this script's own folder, not the ambient working directory: bare
# relative paths only resolve correctly if the script happens to be launched with
# this folder as cwd. Run it from anywhere else (the workspace root, a task runner,
# an IDE with a different default cwd) and they silently look in the wrong place --
# which reads as "no RGB tiles found" / "config not found" even though everything is
# right there under this script's own folder. Anchoring to __file__ makes this
# robust regardless of where the script is invoked from.
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
config = toml.load(os.path.join(SCRIPT_DIR, "config", "predict-config.toml"))

city_name = args.city
target_type = args.type

tiles_dir = os.path.join(SCRIPT_DIR, "results", "02Images", city_name)
struct_dir = os.path.join(SCRIPT_DIR, "results", "05Struct", city_name)
mask_dir = os.path.join(SCRIPT_DIR, "results", "03Masks", target_type, city_name)
tile_size = config["img_size"]
zoom = config.get("zoom", 19)
struct_channels = config.get("struct_channels", 4)
overlap = config.get("overlap", 32)

# 1. Make sure every RGB tile for this city has a matching struct patch. Tiles that
#    already have one are skipped, so this is cheap to re-run.
written = build_struct_patches(
    tiles_dir, struct_dir,
    ndsm_raster=config["ndsm_raster"],
    slope_raster=config["slope_raster"],
    aspect_raster=config["aspect_raster"],
    zoom=zoom, tile_size=tile_size,
    flat_slope_deg=config.get("flat_slope_deg", 3.0),
)
if written:
    print(f"[struct] wrote {written} new struct patches to {struct_dir}")
else:
    print(f"[struct] struct patches already up to date in {struct_dir}")

# 2. Load the fusion checkpoint.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if target_type == "Solar":
    checkpoint_name = config["solar_checkpoint"]
elif target_type == "Green":
    checkpoint_name = config["green_checkpoint"]
else:
    raise ValueError(f"Unknown type '{target_type}', expected 'Green' or 'Solar'")

checkpoint_path = config["checkpoint_path"]
chkpt = torch.load(os.path.join(checkpoint_path, checkpoint_name), map_location=device, weights_only=False)

# 3. Predict. Struct normalisation stats are re-derived from the training struct/
#    folder with the same procedure (and RNG seed) used at training time, since the
#    checkpoint itself does not store them.
training_struct_root = os.path.join(config["dataset_path"], "training", "struct")

predict(
    tiles_dir, struct_dir, mask_dir, tile_size, device, chkpt,
    num_classes=config.get("num_classes", 5),
    struct_channels=struct_channels,
    training_struct_root=training_struct_root,
    overlap=overlap,
)

# 4. Same footprint intersection/extraction as the RGB-only pipeline.
intersection(target_type, city_name, mask_dir)
