# Modified-Roofpedia

### Structure-aware four-class rooftop mapping for Swiss cities

> A pipeline building on [Roofpedia](https://github.com/ualsg/Roofpedia). It maps existing green roofs, solar roofs, potential green roofs and unsuitable flat roofs from orthophotos, and fuses the imagery with elevation-derived layers from swisstopo.

---

## Overview

Modified-Roofpedia extends the Roofpedia segmentation workflow in four ways.

1. **Four roof classes in one network.** Roofpedia uses separate binary networks for green and solar roofs. This pipeline predicts background plus four classes in a single U-Net:
   - **Green (G):** existing vegetated roofs (extensive or intensive).
   - **Potential Green (PG):** bare flat roofs, such as gravel or bitumen, on which greening is plausible.
   - **Solar (S):** flat roofs with PV or solar-thermal installations.
   - **Not Suitable (NS):** flat roofs judged unsuitable from visible material or use, such as glass roofs, terraces, parking decks or dense technical equipment.
2. **Structural fusion.** A second encoder processes four elevation-derived layers: nDSM, slope, sin(aspect) and cos(aspect). Its features are fused with the image features at the bottleneck of the U-Net.
3. **Per-building coverage fractions.** Each building footprint stores the predicted share of every class, in addition to a single dominant-class label.
4. **Spatially blocked evaluation.** Training, validation and evaluation tiles are separated by spatial blocks, so accuracy is reported on unseen areas.

> **Change from the previous version:** the slope filter is no longer applied. Slope, aspect and nDSM are now model inputs. 

## What's new compared to Roofpedia

| Aspect | Roofpedia | Modified-Roofpedia |
|---|---|---|
| Output | Two binary networks (green, solar) | One network, background + 4 classes |
| Inputs | RGB imagery | RGB (or RGB + NIR) imagery + nDSM, slope, aspect |
| Architecture | U-Net, ResNet-50 encoder | Dual-encoder U-Net: ResNet-50 image encoder + ResNet-18 structural encoder, bottleneck fusion |
| Building label | Binary | Dominant class + per-class coverage fractions |
| Evaluation | Random tile split | Spatially blocked split (8 × 8 tile blocks) |
| Geographic focus | Global cities | Swiss cities (tested in Bern) |

## Model architecture

![Fusion network architecture](fusion_unet_architecture.png)

- **Image encoder:** ResNet-50 pretrained on ImageNet. For four-band input (RGB + NIR), the first convolution is widened and the NIR filters are initialized with the mean of the pretrained RGB filters.
- **Structural encoder:** ResNet-18 pretrained on ImageNet, with a four-channel first convolution initialized the same way. A 1 × 1 convolution projects its 512-channel output to 2,048 channels.
- **Fusion block:** the image and structural bottleneck features (each 2,048 × 8 × 8 for a 256 × 256 tile) are concatenated and passed through two 3 × 3 Conv-BN-ReLU blocks (4,096 → 2,048 → 2,048).
- **Decoder:** U-Net decoder with Conv-BN-ReLU blocks. Skip connections come from the image encoder only.
- **Output:** 1 × 1 convolution to 5 channels (background + 4 classes), argmax per pixel.
- **Parameters:** 39.4 M image-only, 164.9 M fusion.

The image-only baseline is the same network without the structural encoder and fusion block.

## Data

| Dataset | Use | Access |
|---|---|---|
| [SWISSIMAGE 10 cm](https://www.swisstopo.admin.ch/en/orthoimage-swissimage-10) | RGB imagery (recommended) | Open (OGD) |
| [SWISSIMAGE RS](https://www.swisstopo.admin.ch/en/orthoimage-swissimage-rs) | NIR-RGB imagery (optional) | On request from swisstopo, cost offer |
| Mapbox satellite tiles | RGB imagery (optional) | API key, free monthly quota |
| [swissTLM3D](https://www.swisstopo.admin.ch/en/landscape-model-swisstlm3d) | Building footprints | Open (OGD) |
| [swissSURFACE3D Raster](https://www.swisstopo.admin.ch/en/height-model-swisssurface3d-raster) | DSM (0.5 m) | Open (OGD) |
| [swissALTI3D](https://www.swisstopo.admin.ch/en/height-model-swissalti3d) | DTM (0.5 m) | Open (OGD) |

SWISSIMAGE 10 cm with the swisstopo elevation models is the only configuration that is fully reproducible from open data.

**Note on SWISSIMAGE RS:** the delivered GeoTIFFs tag the fourth band (blue) as an alpha channel. GIS tile exporters then treat it as transparency, which produces washed-out tiles and drops the NIR band. Read every band as data and mosaic on the native 0.1 m grid before warping to the tile grid.

### Structural layers

- nDSM = swissSURFACE3D DSM − swissALTI3D DTM
- Slope and aspect from the DSM (Horn's method)
- Aspect encoded as sin(aspect) and cos(aspect)
- The four layers are resampled bilinearly to each image tile and standardized with per-channel statistics from the training tiles.

### Tiles

All inputs and labels use 256 × 256 slippy-map tiles at zoom level 19 (about 0.2 m per pixel in Bern).

## Installation

Python ≥ 3.11:

```
pip install -r requirements.txt
```

Training used one NVIDIA RTX 5090 (32 GB). Inference also runs on a CPU.

## Usage

Every pipeline script reads its paths from `config.toml`. Copy `config.example.toml` to
`config.toml` and set them before running anything; `config.toml` is ignored by git, so no
local path or API key ends up in the repository. All commands are run from the repository
root, and everything the pipeline writes goes below `work_dir`.

The keys referred to below are `label_raster`, `ndsm`, `slope`, `aspect`,
`swissimage_tiles`, `mapbox_tiles`, `rs_geotiffs`, `footprints` and `reference_table`.
`mapbox_tiles` and `rs_geotiffs` are optional: leave them out to reproduce the SWISSIMAGE
configurations only. A longer walkthrough, including what each step writes and why, is in
[docs/reproducing.md](docs/reproducing.md).

### 1. Prepare imagery tiles

The RGB tile pyramids are an input, not an output: this repository does not download them.
Obtain zoom-19 XYZ tiles of 256 × 256 pixels in EPSG:3857 and point `swissimage_tiles`, and
optionally `mapbox_tiles`, at the folder that contains the `<x>` folders, so that individual
tiles are at `<dir>/<x>/<y>.png`.

SWISSIMAGE RS needs no pre-tiling. `pipeline/rs_render.py` reads the delivered four-band
GeoTIFFs band by band, mosaics them on their native 0.1 m grid and warps each tile
bilinearly to the tile grid, which is what avoids the alpha-channel problem described above.

```
python pipeline/prep_data.py
```

Under `work_dir/data/` this writes `swissimage/19/<x>/<y>.png` and, if configured,
`mapbox/19/<x>/<y>.png`; `rs/19/<x>/<y>.npy`, uint8 arrays of shape (4, 256, 256) in the
order R, G, B, NIR; and `tiles_all_labelled.csv`, one row per tile with its per-class pixel
counts and a `keep` flag. A tile is kept only when it contains labelled roof pixels and is
fully covered by every configured imagery source, so all configurations see the same tiles.

### 2. Prepare structural layers

The same script builds them, from the `ndsm`, `slope` and `aspect` rasters. Produce those
three beforehand: the nDSM is the DSM minus the DTM in metres, and slope and aspect come
from the DSM by Horn's method.

```
gdaldem slope  dsm.tif slope.tif  -alg Horn -compute_edges
gdaldem aspect dsm.tif aspect.tif -alg Horn -zero_for_flat -compute_edges
```

Each raster is warped bilinearly onto the tile grid. Aspect is circular, so it is stored as
its sine and cosine rather than as degrees, and both are set to zero wherever the slope is
below `FLAT_SLOPE_DEG`, which is 6.5° in `pipeline/config.py`, because the aspect of a flat
surface is undefined and would otherwise inject noise. That constant only suppresses an
undefined aspect; it is not a flat-roof filter, and no slope threshold is applied anywhere
in this pipeline.

The result is `data/struct/19/<x>/<y>.npy`, float32 of shape (4, 256, 256), in the channel
order nDSM, slope, sin(aspect), cos(aspect).

### 3. Prepare labels and the spatial split

`prep_data.py` also renders the label tiles, straight from `label_raster` by nearest
neighbour into `data/labels/19/<x>/<y>.png`, so the class codes are identical in every tile:
0 background, 1 Green, 2 Potential Green, 3 Solar, 4 Not Suitable.

```
python pipeline/split.py
```

This groups the kept tiles into blocks of 8 × 8 zoom-19 tiles, about 420 m on a side, and
assigns whole blocks to training, validation and evaluation in a 70:15:15 ratio, so no block
is split between training and test data. It draws 500 candidate assignments, keeps only
those whose split fractions are within 1.5 percentage points of the target, and among those
keeps the one whose per-class pixel shares deviate least from the overall shares.

Two files are written. `data/split.csv` has one row per tile with `x`, `y`, `split` and
`block`. `data/norm_stats.json` holds the NIR mean and standard deviation and the per-channel
means and standard deviations of the structural layers, estimated from 1,500 randomly chosen
training tiles, plus the block size and the chosen seed. Statistics come from training tiles
only, so nothing crosses into the held-out blocks.

### 4. Train

One configuration:

```
python pipeline/train.py <source> <kind> [epochs] [seed]
```

`source` is `swissimage`, `mapbox` or `rs`; `kind` is `rgb` for the image-only baseline or
`fusion` for the structure-aware network. `epochs` defaults to 200 and `seed` to 1.

All six configurations of the study, each followed automatically by prediction on the
training, validation and evaluation splits:

```
python pipeline/run_all.py 200 2 1
```

The arguments are epochs, jobs to run concurrently on the GPU, and seed. A trailing list
restricts the set, so with SWISSIMAGE data only:

```
python pipeline/run_all.py 200 2 1 swissimage:rgb swissimage:fusion
```

Each run writes `runs/<source>_<kind>_s<seed>/`, containing `config.json` with the settings
and the parameter count, `history.csv` with per-epoch losses and validation metrics,
`best.pth` and `final.pth`, `train.log`, a `DONE` marker, and the `pred_<split>/` folders.
Repeating with seeds 2 and 3 gives run-to-run variability; `pipeline/evaluate.py s2` then
evaluates that seed.

Training settings used in the study:

| Setting | Value |
|---|---|
| Optimizer | Adam, learning rate 5 × 10⁻⁴ |
| Batch size | 8 |
| Epochs | 200 |
| Loss | Binary Lovász hinge (as in the Roofpedia implementation) |
| Augmentation | Random 90° rotations and flips (8 variants), applied jointly to image, structural layers and mask |
| Model selection | Highest validation mIoU over the four roof classes |

The loss is evaluated once per image on the flattened one-hot stack of all five output
channels, on the raw scores, with no softmax or sigmoid applied and no class ignored, and the
per-image values are averaged over the batch.

On Windows each DataLoader worker reserves about 3 GB of virtual memory, so `train.py` uses
two workers and `run_all.py` should run at most two jobs in parallel on a 64 GB machine.

### 5. Predict and assign building labels

Prediction for one run and split, writing paletted PNG masks with the same class codes as the
labels:

```
python pipeline/predict.py <run_name> [split=evaluation] [ckpt=best]
```

`run_all.py` already does this for the three splits, so run it by hand only for a different
checkpoint. City-wide prediction over every zoom-19 tile of the study area, with the
structural layers rendered on the fly from the same rasters and by the same rule as in
training:

```
python pipeline/predict_city.py swissimage_fusion_s1
```

Aggregation to buildings happens in two places. For the reference buildings,
`pipeline/evaluate.py` writes `eval/tile_eval.json` and one `eval/tilebld_<key>.npz` per
configuration, each holding the predicted and reference class pixel counts per footprint
together with flags marking the strictly held-out buildings. For every footprint in the study
area, `pipeline/city_counts.py` writes `eval/city_<key>.npy`:

```
python pipeline/evaluate.py
python pipeline/city_counts.py swissimage_fusion_s1 f_swissimage
python pipeline/city_summary.py
```

Metrics, tables and the spatial block bootstrap:

```
python pipeline/analysis.py
python pipeline/pixel_boot.py
python pipeline/slope_zonal.py
```

For each footprint the pipeline stores:

- **Coverage fractions** `p_G, p_PG, p_S, p_NS`: share of predicted pixels of each class inside the footprint.
- **Detected flag:** a building counts as detected if its total roof-class coverage exceeds τ = 0.05.
- **Area-based label:** the class with the largest coverage fraction (recommended).
- **Priority-based label (optional):** the first class present in the order Solar > Potential Green > Green > Not Suitable.

Use the area-based label for inventories and class statistics. The priority rule overstates Solar and Potential Green area and should only be used to answer questions such as "which roofs carry any PV?".

## Results (Bern, held-out evaluation blocks)

Pixel level, 1,346 held-out tiles:

| Model | OA | mIoU | κ |
|---|---|---|---|
| Image-only, SWISSIMAGE | 0.593 | 0.398 | 0.670 |
| Fusion, SWISSIMAGE | 0.601 | 0.426 | 0.708 |
| Image-only, SWISSIMAGE RS | 0.590 | 0.419 | 0.689 |
| Fusion, SWISSIMAGE RS | 0.603 | 0.421 | 0.710 |
| Image-only, Mapbox | 0.591 | 0.411 | 0.678 |
| Fusion, Mapbox | 0.602 | 0.426 | 0.705 |

Building level, 1,186 held-out buildings: overall accuracy 65.0% to 67.5% for all six configurations. Fusion improves pixel-level κ for every imagery source, but the building-level difference is not statistically significant.

City-scale application (SWISSIMAGE fusion network) flagged 12,611 candidate flat-roof footprints: 4,489 Green, 3,894 Potential Green, 999 Solar and 3,229 Not Suitable.

## Limitations

- **Potential Green is a screening class.** Load-bearing capacity is not available as open data and is not considered. Candidates need a structural assessment.
- **Trained on flat roofs only.** Behaviour on pitched roofs is not characterized. Detections outside the reference set are not verified.
- **One city.** The model was trained and evaluated in Bern only.
- **Coverage fractions** have limited validation, because most reference roofs are labelled with a single class.

## Pretrained models

Please contact [Htet Yamin Ko Ko](mailto:htetyaminkokoedu@gmail.com) for the trained models and supplementary data.

## Citation

The paper describing this pipeline is under review. Until it appears, please cite it as:

```bibtex
@unpublished{kokoStructureAware,
  author = {Ko Ko, Htet Yamin},
  title  = {Structure-Aware Multimodal Fusion for Four-Class Green Roof Potential
            Assessment from Open Swiss Geospatial Data},
  note   = {Manuscript under review, IEEE Journal of Selected Topics in Applied Earth
            Observations and Remote Sensing},
  year   = {2026}
}
```

## Acknowledgements

This project builds on:

- **Roofpedia** by [Abraham Noah Wu](https://ual.sg/authors/abraham/) and [Filip Biljecki](https://ual.sg/authors/filip/), Urban Analytics Lab, National University of Singapore.
  Wu AN, Biljecki F (2021): Roofpedia: Automatic mapping of green and solar roofs for an open roofscape registry and evaluation of urban sustainability. *Landscape and Urban Planning* 214: 104167. [doi:10.1016/j.landurbplan.2021.104167](https://doi.org/10.1016/j.landurbplan.2021.104167)
- **SWISSIMAGE, swissTLM3D, swissSURFACE3D and swissALTI3D** by [swisstopo](https://www.swisstopo.admin.ch/), the Swiss Federal Office of Topography.
- The **Swiss Territorial Data Lab (STDL)** for providing SWISSIMAGE RS data for Bern.

### Packages

- [PyTorch](https://pytorch.org/)
- [GeoPandas](https://geopandas.org/)

## License

Distributed under the MIT License. See `LICENSE` for more information.

## Contact

Htet Yamin Ko Ko, International Space Science Institute (ISSI), Bern, Switzerland. htetyaminkokoedu@gmail.com
