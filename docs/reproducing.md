# Reproducing the paper, step by step

This is the long form of the table in the main README. Every command is run from the
repository root, and every path comes from `config.toml`.

Copy `config.example.toml` to `config.toml` first and set your paths. The keys used below
are:

| Key | What it points at |
|---|---|
| `work_dir` | the folder the pipeline writes into; `data/`, `runs/` and `eval/` are created inside it |
| `label_raster` | the reference raster, class codes 1 G, 2 PG, 3 S, 4 NS |
| `ndsm`, `slope`, `aspect` | the three elevation-derived rasters |
| `swissimage_tiles`, `mapbox_tiles` | existing zoom-19 XYZ tile pyramids, `<dir>/<x>/<y>.png` |
| `rs_geotiffs` | the folder of 4-band SWISSIMAGE RS GeoTIFFs (band order NIR, R, G, B) |
| `footprints`, `reference_table` | the swissTLM3D footprints and the per-footprint reference class |

`mapbox_tiles` and `rs_geotiffs` are optional. Leave them out to reproduce the SWISSIMAGE
configurations only.

## 1. Prepare imagery tiles

The RGB tile pyramids are an input, not an output: this repository does not download them.
Obtain zoom-19 XYZ tiles of 256 x 256 pixels in EPSG:3857 for SWISSIMAGE, and optionally
for Mapbox, and point `swissimage_tiles` and `mapbox_tiles` at the folder that contains the
`<x>` folders.

SWISSIMAGE RS is handled differently and needs no pre-tiling. `pipeline/rs_render.py` reads
the delivered GeoTIFFs band by band, mosaics them on their native 0.1 m grid and warps each
tile bilinearly to the tile grid. Reading the bands as data is what avoids the failure
described in the paper, where a GIS tile export treated the fourth band as an alpha channel,
blended the image with the background and discarded the near-infrared information.

```bash
python pipeline/prep_data.py
```

Under `work_dir/data/` this writes `swissimage/19/<x>/<y>.png` and, if configured,
`mapbox/19/<x>/<y>.png` as RGB copies, `rs/19/<x>/<y>.npy` as uint8 arrays of shape
(4, 256, 256) in the order R, G, B, NIR, and `tiles_all_labelled.csv`, which carries one row
per tile with its per-class pixel counts and a `keep` flag.

A tile is kept only when it contains labelled roof pixels and is fully covered by every
configured imagery source, so all configurations see exactly the same tiles.

## 2. Prepare structural layers

The same script builds the structural layers, from the `ndsm`, `slope` and `aspect` rasters.
Produce those three beforehand: the nDSM is the swissSURFACE3D DSM minus the swissALTI3D
DTM in metres, and slope and aspect come from the DSM by Horn's method, as recorded in the
comments of `config.example.toml`:

```bash
gdaldem slope  dsm.tif slope.tif  -alg Horn -compute_edges
gdaldem aspect dsm.tif aspect.tif -alg Horn -zero_for_flat -compute_edges
```

Each raster is warped bilinearly onto the tile grid. Aspect is a circular variable, so it is
stored as its sine and cosine rather than as degrees. Both are set to zero wherever the slope
is below `FLAT_SLOPE_DEG`, which is 6.5 degrees in `pipeline/config.py`, because the aspect
of a flat surface is undefined and would otherwise inject noise.

That 6.5 degree constant only suppresses an undefined aspect. It is not a flat-roof filter:
no slope threshold is applied anywhere in this pipeline, and the paper's slope analysis
explains why.

The result is `data/struct/19/<x>/<y>.npy`, float32 of shape (4, 256, 256), in the channel
order nDSM, slope, sin(aspect), cos(aspect).

## 3. Prepare labels and the spatial split

`prep_data.py` also renders the label tiles. They are rasterized straight from
`label_raster` by nearest neighbour into `data/labels/19/<x>/<y>.png`, so the class codes
are identical in every tile: 0 background, 1 Green, 2 Potential Green, 3 Solar, 4 Not
Suitable. No per-configuration code harmonization is needed anywhere downstream.

```bash
python pipeline/split.py
```

This groups the kept tiles into blocks of 8 x 8 zoom-19 tiles, about 420 m on a side, and
assigns whole blocks to training, validation and evaluation in a 70:15:15 ratio, so that no
block is split between a training and a test set. It draws 500 candidate assignments, keeps
only those whose split fractions are within 1.5 percentage points of the target, and among
those keeps the one whose per-class pixel shares deviate least from the overall shares.

Two files are written. `data/split.csv` has one row per tile with `x`, `y`, `split` and
`block`. `data/norm_stats.json` holds the near-infrared mean and standard deviation and the
per-channel means and standard deviations of the structural layers, estimated from 1,500
randomly chosen training tiles, together with the block size and the chosen seed. Statistics
come from training tiles only, so no information crosses into the held-out blocks.

## 4. Train

One configuration:

```bash
python pipeline/train.py <source> <kind> [epochs] [seed]
```

`source` is `swissimage`, `mapbox` or `rs`; `kind` is `rgb` for the image-only baseline or
`fusion` for the structure-aware dual-encoder network. `epochs` defaults to 200 and `seed`
to 1.

All six configurations of the paper, each followed automatically by prediction on the
training, validation and evaluation splits:

```bash
python pipeline/run_all.py 200 2 1
```

The arguments are epochs, number of jobs to run concurrently on the GPU, and seed. A trailing
list restricts the set, so with SWISSIMAGE data only:

```bash
python pipeline/run_all.py 200 2 1 swissimage:rgb swissimage:fusion
```

Repeating with seeds 2 and 3 gives run-to-run variability, and `evaluate.py s2` then
evaluates that seed.

Each run writes `runs/<source>_<kind>_s<seed>/` containing `config.json` with the settings
and the parameter count, `history.csv` with per-epoch losses and validation metrics,
`best.pth` and `final.pth`, `train.log`, a `DONE` marker, and the `pred_<split>/` folders.

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
channels, on the raw scores, with no softmax or sigmoid and no ignored class, and the
per-image values are averaged over the batch.

On Windows each DataLoader worker reserves about 3 GB of virtual memory, so `train.py` uses
two workers and `run_all.py` should run at most two jobs in parallel on a 64 GB machine.

## 5. Predict and assign building labels

Prediction for one run and split, writing paletted PNG masks with the same class codes as
the labels:

```bash
python pipeline/predict.py <run_name> [split=evaluation] [ckpt=best]
```

`run_all.py` already does this for the three splits, so run it by hand only for a different
checkpoint. City-wide prediction over every zoom-19 tile of the study area, with the
structural layers rendered on the fly from the same rasters and by the same rule as in
training:

```bash
python pipeline/predict_city.py swissimage_fusion_s1
```

Aggregation to buildings happens in two places. For the reference buildings:

```bash
python pipeline/evaluate.py
```

writes `eval/tile_eval.json` and one `eval/tilebld_<key>.npz` per configuration, each
holding the predicted and reference class pixel counts per footprint, together with flags
marking the strictly held-out buildings. For every footprint in the study area:

```bash
python pipeline/city_counts.py swissimage_fusion_s1 f_swissimage
python pipeline/city_summary.py
```

writes `eval/city_f_swissimage.npy`, an array of per-footprint class pixel counts, and
prints the city table.

The building label itself is a rule, not a model. For a footprint `b`, the coverage fraction
of class `k` is the number of pixels of that class whose centre falls inside the footprint,
divided by the number of pixels inside the footprint. A footprint counts as a detected flat
roof when the four roof-class fractions sum to more than τ = 0.05, which suppresses isolated
pixels, and its label is then the class with the largest fraction. Both the label and the
full fraction vector are kept, because the fractions retain partially greened roofs and small
PV installations that a single label hides. A priority rule, which assigns the first class
present in a fixed order, is implemented for comparison but is not recommended for deriving
statistics; the paper quantifies how far it inflates the Solar and Potential Green shares.

Metrics, tables and the block bootstrap:

```bash
python pipeline/analysis.py
python pipeline/pixel_boot.py
python pipeline/slope_zonal.py
```

## Not yet in this copy of the pipeline

Two experiments reported in the paper live in the authors' working tree and have not been
ported to `pipeline/` here. `pipeline/train.py` accepts only `<source> <kind> [epochs]
[seed]`, so it cannot currently reproduce them:

- the matched **DeepLabv3+** baseline, which needs an `arch` argument selecting the decoder;
- the **structural ablation**, which needs a mode selecting which structural channels reach
  the encoder and whether they still correspond to the image tile (real, shuffled, nDSM only,
  slope and aspect only).

Both are small additions to `train.py`, `models.py`, `data.py` and `run_all.py`.
