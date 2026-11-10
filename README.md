# Modified-Roofpedia: four-class flat-roof classification with structure-aware fusion

This repository classifies flat roofs from aerial orthophotos into **Green, Potential Green, Solar and Not Suitable** roofs. It is the code for:

> Htet Yamin Ko Ko, *Beyond Binary Rooftop Mapping: Structure-Aware Multimodal Fusion for Four-Class Green Roof Potential Assessment from Open Swiss Geospatial Data* (manuscript under review). [[AUTHOR: add citation and DOI]]

It extends Roofpedia (Wu & Biljecki) from two to four roof classes. Optionally, a second encoder fuses the orthophoto with a normalized DSM, slope and aspect. It also includes a building-level evaluation protocol and a complete, reproducible training and evaluation pipeline for Bern, Switzerland.

| | |
|---|---|
| **Use the pretrained models** | `roofpedia_bern/` + weights from the [GitHub release](../../releases). See [docs/inference_guide.md](docs/inference_guide.md). |
| **Reproduce the paper** | `pipeline/` + `figures/`. See below. |

## Repository layout

```
roofpedia_bern/        inference package: models, predict, make_struct, footprints
pipeline/              data preparation, training, prediction, evaluation (paths from config.toml)
figures/               figure scripts (training curves, label shares, architecture, label assignment)
sample/                7 held-out SWISSIMAGE tiles with reference labels and expected outputs
weights/model_cards.json   metadata of the released models (weights: see GitHub release)
docs/inference_guide.md    detailed guidance for applying the models to new data
config.example.toml    template for your local paths
```

## Installation

Python ≥ 3.11:

```bash
pip install -r requirements.txt
```

Training used one NVIDIA RTX 5090 (32 GB). Inference also runs on a CPU.

## Using the pretrained models

1. Download the weight files from the [release page](../../releases) into `weights/`.
2. Run:

```bash
python -m roofpedia_bern.predict --weights weights/swissimage_fusion.pth --tiles sample/tiles --struct sample/struct --out masks
python -m roofpedia_bern.footprints --masks masks --footprints buildings.gpkg --out roofs.gpkg
```

Masks are paletted PNGs with the class codes 0 background, 1 Green, 2 Potential Green, 3 Solar and 4 Not Suitable. Image viewers show them in the class colours. See [docs/inference_guide.md](docs/inference_guide.md) for:
- input requirements (zoom-19 XYZ tiles, structural layers);
- interpretation of the per-building fractions;
- fine-tuning;
- limitations.

## Reproducing the paper

Copy `config.example.toml` to `config.toml` and set the paths to your data:
- **Imagery:** SWISSIMAGE tiles, and optionally Mapbox tiles and the SWISSIMAGE RS GeoTIFFs.
- **Rasters:** the reference label raster, nDSM, slope and aspect.
- **Buildings:** the swissTLM3D footprints and the reference table.

All outputs are written below `work_dir`. Run the steps from the repository root:

| Step | Command | Output |
|---|---|---|
| 1. Tiles, labels, structural layers | `python pipeline/prep_data.py` | `data/` (labels rendered directly from the reference raster; RS bands read as data) |
| 2. Spatially blocked split | `python pipeline/split.py` | `data/split.csv`, `data/norm_stats.json` |
| 3. Train six networks + predict all splits | `python pipeline/run_all.py 200 2 1` | `runs/<source>_<rgb\|fusion>_s1/` |
| 4. Pixel and building evaluation | `python pipeline/evaluate.py` | `eval/tile_eval.json`, `eval/tilebld_*.npz` |
| 5. (optional) existing pipelines | `python pipeline/baseline_counts.py` | `eval/counts_*.npy` |
| 6. All tables and statistics | `python pipeline/analysis.py` and `python pipeline/pixel_boot.py` | `eval/pixel.csv`, `building.csv`, `binary.csv`, `analysis.json`, `pixel_boot.json` |
| 7. City-wide inventory | `python pipeline/predict_city.py swissimage_fusion_s1`, then `python pipeline/city_counts.py swissimage_fusion_s1 f_swissimage`, then `python pipeline/city_summary.py` | `eval/city_f_swissimage.npy` |
| 8. Slope statistics | `python pipeline/slope_zonal.py` | `eval/slope_stats.csv` |
| 9. Figures | `python figures/fig_training.py`, `fig_labelshares.py`, `fig_arch.py`, `fig4_label_assignment.py` | `eval/*.png`, `eval/*.pdf` |

`run_all.py <epochs> <parallel jobs> <seed> [source:kind ...]` trains the listed configurations. The default is all six. If you have no Mapbox or RS data, pass for example `swissimage:rgb swissimage:fusion`. Repeat with seeds 2 and 3 for run-to-run variability. `evaluate.py s2` evaluates another seed.

**Windows notes:**
- On Windows each DataLoader worker reserves about 3 GB of virtual memory. `train.py` therefore uses 2 workers, and `run_all.py` should run at most 2 jobs in parallel on a 64 GB machine.
- Prediction uses no worker processes.

**What the pipeline guarantees.** Every configuration uses:
- identical tiles, labels (one class coding), split, augmentation, decoder (Conv-BN-ReLU), loss and model selection;
- only tiles fully covered by every configured imagery source and free of no-data areas;
- the highest validation mIoU over the four roof classes to choose the checkpoint (`best.pth`).

## Data

| Data | Source | Access |
|---|---|---|
| SWISSIMAGE 10 cm, swissSURFACE3D, swissALTI3D, swissTLM3D | swisstopo | open government data |
| Mapbox satellite tiles | Mapbox Static Tiles API | Mapbox account and terms of service. No API key is stored in this repository; download the tiles with your own key and keep it out of version control (for example in an environment variable). |
| SWISSIMAGE RS (Bern subset) | Swiss Territorial Data Lab | research use, not redistributed here |
| Reference labels (9,898 buildings) | this study | [[AUTHOR: state availability]] |

## Licence

[[AUTHOR: choose a licence, e.g. MIT for the code and CC BY 4.0 for the weights. `pipeline/models.py` and `roofpedia_bern/models.py` adapt the Roofpedia / robosat U-Net; keep their copyright notice and licence terms.]]
