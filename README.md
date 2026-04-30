# Modified-Roofpedia

### Mapping Roofscapes in Switzerland with AI and 3D Surface Data

> **A modified pipeline building on [Roofpedia](https://github.com/ualsg/Roofpedia) - enhanced with Swiss surface 3D data for slope-based roof filtering and automated labelling of building polygons.**

---

## Overview

Modified-Roofpedia extends the original [Roofpedia](https://github.com/ualsg/Roofpedia) deep learning pipeline for detecting green roofs and solar panels from satellite imagery. The key modifications introduced in this project are:

1. **Slope-based roof filtering using Swiss surface 3D data** - The [swissSURFACE3D](https://www.swisstopo.admin.ch/en/height-model-swisssurface3d) dataset is used to compute roof slopes and filter out steep rooftops that are unsuitable for solar panel or green roof installations, improving prediction relevance.
2. **Automated labelling techniques** - A custom labelling pipeline assigns prediction labels (green rooftops/potential flat roofs for rooftop greenery/solar/flat rooftops which are not suitable to install greenery) to building polygons based on the segmentation masks, enabling scalable and reproducible annotation of building footprints.

The underlying model architecture (U-Net with 4 classes) and the slippy map tile prediction workflow remain consistent with the original Roofpedia.

## What's New Compared to Roofpedia

| Aspect | Roofpedia | Modified-Roofpedia |
|---|---|---|
| Roof filtering | None | Slope filtering via swissSURFACE3D point cloud |
| Label assignment | Binary (0 or 1) | Automated labelling method for building polygons based on the segmented areas |
| Geographic focus | 8 global cities | Swiss cities |
| 3D data integration | No | Yes (LiDAR-derived surface model) |

## Running Modified-Roofpedia

### 1. Prerequisites

Create a conda environment using the provided environment file:

```
conda env create -f environment.yml
```

For CPU-only systems:

```
conda env create -f environment_cpu.yml
```

### 2. Data Preparation

**Satellite tiles:** Prepare slippy map tiles at zoom level 19 and place them in `results/02Images/<CityName>/`.

**Building footprints:** Place GeoJSON building polygons in `results/01City/<CityName>.geojson`.

**Swiss surface 3D data:** Download the [swissSURFACE3D](https://www.swisstopo.admin.ch/en/height-model-swisssurface3d) dataset for your area of interest. This is used during the slope filtering step to exclude steep rooftops from the prediction pipeline.

**Pretrained weights:** Place checkpoint files in the configured checkpoint directory (see `config/predict-config.toml`).

### 3. Configuration

Edit `config/predict-config.toml` to set your parameters:

```toml
city_name = "Bern"
target_type = "Green"         
img_size = 512
checkpoint_path = "checkpoints"
solar_checkpoint = "solar_model.pth"
green_checkpoint = "green_model.pth"
```

### 4. Prediction

Run the prediction script:

```
python predict.py
```

Predicted masks will be saved to `results/03Masks/<target_type>/<CityName>/`.

### 5. Slope Filtering and Label Assignment

After prediction, run the slope filtering and labelling pipeline to:
- Filter out building polygons whose rooftops exceed a slope threshold (derived from swissSURFACE3D).
- Assign solar/green labels to the remaining building footprints based on mask overlap.

```
python label_buildings.py
```

Final labelled results are saved as GeoJSON in `results/04Results/`.

## Project Structure

```
Modified-Roofpedia/
├── config/
│   ├── predict-config.toml
│   └── train-config.toml
├── src/
│   ├── datasets.py
│   ├── unet.py
│   ├── transforms.py
│   └── colors.py
├── results/
│   ├── 01City/              # Building footprint GeoJSONs
│   ├── 02Images/            # Satellite tile directories
│   ├── 03Masks/             # Predicted segmentation masks
│   │   └── Green/
│   └── 04Results/           # Final labelled building footprints
├── predict_and_extract.py               # Mask prediction + label assignment
├── train.py                 # Model training
├── dataset.py               # Train/val/test split utility
├── environment.yml
└── README.md
```

## Training

To train your own model with custom labels, configure training options in `config/train-config.toml`, prepare your dataset with `dataset.py`, and run:

```
python train.py
```

The U-Net architecture supports arbitrary rooftop classes — not limited to green roofs or solar panels.

## Methodology

```
Satellite Imagery (Zoom 19)
        │
        ▼
  U-Net Prediction ──► Segmentation Masks
        │
        ▼
  swissSURFACE3D ──► Slope Computation ──► Roof Slope Filter
        │
        ▼
  Building Polygons + Mask Overlay ──► Label Assignment
        │
        ▼
  Labelled GeoJSON (Green/Potential Green/Solar/Flat_but_not_suitable per building)
```

## Acknowledgements

This project builds directly on the work of:

- **Roofpedia** by [Abraham Noah Wu](https://ual.sg/authors/abraham/) and [Filip Biljecki](https://ual.sg/authors/filip/) at the [Urban Analytics Lab](https://ual.sg), National University of Singapore.
  - Wu AN, Biljecki F (2021): Roofpedia: Automatic mapping of green and solar roofs for an open roofscape registry and evaluation of urban sustainability. *Landscape and Urban Planning* 214: 104167. [doi:10.1016/j.landurbplan.2021.104167](https://doi.org/10.1016/j.landurbplan.2021.104167)

- **swissSURFACE3D** by [swisstopo](https://www.swisstopo.admin.ch/) — the Swiss Federal Office of Topography.

### Packages

- [PyTorch](https://pytorch.org/)
- [GeoPandas](https://geopandas.org/)
- [Robosat](https://github.com/mapbox/robosat) — slippy map tile loading adapted from Robosat

## License

Distributed under the MIT License. See `LICENSE` for more information.

## Contact

*[Htet Yamin Ko Ko, International Space Science Institute (Bern, Switzerland), and htetyaminkokoedu@gmail.com]*
