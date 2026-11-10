# Modified-Roofpedia (Bern) – pretrained four-class flat-roof models

Pretrained networks that classify flat roofs from aerial orthophotos into four classes relevant for green-roof planning:

| Code | Class | Meaning |
|---|---|---|
| 0 | background | everything that is not a labelled flat roof (ground, trees, pitched roofs) |
| 1 | Green (G) | flat roof with an existing vegetated cover (extensive or intensive green roof) |
| 2 | Potential Green (PG) | flat roof with a bare surface (gravel, bitumen membrane, etc.) on which greening is plausible |
| 3 | Solar (S) | flat roof occupied by photovoltaic or solar-thermal installations |
| 4 | Not Suitable (NS) | flat roof that cannot be greened because of its surface or use (glass roofs, terraces, parking decks, dense technical equipment) |

The models were trained in Bern, Switzerland, on 9,898 manually labelled flat-roof buildings. They extend the Roofpedia U-Net (Wu & Biljecki) from two to four roof classes. Optionally, they add a second encoder for building structure, which takes the normalized DSM, slope and aspect.

> **Reference.** Htet Yamin Ko Ko, "Beyond Binary Rooftop Mapping: Structure-Aware Multimodal Fusion for Four-Class Green Roof Potential Assessment from Open Swiss Geospatial Data" (manuscript under review). https://doi.org/10.48550/arXiv.2607.22342. 
---

## 1. Contents

```
weights/                 six models (PyTorch state dicts with embedded metadata) + model_cards.json
roofpedia_bern/          inference code
  models.py              network definitions and load_model()
  predict.py             tiles -> class masks
  make_struct.py         nDSM / slope / aspect rasters -> structural tiles (fusion models only)
  footprints.py          class masks + building footprints -> per-building class fractions and labels
sample/                  7 held-out SWISSIMAGE tiles from Bern covering all four roof classes (see preview.png):
  tiles/                 image tiles            struct/            structural tiles (for fusion models)
  reference_labels/      manual reference       expected_masks_*/  outputs of the two SWISSIMAGE models
requirements.txt
```

## 2. Models

All six networks share the same architecture family, training data, split and settings. The only differences are the imagery source and whether structural fusion is used. The metrics refer to held-out spatial blocks in Bern: 1,346 tiles and 1,186 buildings that were not used for training.

| File | Image input | Structural input | Pixel κ | Pixel mIoU (4 classes) | Building accuracy | Size |
|---|---|---|---|---|---|---|
| `swissimage_image_only.pth` | RGB | – | 0.670 | 0.398 | 0.659 | 151 MB |
| `swissimage_fusion.pth` | RGB | nDSM, slope, aspect | 0.708 | 0.426 | 0.650 | 629 MB |
| `mapbox_image_only.pth` | RGB | – | 0.678 | 0.411 | 0.651 | 151 MB |
| `mapbox_fusion.pth` | RGB | nDSM, slope, aspect | 0.705 | 0.426 | 0.660 | 629 MB |
| `rs_image_only.pth` | RGB + NIR | – | 0.689 | 0.419 | 0.657 | 151 MB |
| `rs_fusion.pth` | RGB + NIR | nDSM, slope, aspect | 0.710 | 0.421 | 0.675 | 629 MB |

- **Pixel κ:** Cohen's kappa over all pixels, including background.
- **Building accuracy:** overall accuracy of the area-based building labels (Section 5).
- **Full metadata:** `weights/model_cards.json` lists, for every model, the per-class IoU, the green-roof F1, the normalization constants and the training details. The same metadata is stored inside each `.pth` file.

**Which model to use.**
- **No elevation data:** use an image-only model. Choose `swissimage_image_only` or `mapbox_image_only` for RGB imagery, and `rs_image_only` if you have a near-infrared band.
- **A DSM and DTM available:** use a fusion model. In our tests, fusion gave consistently better pixel-level κ and roof outlines.
- **Accuracy outside Bern:** all models were trained on Swiss imagery from one city, so expect lower accuracy elsewhere (Section 7).

## 3. Installation

Python ≥ 3.10:

```bash
pip install -r requirements.txt     # torch, torchvision, numpy, pillow, rasterio, mercantile, geopandas
```

A GPU is recommended. On a CPU, prediction takes roughly 1–3 s per tile.

**Self-test.** Run this inside the release folder; it should report "masks written: 7":

```bash
python -m roofpedia_bern.predict --weights weights/swissimage_fusion.pth --tiles sample/tiles --struct sample/struct --out test_masks
```

Compare `test_masks` with `sample/expected_masks_swissimage_fusion`. More than 99.8% of pixels should be identical. Small differences come from numerical precision.

The seven sample tiles come from held-out evaluation blocks that were not used in training. `sample/preview.png` shows each tile with its reference label and the prediction of `swissimage_fusion`:
- **Clear examples, 98–100% of roof pixels correct:** an extensive green roof, two gravel roofs (Potential Green), a large PV roof (Solar), and a glass-roof complex (Not Suitable).
- **Mixed tiles, 74% and 62% correct:** one tile with Green, Potential Green and Not Suitable roofs, and a harder one where Solar, Not Suitable and Potential Green surfaces meet.

Together they show both typical successes and typical confusions.

## 4. Preparing your data

### 4.1 Image tiles

The models expect **slippy-map (XYZ) tiles in Web Mercator (EPSG:3857) at zoom level 19, 256 × 256 pixels**, stored as `<folder>/19/<x>/<y>.png`.

| Item | Requirement |
|---|---|
| Ground pixel size | z19 tiles have about 0.20 m pixels in Bern and about 0.19 m in Ghent. Imagery with a native resolution of 0.1–0.25 m works best. Coarser imagery is upsampled and loses detail. |
| Bands (RGB models) | 3-band, 8-bit RGB PNG or JPEG. |
| Bands (RS models) | 4-band `.npy` arrays of shape (4, 256, 256), uint8, in the order **red, green, blue, near-infrared**. |
| Radiometry | Natural colours with an ordinary 8-bit stretch. The networks normalize inputs with fixed constants (stored in the weight files), so strongly different radiometry reduces accuracy. |

Any XYZ tiler works for RGB imagery, for example QGIS *Generate XYZ tiles (Directory)* or `gdal2tiles.py --xyz`.

> **Watch the alpha channel.** Some four-band orthophoto products tag the fourth band as alpha, and tilers then treat it as transparency. That produces washed-out tiles, which is exactly what happened in an earlier version of our own work. For four-band data, read every band explicitly, for example with rasterio, and write the `.npy` tiles yourself.

### 4.2 Structural tiles (fusion models only)

You need three single-band rasters in a projected CRS that cover your tiles:

| Raster | Content | How we made it |
|---|---|---|
| nDSM | object height above ground in metres (DSM − DTM) | swissSURFACE3D DSM − swissALTI3D DTM |
| slope | slope of the DSM in degrees | `gdaldem slope dsm.tif slope.tif -alg Horn -compute_edges` (0.5 m DSM) |
| aspect | aspect of the DSM in degrees | `gdaldem aspect dsm.tif aspect.tif -alg Horn -zero_for_flat -compute_edges` |

Then build the structural tiles. They are resampled bilinearly to every image tile, and aspect is set to zero where the slope is below 6.5°, as in training:

```bash
python -m roofpedia_bern.make_struct --tiles my_tiles --ndsm ndsm.tif --slope slope.tif --aspect aspect.tif --out my_struct
```

We used a LiDAR DSM with 0.5 m resolution. A DSM of 1 m or finer that resolves roof parapets and roof equipment should work. Coarser DSMs make the structural layers less informative.

## 5. Running the workflow

```bash
# 1) pixel classification
python -m roofpedia_bern.predict --weights weights/swissimage_fusion.pth --tiles my_tiles --struct my_struct --out masks --color masks_color

# 2) per-building class fractions and labels (footprints: any vector format that geopandas reads, any CRS)
python -m roofpedia_bern.footprints --masks masks --footprints buildings.gpkg --out roofs.gpkg --tau 0.05
```

**Masks** are paletted PNGs. Their pixel values are the class codes in Section 1 (read them with `numpy.asarray(PIL.Image.open(path))`), and image viewers display them in the class colours: white for background, mint for Green, ochre for Potential Green, blue for Solar and dark red for Not Suitable. `--color` additionally writes plain RGB copies.

**Per-building output (`roofs.gpkg`):**

| Column | Meaning |
|---|---|
| `p_G`, `p_PG`, `p_S`, `p_NS` | Fraction of the footprint covered by each class. |
| `roof_frac` | Sum of the four fractions. |
| `label`, `label_name` | **Area-based** label: the class with the largest fraction. Assigned only if `roof_frac > tau`; otherwise "not detected". |
| `margin` | Difference between the two largest fractions. Buildings with a margin ≥ 0.8 were labelled correctly about 90% of the time in Bern; with a margin < 0.2, only about 50%. |

**Recommendations:**
- **Use the area-based label and keep the class fractions.** A priority rule such as "any solar pixel makes the roof Solar" nearly doubled the share of solar roofs in our evaluation.
- **Use the fractions for area statistics,** for example greenable area as the sum of `p_PG` times footprint area. Label-based areas are biased.

## 6. Fine-tuning on your own labels

We strongly recommend fine-tuning with a few hundred labelled buildings from your study area. The approach:
- **Labels:** rasterize your labelled roofs to the same tiles (codes 0–4).
- **Starting point:** load a model with `load_model()`.
- **Training:** train with a lower learning rate than ours, for example Adam at 1e-4 (we used 5e-4, batch size 8, and the Lovász hinge loss of the Roofpedia code).
- **Splits:** keep validation and test areas spatially separate from the training areas.
- **Normalization:** use the constants in `meta` (`image_mean`, `image_std`, `struct_mean`, `struct_std`) unchanged when fine-tuning.

[[AUTHOR: add a link to the training code repository if you make it public.]]

## 7. Known limitations

- **Trained in one city.** Training used only Bern (2024 SWISSIMAGE, swisstopo elevation models, Swiss roof types). Imagery from other sensors, seasons or sun angles, and other building styles, will reduce accuracy.
- **Flat roofs only.** Pitched roofs were background in training. The models are not designed to classify them, and some pitched roofs may still be detected as flat.
- **Single-interpreter labels.** The reference labels were produced by one interpreter. PV on green roofs, and moss on bare roofs, are ambiguous.
- **Solar is the weakest class** (pixel IoU about 0.3), because PV arrays are small and often mixed with other surfaces.
- **Structural capacity is not considered.** "Potential Green" means greening is plausible from the surface; it does not account for the roof's structural load-bearing capacity.

## 8. Terms of use

[[AUTHOR: state the conditions for academic use here, e.g., a licence (CC BY 4.0 / CC BY-NC 4.0) and the citation requirement.]]

Data used for training:
- **SWISSIMAGE, swissSURFACE3D, swissALTI3D and swissTLM3D** © swisstopo (open government data).
- **Mapbox satellite tiles** © Mapbox and its data providers.
- **SWISSIMAGE RS** for Bern, provided by the Swiss Territorial Data Lab (STDL) for research use.

The use of the `mapbox_*` and `rs_*` weights may be subject to the terms of these data providers. 

Contact: Htet Yamin Ko Ko, International Space Science Institute (ISSI), Bern – koko.htetyamin@issibern.ch
