"""Paths and settings of the training/evaluation pipeline, read from config.toml.

The configuration file is looked up in this order: the environment variable ROOFPEDIA_CONFIG,
then config.toml in the repository root. Copy config.example.toml to config.toml and edit it.
"""
import os
import tomllib

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG_FILE = os.environ.get('ROOFPEDIA_CONFIG', os.path.join(REPO, 'config.toml'))
if not os.path.exists(CONFIG_FILE):
    raise FileNotFoundError(f'{CONFIG_FILE} not found: copy config.example.toml to config.toml and set your paths')
with open(CONFIG_FILE, 'rb') as _f:
    CFG = tomllib.load(_f)

_p = CFG['paths']
WORK = _p['work_dir']                        # everything the pipeline writes goes below this folder
DATA = os.path.join(WORK, 'data')            # tiles, labels, structural layers, split.csv, norm_stats.json
RUNS = os.path.join(WORK, 'runs')            # one folder per trained configuration
EVAL = os.path.join(WORK, 'eval')            # metrics, tables and figures
for _d in (DATA, RUNS, EVAL):
    os.makedirs(_d, exist_ok=True)

LABEL_RASTER = _p['label_raster']            # reference raster, codes 0 bg, 1 G, 2 PG, 3 S, 4 NS
NDSM, SLOPE, ASPECT = _p['ndsm'], _p['slope'], _p['aspect']
SWISS_TILES = _p['swissimage_tiles']         # <dir>/<x>/<y>.png at zoom 19 (the folder that contains the x folders)
MAPBOX_TILES = _p.get('mapbox_tiles', '')    # optional, same layout
RS_DIR = _p.get('rs_geotiffs', '')           # optional, folder with the SWISSIMAGE RS GeoTIFFs
FOOTPRINTS = _p['footprints']                # building footprints (shapefile), CRS given by FOOTPRINT_CRS
REFERENCE_TABLE = _p['reference_table']      # .dbf / table with one row per footprint and the reference class
FOOTPRINT_CRS = CFG.get('crs', {}).get('footprints', 'EPSG:2056')
RS_CRS = CFG.get('crs', {}).get('rs', 'EPSG:2056')
ID_FIELD = CFG.get('fields', {}).get('id', 'UUID')
REF_FIELD = CFG.get('fields', {}).get('reference', 'reference')
BASELINES = CFG.get('baselines', {})         # optional: name -> raster of an existing pipeline (e.g. Roofpedia)

Z, SIZE, FLAT_SLOPE_DEG = 19, 256, 6.5
