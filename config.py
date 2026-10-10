"""Shared paths and experiment settings for the label-free virtual Cell Painting starter."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
# Packed plates live outside the code folder in the competition archive; set VCP_DATA to override.
_ARCHIVE_DATA = ROOT.parents[1] / "data" / "external" / "cpjump1"
DATA = Path(os.environ.get("VCP_DATA", _ARCHIVE_DATA if _ARCHIVE_DATA.exists() else ROOT / "data"))
# Extra read-only folders searched after DATA (e.g. A549 and U2OS plates in separate Kaggle inputs).
DATA_DIRS = [DATA] + [Path(p) for p in os.environ.get("VCP_DATA_EXTRA", "").split(os.pathsep) if p]
RUNS = Path(os.environ.get("VCP_RUNS", ROOT / "runs"))

S3 = "https://cellpainting-gallery.s3.amazonaws.com"
PILOT = "cpg0000-jump-pilot/source_4/workspace"
BATCH = "2020_11_04_CPJUMP1"

# A549, 48 h, 100% density, no anomalies: four replicate plates of the JUMP-Target-1 compound map.
TRAIN_PLATES = os.environ.get("VCP_TRAIN_PLATES", "BR00117015").split(",")
TEST_PLATES = os.environ.get("VCP_TEST_PLATES", "BR00117017,BR00117019").split(",")
# U2OS replication study (separate dataset, never mixed into the A549 search): VCP_TRAIN_PLATES=BR00117011
# VCP_TEST_PLATES=BR00117012 (the only two U2OS 48 h plates of CPJUMP1 without anomalies).
SITES_TRAIN = [1, 6, 11]  # of 16 sites per well
SITES_TEST = [1, 11]

# Stored channel order. Inputs are the three bright-field planes, targets the five Cell Painting stains.
CHANNELS = ["DNA", "ER", "RNA", "AGP", "Mito", "LowZBF", "Brightfield", "HighZBF"]
TARGET_CH = [0, 1, 2, 3, 4]
INPUT_CH = [5, 6, 7]
DOWNSAMPLE = 2  # 1080 -> 540 px (~1.3 um/px), keeps storage at ~4.7 MB per site

# Compound-level split: held-out compounds are never seen in training, on any plate.
SPLIT_SEED = 0
SPLIT_FRAC = {"train": 0.7, "val": 0.1, "test": 0.2}
