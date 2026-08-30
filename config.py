import os
from pathlib import Path

# Repository root is resolved from this file's location so the pipeline runs from a
# clean checkout on any machine. Override the data location with CCTD_DATA_DIR if the
# per-year aggregates are kept outside the repository.
ROOT = Path(__file__).resolve().parent
DATA_DIR = Path(os.environ.get("CCTD_DATA_DIR", ROOT / "data"))
OUT_DIR = ROOT / "outputs"
FIG_DIR = OUT_DIR / "figures"
TABLE_DIR = OUT_DIR / "tables"
INTER_DIR = OUT_DIR / "intermediate"

YEARS = list(range(2007, 2020))
SEED = 20260502
TOST_DELTA = 0.005
CHOW_SPLIT_INDEX = 8
PRE_WINDOW = (2007, 2014)
POST_WINDOW = (2015, 2019)
B_BOOTSTRAP = 1000

COLORS = {
    "purple": "#6c5ce7",
    "pink": "#fd79a8",
    "teal": "#00cec9",
    "red": "#d63031",
    "yellow": "#fdcb6e",
}
