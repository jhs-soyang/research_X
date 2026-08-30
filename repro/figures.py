from pathlib import Path
import os

from config import FIG_DIR, ROOT

os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".mplconfig"))
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def setup_style():
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "figure.figsize": (10, 6),
            "axes.grid": True,
            "grid.alpha": 0.25,
            "grid.linestyle": "-",
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def save_fig(fig, name, dpi=300) -> Path:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    path = FIG_DIR / f"{name}.png"
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return path
