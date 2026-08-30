import numpy as np

from config import COLORS
from .figures import save_fig, setup_style
import matplotlib.pyplot as plt


def make_figure(male_r, female_r):
    setup_style()
    years = male_r.index.to_list()
    row_labels = [
        "Female Believer",
        "Male Believer",
        "Female Denier",
        "Male Denier",
        "Female Neutral",
        "Male Neutral",
    ]
    data = np.vstack(
        [
            female_r["believer"].values,
            male_r["believer"].values,
            female_r["denier"].values,
            male_r["denier"].values,
            female_r["neutral"].values,
            male_r["neutral"].values,
        ]
    )

    fig, ax = plt.subplots(figsize=(12, 5.8))
    im = ax.imshow(data, aspect="auto", cmap="viridis", vmin=0, vmax=max(0.8, data.max()))
    ax.set_title("Gender by Stance Ratio Heatmap")
    ax.set_xticks(np.arange(len(years)), labels=years, rotation=45)
    ax.set_yticks(np.arange(len(row_labels)), labels=row_labels)
    ax.grid(False)

    # Two red vertical lines bracketing the 2014-2015 structural-shift period
    # (left edge of 2014 column and right edge of 2015 column).
    idx_2014 = years.index(2014)
    idx_2015 = years.index(2015)
    for x_pos in [idx_2014 - 0.5, idx_2015 + 0.5]:
        ax.axvline(x=x_pos, color=COLORS["red"], linewidth=2.5, linestyle="-", clip_on=False)

    cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    cbar.set_label("Ratio")
    fig.tight_layout()
    return save_fig(fig, "Figure3")
