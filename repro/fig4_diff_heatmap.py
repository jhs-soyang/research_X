import numpy as np

from config import COLORS
from .figures import save_fig, setup_style
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


def make_figure(male_r, female_r):
    setup_style()
    years = male_r.index.to_list()
    stances = ["believer", "denier", "neutral"]
    data = (female_r[stances] - male_r[stances]).T.to_numpy()
    vmax = float(np.max(np.abs(data)))

    fig, ax = plt.subplots(figsize=(12, 4.8))
    im = ax.imshow(data, aspect="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    ax.set_title("Female-Male Stance Ratio Difference")
    ax.set_xticks(np.arange(len(years)), labels=years, rotation=45)
    ax.set_yticks(np.arange(len(stances)), labels=[s.title() for s in stances])
    ax.grid(False)

    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            value = data[i, j]
            color = "white" if abs(value) > vmax * 0.55 else "black"
            # 3dp formatting so small magnitudes (e.g., -0.004) do not collapse to '-0.00' visually
            label = f"{value:+.3f}".lstrip("+") if value < 0 else f"{value:.3f}"
            ax.text(j, i, label, ha="center", va="center", color=color, fontsize=7.5)

    idx_2014 = years.index(2014)
    # Thicker, darker gold border for print legibility (per IJHSR figure standards)
    ax.add_patch(
        Rectangle(
            (idx_2014 - 0.5, -0.5),
            2,
            len(stances),
            fill=False,
            edgecolor="#daa520",  # darker goldenrod
            linewidth=4.5,
        )
    )
    cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label("Female ratio - Male ratio")
    fig.tight_layout()
    return save_fig(fig, "Figure4")
