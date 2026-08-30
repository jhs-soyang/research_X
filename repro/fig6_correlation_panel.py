import numpy as np
from scipy import stats as sps

from .figures import save_fig, setup_style
import matplotlib.pyplot as plt


def _p_text(p):
    return f"p={p:.3f}" if p >= 0.001 else f"p={p:.1e}"


def _scatter_panel(ax, male_r, female_r, stance):
    x = male_r[stance].values
    y = female_r[stance].values
    years = male_r.index.values
    sc = ax.scatter(x, y, c=years, cmap="viridis", s=48, edgecolor="white", linewidth=0.6)
    slope, intercept, r, p, _ = sps.linregress(x, y)
    xx = np.linspace(min(x.min(), y.min()), max(x.max(), y.max()), 100)
    ax.plot(xx, intercept + slope * xx, color="#2d3436", linestyle="--", linewidth=1.5)
    ax.set_title(f"{stance.title()} Ratios")
    ax.set_xlabel("Male ratio")
    ax.set_ylabel("Female ratio")
    ax.text(
        0.04,
        0.94,
        f"r={r:.3f}, {_p_text(p)}",
        transform=ax.transAxes,
        va="top",
        ha="left",
        bbox={"facecolor": "white", "alpha": 0.75, "edgecolor": "none", "pad": 3},
    )
    return sc


def make_figure(male_r, female_r, l1_df, lr):
    setup_style()
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    sc1 = _scatter_panel(axes[0, 0], male_r, female_r, "believer")
    sc2 = _scatter_panel(axes[0, 1], male_r, female_r, "denier")
    sc3 = _scatter_panel(axes[1, 0], male_r, female_r, "neutral")

    ax = axes[1, 1]
    years = l1_df["Year"].to_numpy()
    l1 = l1_df["L1"].to_numpy()
    colors = []
    for year in years:
        if year <= 2013:
            colors.append("#0984e3")
        elif year <= 2015:
            colors.append("#d63031")
        else:
            colors.append("#00b894")
    ax.plot(years, l1, color="#636e72", linewidth=1, alpha=0.45)
    ax.scatter(years, l1, color=colors, s=55, zorder=3)
    ax.plot(years, lr["fitted"], color="#636e72", linestyle="--", linewidth=1.8)
    ax.set_title("L1 Distance by Period")
    ax.set_xlabel("Year")
    ax.set_ylabel("L1 Distance")
    ax.set_xticks(years)
    ax.tick_params(axis="x", rotation=45)
    ax.text(
        0.04,
        0.94,
        f"slope={lr['slope']:.5f}, p={lr['pvalue']:.3f}",
        transform=ax.transAxes,
        va="top",
        ha="left",
        bbox={"facecolor": "white", "alpha": 0.75, "edgecolor": "none", "pad": 3},
    )

    fig.suptitle("Cross-Gender Correlation and L1 Stability", y=0.995)
    fig.tight_layout()
    # Add a shared horizontal colorbar at the bottom that maps year to dot color
    cbar = fig.colorbar(
        sc1,
        ax=axes.ravel().tolist(),
        orientation="horizontal",
        fraction=0.04,
        pad=0.07,
        aspect=50,
    )
    cbar.set_label("Year (color of scatter dots in believer/denier/neutral panels)")
    cbar.set_ticks([2007, 2010, 2013, 2014, 2015, 2016, 2019])
    return save_fig(fig, "Figure6")
