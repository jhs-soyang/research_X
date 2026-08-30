import numpy as np

from config import COLORS
from .figures import save_fig, setup_style
import matplotlib.pyplot as plt


def make_figure(l1_df, ci_df, lr):
    setup_style()
    fig, ax = plt.subplots(figsize=(10, 6))
    merged = l1_df.merge(ci_df, on="Year", how="left")
    years = merged["Year"].to_numpy()
    l1 = merged["L1"].to_numpy()
    yerr = np.vstack([l1 - merged["L1_lo"].to_numpy(), merged["L1_hi"].to_numpy() - l1])

    ax.errorbar(
        years,
        l1,
        yerr=yerr,
        marker="o",
        color=COLORS["purple"],
        ecolor=COLORS["purple"],
        elinewidth=1.2,
        capsize=3,
        linewidth=2,
        label="L1 distance",
    )
    ax.plot(
        years,
        lr["fitted"],
        linestyle="--",
        color=COLORS["red"],
        linewidth=2,
        label=f"OLS slope={lr['slope']:.5f}, p={lr['pvalue']:.3f}",
    )
    ax.set_title("Temporal Evolution of Gender Stance Gap (L1 Distance)")
    ax.set_xlabel("Year")
    ax.set_ylabel("L1 Distance")
    ax.set_xticks(years)
    ax.tick_params(axis="x", rotation=45)
    ax.legend(loc="best")
    fig.tight_layout()
    return save_fig(fig, "Figure1")
