from config import COLORS
from .figures import save_fig, setup_style
import matplotlib.pyplot as plt


def make_figure(l1_df, male_r, female_r):
    setup_style()
    fig, ax1 = plt.subplots(figsize=(12, 7))

    ax1.plot(
        l1_df["Year"],
        l1_df["L1"],
        "o-",
        color=COLORS["purple"],
        linewidth=3,
        markersize=7,
        label="L1 distance",
    )
    ax1.set_xlabel("Year")
    ax1.set_ylabel("L1 Distance", color=COLORS["purple"])
    ax1.tick_params(axis="y", labelcolor=COLORS["purple"])
    ax1.set_xticks(l1_df["Year"])
    ax1.tick_params(axis="x", rotation=45)

    ax2 = ax1.twinx()
    ax2.plot(
        male_r.index,
        male_r["believer"],
        marker="s",
        linestyle="--",
        color=COLORS["teal"],
        linewidth=2,
        markersize=5,
        label="Male believer ratio",
    )
    ax2.plot(
        female_r.index,
        female_r["believer"],
        marker="^",
        linestyle="--",
        color=COLORS["pink"],
        linewidth=2,
        markersize=5,
        label="Female believer ratio",
    )
    ax2.set_ylabel("Believer Ratio")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="lower right")
    ax1.set_title("Dual-Axis Analysis: L1 Distance and Gender Believer Ratios")
    fig.tight_layout()
    return save_fig(fig, "Figure2")
