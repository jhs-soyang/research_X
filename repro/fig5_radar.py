import numpy as np

from .figures import save_fig, setup_style
import matplotlib.pyplot as plt


def make_figure(male_r, female_r):
    setup_style()
    years = [2008, 2011, 2014, 2015, 2017, 2019]
    stances = ["believer", "denier", "neutral"]
    labels = [s.title() for s in stances]
    angles = np.linspace(0, 2 * np.pi, len(stances), endpoint=False).tolist()
    angles += angles[:1]
    max_ratio = max(male_r.loc[years, stances].max().max(), female_r.loc[years, stances].max().max())
    ylim = min(1.0, max_ratio + 0.1)

    fig, axes = plt.subplots(2, 3, figsize=(12, 8), subplot_kw={"projection": "polar"})
    for ax, year in zip(axes.flat, years):
        if year in {2014, 2015}:
            ax.set_facecolor("#fff0f4")
        male_vals = male_r.loc[year, stances].to_list()
        female_vals = female_r.loc[year, stances].to_list()
        male_closed = male_vals + male_vals[:1]
        female_closed = female_vals + female_vals[:1]

        ax.plot(angles, male_closed, color="#0984e3", linewidth=2, label="Male")
        ax.fill(angles, male_closed, color="#0984e3", alpha=0.12)
        ax.plot(angles, female_closed, color="#d63031", linewidth=2, label="Female")
        ax.fill(angles, female_closed, color="#d63031", alpha=0.12)
        ax.set_title(str(year), pad=14)
        ax.set_xticks(angles[:-1], labels)
        ax.set_ylim(0, ylim)
        ax.set_yticks([0.2, 0.4, 0.6, 0.8])
        ax.set_yticklabels(["0.2", "0.4", "0.6", "0.8"], fontsize=8)
        ax.grid(True, alpha=0.25)

    axes.flat[0].legend(loc="upper right", bbox_to_anchor=(1.35, 1.18))
    fig.suptitle("Selected-Year Gender Stance Profiles", y=0.98)
    fig.tight_layout()
    return save_fig(fig, "Figure5")
