import numpy as np
import pandas as pd

from .l1_metric import compute_l1_from_counts


def bootstrap_l1_ci(counts_df, B=1000, seed=20260502, alpha=0.05):
    """Draw multinomial samples by year and gender, then recompute annual L1."""
    rng = np.random.default_rng(seed)
    rows = []
    by_year = counts_df.groupby("Year", sort=True)
    for year, g in by_year:
        male = (
            g[g.Gender == "Male"][["Believer_Freq", "Denier_Freq", "Neutral_Freq"]]
            .values.flatten()
            .astype(int)
        )
        fem = (
            g[g.Gender == "Female"][["Believer_Freq", "Denier_Freq", "Neutral_Freq"]]
            .values.flatten()
            .astype(int)
        )
        if len(male) != 3 or len(fem) != 3:
            raise ValueError(f"Expected one Male and one Female count row for {year}")
        Nm = int(male.sum())
        Nf = int(fem.sum())
        # No uniform fallback: an empty year x gender bucket is a data error, not a
        # distribution to impute. This mirrors compute_l1_from_counts, which refuses the
        # same substitution downstream.
        if Nm == 0 or Nf == 0:
            raise ValueError(
                f"{year}: empty gender bucket (Male={Nm}, Female={Nf}); "
                "cannot bootstrap an L1 distance."
            )
        pm = male / Nm
        pf = fem / Nf
        l1s = np.empty(B)
        for b in range(B):
            mb = rng.multinomial(Nm, pm)
            fb = rng.multinomial(Nf, pf)
            l1s[b] = compute_l1_from_counts(mb, fb)
        lo = float(np.quantile(l1s, alpha / 2))
        hi = float(np.quantile(l1s, 1 - alpha / 2))
        rows.append({"Year": int(year), "L1_lo": lo, "L1_hi": hi})
    return pd.DataFrame(rows).sort_values("Year").reset_index(drop=True)
