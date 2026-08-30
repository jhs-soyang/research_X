import numpy as np
import pandas as pd


STANCE_COLS = ["believer", "denier", "neutral"]


def compute_annual_l1(male_r, female_r) -> pd.DataFrame:
    common = sorted(set(male_r.index) & set(female_r.index))
    male_arr = male_r.loc[common, STANCE_COLS].to_numpy(dtype=float)
    female_arr = female_r.loc[common, STANCE_COLS].to_numpy(dtype=float)
    # Sanity: each per-year ratio vector must sum to ~1 (probability simplex constraint).
    for arr, label in [(male_arr, "male"), (female_arr, "female")]:
        sums = arr.sum(axis=1)
        if not np.allclose(sums, 1.0, atol=5e-4):
            raise ValueError(
                f"{label} stance ratios do not sum to 1 within tol 5e-4: {sums}"
            )
    diff = male_arr - female_arr
    l1 = np.abs(diff).sum(axis=1)
    return pd.DataFrame({"Year": common, "L1": l1})


def compute_l1_from_counts(m_counts, f_counts) -> float:
    male = np.asarray(m_counts, dtype=float)
    female = np.asarray(f_counts, dtype=float)
    male_total = male.sum()
    female_total = female.sum()
    if male_total == 0 or female_total == 0:
        # Refuse to silently fall back to uniform distributions for empty buckets.
        # Bootstrap callers should resample from non-empty buckets only.
        raise ValueError(
            f"compute_l1_from_counts: empty count bucket detected "
            f"(male_total={int(male_total)}, female_total={int(female_total)}). "
            f"Refusing to substitute uniform 1/3 distributions."
        )
    male_r = male / male_total
    female_r = female / female_total
    return float(np.abs(male_r - female_r).sum())
