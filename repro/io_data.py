from pathlib import Path

import pandas as pd

from config import DATA_DIR


RENAME_MAP = {
    "Believer (Frequency)": "Believer_Freq",
    "Believer (Ratio)": "Believer_Ratio",
    "Denier (Frequency)": "Denier_Freq",
    "Denier (Ratio)": "Denier_Ratio",
    "Neutral (Frequency)": "Neutral_Freq",
    "Neutral (Ratio)": "Neutral_Ratio",
    "Total Tweets": "Total",
}

RATIO_COLS = {
    "Believer_Ratio": "believer",
    "Denier_Ratio": "denier",
    "Neutral_Ratio": "neutral",
}

COUNT_COLS = {
    "Believer_Freq": "believer",
    "Denier_Freq": "denier",
    "Neutral_Freq": "neutral",
}


def load_yearly_stance(start=2007, end=2019, data_dir=DATA_DIR) -> pd.DataFrame:
    rows = []
    data_dir = Path(data_dir)
    for year in range(start, end + 1):
        path = data_dir / f"{year}_GenderStance.csv"
        if not path.exists():
            raise FileNotFoundError(f"Missing input file: {path}")
        frame = pd.read_csv(path)
        if "Year" not in frame.columns:
            frame["Year"] = year
        rows.append(frame)

    df = pd.concat(rows, ignore_index=True)
    # Defensive: normalize Gender capitalisation before filtering (Methods §Preprocessing
    # promises lowercase normalisation; the upstream aggregator already produces title-case
    # values, so we keep title-case as the canonical form but reject anything else).
    df["Gender"] = df["Gender"].astype(str).str.strip().str.title()
    expected_genders = {"Male", "Female", "Undefined"}
    unexpected = sorted(set(df["Gender"].unique()) - expected_genders)
    if unexpected:
        raise ValueError(
            f"io_data.load_yearly_stance: unexpected gender labels found: {unexpected}. "
            f"Expected exactly {sorted(expected_genders)}."
        )
    df = df[df["Gender"] != "Undefined"].copy()
    df = df.rename(columns=RENAME_MAP)

    required = [
        "Year",
        "Gender",
        "Believer_Freq",
        "Believer_Ratio",
        "Denier_Freq",
        "Denier_Ratio",
        "Neutral_Freq",
        "Neutral_Ratio",
        "Total",
    ]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    numeric_cols = [col for col in required if col not in {"Gender"}]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="raise")

    df["Year"] = df["Year"].astype(int)
    count_cols = ["Believer_Freq", "Denier_Freq", "Neutral_Freq", "Total"]
    for col in count_cols:
        df[col] = df[col].astype(int)

    # Defensive consistency check: the per-stance frequencies should sum to Total
    # and the Ratio columns should equal Frequency / Total to within machine
    # precision. This catches upstream CSV corruption / bookkeeping errors.
    freq_sum = df[["Believer_Freq", "Denier_Freq", "Neutral_Freq"]].sum(axis=1)
    if (freq_sum != df["Total"]).any():
        bad_rows = df.loc[freq_sum != df["Total"], ["Year", "Gender", "Total"]]
        raise ValueError(
            f"io_data.load_yearly_stance: stance frequency sums do not match Total in:\n{bad_rows}"
        )
    for stance in ["Believer", "Denier", "Neutral"]:
        ratio_check = df[f"{stance}_Freq"] / df["Total"]
        if not ((ratio_check - df[f"{stance}_Ratio"]).abs() < 1e-9).all():
            raise ValueError(
                f"io_data.load_yearly_stance: {stance}_Freq/Total disagrees with {stance}_Ratio "
                f"by more than 1e-9 in some rows."
            )

    return df[required].sort_values(["Year", "Gender"]).reset_index(drop=True)


def _pivot(df: pd.DataFrame, columns: dict[str, str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    male = (
        df[df["Gender"] == "Male"]
        .set_index("Year")
        .rename(columns=columns)[list(columns.values())]
        .sort_index()
    )
    female = (
        df[df["Gender"] == "Female"]
        .set_index("Year")
        .rename(columns=columns)[list(columns.values())]
        .sort_index()
    )
    male_only = set(male.index) - set(female.index)
    female_only = set(female.index) - set(male.index)
    if male_only or female_only:
        raise ValueError(
            "Asymmetric Male/Female year coverage detected — "
            f"male-only years: {sorted(male_only)}, female-only years: {sorted(female_only)}. "
            "Refusing to silently drop years; fix the upstream CSVs."
        )
    common = sorted(set(male.index) & set(female.index))
    if not common:
        raise ValueError("No common years for Male and Female rows")
    return male.loc[common].copy(), female.loc[common].copy()


def pivot_ratios(df) -> tuple[pd.DataFrame, pd.DataFrame]:
    return _pivot(df, RATIO_COLS)


def pivot_counts(df) -> tuple[pd.DataFrame, pd.DataFrame]:
    return _pivot(df, COUNT_COLS)
