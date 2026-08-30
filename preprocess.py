"""Closes the v2 C1/L2 gap: regenerate per-year GenderStance aggregates from the raw corpus.

Runs the Methods §Preprocessing pipeline end-to-end:
    1. Read the raw Climate Change Twitter Dataset CSV (path given by --raw-csv)
       (chunked, 200K rows at a time, to stay under ~1 GB resident memory).
    2. Keep only the 'created_at', 'gender', 'stance' columns.
    3. Standardise labels: lowercase, strip whitespace; map gender to title-case (Male,
       Female, Undefined) for compatibility with the existing analysis pipeline.
    4. Parse timestamps and extract calendar year.
    5. Group by (Year, Gender) and compute (Believer_Freq, Denier_Freq, Neutral_Freq, Total).
    6. Write thirteen `{year}_GenderStance.csv` files for 2007-2019 (the analysis range).

Outputs are written to --out-dir (default: outputs/intermediate/preprocess_regen).
To replace the analysis inputs in place, point config.DATA_DIR at that directory or copy
the files over data/.

The raw corpus (~1.9 GB) is not distributed with this repository; download it from the
source cited in the manuscript and pass its location:

    python preprocess.py --raw-csv /path/to/The\ Climate\ Change\ Twitter\ Dataset.csv --verify

`--verify` compares the regenerated CSVs against the per-year aggregates in --data-dir
(default: data/) and exits non-zero on any mismatch, closing the reproducibility loop
without overwriting inputs.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
DEFAULT_RAW_CSV = Path(
    os.environ.get("CCTD_RAW_CSV", ROOT / "The Climate Change Twitter Dataset.csv")
)
DEFAULT_DATA_DIR = Path(os.environ.get("CCTD_DATA_DIR", ROOT / "data"))
DEFAULT_OUTPUT_DIR = ROOT / "outputs" / "intermediate" / "preprocess_regen"

CHUNK_SIZE = 200_000
ANALYSIS_YEARS = list(range(2007, 2020))
GENDERS = ["Female", "Male", "Undefined"]
STANCES = ["believer", "denier", "neutral"]


def aggregate(raw_csv: Path) -> pd.DataFrame:
    """Stream raw CSV in chunks and return a long-form (Year, Gender, stance) count table."""
    accum: dict[tuple[int, str], dict[str, int]] = {}
    total_rows = 0
    for chunk in pd.read_csv(
        raw_csv,
        usecols=["created_at", "gender", "stance"],
        chunksize=CHUNK_SIZE,
        low_memory=False,
    ):
        # Standardise labels per Methods §Preprocessing.
        chunk["gender"] = chunk["gender"].astype(str).str.strip().str.lower()
        chunk["stance"] = chunk["stance"].astype(str).str.strip().str.lower()
        # Defensive: reject unknown gender labels rather than silently dropping them
        # (parallel to the stance-label guard below).
        expected_genders = {"male", "female", "undefined"}
        unknown_genders = sorted(set(chunk["gender"].unique()) - expected_genders)
        if unknown_genders:
            raise ValueError(
                f"preprocess.py: unexpected gender labels encountered: {unknown_genders}. "
                f"Expected exactly {sorted(expected_genders)}."
            )
        chunk["gender"] = chunk["gender"].map(
            {"male": "Male", "female": "Female", "undefined": "Undefined"}
        )
        # Parse timestamp -> year.
        chunk["Year"] = pd.to_datetime(chunk["created_at"], errors="coerce", utc=True).dt.year
        chunk = chunk.dropna(subset=["Year", "gender", "stance"])
        chunk["Year"] = chunk["Year"].astype(int)

        grouped = chunk.groupby(["Year", "gender", "stance"]).size().reset_index(name="n")
        unknown_stances: set = set()
        for _, row in grouped.iterrows():
            key = (int(row["Year"]), row["gender"])
            d = accum.setdefault(key, {s: 0 for s in STANCES})
            stance = row["stance"]
            if stance in d:
                d[stance] += int(row["n"])
            else:
                unknown_stances.add(stance)
        if unknown_stances:
            # Loud failure: never silently drop unexpected stance labels.
            raise ValueError(
                f"preprocess.py: unexpected stance labels encountered: {sorted(unknown_stances)}. "
                f"Expected exactly {STANCES}. Refusing to silently drop rows."
            )
        total_rows += len(chunk)
        if total_rows % (CHUNK_SIZE * 5) == 0:
            print(f"  processed {total_rows:,} rows", flush=True)

    print(f"  total rows processed: {total_rows:,}", flush=True)

    rows = []
    for (year, gender), counts in sorted(accum.items()):
        believer = counts["believer"]
        denier = counts["denier"]
        neutral = counts["neutral"]
        total = believer + denier + neutral
        if total == 0:
            continue
        rows.append(
            {
                "Year": year,
                "Gender": gender,
                "Believer (Frequency)": believer,
                "Believer (Ratio)": believer / total,
                "Denier (Frequency)": denier,
                "Denier (Ratio)": denier / total,
                "Neutral (Frequency)": neutral,
                "Neutral (Ratio)": neutral / total,
                "Total Tweets": total,
            }
        )
    return pd.DataFrame(rows)


def write_per_year(df: pd.DataFrame, out_dir: Path) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for year in ANALYSIS_YEARS:
        sub = df[df["Year"] == year].copy()
        if sub.empty:
            print(f"  WARNING: no rows for year {year}", flush=True)
            continue
        # Sort gender to match the existing aggregator's ordering (Male, Female, Undefined).
        order = {"Male": 0, "Female": 1, "Undefined": 2}
        sub["__order"] = sub["Gender"].map(order)
        sub = sub.sort_values("__order").drop(columns="__order")
        path = out_dir / f"{year}_GenderStance.csv"
        sub.to_csv(path, index=False)
        written.append(path)
    return written


def verify_against_existing(regen_dir: Path, existing_dir: Path) -> bool:
    """Return True if every regenerated CSV matches the corresponding existing CSV.

    Uses an OUTER merge with an indicator column so that rows present only in the
    regenerated CSV or only in the existing CSV are surfaced as failures rather than
    being silently dropped (as a default INNER merge would do).
    """
    all_ok = True
    for year in ANALYSIS_YEARS:
        a = regen_dir / f"{year}_GenderStance.csv"
        b = existing_dir / f"{year}_GenderStance.csv"
        if not a.exists() or not b.exists():
            print(f"  {year}: MISSING ({a.exists()=}, {b.exists()=})")
            all_ok = False
            continue
        df_a = pd.read_csv(a)
        df_b = pd.read_csv(b)
        # OUTER merge with `_merge` indicator: any row missing from either side is flagged.
        merged = df_a.merge(
            df_b,
            on=["Year", "Gender"],
            suffixes=("_regen", "_existing"),
            how="outer",
            indicator=True,
        )
        ok_year = True
        # Surface row-level membership mismatches (the silent-false-pass guard).
        membership = merged["_merge"].value_counts().to_dict()
        only_regen = merged[merged["_merge"] == "left_only"][["Year", "Gender"]]
        only_existing = merged[merged["_merge"] == "right_only"][["Year", "Gender"]]
        if len(only_regen) > 0:
            print(f"  {year}: row(s) ONLY in regenerated CSV: {only_regen.to_dict('records')}")
            ok_year = False
            all_ok = False
        if len(only_existing) > 0:
            print(f"  {year}: row(s) ONLY in existing CSV: {only_existing.to_dict('records')}")
            ok_year = False
            all_ok = False
        # Now check value equality on the rows present in both.
        both = merged[merged["_merge"] == "both"]
        for col in ["Believer (Frequency)", "Denier (Frequency)", "Neutral (Frequency)", "Total Tweets"]:
            diff = (both[f"{col}_regen"] - both[f"{col}_existing"]).abs().max()
            if diff != 0:
                print(f"  {year} {col}: max abs diff = {diff}")
                ok_year = False
                all_ok = False
        # Ratios may differ in last decimal due to float formatting; allow 1e-6.
        for col in ["Believer (Ratio)", "Denier (Ratio)", "Neutral (Ratio)"]:
            diff = (both[f"{col}_regen"] - both[f"{col}_existing"]).abs().max()
            if diff > 1e-6:
                print(f"  {year} {col}: max abs diff = {diff}")
                ok_year = False
                all_ok = False
        print(f"  {year}: {'PASS' if ok_year else 'FAIL'} (rows: {membership})")
    return all_ok


def write_audit(df: pd.DataFrame, path: Path) -> Path:
    """Persist the full, unfiltered (Year, Gender) count table as a reproducibility audit.

    Unlike the per-year analysis inputs, this keeps every year present in the raw corpus
    (including 2006) and every gender bucket (including Undefined), and flags which rows
    enter the analysis. It is the artifact that lets a reader re-derive the sample-size
    statements in the manuscript -- the 2006 row counts, the undefined-gender totals, the
    per-year binary-gendered sample sizes, and the analytical total.
    """
    audit = df.copy()
    audit["Analysis_Included"] = (
        audit["Year"].isin(ANALYSIS_YEARS) & (audit["Gender"] != "Undefined")
    )
    audit = audit.sort_values(["Year", "Gender"])
    path.parent.mkdir(parents=True, exist_ok=True)
    audit.to_csv(path, index=False)

    included = audit[audit["Analysis_Included"]]
    print(
        f"Wrote preprocessing audit to {path}: "
        f"{int(audit['Total Tweets'].sum()):,} rows total, "
        f"{int(included['Total Tweets'].sum()):,} in the analytical sample "
        f"across {included['Year'].nunique()} years."
    )
    return path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--raw-csv", type=Path, default=DEFAULT_RAW_CSV,
        help="path to the raw Climate Change Twitter Dataset CSV "
             "(default: $CCTD_RAW_CSV, else ./The Climate Change Twitter Dataset.csv)",
    )
    parser.add_argument(
        "--data-dir", type=Path, default=DEFAULT_DATA_DIR,
        help="directory holding the per-year {year}_GenderStance.csv aggregates to verify "
             "against (default: $CCTD_DATA_DIR, else ./data)",
    )
    parser.add_argument(
        "--out-dir", type=Path, default=DEFAULT_OUTPUT_DIR,
        help="where to write the regenerated per-year CSVs "
             "(default: ./outputs/intermediate/preprocess_regen)",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="compare regenerated CSVs against --data-dir and exit non-zero on any mismatch",
    )
    args = parser.parse_args()

    if not args.raw_csv.exists():
        print(f"ERROR: raw CSV not found at {args.raw_csv}", file=sys.stderr)
        print("       Pass --raw-csv /path/to/dataset.csv (the ~1.9 GB corpus is not "
              "distributed with this repository).", file=sys.stderr)
        return 1

    print(f"Reading raw corpus from {args.raw_csv} "
          f"(~{args.raw_csv.stat().st_size / 1e9:.1f} GB)...", flush=True)
    df = aggregate(args.raw_csv)
    print(f"Aggregation complete: {len(df)} (Year, Gender) cells across {df.Year.nunique()} years.")

    written = write_per_year(df, args.out_dir)
    print(f"Wrote {len(written)} per-year CSVs to {args.out_dir}.")

    write_audit(df, args.out_dir.parent / "preprocessing_audit.csv")

    if args.verify:
        print(f"\nVerifying regenerated CSVs against {args.data_dir}/...")
        ok = verify_against_existing(args.out_dir, args.data_dir)
        print("\nOverall:", "PASS — preprocess.py closes the reproducibility loop." if ok else "FAIL — investigate divergences above.")
        return 0 if ok else 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
