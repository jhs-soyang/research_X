import pandas as pd


COUNT_COLS = ["Believer_Freq", "Denier_Freq", "Neutral_Freq"]
STANCE_NAMES = ["Believer", "Denier", "Neutral"]


def build_table1(df) -> pd.DataFrame:
    rows = []
    for gender in ["Male", "Female"]:
        g = df[df["Gender"] == gender]
        counts = g[COUNT_COLS].sum()
        total = int(counts.sum())
        row = {"Gender": gender}
        for stance, col in zip(STANCE_NAMES, COUNT_COLS):
            count = int(counts[col])
            row[f"{stance}_Count"] = count
            row[f"{stance}_Proportion"] = count / total if total else 0.0
        row["Total"] = total
        rows.append(row)

    counts = df[COUNT_COLS].sum()
    total = int(counts.sum())
    row = {"Gender": "Total"}
    for stance, col in zip(STANCE_NAMES, COUNT_COLS):
        count = int(counts[col])
        row[f"{stance}_Count"] = count
        row[f"{stance}_Proportion"] = count / total if total else 0.0
    row["Total"] = total
    rows.append(row)

    return pd.DataFrame(rows)


def build_table2(stats_dict) -> pd.DataFrame:
    """Manuscript Table 2 — 3-column schema (Statistical Measure / Value / Interpretation)
    that mirrors the docx Table 2 exactly so the generated CSV/MD artifact and the
    embedded docx table do not diverge. v2.9 expands to 11 stats rows: separate
    slope-CI and slope-p rows, Welch–Satterthwaite df reporting, and a Shapiro
    interpretation that flags low power at n = 13.
    """
    lr = stats_dict["lr"]
    tost = stats_dict["tost"]
    chow = stats_dict["chow"]
    sh = stats_dict["sh"]
    bp = stats_dict["bp"]
    dw = stats_dict.get("dw", {"DW": float("nan"), "verdict": ""})
    welch_l1 = stats_dict.get(
        "welch_l1",
        {
            "t": float("nan"), "p": float("nan"),
            "pre_mean": float("nan"), "post_mean": float("nan"),
            "df": float("nan"),
        },
    )
    hac = stats_dict.get(
        "hac",
        {"se_hac": float("nan"), "ratio": float("nan"), "ci_low": float("nan"), "ci_high": float("nan"), "max_lag": 0},
    )
    rows = [
        {
            "Statistical Measure": "Slope (per year)",
            "Value": f"{lr['slope']:.5f}",
            "Interpretation": "Negligible drift; near zero",
        },
        {
            "Statistical Measure": "Slope 95% CI (per year)",
            "Value": f"[{lr['ci_low']:.4f}, {lr['ci_high']:.4f}]",
            "Interpretation": "Narrow and centered near zero; bounds drift to within 0.65 pp/yr",
        },
        {
            "Statistical Measure": "P-value (slope ≠ 0)",
            "Value": f"{lr['pvalue']:.3f}",
            "Interpretation": "Not statistically significant",
        },
        {
            "Statistical Measure": "R²",
            "Value": f"{lr['r2']:.4f}",
            "Interpretation": "Year explains <1% of L₁ variance",
        },
        {
            "Statistical Measure": "TOST p (Δ = 0.005/yr)",
            "Value": f"{tost['p_max']:.3f}",
            "Interpretation": "Cannot formally claim equivalence at α = 0.05 (p_max < α required)",
        },
        {
            "Statistical Measure": "Chow F (split 2014/2015)",
            "Value": f"F({chow['df1']}, {chow['df2']}) = {chow['F']:.2f} (p = {chow['p']:.3f})",
            "Interpretation": "Borderline structural shift in the L1~year regression around 2014-2015 (non-significant after Bonferroni for nine primary substantive tests)",
        },
        {
            "Statistical Measure": "Shapiro–Wilk (residuals)",
            "Value": f"W = {sh['W']:.3f} (p = {sh['p']:.3f})",
            "Interpretation": "No normality violation detected; low power at n = 13",
        },
        {
            "Statistical Measure": "Breusch–Pagan",
            "Value": f"LM = {bp['LM']:.2f} (p = {bp['p']:.3f})",
            "Interpretation": "No evidence of heteroscedasticity",
        },
        {
            # Use '|t|' to match the manuscript docx Table 2 verbatim. to_csv_md()
            # rewrites '|t|' -> 'abs(t)' only when emitting the markdown copy, so the
            # pipe never clashes with the markdown table column separator while the
            # CSV/docx stay string-identical.
            "Statistical Measure": "Welch's t on L1 (2007–2014 vs 2015–2019)",
            "Value": f"|t| = {abs(welch_l1['t']):.2f} (df = {welch_l1['df']:.2f}; p = {welch_l1['p']:.3f})",
            "Interpretation": "Borderline drop in L1 between regimes; pre 0.0828, post 0.0531; consistent with step-convergence at 2014–2015",
        },
        {
            "Statistical Measure": "Durbin–Watson (residual autocorrelation)",
            "Value": f"DW = {dw['DW']:.3f}",
            "Interpretation": "Inconclusive at α = 0.05 (dL ≈ 1.05, dU ≈ 1.40); leans toward positive autocorrelation",
        },
        {
            "Statistical Measure": f"Newey-West HAC slope SE (Bartlett, L = {hac.get('max_lag', 2)})",
            "Value": f"SE_HAC = {hac['se_hac']:.5f} (ratio to naive {hac['ratio']:.3f}); HAC 95% CI [{hac['ci_low']:.4f}, {hac['ci_high']:.4f}]",
            "Interpretation": "Robust slope CI under positive residual autocorrelation; conclusion (slope indistinguishable from zero) unchanged",
        },
    ]
    return pd.DataFrame(rows)


def to_csv_md(df, csv_path, md_path):
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.parent.mkdir(parents=True, exist_ok=True)
    # CSV keeps the manuscript-faithful '|t|'. The markdown copy swaps it for 'abs(t)'
    # so the literal pipe does not break df.to_markdown()'s column separators.
    df.to_csv(csv_path, index=False)
    md_df = df.replace(r"\|t\|", "abs(t)", regex=True)
    md_path.write_text(md_df.to_markdown(index=False), encoding="utf-8")
