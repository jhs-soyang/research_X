import math


REFS = [
    ("L1_mean", 0.071, 5e-4, "abs"),
    ("L1_min", 0.025, 5e-4, "abs"),
    ("L1_max", 0.132, 5e-4, "abs"),
    ("slope", -0.00079, 5e-6, "abs"),
    ("slope_p", 0.767, 5e-3, "abs"),
    ("r2", 0.008, 5e-4, "abs"),
    # t-based 95% CI with df=n-2=11; t.ppf(0.975, 11)=2.20099
    ("ci_low", -0.00652, 5e-5, "abs"),
    ("ci_high", 0.00494, 5e-5, "abs"),
    ("tost_p", 0.067, 5e-3, "abs"),
    ("chow_F", 4.42, 0.05, "abs"),
    ("chow_p", 0.046, 5e-3, "abs"),
    ("shapiro_W", 0.972, 5e-3, "abs"),
    ("shapiro_p", 0.915, 5e-3, "abs"),
    ("bp_LM", 2.43, 0.05, "abs"),
    ("bp_p", 0.119, 5e-3, "abs"),
    ("welch_f_t", 11.68, 0.1, "abs"),
    ("welch_f_p", 3.67e-7, math.log10(2), "log10"),
    ("welch_m_t", 12.32, 0.1, "abs"),
    ("welch_m_p", 4.56e-7, math.log10(2), "log10"),
    ("pearson_believer_r", 0.995, 5e-3, "abs"),
    ("pearson_denier_r", 0.957, 5e-3, "abs"),
    ("pearson_neutral_r", 0.990, 5e-3, "abs"),
    # Newly added checks for v2.1 (Durbin-Watson autocorrelation and Welch's t on L1 itself)
    ("durbin_watson", 1.155, 0.01, "abs"),
    ("welch_l1_t", 1.991, 0.05, "abs"),
    ("welch_l1_p", 0.0756, 5e-3, "abs"),
    ("welch_l1_pre_mean", 0.0828, 5e-4, "abs"),
    ("welch_l1_post_mean", 0.0531, 5e-4, "abs"),
    # Newey-West HAC slope SE robustness (v2.2 addition; bandwidth L=2 Bartlett)
    ("hac_slope_se", 0.00248, 5e-5, "abs"),
    ("hac_se_ratio", 0.9527, 0.01, "abs"),
    ("hac_ci_low", -0.00625, 5e-5, "abs"),
    ("hac_ci_high", 0.00467, 5e-5, "abs"),
    # Values printed in the manuscript that previously sat outside the check set.
    # Welch-Satterthwaite degrees of freedom (Methods vi; Results).
    ("welch_f_df", 10.03, 0.05, "abs"),
    ("welch_m_df", 9.30, 0.05, "abs"),
    ("welch_l1_df", 9.66, 0.05, "abs"),
    # Believer-ratio window means (Results, Synchronized compositional shift).
    ("welch_f_pre_mean", 0.397, 5e-4, "abs"),
    ("welch_f_post_mean", 0.759, 5e-4, "abs"),
    ("welch_m_pre_mean", 0.403, 5e-4, "abs"),
    ("welch_m_post_mean", 0.735, 5e-4, "abs"),
    # Cross-gender Pearson p-values (Results; abstract).
    ("pearson_believer_p", 3.52e-12, math.log10(2), "log10"),
    ("pearson_denier_p", 3.11e-7, math.log10(2), "log10"),
    ("pearson_neutral_p", 9.93e-11, math.log10(2), "log10"),
    # TOST one-sided components. The manuscript reports p_max; p_upper is the
    # informative half (the widening direction is rejected one-sidedly).
    ("tost_p_lower", 0.0672, 5e-3, "abs"),
    ("tost_p_upper", 0.0240, 5e-3, "abs"),
    # HAC bandwidth sensitivity quoted in Methods (iv).
    ("hac_slope_se_L1", 0.00253, 5e-5, "abs"),
    ("hac_slope_se_L3", 0.00235, 5e-5, "abs"),
]


def _computed(inputs):
    l1_df = inputs["l1_df"]
    lr = inputs["lr"]
    tost = inputs["tost"]
    chow = inputs["chow"]
    sh = inputs["sh"]
    bp = inputs["bp"]
    welch_f = inputs["welch_f"]
    welch_m = inputs["welch_m"]
    pear = inputs["pear"]
    dw = inputs.get("dw", {"DW": float("nan")})
    welch_l1 = inputs.get("welch_l1", {"t": float("nan"), "p": float("nan"), "pre_mean": float("nan"), "post_mean": float("nan")})
    hac = inputs.get("hac", {"se_hac": float("nan"), "ratio": float("nan"), "ci_low": float("nan"), "ci_high": float("nan")})
    pear_diff = inputs.get("pear_diff", {})
    pear_period = inputs.get("pear_period", {})
    return {
        "L1_mean": float(l1_df["L1"].mean()),
        "L1_min": float(l1_df["L1"].min()),
        "L1_max": float(l1_df["L1"].max()),
        "slope": float(lr["slope"]),
        "slope_p": float(lr["pvalue"]),
        "r2": float(lr["r2"]),
        "ci_low": float(lr["ci_low"]),
        "ci_high": float(lr["ci_high"]),
        "tost_p": float(tost["p_max"]),
        "chow_F": float(chow["F"]),
        "chow_p": float(chow["p"]),
        "shapiro_W": float(sh["W"]),
        "shapiro_p": float(sh["p"]),
        "bp_LM": float(bp["LM"]),
        "bp_p": float(bp["p"]),
        "welch_f_t": abs(float(welch_f["t"])),
        "welch_f_p": float(welch_f["p"]),
        "welch_m_t": abs(float(welch_m["t"])),
        "welch_m_p": float(welch_m["p"]),
        "pearson_believer_r": float(pear["believer"]["r"]),
        "pearson_denier_r": float(pear["denier"]["r"]),
        "pearson_neutral_r": float(pear["neutral"]["r"]),
        "durbin_watson": float(dw["DW"]),
        "welch_l1_t": abs(float(welch_l1["t"])),
        "welch_l1_p": float(welch_l1["p"]),
        "welch_l1_pre_mean": float(welch_l1["pre_mean"]),
        "welch_l1_post_mean": float(welch_l1["post_mean"]),
        "hac_slope_se": float(hac["se_hac"]),
        "hac_se_ratio": float(hac["ratio"]),
        "hac_ci_low": float(hac.get("ci_low", float("nan"))),
        "hac_ci_high": float(hac.get("ci_high", float("nan"))),
        "welch_f_df": float(welch_f["df"]),
        "welch_m_df": float(welch_m["df"]),
        "welch_l1_df": float(welch_l1["df"]),
        "welch_f_pre_mean": float(welch_f["pre_mean"]),
        "welch_f_post_mean": float(welch_f["post_mean"]),
        "welch_m_pre_mean": float(welch_m["pre_mean"]),
        "welch_m_post_mean": float(welch_m["post_mean"]),
        "pearson_believer_p": float(pear["believer"]["p"]),
        "pearson_denier_p": float(pear["denier"]["p"]),
        "pearson_neutral_p": float(pear["neutral"]["p"]),
        "tost_p_lower": float(tost["p_lower"]),
        "tost_p_upper": float(tost["p_upper"]),
        "hac_slope_se_L1": float(hac.get("se_hac_L1", float("nan"))),
        "hac_slope_se_L3": float(hac.get("se_hac_L3", float("nan"))),
    }


def _passes(value, ref, tol, mode):
    if mode == "log10":
        if value <= 0 or ref <= 0:
            return False
        return abs(math.log10(value) - math.log10(ref)) < tol
    return abs(value - ref) <= tol


def _fmt(value):
    if isinstance(value, float):
        if value != 0 and (abs(value) < 1e-4 or abs(value) >= 1e5):
            return f"{value:.8g}"
        return f"{value:.8f}"
    return str(value)


def run_checks(inputs):
    values = _computed(inputs)
    lines = [
        "# Verification Report",
        "",
        "Deterministic rows are PASS/FAIL checks against the manuscript reference values. Bootstrap CI rows are recorded as INFO only.",
        "Slope 95% CI uses the t-distribution with df=n-2=11 (t.ppf(0.975, 11)=2.20099), which is standard for n=13. The earlier z=1.96 normal-approximation CI [-0.0059, 0.0043] is no longer the reference.",
        "",
    ]
    pass_n = 0
    total_n = 0
    fail_labels = []

    for label, ref, tol, mode in REFS:
        value = values[label]
        ok = _passes(value, ref, tol, mode)
        status = "PASS" if ok else "FAIL"
        pass_n += int(ok)
        total_n += 1
        if not ok:
            fail_labels.append(label)
        tol_text = f"log10 tol {tol:.8g}" if mode == "log10" else f"tol {tol:.8g}"
        lines.append(
            f"[STATS] {status} {label} = {_fmt(value)} (ref {_fmt(ref)}, {tol_text})"
        )

    ci_df = inputs["ci_df"].set_index("Year")
    for year in [2008, 2014]:
        for suffix in ["lo", "hi"]:
            col = f"L1_{suffix}"
            value = float(ci_df.loc[year, col])
            lines.append(f"[STATS] INFO bootstrap_ci_{year}_{suffix} = {_fmt(value)}")

    # First-difference and period-stratified Pearson correlations (v2.3 — INFO).
    # These are robustness checks that filter out the 2014–2015 common shock.
    pear_diff = inputs.get("pear_diff", {})
    if pear_diff:
        for stance in ["believer", "denier", "neutral"]:
            d = pear_diff.get(stance, {})
            r = float(d.get("r", float("nan")))
            p = float(d.get("p", float("nan")))
            n = int(d.get("n", 0))
            lines.append(f"[STATS] INFO pearson_diff_{stance}_r = {_fmt(r)} (p={_fmt(p)}, n={n})")
            if "r_excl_transition" in d:
                r_e = float(d["r_excl_transition"])
                p_e = float(d["p_excl_transition"])
                n_e = int(d["n_excl_transition"])
                lines.append(
                    f"[STATS] INFO pearson_diff_{stance}_r_excl_transition = {_fmt(r_e)} "
                    f"(p={_fmt(p_e)}, n={n_e})"
                )
    pear_period = inputs.get("pear_period", {})
    if pear_period:
        for stance in ["believer", "denier", "neutral"]:
            d = pear_period.get(stance, {})
            lines.append(
                f"[STATS] INFO pearson_period_{stance}_pre_r = {_fmt(d.get('r_pre'))} "
                f"(p={_fmt(d.get('p_pre'))}, n={d.get('n_pre')})"
            )
            lines.append(
                f"[STATS] INFO pearson_period_{stance}_post_r = {_fmt(d.get('r_post'))} "
                f"(p={_fmt(d.get('p_post'))}, n={d.get('n_post')})"
            )

    lines.extend(["", f"Summary: {pass_n}/{total_n} checks PASS"])
    if fail_labels:
        lines.extend(
            [
                "",
                "Divergence notes:",
                "The following deterministic checks did not match the reference table: "
                + ", ".join(fail_labels)
                + ".",
            ]
        )
    else:
        lines.extend(["", "Divergence notes: No deterministic divergences."])
    return pass_n, total_n, "\n".join(lines) + "\n"
