from pathlib import Path
import hashlib
import json

import config as cfg
from repro import bootstrap, io_data, l1_metric, stats_tests, tables, verify
from repro.fig1_l1_trend import make_figure as make_fig1
from repro.fig2_dual_axis import make_figure as make_fig2
from repro.fig3_heatmap_3row import make_figure as make_fig3
from repro.fig4_diff_heatmap import make_figure as make_fig4
from repro.fig5_radar import make_figure as make_fig5
from repro.fig6_correlation_panel import make_figure as make_fig6


def sha256(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    cfg.OUT_DIR.mkdir(parents=True, exist_ok=True)
    cfg.FIG_DIR.mkdir(parents=True, exist_ok=True)
    cfg.TABLE_DIR.mkdir(parents=True, exist_ok=True)
    cfg.INTER_DIR.mkdir(parents=True, exist_ok=True)

    df = io_data.load_yearly_stance(2007, 2019, cfg.DATA_DIR)
    male_r, female_r = io_data.pivot_ratios(df)

    ratio_rows = df[
        ["Year", "Gender", "Believer_Ratio", "Denier_Ratio", "Neutral_Ratio"]
    ].sort_values(["Year", "Gender"])
    ratio_rows.to_csv(cfg.INTER_DIR / "gender_stance_ratios.csv", index=False)

    l1_df = l1_metric.compute_annual_l1(male_r, female_r)
    l1_df.to_csv(cfg.INTER_DIR / "annual_l1.csv", index=False)

    ci_df = bootstrap.bootstrap_l1_ci(df, B=cfg.B_BOOTSTRAP, seed=cfg.SEED)
    ci_df.to_csv(cfg.INTER_DIR / "bootstrap_ci.csv", index=False)

    lr = stats_tests.linreg_l1_vs_year(l1_df.Year, l1_df.L1)
    tost = stats_tests.tost_slope(
        lr["slope"], lr["slope_se"], lr["df"], delta=cfg.TOST_DELTA
    )
    chow = stats_tests.chow_test(lr["x"], lr["y"], split_idx=cfg.CHOW_SPLIT_INDEX)
    sh = stats_tests.shapiro_residuals(lr["residuals"])
    bp = stats_tests.breusch_pagan_nR2(lr["x"], lr["residuals"])
    welch_f = stats_tests.welch_t_period(
        female_r.index.values,
        female_r.believer.values,
        pre_window=cfg.PRE_WINDOW,
        post_window=cfg.POST_WINDOW,
    )
    welch_m = stats_tests.welch_t_period(
        male_r.index.values,
        male_r.believer.values,
        pre_window=cfg.PRE_WINDOW,
        post_window=cfg.POST_WINDOW,
    )
    pear = stats_tests.cross_gender_pearson(male_r, female_r)
    pear_diff = stats_tests.cross_gender_diff_pearson(male_r, female_r, exclude_transition_year=2014)
    pear_period = stats_tests.cross_gender_period_pearson(
        male_r, female_r, pre_window=cfg.PRE_WINDOW, post_window=cfg.POST_WINDOW
    )
    dw = stats_tests.durbin_watson(lr["residuals"])
    welch_l1 = stats_tests.welch_t_l1_pre_post(
        l1_df.Year.values,
        l1_df.L1.values,
        pre_window=cfg.PRE_WINDOW,
        post_window=cfg.POST_WINDOW,
    )
    hac = stats_tests.newey_west_slope_se(lr["x"], lr["residuals"], max_lag=2)
    # t-distributed CI using HAC SE
    from scipy import stats as _sps
    _t_crit = _sps.t.ppf(0.975, lr["df"])
    hac["ci_low"] = lr["slope"] - _t_crit * hac["se_hac"]
    hac["ci_high"] = lr["slope"] + _t_crit * hac["se_hac"]
    # Bandwidth sensitivity quoted in Methods (iv): L=1 and L=3 alternatives.
    for _L in (1, 3):
        hac[f"se_hac_L{_L}"] = stats_tests.newey_west_slope_se(
            lr["x"], lr["residuals"], max_lag=_L
        )["se_hac"]

    t1 = tables.build_table1(df)
    t2 = tables.build_table2({"lr": lr, "tost": tost, "chow": chow, "sh": sh, "bp": bp, "dw": dw, "welch_l1": welch_l1, "hac": hac})
    tables.to_csv_md(t1, cfg.TABLE_DIR / "table1.csv", cfg.TABLE_DIR / "table1.md")
    tables.to_csv_md(t2, cfg.TABLE_DIR / "table2.csv", cfg.TABLE_DIR / "table2.md")

    fig_paths = [
        make_fig1(l1_df, ci_df, lr),
        make_fig2(l1_df, male_r, female_r),
        make_fig3(male_r, female_r),
        make_fig4(male_r, female_r),
        make_fig5(male_r, female_r),
        make_fig6(male_r, female_r, l1_df, lr),
    ]

    pass_n, total_n, report = verify.run_checks(
        {
            "l1_df": l1_df,
            "lr": lr,
            "tost": tost,
            "chow": chow,
            "sh": sh,
            "bp": bp,
            "welch_f": welch_f,
            "welch_m": welch_m,
            "pear": pear,
            "ci_df": ci_df,
            "dw": dw,
            "welch_l1": welch_l1,
            "hac": hac,
            "pear_diff": pear_diff,
            "pear_period": pear_period,
        }
    )
    (cfg.OUT_DIR / "verification_report.md").write_text(report, encoding="utf-8")

    manifest_paths = [
        cfg.INTER_DIR / "annual_l1.csv",
        cfg.INTER_DIR / "gender_stance_ratios.csv",
        cfg.INTER_DIR / "bootstrap_ci.csv",
        cfg.TABLE_DIR / "table1.csv",
        cfg.TABLE_DIR / "table1.md",
        cfg.TABLE_DIR / "table2.csv",
        cfg.TABLE_DIR / "table2.md",
        cfg.OUT_DIR / "verification_report.md",
        *fig_paths,
    ]
    manifest = []
    for p in manifest_paths:
        if p.exists():
            manifest.append(
                {
                    # Repository-relative so the manifest is portable and diffable.
                    "path": p.relative_to(cfg.ROOT).as_posix(),
                    "sha256": sha256(p),
                    "size": p.stat().st_size,
                }
            )
    (cfg.OUT_DIR / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )

    print(f"Summary: {pass_n}/{total_n} checks PASS")


if __name__ == "__main__":
    main()
