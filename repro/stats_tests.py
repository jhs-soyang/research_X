import numpy as np
from scipy import stats as sps


def linreg_l1_vs_year(years, l1):
    """OLS L1 ~ year. Year is centered to 2007 for reproducible intercepts."""
    x = np.asarray(years, dtype=float) - 2007
    y = np.asarray(l1, dtype=float)
    res = sps.linregress(x, y)
    n = len(x)
    df = n - 2
    t_crit = sps.t.ppf(0.975, df)
    z_crit = sps.norm.ppf(0.975)
    ci_low = res.slope - t_crit * res.stderr
    ci_high = res.slope + t_crit * res.stderr
    ci_low_normal = res.slope - z_crit * res.stderr
    ci_high_normal = res.slope + z_crit * res.stderr
    fitted = res.intercept + res.slope * x
    residuals = y - fitted
    return {
        "slope": float(res.slope),
        "intercept": float(res.intercept),
        "r2": float(res.rvalue**2),
        "pvalue": float(res.pvalue),
        "slope_se": float(res.stderr),
        "ci_low": float(ci_low),
        "ci_high": float(ci_high),
        "ci_low_normal": float(ci_low_normal),
        "ci_high_normal": float(ci_high_normal),
        "df": int(df),
        "fitted": fitted,
        "residuals": residuals,
        "n": int(n),
        "x": x,
        "y": y,
    }


def tost_slope(slope, slope_se, df, delta=0.005):
    """Two one-sided t-tests against +/-delta."""
    t_lower = (slope - (-delta)) / slope_se
    t_upper = ((+delta) - slope) / slope_se
    p_lower = 1 - sps.t.cdf(t_lower, df)
    p_upper = 1 - sps.t.cdf(t_upper, df)
    return {
        "t_lower": float(t_lower),
        "t_upper": float(t_upper),
        "p_lower": float(p_lower),
        "p_upper": float(p_upper),
        "p_max": float(max(p_lower, p_upper)),
    }


def chow_test(x, y, split_idx=8):
    """Compare pooled OLS to two sub-period OLS fits. k=2."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    n = len(x)
    k = 2
    # Guard: each regime needs ≥ k points to be identifiable, and the F denominator
    # n - 2k must be positive.
    if split_idx < k or (n - split_idx) < k:
        raise ValueError(
            f"chow_test: each regime needs at least {k} points; got "
            f"split_idx={split_idx}, n={n}."
        )
    if n - 2 * k <= 0:
        raise ValueError(
            f"chow_test: F denominator df = n - 2k = {n - 2 * k} must be > 0."
        )

    def _ssr(xv, yv):
        if len(xv) < 2:
            return 0.0
        slope, intercept, *_ = sps.linregress(xv, yv)
        return float(np.sum((yv - (intercept + slope * xv)) ** 2))

    ssr_p = _ssr(x, y)
    ssr_1 = _ssr(x[:split_idx], y[:split_idx])
    ssr_2 = _ssr(x[split_idx:], y[split_idx:])
    ssr_u = ssr_1 + ssr_2
    df1, df2 = k, n - 2 * k
    F = ((ssr_p - ssr_u) / k) / (ssr_u / df2)
    p = 1 - sps.f.cdf(F, df1, df2)
    return {
        "F": float(F),
        "p": float(p),
        "k": k,
        "df1": df1,
        "df2": df2,
        "ssr_pooled": float(ssr_p),
        "ssr_unrestricted": float(ssr_u),
    }


def shapiro_residuals(residuals):
    W, p = sps.shapiro(residuals)
    return {"W": float(W), "p": float(p)}


def breusch_pagan_nR2(x, residuals):
    """Koenker-style manuscript LM = n * R^2_aux on residuals^2 ~ year."""
    x = np.asarray(x, dtype=float)
    aux_y = np.asarray(residuals, dtype=float) ** 2
    res = sps.linregress(x, aux_y)
    n = len(x)
    LM = n * (res.rvalue**2)
    p = 1 - sps.chi2.cdf(LM, df=1)
    return {"LM": float(LM), "p": float(p)}


def welch_t_period(years, ratios, pre_window=(2007, 2014), post_window=(2015, 2019)):
    """Welch's t-test on yearly ratios split by window.

    Sign convention: scipy ttest_ind(pre, post, equal_var=False) returns
    t = (mean_pre - mean_post) / SE. With believer ratios rising post-2014,
    mean_pre < mean_post, so the raw t is NEGATIVE. The manuscript reports
    |t| (positive) since the two-sided p-value is invariant under sign flip.
    Verification (verify.py) applies abs(t) to match the manuscript.
    """
    years = np.asarray(years)
    ratios = np.asarray(ratios, dtype=float)
    pre_mask = (years >= pre_window[0]) & (years <= pre_window[1])
    post_mask = (years >= post_window[0]) & (years <= post_window[1])
    pre_arr = ratios[pre_mask]
    post_arr = ratios[post_mask]
    if pre_arr.size < 2 or post_arr.size < 2:
        raise ValueError(
            f"welch_t_period: each window must have ≥2 observations "
            f"(pre n={pre_arr.size}, post n={post_arr.size})."
        )
    if pre_arr.var(ddof=1) == 0 and post_arr.var(ddof=1) == 0:
        raise ValueError("welch_t_period: zero variance in both windows; t undefined.")
    t, p = sps.ttest_ind(pre_arr, post_arr, equal_var=False)
    s1 = pre_arr.var(ddof=1)
    n1 = int(pre_mask.sum())
    s2 = post_arr.var(ddof=1)
    n2 = int(post_mask.sum())
    df = (s1 / n1 + s2 / n2) ** 2 / (
        (s1 / n1) ** 2 / (n1 - 1) + (s2 / n2) ** 2 / (n2 - 1)
    )
    return {
        "t": float(t),
        "p": float(p),
        "df": float(df),
        "pre_mean": float(pre_arr.mean()),
        "post_mean": float(post_arr.mean()),
        "pre_n": n1,
        "post_n": n2,
    }


def cross_gender_pearson(male_r, female_r):
    """Per-stance Pearson correlation of yearly ratios between genders.

    Note: a single common shock (such as the 2014→2015 believer step in this
    dataset) can mechanically inflate level-correlations across two series even
    if their year-to-year fluctuations are unrelated. See `cross_gender_diff_pearson`
    and `cross_gender_period_pearson` for tests that filter out that confound.
    """
    out = {}
    for stance in ["believer", "denier", "neutral"]:
        r, p = sps.pearsonr(male_r[stance].values, female_r[stance].values)
        out[stance] = {"r": float(r), "p": float(p)}
    return out


def cross_gender_diff_pearson(male_r, female_r, exclude_transition_year=None):
    """First-difference Pearson correlation of yearly ratios between genders.

    Tests whether year-to-year *changes* in male-attributed and female-attributed
    stance ratios are correlated. With n=13 yearly observations there are 12
    first-difference pairs. Note: a single common shock (such as the 2014→2015
    believer step in this dataset) is preserved as one large Δ pair after
    differencing, so first-difference correlations partially reflect that shock.
    Pass `exclude_transition_year=2014` to also report a sensitivity correlation
    that drops the 2014→2015 Δ pair (n=11).
    """
    if not (male_r.index == female_r.index).all():
        raise ValueError("cross_gender_diff_pearson: male/female index mismatch.")
    years = np.asarray(male_r.index)
    # Defensive: require strictly consecutive integer years (so Δ pairs are well-defined).
    if not np.all(np.diff(years) == 1):
        raise ValueError(
            f"cross_gender_diff_pearson: years are not strictly consecutive integers: {years.tolist()}"
        )
    out = {}
    for stance in ["believer", "denier", "neutral"]:
        m_diff = np.diff(male_r[stance].values)
        f_diff = np.diff(female_r[stance].values)
        r, p = sps.pearsonr(m_diff, f_diff)
        result = {"r": float(r), "p": float(p), "n": int(len(m_diff))}
        if exclude_transition_year is not None:
            # The Δ pair indexed at i corresponds to years[i] -> years[i+1]. The transition
            # year must therefore have a successor year in the index.
            if exclude_transition_year not in set(years.tolist()):
                raise ValueError(
                    f"cross_gender_diff_pearson: exclude_transition_year={exclude_transition_year} not in index {years.tolist()}."
                )
            if exclude_transition_year == int(years[-1]):
                raise ValueError(
                    f"cross_gender_diff_pearson: exclude_transition_year={exclude_transition_year} is the last year in the index; "
                    f"there is no Δ pair starting at the last year to exclude."
                )
            drop_idx = int(np.where(years == exclude_transition_year)[0][0])
            mask = np.ones(len(m_diff), dtype=bool)
            mask[drop_idx] = False
            r_excl, p_excl = sps.pearsonr(m_diff[mask], f_diff[mask])
            result["r_excl_transition"] = float(r_excl)
            result["p_excl_transition"] = float(p_excl)
            result["n_excl_transition"] = int(mask.sum())
            result["transition_year"] = int(exclude_transition_year)
        out[stance] = result
    return out


def cross_gender_period_pearson(male_r, female_r, pre_window=(2007, 2014), post_window=(2015, 2019)):
    """Period-stratified Pearson correlation: 2007-2014 (n=8) and 2015-2019 (n=5)."""
    if not (male_r.index == female_r.index).all():
        raise ValueError("cross_gender_period_pearson: male/female index mismatch.")
    out = {}
    male_idx = np.asarray(male_r.index)
    pre_mask = (male_idx >= pre_window[0]) & (male_idx <= pre_window[1])
    post_mask = (male_idx >= post_window[0]) & (male_idx <= post_window[1])
    n_pre = int(pre_mask.sum())
    n_post = int(post_mask.sum())
    if n_pre < 2 or n_post < 2:
        raise ValueError(
            f"cross_gender_period_pearson: both windows need ≥ 2 observations "
            f"(pre n={n_pre}, post n={n_post})."
        )
    for stance in ["believer", "denier", "neutral"]:
        m_pre = male_r[stance].values[pre_mask]
        f_pre = female_r[stance].values[pre_mask]
        m_post = male_r[stance].values[post_mask]
        f_post = female_r[stance].values[post_mask]
        # Guard against constant-input degenerate cases (Pearson r is undefined).
        for label, arr in [("male pre", m_pre), ("female pre", f_pre), ("male post", m_post), ("female post", f_post)]:
            if np.all(arr == arr[0]):
                raise ValueError(
                    f"cross_gender_period_pearson: {label} {stance} ratios are constant; Pearson r undefined."
                )
        r_pre, p_pre = sps.pearsonr(m_pre, f_pre)
        r_post, p_post = sps.pearsonr(m_post, f_post)
        out[stance] = {
            "r_pre": float(r_pre), "p_pre": float(p_pre), "n_pre": n_pre,
            "r_post": float(r_post), "p_post": float(p_post), "n_post": n_post,
        }
    return out


def durbin_watson(residuals):
    """Durbin-Watson statistic for residual autocorrelation.

    DW ≈ 2: no autocorrelation. DW < 1.5: positive autocorrelation suspected.
    DW > 2.5: negative autocorrelation suspected. The bounds reported below are
    appropriate for the small-n (n ≈ 13) regime used in this study; for other
    n the Savin–White (1977) tables should be consulted.
    """
    r = np.asarray(residuals, dtype=float)
    if r.size < 2:
        return {"DW": float("nan"), "verdict": "n_too_small"}
    denom = float(np.sum(r ** 2))
    if denom == 0:
        return {"DW": float("nan"), "verdict": "zero_residual_variance"}
    DW = float(np.sum(np.diff(r) ** 2) / denom)
    if DW < 1.010:
        verdict = "positive autocorrelation"
    elif DW < 1.340:
        verdict = "inconclusive (likely positive)"
    elif DW <= 2.60:
        verdict = "no detectable autocorrelation"
    elif DW < 2.95:
        verdict = "inconclusive (likely negative)"
    else:
        verdict = "negative autocorrelation"
    return {"DW": DW, "verdict": verdict}


def newey_west_slope_se(x, residuals, max_lag=2):
    """Newey-West HAC standard error for the OLS slope (univariate predictor).

    Implements the Bartlett-kernel HAC sandwich for the simple linear model
    y = alpha + beta*x + e. With n=13 and bandwidth 2 (an ad hoc small-n choice
    consistent with the Newey & West 1987 / Andrews 1991 family of bandwidth
    selectors), this serves as a sensitivity check against the positive
    autocorrelation that the Durbin-Watson statistic flags as 'inconclusive
    but leaning positive'.

    Returns dict with:
        se_naive   -- the OLS sigma/sqrt(Sxx) SE
        se_hac     -- the Newey-West HAC SE
        ratio      -- se_hac / se_naive (1.0 if no autocorrelation)
        max_lag    -- bandwidth used (default 2)

    NOTE: This function returns SE only. The 95% CI is post-computed by main.py
    using the slope and t.ppf(0.975, df). Standalone callers must inject ci_low
    and ci_high before passing the dict to tables.build_table2 or verify.run_checks.
    """
    x = np.asarray(x, dtype=float)
    e = np.asarray(residuals, dtype=float)
    n = x.size
    if n != e.size or n < max_lag + 3:
        raise ValueError(
            f"newey_west_slope_se: incompatible sizes or insufficient n "
            f"(n={n}, max_lag={max_lag})."
        )

    x_centered = x - x.mean()
    Sxx = float(np.sum(x_centered ** 2))
    if Sxx == 0:
        raise ValueError("newey_west_slope_se: zero variance in x")

    # Naive OLS slope SE
    sigma2 = float(np.sum(e ** 2) / (n - 2))
    se_naive = (sigma2 / Sxx) ** 0.5

    # HAC sandwich for slope: var(beta) = (Sxx)^-1 * S * (Sxx)^-1
    # where S = Σ x_t^2 e_t^2 + 2 Σ_{l=1}^L w_l Σ_t x_t x_{t-l} e_t e_{t-l}
    # using Bartlett weights w_l = 1 - l/(L+1).
    psi = x_centered * e
    S = float(np.sum(psi ** 2))
    for lag in range(1, max_lag + 1):
        w = 1.0 - lag / (max_lag + 1)
        S += 2.0 * w * float(np.sum(psi[lag:] * psi[:-lag]))
    var_hac = S / (Sxx ** 2)
    # Small-sample adjustment (n / (n - k)) with k = 2 parameters
    var_hac *= n / (n - 2)
    se_hac = float(max(var_hac, 0.0)) ** 0.5

    return {
        "se_naive": float(se_naive),
        "se_hac": float(se_hac),
        "ratio": float(se_hac / se_naive) if se_naive > 0 else float("nan"),
        "max_lag": int(max_lag),
    }


def welch_t_l1_pre_post(years, l1, pre_window=(2007, 2014), post_window=(2015, 2019)):
    """Welch's t-test on yearly L1 distance, pre vs post 2014/15 split.

    Tests whether the gender L1 gap differs between the two regimes — directly
    addressing the question whether the 2014–2015 step changed the aggregate gap.
    Returns t (raw, signed), p (two-sided), df, and pre/post means.
    """
    years = np.asarray(years)
    l1 = np.asarray(l1, dtype=float)
    pre_mask = (years >= pre_window[0]) & (years <= pre_window[1])
    post_mask = (years >= post_window[0]) & (years <= post_window[1])
    pre = l1[pre_mask]
    post = l1[post_mask]
    if pre.size < 2 or post.size < 2:
        raise ValueError(
            f"welch_t_l1_pre_post: each window must have ≥2 observations "
            f"(pre n={pre.size}, post n={post.size})."
        )
    if pre.var(ddof=1) == 0 and post.var(ddof=1) == 0:
        raise ValueError("welch_t_l1_pre_post: zero variance in both windows; t undefined.")
    t, p = sps.ttest_ind(pre, post, equal_var=False)
    s1 = pre.var(ddof=1)
    n1 = int(pre_mask.sum())
    s2 = post.var(ddof=1)
    n2 = int(post_mask.sum())
    df_w = (s1 / n1 + s2 / n2) ** 2 / (
        (s1 / n1) ** 2 / (n1 - 1) + (s2 / n2) ** 2 / (n2 - 1)
    )
    return {
        "t": float(t),
        "p": float(p),
        "df": float(df_w),
        "pre_mean": float(pre.mean()),
        "post_mean": float(post.mean()),
        "drop_fraction": float((pre.mean() - post.mean()) / pre.mean()) if pre.mean() else float("nan"),
        "pre_n": n1,
        "post_n": n2,
    }
