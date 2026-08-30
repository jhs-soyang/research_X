# Reproduction pipeline — Temporal Stability of Gender Differences in Climate Change Discourse on Twitter (2007–2019)

Deterministic reproduction pipeline for the statistical analyses, tables, and figures
reported in Suh & Yang, *Temporal Stability of Gender Differences in Climate Change
Discourse on Twitter (2007–2019)*.

Everything needed to reproduce the analysis is in this repository. The per-year
gender × stance aggregates that the analysis consumes are included under `data/`;
the raw ~1.9 GB corpus is not, and is only required if you want to regenerate those
aggregates from scratch (see *Regenerating the inputs* below).

## Run

```bash
pip install -r requirements.txt
python main.py
```

The last line should read `Summary: 45/45 checks PASS`. Every deterministic statistic is
checked against the value printed in the manuscript; bootstrap confidence-interval rows
are reported as INFO rather than PASS/FAIL, because NumPy does not guarantee identical
`Generator` streams across versions.

## Manuscript reference values

Checked automatically by `repro/verify.py`; see `outputs/verification_report.md` for the
full list, which is broader than the summary below.

| Quantity | Manuscript value |
|---|---|
| L1 mean / min / max | 0.071 / 0.025 / 0.132 |
| Slope (per year) | −0.00079 |
| Slope p-value | 0.767 |
| R² | 0.008 |
| Slope 95% CI (t, df = 11) | [−0.0065, 0.0049] |
| TOST p (Δ = 0.005/yr) | 0.067 (p_lower 0.067, p_upper 0.024) |
| Chow F / p (split 2014/2015) | 4.42 / 0.046 |
| Shapiro–Wilk W / p | 0.972 / 0.915 |
| Breusch–Pagan LM / p | 2.43 / 0.119 |
| Durbin–Watson | 1.155 |
| Newey–West HAC slope SE (L = 1 / 2 / 3) | 0.00253 / 0.00248 / 0.00235 |
| Welch female t / df / p | 11.68 / 10.03 / 3.67e−7 |
| Welch male t / df / p | 12.32 / 9.30 / 4.56e−7 |
| Welch on L1 t / df / p | 1.99 / 9.66 / 0.076 |
| Pearson believer / denier / neutral r | 0.995 / 0.957 / 0.990 |

## Layout

```
config.py          paths, seed, window and split constants
main.py            end-to-end driver: load -> analyse -> tables -> figures -> verify
preprocess.py      regenerates data/ from the raw corpus (optional)
repro/             io_data, l1_metric, bootstrap, stats_tests, tables, figures, fig1..fig6, verify
data/              per-year gender x stance aggregates, 2007-2019 (analysis inputs)
outputs/           figures, tables, intermediate CSVs, manifest, verification report
```

## Outputs

- `outputs/figures/Figure1.png` … `Figure6.png` (300 dpi)
- `outputs/tables/table1.{csv,md}`, `table2.{csv,md}`
- `outputs/intermediate/annual_l1.csv`, `bootstrap_ci.csv`, `gender_stance_ratios.csv`
- `outputs/manifest.json` (repository-relative paths + SHA-256 for every artifact)
- `outputs/verification_report.md`

## Regenerating the inputs

`preprocess.py` rebuilds `data/` from the raw Climate Change Twitter Dataset, streaming it
in 200k-row chunks. Download the corpus from the source cited in the manuscript, then:

```bash
python preprocess.py --raw-csv /path/to/dataset.csv --verify
```

`--verify` compares the regenerated aggregates against `data/` at the level of row
membership and numeric values, and exits non-zero on any mismatch. It also writes
`outputs/intermediate/preprocessing_audit.csv`, which retains every year present in the
raw corpus (including 2006) and every gender bucket (including `Undefined`), flagging
which rows enter the analysis — this is the artifact that lets a reader re-derive the
sample-size statements in the manuscript.

## Analysis notes

- Bootstrap seed = 20260502, B = 1000, via `np.random.default_rng`.
- Chow split index = 8 (regime 1: 2007–2014, regime 2: 2015–2019). The split point was
  chosen visually from the believer-ratio series before testing, so the Chow result is
  exploratory; an exhaustive scan over all admissible splits locates a larger F elsewhere.
- TOST equivalence margin Δ = 0.005/yr, as specified in the manuscript.
- Newey–West HAC uses a Bartlett kernel, bandwidth L = 2, with a finite-sample factor
  n/(n − k) = 13/11; the sensitivity interval uses t(0.975, 11), not a normal quantile.
- Durbin–Watson bounds for n = 13, k′ = 1 at α = 0.05 are dL = 1.010, dU = 1.340
  (Savin–White).

## Paths

`config.py` resolves the repository root from its own location, so the pipeline runs from
a clean checkout anywhere. Override the input location with `CCTD_DATA_DIR` if you keep
the aggregates outside the repository; `preprocess.py` additionally accepts `--raw-csv`,
`--data-dir`, and `--out-dir`.

## Environment

Python 3.9.6 with the versions pinned in `requirements.txt`. These are the versions used
to produce the figures and statistics reported in the manuscript.
