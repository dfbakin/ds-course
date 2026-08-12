# Data card — `fleet_incidents.csv`

## Summary

Synthetic health snapshots from a **global fleet**: 180 clusters across 10
regions, six instance types, a dozen OS images, three deploy channels. One row
= one node-day. Two targets: `power_draw_watts` (regression, known sparse
mechanism) and `incident` (classification, the lesson's model-ladder target).
The categorical structure is the point — this is the lesson where one-hot
encoding starts to hurt and native categorical handling starts to pay.

| | |
|---|---|
| Rows | 30 000 |
| Features | 8 numeric + 5 categorical (`cluster_id` has 180 levels) |
| Target (clf) | `incident`, binary, 21.7% positive |
| Target (reg) | `power_draw_watts`, linear in a sparse subset of the numerics |
| Splits | train 18 000 / val 6 000 / test 6 000, stratified on `incident`, materialised |
| Unseen clusters | 6 clusters appear in val/test but **never** in train |
| Seed | 20260808 — regeneration is byte-for-byte reproducible |
| Generator | `../src/generate_dataset.py`; ladder: `../src/calibrate_reference.py` |

## Why synthetic

The lesson's five-model ladder (numeric logreg → +OHE → decision tree →
CatBoost → CatBoost+Optuna) only teaches something if every rung is *reliably*
better than the previous one, by a margin that is visible but honest. No
public dataset gives you control over those margins. Here they are dials:

1. **Categorical offsets** (channel, region, instance, OS, and a 180-level
   hierarchical cluster effect) put ≥ 0.02 AUC between the numeric-only and
   the OHE logistic regression — the "categoricals matter" rung.
2. **Nonlinearities** (a latency threshold amplified in risky channels,
   numeric interactions, channel-modulated slopes) plus the high-cardinality
   cluster feature put ≥ 0.02 AUC between OHE logreg and CatBoost.
3. **Unseen clusters** make the unknown-category problem real, not
   hypothetical.
4. The regression target has a **known sparse support**, so lasso's variable
   selection can be *scored* against ground truth instead of admired.

## Columns

| Column | Type | Notes |
|---|---|---|
| `row_id` | int | 0…29999 |
| `region` | str | 10 levels, non-uniform fleet shares (16%…5%) |
| `cluster_id` | str | **180 levels**, nested in region (`eu-west-c07`). Long-tailed sizes: min 8, median 56, max 1839 rows; 66 clusters < 25 rows |
| `instance_type` | str | 6 levels |
| `deploy_channel` | str | `stable`/`beta`/`canary` ≈ 70/20/10%, strong effect |
| `os_image` | str | 12 levels, mild effects |
| `cpu_util` | float | Beta(4.5, 3.2). Bounded, mildly right-skewed |
| `mem_pressure` | float | Beta(3.0, 3.5). Bounded, near-symmetric. Correlated with `cpu_util` (ρ ≈ 0.6) yet **inert for power draw** — the lasso trap |
| `disk_latency_ms` | float | Log-normal. Heavy right tail |
| `request_rate_rps` | float | **Bimodal** — night and day traffic regimes |
| `error_rate_pct` | float | Gamma(0.7). Spike at zero, long tail |
| `queue_depth` | int | Negative binomial. Discrete, over-dispersed |
| `cache_hit_ratio` | float | Beta(12, 1.6). Left-skewed, piled up near 1 |
| `node_age_days` | float | Mixture: young fleet (log-normal, ~140 d) + old tail (~28% around 950 d) |
| `power_draw_watts` | float | Regression target |
| `incident` | int | Classification target |
| `split` | str | `train` / `val` / `test` |

## How it was generated

**Numerics** come from a **Gaussian copula** (as in lesson 1): a correlated
multivariate normal pushed through the normal CDF, then through each feature's
own inverse CDF — arbitrary marginals, specified correlation structure.

**Fleet topology**: each cluster belongs to exactly one region; cluster sizes
are proportional to Gamma(0.33) weights (long tail, min 8 rows). Six small
clusters (10–45 rows) are forced entirely into val/test.

**Power draw** (the regression mechanism, in full):

```
power = 205 + 145·cpu_util + 0.045·request_rate_rps + 2.8·queue_depth
      + instance_type offset + Normal(0, 16)
```

The other five numerics have TRUE coefficient exactly 0. In sd-units the real
effects are ≈ 23, 17 and 14 W/sd against ≈ 25 W of combined noise+instance
spread — big enough that selection is decidable, small enough that shrinkage
visibly matters. Ground truth in `dgp_metadata.json → power_mechanism`.

**Incident** is Bernoulli(sigmoid(logit)) with, on standardised model features
(heavy tails logged):

- numeric mains (deliberately weak — see Calibration),
- categorical offsets: region (±0.3) + hierarchical cluster-within-region +
  channel (stable 0 / beta 0.55 / canary 1.05) + instance (±0.25) + OS (±0.12),
- the cluster effect is **heavy-tailed**: most clusters ~ Normal(0, 0.55), but
  45% of the below-p75-size clusters draw from Normal(0, 3.4) — small
  special-purpose clusters are where the weirdness lives (31 clusters have
  |offset| > 1.5),
- coarse nonlinearities: `disk_latency > 14.1 ms` (≈ p85) adds 0.85, plus
  another 0.90 on beta/canary; a 0.75·z_cpu·z_mem interaction; a
  0.90·z_cpu·z_mem·z_queue three-way term; channel-modulated error slope
  (beta +0.45, canary +0.80),
- fine slope heterogeneity: every region bends the cpu slope (sd 0.42), every
  instance type the queue slope (sd 0.38), every OS image the error slope
  (sd 0.14), and **every cluster carries random slopes** on six features
  (sd 0.30–0.55) — hierarchical random intercepts *and* slopes.

All true parameters, including all 180 cluster offsets and slopes, are in
`dgp_metadata.json`.

## Calibration — change these constants at your peril

The generator's constants are tuned, not arbitrary. The ladder below was
re-fit after every change until all four margins held simultaneously:

- **`INTERCEPT_INCIDENT = -2.70`** → 21.7% positives (spec: 20–24%).
- **Gap (1)→(2) ≥ 0.02**: driven by `DEPLOY_CHANNEL_OFFSETS` and the cluster
  hierarchy. Achieved +0.049.
- **Gap (2)→(4) ≥ 0.02**: driven by the nonlinear terms, by feeding models the
  *raw* heavy-tailed columns (trees are invariant to monotone transforms;
  linear models are not), and by 180-level `cluster_id` where OHE estimates
  small clusters noisily and unseen clusters not at all. Achieved +0.076.
- **Gap (4)→(5) ≥ 0.003** was by far the hardest dial, and its story is worth
  telling. With ordinary effect sizes, *every* sane CatBoost configuration —
  defaults included — converged to the same test AUC within ±0.001: on 18k
  clean rows, ordered boosting with good defaults simply saturates, which is
  precisely the "CatBoost is robust with little tuning" message… taken so far
  that §8 would have nothing to show. Two changes carved out an honest,
  reproducible margin: (a) numeric mains were attenuated (≈ ×0.55) so label
  noise is real and regularisation choices have consequences; (b) the
  three-way `cpu·mem·queue` term and the per-cluster random slopes bury a
  long tail of individually-small real signal that needs ~800 careful
  iterations to harvest. The default (500 iterations, auto lr 0.0875,
  logloss-based early stop → best iteration 319) leaves that tail on the
  table; the tuned model (val-AUC early stop, best iteration 776) collects
  it. Achieved +0.0032 — visible, and honestly modest.
- **Lasso support recovery**: the alpha window that yields *exactly* the true
  support spans 0.22–15.8 (≈ 1.9 decades) — "a reasonable alpha" is a fair
  description, not luck. Plain LassoCV picks α = 0.20 and keeps one extra
  feature (`error_rate_pct`) — the classic "CV-optimal ≠ sparsest-true" bite,
  which §3 uses as a teaching point.

`tests/test_dataset.py` asserts all of these; `reference_results.json` is the
frozen source of truth for the notebook's §9 table and the slides.

## Reference numbers

Fitted by `src/calibrate_reference.py` (train on train, val for early
stopping/tuning, test ROC-AUC reported; seed 20260808 everywhere):

| # | model | features | test AUC |
|---|---|---|---|
| 1 | LogisticRegression, numeric only | 8 | 0.719 |
| 2 | LogisticRegression + OHE on 5 categoricals | 213 | 0.768 |
| 3 | DecisionTree(max_depth=6, min_samples_leaf=50) | 213 | 0.737 |
| 4 | CatBoost defaults, native cats | 13 | 0.844 |
| 5 | CatBoost + Optuna (TPE, 30 trials) | 13 | 0.847 |

| margin | achieved |
|---|---|
| (1)→(2) | **+0.049** |
| (2)→(4) | **+0.076** |
| (4)→(5) | **+0.0032** |
| Optuna best params | depth 6, lr 0.062, l2_leaf_reg 3.0 |
| Bayes AUC (true logit) | 0.891 |
| oracle: mains only / mains+cats | 0.694 / 0.752 |

Regression side (train-standardised numerics):

| quantity | value |
|---|---|
| Lasso support at α ∈ [0.22, 15.8] | exactly {cpu_util, request_rate_rps, queue_depth} |
| Lasso coefs at α = 2.0 (W/sd) | 22.8 / 17.3 / 14.2, all others 0.0 |
| Lasso test R² at α = 2.0 | 0.71 |
| LassoCV α (5-fold) | 0.20 → keeps `error_rate_pct` too (teaching point) |
| Ridge \|coef_cpu − coef_mem\| at α = 0, 10³, 10⁴, 10⁵ | 24.9 → 19.7 → 9.7 → 2.1 (shrinks together) |

## Limitations

- **Rows are independent** given their cluster; real node-days would be
  autocorrelated within a node. Fine for this lesson, wrong for time-series
  claims.
- **The Bayes AUC (0.891) is not reachable**: much of the cluster-level truth
  (offsets and slopes of clusters with < 25 rows) is statistically
  unlearnable from this sample. The gap between CatBoost (0.844) and the
  ceiling is mostly that, not model failure.
- **No missing values and no label noise beyond the mechanism** — both real
  problems, both out of scope here.
- The numeric main effects are deliberately weak (see Calibration), so the
  absolute AUC of the numeric-only baseline (0.719) is lower than a
  practitioner might expect from 8 informative features. That is the price of
  an honest tuning rung; the *ordering* story is unaffected.
- CatBoost numbers are deterministic per machine but not bit-identical across
  machines/thread counts; tests compare with ±0.01 tolerance and never refit
  the Optuna study.
