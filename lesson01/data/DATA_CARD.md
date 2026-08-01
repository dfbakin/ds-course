# Data card — `service_telemetry.csv`

## Summary

Synthetic telemetry from a fleet of 8 API nodes. One row per 5-minute window per
node. The target is whether that window contained an **incident** (an SLA
breach). Partway through the observation period a new version is deployed, which
is what makes "did something change?" a concrete question.

| | |
|---|---|
| Rows | 20 000 |
| Features | 8 numeric |
| Target | `incident`, binary, 24.7% positive |
| Splits | train 12 000 / val 4 000 / test 4 000, stratified, materialised in the CSV |
| Period | 2026-05-04 onward, 5-minute windows, 8 nodes |
| Seed | 20260729 — regeneration is byte-for-byte reproducible |
| Generator | `../src/generate_dataset.py` |

## Why synthetic

The lesson needs four properties at once, and no classic tabular dataset has all
four:

1. **Visibly different marginals** (heavy-tailed, bimodal, discrete, bounded,
   symmetric) so the distribution section has something to say.
2. **A controlled correlation structure**, so covariation is real but not
   trivial.
3. **Genuine pairwise interactions in the label mechanism**, so a cross-feature
   model is *reliably but only slightly* better — the regime where a
   significance test is interesting rather than decorative.
4. **Known ground-truth anomalies**, so §11's detector can be *scored* rather
   than merely admired. On real data that section would be unfalsifiable.

## Columns

| Column | Type | Notes |
|---|---|---|
| `row_id` | int | 0…19999, matches `anomaly_ground_truth.csv` |
| `window_start` | timestamp | 5-minute windows |
| `node_id` | str | `node-00` … `node-07` |
| `phase` | str | `pre_deploy` (11 000) / `post_deploy` (9 000) |
| `cpu_util` | float | Beta(4.5, 3.2). Bounded, mildly right-skewed |
| `mem_pressure` | float | Beta(3.0, 3.5). Bounded, near-symmetric |
| `disk_latency_ms` | float | Log-normal. **Heavy right tail** (skew ≈ 14) |
| `request_rate_rps` | float | **Bimodal** — a night and a day traffic regime |
| `error_rate_pct` | float | Gamma(0.7). Spike at zero, long tail |
| `queue_depth` | int | Negative binomial. **Discrete**, over-dispersed |
| `cache_hit_ratio` | float | Beta(12, 1.6). **Left**-skewed, piled up near 1 |
| `temperature_c` | float | Normal(46, 6). The one symmetric bell |
| `incident` | int | Target |
| `split` | str | `train` / `val` / `test` |

## How it was generated

**Features** come from a **Gaussian copula**: sample a correlated multivariate
normal, push it through the normal CDF to get uniforms, then through each
feature's own inverse CDF. This decouples "what shape is each feature" from
"how do features co-vary", giving arbitrary marginals with a specified
correlation structure. Achieved correlations (copula attenuation is mild):

| pair | ρ |
|---|---|
| `cpu_util` ↔ `temperature_c` | 0.65 |
| `cpu_util` ↔ `mem_pressure` | 0.55 |
| `log_request_rate` ↔ `queue_depth` | 0.46 |
| `cpu_util` ↔ `log_request_rate` | 0.44 |
| `log_disk_latency` ↔ `log1p_error_rate` | 0.36 |
| `log_request_rate` ↔ `cache_hit_ratio` | −0.27 |

**Labels** come from a logistic model on the *standardised modelling
representation* (heavy-tailed features logged):

```
logit = β₀ + Σ βᵢ·zᵢ + Σ γⱼₖ·zⱼ·zₖ
y ~ Bernoulli(sigmoid(logit))
```

with six interaction/quadratic terms. A linear model on plain features can only
reach the main effects; the cross-feature model can reach the rest. **That gap
is the entire point of §5 and §10.** True coefficients are in
`dgp_metadata.json`.

## Calibration — change these constants at your peril

The generator's constants are tuned, not arbitrary. Three properties the lesson
depends on:

- **`INTERCEPT = -2.25`** → 24.7% positives. Makes §6's "a model that never
  fires is already 75% accurate" argument land.
- **`GAMMA`** → test-AUC gap of **+0.020 ± 0.002** across four seeds between the
  plain and cross-feature models. Small enough that you cannot eyeball it,
  large enough that DeLong finds it. With a +0.10 gap, §10 teaches nothing.
- **`POST_DEPLOY_SHIFT`** + the anomaly cohort → an incident-rate increase of
  about **+3 percentage points** (z ≈ 5). Detectable with a test, invisible on
  a bar chart. The pre-deploy A/A split is null, which is what makes the real
  result trustworthy.

`tests/test_dataset.py` asserts all of these.

## Ground truth: `anomaly_ground_truth.csv`

Kept in a **separate file** so the notebook can run its detector before looking
at the answers. 360 injected anomalies (1.8%), post-deploy only:

| kind | n | what it is |
|---|---|---|
| `silent_failure` | 240 | A config regression shipped with the new version. Telemetry looks healthy; the window is an incident anyway (P = 0.95). |
| `sensor_glitch` | 120 | A broken disk-latency probe reporting ~340 ms while the service is fine. `incident` forced to 0. |

**Design note on `silent_failure`.** Their feature values are *not* synthesised.
An earlier version of the generator resampled them from a benign distribution,
which quietly broke the lesson: it gave the cohort a distinctive "every metric
is unusually benign at once" signature — exactly the kind of conjunction a
cross-feature model can learn. Model B then won mostly by detecting anomalies
rather than by capturing interactions, which is not the story §5 tells
(measured: +0.034 AUC on all rows but only +0.009 on clean rows). The fix was to
*select* rows that already look benign and leave their telemetry untouched, so
the cohort is genuinely unpredictable from telemetry.

## Reference numbers

| quantity | value |
|---|---|
| Incident rate, pre → post deploy | 0.233 → 0.264 (z ≈ 5.1) |
| Model A (8 plain features), test AUC | 0.839 |
| Model B (44 cross-features), test AUC | 0.863 |
| DeLong p-value, B vs A | 1.3 × 10⁻⁹ |

The Bayes AUC is measured on **non-anomalous rows only**: on the injected rows
the true probability is not the mechanism's `p_true` — it was forced — so no
ceiling is defined there. **Comparing it against an all-rows model AUC is
therefore invalid**, and the notebook's closing section walks into that trap
deliberately before correcting it. Restricted to the same population:

| measured on clean test rows | value |
|---|---|
| Model A | 0.867 |
| Model B | 0.892 |
| Bayes-optimal ceiling | 0.897 |
| **genuine gap to the ceiling** | **0.005** |
| gap you get by comparing across populations | 0.034 |

So model B is within ~0.005 AUC of optimal on the rows that follow the
mechanism; ~86% of the apparent shortfall is the injected anomalies, which are
unpredictable by construction.

## Limitations

- **Rows are not independent** in the way the bootstrap assumes: consecutive
  windows from one node would be autocorrelated in reality. This generator draws
  windows independently, so the bootstrap intervals in §8–§9 are honest *here*
  but would be too narrow on real telemetry. The notebook says so.
- **No missing values, no categorical features, no leakage** — all real and
  important problems, all deliberately out of scope for lesson 1.
- The mechanism is **logistic** while the lesson fits **linear** models. That is
  intentional: it leaves a visible, closable gap for lesson 2.
