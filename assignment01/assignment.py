# ---
# jupyter:
#   jupytext:
#     formats: py:percent,ipynb
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.6
#   kernelspec:
#     display_name: DS Course
#     language: python
#     name: ds-course
# ---

# %% [markdown]
# # Home assignment 1 — Edge gateway incident model
#
# ## The situation
#
# You have taken over monitoring for an **edge-gateway fleet**: 24 instances
# across 4 regions, logged in 2-hour windows. Each row is one window on one
# instance, and `incident = 1` means that window breached the SLA.
#
# Two things happened while you were away:
#
# 1. **Four deployments** went out during the first 60 days. Someone claims one
#    of them "broke everything" and another "fixed it". Nobody checked.
# 2. **60 more days of logs** have piled up that nobody has looked at.
#
# Your job: find out which deployments actually changed anything, build an
# incident model, make it as good as you can, and then find out whether it still
# works on the logs nobody looked at.
#
# ## What you are practising
#
# This assignment deliberately mixes the two halves of the course:
#
# * **the workflow** — hypothesis tests, train/test discipline, AUC, honest
#   evaluation;
# * **the implementation** — you write the statistics and the model in plain
#   NumPy first, and only then check yourself against `scipy` / `sklearn`.
#
# Wherever a task says *"implement by hand"*, that means NumPy only. You may
# always use a library **afterwards** to check your answer, and several tasks
# require exactly that.
#
# ## The five parts
#
# | Part | Topic |
# |---|---|
# | 1 | Loading and cleaning |
# | 2 | Did the deployments change anything? |
# | 3 | Baseline model, by hand |
# | 4 | Make it better (open-ended) |
# | 5 | The logs nobody looked at |
#
# This is practice, not an exam — nothing here is graded. Work through it in
# order, since each part uses what the last one built.
#
# ## Notation
#
# Used throughout, so the formulas in each task are readable:
#
# | symbol | meaning |
# |---|---|
# | $n$ | number of rows (windows) |
# | $d$ | number of features |
# | $X \in \mathbb{R}^{n \times d}$ | the design matrix, $x_{ij}$ its entries |
# | $y \in \{0,1\}^n$ | the target, `incident` |
# | $s \in \mathbb{R}^n$ | model scores |
# | $n_+,\; n_-$ | number of positives / negatives |
# | $\mathbb{1}[\cdot]$ | 1 if the condition holds, 0 otherwise |
#
# ## Rules
#
# * **Do not touch data after `2026-03-06` until Part 5.** That is the whole
#   point of Part 5. Peeking invalidates it.
# * Every model decision must be made on training data only.
# * Where the notebook prints a `[PASS] / [FAIL]` self-check, get it to `PASS`
#   before moving on.

# %% [markdown]
# ## Part 0 — Setup

# %%
import itertools
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

RANDOM_SEED = 7
rng = np.random.default_rng(RANDOM_SEED)

BASE = Path.cwd()
if not (BASE / "data").exists() and (BASE / "assignment01" / "data").exists():
    BASE = BASE / "assignment01"
DATA = BASE / "data"
FIG = BASE / "figures"
FIG.mkdir(exist_ok=True)

pd.set_option("display.width", 120)
pd.set_option("display.max_columns", 40)

# Course palette — series 1 blue, series 2 orange, series 3 aqua, fixed order.
C = {"blue": "#2a78d6", "orange": "#eb6834", "aqua": "#1baf7a"}
INK = {"primary": "#0b0b0b", "secondary": "#52514e", "muted": "#898781",
       "grid": "#e1e0d9", "axis": "#c3c2b7", "surface": "#fcfcfb"}
plt.rcParams.update({
    "figure.facecolor": INK["surface"], "axes.facecolor": INK["surface"],
    "savefig.facecolor": INK["surface"], "axes.edgecolor": INK["axis"],
    "axes.labelcolor": INK["secondary"], "axes.titlecolor": INK["primary"],
    "axes.titlesize": 11, "axes.titleweight": "semibold",
    "axes.titlelocation": "left", "axes.labelsize": 9,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
    "grid.color": INK["grid"], "grid.linewidth": 0.8, "grid.linestyle": "-",
    "xtick.color": INK["muted"], "ytick.color": INK["muted"],
    "xtick.labelsize": 8.5, "ytick.labelsize": 8.5, "legend.frameon": False,
    "legend.fontsize": 9, "lines.linewidth": 2.0, "font.size": 10,
    "figure.dpi": 110,
})


def finish(ax, title=None, xlabel=None, ylabel=None, legend=False):
    if title:
        ax.set_title(title, pad=10)
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    ax.set_axisbelow(True)
    if legend:
        ax.legend()
    return ax


def check(label, ok, detail=""):
    """Self-check. Prints rather than raising, so you can keep working."""
    print(f"  [{'PASS' if ok else 'FAIL'}] {label}" + (f"   {detail}" if detail else ""))
    return bool(ok)


NUMERIC = ["cpu_util", "mem_util", "p99_latency_ms", "request_rate",
           "error_rate_pct", "queue_depth", "cache_hit_ratio", "gc_pause_ms",
           "active_connections", "disk_io_wait_ms"]
CATEGORICAL = ["region", "instance_type", "service_tier"]
TARGET = "incident"

DEV_END = pd.Timestamp("2026-03-06")          # <- do not look past this until Part 5
DEPLOYS = [pd.Timestamp("2026-01-17"), pd.Timestamp("2026-01-29"),
           pd.Timestamp("2026-02-10"), pd.Timestamp("2026-02-22")]
print("data:", DATA)

# %% [markdown]
# ## Part 1 — Loading and cleaning
#
# ### Task 1.1 — Load the logs and look at them
#
# Load `data/edge_gateway_logs.csv`, parsing `timestamp` as a datetime. Print
# the shape, the column dtypes, and the first few rows.

# %%
# ============================================================================
# TASK 1.1 — YOUR CODE HERE
# ============================================================================
# Load the CSV (parse `timestamp` as a datetime) into `logs`,
# then print shape / period / dtypes and show `.head()`.
#
# Define: logs
# ============================================================================

raise NotImplementedError('Task 1.1')

# %% [markdown]
# ### Task 1.2 — Find and fix the data quality problems
#
# These logs came off a real collector, so they have real problems. There are
# **three** of them. Find all three and fix them.
#
# Hints, in ascending order of how much they give away:
#
# * one is about rows, one is about empty cells, one is about a value that is
#   *present* but impossible;
# * `.duplicated()`, `.isna()`, and `.describe()` will each reveal one;
# * a latency of `-1` ms has not been measured — it is a collector convention
#   for "no data", and treating it as a number will quietly poison every model
#   you build.
#
# For each problem: report how many rows it affects, then fix it. Justify your
# imputation choice in a comment.

# %%
# ============================================================================
# TASK 1.2 — YOUR CODE HERE
# ============================================================================
# Report the count of each of the three problems, then build
# `clean`: de-duplicated, with the -1 latency sentinel turned
# into a proper NaN.
#
# Do NOT drop the rows with missing values -- that would throw
# away ~1000 perfectly good rows. Impute them in Task 1.3.
#
# Define: clean
# ============================================================================

raise NotImplementedError('Task 1.2')

# %% [markdown]
# ### Task 1.3 — Split off the horizon
#
# Split into:
#
# * **`dev`** — everything strictly before `DEV_END` (2026-03-06). This is all
#   you may use for Parts 2–4.
# * **`horizon`** — everything from `DEV_END` onward. **Do not look at it until
#   Part 5.**
#
# Then impute the missing values. Careful: the imputation statistic must come
# from `dev` only. Using the median of the whole file leaks the future into
# your model — a small leak here, but exactly the same mistake that invalidates
# real evaluations.

# %%
# ============================================================================
# TASK 1.3 — YOUR CODE HERE
# ============================================================================
# Build `dev` (before DEV_END) and `horizon` (from DEV_END on),
# then fill the missing numeric values.
#
# The imputation statistic must be computed on `dev` only, and
# then applied to BOTH frames. Computing it over the whole file
# leaks the future into your model.
#
# Store the values you used in `impute_values` -- Part 5 needs
# them again.
#
# Define: dev, horizon, impute_values
# ============================================================================

raise NotImplementedError('Task 1.3')

check("no missing values left", dev[NUMERIC].isna().sum().sum() == 0)
check("no duplicate rows left", dev.duplicated().sum() == 0)
check("no negative latencies", (dev.p99_latency_ms <= 0).sum() == 0)
check("dev and horizon do not overlap", dev.timestamp.max() < horizon.timestamp.min())

# %% [markdown]
# ### Task 1.4 — Two sanity plots
#
# Plot (a) the distribution of one heavily skewed feature before and after a
# log transform, and (b) the incident rate by hour of day. Write one sentence
# under each about what it tells you.

# %%
# ============================================================================
# TASK 1.4 — YOUR CODE HERE
# ============================================================================
# Three panels: p99_latency_ms raw, the same log-transformed,
# and the incident rate by hour of day.
# Write one sentence under the figure about each.
# ============================================================================

raise NotImplementedError('Task 1.4')

# %% [markdown]
# *(a)* Latency spans three orders of magnitude and is unusable raw; the log
# turns it into a workable near-symmetric distribution.
#
# *(b)* Incidents follow a clear daily cycle, peaking in the busy afternoon
# hours. Two consequences, and they pull in opposite directions: the cycle means
# **rows close together in time are not independent**, so a random train/test
# split would leak (we handle that in Task 3.1) — but it does *not* automatically
# make "hour of day" a useful model feature, because the thing that varies over
# the day is **load**, and we already measure load directly. We come back to this
# in Task 4.6.

# %% [markdown]
# ## Part 2 — Did the deployments change anything?
#
# Four deployments went out, splitting the development period into five phases
# `A`–`E` (already labelled in `deploy_phase`):
#
# | | date | phases compared |
# |---|---|---|
# | D1 | 2026-01-17 | A → B |
# | D2 | 2026-01-29 | B → C |
# | D3 | 2026-02-10 | C → D |
# | D4 | 2026-02-22 | D → E |
#
# ### Task 2.1 — Per-phase statistics
#
# Build a table with, for each phase: number of windows, number of incidents,
# incident rate, and a 95% confidence interval for that rate.
#
# Implement the interval **by hand**. Use the **Wilson** interval, not the
# textbook normal one — the normal interval misbehaves near 0 and 1 and can
# produce bounds outside `[0, 1]`.
#
# With $x$ incidents in $n$ windows, $\hat p = x/n$, and $z$ the two-sided
# normal critical value
#
# $$z = \Phi^{-1}\!\left(1 - \tfrac{1-\text{conf}}{2}\right)
#   \;=\; 1.96 \text{ for conf} = 0.95$$
#
# the interval is centred at $c$ with half-width $h$:
#
# $$c = \frac{\hat p + \dfrac{z^2}{2n}}{1 + \dfrac{z^2}{n}}, \qquad
#   h = \frac{z\sqrt{\dfrac{\hat p(1-\hat p)}{n} + \dfrac{z^2}{4n^2}}}
#            {1 + \dfrac{z^2}{n}}, \qquad
#   \text{CI} = [\,c - h,\; c + h\,]$$
#
# `stats.norm.ppf` gives you $\Phi^{-1}$. Note the interval is **not** centred
# on $\hat p$ — that shift toward $1/2$ is exactly what keeps it inside
# $[0,1]$.

# %%
# ============================================================================
# TASK 2.1 — YOUR CODE HERE
# ============================================================================
# Implement `wilson_interval(x, n, conf=0.95)` by hand, then
# build `phase_stats`: one row per phase A-E with columns
# phase / windows / incidents / rate / ci_low / ci_high.
#
# Define: wilson_interval, phase_stats
# ============================================================================

def wilson_interval(x, n, conf=0.95):
    raise NotImplementedError('wilson_interval')


phase_stats = None

check("all five phases present", len(phase_stats) == 5)
check("intervals inside [0,1]",
      bool((phase_stats.ci_low >= 0).all() and (phase_stats.ci_high <= 1).all()))
check("every interval contains its own point estimate",
      bool(((phase_stats.ci_low <= phase_stats.rate)
            & (phase_stats.rate <= phase_stats.ci_high)).all()))

# %% [markdown]
# ### Task 2.2 — Test each deployment
#
# For each of the four deployments, test whether the incident rate changed,
# using a **two-proportion z-test implemented by hand**.
#
# The hypotheses, stated properly. With $p_1, p_2$ the true incident rates
# before and after a deployment:
#
# $$H_0: p_1 = p_2 \qquad \text{vs.} \qquad H_1: p_1 \ne p_2$$
#
# Under $H_0$ both phases share one rate, so **pool** them to estimate it, and
# build the standard error of the difference from that pooled estimate:
#
# $$\hat p = \frac{x_1+x_2}{n_1+n_2}, \qquad
#   \mathrm{SE} = \sqrt{\hat p\,(1-\hat p)
#                 \left(\frac{1}{n_1}+\frac{1}{n_2}\right)}, \qquad
#   z = \frac{\hat p_2 - \hat p_1}{\mathrm{SE}}$$
#
# The two-sided p-value is the probability of a $|z|$ at least this large under
# the standard normal:
#
# $$p = 2\left(1 - \Phi(|z|)\right) = 2\,\Phi(-|z|)$$
#
# Use `stats.norm.sf(abs(z))` for $1 - \Phi(|z|)$ — it is more accurate in the
# far tail than `1 - stats.norm.cdf(abs(z))`, which loses precision to
# cancellation once $p$ drops below about $10^{-8}$.
#
# Report the difference, z, and the p-value for each deployment. Then **check
# your implementation** against `scipy.stats.chi2_contingency` on the 2×2 table
# with `correction=False`. For a 2×2 table the two tests are algebraically the
# same, so
#
# $$\chi^2 = z^2$$
#
# must hold to floating-point precision. If it does not, your SE is wrong.

# %%
# ============================================================================
# TASK 2.2 — YOUR CODE HERE
# ============================================================================
# Implement `two_proportion_test(x1, n1, x2, n2)` returning
# (difference, z, p_value), then apply it to all four
# deployments and collect the answers in a DataFrame
# `results` with a `p_value` and a `z` column.
#
# Then cross-check every one against
# `stats.chi2_contingency(table, correction=False)` and collect
# the booleans in `chi2_matches`.
#
# Define: two_proportion_test, results, chi2_matches
# ============================================================================

def two_proportion_test(x1, n1, x2, n2):
    raise NotImplementedError('two_proportion_test')


results = None
chi2_matches = []

# cross-check against a chi-square on the same 2x2 tables
print("\ncross-check: chi2 vs z^2")
chi2_matches = []
for k, (a, b) in enumerate(zip("ABCD", "BCDE"), start=1):
    ga = dev[dev.deploy_phase == a][TARGET]
    gb = dev[dev.deploy_phase == b][TARGET]
    table = np.array([[ga.sum(), len(ga) - ga.sum()],
                      [gb.sum(), len(gb) - gb.sum()]])
    chi2 = stats.chi2_contingency(table, correction=False)[0]
    z = results.loc[k - 1, "z"]
    chi2_matches.append(np.isclose(chi2, z ** 2))
    print(f"  D{k}: chi2 = {chi2:9.5f}   z^2 = {z ** 2:9.5f}   "
          f"match = {chi2_matches[-1]}")

check("z^2 equals chi2 for every deployment", all(chi2_matches))

# %% [markdown]
# ### Task 2.3 — You just ran four tests
#
# Look at your p-values before continuing.
#
# With $m$ independent tests each at level $\alpha$, the probability of **at
# least one** false positive when every null is true — the *family-wise error
# rate* — is
#
# $$\mathrm{FWER} = 1 - (1-\alpha)^m$$
#
# For $m = 4$, $\alpha = 0.05$ that is $1 - 0.95^4 \approx 18.5\%$. Nearly one
# chance in five of finding a "significant" deployment in a world where all four
# did nothing. That is why "one of our four was significant at p = 0.04" is
# nearly worthless on its own.
#
# Apply two corrections, both by hand.
#
# **Bonferroni** — divide the budget evenly. Reject $H_i$ iff
#
# $$p_i \le \frac{\alpha}{m}$$
#
# Simple, always valid, and conservative: it controls FWER at $\le \alpha$
# because $\mathrm{FWER} \le \sum_i P(p_i \le \alpha/m) = m \cdot \alpha/m$.
#
# **Holm–Bonferroni** — a *step-down* procedure. Sort the p-values ascending,
# $p_{(1)} \le p_{(2)} \le \dots \le p_{(m)}$, and walk down the list comparing
# each against a threshold that relaxes as you go:
#
# $$k^{\ast} = \min\left\{k : p_{(k)} > \frac{\alpha}{m-k+1}\right\}$$
#
# Reject $H_{(1)}, \dots, H_{(k^{\ast}-1)}$ and **nothing after** — once one test
# fails, you stop, even if a later p-value happens to sit below its own
# threshold. Holm controls FWER at exactly the same level as Bonferroni while
# rejecting at least as much, so there is never a reason to prefer plain
# Bonferroni.
#
# Note the first threshold is $\alpha/m$ — identical to Bonferroni — and the
# last is $\alpha$, so Holm interpolates between "correct hard" and "do not
# correct".
#
# Report, for each deployment, whether it is significant raw / after Bonferroni
# / after Holm. **Then state your conclusion in words.**

# %%
# ============================================================================
# TASK 2.3 — YOUR CODE HERE
# ============================================================================
# Implement `holm_bonferroni(pvals, alpha=0.05)` -> boolean array.
#
#   sort the p-values ascending; the k-th smallest (k from 1) is
#   compared against alpha / (m - k + 1); STOP at the first
#   failure and reject nothing after it.
#
# Add `sig_raw`, `sig_bonferroni` and `sig_holm` columns to
# `results`, print the family-wise error rate of 4 uncorrected
# tests, and print which deployments change verdict.
#
# Define: holm_bonferroni
# ============================================================================

def holm_bonferroni(pvals, alpha=0.05):
    raise NotImplementedError('holm_bonferroni')

# %% [markdown]
# **Conclusion.**
#
# * **D2 (B→C) is the bad deploy.** The incident rate rose about 5 percentage
#   points, at p far below any threshold. This is real by any standard.
# * **D3 (C→D) is the fix.** It brought the rate back down by roughly 4.7
#   points, also unambiguous. It did not quite return to the pre-D2 level.
# * **D1 (A→B) did nothing.** p ≈ 0.40 — entirely consistent with noise.
# * **D4 (D→E) is the interesting one.** Taken alone it is "significant"
#   (p ≈ 0.039). But we ran four tests, and after correcting for that it does
#   **not** survive. The honest report is *"no evidence of a change, and we
#   would need more data to call it"* — not *"D4 raised incidents"*.
#
# The lesson: the number of tests you ran is part of the result. A p-value of
# 0.04 from the only test you ran and a p-value of 0.04 from the best of four
# are very different pieces of evidence.

# %% [markdown]
# ### Task 2.4 — Plot it
#
# Plot the incident rate per phase with its 95% interval, and mark the four
# deployments. The intervals matter more than the dots.

# %%
# ============================================================================
# TASK 2.4 — YOUR CODE HERE
# ============================================================================
# Plot the per-phase rate with its 95% interval and mark the
# four deployments. The intervals are the point of the chart.
# ============================================================================

raise NotImplementedError('Task 2.4')

# %% [markdown]
# ## Part 3 — Baseline model, by hand
#
# ### Task 3.1 — Split by time, not at random
#
# Split `dev` into train (first 75% of the period) and test (last 25%) **by
# timestamp**.
#
# Why not a random split? Because rows from the same hour on neighbouring
# instances are nearly duplicates. A random split puts some of them in train and
# some in test, so the test set is no longer independent and your score comes
# out flattering. Splitting on time is how the model will actually be used:
# fit on the past, predict the future.

# %%
# ============================================================================
# TASK 3.1 — YOUR CODE HERE
# ============================================================================
# Split `dev` by TIME at the 75% quantile of `timestamp`.
# Define `tr_mask`, `te_mask` (boolean numpy arrays) and
# `y_tr`, `y_te`.
#
# Define: tr_mask, te_mask, y_tr, y_te
# ============================================================================

split_at = None
tr_mask = te_mask = None
y_tr = y_te = None

check("train comes strictly before test",
      dev.timestamp[tr_mask].max() < dev.timestamp[te_mask].min())
check("split is roughly 75/25", abs(tr_mask.mean() - 0.75) < 0.02,
      f"train fraction = {tr_mask.mean():.3f}")

# %% [markdown]
# ### Task 3.2 — Least squares by hand
#
# Fit a linear regression on the **10 raw numeric columns** using the normal
# equation, implemented in NumPy.
#
# We are minimising the squared error over $w \in \mathbb{R}^d$, $b \in
# \mathbb{R}$:
#
# $$L(w, b) = \sum_{i=1}^{n}\left(x_i^\top w + b - y_i\right)^2
#           = \lVert Xw + b\mathbf{1} - y \rVert_2^2$$
#
# The clean way to handle the intercept is to fold it into the matrix. Append a
# column of ones,
#
# $$\tilde X = \begin{bmatrix} \mathbf{1} & X \end{bmatrix}
#              \in \mathbb{R}^{n \times (d+1)}, \qquad
#   \beta = \begin{bmatrix} b \\ w \end{bmatrix} \in \mathbb{R}^{d+1}$$
#
# so that $L(\beta) = \lVert \tilde X\beta - y\rVert^2$. Setting
# $\nabla_\beta L = 2\tilde X^\top(\tilde X\beta - y) = 0$ gives the **normal
# equations** $\tilde X^\top \tilde X \beta = \tilde X^\top y$, hence
#
# $$\boxed{\;\hat\beta = (\tilde X^\top \tilde X)^{-1}\tilde X^\top y\;}
#   \qquad \hat b = \hat\beta_0,\quad \hat w = \hat\beta_{1:d}$$
#
# **Solve, do not invert.** `np.linalg.solve(A, c)` computes $A^{-1}c$ without
# ever forming $A^{-1}$: it is roughly 2× faster and numerically better
# conditioned. `np.linalg.inv(A) @ c` is the classic beginner's line and is
# wrong for both reasons.
#
# (Yes, we are fitting a regression to a 0/1 target. That is a *linear
# probability model*: the scores are not calibrated probabilities and can fall
# outside [0, 1], but AUC only cares about the **ranking**, so it is fine here.)

# %%
# ============================================================================
# TASK 3.2 — YOUR CODE HERE
# ============================================================================
# Implement `fit_ols(X, y)` -> (weights, intercept) using the
# normal equation and `np.linalg.solve`.
# Fit it on the raw numeric columns and print the weights.
# Keep the design matrix in `X_raw` and the fit in w_raw/b_raw.
#
# Define: fit_ols, X_raw, w_raw, b_raw
# ============================================================================

def fit_ols(X, y):
    raise NotImplementedError('fit_ols')


X_raw = dev[NUMERIC].to_numpy(float)
w_raw = b_raw = None

check("weights match sklearn", np.allclose(w_raw, sk.coef_, atol=1e-8))
check("intercept matches sklearn", np.isclose(b_raw, sk.intercept_, atol=1e-8))

# %% [markdown]
# ### Task 3.3 — AUC by hand
#
# Implement AUC yourself. Use the rank identity rather than integrating a curve
# — it is three lines and it is exactly what AUC *means*:
#
# > AUC is the probability that a randomly chosen positive scores above a
# > randomly chosen negative.
#
# Formally, with $\psi$ counting a tie as half a win,
#
# $$\psi(a, b) = \mathbb{1}[a > b] + \tfrac{1}{2}\,\mathbb{1}[a = b]$$
#
# the definition is an average over **all** positive–negative pairs:
#
# $$\mathrm{AUC} = \frac{1}{n_+ n_-}
#   \sum_{i:\,y_i=1}\;\sum_{j:\,y_j=0} \psi(s_i, s_j)$$
#
# Computing that double sum directly costs $O(n_+ n_-)$ — about 4 million
# operations here. The **Mann–Whitney identity** collapses it to a single sort.
# Let $r_i$ be the rank of $s_i$ in the *pooled* sample (ranks $1 \dots n$) and
# $R_+ = \sum_{i:\,y_i=1} r_i$. Then
#
# $$\boxed{\;\mathrm{AUC}
#   = \frac{R_+ - \dfrac{n_+(n_++1)}{2}}{n_+ n_-}\;}$$
#
# The intuition for the correction term: if the positives took the top $n_+$
# ranks they would sum to $n_- n_+ + \frac{n_+(n_++1)}{2}$, so subtracting
# $\frac{n_+(n_++1)}{2}$ removes the ranks positives contribute *among
# themselves*, leaving only wins over negatives.
#
# Ties must get **average** ranks, which is precisely why $\psi$ scores them
# $\tfrac12$ — the two conventions agree. `scipy.stats.rankdata` averages ties
# by default; using an ordinal rank instead is the classic bug here.
#
# Check against `sklearn.metrics.roc_auc_score`.

# %%
# ============================================================================
# TASK 3.3 — YOUR CODE HERE
# ============================================================================
# Implement `auc_score(y_true, score)` with the rank identity.
# Store the baseline test AUC in `AUC_BASELINE`.
#
# Define: auc_score, AUC_BASELINE
# ============================================================================

def auc_score(y_true, score):
    raise NotImplementedError('auc_score')


scores_te = None
AUC_BASELINE = None

check("AUC matches sklearn",
      np.isclose(AUC_BASELINE, roc_auc_score(y_te, scores_te), atol=1e-12))
check("AUC is in a sensible range for this task", 0.75 < AUC_BASELINE < 0.85,
      f"AUC = {AUC_BASELINE:.4f}")

# %% [markdown]
# ### Task 3.4 — Sanity baselines
#
# A number needs something to be compared against. Report the AUC of:
#
# 1. random scores,
# 2. the single best individual feature,
#
# and confirm your model beats both.

# %%
# ============================================================================
# TASK 3.4 — YOUR CODE HERE
# ============================================================================
# Compute the AUC of random scores and of each single feature.
# Put the per-feature AUCs in a Series `single` and the name of
# the most informative one in `best_feature`.
#
# Define: single, best_feature
# ============================================================================

random_auc = None
single = None
best_feature = None

check("beats random", AUC_BASELINE > 0.65)
check("beats the best single feature",
      AUC_BASELINE > max(single[best_feature], 1 - single[best_feature]))

# %% [markdown]
# ## Part 4 — Make it better
#
# Baseline is about **0.81**. Your target is **0.87 or better** on the same test
# split, using the same model class (linear regression). Everything you gain has
# to come from the *features*.
#
# Work through 4.1–4.4, then report your best in 4.5.
#
# ### Task 4.1 — Standardisation: a checkpoint
#
# Standardise every feature to mean 0 / std 1, refit, and recompute the AUC:
#
# $$\tilde x_{ij} = \frac{x_{ij} - \mu_j}{\sigma_j}, \qquad
#   \mu_j = \frac{1}{n}\sum_{i=1}^{n} x_{ij}, \qquad
#   \sigma_j = \sqrt{\frac{1}{n}\sum_{i=1}^{n}(x_{ij}-\mu_j)^2}$$
#
# Compute $\mu_j, \sigma_j$ on the **training rows only** and apply those same
# numbers to the test rows — that is why the function takes `mean` and `std` as
# optional arguments. Guard $\sigma_j = 0$ (a constant column) by substituting
# 1, or you divide by zero.
#
# **Before you run it, write down what you expect.** Then look at the answer.
#
# To predict it, ask: in matrix form standardising is
# $\tilde X = (X - \mathbf{1}\mu^\top)S^{-1}$ with
# $S = \mathrm{diag}(\sigma_1,\dots,\sigma_d)$, which is invertible. So is the
# set of functions
#
# $$\left\{\,x \mapsto \tilde x^\top \tilde w + \tilde b \;:\;
#   \tilde w \in \mathbb{R}^d,\ \tilde b \in \mathbb{R}\right\}$$
#
# the same set as $\{x \mapsto x^\top w + b\}$, or a different one? And if it is
# the same set, what can least squares possibly do differently?

# %%
# ============================================================================
# TASK 4.1 — YOUR CODE HERE
# ============================================================================
# Implement `standardize(X, mean=None, std=None)` returning
# (X_scaled, mean, std). Standardise with TRAIN statistics,
# refit, and put the result in `auc_standardised`.
#
# Write down your prediction BEFORE running it.
#
# Define: standardize, auc_standardised
# ============================================================================

def standardize(X, mean=None, std=None):
    raise NotImplementedError('standardize')


auc_standardised = None

check("standardisation changed the AUC by nothing at all",
      np.isclose(auc_standardised, AUC_BASELINE, atol=1e-9))

# %% [markdown]
# **Why it is exactly zero.**
#
# Standardising is an *affine, invertible* change of variables. If
# $\tilde X = (X - \mu)/\sigma$ then any linear function of $\tilde X$ is also a
# linear function of $X$:
#
# $$\tilde X \tilde w + \tilde b
#   = X\left(\tfrac{\tilde w}{\sigma}\right)
#     + \left(\tilde b - \tfrac{\mu^\top \tilde w}{\sigma}\right)$$
#
# So the two problems have the *same* set of reachable predictions, and least
# squares finds the best one in that set either way. The fitted values are
# identical to floating-point noise — so every metric is too, AUC included.
#
# **Then why does everyone standardise?** Because it matters for things that are
# *not* plain OLS:
#
# * **gradient descent** — the usable learning rate is set by the largest
#   feature scale, so without standardising, small-scale features barely move
#   (this is the §4 result from lesson 1);
# * **regularisation** — ridge/lasso penalise $\|w\|$, and that penalty is not
#   scale-invariant, so unstandardised features get penalised unequally;
# * **distance-based models** — kNN, SVM, k-means are meaningless without it;
# * **reading coefficients** — standardised weights are comparable to each other.
#
# It is good practice. It is not a way to raise AUC on an OLS fit, and any
# tutorial implying otherwise is wrong.

# %% [markdown]
# ### Task 4.2 — Transform the skewed features
#
# *This* changes the predictions, because $\log$ is **not** an affine map:
# there is no pair $(a, b)$ with $\log x = ax + b$, so the set of reachable
# functions genuinely grows.
#
# Apply
#
# $$x \mapsto \log x \quad\text{or}\quad
#   x \mapsto \log(1+x) \;\;(\texttt{np.log1p})$$
#
# to the heavy-tailed columns, refit, and report the gain.
#
# Use `log1p` wherever a column can be exactly $0$, since $\log 0 = -\infty$.
# `error_rate_pct` is exactly zero for many windows; `disk_io_wait_ms` gets
# arbitrarily close, so shift it by a small constant first. Check
# `(df[col] == 0).any()` rather than guessing which columns need it.
#
# Why a log at all: these columns are roughly log-normal, so $\log x$ is roughly
# symmetric. A squared-error fit is dominated by the largest residuals, and on a
# raw heavy-tailed column a handful of extreme rows drag the whole hyperplane
# toward themselves.

# %%
# ============================================================================
# TASK 4.2 — YOUR CODE HERE
# ============================================================================
# Write `build_numeric(frame)` -> DataFrame with the skewed
# columns log-transformed (use log1p where a column can be
# exactly zero). Refit, and store the AUC in `auc_logs`.
#
# Keep the numeric design matrix in `X_log` -- later tasks
# reuse it.
#
# Define: build_numeric, X_log, auc_logs
# ============================================================================

def build_numeric(frame):
    raise NotImplementedError('build_numeric')


X_log = None
auc_logs = None

check("log transforms helped", auc_logs > AUC_BASELINE)

# %% [markdown]
# ### Task 4.3 — Use the categorical columns
#
# So far you have thrown away `region`, `instance_type` and `service_tier`
# entirely. A linear model cannot consume strings, so encode them.
#
# A categorical column with levels $\{c_1, \dots, c_K\}$ becomes $K$ indicator
# columns:
#
# $$D_{ik} = \mathbb{1}\!\left[\text{category}_i = c_k\right],
#   \qquad k = 1, \dots, K$$
#
# Implement it by hand — `np.unique(..., return_inverse=True)` gives an integer
# code per row, and then row $k$ of `np.eye(K)` *is* the one-hot vector for
# category $k$, so `np.eye(K)[codes]` encodes the whole column at once.
#
# **Drop one level per column.** Every row belongs to exactly one category, so
#
# $$\sum_{k=1}^{K} D_{ik} = 1 \quad \text{for every } i
#   \qquad\Longleftrightarrow\qquad D\mathbf{1}_K = \mathbf{1}_n$$
#
# which is *exactly* the intercept column. So $[\mathbf{1}, D]$ has rank $K$,
# not $K+1$, making $\tilde X^\top \tilde X$ **singular** and
# `np.linalg.solve` raise `LinAlgError`. Dropping one level (the "reference"
# or "baseline") fixes it, and every remaining coefficient is then read as
# *"effect relative to the dropped level"*.
#
# Then refit and report the gain.

# %%
# ============================================================================
# TASK 4.3 — YOUR CODE HERE
# ============================================================================
# Implement `one_hot(labels, categories=None, drop_first=True)`
# and `encode_categoricals(frame, cats=None)`.
#
# `encode_categoricals` must be able to reuse the categories it
# learned on `dev` (return them as `CATS`), or Part 5 will build
# a different number of columns and the weights will not line up.
#
# Store the AUC in `auc_cats` and the design in `X_cat`.
#
# Define: one_hot, encode_categoricals, cat_matrix, cat_names, CATS, auc_cats
# ============================================================================

def one_hot(labels, categories=None, drop_first=True):
    raise NotImplementedError('one_hot')


def encode_categoricals(frame, cats=None):
    raise NotImplementedError('encode_categoricals')


cat_matrix = cat_names = CATS = None
auc_cats = None

check("one-hot encoding helped", auc_cats > auc_logs)
check("dummies are 0/1 only", set(np.unique(cat_matrix)) <= {0.0, 1.0})

# %% [markdown]
# The learned effects are readable: `ap-south-1` carries a much higher incident
# rate than the other regions, `enterprise` traffic breaches its (stricter) SLA
# more often than `free`, and the memory-heavy `r5.2xlarge` instances fail
# least. None of this was recoverable from the numeric columns — which is why
# one-hot encoding is the single biggest win available here.

# %% [markdown]
# ### Task 4.4 — Interactions
#
# A linear model can only add effects up: it predicts $\sum_j w_j x_j$, so the
# contribution of feature $j$ never depends on feature $k$. If "high CPU is
# fine, and a deep queue is fine, but high CPU **while** the queue is deep means
# trouble", no pair of weights can express that.
#
# The fix is to enlarge the feature map. Replace $x$ with
#
# $$\phi(x) = \bigl(\underbrace{x_1, \dots, x_d}_{\text{main effects}},\;
#   \underbrace{\{x_i x_j\}_{1 \le i \le j \le d}}_{\text{interactions and squares}}\bigr)$$
#
# which has dimension
#
# $$d + \binom{d+1}{2} = d + \frac{d(d+1)}{2}
#   \;=\; 10 + 55 = 65 \text{ columns here}$$
#
# The model stays **linear in its parameters** — so the normal equation still
# applies unchanged — while becoming quadratic in the original features. Taking
# $i \le j$ rather than $i < j$ keeps the squares $x_i^2$, which let the model
# express "moderate is best" curvature that pure cross terms cannot.
#
# `np.triu_indices(d)` gives exactly the $i \le j$ index pairs, and
# `X[:, :, None] * X[:, None, :]` broadcasts to an $(n, d, d)$ array of all
# products — index it with those pairs.
#
# **Standardise the numeric block first.** Products of raw features vary
# wildly in scale ($x_i x_j$ has units of $x_i$ times $x_j$), which wrecks the
# conditioning of $\tilde X^\top \tilde X$. Keep the dummies as they are — they
# are already $0/1$, and squaring a dummy just reproduces it
# ($D_{ik}^2 = D_{ik}$), which would add exact duplicate columns.

# %%
# ============================================================================
# TASK 4.4 — YOUR CODE HERE
# ============================================================================
# Implement `add_interactions(X)` -> (expanded, pairs), adding
# every product x_i * x_j with i <= j.
# `np.triu_indices` plus broadcasting is the whole trick.
#
# Standardise the numeric block FIRST, then expand, then stick
# the (already 0/1) dummies on the end. Store `X_full`, `w_f`,
# `b_f` and `auc_full` -- Part 5 reuses all four.
#
# Define: add_interactions, X_full, w_f, b_f, auc_full
# ============================================================================

def add_interactions(X):
    raise NotImplementedError('add_interactions')


X_full = w_f = b_f = None
auc_full = None

check("interactions helped", auc_full > auc_cats)

# %% [markdown]
# ### Task 4.5 — The same thing with an sklearn Pipeline
#
# Everything above, expressed as a `Pipeline` + `ColumnTransformer`.
#
# This is not just tidier. A pipeline **cannot leak**: `fit` learns the scaler
# means and the encoder categories from training data only, and `transform`
# reapplies them. The hand-rolled version above is correct only because we were
# careful to pass training statistics around by hand every single time — and
# that is exactly the kind of care that fails silently in a real project.

# %%
# ============================================================================
# TASK 4.5 — YOUR CODE HERE
# ============================================================================
# Express the same preprocessing as a sklearn Pipeline with a
# ColumnTransformer (StandardScaler / FunctionTransformer for
# the numerics, OneHotEncoder for the categoricals,
# PolynomialFeatures for the interactions).
# Store the AUC in `auc_pipeline`.
#
# Define: pipe, auc_pipeline
# ============================================================================

pipe = None
auc_pipeline = None

check("pipeline is in the same league as the hand-built model",
      abs(auc_pipeline - auc_full) < 0.05,
      f"difference {auc_pipeline - auc_full:+.4f}")

# %% [markdown]
# The two differ slightly because they are not quite the same model — the
# pipeline logs a different subset of columns and expands interactions across
# the dummies too. That is fine; the point is that the pipeline expresses the
# whole preprocessing story as one fitted object you can hand to
# `cross_val_score` or save to disk without it silently forgetting a scaler.
#
# ### Task 4.6 — One more idea, and a trap
#
# Task 1.4 showed a clear daily cycle in the incident rate. That looks like a
# free feature: add hour-of-day and let the model use it.
#
# Try it — add hour-of-day (as `sin`/`cos`, and as dummies) on top of your best
# model — and report what happens **before** reading the explanation below.

# %%
# ============================================================================
# TASK 4.6 — YOUR CODE HERE
# ============================================================================
# Add hour-of-day to your best model, as sin/cos and as dummies,
# and report both AUCs. Then build the summary `ladder` table,
# set `BEST_AUC`, and plot the ladder as a horizontal bar chart.
#
# Predict what hour-of-day will do BEFORE you run it, and write
# your explanation of the result in the markdown cell below.
#
# Define: ladder, BEST_AUC
# ============================================================================

auc_cyc = auc_hod = None
ladder = None
BEST_AUC = None

# %% [markdown]
# ## Part 5 — The logs nobody looked at
#
# Now open the horizon: 60 days of logs that played no part in anything above.
#
# Your model was validated on data from the same 60-day window it was trained
# on. The real question — the only one that matters in production — is whether
# it still works months later.
#
# ### Task 5.1 — Does it still work?
#
# Score the horizon with the model from Task 4.4 and compute AUC in a **rolling
# window** over time (14 days wide, stepped every 3 days). Plot it.
#
# For a window starting at time $t$ with width $\Delta$, the index set is
#
# $$W_t = \{\,i \;:\; t \le \tau_i < t + \Delta\,\}$$
#
# where $\tau_i$ is row $i$'s timestamp, and you report
# $\mathrm{AUC}\left(y_{W_t},\, s_{W_t}\right)$ plotted at the window's centre
# $t + \Delta/2$. Step $t$ forward by 3 days and repeat.
#
# Two details worth getting right:
#
# * a window with **only one class** has undefined AUC ($n_+ n_- = 0$), so skip
#   it rather than dividing by zero;
# * the windows **overlap** (14-day width, 3-day step), so neighbouring points
#   share most of their data and the curve is smoother than independent
#   estimates would be. That is the point — it trades independence for
#   readability — but it means you should not read a single wobble as a signal.
#
# Implement the rolling evaluation yourself.

# %%
# ============================================================================
# TASK 5.1 — YOUR CODE HERE
# ============================================================================
# Write `build_design(frame, mu, sd, cats)` that reproduces the
# Task 4.4 design matrix for ANY frame, and
# `rolling_auc(timestamps, y, scores, window_days, step_days)`
# returning (centres, aucs, counts).
#
# Score the horizon with the Task 4.4 model and compute the
# rolling AUC. Store `centres` and `aucs`.
#
# Define: build_design, rolling_auc, centres, aucs, mu_log, sd_log, hor_scores, y_hor
# ============================================================================

def build_design(frame, mu, sd, cats):
    raise NotImplementedError('build_design')


def rolling_auc(timestamps, y_true, scores, window_days=14,
                step_days=3):
    raise NotImplementedError('rolling_auc')


mu_log = sd_log = None
X_hor = y_hor = hor_scores = None
centres = aucs = counts = None

check("quality decayed over the horizon", aucs[-3:].mean() < aucs[:3].mean(),
      f"{aucs[:3].mean():.4f} -> {aucs[-3:].mean():.4f}")

# %%
# ============================================================================
# TASK 5.1b — YOUR CODE HERE
# ============================================================================
# Plot the rolling AUC over time, with a horizontal line for the
# development-period AUC and a fitted linear trend.
# ============================================================================

raise NotImplementedError('Task 5.1b')

# %% [markdown]
# ### Task 5.2 — Diagnose it
#
# Quality dropped. There are two different reasons that can happen, and they
# call for different responses:
#
# Write $P_{\text{old}}$ for the joint distribution during development and
# $P_{\text{new}}$ for the one at the end of the horizon. Since
# $P(x, y) = P(x)\,P(y \mid x)$, exactly one of the two factors can move:
#
# * **Covariate shift** — the inputs moved, the rule did not:
#
#   $$P_{\text{new}}(x) \ne P_{\text{old}}(x), \qquad
#     P_{\text{new}}(y \mid x) = P_{\text{old}}(y \mid x)$$
#
#   The model is still *right*; it is just being asked about regions of feature
#   space it saw rarely. Often survivable, and fixable by reweighting.
#
# * **Concept drift** — the rule itself changed:
#
#   $$P_{\text{new}}(y \mid x) \ne P_{\text{old}}(y \mid x)$$
#
#   What the model learned is no longer true. **Only retraining fixes this.**
#
# Test for both:
#
# 1. **Covariate shift.** Compare feature means in `dev` against the last 21
#    days, in units of the dev standard deviation so the columns are
#    comparable:
#
#    $$\delta_j = \frac{\bar x_j^{\,\text{new}} - \bar x_j^{\,\text{old}}}
#                      {\sigma_j^{\,\text{old}}}$$
#
#    Treat $|\delta_j| \gtrsim 0.3$ as worth noticing.
#
# 2. **Concept drift.** Refit on the last 21 days and compare coefficients:
#
#    $$\Delta_j = \hat w_j^{\,\text{new}} - \hat w_j^{\,\text{old}}$$
#
#    A large $|\Delta_j|$ is suspicious; a **sign flip**,
#    $\mathrm{sign}(\hat w_j^{\,\text{new}}) \ne
#     \mathrm{sign}(\hat w_j^{\,\text{old}})$, is conclusive — the model is now
#    pushing that feature in the wrong direction.

# %%
# ============================================================================
# TASK 5.2 — YOUR CODE HERE
# ============================================================================
# Quantify BOTH kinds of shift:
#
#   covariate shift -- compare feature means in `dev` against the
#     last 21 days of the horizon, expressed in dev standard
#     deviations;
#   concept drift  -- refit on the last 21 days and compare the
#     main-effect coefficients with the dev-trained ones. Build
#     a `coef` frame with a `flipped_sign` column.
#
# Then say in words which one is doing the damage, and why.
#
# Define: late, coef
# ============================================================================

late = None
shift = None
coef = None

check("found at least one sign flip (concept drift)", coef.flipped_sign.any())

# %% [markdown]
# **Diagnosis.** Both are present, but they are not equally important.
#
# There *is* covariate shift — traffic and latency have crept up, so
# `log_request_rate`, `log_active_connections` and `log_p99_latency` all sit
# noticeably higher than in the development period. On its own that would be
# survivable.
#
# The real problem is **concept drift**: `cache_hit_ratio` has changed sign. In
# the development period a high cache-hit ratio meant *fewer* incidents; by the
# end of the horizon it means *more*. A model carrying the old coefficient is
# now actively penalising the wrong windows. `log1p_error_rate` has also lost
# most of its weight, and `log_gc_pause` has gained a lot.
#
# That is not something better features fix. The relationship changed, so the
# model has to be refit.

# %% [markdown]
# ### Task 5.3 — Retrain
#
# Simulate what an on-call team would actually do: hold out the **last 21 days**
# as a fresh test set, and compare
#
# * the original dev-trained model, versus
# * the same model retrained on the **21 days immediately before** that.
#
# Note how little data the retrained model gets — and that it wins anyway.

# %%
# ============================================================================
# TASK 5.3 — YOUR CODE HERE
# ============================================================================
# Hold out the last 21 days as `final`; train on the 21 days
# immediately before (`recent`). Compare the stale dev-trained
# model against the retrained one on `final`.
# Store `auc_stale`, `auc_fresh`, `cut`, `X_final`, `y_final`.
#
# Define: cut, recent, final, X_final, y_final, auc_stale, auc_fresh
# ============================================================================

cut = None
recent = final = None
X_final = y_final = None
auc_stale = auc_fresh = None

check("retraining improved the model", auc_fresh > auc_stale)

# %% [markdown]
# ### Task 5.4 — How much history should you keep?
#
# Retraining on 21 days worked. Would 60 days work better — more data — or
# worse, because older data describes a world that no longer exists?
#
# This is a bias–variance trade-off in an unusual variable, the **window
# length** $L$. Training on $\{i : \text{cut} - L \le \tau_i < \text{cut}\}$:
#
# * larger $L$ $\Rightarrow$ more rows $\Rightarrow$ **lower variance** in
#   $\hat w$ (roughly $\propto 1/\sqrt{n}$, and $n$ grows linearly with $L$);
# * larger $L$ $\Rightarrow$ older rows, drawn from a
#   $P(y \mid x)$ further from today's $\Rightarrow$ **higher bias**.
#
# The optimum $L^{\ast}$ is wherever those two curves cross, and it depends on how
# fast the concept is drifting — there is no universal answer, which is why you
# sweep it.
#
# Sweep the training-window length and find out.

# %%
# ============================================================================
# TASK 5.4 — YOUR CODE HERE
# ============================================================================
# Sweep the training-window length (7, 14, 21, 30, 45, 60, 90
# days before `cut`), refit on each, and evaluate on `final`.
# Build a `sweep` DataFrame with `window_days` and `auc`, then
# plot AUC against window length with the stale model's AUC as a
# reference line.
#
# Careful: the older windows reach back before DEV_END, so build
# them from `clean` (and impute with `impute_values`), not from
# `horizon`.
#
# Is more data better here? Explain the shape you get.
#
# Define: sweep
# ============================================================================

sweep = None
