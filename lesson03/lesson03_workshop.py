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
# # Lesson 3 — From lines to leaves
#
# ## The situation
#
# In lesson 1 you ran one service on eight nodes. Congratulations: you have been
# promoted. You now run a **global fleet** — 180 clusters across 10 regions, six
# instance types, a dozen OS images, three deploy channels. One row of the
# dataset is one node-day health snapshot, and there are 30 000 of them.
#
# Two questions, one per target:
#
# > **Q1 (regression).** Predict `power_draw_watts` — capacity planning wants a
# > number, and wants to know *which* signals actually drive it.
#
# > **Q2 (classification).** Predict whether a node-day contains an `incident` —
# > the on-call rotation wants risk ranking.
#
# The new ingredient compared to lessons 1–2 is not size. It is **structure**:
# five categorical columns, one of which has 180 levels. Everything painful and
# interesting in this lesson flows from that. Linear models need categoricals
# flattened into one-hot columns, and at 180 levels that starts to hurt. Trees
# split on anything. Gradient boosting stacks trees. CatBoost handles the
# categoricals natively — for reasons that are widely misquoted, which we will
# correct.
#
# The lesson climbs a **ladder of five models** on Q2, and every rung exists to
# teach exactly one idea:
#
# | # | model | the idea |
# |---|-------|----------|
# | 1 | logistic regression, numerics only | the baseline you must always build |
# | 2 | + one-hot on 5 categoricals | categoricals matter; OHE is the price |
# | 3 | one decision tree | nonlinearity for free, but one tree is weak |
# | 4 | CatBoost, defaults | boosting + native categoricals |
# | 5 | CatBoost + Optuna | what honest tuning buys (spoiler: a little) |
#
# ### The rule of this notebook
#
# Every number quoted in prose is computed by a nearby cell, and §9 checks the
# headline results against `data/reference_results.json` — the frozen reference
# fitted by `src/calibrate_reference.py`. If this notebook ever drifts from the
# reference, §9 fails loudly instead of teaching you stale numbers.

# %% [markdown]
# ## §0. Setup, load, splits
#
# Everything is seeded. The dataset ships with a materialised `split` column
# (train 18 000 / val 6 000 / test 6 000, stratified on `incident`) so that
# every model in the course — and the reference results — sees exactly the same
# rows. Discipline for the whole lesson:
#
# * **train** — fit parameters,
# * **val** — early stopping and hyperparameter choices,
# * **test** — touched once per model, to report. Never to choose.

# %%
import json
import time
import warnings
from pathlib import Path

from tqdm import TqdmWarning

# tqdm (pulled in by optuna's progress machinery) warns that this kernel has no
# ipywidgets progress bars; we never render any, so silence exactly that one.
warnings.filterwarnings("ignore", category=TqdmWarning)

import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
from catboost import CatBoostClassifier
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import (Lasso, LassoCV, LinearRegression,
                                  LogisticRegression, Ridge)
from sklearn.metrics import r2_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor, plot_tree

RANDOM_SEED = 20260808
rng = np.random.default_rng(RANDOM_SEED)

CANDIDATES = [Path.cwd() / "data", Path.cwd() / "lesson03" / "data"]
DATA_DIR = next(p.resolve() for p in CANDIDATES
                if (p / "fleet_incidents.csv").exists())
FIG_DIR = DATA_DIR.parent / "figures"
FIG_DIR.mkdir(exist_ok=True)

NUMERIC = ["cpu_util", "mem_pressure", "disk_latency_ms", "request_rate_rps",
           "error_rate_pct", "queue_depth", "cache_hit_ratio", "node_age_days"]
CATEGORICAL = ["region", "cluster_id", "instance_type", "deploy_channel",
               "os_image"]

pd.set_option("display.width", 110)
pd.set_option("display.max_columns", 40)
np.set_printoptions(precision=4, suppress=True, linewidth=100)
print("data:", DATA_DIR)

# %%
# Same palette as lessons 1-2, so the course reads as one system.
C = {"blue": "#2a78d6", "orange": "#eb6834", "aqua": "#1baf7a",
     "red": "#e34948", "violet": "#4a3aa7"}
INK = {"primary": "#0b0b0b", "secondary": "#52514e", "muted": "#898781",
       "grid": "#e1e0d9", "axis": "#c3c2b7", "surface": "#fcfcfb"}

plt.rcParams.update({
    "figure.facecolor": INK["surface"], "axes.facecolor": INK["surface"],
    "savefig.facecolor": INK["surface"], "axes.edgecolor": INK["axis"],
    "axes.labelcolor": INK["secondary"], "axes.titlecolor": INK["primary"],
    "axes.titlesize": 11, "axes.titleweight": "semibold",
    "axes.titlelocation": "left", "axes.labelsize": 9,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": INK["grid"], "grid.linewidth": 0.8,
    "grid.linestyle": "-", "xtick.color": INK["muted"],
    "ytick.color": INK["muted"], "xtick.labelsize": 8.5,
    "ytick.labelsize": 8.5, "legend.frameon": False, "legend.fontsize": 9,
    "lines.linewidth": 2.0, "font.size": 10, "figure.dpi": 110,
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


# %% [markdown]
# The categoricals are loaded as strings explicitly. That matters twice: pandas
# would otherwise be free to infer something clever, and CatBoost in §7 expects
# its categorical columns to be strings, not floats.

# %%
def load_splits(data_dir):
    """Load fleet_incidents.csv and slice it by the materialised split column.

    Returns (df, parts) where parts maps "train"/"val"/"test" to a tuple
    (X, y, power): X is a DataFrame with the 8 numeric + 5 categorical feature
    columns, y is the binary incident array, power the regression target.
    """
    df = pd.read_csv(data_dir / "fleet_incidents.csv",
                     dtype={c: str for c in CATEGORICAL})
    parts = {}
    for s in ("train", "val", "test"):
        d = df[df["split"] == s]
        parts[s] = (d[NUMERIC + CATEGORICAL], d["incident"].to_numpy(),
                    d["power_draw_watts"].to_numpy())
    return df, parts


df, parts = load_splits(DATA_DIR)
(Xtr, ytr, ptr), (Xva, yva, pva), (Xte, yte, pte) = (
    parts["train"], parts["val"], parts["test"])

print(f"{'split':8s}{'rows':>8}{'incident rate':>16}")
for s in ("train", "val", "test"):
    print(f"{s:8s}{len(parts[s][1]):8,}{parts[s][1].mean():16.4f}")
print(f"{'all':8s}{len(df):8,}{df['incident'].mean():16.4f}")

df.head()

# %% [markdown]
# ## §1. EDA for mixed types
#
# Lesson 1 taught you to look at marginals before modelling. Here the numerics
# get one glance — the real EDA problem in this dataset is categorical.
#
# ### The numerics, in one table
#
# Note the shapes: `disk_latency_ms` and `error_rate_pct` have a p99 several
# times their median (heavy right tails), `request_rate_rps` is bimodal
# (night/day traffic), `cache_hit_ratio` piles up near 1. We feed the models the
# **raw** columns on purpose. A linear model suffers on a heavy-tailed feature;
# a tree does not care, because trees only use the *ordering* of a feature, not
# its scale. That asymmetry is part of what the ladder measures.

# %%
glance = df[NUMERIC].describe(percentiles=[0.5, 0.99]).T[
    ["mean", "std", "50%", "99%", "max"]]
glance["skew"] = df[NUMERIC].skew()
glance.round(2)

# %% [markdown]
# ### Cardinalities — the shape of the categorical problem
#
# Five categorical columns, wildly different sizes. And one detail with real
# consequences: the train split contains only **174** of the 180 clusters. Six
# clusters live entirely in val/test — new capacity that came online after your
# training snapshot, which is exactly how it happens in production. Any encoding
# fitted on train has, by construction, never seen them.

# %%
card = pd.DataFrame({
    "levels_total": [df[c].nunique() for c in CATEGORICAL],
    "levels_in_train": [Xtr[c].nunique() for c in CATEGORICAL],
}, index=CATEGORICAL)
card["unseen_in_train"] = card["levels_total"] - card["levels_in_train"]
print(card)

unseen_clusters = sorted(set(df["cluster_id"]) - set(Xtr["cluster_id"]))
print("\nclusters never seen in train:", unseen_clusters)
assert len(unseen_clusters) == 6

# %% [markdown]
# ### Incident rate per level — always with uncertainty
#
# A rate from 12 rows and a rate from 12 000 rows are not the same kind of
# number, so every per-level rate below carries a 95% interval. We use the
# Wilson score interval — unlike the naive ±1.96·SE it behaves at small n and
# near 0 and 1, which is exactly where fleet data lives.

# %%
def wilson_ci(k, n, z=1.96):
    """95% Wilson score interval for a binomial proportion.

    k successes out of n trials (arrays or scalars). Returns (lo, hi).
    Unlike the normal approximation p +/- z*sqrt(p(1-p)/n), this never leaves
    [0, 1] and stays honest for small n.
    """
    # TODO (§1)   the Wilson score interval, vectorised
    #
    # With p = k/n and denom = 1 + z^2/n:
    #   centre = (p + z^2 / (2n)) / denom
    #   half   = (z / denom) * sqrt( p(1-p)/n + z^2 / (4 n^2) )
    #
    # Cast k and n to float arrays first (`np.asarray(..., dtype=float)`) --
    # rate_per_level below calls this with whole columns, and integer n
    # would make z**2/n integer-divide-adjacent nonsense on old habits.
    #
    # return: (centre - half, centre + half)
    raise NotImplementedError('wilson_ci')


def rate_per_level(frame, col, target="incident"):
    """Incident rate per level of a categorical column, with Wilson 95% CI.

    Returns a DataFrame indexed by level with columns n, rate, lo, hi,
    sorted by rate descending.
    """
    g = frame.groupby(col)[target].agg(n="size", rate="mean")
    lo, hi = wilson_ci(g["rate"] * g["n"], g["n"])
    g["lo"], g["hi"] = lo, hi
    return g.sort_values("rate", ascending=False)


by_channel = rate_per_level(df, "deploy_channel")
by_region = rate_per_level(df, "region")
print(by_channel.round(4))
print()
print(by_region.round(4))

# %% [markdown]
# `deploy_channel` is the loudest signal in the table: canary node-days hit
# incidents at ~33% against ~19% on stable, and the intervals are nowhere near
# overlapping — with thousands of rows per level, those differences are real.
# The region spread is smaller but also solid.
#
# ### Why you cannot eyeball a 180-level categorical
#
# Now try the same trick on `cluster_id` and watch it fall apart. 66 of the 180
# clusters have fewer than 25 rows. At n = 15, *one* incident moves the observed
# rate by 6.7 points — so a ranking of raw per-cluster rates mixes genuine
# monsters with lucky dice, and the rates alone cannot tell you which is which.

# %%
by_cluster = rate_per_level(df, "cluster_id")
sizes = by_cluster["n"]
print(f"cluster sizes: min {sizes.min()}, median {sizes.median():.0f}, "
      f"max {sizes.max()}, n<25: {(sizes < 25).sum()}")

fleet_rate = df["incident"].mean()
print(f"\nclusters with observed rate 0        : {(by_cluster['rate'] == 0).sum()}")
print(f"clusters with rate > 2x fleet average: "
      f"{(by_cluster['rate'] > 2 * fleet_rate).sum()}")

print("\ntop 5 clusters by raw incident rate — look at the n column:")
print(by_cluster.head(5).round(3))

# How many clusters are even distinguishable from the fleet average?
lo_b, hi_b = wilson_ci(fleet_rate * sizes, sizes)
outside = (by_cluster["rate"] < lo_b) | (by_cluster["rate"] > hi_b)
print(f"\nclusters outside their own 95% band around the fleet rate: "
      f"{outside.sum()} of {len(by_cluster)}")

# %% [markdown]
# Read that top-5 with the intervals, not the rates. The two large clusters
# near 90% are genuinely broken: their intervals exclude every sane rate. The
# three 8-row clusters at 87.5% are 7-of-8 streaks whose intervals stretch down
# to 0.53 — suggestive, provable for none of them individually. Zoom out and
# the funnel count says 81 of 180 clusters sit outside their own 95% band,
# where chance alone would put about 9: cluster identity carries *real* signal
# (the generator planted heavy-tailed cluster effects on purpose), much of it
# in small special-purpose clusters where per-level estimates are at their
# worst. A bar chart of 180 rates therefore fails in both directions — it
# flags healthy tiny clusters and it cannot certify the broken ones. You need
# estimates that share statistical strength across levels: a hierarchical
# model, or an encoding built for exactly this. Hold that thought until
# CatBoost's *ordered target statistics* in §7.

# %%
fig, axes = plt.subplots(1, 3, figsize=(12.5, 4.1),
                         gridspec_kw={"width_ratios": [0.75, 1.05, 1.5]})

for ax, tab, name in [(axes[0], by_channel, "deploy_channel"),
                      (axes[1], by_region, "region")]:
    tab = tab.iloc[::-1]                      # highest rate on top
    ypos = np.arange(len(tab))
    ax.errorbar(tab["rate"], ypos,
                xerr=[tab["rate"] - tab["lo"], tab["hi"] - tab["rate"]],
                fmt="o", color=C["blue"], ecolor=C["blue"],
                elinewidth=1.6, capsize=0, markersize=5.5)
    ax.axvline(fleet_rate, color=INK["muted"], linewidth=1.1, linestyle="--")
    ax.set_yticks(ypos)
    ax.set_yticklabels(tab.index, fontsize=8)
    finish(ax, name, xlabel="incident rate")
axes[0].annotate(f"fleet average\n{fleet_rate:.3f}",
                 xy=(fleet_rate + 0.004, 1.45), fontsize=8,
                 color=INK["muted"])

ax = axes[2]
ns = np.geomspace(7, 2100, 200)
f_lo, f_hi = wilson_ci(fleet_rate * ns, ns)
ax.fill_between(ns, f_lo, f_hi, color=INK["grid"], alpha=0.55, linewidth=0)
ax.plot(ns, f_lo, color=INK["muted"], linewidth=1.0, linestyle="--")
ax.plot(ns, f_hi, color=INK["muted"], linewidth=1.0, linestyle="--")
ax.axhline(fleet_rate, color=INK["muted"], linewidth=1.1)
ax.scatter(by_cluster["n"], by_cluster["rate"], s=16, color=C["blue"],
           alpha=0.65, linewidths=0)
ax.set_xscale("log")
ax.annotate("95% band if every cluster\nshared the fleet rate",
            xy=(11, 0.62), fontsize=8.5, color=INK["secondary"])
finish(ax, "cluster_id — 180 levels", xlabel="cluster size (rows, log)",
       ylabel="observed incident rate")

fig.suptitle("Per-level incident rates need uncertainty — and 180 levels defeat the eyeball",
             x=0.005, ha="left", fontsize=12, fontweight="semibold",
             color=INK["primary"])
fig.tight_layout(rect=(0, 0, 1, 0.94))
fig.savefig(FIG_DIR / "01_incident_rates.png", bbox_inches="tight", dpi=140)
plt.show()

# %% [markdown]
# ## §2. Linear regression on power draw
#
# Q1 first, because power draw is the honest place to *see* linear-model
# mechanics: the generator's mechanism is genuinely linear —
#
# $$\text{power} = 205 + 145\,\text{cpu} + 0.045\,\text{rps} + 2.8\,\text{queue}
#   + \text{instance offset} + \varepsilon$$
#
# — and the other **five numerics have a true coefficient of exactly zero**
# (ground truth in `data/dgp_metadata.json`, which we load and check against).
# The instance-type offsets and the noise are what OLS on numerics cannot
# explain; on standardised features the three real effects are roughly 24, 18
# and 16 W per standard deviation.
#
# We standardise (train statistics only!) so coefficients are comparable across
# features — and, in §3, so the penalty treats them symmetrically.

# %%
META = json.loads((DATA_DIR / "dgp_metadata.json").read_text())
true_coef_units = META["power_mechanism"]["coefficients"]
TRUE_SUPPORT = META["power_mechanism"]["true_support"]

scaler_reg = StandardScaler().fit(Xtr[NUMERIC])
Ztr_reg = scaler_reg.transform(Xtr[NUMERIC])
Zte_reg = scaler_reg.transform(Xte[NUMERIC])

ols = LinearRegression().fit(Ztr_reg, ptr)
sd = np.asarray(scaler_reg.scale_)
coef_table = pd.DataFrame({
    "true_per_unit": [true_coef_units[f] for f in NUMERIC],
    "true_per_sd": [true_coef_units[f] * s for f, s in zip(NUMERIC, sd)],
    "ols_per_sd": ols.coef_,
}, index=NUMERIC)
print(coef_table.round(3))
print(f"\nOLS test R^2: {r2_score(pte, ols.predict(Zte_reg)):.4f}")
print("(the missing ~0.29 is instance-type offsets + noise -- not ours to explain "
      "with numerics)")

# %% [markdown]
# OLS at n = 18 000 nails the truth: the three real coefficients land within a
# fraction of a watt of their true per-sd values, and the five true zeros
# estimate near zero. So what is the problem?
#
# ### What collinearity actually does
#
# `cpu_util` and `mem_pressure` are correlated (by design — busy nodes are busy
# everywhere), yet mem_pressure's true effect on power is **zero**:

# %%
r = df["cpu_util"].corr(df["mem_pressure"])
print(f"corr(cpu_util, mem_pressure) = {r:.3f}")
assert r > 0.5

# %% [markdown]
# Correlated columns make the least-squares problem *ill-conditioned in one
# direction*: many coefficient pairs $(\beta_{cpu}, \beta_{mem})$ with nearly
# the same sum produce nearly the same predictions, so the data pins down the
# sum tightly and the individual values loosely. At n = 18 000 there is enough
# signal to break the tie. Starve the model — refit on random 300-row
# subsamples — and you watch the pair wobble in opposite directions while their
# sum barely moves:

# %%
i_cpu, i_mem = NUMERIC.index("cpu_util"), NUMERIC.index("mem_pressure")
boots = np.empty((400, 2))
for b in range(400):
    idx = rng.integers(0, len(Ztr_reg), 300)
    fit_b = LinearRegression().fit(Ztr_reg[idx], ptr[idx])
    boots[b] = fit_b.coef_[i_cpu], fit_b.coef_[i_mem]

print(f"across 400 subsample fits (n=300 each):")
print(f"  sd(beta_cpu)          : {boots[:, 0].std():6.2f} W")
print(f"  sd(beta_mem)          : {boots[:, 1].std():6.2f} W")
print(f"  sd(beta_cpu+beta_mem) : {boots.sum(axis=1).std():6.2f} W  <- the sum is the stable thing")
print(f"  corr(beta_cpu, beta_mem): {np.corrcoef(boots.T)[0, 1]:+.2f}")

# %% [markdown]
# That −0.8 correlation between the two *estimates* is the fingerprint of
# collinearity: whatever one coefficient grabs, the other gives back. On small
# or noisy data this is how you end up presenting a large positive effect for a
# feature whose true effect is zero. Regularisation is the fix — which is §3.
#
# ## §3. Regularisation: ridge, lasso, and who recovers the truth
#
# Both penalties shrink coefficients toward zero; they differ in geometry and
# therefore in behaviour:
#
# * **Ridge** (L2, $\alpha\sum\beta_j^2$) shrinks smoothly and *shares* weight
#   across correlated features — it hates one big coefficient more than two
#   medium ones.
# * **Lasso** (L1, $\alpha\sum|\beta_j|$) has corners at zero, so as $\alpha$
#   grows it sets coefficients **exactly** to zero: built-in feature selection.
#
# ### Standardisation is not optional
#
# The penalty is on coefficient *magnitude*, and magnitude depends on units.
# `cpu_util` lives in [0, 1], so its true coefficient is 145 per unit;
# `request_rate_rps` lives in the hundreds, so its true coefficient is 0.045
# per unit. Penalise those magnitudes directly and you punish cpu_util ~3000×
# harder for the same predictive effect. Watch lasso get the support **wrong**
# on raw features and right on standardised ones — same $\alpha$:

# %%
def support_of(coefs, names, tol=1e-6):
    """Names of the features a fitted linear model actually uses.

    coefs: 1-D coefficient array. Returns the list of names whose coefficient
    magnitude exceeds tol, preserving input order.
    """
    return [n for n, c in zip(names, coefs) if abs(c) > tol]


lasso_raw = Lasso(alpha=2.0, max_iter=50_000).fit(Xtr[NUMERIC], ptr)
lasso_std = Lasso(alpha=2.0).fit(Ztr_reg, ptr)

print("true support           :", TRUE_SUPPORT)
print("lasso on RAW features  :", support_of(lasso_raw.coef_, NUMERIC))
print("lasso on STANDARDISED  :", support_of(lasso_std.coef_, NUMERIC))
assert support_of(lasso_std.coef_, NUMERIC) == TRUE_SUPPORT
assert support_of(lasso_raw.coef_, NUMERIC) != TRUE_SUPPORT

# %% [markdown]
# On raw features lasso keeps `disk_latency_ms` — a true zero — while crushing
# cpu_util's coefficient (which *must* be huge in raw units) to a third of its
# true value. Standardised, it recovers exactly the truth. Every penalised
# model in this course sits behind a scaler; now you know why.
#
# ### Coefficient paths
#
# The honest way to look at a penalty is the whole path: refit across a grid of
# $\alpha$ and watch every coefficient. (`lasso_path_coefs` refits per alpha —
# a deliberate, transparent inefficiency.)

# %%
def lasso_path_coefs(Z, y, alphas):
    """Lasso coefficients along an alpha grid, on already-standardised Z.

    Returns an array of shape (len(alphas), n_features): row i holds the
    fitted coefficient vector at alphas[i].
    """
    # TODO (§3)   refit Lasso once per alpha, stack the coefficient rows
    #
    # One list comprehension: `Lasso(alpha=a).fit(Z, y).coef_` for each a,
    # wrapped in np.array. Deliberately transparent and slow -- sklearn's
    # lasso_path does it faster, but you would not see the refits.
    #
    # return: array of shape (len(alphas), n_features)
    raise NotImplementedError('lasso_path_coefs')


def ridge_path_coefs(Z, y, alphas):
    """Ridge coefficients along an alpha grid, on already-standardised Z.

    Returns an array of shape (len(alphas), n_features), like lasso_path_coefs.
    """
    # TODO (§3)   the same recipe with Ridge
    #
    # Identical loop, `Ridge(alpha=a)` instead of `Lasso(alpha=a)`. Note
    # the RIDGE_ALPHAS grid below runs to 1e6 where lasso's stops at 100:
    # the L2 penalty has no corners, so it needs far bigger alphas to move.
    #
    # return: array of shape (len(alphas), n_features)
    raise NotImplementedError('ridge_path_coefs')


LASSO_ALPHAS = np.logspace(-2, 2, 81)      # the reference grid
RIDGE_ALPHAS = np.logspace(-1, 6, 71)
lasso_path = lasso_path_coefs(Ztr_reg, ptr, LASSO_ALPHAS)
ridge_path = ridge_path_coefs(Ztr_reg, ptr, RIDGE_ALPHAS)

# %% [markdown]
# ### Scoring lasso's variable selection against ground truth
#
# On real data you can only *admire* the features lasso picked. Here we can
# **score** it: for which alphas is the active set *exactly*
# {cpu_util, request_rate_rps, queue_depth}?

# %%
recovering = [a for a, coefs in zip(LASSO_ALPHAS, lasso_path)
              if sorted(support_of(coefs, NUMERIC)) == sorted(TRUE_SUPPORT)]
print(f"alphas recovering the exact true support: {len(recovering)} of "
      f"{len(LASSO_ALPHAS)}")
print(f"recovery window: alpha in [{min(recovering):.3f}, {max(recovering):.2f}]"
      f"  (~{np.log10(max(recovering) / min(recovering)):.1f} decades)")

example_alpha = recovering[len(recovering) // 2]
lasso_example = Lasso(alpha=example_alpha).fit(Ztr_reg, ptr)
example_r2 = r2_score(pte, lasso_example.predict(Zte_reg))
print(f"\nat alpha = {example_alpha:.2f} (mid-window):")
for n, c in zip(NUMERIC, lasso_example.coef_):
    mark = "  <- true support" if n in TRUE_SUPPORT else ""
    print(f"  {n:18s} {c:8.3f}{mark}")
print(f"test R^2: {example_r2:.4f}  (OLS with all 8: "
      f"{r2_score(pte, ols.predict(Zte_reg)):.4f})")

# %% [markdown]
# A window from 0.22 to 15.8 — nearly two decades of $\alpha$ — yields exactly
# the true support. "Lasso at a reasonable alpha recovers the truth" is a fair
# summary here, not luck. And the price of dropping five features is under a
# point of R².
#
# ### Choosing alpha by cross-validation — and its fine print

# %%
lasso_cv = LassoCV(alphas=LASSO_ALPHAS, cv=5).fit(Ztr_reg, ptr)
cv_support = support_of(lasso_cv.coef_, NUMERIC)
print(f"LassoCV picks alpha = {lasso_cv.alpha_:.4f}")
print(f"its support: {cv_support}")
print(f"recovers the exact truth: {sorted(cv_support) == sorted(TRUE_SUPPORT)}")

# %% [markdown]
# The CV-optimal alpha (0.20) sits *just below* the recovery window and keeps
# one extra feature, `error_rate_pct`. That is not a bug: CV optimises
# **prediction error**, and a slightly-too-rich model predicts marginally
# better than the sparsest true one. If your goal is *support recovery*
# (which signals are real?), CV-optimal alpha is systematically too small.
# Know which question you are asking.
#
# ### Ridge and the collinear pair
#
# Ridge never zeroes anything — instead watch what it does to our collinear
# pair from §2. The reference checkpoints: as alpha climbs 0 → 10³ → 10⁴ →
# 10⁵, the gap |β_cpu − β_mem| shrinks 24.9 → 19.7 → 9.7 → 2.1.

# %%
ridge_gaps = []
for a in (0.0, 1e3, 1e4, 1e5):
    m = LinearRegression().fit(Ztr_reg, ptr) if a == 0 else \
        Ridge(alpha=a).fit(Ztr_reg, ptr)
    gap = abs(m.coef_[i_cpu] - m.coef_[i_mem])
    ridge_gaps.append(gap)
    print(f"  alpha {a:>8.0f}: beta_cpu {m.coef_[i_cpu]:7.2f}  "
          f"beta_mem {m.coef_[i_mem]:6.2f}   |gap| {gap:6.2f}")
assert all(a > b for a, b in zip(ridge_gaps, ridge_gaps[1:])), \
    "ridge must shrink the collinear pair toward each other"

# %% [markdown]
# Ridge spreads weight across correlated features — stabilising, but it hands
# a visible coefficient to mem_pressure, whose true effect is zero. Lasso, in
# its window, gives mem_pressure exactly 0 and lets cpu_util keep the credit.
# Neither is "better": ridge is the choice for prediction under collinearity,
# lasso for a sparse explanation. The paths below are the whole story in one
# figure.

# %%
fig, axes = plt.subplots(1, 2, figsize=(11.8, 4.4), sharey=True)
path_colors = {"cpu_util": C["blue"], "request_rate_rps": C["orange"],
               "queue_depth": C["aqua"]}

for ax, path, alphas, name in [
        (axes[0], ridge_path, RIDGE_ALPHAS, "ridge — shrinks, never kills"),
        (axes[1], lasso_path, LASSO_ALPHAS, "lasso — kills, feature by feature")]:
    for j, feat in enumerate(NUMERIC):
        col = path_colors.get(feat, "#b9b7ae")
        lw = 2.2 if feat in path_colors else 1.2
        ax.plot(alphas, path[:, j], color=col, linewidth=lw)
    ax.set_xscale("log")
    ax.axhline(0, color=INK["axis"], linewidth=0.8)
    finish(ax, name, xlabel="alpha (log scale)")

axes[0].set_ylabel("coefficient (W per sd)")
for feat, col in path_colors.items():
    j = NUMERIC.index(feat)
    axes[0].annotate(feat, xy=(RIDGE_ALPHAS[0], ridge_path[0, j]),
                     xytext=(3, 3), textcoords="offset points",
                     fontsize=8.5, color=col, fontweight="semibold")
axes[0].annotate("5 true zeros", xy=(RIDGE_ALPHAS[0], 1.2), xytext=(3, 3),
                 textcoords="offset points", fontsize=8.5, color=INK["muted"])

axes[1].axvspan(min(recovering), max(recovering), color=INK["grid"], alpha=0.5,
                linewidth=0)
axes[1].annotate("exact-support\nwindow", xy=(1.8, 8.5), fontsize=8.5,
                 color=INK["secondary"], ha="center")
axes[1].axvline(lasso_cv.alpha_, color=INK["muted"], linewidth=1.1,
                linestyle="--")
axes[1].annotate(f"LassoCV\n{lasso_cv.alpha_:.2f}", xy=(lasso_cv.alpha_, 4.5),
                 xytext=(-5, 0), textcoords="offset points", fontsize=8,
                 color=INK["muted"], ha="right")
fig.suptitle("Coefficient paths on standardised features",
             x=0.005, ha="left", fontsize=12, fontweight="semibold",
             color=INK["primary"])
fig.tight_layout(rect=(0, 0, 1, 0.95))
fig.savefig(FIG_DIR / "02_regularization_paths.png", bbox_inches="tight",
            dpi=140)
plt.show()

# %% [markdown]
# ## §4. Logistic regression on incidents — the ladder begins
#
# Back to Q2. From here on, every model follows the same protocol: **fit on
# train, use val for anything that needs choosing, report test ROC-AUC.**
#
# ### Model 1 — numerics only
#
# The baseline you always build first. Eight standardised numerics, nothing
# else. It cannot see regions, channels or clusters, and it meets the raw
# heavy-tailed columns head-on.

# %%
def fit_logreg_numeric(parts, numeric):
    """Ladder model 1: logistic regression on standardised numerics only.

    parts is the split dict from load_splits; numeric the list of numeric
    column names. Fits StandardScaler + LogisticRegression(max_iter=1000) on
    train. Returns a dict with keys model, val_auc, test_auc, n_features.
    """
    # TODO (§4)   ladder model 1 -- the baseline you always build first
    #
    # Pipeline: StandardScaler -> LogisticRegression(max_iter=1000), fitted
    # on Xtr[numeric] ONLY -- this model must not see a single categorical.
    #
    # Score with `predict_proba(...)[:, 1]`: AUC ranks probabilities, and
    # .predict() would throw the ranking away.
    #
    # return: dict with keys model, val_auc, test_auc, n_features
    raise NotImplementedError('fit_logreg_numeric')


ladder = {}
ladder["logreg_numeric"] = fit_logreg_numeric(parts, NUMERIC)
print(f"model 1 — logreg, numerics only")
print(f"  val  AUC {ladder['logreg_numeric']['val_auc']:.4f}")
print(f"  test AUC {ladder['logreg_numeric']['test_auc']:.4f}")

# %% [markdown]
# Test AUC **0.719**. Underwhelming — deliberately so: the numeric main effects
# in this dataset are weak, and the strong signals (channel, the cluster
# hierarchy, thresholds, interactions) are all invisible from these eight
# columns. This number is the floor the categoricals will beat.
#
# ### Model 2 — add the categoricals, pay the OHE price
#
# One-hot encoding maps each level to its own 0/1 column inside a
# `ColumnTransformer`, so scaling (numerics) and encoding (categoricals) live
# in one fitted object — fit on train, applied identically everywhere. Two
# choices matter:
#
# * `handle_unknown="ignore"` — an unseen level encodes as **all zeros**
#   instead of crashing. Remember the six unseen clusters.
# * no `drop="first"` — with L2 regularisation the redundancy is harmless, and
#   coefficients stay readable per level.

# %%
def build_ohe_pipeline(numeric, categorical):
    """The model-2/3 preprocessor: scale numerics, one-hot the categoricals.

    Returns an (unfitted) ColumnTransformer that standardises the numeric
    columns and one-hot encodes the categorical ones with
    handle_unknown="ignore", so levels never seen in train encode as all-zero
    rows instead of raising.
    """
    # TODO (§4)   ColumnTransformer: scale the numerics, one-hot the cats
    #
    # Two named transformers: ("num", StandardScaler(), numeric) and
    # ("cat", OneHotEncoder(...), categorical).
    #
    # handle_unknown="ignore" is the load-bearing choice -- six clusters
    # exist only in val/test, and a level the encoder never saw must
    # become an all-zero block, not a crash. No drop="first": with L2 the
    # redundancy is harmless and coefficients stay readable per level.
    #
    # return: the (unfitted) ColumnTransformer
    raise NotImplementedError('build_ohe_pipeline')


def fit_logreg_ohe(parts, numeric, categorical):
    """Ladder model 2: logistic regression on numerics + one-hot categoricals.

    Uses build_ohe_pipeline for preprocessing, then
    LogisticRegression(max_iter=1000). Returns a dict with keys model,
    val_auc, test_auc, n_features (the post-encoding column count).
    """
    # TODO (§4)   ladder model 2 -- numerics + all 5 categoricals
    #
    # Pipeline: build_ohe_pipeline(numeric, categorical) ->
    # LogisticRegression(max_iter=1000), fitted on the full Xtr.
    #
    # n_features is the POST-encoding column count: transform a single row
    # (`model["prep"].transform(Xtr[:1]).shape[1]`). Expect ~213 -- the
    # explosion is the point of the section, so report it honestly.
    #
    # return: dict with keys model, val_auc, test_auc, n_features
    raise NotImplementedError('fit_logreg_ohe')


ladder["logreg_ohe"] = fit_logreg_ohe(parts, NUMERIC, CATEGORICAL)
print(f"model 2 — logreg + OHE on 5 categoricals")
print(f"  val  AUC {ladder['logreg_ohe']['val_auc']:.4f}")
print(f"  test AUC {ladder['logreg_ohe']['test_auc']:.4f}")
print(f"  features after encoding: {ladder['logreg_ohe']['n_features']}")
print(f"  gain over model 1: "
      f"{ladder['logreg_ohe']['test_auc'] - ladder['logreg_numeric']['test_auc']:+.4f}")

# %% [markdown]
# Test AUC **0.768** — the categoricals are worth about five AUC points. Now
# look at what it cost:

# %%
prep = ladder["logreg_ohe"]["model"]["prep"]
feat_names = prep.get_feature_names_out()
blocks = {"numeric": sum(1 for n in feat_names if n.startswith("num__"))}
for c in CATEGORICAL:
    blocks[c] = sum(1 for n in feat_names if n.startswith(f"cat__{c}_"))
print("column budget after one-hot encoding:")
for k, v in blocks.items():
    print(f"  {k:15s} {v:4d}")
print(f"  {'TOTAL':15s} {len(feat_names):4d}")
assert len(feat_names) == 213

# %% [markdown]
# **8 features became 213 columns**, and 174 of them belong to `cluster_id`
# alone (174, not 180 — the encoder can only learn levels train contains). Each
# of those 174 columns gets its own coefficient estimated from however many
# rows that cluster has; for the 8-row clusters that is a coefficient hanging
# from 8 observations, held in place mostly by the L2 penalty. The explosion is
# not just memory — it is *statistical dilution*.
#
# ### Unseen levels: graceful degradation, not magic

# %%
one_unseen = unseen_clusters[0]
row = Xva[Xva["cluster_id"] == one_unseen].iloc[:1]
encoded = prep.transform(row)
cluster_cols = [i for i, n in enumerate(feat_names)
                if n.startswith("cat__cluster_id_")]
block_sum = encoded[:, cluster_cols].sum()
print(f"val row from unseen cluster {one_unseen!r}:")
print(f"  sum over all 174 cluster_id columns: {block_sum:.0f}")
assert block_sum == 0

# %% [markdown]
# The whole cluster block is zeros: for this row, the model silently falls back
# to "average cluster, judged by numerics + region + channel + instance + OS".
# No crash — but also no cluster information, on exactly the nodes newest in
# the fleet. Keep this in mind as a *structural* limit of one-hot encoding.
#
# ### Reading coefficients as odds ratios
#
# A logistic coefficient β is a log-odds-ratio: holding everything else fixed,
# the level multiplies the odds of an incident by $e^{\beta}$. Because we kept
# all levels (no dropped baseline), the meaningful quantity is a *difference*
# between two levels' coefficients:

# %%
coefs2 = ladder["logreg_ohe"]["model"]["clf"].coef_[0]
ch_coef = {n.split("deploy_channel_")[1]: coefs2[i]
           for i, n in enumerate(feat_names) if "deploy_channel_" in n}
print("deploy_channel coefficients:",
      {k: round(v, 3) for k, v in ch_coef.items()})
print(f"\nodds ratio canary vs stable: "
      f"{np.exp(ch_coef['canary'] - ch_coef['stable']):.2f}")
print(f"odds ratio beta   vs stable: "
      f"{np.exp(ch_coef['beta'] - ch_coef['stable']):.2f}")

raw_odds = (by_channel.loc["canary", "rate"] / (1 - by_channel.loc["canary", "rate"])) \
    / (by_channel.loc["stable", "rate"] / (1 - by_channel.loc["stable", "rate"]))
print(f"\nfor contrast, the *unadjusted* odds ratio from §1's table: {raw_odds:.2f}")

# %% [markdown]
# Canary multiplies the odds of an incident by **~2.7** relative to stable,
# *after adjusting* for everything else in the model — workload, region,
# instance, cluster. The unadjusted table-level ratio (~2.2) is a different
# number answering a different question; the adjusted one is larger here
# because canary nodes do not otherwise look especially risky. This
# interpretability — one number per level, with a causal-flavoured reading you
# can defend in a postmortem — is the argument for logistic regression that
# AUC tables never show.
#
# ## §5. One decision tree: the staircase
#
# A decision tree asks one question per node — always of the form
# `feature <= threshold`, a single **axis-parallel** cut — and routes each row
# down to a leaf, predicting the leaf's training rate. Fitting is greedy: try
# every feature and threshold, keep the split that most reduces impurity,
# recurse.
#
# Remember this and half of tree folklore evaporates: *every* tree in every
# library — CART, random forests, XGBoost, LightGBM, CatBoost — splits
# axis-parallel. A tree's decision boundary is therefore a **staircase**. On
# two features you can watch it try to approximate an oblique, interacting
# truth:

# %%
two = ["cpu_util", "mem_pressure"]
X2 = Xtr[two].to_numpy()
tree_shallow = DecisionTreeClassifier(max_depth=3, min_samples_leaf=50,
                                      random_state=RANDOM_SEED).fit(X2, ytr)
tree_deep = DecisionTreeClassifier(max_depth=12, min_samples_leaf=5,
                                   random_state=RANDOM_SEED).fit(X2, ytr)

gx = np.linspace(X2[:, 0].min(), X2[:, 0].max(), 300)
gy = np.linspace(X2[:, 1].min(), X2[:, 1].max(), 300)
GX, GY = np.meshgrid(gx, gy)
grid = np.c_[GX.ravel(), GY.ravel()]

RISK_CMAP = LinearSegmentedColormap.from_list(
    "risk", ["#5795de", "#cde2fb", "#f0efec", "#f8cdb0", "#ef8f5c"])
norm = TwoSlopeNorm(vmin=0.0, vcenter=fleet_rate, vmax=1.0)

pick = rng.choice(len(X2), 1500, replace=False)
fig, axes = plt.subplots(1, 2, figsize=(11.6, 4.9), sharey=True)
for ax, model, title in [
        (axes[0], tree_shallow, "depth 3 — a staircase, and that is fine"),
        (axes[1], tree_deep, "depth 12, min_samples_leaf=5 — the staircase memorises")]:
    P = model.predict_proba(grid)[:, 1].reshape(GX.shape)
    mesh = ax.pcolormesh(GX, GY, P, cmap=RISK_CMAP, norm=norm, shading="auto")
    ax.contour(GX, GY, P, levels=[fleet_rate], colors=[INK["primary"]],
               linewidths=1.1)
    for cls, col in [(0, C["blue"]), (1, C["orange"])]:
        m = ytr[pick] == cls
        ax.scatter(X2[pick][m, 0], X2[pick][m, 1], s=7, color=col, alpha=0.75,
                   linewidths=0.2, edgecolors="white",
                   label=f"incident={cls}")
    ax.grid(False)
    finish(ax, title, xlabel="cpu_util")
axes[0].set_ylabel("mem_pressure")
axes[0].legend(loc="upper left", markerscale=1.8)
cb = fig.colorbar(mesh, ax=axes, fraction=0.035, pad=0.02)
cb.set_label("predicted P(incident)", fontsize=8.5, color=INK["secondary"])
cb.outline.set_visible(False)
fig.suptitle("A tree's boundary is axis-parallel by construction — in every library",
             x=0.005, ha="left", fontsize=12, fontweight="semibold",
             color=INK["primary"])
fig.savefig(FIG_DIR / "03_tree_staircase.png", bbox_inches="tight", dpi=140)
plt.show()

va2 = Xva[two].to_numpy()
print(f"val AUC, depth 3 : {roc_auc_score(yva, tree_shallow.predict_proba(va2)[:, 1]):.4f}")
print(f"val AUC, depth 12: {roc_auc_score(yva, tree_deep.predict_proba(va2)[:, 1]):.4f}")

# %% [markdown]
# The shallow tree carves the high-cpu, high-mem corner — a real interaction
# from the generator, which no linear model on these two raw features could
# express. The deep tree shatters the plane into confetti *and scores worse on
# val*: those tiny boxes are individual training rows, memorised.
#
# ### Depth is the overfitting dial
#
# Sweep depth on the full 213-column matrix (same preprocessing as model 2)
# and score train vs val at each step — once with no leaf-size floor, once
# with `min_samples_leaf=50`:

# %%
def tree_depth_sweep(Ztr, ytr, Zva, yva, depths, min_samples_leaf=1):
    """Train and val AUC of a decision tree at each depth in `depths`.

    Fits DecisionTreeClassifier(max_depth=d, min_samples_leaf=...,
    random_state=RANDOM_SEED) on the (already preprocessed) train matrix for
    each d. Returns (train_aucs, val_aucs) as two lists aligned with depths.
    """
    # TODO (§5)   train-vs-val AUC of a tree at every depth in `depths`
    #
    # For each d: fit DecisionTreeClassifier(max_depth=d,
    # min_samples_leaf=min_samples_leaf, random_state=RANDOM_SEED) on the
    # already-preprocessed (Ztr, ytr), then score BOTH matrices with
    # predict_proba(...)[:, 1]. The train/val gap is the overfitting
    # figure; forget the train score and the plot has nothing to say.
    #
    # return: (train_aucs, val_aucs), two lists aligned with depths
    raise NotImplementedError('tree_depth_sweep')


prep_tree = build_ohe_pipeline(NUMERIC, CATEGORICAL)
Ztr_full = prep_tree.fit_transform(Xtr)
Zva_full = prep_tree.transform(Xva)

DEPTHS = list(range(1, 17))
t0 = time.time()
sweep = {leaf: tree_depth_sweep(Ztr_full, ytr, Zva_full, yva, DEPTHS,
                                min_samples_leaf=leaf)
         for leaf in (1, 50)}
print(f"32 trees fitted in {time.time() - t0:.1f}s")
for leaf in (1, 50):
    tr_a, va_a = sweep[leaf]
    best = int(np.argmax(va_a))
    print(f"min_samples_leaf={leaf:2d}: best val AUC {max(va_a):.4f} "
          f"at depth {DEPTHS[best]}")

# %%
fig, axes = plt.subplots(1, 2, figsize=(11.4, 4.2), sharey=True)
for ax, leaf, title in [(axes[0], 1, "min_samples_leaf = 1 — free to memorise"),
                        (axes[1], 50, "min_samples_leaf = 50 — a floor under every leaf")]:
    tr_a, va_a = sweep[leaf]
    ax.plot(DEPTHS, tr_a, color=C["blue"], marker="o", markersize=3.5)
    ax.plot(DEPTHS, va_a, color=C["orange"], marker="o", markersize=3.5)
    ax.annotate("train", xy=(DEPTHS[-1], tr_a[-1]), xytext=(5, 0),
                textcoords="offset points", color=C["blue"], fontsize=9,
                fontweight="semibold", va="center")
    ax.annotate("val", xy=(DEPTHS[-1], va_a[-1]), xytext=(5, 0),
                textcoords="offset points", color=C["orange"], fontsize=9,
                fontweight="semibold", va="center")
    best = int(np.argmax(va_a))
    ax.scatter([DEPTHS[best]], [va_a[best]], s=42, facecolors="none",
               edgecolors=C["orange"], linewidths=1.4, zorder=5)
    finish(ax, title, xlabel="max_depth")
axes[0].set_ylabel("ROC-AUC")
fig.suptitle("Depth sweep on the full 213-column matrix: the gap is the overfit",
             x=0.005, ha="left", fontsize=12, fontweight="semibold",
             color=INK["primary"])
fig.tight_layout(rect=(0, 0, 1, 0.94))
fig.savefig(FIG_DIR / "04_tree_depth_sweep.png", bbox_inches="tight", dpi=140)
plt.show()

# %% [markdown]
# Left panel, the classic overfitting picture: past depth 8 train AUC keeps
# climbing toward 0.9+ while val *falls off a cliff* — every extra level is
# fitting noise. Right panel: a leaf-size floor of 50 rows caps how personal a
# leaf can get, so val creeps up and plateaus around 0.78 instead of
# collapsing. Regularising a tree means bounding how few rows may vote.
#
# ### Model 3 — one tree for the ladder
#
# The ladder's tree is fixed at `max_depth=6, min_samples_leaf=50`: deep
# enough to express the interactions, shallow enough to still read.

# %%
def fit_tree(parts, numeric, categorical, max_depth=6, min_samples_leaf=50):
    """Ladder model 3: a single decision tree on the same matrix as model 2.

    Same build_ohe_pipeline preprocessing, then DecisionTreeClassifier with
    the given depth/leaf settings and random_state=RANDOM_SEED. Returns a
    dict with keys model, val_auc, test_auc, n_features.
    """
    # TODO (§5)   ladder model 3 -- one tree, same 213 columns as model 2
    #
    # Pipeline: build_ohe_pipeline(numeric, categorical) ->
    # DecisionTreeClassifier(max_depth=max_depth,
    # min_samples_leaf=min_samples_leaf, random_state=RANDOM_SEED).
    # Same protocol as models 1-2: fit on train, AUC on val and test,
    # n_features from the fitted preprocessor.
    #
    # return: dict with keys model, val_auc, test_auc, n_features
    raise NotImplementedError('fit_tree')


ladder["tree"] = fit_tree(parts, NUMERIC, CATEGORICAL)
print(f"model 3 — decision tree (depth 6, min_samples_leaf 50)")
print(f"  val  AUC {ladder['tree']['val_auc']:.4f}")
print(f"  test AUC {ladder['tree']['test_auc']:.4f}")

# %% [markdown]
# Test AUC **0.737** — above the numeric-only logreg (nonlinearities are worth
# something) but *below* the OHE logreg. A single depth-6 tree has at most 64
# leaves; against 213 columns it must spend whole levels isolating individual
# dummy columns, and each split it takes fragments the data it can use below.
# One tree is a weak learner. That is not an insult — it is the job
# description for §6.
#
# ### Reading a tree
#
# The one unbeatable feature of a shallow tree: you can print it. Fit a
# depth-3 tree on the same matrix and look at what it chose:

# %%
clean_names = [n.split("__", 1)[1] for n in prep_tree.get_feature_names_out()]
tree3 = DecisionTreeClassifier(max_depth=3, min_samples_leaf=50,
                               random_state=RANDOM_SEED).fit(Ztr_full, ytr)

root_feat = clean_names[tree3.tree_.feature[0]]
root_thr = tree3.tree_.threshold[0]
lat_mean = Xtr["disk_latency_ms"].mean()
lat_sd = Xtr["disk_latency_ms"].std(ddof=0)
print(f"root split: {root_feat} <= {root_thr:.3f} (standardised)")
print(f"  in raw units: disk_latency_ms <= {lat_mean + root_thr * lat_sd:.1f} ms")
print(f"  true threshold planted by the generator: "
      f"{META['incident_mechanism']['disk_latency_threshold_ms']} ms")
assert root_feat == "disk_latency_ms"

fig, ax = plt.subplots(figsize=(13.5, 5.6))
plot_tree(tree3, feature_names=clean_names, class_names=["fine", "incident"],
          filled=True, impurity=False, proportion=True, rounded=True,
          fontsize=8, ax=ax)
ax.set_title("A depth-3 tree is a document you can read", pad=10)
fig.savefig(FIG_DIR / "05_tree_read.png", bbox_inches="tight", dpi=140)
plt.show()

# %% [markdown]
# Nobody told the tree about the generator's latency threshold at 14.1 ms
# (≈ p85). It *found* it — the root split sits at ≈ 14.0 ms — and one level
# down it splits on `deploy_channel_stable`, which is precisely the
# "high latency hurts more on beta/canary" interaction the generator planted.
# Notice it also spends a split isolating a single cluster's dummy column:
# with one-hot encoding, that is the only way a tree can use `cluster_id` —
# one level per split. File that under "why §7 exists".
#
# ## §6. Boosting, by hand
#
# One tree underfits. Boosting's idea is embarrassingly simple: **fit a weak
# model, look at what it got wrong, fit the next weak model to exactly that,
# repeat.** For squared-error regression, "what it got wrong" is literally the
# residual $y - F_m(x)$ — which is also the negative gradient of the loss, and
# that little identity is what generalises this trick to any differentiable
# loss (for log-loss you boost on gradients instead of raw residuals).
#
# Watch it happen on a toy 1-D problem, with stumps — depth-1 trees, the
# weakest learner that does anything:

# %%
def boost_by_hand(x, y, n_rounds=3):
    """Gradient boosting for squared loss, from scratch, in its smallest form.

    Stage 0 predicts the constant mean of y. Each round fits a depth-1
    regression tree (a stump) to the current residuals y - F(x) and adds its
    prediction, at full weight (no shrinkage), to the ensemble.

    Returns a dict with keys:
      f0         -- the stage-0 constant (float),
      stumps     -- the fitted DecisionTreeRegressor stumps, in order,
      train_pred -- predictions on x after 0..n_rounds stages (list of arrays),
      train_mse  -- MSE on (x, y) after 0..n_rounds stages (list of floats).
    """
    # TODO (§6)   gradient boosting for squared loss, smallest honest form
    #
    # Stage 0: pred = the constant y.mean(). Then, n_rounds times:
    #   residual = y - pred          <- recomputed EVERY round, not once
    #   stump = DecisionTreeRegressor(max_depth=1,
    #           random_state=RANDOM_SEED).fit(X, residual)
    #   pred = pred + stump.predict(X)    (full weight -- no shrinkage yet)
    #
    # X is x.reshape(-1, 1): sklearn wants 2-D. Record pred and the MSE
    # after stage 0 AND after every round, and append pred.copy() --
    # without the copy every history entry aliases the same array.
    #
    # return: dict with keys f0, stumps, train_pred, train_mse
    #         (train_pred and train_mse hold n_rounds + 1 entries)
    raise NotImplementedError('boost_by_hand')


x_toy = np.sort(rng.uniform(0, 3, 90))
y_toy = np.sin(2.0 * x_toy) + rng.normal(0, 0.25, x_toy.size)
boosted = boost_by_hand(x_toy, y_toy, n_rounds=3)

for m, mse in enumerate(boosted["train_mse"]):
    what = "predict the mean" if m == 0 else f"+ stump {m}"
    print(f"stage {m} ({what:18s}) MSE {mse:.3f}")
assert all(a > b for a, b in zip(boosted["train_mse"], boosted["train_mse"][1:]))

# %%
xs_grid = np.linspace(0, 3, 600)
fig, axes = plt.subplots(2, 2, figsize=(10.8, 6.6), sharex=True, sharey=True)
for m, ax in enumerate(axes.ravel()):
    grid_pred = np.full(xs_grid.size, boosted["f0"])
    for stump in boosted["stumps"][:m]:
        grid_pred += stump.predict(xs_grid.reshape(-1, 1))
    ax.scatter(x_toy, y_toy, s=13, color=INK["muted"], alpha=0.75, linewidths=0)
    ax.plot(xs_grid, grid_pred, color=C["blue"], linewidth=2.2)
    title = "stage 0 — the mean" if m == 0 else f"stage {m} — {m} stump{'s' if m > 1 else ''}"
    finish(ax, title)
    ax.annotate(f"MSE {boosted['train_mse'][m]:.3f}", xy=(0.97, 0.06),
                xycoords="axes fraction", ha="right", fontsize=9,
                color=INK["secondary"], fontweight="semibold")
for ax in axes[1]:
    ax.set_xlabel("x")
fig.suptitle("Boosting by hand: each stump fits the previous ensemble's residuals",
             x=0.005, ha="left", fontsize=12, fontweight="semibold",
             color=INK["primary"])
fig.tight_layout(rect=(0, 0, 1, 0.95))
fig.savefig(FIG_DIR / "06_boosting_by_hand.png", bbox_inches="tight", dpi=140)
plt.show()

# %% [markdown]
# Three stumps cut the MSE from 0.50 to 0.10. Each panel's step function is the
# *sum* of everything before it; no single stump is any good, and the ensemble
# is already respectable.
#
# A production GBDT is this loop plus exactly three upgrades:
#
# 1. **shrinkage** — each tree is added scaled by a learning rate
#    (`pred += lr * tree.predict(X)`), so no single tree can dominate and
#    hundreds of small corrections average out each other's noise;
# 2. **depth-d trees** instead of stumps, to capture interactions;
# 3. **a stopping rule** — monitor a validation set and stop when it stops
#    improving, which you are about to see as `early_stopping_rounds`.
#
# That is the entire conceptual content of XGBoost, LightGBM and CatBoost.
# Everything else is engineering — some of it brilliant, as next section shows.
#
# ## §7. CatBoost: boosting that speaks categorical
#
# ### Model 4 — defaults, native categoricals
#
# No encoding pipeline. You hand CatBoost the raw DataFrame and *tell it which
# columns are categorical*; 13 features in, not 213. Val is the early-stopping
# monitor, and otherwise we touch nothing.

# %%
def fit_catboost_default(parts, categorical):
    """Ladder model 4: CatBoostClassifier with default settings.

    iterations=500, random_seed=RANDOM_SEED, native cat_features, early
    stopping on the val set (50 rounds, best model kept), otherwise pure
    defaults -- notably eval_metric stays Logloss. Returns a dict with keys
    model, val_auc, test_auc, best_iteration, n_features.
    """
    # TODO (§7)   ladder model 4 -- CatBoost defaults, native categoricals
    #
    # No encoding pipeline: pass the raw DataFrame and
    # cat_features=categorical -- COLUMN NAMES, because Xtr is a DataFrame.
    # CatBoostClassifier(iterations=500, random_seed=RANDOM_SEED,
    # verbose=0, allow_writing_files=False); fit with eval_set=(Xva, yva),
    # early_stopping_rounds=50, use_best_model=True. Touch nothing else --
    # eval_metric stays Logloss; "defaults" is the whole rung.
    #
    # return: dict with keys model, val_auc, test_auc, best_iteration
    #         (int(model.get_best_iteration())), n_features
    raise NotImplementedError('fit_catboost_default')


t0 = time.time()
ladder["catboost_default"] = fit_catboost_default(parts, CATEGORICAL)
cb = ladder["catboost_default"]
print(f"model 4 — CatBoost defaults, native cats  ({time.time() - t0:.1f}s)")
print(f"  val  AUC {cb['val_auc']:.4f}")
print(f"  test AUC {cb['test_auc']:.4f}")
print(f"  best iteration {cb['best_iteration']} of 500 "
      f"(auto learning rate "
      f"{cb['model'].get_all_params()['learning_rate']:.4f})")
print(f"  gain over model 2: "
      f"{cb['test_auc'] - ladder['logreg_ohe']['test_auc']:+.4f}")

# %% [markdown]
# Test AUC **0.844** — a jump of nearly eight points over the best linear
# model, with zero tuning. That gap is the honest sum of three things the
# linear ladder could not do: the nonlinearities (§5 found them one at a
# time; 300 trees find them all), invariance to the heavy-tailed raw features,
# and a 180-level categorical actually being *used* instead of diluted across
# 174 dummy columns.
#
# ### What CatBoost actually is — the corrected story
#
# First, the myth. You will hear — in interviews, in blog posts — that
# CatBoost is special because "it splits with vertical and horizontal lines" /
# "uses axis-parallel splits". **That is not a distinguishing feature; it is
# the definition of a decision tree.** Every library named in this lesson
# splits axis-parallel; the staircase of §5 belongs to all of them equally,
# and it is a *limitation* they share — an oblique boundary must be
# approximated by stairs. What is actually specific to CatBoost:
#
# 1. **Oblivious (symmetric) trees.** All nodes at the same level of a tree
#    share **one** split condition, so a depth-6 tree is just 6 questions and
#    a leaf is the 6-bit answer string. Individually even weaker than free-form
#    CART trees — which is fine (§6: weak is the job) — plus a strong built-in
#    regulariser and extremely fast inference: evaluate 6 conditions, index
#    into 64 leaves.
#
# 2. **Ordered boosting.** In classic GBDT, tree $m$ is fit to gradients of a
#    model that was itself trained on the *same rows* — every row helps grade
#    its own homework, a subtle self-leak ("prediction shift") that inflates
#    train fit. CatBoost fights it with random permutations: the gradient for
#    row $i$ comes from a model state trained only on rows *before* $i$ in the
#    permutation.
#
# 3. **Ordered target statistics** for categoricals — the reason this lesson's
#    dataset has a 180-level column. Instead of 174 dummy columns, encode a
#    level by (roughly) *the average target among earlier rows of that level*
#    in a random permutation, plus a prior: row i's own label is **never** in
#    its own encoding. One dense number per categorical; rare levels get
#    shrunk toward the prior automatically (the §1 funnel problem, solved
#    en passant); and CatBoost builds *combinations* (e.g. cluster × channel)
#    on the fly.
#
# 4. **Defaults that hold up.** Auto learning rate, sensible depth, ordered
#    boosting on by default on data this size. Remember 0.844-with-no-tuning
#    when §8 tells you what 30 trials of tuning bought.
#
# ### Feature importances — a quick look, with the right disclaimer
#
# CatBoost's default importance ("how much do predictions change when this
# feature moves") is a *usage* report, not ground truth about the world:

# %%
imp = pd.Series(cb["model"].get_feature_importance(),
                index=NUMERIC + CATEGORICAL).sort_values()
is_cat = [f in CATEGORICAL for f in imp.index]

fig, ax = plt.subplots(figsize=(8.6, 4.6))
ax.barh(np.arange(len(imp)), imp.to_numpy(), height=0.62,
        color=[C["orange"] if c else C["blue"] for c in is_cat])
ax.set_yticks(np.arange(len(imp)))
ax.set_yticklabels(imp.index, fontsize=8.5)
for i, v in enumerate(imp.to_numpy()):
    ax.annotate(f"{v:.1f}", xy=(v, i), xytext=(4, 0),
                textcoords="offset points", va="center", fontsize=8,
                color=INK["secondary"])
ax.scatter([], [], color=C["blue"], label="numeric", marker="s")
ax.scatter([], [], color=C["orange"], label="categorical", marker="s")
finish(ax, "CatBoost feature importance (prediction value change, %)",
       xlabel="importance", legend=True)
fig.tight_layout()
fig.savefig(FIG_DIR / "07_catboost_importance.png", bbox_inches="tight",
            dpi=140)
plt.show()

print("top 4:", ", ".join(f"{k} ({v:.1f})" for k, v in
                          imp.sort_values(ascending=False).head(4).items()))

# %% [markdown]
# `cluster_id` — the feature one-hot encoding turned into 174 starving dummy
# columns — is right up with the top numerics. Target statistics made a
# 180-level categorical *cheap*. Note also cpu_util and mem_pressure both
# ranking high: the model leans on their interaction, which no additive story
# of "importance per feature" can honestly decompose. Treat importances as a
# map of what the model uses, never as effect sizes.
#
# ## §8. Tuning with Optuna — what is it actually worth?
#
# Model 4 used defaults. The final rung asks, with a proper budget and
# protocol, how much a careful hyperparameter search adds. Vocabulary:
# a **study** is the search; each **trial** picks a parameter set from the
# search space and returns a score; the **sampler** (TPE — tree-structured
# Parzen estimator) fits light density models of "good" vs "bad" regions and
# proposes where to look next — noticeably better than random search at ~30
# trials, and seeded here so the whole study is reproducible.
#
# The search space, per the reference protocol:
#
# | parameter | range | scale |
# |---|---|---|
# | `depth` | 4 … 8 | int |
# | `learning_rate` | 0.02 … 0.3 | log |
# | `l2_leaf_reg` | 1 … 30 | log |
#
# with `iterations=800` and early stopping on val inside every trial — so the
# *effective* tree count is tuned for free — and `eval_metric="AUC"`, so early
# stopping now watches val AUC (the thing we optimise) rather than Logloss.
# The objective returns **val** AUC. Test appears nowhere in the loop.

# %%
def fit_catboost_eval_auc(parts, categorical, params):
    """Fit CatBoost for the tuning protocol: 800 iterations max,
    eval_metric="AUC", early stopping (50 rounds) on val, best model kept.

    params supplies depth / learning_rate / l2_leaf_reg. Deliberately never
    touches the test split. Returns (model, val_auc).
    """
    # TODO (§8)   the tuning protocol's fit: 800 iterations, AUC-watched
    #
    # Like model 4 but iterations=800, eval_metric="AUC" (early stopping
    # now monitors the thing we optimise, not Logloss), plus **params for
    # depth / learning_rate / l2_leaf_reg. Unpack ONLY train and val from
    # parts -- this function must never touch the test split.
    #
    # return: (model, val_auc)
    raise NotImplementedError('fit_catboost_eval_auc')


def make_optuna_objective(parts, categorical):
    """Build the Optuna objective for the CatBoost search.

    Returns a function objective(trial) -> val AUC that samples depth in
    [4, 8], learning_rate log-uniform in [0.02, 0.3] and l2_leaf_reg
    log-uniform in [1, 30], then fits via fit_catboost_eval_auc.
    """
    # TODO (§8)   build and return objective(trial) -> val AUC
    #
    # Inside the closure, sample the reference search space:
    #   depth          trial.suggest_int("depth", 4, 8)
    #   learning_rate  trial.suggest_float(..., 0.02, 0.3, log=True)
    #   l2_leaf_reg    trial.suggest_float(..., 1.0, 30.0, log=True)
    # then fit via fit_catboost_eval_auc and return its val AUC.
    #
    # The objective returns VAL AUC -- never touch test inside it. Any
    # split you optimise against stops measuring generalisation; test
    # buys its meaning by being spent once, in the refit cell below.
    #
    # return: the objective function (a closure over parts / categorical)
    raise NotImplementedError('make_optuna_objective')


optuna.logging.set_verbosity(optuna.logging.WARNING)  # 30 trials, not 30 pages

t0 = time.time()
study = optuna.create_study(
    direction="maximize",
    sampler=optuna.samplers.TPESampler(seed=RANDOM_SEED))
study.optimize(make_optuna_objective(parts, CATEGORICAL), n_trials=30)
elapsed = time.time() - t0

print(f"30 trials in {elapsed / 60:.1f} min")
print(f"best val AUC {study.best_value:.4f} at trial {study.best_trial.number}")
for k, v in study.best_params.items():
    print(f"  {k:14s} {v:.4f}" if isinstance(v, float) else f"  {k:14s} {v}")

# %%
tdf = study.trials_dataframe()[
    ["number", "value", "params_depth", "params_learning_rate",
     "params_l2_leaf_reg"]]
print("top 5 trials by val AUC:")
print(tdf.sort_values("value", ascending=False).head(5).round(4).to_string(index=False))

# %% [markdown]
# ### Refit the winner, spend the test set once

# %%
best_model, best_val = fit_catboost_eval_auc(parts, CATEGORICAL,
                                             study.best_params)
tuned_test = roc_auc_score(yte, best_model.predict_proba(Xte)[:, 1])
ladder["catboost_tuned"] = {
    "model": best_model,
    "val_auc": float(study.best_value),
    "test_auc": float(tuned_test),
    "best_iteration": int(best_model.get_best_iteration()),
    "best_params": dict(study.best_params),
    "n_features": Xtr.shape[1],
}
tuned = ladder["catboost_tuned"]
print(f"model 5 — CatBoost tuned (30 TPE trials)")
print(f"  val  AUC {tuned['val_auc']:.4f}")
print(f"  test AUC {tuned['test_auc']:.4f}")
print(f"  best iteration {tuned['best_iteration']} of 800")
print(f"  gain over defaults: "
      f"{tuned['test_auc'] - ladder['catboost_default']['test_auc']:+.4f}")

# %% [markdown]
# Test AUC **0.847**, a gain of **+0.003** over defaults. Modest — and that is
# the honest, general lesson: on clean tabular data of this size, CatBoost's
# defaults are close to saturated, and tuning buys tenths of a point, not
# whole points (recall the ladder's *big* gaps came from features and model
# class). Where did even this gain come from? Look at the winning
# configuration: a lower learning rate (0.06 vs the default's auto 0.087) run
# for ~780 AUC-monitored iterations, versus the default's Logloss-monitored
# stop at 319. Slower, longer, watched by the right metric — that harvests a
# long tail of small real signal the defaults leave on the table.

# %%
fig, axes = plt.subplots(1, 2, figsize=(11.6, 4.2),
                         gridspec_kw={"width_ratios": [1.6, 1.0]})
ax = axes[0]
values = [t.value for t in study.trials]
ax.scatter(range(len(values)), values, s=22, color=C["blue"], alpha=0.8,
           linewidths=0)
ax.plot(range(len(values)), np.maximum.accumulate(values), color=C["orange"],
        linewidth=1.8, drawstyle="steps-post")
ax.annotate("best so far", xy=(len(values) - 1, study.best_value),
            xytext=(-6, -14), textcoords="offset points", ha="right",
            fontsize=9, color=C["orange"], fontweight="semibold")
ax.axhline(ladder["catboost_default"]["val_auc"], color=INK["muted"],
           linewidth=1.1, linestyle="--")
ax.annotate(f"defaults {ladder['catboost_default']['val_auc']:.4f}",
            xy=(0, ladder["catboost_default"]["val_auc"]), xytext=(2, -11),
            textcoords="offset points", fontsize=8.5, color=INK["muted"])
finish(ax, "optimisation history", xlabel="trial", ylabel="val AUC")

ax = axes[1]
importances = optuna.importance.get_param_importances(
    study, evaluator=optuna.importance.FanovaImportanceEvaluator(
        seed=RANDOM_SEED))
keys = list(importances)[::-1]
vals = [importances[k] for k in keys]
ax.barh(np.arange(len(keys)), vals, height=0.5, color=C["blue"])
ax.set_yticks(np.arange(len(keys)))
ax.set_yticklabels(keys, fontsize=9)
for i, v in enumerate(vals):
    ax.annotate(f"{v:.2f}", xy=(v, i), xytext=(4, 0),
                textcoords="offset points", va="center", fontsize=8.5,
                color=INK["secondary"])
finish(ax, "parameter importance (fANOVA)", xlabel="share of val-AUC variance")
fig.suptitle("The Optuna study: 30 seeded TPE trials",
             x=0.005, ha="left", fontsize=12, fontweight="semibold",
             color=INK["primary"])
fig.tight_layout(rect=(0, 0, 1, 0.94))
fig.savefig(FIG_DIR / "08_optuna_study.png", bbox_inches="tight", dpi=140)
plt.show()

# %% [markdown]
# ### Never tune on test
#
# The protocol was: train fits, **val chooses** (early stopping *and*
# hyperparameters — 30 times over), test is spent exactly once, at the end.
# Val's number is now contaminated by all that choosing — it is the maximum
# of 30 noisy draws, so it flatters:

# %%
opt_gap = tuned["val_auc"] - tuned["test_auc"]
print(f"tuned model: val AUC {tuned['val_auc']:.4f} vs test AUC "
      f"{tuned['test_auc']:.4f}")
print(f"optimism of the split that did the choosing: {opt_gap:+.4f}")

# %% [markdown]
# That ~0.01 is with an honest protocol — val paid for its role and test
# stayed clean. If we had let the objective *see test*, the reported number
# would carry that optimism invisibly, and nothing in the code would look
# wrong. Selection bias does not announce itself: any set you optimise
# against — by tuning, by early stopping, or just by re-running until you
# like the number — silently stops measuring generalisation. The test set
# buys its meaning by being spent once.
#
# ## §9. The ladder, and when a line is still the right answer
#
# Every rung, side by side — and checked against the frozen reference
# (`data/reference_results.json`), so this notebook cannot silently drift:

# %%
LABELS = {
    "logreg_numeric": "1 logreg, numeric only",
    "logreg_ohe": "2 logreg + OHE",
    "tree": "3 decision tree (d6)",
    "catboost_default": "4 CatBoost defaults",
    "catboost_tuned": "5 CatBoost + Optuna",
}
rows, prev = [], None
for key, label in LABELS.items():
    r = ladder[key]
    rows.append({"model": label, "features": r["n_features"],
                 "val AUC": round(r["val_auc"], 4),
                 "test AUC": round(r["test_auc"], 4),
                 "vs prev": "" if prev is None
                 else f"{r['test_auc'] - prev:+.4f}"})
    prev = r["test_auc"]
print(pd.DataFrame(rows).set_index("model").to_string())

# %%
REF = json.loads((DATA_DIR / "reference_results.json").read_text())

print(f"{'model':18s}{'ours':>10}{'reference':>11}{'diff':>10}")
for key in LABELS:
    ours, ref = ladder[key]["test_auc"], REF["ladder"][key]["test_auc"]
    print(f"{key:18s}{ours:10.4f}{ref:11.4f}{ours - ref:10.6f}")
    assert abs(ours - ref) < 0.005, f"{key} test AUC drifted from reference"
    assert abs(ladder[key]["val_auc"] - REF["ladder"][key]["val_auc"]) < 0.005
    if "n_features" in REF["ladder"][key]:
        assert ladder[key]["n_features"] == REF["ladder"][key]["n_features"]

# the ladder's ordering and margins -- the whole point of the lesson
a1, a2, a3, a4, a5 = (ladder[k]["test_auc"] for k in LABELS)
assert a1 < a2 < a4 < a5 and a1 < a3 < a4, "ladder ordering broke"
assert abs((a2 - a1) - REF["gaps"]["numeric_to_ohe"]) < 0.005
assert abs((a4 - a2) - REF["gaps"]["ohe_to_catboost"]) < 0.005
assert abs((a5 - a4) - REF["gaps"]["default_to_tuned"]) < 0.005

# the tuning result
ref_t = REF["ladder"]["catboost_tuned"]
assert tuned["best_params"]["depth"] == ref_t["best_params"]["depth"]
for p in ("learning_rate", "l2_leaf_reg"):
    rel = abs(tuned["best_params"][p] - ref_t["best_params"][p]) \
        / ref_t["best_params"][p]
    assert rel < 0.25, f"optuna best {p} drifted from reference"
assert abs(ladder["catboost_default"]["best_iteration"]
           - REF["ladder"]["catboost_default"]["best_iteration"]) <= 100
assert abs(tuned["best_iteration"] - ref_t["best_iteration"]) <= 100

# the regression story
ref_l = REF["lasso_support_recovery"]
assert len(recovering) == ref_l["n_alphas_recovering"]
assert np.isclose(min(recovering), ref_l["recovering_alpha_min"])
assert np.isclose(max(recovering), ref_l["recovering_alpha_max"])
assert np.isclose(lasso_cv.alpha_, ref_l["lasso_cv_alpha"])
assert sorted(cv_support) == sorted(ref_l["lasso_cv_support"])
assert np.allclose(lasso_example.coef_,
                   [ref_l["example_coefficients"][f] for f in NUMERIC],
                   atol=0.05)
assert abs(example_r2 - ref_l["test_r2_at_example_alpha"]) < 0.01
assert np.allclose(ridge_gaps, REF["ridge_collinearity"]["abs_gap_path"],
                   atol=0.05)
assert abs(df["incident"].mean() - REF["incident_rate"]) < 1e-6

print("\nall reference checks passed — this notebook matches "
      "reference_results.json")

# %% [markdown]
# ### Reading the ladder
#
# * **+0.049** for the categoricals (1→2): structure you refuse to encode is
#   signal you refuse to use.
# * **+0.076** for boosting + native categoricals (2→4): the single biggest
#   step — nonlinearities, raw-scale robustness, and a usable `cluster_id`.
# * **+0.003** for tuning (4→5): real, reproducible, and *small*. Model class
#   and features move AUC by points; hyperparameters, on data like this, by
#   tenths of points. Budget your effort in that order.
# * The tree (0.737) sits between the linear models — nonlinear but weak
#   alone. Its job in practice is to be the unit inside an ensemble.
#
# And a ceiling for honesty: the generator knows its own true incident
# probabilities, and even *they* only score AUC ≈ 0.89 on this sample — much
# of the remaining gap is cluster-level truth that 8-row clusters simply
# cannot reveal. CatBoost at 0.847 is closer to the ceiling than to the
# best linear model.

# %%
print(f"Bayes ceiling (true probabilities, from the generator): "
      f"AUC {META['bayes_auc']:.3f}")
print(f"best model in this lesson:                              "
      f"AUC {a5:.3f}")
print(f"best linear model in this lesson:                       "
      f"AUC {a2:.3f}")

# %% [markdown]
# ### When are linear models still the right answer?
#
# After a chapter where gradient boosting wins by eight points, it would be
# easy to conclude "always boost". Wrong conclusion. Reach for the linear
# model when:
#
# * **You must explain, not just rank.** Model 2's "canary multiplies incident
#   odds by 2.7, adjusted for workload and region" survives a postmortem
#   review; feature importances do not — they say *used*, not *why*, and they
#   silently absorb interactions.
# * **The decision consumes probabilities, not rankings.** A monotone,
#   additive log-odds model is well-calibrated almost for free; boosted
#   ensembles usually need a calibration step before their probabilities mean
#   anything.
# * **Data is small.** Our 8-row clusters were noise even for CatBoost; with
#   hundreds of rows and dozens of features, a regularised linear model is
#   frequently the *accuracy* winner too, and §2's coefficient machinery tells
#   you exactly what it believes.
# * **You extrapolate.** Trees predict a constant outside the training range —
#   power draw at a cpu_util no node has reached yet is a question only a
#   parametric model can even attempt.
# * **Latency and simplicity bind.** `w @ x + b` deploys anywhere, audits
#   trivially, and never surprises you.
#
# And in every case, the numeric-only baseline remains the first model you
# build — not because it wins, but because every gap in the ladder is measured
# from it. A 0.847 means nothing until you know the floor was 0.719.

# %%
print("checks that ran in this notebook:")
for line in [
    "unseen clusters in val/test          -> exactly the 6 the data card promises",
    "lasso on standardised features       -> recovers the exact true support",
    "lasso on raw features                -> gets the support wrong (units!)",
    "ridge on the collinear pair          -> gap shrinks monotonically",
    "depth-3 tree root split              -> the generator's 14.1 ms latency threshold",
    "boost_by_hand                        -> train MSE falls every stage",
    "OHE of an unseen cluster             -> encodes to an all-zero block",
    "full 5-model ladder                  -> matches reference_results.json within 0.005",
    "optuna best params                   -> match the reference study",
]:
    print(f"  [ok] {line}")
