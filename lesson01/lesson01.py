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
#     display_name: DS Course (lesson01)
#     language: python
#     name: ds-course
# ---

# %% [markdown]
# # Lesson 1 — From a table to a defensible decision
#
# ## What this lesson is about
#
# Almost every applied data-science request, once you strip the wrapping off,
# is one of two questions:
#
# > **Q1. Did something change?**
# > The deploy went out on Tuesday. Incidents look worse. Are they *actually*
# > worse, or is this the same noise we see every week?
#
# > **Q2. Where are the anomalies?**
# > Twenty thousand telemetry windows. Which handful deserves a human's
# > attention?
#
# Both questions have the same uncomfortable property: **the answer is a number
# you cannot observe directly.** You observe *one* sample. The incident rate you
# measure is not the incident rate — it is one draw from a distribution of
# incident rates you could have measured. So every honest answer in this lesson
# comes with an interval attached, and a large part of the lesson is the
# machinery for producing that interval.
#
# The path we take:
#
# | § | Topic |
# |---|-------|
# | 1 | The two questions, and why a point estimate is never an answer |
# | 2 | The data: binary classification, distributions, covariation |
# | 3 | Linear regression, the normal equation, and why we don't use it |
# | 4 | Gradient descent |
# | 5 | Two models: plain features vs cross-features |
# | 6 | Thresholds. Accuracy, precision, recall — by hand, then by library |
# | 7 | F1 and ROC-AUC |
# | 8 | A test-set score is a random variable. The bootstrap |
# | 9 | Bootstrap by hand, then by library |
# | 10 | DeLong's test: is model B *really* better than model A? |
# | 11 | Anomalies from the residual distribution — and where that idea breaks |
#
# ### A note on honesty
#
# This notebook uses a **synthetic** dataset. That is a deliberate choice, not a
# shortcut: we need to know the ground truth (which rows are genuinely
# anomalous, what the true coefficients are, what the achievable ceiling is) in
# order to *check* that our methods work. On a real dataset every method in
# section 11 would be unfalsifiable. Here we can score it.
#
# The generator is `src/generate_dataset.py` and is worth reading after the
# lesson — but not before.

# %% [markdown]
# ## §0. Setup
#
# Everything is seeded. Re-running this notebook top to bottom reproduces every
# number in it.

# %%
from __future__ import annotations

import itertools
import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from scipy import stats

RANDOM_SEED = 20260729
rng = np.random.default_rng(RANDOM_SEED)

BASE_DIR = Path.cwd()
if not (BASE_DIR / "data").exists() and (BASE_DIR / "lesson01" / "data").exists():
    BASE_DIR = BASE_DIR / "lesson01"

DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "models"
FIG_DIR = BASE_DIR / "figures"
MODEL_DIR.mkdir(exist_ok=True)
FIG_DIR.mkdir(exist_ok=True)

pd.set_option("display.width", 110)
pd.set_option("display.max_columns", 40)
pd.set_option("display.float_format", lambda v: f"{v:,.4f}")

print("data dir:", DATA_DIR)

# %% [markdown]
# ### Plot styling
#
# One place to define colours, so every chart in the notebook reads as one
# system. The categorical hues are used **in a fixed order** — series 1 is
# always blue, series 2 always orange — so that "blue = model A" holds from the
# first chart to the last. We never use more than three categorical colours at
# once, which is the documented all-pairs colour-blind-safe limit for this
# palette.

# %%
C = {
    "blue": "#2a78d6",     # series 1
    "orange": "#eb6834",   # series 2
    "aqua": "#1baf7a",     # series 3
    "red": "#e34948",
    "violet": "#4a3aa7",
}
INK = {
    "primary": "#0b0b0b",
    "secondary": "#52514e",
    "muted": "#898781",
    "grid": "#e1e0d9",
    "axis": "#c3c2b7",
    "surface": "#fcfcfb",
}

# Diverging map for correlations: blue = negative, neutral grey = zero,
# red = positive. Never a rainbow, and never a hue at the midpoint.
CORR_CMAP = LinearSegmentedColormap.from_list(
    "blue_grey_red",
    ["#0d366b", "#1c5cab", "#2a78d6", "#86b6ef", "#cde2fb",
     "#f0efec",
     "#f6c9c9", "#ef8f8f", "#e34948", "#b32b2b", "#7a1616"],
)

plt.rcParams.update({
    "figure.facecolor": INK["surface"],
    "axes.facecolor": INK["surface"],
    "savefig.facecolor": INK["surface"],
    "axes.edgecolor": INK["axis"],
    "axes.labelcolor": INK["secondary"],
    "axes.titlecolor": INK["primary"],
    "axes.titlesize": 11,
    "axes.titleweight": "semibold",
    "axes.titlelocation": "left",
    "axes.labelsize": 9,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.color": INK["grid"],
    "grid.linewidth": 0.8,
    "grid.linestyle": "-",       # solid hairlines; dashes read as "threshold"
    "xtick.color": INK["muted"],
    "ytick.color": INK["muted"],
    "xtick.labelsize": 8.5,
    "ytick.labelsize": 8.5,
    "legend.frameon": False,
    "legend.fontsize": 9,
    "lines.linewidth": 2.0,
    "lines.markersize": 5,
    "font.size": 10,
    "figure.dpi": 110,
})


def finish(ax, title=None, xlabel=None, ylabel=None, legend=False):
    """Apply the recessive-chrome conventions to an axis."""
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


def save_fig(fig, name):
    fig.savefig(FIG_DIR / f"{name}.png", bbox_inches="tight", dpi=140)


# %% [markdown]
# ## §1. The two questions
#
# ### The scenario
#
# We run an API on 8 nodes. Every 5 minutes each node emits a telemetry window:
# CPU, memory, disk latency, request rate, error rate, queue depth, cache hit
# ratio, temperature. A window is labelled `incident = 1` if it breached the
# SLA.
#
# Partway through the observation period, **a new version was deployed.** Rows
# are tagged `pre_deploy` / `post_deploy`.
#
# Let us ask Q1 in its crudest form.

# %%
df = pd.read_csv(DATA_DIR / "service_telemetry.csv", parse_dates=["window_start"])
print(f"{len(df):,} rows x {df.shape[1]} columns")
df.head()

# %%
rate_by_phase = df.groupby("phase")["incident"].agg(["size", "sum", "mean"])
rate_by_phase.columns = ["n_windows", "n_incidents", "incident_rate"]
rate_by_phase = rate_by_phase.loc[["pre_deploy", "post_deploy"]]
rate_by_phase

# %% [markdown]
# The incident rate went **up**. So: ship a rollback?
#
# Not yet. That difference is about three percentage points. Before acting we
# need to know whether a difference that size is surprising, or whether two
# halves of an *unchanged* process routinely differ by that much.
#
# ### The answer requires a null model
#
# The reasoning is: *if nothing had changed*, how often would we see a gap this
# large by chance alone? That is a **two-proportion z-test**. Under the null,
# both phases share one rate $p$, estimated by pooling:
#
# $$\hat p = \frac{x_1 + x_2}{n_1 + n_2}, \qquad
#   \mathrm{SE} = \sqrt{\hat p (1-\hat p)\left(\tfrac{1}{n_1}+\tfrac{1}{n_2}\right)},
#   \qquad z = \frac{\hat p_2 - \hat p_1}{\mathrm{SE}}$$
#
# We implement it, then check it against a library.

# %%
def two_proportion_test(x1, n1, x2, n2):
    """Compare two binomial rates. Returns (diff, z, p_value, se)."""
    p1, p2 = x1 / n1, x2 / n2
    p_pool = (x1 + x2) / (n1 + n2)
    se = np.sqrt(p_pool * (1 - p_pool) * (1 / n1 + 1 / n2))
    z = (p2 - p1) / se
    p_value = 2 * stats.norm.sf(abs(z))
    return p2 - p1, z, p_value, se


pre = df[df.phase == "pre_deploy"]["incident"]
post = df[df.phase == "post_deploy"]["incident"]

diff, z_stat, p_value, se = two_proportion_test(
    pre.sum(), len(pre), post.sum(), len(post)
)
print(f"pre-deploy  rate : {pre.mean():.4f}  (n = {len(pre):,})")
print(f"post-deploy rate : {post.mean():.4f}  (n = {len(post):,})")
print(f"difference       : {diff:+.4f}  ({diff * 100:+.2f} percentage points)")
print(f"standard error   : {se:.4f}")
print(f"z                : {z_stat:.3f}")
print(f"p-value          : {p_value:.3e}")

# A 2x2 chi-square is the same test; chi2 should equal z^2.
table = np.array([[pre.sum(), len(pre) - pre.sum()],
                  [post.sum(), len(post) - post.sum()]])
chi2, chi2_p, _, _ = stats.chi2_contingency(table, correction=False)
print(f"\ncross-check: chi2 = {chi2:.4f}  vs  z^2 = {z_stat ** 2:.4f}")
print(f"            chi2 p = {chi2_p:.3e}  vs  z p = {p_value:.3e}")
assert np.isclose(chi2, z_stat ** 2), "chi-square and z-test must agree"

# %% [markdown]
# ### The control that makes this trustworthy
#
# A small p-value is easy to produce and easy to fool yourself with. The
# discipline that separates a result from a coincidence is running the *same
# procedure* on a comparison where you know the answer must be "no change".
#
# So: split the **pre-deploy period alone** into two halves. Nothing happened
# between them. If our test flags *that* as significant, the test is broken (or
# the data has structure we have not modelled, like drift or autocorrelation).
# This is often called an A/A test.

# %%
pre_rows = df[df.phase == "pre_deploy"].sort_values("window_start")
half = len(pre_rows) // 2
a, b = pre_rows.iloc[:half]["incident"], pre_rows.iloc[half:]["incident"]

pdiff, pz, pp, _ = two_proportion_test(a.sum(), len(a), b.sum(), len(b))
print("PLACEBO (pre-deploy first half vs second half — nothing changed here)")
print(f"  difference : {pdiff:+.4f}  ({pdiff * 100:+.2f} pp)")
print(f"  z          : {pz:.3f}")
print(f"  p-value    : {pp:.3f}")
print()
print("REAL COMPARISON (pre vs post deploy)")
print(f"  difference : {diff:+.4f}  ({diff * 100:+.2f} pp)")
print(f"  z          : {z_stat:.3f}")
print(f"  p-value    : {p_value:.3e}")

# %% [markdown]
# The placebo comparison comes back null and the real one does not. *Now* the
# result means something.
#
# Let us draw it. A rate with no interval on it is not a finding, so the bars
# carry 95% confidence intervals — the interval is the point of the chart.

# %%
def wilson_interval(x, n, conf=0.95):
    """Wilson score interval for a binomial proportion.

    Preferred over the textbook normal interval, which misbehaves when p is
    near 0 or 1 and can produce bounds outside [0, 1].
    """
    zc = stats.norm.ppf(1 - (1 - conf) / 2)
    p = x / n
    denom = 1 + zc ** 2 / n
    centre = (p + zc ** 2 / (2 * n)) / denom
    halfwidth = zc * np.sqrt(p * (1 - p) / n + zc ** 2 / (4 * n ** 2)) / denom
    return centre - halfwidth, centre + halfwidth


groups = [
    ("pre-deploy\n(first half)", a.sum(), len(a), C["blue"]),
    ("pre-deploy\n(second half)", b.sum(), len(b), C["blue"]),
    ("post-deploy", post.sum(), len(post), C["orange"]),
]

fig, ax = plt.subplots(figsize=(7.0, 4.0))
for i, (label, x, n, colour) in enumerate(groups):
    lo, hi = wilson_interval(x, n)
    rate = x / n
    ax.plot([i, i], [lo, hi], color=colour, linewidth=2, solid_capstyle="round")
    ax.plot([i], [rate], "o", color=colour, markersize=9,
            markeredgecolor=INK["surface"], markeredgewidth=2)
    ax.annotate(f"{rate:.3f}", (i, hi), textcoords="offset points",
                xytext=(0, 9), ha="center", fontsize=9, color=INK["primary"])

ax.set_xticks(range(len(groups)))
ax.set_xticklabels([g[0] for g in groups])
ax.set_xlim(-0.5, len(groups) - 0.5)
finish(ax, "Incident rate with 95% confidence intervals",
       ylabel="incident rate")
ax.annotate("the interval is the finding —\nnot the dot",
            xy=(2, post.mean()), xytext=(0.55, post.mean() + 0.021), fontsize=9,
            color=INK["secondary"],
            arrowprops=dict(arrowstyle="-", color=INK["axis"], linewidth=1))
save_fig(fig, "01_incident_rate_by_phase")
plt.show()

# %% [markdown]
# One warning about reading that chart, which we will come back to properly in
# §10: **do not decide significance by checking whether the error bars overlap.**
# The intervals are there to show you the precision of each estimate. The
# question "is the *difference* real?" needs the distribution of the difference,
# which is what the test computed — and the two are not the same question.
#
# **Q1 is answered: yes, something changed.** The rest of the lesson is
# largely about Q2 — *what* changed and *which rows* are responsible — and
# about doing the "is this real?" move again in harder settings, where no
# textbook formula exists and we will have to build the interval ourselves.

# %% [markdown]
# ## §2. Binary classification: the data, its distributions, its covariation
#
# ### What binary classification is
#
# We have $n$ objects. Each carries a feature vector $x_i \in \mathbb{R}^d$ and
# a label $y_i \in \{0, 1\}$. We want a function $f$ that, shown a new $x$,
# produces a **score** $s = f(x) \in \mathbb{R}$ — higher meaning "more likely
# to be a 1".
#
# Note what the model does *not* do: it does not output a decision. Turning a
# score into a yes/no requires a **threshold**, and choosing that threshold is a
# separate, business-driven act (§6). Keeping "ranking" and "deciding" apart is
# one of the most useful habits in applied ML.

# %%
RAW_FEATURES = [
    "cpu_util", "mem_pressure", "disk_latency_ms", "request_rate_rps",
    "error_rate_pct", "queue_depth", "cache_hit_ratio", "temperature_c",
]
TARGET = "incident"

print(df[RAW_FEATURES].describe().T[["mean", "std", "min", "50%", "max"]])
print(f"\nmissing values: {int(df[RAW_FEATURES].isna().sum().sum())}")
print(f"class balance : {df[TARGET].mean():.4f} positive "
      f"({df[TARGET].sum():,} of {len(df):,})")
print(f"splits        : {df['split'].value_counts().to_dict()}")

# %% [markdown]
# 24.7% positives. Worth writing down now, because in §6 it is the number that
# makes accuracy hard to read: a model that predicts "never an incident" is
# already **75.3% accurate**.
#
# ### Distributions, one feature at a time
#
# Before any modelling, look at every feature's marginal distribution. You are
# looking for: the shape, the range, spikes, and tails.

# %%
fig, axes = plt.subplots(2, 4, figsize=(15, 6.4))
for ax, col in zip(axes.ravel(), RAW_FEATURES):
    ax.hist(df[col], bins=60, color=C["blue"], alpha=0.85, edgecolor="none")
    finish(ax, col, ylabel="windows")
    ax.tick_params(labelleft=False)
    sk = stats.skew(df[col])
    ax.annotate(f"skew {sk:+.2f}", xy=(0.97, 0.93), xycoords="axes fraction",
                ha="right", fontsize=8.5, color=INK["secondary"])
fig.suptitle("Marginal distribution of each feature", x=0.077, y=1.0,
             ha="left", fontsize=12, fontweight="semibold")
fig.tight_layout()
save_fig(fig, "02_feature_distributions")
plt.show()

# %% [markdown]
# Six of the eight deserve comment, and no two are the same shape:
#
# * **`disk_latency_ms`** — skew ≈ +14. A log-normal latency tail. The
#   histogram is useless as drawn: everything is jammed into the first bin
#   because a few windows are 100× the median.
# * **`request_rate_rps`** — **bimodal**. Two humps, because night traffic and
#   day traffic are two different regimes. A single mean is meaningless here;
#   "average traffic" describes no actual moment.
# * **`error_rate_pct`** — a spike at zero with a long tail. Most windows have
#   no errors at all.
# * **`cache_hit_ratio`** — bounded and *left*-skewed, piled up near 1.
# * **`queue_depth`** — discrete counts, so the histogram is a picket fence.
# * **`temperature_c`** — the one clean symmetric bell, useful as a contrast.
#
# ### The tail problem, and the fix
#
# A feature spanning three orders of magnitude will dominate any distance- or
# gradient-based method purely through its scale, and a linear model will be
# dragged around by a handful of extreme rows. The standard move is to model
# the **logarithm** of a positive, heavy-tailed quantity.

# %%
fig, axes = plt.subplots(1, 2, figsize=(11, 3.6))
axes[0].hist(df["disk_latency_ms"], bins=80, color=C["blue"], edgecolor="none")
finish(axes[0], "disk_latency_ms (raw)", xlabel="ms", ylabel="windows")
axes[1].hist(np.log(df["disk_latency_ms"]), bins=80, color=C["blue"], edgecolor="none")
finish(axes[1], "log(disk_latency_ms)", xlabel="log ms", ylabel="windows")
fig.tight_layout()
save_fig(fig, "03_log_transform")
plt.show()

print(f"raw  skew: {stats.skew(df['disk_latency_ms']):+.2f}")
print(f"log  skew: {stats.skew(np.log(df['disk_latency_ms'])):+.2f}")

# %% [markdown]
# The log turns an unusable spike into a workable, roughly bell-shaped
# distribution. A handful of extreme readings still survive on the right, and
# they are the reason the log skew is not closer to zero — we will meet them
# again in §11, because they are not noise, they are broken sensors.
#
# We apply the same reasoning to the other two positive skewed features and fix
# the **modelling representation** used for the rest of the notebook.

# %%
def build_model_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Raw telemetry columns -> the representation we actually model on.

    Only the heavy-tailed positive quantities are transformed. `log1p` rather
    than `log` for the error rate, because it is exactly zero for many windows.
    """
    out = pd.DataFrame(index=frame.index)
    out["cpu_util"] = frame["cpu_util"]
    out["mem_pressure"] = frame["mem_pressure"]
    out["log_disk_latency"] = np.log(frame["disk_latency_ms"])
    out["log_request_rate"] = np.log(frame["request_rate_rps"])
    out["log1p_error_rate"] = np.log1p(frame["error_rate_pct"])
    out["queue_depth"] = frame["queue_depth"].astype(float)
    out["cache_hit_ratio"] = frame["cache_hit_ratio"]
    out["temperature_c"] = frame["temperature_c"]
    return out


FEATURES = ["cpu_util", "mem_pressure", "log_disk_latency", "log_request_rate",
            "log1p_error_rate", "queue_depth", "cache_hit_ratio", "temperature_c"]

X_all = build_model_features(df)
y_all = df[TARGET].to_numpy()
assert list(X_all.columns) == FEATURES
X_all.head()

# %% [markdown]
# ### Split first, then explore
#
# From here on, everything that *informs a decision* — which features look
# useful, what transform to apply, which model to pick, where to put the
# threshold — is computed on the **training split only**.
#
# The reason is not ceremony. Any choice you make after looking at the test set
# leaks information from it into your model, and the test score stops being an
# estimate of performance on new data. That includes choices you make with your
# eyes: "feature 7 looks useless, drop it" is a fitted parameter if you looked at
# the test set to decide.
#
# The splits are materialised in the CSV, so they are identical for everyone.

# %%
train_mask = (df["split"] == "train").to_numpy()
val_mask = (df["split"] == "val").to_numpy()
test_mask = (df["split"] == "test").to_numpy()

X_train_frame, y_train = X_all[train_mask], y_all[train_mask]
X_val_frame, y_val = X_all[val_mask], y_all[val_mask]
X_test_frame, y_test = X_all[test_mask], y_all[test_mask]

print(f"train {len(X_train_frame):,}   val {len(X_val_frame):,}   "
      f"test {len(X_test_frame):,}")
print(f"positive rate — train {y_train.mean():.4f}  val {y_val.mean():.4f}  "
      f"test {y_test.mean():.4f}   (stratified, so these match)")

# %% [markdown]
# ### Distributions *conditional on the label*
#
# The marginal shape tells you about the feature. What tells you about
# **usefulness** is how much the distribution moves when you condition on the
# target. A feature whose two conditional densities sit on top of each other
# carries no signal.

# %%
fig, axes = plt.subplots(2, 4, figsize=(15, 6.4))
for ax, col in zip(axes.ravel(), FEATURES):
    v0 = X_train_frame.loc[y_train == 0, col]
    v1 = X_train_frame.loc[y_train == 1, col]
    bins = np.histogram_bin_edges(X_train_frame[col], bins=50)
    # Thin outlined steps, not heavy blocks: two overlapping filled histograms
    # at this size read as mud.
    for v, colour, label in ((v0, C["blue"], "no incident"),
                             (v1, C["orange"], "incident")):
        ax.hist(v, bins=bins, density=True, histtype="step", linewidth=1.8,
                color=colour, label=label)
        ax.hist(v, bins=bins, density=True, color=colour, alpha=0.10)
    finish(ax, col, ylabel="density")
    ax.tick_params(labelleft=False)
handles, labels = axes[0, 0].get_legend_handles_labels()
fig.legend(handles, labels, loc="upper right", bbox_to_anchor=(0.995, 1.045), ncol=2)
fig.suptitle("Feature distribution by class (densities, so the 3:1 class "
             "imbalance does not hide the orange)", x=0.077, y=1.0, ha="left",
             fontsize=12, fontweight="semibold")
fig.tight_layout()
save_fig(fig, "04_distributions_by_class")
plt.show()

# %% [markdown]
# `cpu_util`, `log_request_rate` and `log1p_error_rate` separate visibly.
# `temperature_c` barely moves, and `log_disk_latency` separates far less than
# its dramatic tail might suggest — a reminder that a striking marginal
# distribution and a useful feature are different things.
#
# Note the `density=True`: with 3× as many negatives as positives, raw counts
# would show a big blue histogram and a small orange one, and you would learn
# the class balance instead of the shape. Normalising each class to its own
# density is what makes them comparable.
#
# ### Quantifying it
#
# Eyeballing eight panels does not scale. Two quick numbers per feature:
# the **point-biserial correlation** with the target (a Pearson correlation
# where one variable is binary), and the **AUC of that feature used alone as a
# score** (§7 defines AUC properly; for now, 0.5 = useless, 1.0 = perfect,
# below 0.5 = useful but pointing the other way).

# %%
from sklearn.metrics import roc_auc_score

single = pd.DataFrame({
    "corr_with_target": [np.corrcoef(X_train_frame[c], y_train)[0, 1]
                         for c in FEATURES],
    "auc_alone": [roc_auc_score(y_train, X_train_frame[c]) for c in FEATURES],
}, index=FEATURES)
single["abs_corr"] = single["corr_with_target"].abs()
single.sort_values("abs_corr", ascending=False).drop(columns="abs_corr")

# %% [markdown]
# `cache_hit_ratio` has AUC **below** 0.5 — as expected, a higher cache hit
# ratio means *fewer* incidents. That is a perfectly good feature; the sign is
# information, not a problem.
#
# ### Covariation
#
# Features are not independent, and how they move together matters enormously —
# it is the direct cause of the numerical problem we hit in §3.

# %%
corr = X_train_frame.corr()

fig, ax = plt.subplots(figsize=(7.4, 6.2))
im = ax.imshow(corr, cmap=CORR_CMAP, vmin=-1, vmax=1)
ax.set_xticks(range(len(FEATURES)))
ax.set_xticklabels(FEATURES, rotation=45, ha="right")
ax.set_yticks(range(len(FEATURES)))
ax.set_yticklabels(FEATURES)
ax.grid(False)
for i in range(len(FEATURES)):
    for j in range(len(FEATURES)):
        v = corr.iloc[i, j]
        ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=8,
                color="#ffffff" if abs(v) > 0.55 else INK["secondary"])
cb = fig.colorbar(im, ax=ax, fraction=0.045, pad=0.03)
cb.set_label("Pearson correlation", color=INK["secondary"], fontsize=9)
cb.outline.set_visible(False)
finish(ax, "Feature correlation matrix")
fig.tight_layout()
save_fig(fig, "05_correlation_heatmap")
plt.show()

# %% [markdown]
# The colour map is **diverging**: blue for negative, neutral grey at zero, red
# for positive. That is the right encoding when the value has a meaningful
# midpoint and a sign. (A single-hue ramp here would make −0.6 and +0.05 look
# similar, and a rainbow would invent structure that is not there.) Every cell
# is also printed numerically, so nothing is encoded in colour alone.
#
# The strong pairs are physically sensible: CPU with temperature (0.65), CPU
# with memory (0.55), request rate with queue depth (0.46).
#
# Let us look at one pair directly, because a correlation coefficient is a
# one-number summary of a picture and the picture can be very different.

# %%
fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))
pairs = [("cpu_util", "temperature_c"), ("log_request_rate", "queue_depth"),
         ("temperature_c", "cache_hit_ratio")]
sample = rng.choice(len(X_train_frame), 4000, replace=False)
for ax, (xa, xb) in zip(axes, pairs):
    ax.scatter(X_train_frame[xa].iloc[sample], X_train_frame[xb].iloc[sample],
               s=5, alpha=0.18, color=C["blue"], edgecolors="none")
    r = corr.loc[xa, xb]
    finish(ax, f"r = {r:+.2f}", xlabel=xa, ylabel=xb)
fig.suptitle("Covariation: what a correlation coefficient does and does not say",
             x=0.077, y=1.02, ha="left", fontsize=12, fontweight="semibold")
fig.tight_layout()
save_fig(fig, "06_covariation_scatter")
plt.show()

# %% [markdown]
# Two warnings that will cost you real money if ignored:
#
# 1. **Correlation is not causation.** CPU and temperature correlate at 0.65
#    because both respond to load. Cooling the CPU will not reduce load.
# 2. **Correlated features make the linear-algebra solution unstable.** That is
#    not a philosophical point — it is the subject of the next section.

# %% [markdown]
# ## §3. Linear regression and the normal equation
#
# ### The model
#
# Predict a target as a weighted sum of features:
#
# $$\hat y = Xw, \qquad X \in \mathbb{R}^{n \times d},\ w \in \mathbb{R}^{d}$$
#
# (The intercept is folded in by appending a column of ones.) We choose $w$ to
# minimise the **sum of squared errors**:
#
# $$L(w) = \lVert y - Xw \rVert_2^2$$
#
# ### Deriving the closed-form solution
#
# Expand, using $a^\top b = b^\top a$ for scalars:
#
# $$L(w) = (y - Xw)^\top (y - Xw) = y^\top y - 2 w^\top X^\top y + w^\top X^\top X w$$
#
# Differentiate with respect to $w$, using $\nabla_w (w^\top a) = a$ and
# $\nabla_w (w^\top A w) = 2Aw$ for symmetric $A$ (and $X^\top X$ is symmetric):
#
# $$\nabla_w L(w) = -2 X^\top y + 2 X^\top X w$$
#
# Set the gradient to zero. This gives the **normal equations**:
#
# $$X^\top X w = X^\top y$$
#
# and if $X^\top X$ is invertible:
#
# $$\boxed{\;\hat w = (X^\top X)^{-1} X^\top y\;}$$
#
# This really is a minimum, not a saddle: the Hessian is
# $\nabla^2_w L = 2 X^\top X \succeq 0$ for any $X$ (since
# $v^\top X^\top X v = \lVert Xv\rVert^2 \ge 0$), so $L$ is convex and any
# stationary point is a global minimum.
#
# ### Implementing it

# %%
def fit_normal_equation(X: np.ndarray, y: np.ndarray, fit_intercept: bool = True):
    """Least squares by the normal equations. Returns (weights, intercept).

    Note `np.linalg.solve`, not `np.linalg.inv`: forming the inverse explicitly
    is slower and numerically worse. You almost never want an explicit inverse.
    """
    X = np.asarray(X, dtype=float)
    if fit_intercept:
        X = np.hstack([np.ones((X.shape[0], 1)), X])
    gram = X.T @ X
    rhs = X.T @ y
    beta = np.linalg.solve(gram, rhs)
    return (beta[1:], beta[0]) if fit_intercept else (beta, 0.0)


# Fit on the training split, standardised (we will reuse this scaling later).
# The scaling constants come from the training data only -- computing them over
# all rows would leak the test set's distribution into the model.
X_train_raw = X_train_frame.to_numpy()
feat_mean = X_train_raw.mean(axis=0)
feat_std = X_train_raw.std(axis=0, ddof=0)

X_train = (X_train_raw - feat_mean) / feat_std

w_ne, b_ne = fit_normal_equation(X_train, y_train)

from sklearn.linear_model import LinearRegression

sk = LinearRegression().fit(X_train, y_train)
comparison = pd.DataFrame({
    "feature": FEATURES,
    "ours (normal eq.)": w_ne,
    "sklearn": sk.coef_,
})
print(comparison.to_string(index=False))
print(f"\nintercept  ours = {b_ne:.6f}   sklearn = {sk.intercept_:.6f}")
assert np.allclose(w_ne, sk.coef_, atol=1e-8)
assert np.isclose(b_ne, sk.intercept_, atol=1e-8)
print("\nmatches sklearn to 1e-8")

# %% [markdown]
# ### A caveat we are choosing to live with
#
# We are fitting a *regression* to a 0/1 target. This is called a **linear
# probability model**, and it is not the textbook-correct tool: nothing stops it
# from predicting −0.3 or 1.4, and those are not probabilities.

# %%
train_pred = X_train @ w_ne + b_ne
outside = (train_pred < 0) | (train_pred > 1)
print(f"predictions outside [0, 1]: {outside.sum():,} of {len(train_pred):,} "
      f"({outside.mean():.2%})")
print(f"range: [{train_pred.min():.3f}, {train_pred.max():.3f}]")

# %% [markdown]
# We keep it anyway, for this lesson, for two reasons: the derivation above is
# the cleanest closed form in all of ML, and every metric in §6–§10 needs only
# that the score **ranks** objects correctly, not that it is calibrated. Where
# calibration *would* matter, we will say so. Logistic regression — the right
# tool — is the subject of lesson 2.
#
# ### Why we will not use this formula in practice
#
# #### Problem 1: $X^\top X$ can be singular
#
# $X^\top X$ is invertible if and only if the columns of $X$ are linearly
# independent. Two ways that fails, both routine in real data:
#
# * a feature is a duplicate or an exact linear combination of others (a
#   one-hot encoding that kept all levels *and* an intercept is the classic);
# * $d > n$ — more features than rows makes it singular automatically.

# %%
X_dup = np.hstack([X_train, X_train[:, [0]]])  # duplicate the first column
gram_dup = X_dup.T @ X_dup

print(f"rank of X^T X : {np.linalg.matrix_rank(gram_dup)} "
      f"(needs to be {gram_dup.shape[0]} to be invertible)")
print(f"determinant   : {np.linalg.det(gram_dup):.3e}")

try:
    np.linalg.solve(gram_dup, X_dup.T @ y_train)
except np.linalg.LinAlgError as exc:
    print(f"\nnp.linalg.solve raises: LinAlgError({exc})")

# %% [markdown]
# #### The subtler, more dangerous case: *nearly* singular
#
# An exact duplicate raises an exception, which is the kind thing to do. Real
# data rarely obliges. It gives you features that are *almost* collinear —
# and then the matrix inverts fine, returns enormous coefficients with
# arbitrary signs, and nothing warns you.

# %%
noise_levels = [1e-1, 1e-3, 1e-5, 1e-7]
rows = []
for eps in noise_levels:
    X_near = np.hstack([
        X_train,
        (X_train[:, [0]] + rng.normal(0, eps, size=(len(X_train), 1))),
    ])
    w_near, _ = fit_normal_equation(X_near, y_train)
    rows.append({
        "noise added": eps,
        "cond(X^T X)": np.linalg.cond(X_near.T @ X_near),
        "w[cpu_util]": w_near[0],
        "w[near-copy]": w_near[-1],
        "largest |w|": np.abs(w_near).max(),
    })
near_singular = pd.DataFrame(rows)
print(near_singular.to_string(index=False,
                              formatters={"cond(X^T X)": lambda v: f"{v:.2e}",
                                          "noise added": lambda v: f"{v:.0e}"}))

# %% [markdown]
# As the copy gets closer to the original, the condition number explodes and the
# two coefficients blow up in opposite directions, cancelling each other. The
# *predictions* stay fine; the *coefficients* become garbage. If anyone is
# reading those coefficients as "feature importance" — and someone always is —
# they are reading noise.
#
# A rule of thumb: $\mathrm{cond}(X^\top X) > 10^{10}$ means roughly 10 of your
# ~16 digits of precision are gone.
#
# The fixes: drop redundant columns, or add **regularisation** — ridge
# regression solves $(X^\top X + \lambda I)^{-1} X^\top y$, and that
# $+\lambda I$ makes the matrix invertible by construction. (Lesson 2.)
#
# #### Problem 2: cost
#
# Forming $X^\top X$ costs $O(n d^2)$; solving costs $O(d^3)$. The cubic term is
# what kills you. Let us watch it happen.

# %%
def time_normal_equation(n: int, d: int, repeats: int = 5):
    """Time the two phases separately. Returns (gram_seconds, solve_seconds).

    Measuring them apart matters: the total is O(n d^2 + d^3), and with n fixed
    the first term dominates until d grows large. Timing only the sum would show
    neither exponent cleanly.
    """
    Xt = rng.normal(size=(n, d))
    yt = rng.normal(size=n)
    gram, rhs = Xt.T @ Xt, Xt.T @ yt
    np.linalg.solve(gram, rhs)          # warm up the BLAS thread pool first

    def best_of(fn):
        best = np.inf
        for _ in range(repeats):
            t0 = time.perf_counter()
            fn()
            best = min(best, time.perf_counter() - t0)
        return best

    return best_of(lambda: Xt.T @ Xt), best_of(lambda: np.linalg.solve(gram, rhs))


dims = [200, 400, 800, 1600, 3200]
gram_t, solve_t = zip(*[time_normal_equation(5000, d) for d in dims])

fig, ax = plt.subplots(figsize=(6.8, 4.4))
ax.plot(dims, solve_t, "o-", color=C["blue"], label="solve  —  $O(d^3)$",
        markeredgecolor=INK["surface"], markeredgewidth=1.5)
ax.plot(dims, gram_t, "o-", color=C["orange"], label="form $X^\\top X$  —  $O(nd^2)$",
        markeredgecolor=INK["surface"], markeredgewidth=1.5)
ax.plot(dims, np.array(solve_t[-1]) * (np.array(dims) / dims[-1]) ** 3,
        color=INK["muted"], linewidth=1.3, label="$d^3$ reference")
ax.plot(dims, np.array(gram_t[-1]) * (np.array(dims) / dims[-1]) ** 2,
        color=INK["axis"], linewidth=1.3, label="$d^2$ reference")
ax.set_xscale("log")
ax.set_yscale("log")
finish(ax, "Cost of the normal equations, by phase (n = 5000)",
       xlabel="number of features $d$", ylabel="seconds", legend=True)
save_fig(fig, "07_normal_equation_cost")
plt.show()

print(f"  {'d':>6} {'form X^T X':>12} {'solve':>10}")
for d, g, s in zip(dims, gram_t, solve_t):
    print(f"  {d:6d} {g * 1000:10.1f} ms {s * 1000:8.1f} ms")

slope = lambda v: np.log(v[-1] / v[-2]) / np.log(dims[-1] / dims[-2])
print(f"\n  empirical exponent over the last doubling:")
print(f"    solve      : d^{slope(solve_t):.2f}   (theory: 3)")
print(f"    form X^T X : d^{slope(gram_t):.2f}   (theory: 2, at fixed n)")

# %% [markdown]
# Read that carefully, because the naive version of this demo is misleading.
#
# At $d = 200$ the solve is a rounding error and **forming** $X^\top X$ is
# essentially the whole cost. Only as $d$ approaches $n$ does the cubic term take
# over. If you time the two together at fixed $n$, you measure a blend of $d^2$
# and $d^3$ and see neither — which is why they are separated here.
#
# The exponents printed above are also measured over a single doubling on a
# multi-threaded BLAS, so expect them to be noisy. The cubic is real; it just
# needs $d$ large enough to dominate the constant factors.
#
# Now extrapolate. The cubic term at $d = 10^5$ — an entirely ordinary size once
# you one-hot encode a few high-cardinality columns — costs roughly:

# %%
solve_at_1e5 = solve_t[-1] * (1e5 / dims[-1]) ** 3
print(f"solve alone at d = 1e5 : {solve_at_1e5 / 3600:,.0f} hours")
print(f"X^T X alone would need : {(1e5 ** 2 * 8) / 1e9:,.0f} GB of memory")
print("\n(and this assumes n grows with d — at n = 5000 and d = 1e5 the matrix")
print(" is singular by Problem 1 above, so the method does not merely become")
print(" slow, it stops being defined.)")

# %% [markdown]
# Hours of compute and 80 GB of RAM just to *hold* the matrix — and that is for
# the one model in ML that has a closed form at all. Neural networks, gradient
# boosting, logistic regression: no closed form exists.
#
# So we need a method that never forms $X^\top X$, scales to large $d$, and
# generalises beyond least squares.

# %% [markdown]
# ## §4. How linear regression is actually trained: gradient descent
#
# ### The idea
#
# $L$ is differentiable, and $-\nabla L$ is the direction of steepest local
# decrease. So: start somewhere, repeatedly take a small step downhill.
#
# $$w \leftarrow w - \eta \nabla_w L(w)$$
#
# with $\eta$ the **learning rate**. For the mean squared error
# $L(w, b) = \frac{1}{n}\lVert Xw + b - y\rVert^2$:
#
# $$\nabla_w L = \frac{2}{n} X^\top (Xw + b - y), \qquad
#   \frac{\partial L}{\partial b} = \frac{2}{n}\sum_i (\hat y_i - y_i)$$
#
# Each step costs $O(nd)$ — **linear** in $d$, not cubic — and never
# materialises a $d \times d$ matrix.
#
# ### How large may $\eta$ be?
#
# Not arbitrary. For a quadratic loss, gradient descent converges iff
# $\eta < 2 / L_{\max}$, where $L_{\max}$ is the largest eigenvalue of the
# Hessian $\frac{2}{n}X^\top X$. Above that, steps overshoot and grow. We can
# compute the exact cutoff for our data and then verify it empirically.

# %%
def mse_loss(X, y, w, b):
    resid = X @ w + b - y
    return float(np.mean(resid ** 2))


def fit_gradient_descent(X, y, lr=0.1, n_iters=2000, record_every=1):
    """Full-batch gradient descent on the mean squared error.

    Returns (w, b, history) where history is the loss at each recorded step.
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    n, d = X.shape

    w = np.zeros(d)
    b = 0.0
    history = []

    for it in range(n_iters):
        resid = X @ w + b - y            # (n,)
        grad_w = (2.0 / n) * (X.T @ resid)
        grad_b = (2.0 / n) * resid.sum()
        w -= lr * grad_w
        b -= lr * grad_b
        if it % record_every == 0:
            history.append(float(np.mean(resid ** 2)))
    return w, b, np.array(history)


hessian_max_eig = 2 * np.linalg.eigvalsh(X_train.T @ X_train / len(X_train)).max()
lr_critical = 2 / hessian_max_eig
print(f"largest Hessian eigenvalue : {hessian_max_eig:.4f}")
print(f"=> gradient descent diverges for lr >= {lr_critical:.4f}")

# %%
fig, ax = plt.subplots(figsize=(7.2, 4.4))
loss_star = mse_loss(X_train, y_train, w_ne, b_ne)

for lr, colour, style in [(0.5, C["blue"], "-"),
                          (0.05, C["orange"], "-"),
                          (0.005, C["aqua"], "-")]:
    _, _, hist = fit_gradient_descent(X_train, y_train, lr=lr, n_iters=300)
    ax.plot(hist - loss_star, style, color=colour, label=f"lr = {lr}")
    # aqua sits below 3:1 contrast on this surface, so it is directly labelled
    # rather than relying on the legend swatch alone
    ax.annotate(f"lr = {lr}", xy=(299, hist[-1] - loss_star),
                xytext=(6, 0), textcoords="offset points", fontsize=9,
                color=colour, va="center")

ax.set_yscale("log")
ax.set_xlim(0, 340)
finish(ax, "Convergence to the closed-form optimum",
       xlabel="iteration", ylabel="loss − optimal loss")
save_fig(fig, "08_gd_learning_rates")
plt.show()

# %% [markdown]
# Plotting $L - L^*$ on a log scale (rather than $L$ itself) is what makes the
# behaviour legible: the curves are straight lines, and their slope *is* the
# convergence rate. A plain loss curve would show three lines converging to the
# same flat value and tell you nothing.
#
# Now the divergence claim, tested rather than asserted:

# %%
for lr in [lr_critical * 0.9, lr_critical * 1.01, lr_critical * 1.1]:
    _, _, hist = fit_gradient_descent(X_train, y_train, lr=lr, n_iters=200)
    final = hist[-1]
    verdict = "converged" if np.isfinite(final) and final < 1 else "DIVERGED"
    shown = f"{final:.6f}" if np.isfinite(final) else "inf/nan"
    print(f"  lr = {lr:.4f}  ({lr / lr_critical:.2f} x critical)  "
          f"final loss = {shown:>12s}   {verdict}")

# %% [markdown]
# The theory holds exactly. This is also the practical argument for
# **standardising features**: $L_{\max}$ depends on feature scale, so with one
# feature in the thousands and another in [0,1] the usable learning rate is set
# by the largest, and the smallest-scale feature learns nothing.
#
# ### Does gradient descent find the same answer as the closed form?
#
# It must — the problem is convex with a unique minimum (given full rank).
# Verify rather than trust:

# %%
w_gd, b_gd, hist_gd = fit_gradient_descent(X_train, y_train, lr=0.3, n_iters=5000)

check = pd.DataFrame({
    "feature": FEATURES,
    "normal equation": w_ne,
    "gradient descent": w_gd,
    "abs diff": np.abs(w_ne - w_gd),
})
print(check.to_string(index=False))
print(f"\nintercept: closed form {b_ne:.8f}   GD {b_gd:.8f}")
print(f"max |w difference| : {np.abs(w_ne - w_gd).max():.3e}")
print(f"loss  closed form  : {loss_star:.10f}")
print(f"loss  GD           : {mse_loss(X_train, y_train, w_gd, b_gd):.10f}")
assert np.allclose(w_gd, w_ne, atol=1e-6), "GD must reach the closed-form optimum"
print("\nsame solution to 1e-6")

# %% [markdown]
# ### What we skipped
#
# In practice you would use **mini-batch** gradient descent: estimate the
# gradient from a random subset of rows instead of all $n$. That makes each step
# $O(\text{batch} \cdot d)$, independent of dataset size, so training scales to
# data that does not fit in memory. The noise it introduces turns out to help
# escape bad regions in non-convex problems. Momentum, Adam, and learning-rate
# schedules all build on the same three lines above.

# %% [markdown]
# ## §5. Two models: plain features, and cross-features
#
# A linear model can only add features up. If the real mechanism says *"high CPU
# is fine, and a deep queue is fine, but high CPU **while** the queue is deep
# means trouble"*, no choice of weights on `cpu_util` and `queue_depth`
# separately can express it. That is an **interaction**, and the fix is to hand
# the model the product as a new feature:
#
# $$x_{\text{new}} = x_i \cdot x_j$$
#
# The model stays linear *in its parameters* — so all our machinery still
# applies — while becoming non-linear in the original features.
#
# With $d = 8$ we add every pair including squares: $\binom{8}{2} + 8 = 36$ new
# columns, for 44 total. ($x_i^2$ terms let the model express "moderate is
# best" curvature, which pure interactions cannot.)

# %%
def make_cross_features(X: np.ndarray, feature_names: list[str]):
    """Append every product x_i * x_j (i <= j) to the design matrix."""
    pairs = list(itertools.combinations_with_replacement(range(X.shape[1]), 2))
    extra = np.column_stack([X[:, i] * X[:, j] for i, j in pairs])
    names = feature_names + [
        f"{feature_names[i]}^2" if i == j else f"{feature_names[i]}*{feature_names[j]}"
        for i, j in pairs
    ]
    return np.hstack([X, extra]), names, pairs


_check, cross_names, CROSS_PAIRS = make_cross_features(X_train, FEATURES)
print(f"{X_train.shape[1]} features -> {_check.shape[1]} features "
      f"({len(CROSS_PAIRS)} products added)")
print("examples:", cross_names[8:12], "...", cross_names[-2:])

# %% [markdown]
# ### Packaging a model
#
# Both models share the same shape: standardise, optionally add cross-features,
# standardise again, then apply a linear score. We wrap that in a small class so
# the rest of the notebook can treat a model as a black box with `.predict()` —
# and so §11's anomaly finder works on either model unchanged.

# %%
@dataclass
class LinearProbabilityModel:
    """A linear model on (optionally cross-expanded) standardised features."""

    name: str
    feature_names: list[str]
    use_cross: bool
    feat_mean: np.ndarray
    feat_std: np.ndarray
    design_mean: np.ndarray | None = None
    design_std: np.ndarray | None = None
    w: np.ndarray | None = None
    b: float = 0.0
    loss_history: np.ndarray = field(default_factory=lambda: np.array([]))

    def design_matrix(self, frame: pd.DataFrame) -> np.ndarray:
        """Raw model-features -> the exact matrix the weights multiply."""
        X = (frame[self.feature_names].to_numpy(float) - self.feat_mean) / self.feat_std
        if self.use_cross:
            X, _, _ = make_cross_features(X, self.feature_names)
        if self.design_mean is not None:
            X = (X - self.design_mean) / self.design_std
        return X

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        return self.design_matrix(frame) @ self.w + self.b

    def to_bundle(self) -> dict:
        """Plain arrays only — a backup that unpickles without this class."""
        return {
            "name": self.name, "feature_names": list(self.feature_names),
            "use_cross": bool(self.use_cross),
            "feat_mean": self.feat_mean, "feat_std": self.feat_std,
            "design_mean": self.design_mean, "design_std": self.design_std,
            "w": self.w, "b": float(self.b),
        }


def critical_learning_rate(X: np.ndarray) -> float:
    """Largest stable step size for full-batch GD on the MSE — the §4 formula.

    (This uses a full eigendecomposition, which is itself O(d^3). For a large d
    you would estimate the top eigenvalue with a few power iterations instead.)
    """
    hessian_max_eig = 2 * np.linalg.eigvalsh(X.T @ X / len(X)).max()
    return 2 / hessian_max_eig


def train_model(name: str, frame: pd.DataFrame, y: np.ndarray, *,
                use_cross: bool, n_iters: int,
                lr_fraction: float = 0.9) -> LinearProbabilityModel:
    raw = frame[FEATURES].to_numpy(float)
    mu, sigma = raw.mean(axis=0), raw.std(axis=0, ddof=0)
    X = (raw - mu) / sigma

    if use_cross:
        X, _, _ = make_cross_features(X, FEATURES)
    # Re-standardise: products of standardised variables are not themselves
    # standardised, and their spread drives the usable learning rate.
    dmu, dsigma = X.mean(axis=0), X.std(axis=0, ddof=0)
    X = (X - dmu) / dsigma

    # Derive the step size from the data rather than hardcoding one.
    lr = lr_fraction * critical_learning_rate(X)
    print(f"{name:22s}  d = {X.shape[1]:3d}   "
          f"critical lr = {critical_learning_rate(X):.4f}   using {lr:.4f}")

    w, b, hist = fit_gradient_descent(X, y, lr=lr, n_iters=n_iters)
    return LinearProbabilityModel(
        name=name, feature_names=FEATURES, use_cross=use_cross,
        feat_mean=mu, feat_std=sigma, design_mean=dmu, design_std=dsigma,
        w=w, b=b, loss_history=hist,
    )


model_a = train_model("A · plain features", X_train_frame, y_train,
                      use_cross=False, n_iters=5000)
model_b = train_model("B · cross-features", X_train_frame, y_train,
                      use_cross=True, n_iters=20000)

print(f"\nmodel A: {len(model_a.w)} weights")
print(f"model B: {len(model_b.w)} weights")

# %% [markdown]
# Note the two critical learning rates: **0.47** for model A, **0.26** for model
# B. This is not a detail — a rate of 0.3, perfectly good for the 8-feature
# model, makes the 44-feature model diverge to `nan`. (This is not hypothetical;
# it happened while writing this notebook.) The §4 formula told us the safe
# range before we ran anything, which is why we compute it instead of guessing.
#
# Model B also needs 4× as many iterations. Both facts trace to the same cause:
# its design matrix is far worse conditioned, because products of features are
# correlated with each other *and* with the features they came from.
#
# Let us confirm both actually converged, by comparing against the closed form.

# %%
for model, frame in [(model_a, X_train_frame), (model_b, X_train_frame)]:
    X_design = model.design_matrix(frame)
    w_exact, b_exact = fit_normal_equation(X_design, y_train)
    loss_gd = mse_loss(X_design, y_train, model.w, model.b)
    loss_exact = mse_loss(X_design, y_train, w_exact, b_exact)
    print(f"{model.name:22s}  cond(X^T X) = "
          f"{np.linalg.cond(X_design.T @ X_design):8.1f}   "
          f"max|w_gd − w_exact| = {np.abs(model.w - w_exact).max():.2e}   "
          f"loss gap = {loss_gd - loss_exact:.2e}")
    # Assert on the loss, not on individual weights: for an ill-conditioned
    # design many weight vectors give near-identical loss, and the loss is what
    # gradient descent is actually minimising.
    assert loss_gd - loss_exact < 1e-9, f"{model.name} has not converged"

# %%
fig, ax = plt.subplots(figsize=(7.2, 4.2))
for model, colour in [(model_a, C["blue"]), (model_b, C["orange"])]:
    ax.plot(model.loss_history, color=colour, label=model.name)
ax.set_xscale("log")
finish(ax, "Training loss (MSE)", xlabel="iteration", ylabel="MSE", legend=True)
save_fig(fig, "09_training_loss")
plt.show()

# %%
scores = {}
for split_name, frame, y_true in [("train", X_train_frame, y_train),
                                  ("val", X_val_frame, y_val),
                                  ("test", X_test_frame, y_test)]:
    for model in (model_a, model_b):
        scores[(model.name, split_name)] = model.predict(frame)

summary = pd.DataFrame([
    {
        "model": model.name,
        "train AUC": roc_auc_score(y_train, scores[(model.name, "train")]),
        "val AUC": roc_auc_score(y_val, scores[(model.name, "val")]),
        "test AUC": roc_auc_score(y_test, scores[(model.name, "test")]),
    }
    for model in (model_a, model_b)
])
print(summary.to_string(index=False))
print(f"\ntest AUC gain from cross-features: "
      f"{summary['test AUC'][1] - summary['test AUC'][0]:+.4f}")

# %% [markdown]
# Model B is better — by about **+0.024 AUC**. Which raises exactly the question
# this lesson exists to answer: *is a gap that small real, or is it what you get
# from evaluating two similar models on one particular sample of 4000 rows?*
#
# Hold that thought until §10. First we need to know what these numbers mean.
#
# Note also the train-vs-val gap: model B has 44 parameters and shows a slightly
# larger gap than model A's 8. With 12 000 training rows neither is meaningfully
# overfitting, but the direction is visible, and it is why the honest comparison
# happens on held-out data.

# %% [markdown]
# ## §6. Thresholds, and the first three metrics
#
# The model outputs a score. A decision needs a **threshold** $t$:
#
# $$\hat y_i = \mathbb{1}[s_i \ge t]$$
#
# Every prediction now falls into one of four cells — the **confusion matrix**:
#
# |  | predicted 0 | predicted 1 |
# |---|---|---|
# | **actual 0** | TN | FP (false alarm) |
# | **actual 1** | FN (missed incident) | TP |
#
# And the three metrics:
#
# $$\text{accuracy} = \frac{TP + TN}{TP+TN+FP+FN}, \qquad
#   \text{precision} = \frac{TP}{TP + FP}, \qquad
#   \text{recall} = \frac{TP}{TP + FN}$$
#
# In words, and this is the part worth memorising:
#
# * **Precision** — of the windows we flagged, what fraction were real? *Low
#   precision means alert fatigue.*
# * **Recall** — of the real incidents, what fraction did we catch? *Low recall
#   means outages nobody noticed.*
#
# They trade off against each other, and the threshold is the dial.

# %%
def confusion_counts(y_true, y_pred):
    """Return (tn, fp, fn, tp) without any library help."""
    y_true = np.asarray(y_true).astype(bool)
    y_pred = np.asarray(y_pred).astype(bool)
    tp = int(np.sum(y_true & y_pred))
    tn = int(np.sum(~y_true & ~y_pred))
    fp = int(np.sum(~y_true & y_pred))
    fn = int(np.sum(y_true & ~y_pred))
    return tn, fp, fn, tp


def accuracy_manual(y_true, y_pred):
    tn, fp, fn, tp = confusion_counts(y_true, y_pred)
    return (tp + tn) / (tp + tn + fp + fn)


def precision_manual(y_true, y_pred):
    _, fp, _, tp = confusion_counts(y_true, y_pred)
    # No positive predictions -> precision is undefined. Returning 0 matches
    # sklearn's zero_division=0 default and keeps threshold sweeps plottable.
    return tp / (tp + fp) if (tp + fp) > 0 else 0.0


def recall_manual(y_true, y_pred):
    _, _, fn, tp = confusion_counts(y_true, y_pred)
    return tp / (tp + fn) if (tp + fn) > 0 else 0.0


val_scores_b = scores[(model_b.name, "val")]
THRESHOLD_DEMO = 0.5
y_hat = (val_scores_b >= THRESHOLD_DEMO).astype(int)

tn, fp, fn, tp = confusion_counts(y_val, y_hat)
print(f"threshold = {THRESHOLD_DEMO}   (validation set, model B)\n")
print(f"                predicted 0   predicted 1")
print(f"  actual 0      {tn:>11,}   {fp:>11,}")
print(f"  actual 1      {fn:>11,}   {tp:>11,}")
print(f"\n  accuracy  = (TP+TN)/n = ({tp:,}+{tn:,})/{len(y_val):,} "
      f"= {accuracy_manual(y_val, y_hat):.4f}")
print(f"  precision = TP/(TP+FP) = {tp:,}/({tp:,}+{fp:,}) "
      f"= {precision_manual(y_val, y_hat):.4f}")
print(f"  recall    = TP/(TP+FN) = {tp:,}/({tp:,}+{fn:,}) "
      f"= {recall_manual(y_val, y_hat):.4f}")

# %% [markdown]
# ### Now check against the library
#
# Always do this once. If your hand implementation and the library disagree, one
# of them is wrong and you want to find out here, not in production.

# %%
from sklearn.metrics import (accuracy_score, confusion_matrix, precision_score,
                             recall_score)

sk_tn, sk_fp, sk_fn, sk_tp = confusion_matrix(y_val, y_hat).ravel()
assert (tn, fp, fn, tp) == (sk_tn, sk_fp, sk_fn, sk_tp)

pairs = [
    ("accuracy", accuracy_manual(y_val, y_hat), accuracy_score(y_val, y_hat)),
    ("precision", precision_manual(y_val, y_hat),
     precision_score(y_val, y_hat, zero_division=0)),
    ("recall", recall_manual(y_val, y_hat), recall_score(y_val, y_hat)),
]
print(f"{'metric':<12}{'ours':>12}{'sklearn':>12}{'diff':>10}")
for nm, ours, theirs in pairs:
    print(f"{nm:<12}{ours:>12.6f}{theirs:>12.6f}{abs(ours - theirs):>10.1e}")
    assert np.isclose(ours, theirs), nm
print("\nall metrics agree")

# %% [markdown]
# ### The baseline that makes accuracy look silly

# %%
never = np.zeros_like(y_val)
print(f"model at t=0.5     : accuracy {accuracy_manual(y_val, y_hat):.4f}   "
      f"recall {recall_manual(y_val, y_hat):.4f}")
print(f"'never an incident': accuracy {accuracy_manual(y_val, never):.4f}   "
      f"recall {recall_manual(y_val, never):.4f}")

# %% [markdown]
# Look at what that comparison actually says. A model that **catches zero
# incidents** — that literally never fires — is already 75.3% accurate. Our real
# model is 85.2%.
#
# So the entire useful range of accuracy on this problem is the ten points
# between 75.3 and 100, and it starts at "does nothing". "The model is 85%
# accurate" sounds like a report of success while sitting barely a third of the
# way up that range — and a reader who does not know the base rate cannot tell
# the difference between our model and one that catches nothing at all.
#
# Accuracy is a fine metric when classes are balanced and both error types cost
# the same. Neither is true here, and neither is true in most real problems.
#
# ### Sweeping the threshold

# %%
def sweep_thresholds(y_true, score, grid):
    return pd.DataFrame([
        {
            "threshold": t,
            "accuracy": accuracy_manual(y_true, score >= t),
            "precision": precision_manual(y_true, score >= t),
            "recall": recall_manual(y_true, score >= t),
            "flagged": int(np.sum(score >= t)),
        }
        for t in grid
    ])


grid = np.linspace(val_scores_b.min(), val_scores_b.max(), 220)
sweep = sweep_thresholds(y_val, val_scores_b, grid)

fig, ax = plt.subplots(figsize=(7.6, 4.6))
for col, colour in [("precision", C["blue"]), ("recall", C["orange"]),
                    ("accuracy", C["aqua"])]:
    ax.plot(sweep["threshold"], sweep[col], color=colour, label=col)
    j = int(len(sweep) * 0.62)
    ax.annotate(col, xy=(sweep["threshold"][j], sweep[col][j]),
                xytext=(6, 6), textcoords="offset points",
                fontsize=9, color=colour, fontweight="semibold")
ax.axvline(THRESHOLD_DEMO, color=INK["axis"], linewidth=1.2)
ax.annotate("t = 0.5", xy=(THRESHOLD_DEMO, 0.02), xytext=(5, 0),
            textcoords="offset points", fontsize=8.5, color=INK["muted"])
finish(ax, "Every metric is a function of the threshold — model B, validation set",
       xlabel="threshold", ylabel="metric value")
ax.set_ylim(-0.02, 1.02)
save_fig(fig, "10_threshold_sweep")
plt.show()

# %% [markdown]
# The classic shape: as the threshold rises we flag fewer windows, so precision
# climbs and recall falls. Accuracy peaks somewhere unhelpful and is flat and
# uninformative across a wide range.
#
# **There is no "correct" threshold** — only a correct threshold *given a cost
# ratio*. If a missed incident costs 10× a false alarm, that is an input to the
# choice, and it comes from the business, not the data.
#
# What we can say without cost information is which threshold balances the two.
# That is F1, next. Note that we pick it on the **validation** set — choosing a
# threshold on test data means the test score is no longer an unbiased estimate
# of anything.

# %% [markdown]
# ## §7. F1 and ROC-AUC
#
# ### F1
#
# The harmonic mean of precision and recall:
#
# $$F_1 = 2\cdot\frac{\text{precision} \cdot \text{recall}}
#                     {\text{precision} + \text{recall}}$$
#
# Harmonic, not arithmetic, and that is the whole point: the harmonic mean is
# dominated by the smaller value. Precision 1.0 with recall 0.0 gives an
# arithmetic mean of 0.5 but $F_1 = 0$. You cannot game it by sacrificing one
# side.

# %%
def f1_manual(y_true, y_pred):
    p = precision_manual(y_true, y_pred)
    r = recall_manual(y_true, y_pred)
    return 2 * p * r / (p + r) if (p + r) > 0 else 0.0


from sklearn.metrics import f1_score

assert np.isclose(f1_manual(y_val, y_hat), f1_score(y_val, y_hat))
print(f"F1 at t=0.5: ours {f1_manual(y_val, y_hat):.6f}  "
      f"sklearn {f1_score(y_val, y_hat):.6f}")

sweep["f1"] = [f1_manual(y_val, val_scores_b >= t) for t in sweep["threshold"]]
best = sweep.loc[sweep["f1"].idxmax()]
BEST_THRESHOLD = float(best["threshold"])
print(f"\nbest F1 on validation = {best['f1']:.4f} at threshold "
      f"{BEST_THRESHOLD:.4f}")
print(f"  precision {best['precision']:.4f}   recall {best['recall']:.4f}   "
      f"flagged {int(best['flagged']):,} of {len(y_val):,}")
print(f"\n(compare with t = 0.5: F1 = {f1_manual(y_val, y_hat):.4f})")

# %% [markdown]
# ### ROC-AUC
#
# A threshold-free summary. Sweep the threshold across every possible value and
# plot two rates against each other:
#
# $$\mathrm{TPR} = \frac{TP}{TP+FN} \ (= \text{recall}), \qquad
#   \mathrm{FPR} = \frac{FP}{FP+TN}$$
#
# The **ROC curve** is that path; **AUC** is the area under it.
#
# AUC has an interpretation far more useful than "area":
#
# > **AUC is the probability that a randomly chosen positive is scored above a
# > randomly chosen negative.**
#
# So 0.5 is coin-flipping and 1.0 is perfect separation. It measures **ranking
# quality only** — it is completely blind to calibration, which is why our
# linear probability model's out-of-range predictions do not hurt it.
#
# Let us build the curve from scratch.

# %%
def roc_curve_manual(y_true, score):
    """ROC curve without library help.

    Sort by descending score; sweeping the threshold down the sorted list means
    each step moves one object from 'predicted 0' to 'predicted 1', so TP and FP
    counts are just cumulative sums.
    """
    y_true = np.asarray(y_true)
    score = np.asarray(score, dtype=float)

    order = np.argsort(-score, kind="mergesort")
    s_sorted = score[order]
    y_sorted = y_true[order]

    tps = np.cumsum(y_sorted)
    fps = np.cumsum(1 - y_sorted)

    # Only threshold values *between distinct scores* are real operating
    # points; tied scores must be committed to as a block.
    distinct = np.flatnonzero(np.diff(s_sorted))
    idx = np.r_[distinct, s_sorted.size - 1]

    tpr = np.r_[0.0, tps[idx] / tps[-1]]
    fpr = np.r_[0.0, fps[idx] / fps[-1]]
    thresholds = np.r_[np.inf, s_sorted[idx]]
    return fpr, tpr, thresholds


def auc_trapezoid(fpr, tpr):
    return float(np.trapezoid(tpr, fpr))


from sklearn.metrics import roc_curve

test_scores_a = scores[(model_a.name, "test")]
test_scores_b = scores[(model_b.name, "test")]

fpr_ours, tpr_ours, thr_ours = roc_curve_manual(y_test, test_scores_b)
fpr_sk, tpr_sk, thr_sk = roc_curve(y_test, test_scores_b, drop_intermediate=False)

print(f"points: ours {len(fpr_ours):,}   sklearn {len(fpr_sk):,}")
assert np.allclose(fpr_ours, fpr_sk) and np.allclose(tpr_ours, tpr_sk)
print("ROC curves identical to sklearn")

auc_manual = auc_trapezoid(fpr_ours, tpr_ours)
print(f"\nAUC by trapezoid : {auc_manual:.10f}")
print(f"AUC by sklearn   : {roc_auc_score(y_test, test_scores_b):.10f}")

# %% [markdown]
# ### The same number, a completely different way
#
# The "probability a random positive outranks a random negative" reading is not
# a metaphor — it is literally computable as a rank statistic, the
# **Mann–Whitney U**:
#
# $$\mathrm{AUC} = \frac{U}{n_{+} n_{-}}, \qquad
#   U = R_{+} - \frac{n_{+}(n_{+}+1)}{2}$$
#
# where $R_+$ is the sum of the ranks of the positives in the pooled sample.
# Ties get **average ranks**, which is exactly how the trapezoid rule handles
# the diagonal segments they produce.
#
# This equivalence is not a curiosity: it is the foundation of DeLong's test in
# §10.

# %%
def auc_via_ranks(y_true, score):
    y_true = np.asarray(y_true)
    ranks = stats.rankdata(score)          # average ranks for ties
    n_pos = int(y_true.sum())
    n_neg = len(y_true) - n_pos
    r_pos = ranks[y_true == 1].sum()
    u = r_pos - n_pos * (n_pos + 1) / 2
    return u / (n_pos * n_neg)


print(f"AUC via trapezoid    : {auc_manual:.12f}")
print(f"AUC via Mann-Whitney : {auc_via_ranks(y_test, test_scores_b):.12f}")
print(f"AUC via sklearn      : {roc_auc_score(y_test, test_scores_b):.12f}")
assert np.isclose(auc_manual, auc_via_ranks(y_test, test_scores_b))
assert np.isclose(auc_manual, roc_auc_score(y_test, test_scores_b))
print("\nall three agree to machine precision")

# %%
fig, ax = plt.subplots(figsize=(6.2, 6.0))
ax.plot([0, 1], [0, 1], color=INK["axis"], linewidth=1.2, zorder=1)
ax.annotate("random guessing", xy=(0.62, 0.62), xytext=(6, -14),
            textcoords="offset points", fontsize=8.5, color=INK["muted"],
            rotation=45, rotation_mode="anchor")

for sc, model, colour in [(test_scores_a, model_a, C["blue"]),
                          (test_scores_b, model_b, C["orange"])]:
    f, t, _ = roc_curve_manual(y_test, sc)
    ax.plot(f, t, color=colour, zorder=3,
            label=f"{model.name} — AUC {auc_trapezoid(f, t):.4f}")

ax.set_xlim(-0.01, 1.01)
ax.set_ylim(-0.01, 1.01)
ax.set_aspect("equal")
finish(ax, "ROC curves on the test set",
       xlabel="false positive rate", ylabel="true positive rate", legend=True)
ax.legend(loc="lower right")
save_fig(fig, "11_roc_curves")
plt.show()

# %% [markdown]
# The two curves are close, with orange above blue nearly everywhere. Zoom in on
# the region an on-call team would actually operate in — you cannot afford a 40%
# false-positive rate, so only the left edge matters:

# %%
fig, ax = plt.subplots(figsize=(6.4, 4.6))
for sc, model, colour in [(test_scores_a, model_a, C["blue"]),
                          (test_scores_b, model_b, C["orange"])]:
    f, t, _ = roc_curve_manual(y_test, sc)
    ax.plot(f, t, color=colour, label=model.name)
ax.set_xlim(0, 0.2)
ax.set_ylim(0, 0.75)
finish(ax, "The operating region you actually care about (FPR ≤ 0.2)",
       xlabel="false positive rate", ylabel="true positive rate", legend=True)
ax.legend(loc="lower right")
save_fig(fig, "12_roc_zoom")
plt.show()

# %% [markdown]
# ### One caveat on AUC
#
# ROC-AUC is computed over *all* thresholds, including absurd ones. Under heavy
# class imbalance it can look reassuring while the model is useless at the only
# thresholds you would ever deploy, because the FPR denominator ($n_-$) is huge
# and swamps a small number of false positives. When positives are rare
# (< ~5%), report **average precision** (area under the precision–recall curve)
# alongside it. At 25% positives we are fine.

# %% [markdown]
# ## §8. A test-set score is a random variable
#
# Here is the conceptual centre of the lesson.
#
# We reported model B's test AUC as 0.8634. That number is **not** the model's
# AUC. It is an estimate of it, computed from 4000 rows that happened to land in
# the test split. A different 4000 rows would have given a different number.
#
# Formally: the test set is a sample from the population, so any statistic
# computed on it — AUC, F1, precision — is a **random variable**. Reporting it
# without an interval is claiming a precision you do not have.
#
# The claim is easy to demonstrate. Split the test set in half and evaluate
# each half:

# %%
perm = rng.permutation(len(y_test))
halves = np.array_split(perm, 2)
for i, h in enumerate(halves):
    print(f"  test half {i + 1} (n = {len(h):,}): "
          f"AUC = {roc_auc_score(y_test[h], test_scores_b[h]):.4f}")
print(f"  full test set  (n = {len(y_test):,}): "
      f"AUC = {roc_auc_score(y_test, test_scores_b):.4f}")

print("\n20 random halves:")
aucs_halves = [
    roc_auc_score(y_test[idx], test_scores_b[idx])
    for idx in (rng.permutation(len(y_test))[:len(y_test) // 2] for _ in range(20))
]
print(f"  min {np.min(aucs_halves):.4f}   max {np.max(aucs_halves):.4f}   "
      f"spread {np.max(aucs_halves) - np.min(aucs_halves):.4f}")

# %% [markdown]
# The spread across random halves is comparable to the entire gap between our
# two models. If you are not quantifying that, you are not measuring anything.
#
# ### So we need a confidence interval. But from what distribution?
#
# For a simple mean, the central limit theorem hands you $\mathrm{SE} =
# \sigma/\sqrt{n}$ and you are done. AUC is **not** a mean of independent
# terms — it is a rank statistic over all $n_+ \times n_-$ pairs, and those
# pairs share objects, so they are dependent. There is no elementary formula.
#
# ### The bootstrap
#
# Efron's insight (1979): we cannot resample from the population, but the sample
# *is* our best estimate of the population. So resample from the sample.
#
# > **The bootstrap.** Given a sample $D$ of size $n$ and a statistic
# > $\hat\theta = s(D)$:
# >
# > 1. Draw $D^*_b$ by sampling $n$ objects from $D$ **with replacement**.
# > 2. Compute $\hat\theta^*_b = s(D^*_b)$.
# > 3. Repeat for $b = 1 \ldots B$ (typically 1000–10000).
# >
# > The spread of $\{\hat\theta^*_b\}$ estimates the spread of $\hat\theta$.
# > The percentile interval is the empirical 2.5th and 97.5th percentiles.
#
# *With replacement* is the essential part: it makes each resample differ from
# the original, and on average each contains about 63.2% of the distinct
# original objects ($1 - e^{-1}$).
#
# What makes it remarkable is that it needs no formula for the statistic. It
# works for AUC, for the median, for the gap between two models' F1 scores — for
# anything you can compute.
#
# **It is not magic.** The bootstrap assumes your sample is representative and
# your objects are independent. If the test set is small, or the rows are
# correlated (repeated users, time series, several rows per node), the interval
# will be too narrow and confidently wrong.

# %% [markdown]
# ## §9. Implementing the bootstrap
#
# ### By hand

# %%
def bootstrap_metric(y_true, score, metric_fn, n_boot=2000, seed=0,
                     stratified=True):
    """Percentile bootstrap for a metric of the form metric(y_true, score)."""
    boot_rng = np.random.default_rng(seed)
    y_true = np.asarray(y_true)
    score = np.asarray(score)

    if stratified:
        # Resample within each class so every resample keeps the class balance.
        # Unstratified resampling can (rarely) produce a resample with one class
        # absent, for which AUC is undefined.
        pos = np.flatnonzero(y_true == 1)
        neg = np.flatnonzero(y_true == 0)

    stats_out = np.empty(n_boot)
    for b in range(n_boot):
        if stratified:
            idx = np.concatenate([
                boot_rng.choice(pos, pos.size, replace=True),
                boot_rng.choice(neg, neg.size, replace=True),
            ])
        else:
            idx = boot_rng.integers(0, len(y_true), len(y_true))
        stats_out[b] = metric_fn(y_true[idx], score[idx])
    return stats_out


t0 = time.perf_counter()
boot_auc_b = bootstrap_metric(y_test, test_scores_b, roc_auc_score,
                              n_boot=2000, seed=RANDOM_SEED)
elapsed = time.perf_counter() - t0

point = roc_auc_score(y_test, test_scores_b)
lo, hi = np.percentile(boot_auc_b, [2.5, 97.5])
print(f"model B test AUC   : {point:.4f}")
print(f"bootstrap SE       : {boot_auc_b.std(ddof=1):.4f}")
print(f"95% percentile CI  : [{lo:.4f}, {hi:.4f}]  (width {hi - lo:.4f})")
print(f"2000 resamples in {elapsed:.2f} s")

# %%
fig, ax = plt.subplots(figsize=(7.4, 4.2))
ax.hist(boot_auc_b, bins=60, color=C["blue"], alpha=0.85, edgecolor="none")
for value, colour, label, offset in [
    (point, INK["primary"], f"observed  {point:.4f}", 8),
    (lo, INK["muted"], f"2.5%  {lo:.4f}", -8),
    (hi, INK["muted"], f"97.5%  {hi:.4f}", 8),
]:
    ax.axvline(value, color=colour, linewidth=1.6)
    ax.annotate(label, xy=(value, ax.get_ylim()[1] * 0.94),
                xytext=(offset, 0), textcoords="offset points", fontsize=8.5,
                color=colour, ha="left" if offset > 0 else "right")
finish(ax, "Bootstrap distribution of model B's test AUC (2000 resamples)",
       xlabel="AUC", ylabel="resamples")
save_fig(fig, "13_bootstrap_auc")
plt.show()

# %% [markdown]
# That histogram is the answer to "how precisely do we know the AUC?". It is
# roughly normal here — which is common but never guaranteed, and is exactly the
# kind of assumption the bootstrap lets you *check* instead of assume.
#
# ### The same thing with a library

# %%
res = stats.bootstrap(
    (y_test, test_scores_b),
    statistic=lambda yy, ss: roc_auc_score(yy, ss),
    n_resamples=2000,
    paired=True,          # y and score must be resampled together
    vectorized=False,
    method="percentile",
    random_state=RANDOM_SEED,
)
print(f"{'':<22}{'low':>10}{'high':>10}{'SE':>10}")
print(f"{'ours (stratified)':<22}{lo:>10.4f}{hi:>10.4f}"
      f"{boot_auc_b.std(ddof=1):>10.4f}")
print(f"{'scipy percentile':<22}{res.confidence_interval.low:>10.4f}"
      f"{res.confidence_interval.high:>10.4f}{res.standard_error:>10.4f}")

res_bca = stats.bootstrap(
    (y_test, test_scores_b),
    statistic=lambda yy, ss: roc_auc_score(yy, ss),
    n_resamples=2000, paired=True, vectorized=False,
    method="BCa", random_state=RANDOM_SEED,
)
print(f"{'scipy BCa':<22}{res_bca.confidence_interval.low:>10.4f}"
      f"{res_bca.confidence_interval.high:>10.4f}{res_bca.standard_error:>10.4f}")

# %% [markdown]
# All three agree to about the third decimal, and the residual differences here
# are Monte-Carlo noise rather than anything meaningful — with only 2000
# resamples the interval endpoints themselves carry a couple of units in the
# fourth decimal. Do not read anything into which is largest.
#
# The two *design* differences are still worth understanding:
#
# * **ours** resamples *within each class*, holding $n_+$ and $n_-$ fixed; scipy
#   resamples rows freely. At 25% positives in 4000 rows this changes almost
#   nothing. It matters when a class is small: free resampling can produce a
#   resample containing no positives at all, for which AUC is undefined.
# * **BCa** ("bias-corrected and accelerated") adjusts the percentiles for skew
#   and bias in the bootstrap distribution. It is the better default for skewed
#   statistics, at the cost of an extra jackknife pass. For a near-symmetric
#   distribution like this one it barely moves.
#
# ### Bootstrapping anything: F1 at our chosen threshold
#
# The point of the bootstrap is that it does not care what the statistic is.

# %%
f1_at_threshold = lambda yy, ss: f1_score(yy, (ss >= BEST_THRESHOLD).astype(int))
boot_f1 = bootstrap_metric(y_test, test_scores_b, f1_at_threshold,
                           n_boot=2000, seed=RANDOM_SEED)
f1_point = f1_at_threshold(y_test, test_scores_b)
f1_lo, f1_hi = np.percentile(boot_f1, [2.5, 97.5])
print(f"model B test F1 (threshold {BEST_THRESHOLD:.4f} chosen on validation)")
print(f"  point estimate : {f1_point:.4f}")
print(f"  95% CI         : [{f1_lo:.4f}, {f1_hi:.4f}]")

# %% [markdown]
# ### Closing the loop on §1
#
# We answered Q1 with a two-proportion z-test that required knowing the
# sampling distribution of a difference of proportions. The bootstrap needs no
# such knowledge. If the two agree, that is real evidence both are right.

# %%
pre_arr = pre.to_numpy()
post_arr = post.to_numpy()
boot_diffs = np.array([
    rng.choice(post_arr, post_arr.size, replace=True).mean()
    - rng.choice(pre_arr, pre_arr.size, replace=True).mean()
    for _ in range(4000)
])
b_lo, b_hi = np.percentile(boot_diffs, [2.5, 97.5])

z_lo = diff - 1.96 * np.sqrt(pre.mean() * (1 - pre.mean()) / len(pre)
                             + post.mean() * (1 - post.mean()) / len(post))
z_hi = diff + 1.96 * np.sqrt(pre.mean() * (1 - pre.mean()) / len(pre)
                             + post.mean() * (1 - post.mean()) / len(post))

print(f"difference in incident rate (post − pre) = {diff:+.4f}")
print(f"  analytic 95% CI  : [{z_lo:+.4f}, {z_hi:+.4f}]")
print(f"  bootstrap 95% CI : [{b_lo:+.4f}, {b_hi:+.4f}]")
print(f"  bootstrap P(difference <= 0) = {(boot_diffs <= 0).mean():.4f}")
print("\nNeither interval contains zero.")

# %% [markdown]
# ## §10. Is model B really better? DeLong's test
#
# We have two AUCs on the same test set: 0.8394 and 0.8634. We now know each
# carries a 95% CI roughly ±0.015 wide — so model A spans about
# [0.824, 0.855] and model B about [0.848, 0.878]. Those intervals **overlap**.
#
# It is tempting to conclude "not significant". That reasoning is **wrong**, and
# it is one of the most common statistical errors in applied ML.
#
# ### Why overlapping intervals do not settle it
#
# There are **two** separate reasons, and they are worth keeping apart.
#
# **First, and true even for completely independent estimates:** "do the error
# bars overlap?" is simply not the same question as "is the difference
# significant?". The interval for a difference is built from
# $\sqrt{\mathrm{se}_1^2 + \mathrm{se}_2^2}$, not from $\mathrm{se}_1 +
# \mathrm{se}_2$, and the first is always smaller. Two 95% intervals can overlap
# while the difference is significant at $p < 0.05$, with zero covariance
# anywhere. Testing a difference requires the distribution *of the difference*.
#
# **Second, and specific to our situation:** the two models are evaluated on
# **the same rows**, so their errors are strongly correlated — a test set that
# happens to contain easy positives inflates *both* AUCs together. That common
# fluctuation cancels in the difference:
#
# $$\mathrm{Var}(A_1 - A_2) = \mathrm{Var}(A_1) + \mathrm{Var}(A_2)
#   - 2\,\mathrm{Cov}(A_1, A_2)$$
#
# With a large positive covariance the difference is far more precisely
# determined than either AUC alone. So the eyeball test is over-conservative
# twice over — and the fix in both cases is the same: estimate the distribution
# of the difference directly.
#
# The paired bootstrap already gets this right, as long as you resample **rows**
# and re-evaluate both models on the same resample:

# %%
def paired_bootstrap_auc_diff(y_true, score_1, score_2, n_boot=4000, seed=0):
    """Bootstrap the AUC difference, resampling rows once for both models."""
    boot_rng = np.random.default_rng(seed)
    y_true = np.asarray(y_true)
    pos = np.flatnonzero(y_true == 1)
    neg = np.flatnonzero(y_true == 0)

    diffs = np.empty(n_boot)
    for b in range(n_boot):
        idx = np.concatenate([
            boot_rng.choice(pos, pos.size, replace=True),
            boot_rng.choice(neg, neg.size, replace=True),
        ])
        diffs[b] = (roc_auc_score(y_true[idx], score_2[idx])
                    - roc_auc_score(y_true[idx], score_1[idx]))
    return diffs


boot_a = bootstrap_metric(y_test, test_scores_a, roc_auc_score, 2000,
                          seed=RANDOM_SEED)
observed_diff = (roc_auc_score(y_test, test_scores_b)
                 - roc_auc_score(y_test, test_scores_a))

paired_diffs = paired_bootstrap_auc_diff(y_test, test_scores_a, test_scores_b,
                                         n_boot=4000, seed=RANDOM_SEED)
pd_lo, pd_hi = np.percentile(paired_diffs, [2.5, 97.5])

print(f"SE of model A's AUC       : {boot_a.std(ddof=1):.5f}")
print(f"SE of model B's AUC       : {boot_auc_b.std(ddof=1):.5f}")
print(f"if they were independent  : "
      f"{np.sqrt(boot_a.std(ddof=1) ** 2 + boot_auc_b.std(ddof=1) ** 2):.5f}")
print(f"SE of the DIFFERENCE      : {paired_diffs.std(ddof=1):.5f}   "
      f"<- much smaller\n")
print(f"observed difference       : {observed_diff:+.5f}")
print(f"95% CI for the difference : [{pd_lo:+.5f}, {pd_hi:+.5f}]")
print(f"bootstrap P(difference <= 0) = {(paired_diffs <= 0).mean():.4f}")

# %% [markdown]
# The SE of the difference is roughly **a third** of what independence would
# predict. The interval excludes zero comfortably.
#
# ### DeLong's test: the same answer without resampling
#
# The paired bootstrap above took a few seconds and gives a slightly different
# answer every run. DeLong, DeLong & Clarke-Pearson (1988) derived the
# covariance of two correlated AUCs **in closed form**, using the Mann–Whitney
# representation from §7. Sun & Xu (2014) made it $O(n \log n)$.
#
# The idea: AUC is a two-sample U-statistic, and U-statistics have a known
# asymptotic variance built from *structural components* — one number per
# object, measuring how much that object contributes to the AUC:
#
# $$V^{(1)}_i = \frac{1}{n_-}\sum_{j} \psi(X_i, Y_j), \qquad
#   V^{(0)}_j = \frac{1}{n_+}\sum_{i} \psi(X_i, Y_j)$$
#
# where $\psi(x, y) = \mathbb{1}[x > y] + \tfrac12 \mathbb{1}[x = y]$. So
# $V^{(1)}_i$ is the fraction of **negatives that positive $i$ outranks**, and
# $V^{(0)}_j$ is the fraction of **positives that outrank negative $j$**. (Note
# the asymmetry: both are written with $\psi$ in the same order, so $V^{(0)}$
# counts *against* the negative. The code's `v10` therefore stores
# $1 - V^{(0)}$, which is what the variance formula needs.) The covariance of two
# AUCs is then estimated from the sample covariance of these components:
#
# $$\widehat{\mathrm{Cov}}(A_1, A_2) =
#   \frac{S^{(1)}_{12}}{n_+} + \frac{S^{(0)}_{12}}{n_-}$$
#
# The trick that makes it fast is that all those sums can be read off
# **midranks** instead of computed pairwise.

# %%
def midrank(x):
    """Ranks 1..n with ties averaged."""
    order = np.argsort(x)
    z = x[order]
    n = len(x)
    t = np.zeros(n, dtype=float)
    i = 0
    while i < n:
        j = i
        while j < n and z[j] == z[i]:
            j += 1
        t[i:j] = 0.5 * (i + j - 1) + 1
        i = j
    out = np.empty(n, dtype=float)
    out[order] = t
    return out


def delong_auc_cov(score_matrix, y_true):
    """Fast DeLong. Returns (aucs, covariance matrix) for k score vectors."""
    score_matrix = np.atleast_2d(np.asarray(score_matrix, dtype=float))
    y_true = np.asarray(y_true).astype(int)

    pos = y_true == 1
    n_pos, n_neg = int(pos.sum()), int((~pos).sum())
    k = score_matrix.shape[0]

    ordered = np.hstack([score_matrix[:, pos], score_matrix[:, ~pos]])

    tx = np.empty((k, n_pos))       # midranks of positives among positives
    ty = np.empty((k, n_neg))       # midranks of negatives among negatives
    tz = np.empty((k, n_pos + n_neg))   # midranks in the pooled sample
    for r in range(k):
        tx[r] = midrank(ordered[r, :n_pos])
        ty[r] = midrank(ordered[r, n_pos:])
        tz[r] = midrank(ordered[r])

    # Mann-Whitney AUC straight from the pooled ranks of the positives
    aucs = tz[:, :n_pos].sum(axis=1) / n_pos / n_neg - (n_pos + 1.0) / 2.0 / n_neg

    v01 = (tz[:, :n_pos] - tx) / n_neg        # structural components, positives
    v10 = 1.0 - (tz[:, n_pos:] - ty) / n_pos  # structural components, negatives

    cov = (np.cov(v01, ddof=1).reshape(k, k) / n_pos
           + np.cov(v10, ddof=1).reshape(k, k) / n_neg)
    return aucs, cov


def delong_test(y_true, score_1, score_2):
    """Two-sided DeLong test for AUC_1 == AUC_2 on paired data."""
    aucs, cov = delong_auc_cov(np.vstack([score_1, score_2]), y_true)
    var_diff = cov[0, 0] + cov[1, 1] - 2 * cov[0, 1]
    if var_diff <= 0:
        return dict(auc_1=aucs[0], auc_2=aucs[1], diff=aucs[1] - aucs[0],
                    se=0.0, z=0.0, p_value=1.0, cov=cov)
    se = np.sqrt(var_diff)
    z = (aucs[1] - aucs[0]) / se
    return dict(auc_1=aucs[0], auc_2=aucs[1], diff=aucs[1] - aucs[0],
                se=se, z=z, p_value=2 * stats.norm.sf(abs(z)), cov=cov)


result = delong_test(y_test, test_scores_a, test_scores_b)

print(f"AUC model A     : {result['auc_1']:.5f}")
print(f"AUC model B     : {result['auc_2']:.5f}")
print(f"difference      : {result['diff']:+.5f}")
print(f"SE(difference)  : {result['se']:.5f}")
print(f"z statistic     : {result['z']:.4f}")
print(f"p-value         : {result['p_value']:.3e}")

corr_ab = result["cov"][0, 1] / np.sqrt(result["cov"][0, 0] * result["cov"][1, 1])
print(f"\ncorrelation between the two AUC estimates: {corr_ab:.4f}")
print("(that is the covariance the naive 'do the CIs overlap?' check discards)")

# %% [markdown]
# ### Verifying the implementation
#
# Three checks, because a subtly wrong variance formula produces plausible
# numbers forever.

# %%
# 1. DeLong's AUCs must equal sklearn's, exactly.
assert np.isclose(result["auc_1"], roc_auc_score(y_test, test_scores_a))
assert np.isclose(result["auc_2"], roc_auc_score(y_test, test_scores_b))

# 2. A model against itself: zero difference, zero variance, p = 1.
self_test = delong_test(y_test, test_scores_b, test_scores_b)

# 3. The closed-form SE must match the paired bootstrap's SE.
print(f"1. AUCs match sklearn                       ok")
print(f"2. model vs itself: diff={self_test['diff']:.1e}, "
      f"p={self_test['p_value']:.4f}")
print(f"3. SE(diff)  DeLong    = {result['se']:.5f}")
print(f"   SE(diff)  bootstrap = {paired_diffs.std(ddof=1):.5f}")
print(f"   relative gap        = "
      f"{abs(result['se'] - paired_diffs.std(ddof=1)) / result['se']:.2%}")
assert abs(result["se"] - paired_diffs.std(ddof=1)) / result["se"] < 0.10

# %%
fig, ax = plt.subplots(figsize=(7.4, 4.2))
ax.hist(paired_diffs, bins=60, color=C["blue"], alpha=0.85, edgecolor="none",
        label="paired bootstrap")
xs = np.linspace(paired_diffs.min(), paired_diffs.max(), 300)
density = stats.norm.pdf(xs, result["diff"], result["se"])
scale = len(paired_diffs) * (paired_diffs.max() - paired_diffs.min()) / 60
ax.plot(xs, density * scale, color=C["orange"], label="DeLong normal approx.")
ax.axvline(0, color=INK["primary"], linewidth=1.6)
ax.annotate("no difference", xy=(0, ax.get_ylim()[1] * 0.9), xytext=(-8, 0),
            textcoords="offset points", fontsize=8.5, ha="right",
            color=INK["primary"])
finish(ax, "Sampling distribution of AUC(B) − AUC(A): two routes, one answer",
       xlabel="AUC difference", ylabel="resamples", legend=True)
save_fig(fig, "14_delong_vs_bootstrap")
plt.show()

# %% [markdown]
# The closed-form curve lands on the bootstrap histogram. Two methods with
# nothing in common agreeing to the third decimal is about as good as
# verification gets.
#
# **Conclusion: model B is genuinely better** (p ≈ 1e-9). Small, but real.
#
# ### When to use which
#
# | | DeLong | Paired bootstrap |
# |---|---|---|
# | speed | milliseconds | seconds to minutes |
# | determinism | exact | varies run to run |
# | scope | AUC only | any statistic |
# | assumptions | asymptotic normality | sample is representative |
#
# Use DeLong for AUC. Use the bootstrap for everything else — F1, precision at a
# fixed recall, revenue per session.
#
# ### Two caveats worth more than the test itself
#
# 1. **Statistical significance is not practical significance.** +0.024 AUC is
#    real. Whether it justifies 36 extra features and a harder-to-debug model is
#    a different question, and the p-value has no opinion on it. §11 will give
#    us a concrete reason to hesitate.
# 2. **This is one comparison.** If you test twenty model variants against a
#    baseline on the same test set, you should expect one to look significant at
#    p < 0.05 by chance alone. Repeated use of the same test set erodes it —
#    which is what validation splits are for.

# %% [markdown]
# ## §11. Finding anomalies in the error distribution
#
# ### The proposal, and an honest examination of it
#
# > *Rank objects by model error. Take the top 1%. Call those anomalies.*
#
# This is a real technique — model-residual-based anomaly detection — and the
# intuition behind it is sound: an object the model gets badly wrong is an
# object that does not behave like the rest of the data.
#
# But four things have to be said before we implement it, because each one is a
# way to get a confident, wrong answer.
#
# **1. A percentile threshold cannot tell you *whether* there are anomalies.**
# If you flag the top 1% of errors you will flag exactly 1% of your rows —
# on clean data, on corrupted data, always. The count is fixed by construction,
# so "how many anomalies did we find?" is a question a percentile *cannot*
# answer. Only an **absolute** error threshold produces a count that carries
# information.
#
# **2. Errors must be computed out-of-fold.** Residuals on training data are
# shrunk by the fitting process itself. Use held-out data or cross-validated
# predictions.
#
# **3. A large residual has at least four different causes**, and they call for
# opposite responses: a genuinely anomalous object; a *region* where the model
# is simply misspecified (a systematic failure, not an outlier); a mislabelled
# row; or plain irreducible noise — for an object whose true probability is 0.5,
# a large residual is guaranteed no matter how good the model is. **Residual
# ranking finds "where the model is surprised", which overlaps with "anomalous"
# but is not the same set.**
#
# **4. For a binary target the *signed* residual splits into two populations**,
# because $y$ is 0 or 1: rows the model scored too low sit near $+1$, rows it
# scored too high sit near $-1$. Taking $|y - \hat y|$ folds those two
# populations onto each other, so a single percentile cut mixes "confidently
# predicted 0, actually 1" with the exact reverse. The **sign** carries most of
# the diagnostic value, and the absolute value throws it away.
#
# So: we implement it, and then we *test* whether it works — which we can do
# only because this dataset has ground truth.

# %%
def find_anomalies(model, frame, y_true, percentile=99.0, abs_threshold=None,
                   signed=False):
    """Flag rows whose prediction error is extreme.

    Parameters
    ----------
    model : anything with .predict(frame) -> scores
    frame, y_true : the data to score. Must be held-out w.r.t. `model`.
    percentile : flag errors at or above this percentile of the error
        distribution. Note this always flags (100 - percentile)% of rows.
    abs_threshold : if given, overrides `percentile` and flags rows with error
        at or above this absolute value. This is the mode whose *count* is
        informative.
    signed : if True, rank by the signed residual (y - y_hat) instead of its
        magnitude, so under- and over-prediction are separated.

    Returns
    -------
    (flagged, cutoff)
        `flagged` is a DataFrame indexed like `frame`, containing only the
        flagged rows, sorted by decreasing `error`. Under ``signed=True`` that
        sorts by most-positive residual rather than most-extreme, since the
        point of that mode is to isolate one tail. `cutoff` is the error
        threshold actually applied.
    """
    score = model.predict(frame)
    residual = np.asarray(y_true, dtype=float) - score
    error = residual if signed else np.abs(residual)

    if abs_threshold is not None:
        cutoff = abs_threshold
    else:
        cutoff = float(np.percentile(error, percentile))

    flagged = error >= cutoff
    out = pd.DataFrame({
        "score": score,
        "y_true": np.asarray(y_true),
        "residual": residual,
        "error": error,
    }, index=frame.index)[flagged]
    return out.sort_values("error", ascending=False), cutoff


# Anomalies are rare, so we pool BOTH held-out splits to get a usable sample of
# them. Both are legitimate: the model never trained on either, which is the
# only property section 11 needs. (We are not measuring model quality here, so
# the "don't touch the test set" rule of §6 is not in play.)
heldout_mask = val_mask | test_mask
X_heldout_frame = X_all[heldout_mask]
y_heldout = y_all[heldout_mask]

anomalies, cutoff = find_anomalies(model_b, X_heldout_frame, y_heldout,
                                   percentile=99.0)
print(f"held-out rows: {len(X_heldout_frame):,}")
print(f"error cutoff at the 99th percentile: {cutoff:.4f}")
print(f"flagged {len(anomalies):,} rows "
      f"({len(anomalies) / len(X_heldout_frame):.2%})\n")
print(anomalies.head(8).to_string())

# %% [markdown]
# ### The error distribution
#
# Claim 4 above, made visible:

# %%
resid = y_heldout - model_b.predict(X_heldout_frame)

fig, axes = plt.subplots(1, 2, figsize=(13, 4.2))

axes[0].hist(np.abs(resid), bins=70, color=C["blue"], alpha=0.85,
             edgecolor="none")
axes[0].axvline(cutoff, color=INK["primary"], linewidth=1.6)
axes[0].annotate(f"99th pct = {cutoff:.2f}", xy=(cutoff, axes[0].get_ylim()[1] * 0.8),
                 xytext=(-8, 0), textcoords="offset points", ha="right",
                 fontsize=8.5, color=INK["primary"])
finish(axes[0], "|error| — one pile, both failure directions folded together",
       xlabel="|y − ŷ|", ylabel="windows")

bins = np.linspace(resid.min(), resid.max(), 70)
for cls, colour, label in [(0, C["blue"], "actual 0"), (1, C["orange"], "actual 1")]:
    axes[1].hist(resid[y_heldout == cls], bins=bins, histtype="step",
                 linewidth=1.8, color=colour, label=label)
    axes[1].hist(resid[y_heldout == cls], bins=bins, color=colour, alpha=0.10)
finish(axes[1], "signed residual y − ŷ, split by true class",
       xlabel="y − ŷ", ylabel="windows", legend=True)

fig.tight_layout()
save_fig(fig, "15_error_distribution")
plt.show()

# %% [markdown]
# The right panel is the informative one. Two distinct populations — negatives
# the model scored too high (left) and positives it scored too low (right) —
# which do overlap in the middle, but whose extremes mean opposite things.
#
# The left panel is what you get by taking the absolute value: a single pile
# decaying away from zero, with the two tails of the right panel folded on top
# of each other. Note that $|y - \hat y|$ is *not* bimodal — it is unimodal at 0
# with a long shoulder — so you cannot recover the split by looking at it.
#
# ### Count versus threshold — the point about percentiles
#
# The user's question was "plot the number of anomalies against the percentile,
# or against the magnitude of the error". Both, side by side, because the
# contrast *is* the lesson.

# %%
abs_err = np.abs(resid)

percentiles = np.linspace(80, 99.9, 200)
counts_by_pct = [(abs_err >= np.percentile(abs_err, p)).sum() for p in percentiles]

thresholds = np.linspace(0, abs_err.max(), 200)
counts_by_thr = [(abs_err >= t).sum() for t in thresholds]

fig, axes = plt.subplots(1, 2, figsize=(13, 4.4))

axes[0].plot(percentiles, counts_by_pct, color=C["blue"], label="observed")
axes[0].plot(percentiles, (1 - percentiles / 100) * len(y_heldout),
             color=INK["muted"], linewidth=1.4, label="(1 − p)·n, exactly")
finish(axes[0], "Count vs percentile — determined by arithmetic, not by data",
       xlabel="percentile threshold", ylabel="rows flagged", legend=True)

axes[1].plot(thresholds, counts_by_thr, color=C["orange"])
axes[1].set_yscale("log")
finish(axes[1], "Count vs absolute error threshold — this one carries information",
       xlabel="|y − ŷ| threshold", ylabel="rows flagged (log)")

fig.tight_layout()
save_fig(fig, "16_anomaly_counts")
plt.show()

# %% [markdown]
# The left panel is a straight line lying exactly on $(1-p)\cdot n$. It contains
# no information about the data whatsoever — the same plot appears for pure
# noise or for perfectly clean data.
#
# The right panel has structure: a steep fall, then a shoulder, then a thin tail
# that persists past $|y - \hat y| = 1$ — errors that *exceed what a probability
# is even capable of*, which is only possible because our linear model predicts
# outside [0,1]. Those extreme rows are the interesting ones.
#
# **Practical rule:** set the threshold on a *reference* period with a percentile
# once, then hold that absolute threshold fixed and watch the count over time. A
# rising count is a signal. A percentile recomputed every day can never rise.

# %% [markdown]
# ### Does it actually work?
#
# Now the part that is impossible on real data. This dataset contains 360
# deliberately injected anomalies of two kinds, and we have not touched the
# ground-truth file until this moment.
#
# * **`silent_failure`** (240) — a config regression shipped with the new
#   version. Telemetry looks completely healthy; the window is an incident
#   anyway, with probability 0.95 (so a handful are labelled 0). By construction
#   *no* telemetry-based model can predict these.
# * **`sensor_glitch`** (120) — a broken disk-latency probe reporting absurd
#   values while the service is fine. The model panics; nothing happens.

# %%
truth = pd.read_csv(DATA_DIR / "anomaly_ground_truth.csv")
df_truth = df.merge(truth, on="row_id")
assert (df_truth["row_id"].to_numpy() == df["row_id"].to_numpy()).all(), \
    "merge must preserve row order, or every mask below is misaligned"
heldout_truth = df_truth[heldout_mask].reset_index(drop=True)

print(heldout_truth["anomaly_kind"].value_counts().to_string())
print(f"\nanomaly base rate in the held-out set: "
      f"{heldout_truth['is_anomaly'].mean():.4%}")

# %%
is_anom = heldout_truth["is_anomaly"].to_numpy().astype(bool)

rows = []
for p in [99.5, 99, 98, 95, 90, 80]:
    thr = np.percentile(abs_err, p)
    flagged = abs_err >= thr
    tp = int((flagged & is_anom).sum())
    rows.append({
        "percentile": p,
        "flagged": int(flagged.sum()),
        "true anomalies found": tp,
        "precision": tp / flagged.sum(),
        "recall": tp / is_anom.sum(),
        "lift over base rate": (tp / flagged.sum()) / is_anom.mean(),
    })
detector = pd.DataFrame(rows)
print(detector.to_string(index=False))

# %% [markdown]
# **It works.** At the top 1% the detector is right about 5.6 times out of 10,
# against a base rate of 1.83% — a lift of roughly **31×**. It recovers 31% of
# all injected anomalies, and going down to the top 5% recovers about 75%.
#
# It is also, just as importantly, **not** a solved problem: at the top 1%, more
# than 4 flagged rows in 10 are not injected anomalies at all. These are the
# **irreducible-noise** case from the caveat list — ordinary rows with genuinely
# ambiguous telemetry, where a large residual is unavoidable no matter how good
# the model is. A human triaging this queue would spend over 40% of their time
# on false leads. That is what the method costs, and it is worth knowing before
# you ship it.

# %%
fig, axes = plt.subplots(1, 2, figsize=(13, 4.4))

axes[0].plot(detector["percentile"], detector["precision"], "o-",
             color=C["blue"], label="precision",
             markeredgecolor=INK["surface"], markeredgewidth=1.5)
axes[0].plot(detector["percentile"], detector["recall"], "o-", color=C["orange"],
             label="recall", markeredgecolor=INK["surface"], markeredgewidth=1.5)
axes[0].axhline(is_anom.mean(), color=INK["muted"], linewidth=1.4)
axes[0].annotate("base rate (random guessing)", xy=(90, is_anom.mean()),
                 xytext=(0, 7), textcoords="offset points", fontsize=8.5,
                 color=INK["muted"])
finish(axes[0], "Detector quality vs percentile cut",
       xlabel="percentile threshold", ylabel="rate", legend=True)

bins = np.linspace(-1.3, 1.6, 70)
axes[1].hist(resid[~is_anom], bins=bins, color=C["blue"], alpha=0.85,
             edgecolor="none", label="normal windows")
axes[1].hist(resid[is_anom], bins=bins, color=C["orange"], alpha=0.95,
             edgecolor="none", label="injected anomalies")
axes[1].set_yscale("log")
finish(axes[1], "Where the injected anomalies actually live",
       xlabel="signed residual y − ŷ", ylabel="windows (log)", legend=True)

fig.tight_layout()
save_fig(fig, "17_anomaly_detector_quality")
plt.show()

# %% [markdown]
# ### The sign of the residual — and a warning hiding underneath it
#
# The injected anomalies appear in **both** tails, and the two kinds should be
# distinguishable by sign: a silent failure means the model said "fine" and
# reality said "incident" (residual $\approx +1$), while a sensor glitch means
# the model panicked and nothing happened (residual $\approx -1$).
#
# Let us check that against both models, because it turns out to matter which
# model you attach the detector to.

# %%
kind_series = heldout_truth["anomaly_kind"]
rows = []
for model in (model_a, model_b):
    r = y_heldout - model.predict(X_heldout_frame)
    cut = np.percentile(np.abs(r), 99)
    for kind in ["none", "silent_failure", "sensor_glitch"]:
        m = (kind_series == kind).to_numpy()
        rows.append({
            "model": model.name.split(" · ")[0],
            "kind": kind,
            "n": int(m.sum()),
            "median resid": float(np.median(r[m])),
            "min": float(r[m].min()),
            "max": float(r[m].max()),
            "caught @ top 1%": int(((np.abs(r) >= cut) & m).sum()),
        })
print(pd.DataFrame(rows).to_string(index=False))

# %% [markdown]
# **Silent failures behave as predicted** under both models: median residual
# around +0.90 (model A) and +0.95 (model B), clustered near the top of the
# range. Not *quite* always positive — model B's minimum is −0.12 — but close.
#
# **Sensor glitches split the two models apart.** Under model A they sit cleanly
# in the negative tail, every one of them, spanning roughly [−1.19, −0.25].
# Under model B their residuals are scattered across **both** signs,
# [−1.50, +1.15].
#
# Look carefully at what that does and does not cost. Model B still *flags* just
# as many glitches as model A at the top 1% (8 of 45 versus 7), and it flags far
# more silent failures (37 versus 24). By raw detection, model B is the better
# detector. What it has lost is the ability to tell you **which kind** of
# anomaly it found: with residuals on both sides of zero, "negative residual ⇒
# broken sensor" stops working as a routing rule.
#
# The reason is worth more than the anomaly detector itself:

# %%
glitch = (kind_series == "sensor_glitch").to_numpy()
lat_idx = FEATURES.index("log_disk_latency")

z_latency = ((np.log(df.loc[heldout_mask, "disk_latency_ms"]).to_numpy()
              - model_b.feat_mean[lat_idx]) / model_b.feat_std[lat_idx])

# Find the column of model B's design matrix that holds log_disk_latency^2, and
# look at the values the *weight actually multiplies* -- model B re-standardises
# the design, so that is (z^2 - mean) / std, not z^2.
sq_col = len(FEATURES) + CROSS_PAIRS.index((lat_idx, lat_idx))
design_b = model_b.design_matrix(X_heldout_frame)

print("raw z-score of log_disk_latency")
print(f"  normal held-out windows : [{z_latency[~glitch].min():+.2f}, "
      f"{z_latency[~glitch].max():+.2f}]")
print(f"  sensor glitches         : [{z_latency[glitch].min():+.2f}, "
      f"{z_latency[glitch].max():+.2f}]")
print(f"  mean of z^2 on normal windows = {np.mean(z_latency[~glitch] ** 2):.2f}"
      "   (a squared standardised feature averages ~1, not 0)")

print(f"\nthe column model B's weight actually multiplies, (z^2 − mean)/std:")
print(f"  normal held-out windows : [{design_b[~glitch, sq_col].min():+.2f}, "
      f"{design_b[~glitch, sq_col].max():+.2f}]")
print(f"  sensor glitches         : [{design_b[glitch, sq_col].min():+.2f}, "
      f"{design_b[glitch, sq_col].max():+.2f}]")
print(f"\n  -> glitches sit about "
      f"{design_b[glitch, sq_col].max() / design_b[~glitch, sq_col].max():.1f}x "
      "beyond the largest value the fit ever saw.")

# %% [markdown]
# Model B contains terms like $z_{\text{latency}}^2$. Squaring is the problem:
# it turns a feature that is 2× out of range into a term that is 4× out of
# range. A glitched window sits at $z \approx 7$ where ordinary windows reach
# $3.4$ — but on the *squared* column it lands several times beyond anything in
# the training data. The coefficient fitted there was never constrained by data
# at that magnitude, so multiplying it by a large number produces whatever it
# produces. Polynomials extrapolate *badly*: far from the data they shoot off in
# whichever direction their highest-order term happens to point.
#
# Three conclusions, in increasing order of importance:
#
# 1. **The anomaly detector inherits the behaviour of the model behind it.**
#    Residual-based detection is never a property of the data alone.
# 2. **"Better" is task-relative.** Model B wins on AUC by a statistically
#    significant margin (§10), and it detects marginally more anomalies — but it
#    can no longer tell you what kind. The metric you optimised is not
#    automatically the metric you needed.
# 3. **Polynomial features are unreliable far from the training distribution** —
#    which is exactly where anomalies live, by definition. That is an
#    uncomfortable combination, and it is a genuine argument for doing anomaly
#    detection with a deliberately simple model, or with a method that does not
#    involve a supervised model at all.
#
# The operational reading of the sign is still correct, and still worth building
# in:
#
# * **Positive residual** — a monitoring blind spot. Escalate to a human.
# * **Negative residual** — a broken input. Fix the sensor, do not page anyone.
#
# A detector reporting only `|error|` merges both into one queue and throws that
# distinction away.
#
# ### The out-of-fold point, measured

# %%
train_resid = y_train - model_b.predict(X_train_frame)
print(f"{'':<26}{'99th pct |error|':>18}{'mean |error|':>15}")
print(f"{'train (in-sample)':<26}{np.percentile(np.abs(train_resid), 99):>18.4f}"
      f"{np.abs(train_resid).mean():>15.4f}")
print(f"{'held out':<26}{np.percentile(abs_err, 99):>18.4f}"
      f"{abs_err.mean():>15.4f}")
print(f"\ninflation of the held-out threshold vs train: "
      f"{np.percentile(abs_err, 99) / np.percentile(np.abs(train_resid), 99) - 1:+.2%}")

# %% [markdown]
# The gap is small here — 44 parameters against 12 000 rows barely overfits, so
# in-sample residuals are only slightly optimistic. Do not generalise from that:
# with a gradient-boosted model of a few thousand trees, training residuals go
# to nearly zero and a percentile threshold set on them would flag essentially
# nothing. **Always compute residuals out-of-fold.**

# %% [markdown]
# ## §12. Saving the artifacts
#
# Models are saved as plain dictionaries of arrays rather than pickled objects,
# so a backup can be loaded without this notebook's class definitions being
# importable. We verify the round-trip rather than assuming it.

# %%
def predict_from_bundle(bundle: dict, frame: pd.DataFrame) -> np.ndarray:
    """Reproduce a model's predictions using only numpy and the saved arrays."""
    X = (frame[bundle["feature_names"]].to_numpy(float)
         - bundle["feat_mean"]) / bundle["feat_std"]
    if bundle["use_cross"]:
        pairs = list(itertools.combinations_with_replacement(range(X.shape[1]), 2))
        X = np.hstack([X, np.column_stack([X[:, i] * X[:, j] for i, j in pairs])])
    if bundle["design_mean"] is not None:
        X = (X - bundle["design_mean"]) / bundle["design_std"]
    return X @ bundle["w"] + bundle["b"]


metadata = {
    "seed": RANDOM_SEED,
    "features": FEATURES,
    "best_threshold_on_val": BEST_THRESHOLD,
    "test_auc": {model_a.name: float(roc_auc_score(y_test, test_scores_a)),
                 model_b.name: float(roc_auc_score(y_test, test_scores_b))},
    "delong": {k: float(v) for k, v in result.items() if k != "cov"},
}

for model, fname in [(model_a, "model_a_plain.joblib"),
                     (model_b, "model_b_cross.joblib")]:
    bundle = model.to_bundle()
    joblib.dump(bundle, MODEL_DIR / fname)
    reloaded = joblib.load(MODEL_DIR / fname)
    assert np.allclose(predict_from_bundle(reloaded, X_test_frame),
                       model.predict(X_test_frame))
    print(f"saved + verified round-trip: {fname}")

with open(MODEL_DIR / "metadata.json", "w") as fh:
    json.dump(metadata, fh, indent=2)
print("saved metadata.json")

# %% [markdown]
# ## Recap
#
# What we did, and the one sentence worth keeping from each part:
#
# 1. **Two questions.** "Did it change?" and "what's anomalous?" — both need an
#    interval, not a point. A placebo comparison is what makes a p-value
#    trustworthy.
# 2. **Look at the data first.** Marginal shapes told us to log three features
#    before any model existed. Conditional distributions told us which features
#    carry signal.
# 3. **The normal equation** $\hat w = (X^\top X)^{-1}X^\top y$ is exact, and
#    unusable at scale: $O(d^3)$, and silently catastrophic when features are
#    collinear.
# 4. **Gradient descent** costs $O(nd)$ per step, converges for
#    $\eta < 2/L_{\max}$, and generalises to every model that has no closed
#    form — which is all of them.
# 5. **Cross-features** let a linear model express interactions. Ours bought
#    +0.024 AUC — and cost us the ability to tell one anomaly type from another
#    by residual sign (§11).
# 6. **Thresholds are a business decision.** Accuracy is nearly useless under
#    imbalance; precision and recall trade off; pick the threshold on
#    validation.
# 7. **AUC = P(random positive ranks above random negative)**, computable three
#    equivalent ways, and blind to calibration.
# 8. **Every test-set metric is a random variable.** Random halves of our test
#    set differ by as much as the gap between our two models.
# 9. **The bootstrap** gives an interval for any statistic, with no formula.
# 10. **DeLong** does it in closed form for AUC — and correctly credits the
#     covariance that "do the error bars overlap?" throws away.
# 11. **Residual-percentile anomaly detection works** (31× lift here), but a
#     percentile can never tell you *how many* anomalies exist, the sign of the
#     residual separates two failure modes that need opposite responses, and the
#     detector inherits the blind spots of whatever model you hang it on.
#
# ### Where the remaining error lives
#
# Our best model reaches 0.863 test AUC. The generator's metadata records the
# Bayes-optimal AUC — the ceiling no telemetry-based model can pass.
#
# Comparing the two is a trap, and it is worth walking into it deliberately,
# because it is the same mistake as every other one in this lesson: **comparing
# two numbers computed on different populations.** The recorded ceiling is
# measured on *clean rows only* (on the injected rows the true probability was
# forced, so no ceiling is defined there). Our 0.863 is measured on *all* test
# rows. So we must restrict both to the same population before subtracting.

# %%
with open(DATA_DIR / "dgp_metadata.json") as fh:
    dgp = json.load(fh)

clean_test = (heldout_truth.loc[
    (df_truth[heldout_mask]["split"] == "test").to_numpy(), "anomaly_kind"
] == "none").to_numpy()

print("measured on ALL test rows")
print(f"  model A (plain)          : {roc_auc_score(y_test, test_scores_a):.4f}")
print(f"  model B (cross-features) : {roc_auc_score(y_test, test_scores_b):.4f}")

print("\nmeasured on CLEAN test rows only — the like-for-like comparison")
print(f"  model A (plain)          : "
      f"{roc_auc_score(y_test[clean_test], test_scores_a[clean_test]):.4f}")
print(f"  model B (cross-features) : "
      f"{roc_auc_score(y_test[clean_test], test_scores_b[clean_test]):.4f}")
print(f"  Bayes-optimal ceiling    : {dgp['bayes_auc_clean_rows']:.4f}")

naive = dgp["bayes_auc_clean_rows"] - roc_auc_score(y_test, test_scores_b)
real = (dgp["bayes_auc_clean_rows"]
        - roc_auc_score(y_test[clean_test], test_scores_b[clean_test]))
print(f"\n  apparent gap (the trap)  : {naive:.4f} AUC")
print(f"  genuine gap              : {real:.4f} AUC")
print(f"  -> {1 - real / naive:.0%} of the apparent gap is not modelling error at all")

# %% [markdown]
# That changes the conclusion completely.
#
# Measured honestly, model B is within **~0.005 AUC** of the best any
# telemetry-based model could possibly do on the rows that follow the
# mechanism. It is very nearly optimal. Almost all of the apparent 0.03 shortfall
# is the injected anomalies — rows that are *unpredictable by construction*, and
# no amount of modelling will recover them.
#
# That is the difference between a productive week and a wasted quarter. Facing
# the naive number you would go looking for a better model; facing the real one
# you would go looking for **a better feature** — something that actually
# observes the config regression — or accept the ceiling and move on.
#
# Lesson 2 still has work to do: the true mechanism is *logistic* and we fitted a
# *linear* approximation, so logistic regression should close some of that last
# 0.005, and regularisation fixes §3's collinearity problem properly. But the
# honest headline is that **the model is close to done and the data is not** —
# and you can only see that by comparing like with like.
