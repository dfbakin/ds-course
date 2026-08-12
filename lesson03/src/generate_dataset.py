"""
Generate the synthetic dataset for Lesson 03.

Story
-----
Lesson 1 watched one service on 8 nodes.  The platform team now runs a
**global fleet**: ~180 clusters spread over 10 regions, six instance types, a
dozen OS images, three deploy channels.  One row = one node-day health
snapshot.  Two questions:

* ``power_draw_watts`` -- regression.  Power is linear in a *sparse* subset of
  the numerics (cpu_util, request_rate_rps, queue_depth carry real
  coefficients; the other five numerics have TRUE coefficient exactly 0),
  plus instance-type offsets and Gaussian noise.  cpu_util and mem_pressure
  are correlated, which is what makes ridge-vs-lasso interesting.
* ``incident`` -- classification, ~22% positive.  The mechanism mixes linear
  main effects, categorical offsets (including a hierarchical
  cluster-within-region effect over ~180 clusters), and nonlinearities that
  trees can reach and linear models cannot.

Why synthetic and not a classic dataset
---------------------------------------
The lesson's model ladder (numeric logreg -> +OHE -> tree -> CatBoost ->
CatBoost+Optuna) only teaches something if every rung is *reliably* better
than the previous one, by a margin that is visible but honest.  No public
dataset gives you control over those margins.  Here the gaps are dials:

1. Categorical offsets (region, channel, instance, OS, and a 180-level
   cluster effect) put >= 0.02 AUC between the numeric-only and the OHE
   logistic regression -- the "categoricals matter" rung.
2. Nonlinear terms (a latency threshold amplified in risky channels, a
   cpu x mem interaction, channel-dependent error-rate slopes) plus the
   high-cardinality cluster feature put >= 0.02 AUC between the OHE logistic
   regression and CatBoost -- the "native categoricals + trees" rung.
3. A handful of clusters appear in val/test but NOT in train, so the
   unknown-category problem is real, not hypothetical.
4. The regression target has a known sparse support, so lasso's variable
   selection can be *scored* against ground truth instead of admired.

Design
------
Numeric features are drawn through a **Gaussian copula** (as in lesson 1):
sample a correlated multivariate normal, push it through the normal CDF to
get uniforms, then through each feature's own inverse CDF.  Marginals are
visibly different shapes; the correlation structure is specified exactly.

Categorical structure: each row belongs to one of ``N_CLUSTERS`` clusters;
each cluster belongs to exactly one region (nesting is by construction).
Cluster sizes are long-tailed (Gamma weights), so some clusters have < 25
rows.  Incident risk gets a per-cluster random effect on top of the region
offset -- a textbook hierarchical effect.

The label comes from a logistic model on the *standardised modelling
representation* (heavy tails logged).  The nonlinear terms come in the three
families named in the lesson -- threshold, numeric x numeric interaction,
categorical x numeric interaction -- each with a coarse, teachable instance
(the p85 latency threshold, cpu x mem, channel-modulated error slope) plus a
long tail of individually-small instances (a cpu x mem x queue 3-way term and
per-region / per-instance / per-OS / per-cluster slope heterogeneity).  The
coarse instances are what the lesson reads off feature importances; the fine
tail is what makes hyperparameter tuning genuinely pay (see DATA_CARD.md
"Calibration").

Everything is seeded.  Re-running this script reproduces the CSV byte for
byte.

Usage
-----
    python src/generate_dataset.py [--out-dir DATA_DIR]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------

SEED = 20260808

N_ROWS = 30_000
TRAIN_ROWS, VAL_ROWS, TEST_ROWS = 18_000, 6_000, 6_000

# The eight numeric features as they are *stored* in the CSV.
RAW_FEATURES = [
    "cpu_util",
    "mem_pressure",
    "disk_latency_ms",
    "request_rate_rps",
    "error_rate_pct",
    "queue_depth",
    "cache_hit_ratio",
    "node_age_days",
]

# The numeric features as the *incident mechanism* sees them (heavy tails
# logged).  The ladder models are fed the raw columns on purpose: part of
# what separates trees from linear models here is that trees are invariant
# to monotone transforms and linear models are not.
MODEL_FEATURES = [
    "cpu_util",
    "mem_pressure",
    "log_disk_latency",
    "log_request_rate",
    "log1p_error_rate",
    "queue_depth",
    "cache_hit_ratio",
    "log1p_node_age",
]

CATEGORICAL_FEATURES = [
    "region",
    "cluster_id",
    "instance_type",
    "deploy_channel",
    "os_image",
]

# ---- fleet topology -------------------------------------------------------

REGIONS = [
    "us-east", "us-west", "eu-west", "eu-central", "ap-south",
    "ap-northeast", "sa-east", "ca-central", "af-south", "me-central",
]
# Fleet share of each region.  Deliberately non-uniform: the EDA section
# wants believable base-rate differences, not a flat bar chart.
REGION_WEIGHTS = [0.16, 0.13, 0.15, 0.10, 0.12, 0.09, 0.07, 0.08, 0.05, 0.05]

N_CLUSTERS = 180
# Cluster sizes are proportional to Gamma(0.38) weights: a long tail of small
# clusters (many < 25 rows) and a few giants -- the regime where OHE
# estimates per-cluster dummies badly and ordered target statistics shine.
CLUSTER_SIZE_GAMMA_SHAPE = 0.33
CLUSTER_MIN_ROWS = 8  # every cluster exists in the data (cardinality = 180)

# A handful of clusters are held out of train entirely: they appear only in
# val/test.  This is the unknown-category problem made concrete.  Candidates
# are small clusters, so the split sizes stay intact.
N_UNSEEN_CLUSTERS = 6
UNSEEN_SIZE_RANGE = (10, 45)  # rows; eligibility window for holdout clusters

INSTANCE_TYPES = ["m6i.large", "m6i.2xlarge", "c6i.xlarge",
                  "c6i.4xlarge", "r6i.2xlarge", "t3.large"]
INSTANCE_WEIGHTS = [0.22, 0.16, 0.22, 0.10, 0.14, 0.16]

DEPLOY_CHANNELS = ["stable", "beta", "canary"]
DEPLOY_CHANNEL_WEIGHTS = [0.70, 0.20, 0.10]

OS_IMAGES = [
    "ubuntu-22.04-v1", "ubuntu-22.04-v2", "ubuntu-22.04-v3",
    "ubuntu-24.04-v1", "ubuntu-24.04-v2", "ubuntu-24.04-v3",
    "debian-12-v1", "debian-12-v2", "debian-12-v3",
    "al2023-v1", "al2023-v2", "al2023-v3",
]
OS_IMAGE_WEIGHTS = [0.10, 0.12, 0.14, 0.06, 0.08, 0.10,
                    0.07, 0.09, 0.08, 0.05, 0.06, 0.05]

# ---- incident mechanism ---------------------------------------------------
#
# logit = INTERCEPT + sum(beta_i * z_i)                      (numeric mains)
#       + region + cluster-within-region + channel
#         + instance + os                                    (categorical offsets)
#       + threshold + cpu*mem + cpu*mem*queue
#         + channel-dependent err slope                      (coarse nonlinear)
#       + region/instance/os slope tweaks
#         + per-cluster random slopes                        (fine nonlinear)
#
# All betas act on *standardised model features* (see MODEL_FEATURES).

BETA_INCIDENT = {
    "cpu_util": 0.26,
    "mem_pressure": 0.08,
    "log_disk_latency": 0.22,
    "log_request_rate": 0.06,
    "log1p_error_rate": 0.36,
    "queue_depth": 0.11,
    "cache_hit_ratio": -0.17,
    "log1p_node_age": 0.05,
}

# Tuned for an incident rate inside 20-24% given every other effect below.
INTERCEPT_INCIDENT = -2.70

# Moderate region base-rate offsets (sum to 0 by construction).
REGION_INCIDENT_OFFSETS = {
    "us-east": -0.10, "us-west": 0.05, "eu-west": -0.22, "eu-central": 0.12,
    "ap-south": 0.30, "ap-northeast": -0.05, "sa-east": 0.20,
    "ca-central": -0.28, "af-south": 0.15, "me-central": -0.17,
}

# Hierarchical cluster effect *on top of* the region offset, with a
# heavy-tailed prior: most clusters sit near their region's base rate
# (Normal(0, CLUSTER_EFFECT_SD)), but a minority of *small* clusters are
# genuinely rotten or golden (Normal(0, CLUSTER_EXTREME_SD)) -- in a real
# fleet the special-purpose weirdness lives in the small clusters.  The
# base effect drives the numeric-only -> +OHE gap (real signal a numeric
# model cannot see, spread over 180 levels); the small-and-extreme minority
# is deliberately hard: too few rows to estimate cleanly, so lightly
# regularised models chase noise there.  That is where the default-vs-tuned
# CatBoost gap comes from (see DATA_CARD.md "Calibration").
CLUSTER_EFFECT_SD = 0.55
CLUSTER_EXTREME_FRAC = 0.45   # among clusters below the p75 size
CLUSTER_EXTREME_SD = 3.4

# Strong channel effect (the "canary releases are risky" story).
DEPLOY_CHANNEL_OFFSETS = {"stable": 0.0, "beta": 0.55, "canary": 1.05}

INSTANCE_INCIDENT_OFFSETS = {
    "m6i.large": -0.12, "m6i.2xlarge": 0.06, "c6i.xlarge": 0.02,
    "c6i.4xlarge": 0.25, "r6i.2xlarge": -0.18, "t3.large": 0.10,
}

# Mild OS-image effects (sum to 0): visible in a grouped bar chart with CIs,
# not decisive for any model.
OS_IMAGE_OFFSETS = {
    "ubuntu-22.04-v1": 0.04, "ubuntu-22.04-v2": -0.02, "ubuntu-22.04-v3": 0.10,
    "ubuntu-24.04-v1": -0.06, "ubuntu-24.04-v2": 0.02, "ubuntu-24.04-v3": -0.12,
    "debian-12-v1": 0.08, "debian-12-v2": 0.00, "debian-12-v3": -0.08,
    "al2023-v1": 0.12, "al2023-v2": -0.05, "al2023-v3": -0.03,
}

# Nonlinear terms -- the part of the mechanism no linear model can reach.
# Sizes are calibrated so the fitted OHE-logreg -> CatBoost test-AUC gap is
# >= 0.02 (see DATA_CARD.md "Calibration").
#
# (a) threshold: disk latency above ~p85 of its marginal adds risk, and the
#     jump is twice as large on beta/canary nodes.
DISK_LATENCY_THRESHOLD_MS = 14.1  # = lognorm(s=0.55, scale=8).ppf(0.85)
THRESHOLD_BASE = 0.85
THRESHOLD_RISKY_BOOST = 0.90      # extra jump when channel != stable
# (b) numeric x numeric interaction.
CPU_MEM_INTERACTION = 0.75        # on z_cpu * z_mem
CPU_MEM_QUEUE_INTERACTION = 0.90  # on z_cpu * z_mem * z_queue (3-way)
# (c) categorical x numeric interaction: risky channels amplify the
#     error-rate slope (on top of BETA_INCIDENT's main effect).
ERROR_SLOPE_EXTRA = {"stable": 0.0, "beta": 0.45, "canary": 0.80}
# (c') fine-grained slope heterogeneity, same cat x num family as (c) but
#     spread thin across many levels: each region bends the cpu slope a
#     little, each instance type bends the queue slope a little.  Individually
#     tiny, collectively real -- this is the "long tail of small interactions"
#     that keeps boosting improving after the first couple hundred trees,
#     and therefore what separates a tuned CatBoost from the default's
#     early-stopped fit.  Slopes are drawn Normal(0, SD) per level (seeded,
#     recorded in dgp_metadata.json).
REGION_CPU_SLOPE_SD = 0.42
INSTANCE_QUEUE_SLOPE_SD = 0.38
OS_ERROR_SLOPE_SD = 0.14
# Random *slopes* to go with the clusters' random intercepts: every cluster
# responds to load in its own way (hierarchical random slopes on six of the
# model features).  ~1000 individually-small real interactions that boosting
# can only hoover up slowly -- the late-training signal that separates a
# generously-budgeted tuned CatBoost from the default's 500 fast iterations.
CLUSTER_SLOPE_SDS = {
    "log_disk_latency": 0.55,
    "cpu_util": 0.40,
    "queue_depth": 0.35,
    "log1p_error_rate": 0.35,
    "mem_pressure": 0.30,
    "cache_hit_ratio": 0.30,
}

# ---- power draw (regression target) ---------------------------------------
#
# power = BASE + 145*cpu_util + 0.045*request_rate + 2.8*queue_depth
#       + instance offset + Normal(0, 16).
# The other five numerics have TRUE coefficient exactly 0 -- that zero *is*
# the ground truth lasso must recover.  mem_pressure is the trap: it is
# correlated with cpu_util (rho ~ 0.6) yet truly inert.

POWER_BASE_WATTS = 205.0
POWER_COEF = {  # raw units: W per unit of the raw feature
    "cpu_util": 145.0,
    "request_rate_rps": 0.045,
    "queue_depth": 2.8,
}
POWER_TRUE_SUPPORT = list(POWER_COEF)
POWER_INSTANCE_OFFSETS = {
    "m6i.large": -20.0, "m6i.2xlarge": 14.0, "c6i.xlarge": -6.0,
    "c6i.4xlarge": 38.0, "r6i.2xlarge": 20.0, "t3.large": -30.0,
}
POWER_NOISE_SD = 16.0

# ---- copula correlation structure (RAW_FEATURES order) --------------------

CORRELATIONS = {
    ("cpu_util", "mem_pressure"): 0.62,       # the ridge-vs-lasso pair
    ("cpu_util", "request_rate_rps"): 0.45,
    ("request_rate_rps", "queue_depth"): 0.50,
    ("disk_latency_ms", "queue_depth"): 0.30,
    ("disk_latency_ms", "error_rate_pct"): 0.30,
    ("cache_hit_ratio", "request_rate_rps"): -0.30,
    ("mem_pressure", "queue_depth"): 0.20,
    ("error_rate_pct", "cpu_util"): 0.15,
    ("node_age_days", "disk_latency_ms"): 0.20,
    ("node_age_days", "error_rate_pct"): 0.15,
}


# --------------------------------------------------------------------------
# Small helpers
# --------------------------------------------------------------------------


def _largest_remainder(total: int, weights) -> np.ndarray:
    """Split `total` into integers proportional to `weights`, exactly."""
    exact = np.asarray(weights, dtype=float)
    exact = exact * total / exact.sum()
    base = np.floor(exact).astype(int)
    short = total - int(base.sum())
    order = np.argsort(-(exact - base), kind="stable")
    base[order[:short]] += 1
    return base


def build_correlation_matrix() -> np.ndarray:
    """Assemble the latent correlation matrix and make it positive definite."""
    d = len(RAW_FEATURES)
    idx = {name: i for i, name in enumerate(RAW_FEATURES)}
    corr = np.eye(d)
    for (a, b), rho in CORRELATIONS.items():
        corr[idx[a], idx[b]] = rho
        corr[idx[b], idx[a]] = rho

    eigvals, eigvecs = np.linalg.eigh(corr)
    if eigvals.min() < 1e-6:
        eigvals = np.clip(eigvals, 1e-6, None)
        corr = eigvecs @ np.diag(eigvals) @ eigvecs.T
        scale = np.sqrt(np.diag(corr))
        corr = corr / np.outer(scale, scale)
        corr = (corr + corr.T) / 2.0
        np.fill_diagonal(corr, 1.0)
    return corr


def mixture_ppf(u: np.ndarray, components, weights, lo: float, hi: float,
                n_grid: int = 200_001) -> np.ndarray:
    """Inverse CDF of a finite mixture, by inverting a fine CDF grid."""
    grid = np.linspace(lo, hi, n_grid)
    cdf = np.zeros_like(grid)
    for w, comp in zip(weights, components):
        cdf += w * comp.cdf(grid)
    cdf = np.maximum.accumulate(cdf)
    cdf += np.arange(cdf.size) * 1e-12
    return np.interp(np.clip(u, cdf[0], cdf[-1]), cdf, grid)


def marginal_transform(name: str, u: np.ndarray) -> np.ndarray:
    """Push uniforms through one feature's inverse CDF.

    Each marginal looks like the physical quantity it stands for *and* gives
    the EDA panel a different shape to talk about.
    """
    if name == "cpu_util":
        # Bounded, mildly right-skewed.
        return stats.beta.ppf(u, a=4.5, b=3.2)
    if name == "mem_pressure":
        # Bounded, near-symmetric.  Correlated with cpu_util via the copula,
        # yet truly inert for power draw -- the lasso trap.
        return stats.beta.ppf(u, a=3.0, b=3.5)
    if name == "disk_latency_ms":
        # Log-normal: the classic heavy right tail.
        return stats.lognorm.ppf(u, s=0.55, scale=8.0)
    if name == "request_rate_rps":
        # Bimodal: night traffic and day traffic.
        return mixture_ppf(
            u,
            components=[
                stats.lognorm(s=0.35, scale=120.0),
                stats.lognorm(s=0.40, scale=700.0),
            ],
            weights=[0.42, 0.58],
            lo=1.0,
            hi=8000.0,
        )
    if name == "error_rate_pct":
        # Gamma with shape < 1: a spike at zero and a long tail.
        return stats.gamma.ppf(u, a=0.7, scale=0.9)
    if name == "queue_depth":
        # Discrete counts, over-dispersed relative to Poisson.
        return stats.nbinom.ppf(u, n=3.5, p=0.28)
    if name == "cache_hit_ratio":
        # Bounded and strongly *left*-skewed, piled up near 1.
        return stats.beta.ppf(u, a=12.0, b=1.6)
    if name == "node_age_days":
        # Mixture: a mostly-young fleet plus an old tail of legacy boxes.
        return mixture_ppf(
            u,
            components=[
                stats.lognorm(s=0.70, scale=140.0),
                stats.norm(loc=950.0, scale=220.0),
            ],
            weights=[0.72, 0.28],
            lo=1.0,
            hi=3000.0,
        )
    raise ValueError(f"unknown feature {name!r}")


def to_model_features(df: pd.DataFrame) -> pd.DataFrame:
    """Raw stored columns -> the representation the incident mechanism uses."""
    out = pd.DataFrame(index=df.index)
    out["cpu_util"] = df["cpu_util"]
    out["mem_pressure"] = df["mem_pressure"]
    out["log_disk_latency"] = np.log(df["disk_latency_ms"])
    out["log_request_rate"] = np.log(df["request_rate_rps"])
    out["log1p_error_rate"] = np.log1p(df["error_rate_pct"])
    out["queue_depth"] = df["queue_depth"]
    out["cache_hit_ratio"] = df["cache_hit_ratio"]
    out["log1p_node_age"] = np.log1p(df["node_age_days"])
    return out[MODEL_FEATURES]


def sigmoid(x: np.ndarray) -> np.ndarray:
    """Numerically stable logistic function."""
    out = np.empty_like(x, dtype=float)
    pos = x >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
    exp_x = np.exp(x[~pos])
    out[~pos] = exp_x / (1.0 + exp_x)
    return out


# --------------------------------------------------------------------------
# Fleet topology
# --------------------------------------------------------------------------


def build_cluster_structure(rng: np.random.Generator):
    """Clusters nested in regions, with long-tailed sizes.

    Returns (cluster_ids, cluster_region, cluster_sizes, cluster_effects,
    unseen_cluster_indices).  Cluster ids look like ``eu-west-c07`` so the
    nesting is readable in the CSV itself.
    """
    region_rows = _largest_remainder(N_ROWS, REGION_WEIGHTS)
    clusters_per_region = _largest_remainder(N_CLUSTERS, REGION_WEIGHTS)

    cluster_ids: list[str] = []
    cluster_region: list[str] = []
    sizes_all: list[int] = []
    for r_i, region in enumerate(REGIONS):
        n_c = int(clusters_per_region[r_i])
        weights = rng.gamma(CLUSTER_SIZE_GAMMA_SHAPE, 1.0, size=n_c)
        sizes = _largest_remainder(int(region_rows[r_i]), weights)
        # Clamp: every cluster keeps at least CLUSTER_MIN_ROWS rows, taken
        # from the region's largest cluster (which can spare them).
        while sizes.min() < CLUSTER_MIN_ROWS:
            i, j = int(np.argmin(sizes)), int(np.argmax(sizes))
            take = min(CLUSTER_MIN_ROWS - sizes[i], sizes[j] - CLUSTER_MIN_ROWS)
            if take <= 0:
                raise RuntimeError("cannot satisfy CLUSTER_MIN_ROWS")
            sizes[i] += take
            sizes[j] -= take
        for c_i in range(n_c):
            cluster_ids.append(f"{region}-c{c_i:02d}")
            cluster_region.append(region)
            sizes_all.append(int(sizes[c_i]))

    cluster_sizes = np.asarray(sizes_all)
    base_effects = rng.normal(0.0, CLUSTER_EFFECT_SD, size=N_CLUSTERS)
    small = cluster_sizes < np.percentile(cluster_sizes, 75)
    extreme_mask = (rng.random(N_CLUSTERS) < CLUSTER_EXTREME_FRAC) & small
    extreme_effects = rng.normal(0.0, CLUSTER_EXTREME_SD, size=N_CLUSTERS)
    cluster_effects = np.where(extreme_mask, extreme_effects, base_effects)

    eligible = np.flatnonzero(
        (cluster_sizes >= UNSEEN_SIZE_RANGE[0])
        & (cluster_sizes <= UNSEEN_SIZE_RANGE[1]))
    if eligible.size < N_UNSEEN_CLUSTERS:
        raise RuntimeError(
            f"only {eligible.size} clusters in the {UNSEEN_SIZE_RANGE} size "
            "window; retune CLUSTER_SIZE_GAMMA_SHAPE")
    unseen = np.sort(rng.choice(eligible, size=N_UNSEEN_CLUSTERS, replace=False))
    return cluster_ids, cluster_region, cluster_sizes, cluster_effects, unseen


# --------------------------------------------------------------------------
# Splits
# --------------------------------------------------------------------------


def assign_splits(y: np.ndarray, unseen_row_mask: np.ndarray,
                  rng: np.random.Generator) -> np.ndarray:
    """Materialised train/val/test split, stratified on `incident`.

    Rows of held-out clusters go to val/test only (that is the point of the
    holdout); every other row is then allocated so that each split hits its
    exact size and an incident count within one row of perfect stratification.
    """
    split = np.full(len(y), "", dtype=object)

    hold = rng.permutation(np.flatnonzero(unseen_row_mask))
    half = hold.size // 2
    split[hold[:half]] = "val"
    split[hold[half:]] = "test"

    sizes = {"train": TRAIN_ROWS, "val": VAL_ROWS, "test": TEST_ROWS}
    names = list(sizes)

    # Target positives per split: proportional allocation, exact in total.
    pos_total = int(y.sum())
    targets = dict(zip(names, _largest_remainder(
        pos_total, [sizes[s] / len(y) for s in names])))

    need = {1: {}, 0: {}}
    for s in names:
        in_s = split == s
        held_pos = int(y[in_s].sum())
        held_neg = int(in_s.sum()) - held_pos
        need[1][s] = targets[s] - held_pos
        need[0][s] = (sizes[s] - targets[s]) - held_neg
        if need[1][s] < 0 or need[0][s] < 0:
            raise RuntimeError("holdout clusters overflow a split stratum")

    for label in (1, 0):
        pool = rng.permutation(np.flatnonzero((y == label) & (split == "")))
        start = 0
        for s in names:
            split[pool[start:start + need[label][s]]] = s
            start += need[label][s]
        if start != pool.size:
            raise RuntimeError("split allocation does not add up")
    return split


# --------------------------------------------------------------------------
# Assembly
# --------------------------------------------------------------------------


def generate(seed: int = SEED) -> tuple[pd.DataFrame, dict]:
    rng = np.random.default_rng(seed)

    # ---- fleet topology --------------------------------------------------
    (cluster_ids, cluster_region, cluster_sizes,
     cluster_effects, unseen) = build_cluster_structure(rng)

    row_cluster = np.repeat(np.arange(N_CLUSTERS), cluster_sizes)
    row_cluster = row_cluster[rng.permutation(N_ROWS)]

    channel_idx = rng.choice(len(DEPLOY_CHANNELS), size=N_ROWS,
                             p=DEPLOY_CHANNEL_WEIGHTS)
    instance_idx = rng.choice(len(INSTANCE_TYPES), size=N_ROWS,
                              p=INSTANCE_WEIGHTS)
    os_idx = rng.choice(len(OS_IMAGES), size=N_ROWS, p=OS_IMAGE_WEIGHTS)

    region = np.array([cluster_region[c] for c in row_cluster])
    cluster = np.array([cluster_ids[c] for c in row_cluster])
    channel = np.array(DEPLOY_CHANNELS, dtype=object)[channel_idx]
    instance = np.array(INSTANCE_TYPES, dtype=object)[instance_idx]
    os_image = np.array(OS_IMAGES, dtype=object)[os_idx]

    # ---- correlated numerics via the Gaussian copula ---------------------
    corr = build_correlation_matrix()
    chol = np.linalg.cholesky(corr)
    latent = rng.standard_normal((N_ROWS, len(RAW_FEATURES))) @ chol.T
    uniforms = stats.norm.cdf(latent)
    uniforms = np.clip(uniforms, 1e-9, 1 - 1e-9)
    data = {name: marginal_transform(name, uniforms[:, i])
            for i, name in enumerate(RAW_FEATURES)}

    df = pd.DataFrame(data)
    df.insert(0, "row_id", np.arange(N_ROWS))
    df.insert(1, "region", region)
    df.insert(2, "cluster_id", cluster)
    df.insert(3, "instance_type", instance)
    df.insert(4, "deploy_channel", channel)
    df.insert(5, "os_image", os_image)

    # ---- regression target ----------------------------------------------
    power = np.full(N_ROWS, POWER_BASE_WATTS)
    for name, coef in POWER_COEF.items():
        power = power + coef * df[name].to_numpy()
    power = power + np.array([POWER_INSTANCE_OFFSETS[t] for t in instance])
    power = power + rng.normal(0.0, POWER_NOISE_SD, size=N_ROWS)
    df["power_draw_watts"] = power

    # ---- classification target ------------------------------------------
    model_df = to_model_features(df)
    standardiser = {
        name: (float(model_df[name].mean()), float(model_df[name].std(ddof=0)))
        for name in MODEL_FEATURES
    }
    z = {name: (model_df[name].to_numpy() - standardiser[name][0])
         / standardiser[name][1] for name in MODEL_FEATURES}

    logit = np.full(N_ROWS, INTERCEPT_INCIDENT)
    for name, coef in BETA_INCIDENT.items():
        logit += coef * z[name]

    logit_linear = logit.copy()  # mains + categorical offsets, no nonlinear
    cat_offset = (
        np.array([REGION_INCIDENT_OFFSETS[r] for r in region])
        + cluster_effects[row_cluster]
        + np.array([DEPLOY_CHANNEL_OFFSETS[c] for c in channel])
        + np.array([INSTANCE_INCIDENT_OFFSETS[t] for t in instance])
        + np.array([OS_IMAGE_OFFSETS[o] for o in os_image])
    )
    logit += cat_offset
    logit_linear += cat_offset

    risky = (channel != "stable")
    over_threshold = df["disk_latency_ms"].to_numpy() > DISK_LATENCY_THRESHOLD_MS
    logit += over_threshold * (THRESHOLD_BASE + THRESHOLD_RISKY_BOOST * risky)
    logit += CPU_MEM_INTERACTION * z["cpu_util"] * z["mem_pressure"]
    logit += (CPU_MEM_QUEUE_INTERACTION * z["cpu_util"] * z["mem_pressure"]
              * z["queue_depth"])
    logit += (np.array([ERROR_SLOPE_EXTRA[c] for c in channel])
              * z["log1p_error_rate"])

    # (c') the long tail of small cat x num interactions.
    region_cpu_slope = {r: s for r, s in zip(
        REGIONS, rng.normal(0.0, REGION_CPU_SLOPE_SD, len(REGIONS)))}
    instance_queue_slope = {t: s for t, s in zip(
        INSTANCE_TYPES, rng.normal(0.0, INSTANCE_QUEUE_SLOPE_SD,
                                   len(INSTANCE_TYPES)))}
    os_error_slope = {o: s for o, s in zip(
        OS_IMAGES, rng.normal(0.0, OS_ERROR_SLOPE_SD, len(OS_IMAGES)))}
    logit += np.array([region_cpu_slope[r] for r in region]) * z["cpu_util"]
    logit += (np.array([instance_queue_slope[t] for t in instance])
              * z["queue_depth"])
    logit += (np.array([os_error_slope[o] for o in os_image])
              * z["log1p_error_rate"])
    cluster_slopes = {f: rng.normal(0.0, sd, N_CLUSTERS)
                      for f, sd in CLUSTER_SLOPE_SDS.items()}
    for f, slopes in cluster_slopes.items():
        logit += slopes[row_cluster] * z[f]

    p_true = sigmoid(logit)
    df["incident"] = rng.binomial(1, p_true)

    # ---- splits ----------------------------------------------------------
    unseen_row_mask = np.isin(row_cluster, unseen)
    df["split"] = assign_splits(df["incident"].to_numpy(), unseen_row_mask, rng)

    # ---- rounding, for a CSV that looks like real telemetry --------------
    df["cpu_util"] = df["cpu_util"].round(4)
    df["mem_pressure"] = df["mem_pressure"].round(4)
    df["disk_latency_ms"] = df["disk_latency_ms"].round(3)
    df["request_rate_rps"] = df["request_rate_rps"].round(2)
    df["error_rate_pct"] = df["error_rate_pct"].round(4)
    df["queue_depth"] = df["queue_depth"].astype(int)
    df["cache_hit_ratio"] = df["cache_hit_ratio"].round(4)
    df["node_age_days"] = df["node_age_days"].round(1)
    df["power_draw_watts"] = df["power_draw_watts"].round(2)

    # ---- diagnostics for the data card ----------------------------------
    from sklearn.metrics import roc_auc_score

    y = df["incident"].to_numpy()
    # Oracle AUCs: what a model that knew the TRUE coefficients could score.
    # They bracket the fitted ladder from above and make calibration cheap:
    #   numeric-mains oracle  ~ ceiling for model 1,
    #   linear oracle (mains + categorical offsets) ~ ceiling for model 2,
    #   bayes (full logit)    ~ ceiling for models 4/5.
    logit_numeric = np.full(N_ROWS, INTERCEPT_INCIDENT)
    for name, coef in BETA_INCIDENT.items():
        logit_numeric += coef * z[name]
    oracle_numeric_auc = float(roc_auc_score(y, logit_numeric))
    oracle_linear_auc = float(roc_auc_score(y, logit_linear))
    bayes_auc = float(roc_auc_score(y, logit))

    meta = {
        "seed": seed,
        "n_rows": int(N_ROWS),
        "raw_features": RAW_FEATURES,
        "model_features": MODEL_FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
        "incident_mechanism": {
            "intercept": INTERCEPT_INCIDENT,
            "beta": {k: float(v) for k, v in BETA_INCIDENT.items()},
            "region_offsets": REGION_INCIDENT_OFFSETS,
            "deploy_channel_offsets": DEPLOY_CHANNEL_OFFSETS,
            "instance_offsets": INSTANCE_INCIDENT_OFFSETS,
            "os_image_offsets": OS_IMAGE_OFFSETS,
            "cluster_effect_sd": CLUSTER_EFFECT_SD,
            "cluster_extreme_frac": CLUSTER_EXTREME_FRAC,
            "cluster_extreme_sd": CLUSTER_EXTREME_SD,
            "disk_latency_threshold_ms": DISK_LATENCY_THRESHOLD_MS,
            "threshold_base": THRESHOLD_BASE,
            "threshold_risky_boost": THRESHOLD_RISKY_BOOST,
            "cpu_mem_interaction": CPU_MEM_INTERACTION,
            "cpu_mem_queue_interaction": CPU_MEM_QUEUE_INTERACTION,
            "error_slope_extra": ERROR_SLOPE_EXTRA,
            "region_cpu_slope": {k: float(v)
                                 for k, v in region_cpu_slope.items()},
            "instance_queue_slope": {k: float(v)
                                     for k, v in instance_queue_slope.items()},
            "os_error_slope": {k: float(v)
                               for k, v in os_error_slope.items()},
        },
        "power_mechanism": {
            "base_watts": POWER_BASE_WATTS,
            # Explicit zeros: this dict IS the ground truth for the lasso
            # support-recovery check.
            "coefficients": {name: float(POWER_COEF.get(name, 0.0))
                             for name in RAW_FEATURES},
            "true_support": POWER_TRUE_SUPPORT,
            "instance_offsets": POWER_INSTANCE_OFFSETS,
            "noise_sd": POWER_NOISE_SD,
        },
        "standardiser": {k: [v[0], v[1]] for k, v in standardiser.items()},
        "clusters": {
            cluster_ids[i]: {
                "region": cluster_region[i],
                "size": int(cluster_sizes[i]),
                "incident_offset": float(cluster_effects[i]),
                "slopes": {f: float(cluster_slopes[f][i])
                           for f in CLUSTER_SLOPE_SDS},
            }
            for i in range(N_CLUSTERS)
        },
        "unseen_clusters": [cluster_ids[i] for i in unseen],
        "incident_rate": float(y.mean()),
        "oracle_numeric_auc": oracle_numeric_auc,
        "oracle_linear_auc": oracle_linear_auc,
        "bayes_auc": bayes_auc,
        "split_sizes": {k: int(v) for k, v in df["split"].value_counts().items()},
        "cluster_size_stats": {
            "min": int(cluster_sizes.min()),
            "median": float(np.median(cluster_sizes)),
            "max": int(cluster_sizes.max()),
            "n_below_25": int((cluster_sizes < 25).sum()),
        },
    }
    return df, meta


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path,
                        default=Path(__file__).resolve().parents[1] / "data")
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    df, meta = generate()

    public_cols = (["row_id"] + CATEGORICAL_FEATURES + RAW_FEATURES
                   + ["power_draw_watts", "incident", "split"])
    df[public_cols].to_csv(args.out_dir / "fleet_incidents.csv", index=False)

    with open(args.out_dir / "dgp_metadata.json", "w") as fh:
        json.dump(meta, fh, indent=2)

    print(f"wrote {len(df):,} rows to {args.out_dir}")
    print(f"  incident rate      : {meta['incident_rate']:.4f}")
    print(f"  oracle numeric AUC : {meta['oracle_numeric_auc']:.4f}")
    print(f"  oracle linear AUC  : {meta['oracle_linear_auc']:.4f}")
    print(f"  Bayes AUC          : {meta['bayes_auc']:.4f}")
    print(f"  splits             : {meta['split_sizes']}")
    print(f"  unseen clusters    : {meta['unseen_clusters']}")
    print(f"  cluster sizes      : {meta['cluster_size_stats']}")


if __name__ == "__main__":
    main()
