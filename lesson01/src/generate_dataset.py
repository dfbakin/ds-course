"""
Generate the synthetic dataset for Lesson 01.

Story
-----
We monitor a fleet of 8 nodes serving a web API.  Every 5 minutes each node
emits a telemetry window.  The target is whether that window contained an
*incident* (an SLA breach).  Halfway through the observation period a new
version is deployed, which is what makes the "was there a significant change?"
question concrete.

Why synthetic and not a classic dataset
---------------------------------------
The lesson needs four properties simultaneously, and no classic tabular
dataset has all four:

1. Marginals that are *visibly* different from one another (skewed, bimodal,
   discrete, bounded, symmetric) so the distribution section has something to
   say.
2. A controlled correlation structure, so covariation is real but not trivial.
3. Genuine pairwise interaction effects in the label mechanism, so that a
   second model built on cross-features x_i * x_j is *reliably but only
   slightly* better -- which is exactly the regime where the DeLong test is
   interesting rather than decorative.
4. Known ground-truth anomalies, so the residual-percentile anomaly detector in
   the last section can be *scored* instead of merely admired.

Design
------
Features are drawn through a **Gaussian copula**: sample a correlated
multivariate normal, push it through the normal CDF to get uniforms, then
through each feature's own inverse CDF.  This decouples "what shape is each
feature" from "how do features co-vary" -- we get arbitrary marginals with an
exactly specified rank-correlation structure.

The label comes from a logistic model on the *standardised modelling
representation* of the features, with main effects, six interaction terms and
one quadratic term.  A linear model on plain features can only capture the
main effects; the cross-feature model can reach the rest.  That gap is the
whole point of section 5.

Everything is seeded.  Re-running this script reproduces the CSVs byte for
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

SEED = 20260729

N_ROWS = 20_000
N_NODES = 8
DEPLOY_FRACTION = 0.55  # first 55% of each node's timeline is the old version

# Anomaly cohorts (see `inject_anomalies`)
N_SILENT_FAILURE = 240  # benign telemetry, incident anyway
N_SENSOR_GLITCH = 120  # alarming telemetry, no incident (~24 land in the test
# split, enough to characterise the cohort's residuals stably)

# The eight features as they are *stored* in the CSV.
RAW_FEATURES = [
    "cpu_util",
    "mem_pressure",
    "disk_latency_ms",
    "request_rate_rps",
    "error_rate_pct",
    "queue_depth",
    "cache_hit_ratio",
    "temperature_c",
]

# The eight features as they are *modelled* (heavy tails logged).  The notebook
# re-derives these from the raw columns during EDA; we need them here because
# the label mechanism is defined on them.
MODEL_FEATURES = [
    "cpu_util",
    "mem_pressure",
    "log_disk_latency",
    "log_request_rate",
    "log1p_error_rate",
    "queue_depth",
    "cache_hit_ratio",
    "temperature_c",
]

# Main effects of the label mechanism, on standardised model features.
BETA = {
    "cpu_util": 0.62,
    "mem_pressure": 0.45,
    "log_disk_latency": 0.70,
    "log_request_rate": 0.30,
    "log1p_error_rate": 0.85,
    "queue_depth": 0.40,
    "cache_hit_ratio": -0.55,
    "temperature_c": 0.22,
}

# Interaction / quadratic effects, on standardised model features.  These are
# invisible to a linear model on the plain features.
#
# Their size is deliberately calibrated: strong enough that the cross-feature
# model wins reliably, small enough that the win is *not* obvious by eye.
# Measured over four independent seeds these values give a test-AUC gap of
# +0.0204 +/- 0.0023 -- always positive, never dramatic.  That is the regime
# where a significance test earns its keep; with a +0.10 gap DeLong would be a
# formality and section 10 would teach nothing.
GAMMA = {
    ("cpu_util", "queue_depth"): 0.55,
    ("log_disk_latency", "log_request_rate"): 0.45,
    ("cache_hit_ratio", "log_request_rate"): -0.40,
    ("cpu_util", "mem_pressure"): 0.35,
    ("log1p_error_rate", "log_request_rate"): 0.30,
    ("cpu_util", "cpu_util"): 0.28,
}

INTERCEPT = -2.25  # tuned for a base incident rate near 25%

# Post-deploy drift, in units of the latent normal's standard deviation.
# Small on purpose: together with the anomalous cohort it moves the incident
# rate by under 3 points, which no one can eyeball off a bar chart.
POST_DEPLOY_SHIFT = {
    "disk_latency_ms": 0.060,
    "error_rate_pct": 0.045,
}

# Target correlation matrix of the latent normals, in RAW_FEATURES order.
CORRELATIONS = {
    ("cpu_util", "mem_pressure"): 0.55,
    ("cpu_util", "temperature_c"): 0.65,
    ("cpu_util", "request_rate_rps"): 0.45,
    ("request_rate_rps", "queue_depth"): 0.50,
    ("disk_latency_ms", "queue_depth"): 0.35,
    ("cache_hit_ratio", "request_rate_rps"): -0.30,
    ("error_rate_pct", "disk_latency_ms"): 0.30,
    ("mem_pressure", "queue_depth"): 0.20,
    ("temperature_c", "request_rate_rps"): 0.25,
    ("error_rate_pct", "cpu_util"): 0.15,
}


# --------------------------------------------------------------------------
# Copula machinery
# --------------------------------------------------------------------------


def build_correlation_matrix() -> np.ndarray:
    """Assemble the latent correlation matrix and make it positive definite."""
    d = len(RAW_FEATURES)
    idx = {name: i for i, name in enumerate(RAW_FEATURES)}
    corr = np.eye(d)
    for (a, b), rho in CORRELATIONS.items():
        corr[idx[a], idx[b]] = rho
        corr[idx[b], idx[a]] = rho

    # Nearest-PSD by eigenvalue clipping, then renormalise the diagonal to 1.
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
    """Inverse CDF of a finite mixture, by inverting a fine CDF grid.

    scipy has no closed-form ppf for a mixture, and we need one to push
    uniforms through the copula.  Tabulating the CDF on a dense grid and
    interpolating is accurate to well under the noise we add anyway.
    """
    grid = np.linspace(lo, hi, n_grid)
    cdf = np.zeros_like(grid)
    for w, comp in zip(weights, components):
        cdf += w * comp.cdf(grid)
    # np.interp needs a non-decreasing x; enforce strict monotonicity so that
    # flat regions of the CDF cannot produce ties.
    cdf = np.maximum.accumulate(cdf)
    cdf += np.arange(cdf.size) * 1e-12
    return np.interp(np.clip(u, cdf[0], cdf[-1]), cdf, grid)


def marginal_transform(name: str, u: np.ndarray) -> np.ndarray:
    """Push uniforms through one feature's inverse CDF.

    Each marginal is picked to look like the physical quantity it stands for
    *and* to give the histogram section a different shape to talk about.
    """
    if name == "cpu_util":
        # Bounded, mildly right-skewed: the everyday "bounded ratio" shape.
        return stats.beta.ppf(u, a=4.5, b=3.2)
    if name == "mem_pressure":
        # Bounded and near-symmetric.
        return stats.beta.ppf(u, a=3.0, b=3.5)
    if name == "disk_latency_ms":
        # Log-normal: the classic heavy right tail of a latency metric.
        return stats.lognorm.ppf(u, s=0.55, scale=8.0)
    if name == "request_rate_rps":
        # Bimodal: night traffic and day traffic are two different regimes.
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
    if name == "temperature_c":
        # The one boring, symmetric, Gaussian feature -- a useful contrast.
        return stats.norm.ppf(u, loc=46.0, scale=6.0)
    raise ValueError(f"unknown feature {name!r}")


def to_model_features(df: pd.DataFrame) -> pd.DataFrame:
    """Raw stored columns -> the representation the label mechanism uses.

    The notebook performs exactly these transforms in the EDA section, and it
    motivates them from the histograms rather than being told.
    """
    out = pd.DataFrame(index=df.index)
    out["cpu_util"] = df["cpu_util"]
    out["mem_pressure"] = df["mem_pressure"]
    out["log_disk_latency"] = np.log(df["disk_latency_ms"])
    out["log_request_rate"] = np.log(df["request_rate_rps"])
    out["log1p_error_rate"] = np.log1p(df["error_rate_pct"])
    out["queue_depth"] = df["queue_depth"]
    out["cache_hit_ratio"] = df["cache_hit_ratio"]
    out["temperature_c"] = df["temperature_c"]
    return out[MODEL_FEATURES]


# --------------------------------------------------------------------------
# Label mechanism
# --------------------------------------------------------------------------


def latent_logit(model_df: pd.DataFrame, standardiser: dict) -> np.ndarray:
    """The true log-odds: intercept + main effects + interactions."""
    z = {
        name: (model_df[name].to_numpy() - standardiser[name][0]) / standardiser[name][1]
        for name in MODEL_FEATURES
    }
    logit = np.full(len(model_df), INTERCEPT, dtype=float)
    for name, coef in BETA.items():
        logit += coef * z[name]
    for (a, b), coef in GAMMA.items():
        logit += coef * z[a] * z[b]
    return logit


def sigmoid(x: np.ndarray) -> np.ndarray:
    """Numerically stable logistic function."""
    out = np.empty_like(x, dtype=float)
    pos = x >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
    exp_x = np.exp(x[~pos])
    out[~pos] = exp_x / (1.0 + exp_x)
    return out


# --------------------------------------------------------------------------
# Anomalies
# --------------------------------------------------------------------------


def pick_sensor_glitches(df: pd.DataFrame, rng: np.random.Generator) -> np.ndarray:
    """Choose the rows whose disk-latency probe will be rewritten as garbage.

    A broken probe reports absurd values while the service is actually fine.
    The model screams, nothing happens.  These land in the *lower* tail of the
    signed residual, which is why section 11 does not stop at |error|.
    """
    post = np.flatnonzero((df["phase"] == "post_deploy").to_numpy())
    return rng.choice(post, size=N_SENSOR_GLITCH, replace=False)


def pick_silent_failures(df: pd.DataFrame, p_true: np.ndarray, exclude: np.ndarray,
                         rng: np.random.Generator) -> np.ndarray:
    """Choose the rows that will be incidents despite healthy telemetry.

    A config regression shipped with the new version: the window is an incident
    although every metric looks fine.

    The important design choice is that we do **not** synthesise new feature
    values for these rows.  An earlier version did, and it quietly broke the
    lesson: resampling gave the cohort a distinctive "every metric is
    unusually benign at once" signature, which is precisely the kind of
    conjunction a cross-feature model can learn.  Model 2 then won mostly by
    detecting anomalies rather than by capturing the interactions in the
    mechanism, which is not the story section 5 tells.

    So instead we *select* rows that already look benign (bottom 40% of true
    risk) and leave their telemetry untouched.  The cohort is then genuinely
    unpredictable from telemetry -- no model can beat another on it -- and it
    still lands in the upper residual tail, which is what section 11 needs.
    """
    post = (df["phase"] == "post_deploy").to_numpy()
    benign = p_true <= np.quantile(p_true, 0.40)
    eligible = np.flatnonzero(post & benign)
    eligible = np.setdiff1d(eligible, exclude)
    return rng.choice(eligible, size=N_SILENT_FAILURE, replace=False)


# --------------------------------------------------------------------------
# Assembly
# --------------------------------------------------------------------------


def generate(seed: int = SEED) -> tuple[pd.DataFrame, dict]:
    rng = np.random.default_rng(seed)

    # ---- timeline: 8 nodes x 2500 five-minute windows -------------------
    windows_per_node = N_ROWS // N_NODES
    start = pd.Timestamp("2026-05-04 00:00:00")
    offsets = pd.to_timedelta(np.arange(windows_per_node) * 5, unit="m")
    timestamps = np.tile((start + offsets).to_numpy(), N_NODES)
    node_ids = np.repeat([f"node-{i:02d}" for i in range(N_NODES)], windows_per_node)

    window_index = np.tile(np.arange(windows_per_node), N_NODES)
    deploy_at = int(windows_per_node * DEPLOY_FRACTION)
    phase = np.where(window_index < deploy_at, "pre_deploy", "post_deploy")

    # ---- correlated latent normals --------------------------------------
    corr = build_correlation_matrix()
    chol = np.linalg.cholesky(corr)
    latent = rng.standard_normal((N_ROWS, len(RAW_FEATURES))) @ chol.T

    # Post-deploy drift: shift the latent mean of two features.  The drift is
    # real, modest, and flows through to the incident rate -- which is what
    # section 1 goes looking for.
    post_mask = phase == "post_deploy"
    for name, shift in POST_DEPLOY_SHIFT.items():
        latent[post_mask, RAW_FEATURES.index(name)] += shift

    # ---- copula: normals -> uniforms -> arbitrary marginals --------------
    uniforms = stats.norm.cdf(latent)
    uniforms = np.clip(uniforms, 1e-9, 1 - 1e-9)

    data = {name: marginal_transform(name, uniforms[:, i])
            for i, name in enumerate(RAW_FEATURES)}

    df = pd.DataFrame(data)
    df.insert(0, "row_id", np.arange(N_ROWS))
    df.insert(1, "window_start", timestamps)
    df.insert(2, "node_id", node_ids)
    df.insert(3, "phase", phase)
    df["anomaly_kind"] = "none"

    # ---- sensor glitches rewrite telemetry, so they come first -----------
    # Their garbage readings are part of the observed world and therefore must
    # be present before we standardise anything.
    glitch = pick_sensor_glitches(df, rng)
    df.loc[df.index[glitch], "disk_latency_ms"] = rng.lognormal(np.log(340.0), 0.30,
                                                                size=glitch.size)
    df.loc[df.index[glitch], "error_rate_pct"] = rng.gamma(2.5, 3.0, size=glitch.size)

    # ---- labels ----------------------------------------------------------
    model_df = to_model_features(df)
    standardiser = {
        name: (float(model_df[name].mean()), float(model_df[name].std(ddof=0)))
        for name in MODEL_FEATURES
    }
    logit = latent_logit(model_df, standardiser)
    p_true = sigmoid(logit)

    df["incident"] = rng.binomial(1, p_true)

    # ---- now force the two anomalous cohorts' labels ---------------------
    silent = pick_silent_failures(df, p_true, exclude=glitch, rng=rng)
    df.loc[df.index[silent], "incident"] = rng.binomial(1, 0.95, size=silent.size)
    df.loc[df.index[silent], "anomaly_kind"] = "silent_failure"

    df.loc[df.index[glitch], "incident"] = 0
    df.loc[df.index[glitch], "anomaly_kind"] = "sensor_glitch"

    normal_mask = (df["anomaly_kind"] == "none").to_numpy()

    # ---- splits ----------------------------------------------------------
    # Stratified 60/20/20.  Materialised in the CSV so the notebook does not
    # depend on any library's splitting internals staying stable.
    split = np.empty(N_ROWS, dtype=object)
    y = df["incident"].to_numpy()
    for label in (0, 1):
        idx = np.flatnonzero(y == label)
        idx = rng.permutation(idx)
        n_train = int(round(0.60 * idx.size))
        n_val = int(round(0.20 * idx.size))
        split[idx[:n_train]] = "train"
        split[idx[n_train:n_train + n_val]] = "val"
        split[idx[n_train + n_val:]] = "test"
    df["split"] = split

    # ---- rounding, for a CSV that looks like real telemetry --------------
    df["cpu_util"] = df["cpu_util"].round(4)
    df["mem_pressure"] = df["mem_pressure"].round(4)
    df["disk_latency_ms"] = df["disk_latency_ms"].round(3)
    df["request_rate_rps"] = df["request_rate_rps"].round(2)
    df["error_rate_pct"] = df["error_rate_pct"].round(4)
    df["queue_depth"] = df["queue_depth"].astype(int)
    df["cache_hit_ratio"] = df["cache_hit_ratio"].round(4)
    df["temperature_c"] = df["temperature_c"].round(2)

    # ---- diagnostics for the data card ----------------------------------
    from sklearn.metrics import roc_auc_score

    # The ceiling is measured on the rows that actually follow the mechanism.
    # On the injected-anomaly rows the true probability is NOT p_true -- it was
    # forced to 0.95 / 0.0 -- so scoring p_true there would measure nothing.
    # Restricted to clean rows, `bayes_auc` is a genuine upper bound on what any
    # telemetry-based model can achieve, and the distance from it to our fitted
    # models is the honest "how much is left on the table" number.
    zs = {n: (model_df[n].to_numpy() - standardiser[n][0]) / standardiser[n][1]
          for n in MODEL_FEATURES}
    main_only = np.full(N_ROWS, INTERCEPT, dtype=float)
    for name, coef in BETA.items():
        main_only += coef * zs[name]

    bayes_auc = float(roc_auc_score(y[normal_mask], p_true[normal_mask]))
    main_only_auc = float(roc_auc_score(y[normal_mask], main_only[normal_mask]))

    meta = {
        "seed": seed,
        "n_rows": int(N_ROWS),
        "raw_features": RAW_FEATURES,
        "model_features": MODEL_FEATURES,
        "intercept": INTERCEPT,
        "beta": {k: float(v) for k, v in BETA.items()},
        "gamma": {f"{a}*{b}": float(v) for (a, b), v in GAMMA.items()},
        "standardiser": {k: [v[0], v[1]] for k, v in standardiser.items()},
        "post_deploy_shift": POST_DEPLOY_SHIFT,
        "incident_rate": float(y.mean()),
        "incident_rate_pre": float(y[~post_mask].mean()),
        "incident_rate_post": float(y[post_mask].mean()),
        # Both measured on non-anomalous rows only; see the comment above.
        "bayes_auc_clean_rows": bayes_auc,
        "main_effects_only_auc_clean_rows": main_only_auc,
        "n_silent_failure": int(N_SILENT_FAILURE),
        "n_sensor_glitch": int(N_SENSOR_GLITCH),
        "split_sizes": {k: int(v) for k, v in df["split"].value_counts().items()},
    }
    return df, meta


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path,
                        default=Path(__file__).resolve().parents[1] / "data")
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    df, meta = generate()

    public_cols = (["row_id", "window_start", "node_id", "phase"]
                   + RAW_FEATURES + ["incident", "split"])
    df[public_cols].to_csv(args.out_dir / "service_telemetry.csv", index=False)

    # The anomaly ground truth lives in a separate file: the notebook only
    # opens it in the final section, after the detector has already run.
    labels = df[["row_id", "anomaly_kind"]].copy()
    labels["is_anomaly"] = (labels["anomaly_kind"] != "none").astype(int)
    labels.to_csv(args.out_dir / "anomaly_ground_truth.csv", index=False)

    with open(args.out_dir / "dgp_metadata.json", "w") as fh:
        json.dump(meta, fh, indent=2)

    print(f"wrote {len(df):,} rows to {args.out_dir}")
    print(f"  incident rate      : {meta['incident_rate']:.4f}")
    print(f"  pre-deploy rate    : {meta['incident_rate_pre']:.4f}")
    print(f"  post-deploy rate   : {meta['incident_rate_post']:.4f}")
    print(f"  Bayes AUC (clean)  : {meta['bayes_auc_clean_rows']:.4f}")
    print(f"  main-effects (clean): {meta['main_effects_only_auc_clean_rows']:.4f}")
    print(f"  splits             : {meta['split_sizes']}")


if __name__ == "__main__":
    main()
