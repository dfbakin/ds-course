"""Generate `edge_gateway_logs.csv` — the dataset for home assignment 1.

Scenario
--------
An edge-gateway fleet: 24 instances across 4 regions, logged in 2-hour windows
for 120 days. The target is whether the window contained an **incident** (an SLA
breach). Four deployments went out during the first 60 days. The remaining 60
days are the "future horizon" the student does not touch until Part 5.

What the data has to support, task by task
------------------------------------------
Part 1  Realistic mess: missing values, a `-1` sentinel, duplicate rows.
Part 2  Four deployments whose effects are *deliberately mixed* — two clearly
        real, one null, and one that is significant on its own but does NOT
        survive a multiple-testing correction. That last one is the point of
        the exercise; a dataset where all four are significant teaches nothing.
Part 3  A baseline linear model on raw numeric columns that lands near 0.78 —
        good enough to be worth improving, bad enough to leave room.
Part 4  Headroom that is only reachable through feature engineering:
          * the true mechanism acts on LOGS of the heavy-tailed columns, so a
            log transform pays;
          * the categoricals carry signal that no numeric column encodes, so
            one-hot pays;
          * two interaction terms exist, so cross-features pay.
        Standardisation is deliberately NOT one of these: it cannot change an
        OLS fit at all, which the assignment makes the student verify.
Part 5  Concept drift across the horizon: the coefficient on `cache_hit_ratio`
        decays toward zero while `gc_pause_ms` takes over, plus covariate shift
        in traffic. A model trained on the development period therefore decays,
        and retraining on recent data recovers.

Calibration
-----------
Phase incident rates are not tuned by hand. For each phase we *solve* for the
intercept offset that produces the target rate exactly (bisection on the mean
of a sigmoid), so the Part 2 answers are locked in by construction.

Usage
-----
    python src/generate_logs.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

SEED = 20260815

# ---------------------------------------------------------------- geometry --
N_INSTANCES = 24
WINDOW_HOURS = 2
N_DAYS = 120
DEV_DAYS = 60                      # days 0-59 are the development period
WINDOWS_PER_DAY = 24 // WINDOW_HOURS
START = pd.Timestamp("2026-01-05 00:00:00")

DEPLOY_DAYS = [12, 24, 36, 48]     # four deployments inside the dev period
PHASES = ["A", "B", "C", "D", "E"]

# Target incident rate per phase. The *mix* is the pedagogy -- a dataset where
# all four deployments are significant teaches nothing. As realised on the
# shipped seed:
#   D1  A -> B   -0.0pp   p ~ 0.98   plainly null
#   D2  B -> C   +5.1pp   p < 1e-6   the bad deploy, unmissable
#   D3  C -> D   -4.6pp   p < 1e-5   the fix
#   D4  D -> E   +2.1pp   p ~ 0.040  significant ALONE, but not after
#                                    correcting for four tests -- which is the
#                                    whole point of Part 2c.
PHASE_RATES = {"A": 0.200, "B": 0.205, "C": 0.255, "D": 0.216, "E": 0.2244}

REGIONS = ["us-east-1", "us-west-2", "eu-central-1", "ap-south-1"]
INSTANCE_TYPES = ["m5.large", "c5.xlarge", "r5.2xlarge"]
SERVICE_TIERS = ["free", "pro", "enterprise"]

NUMERIC = ["cpu_util", "mem_util", "p99_latency_ms", "request_rate",
           "error_rate_pct", "queue_depth", "cache_hit_ratio", "gc_pause_ms",
           "active_connections", "disk_io_wait_ms"]

# ------------------------------------------------------------ true effects --
# Coefficients act on standardised, log-where-appropriate features. A model on
# RAW columns cannot see the log structure, which is the Part 4 headroom.
BETA = {
    "cpu_util": 0.55,
    "mem_util": 0.34,
    "log_p99_latency": 0.62,
    "log_request_rate": 0.26,
    "log1p_error_rate": 0.70,
    "queue_depth": 0.38,
    "cache_hit_ratio": -0.46,
    "log_gc_pause": 0.30,
    "log_active_connections": 0.18,
    "log_disk_io_wait": 0.22,
}

# Interactions -- reachable only with cross-features.
GAMMA = {
    ("cpu_util", "queue_depth"): 0.60,
    ("log_p99_latency", "log_request_rate"): 0.52,
}

# Categorical effects, in logits. Nothing in the numeric columns encodes these,
# so one-hot encoding is the only way to reach them -- which is what makes it
# the single biggest win available in Part 4 (+0.040 AUC, measured).
CAT_EFFECT = {
    "region": {"us-east-1": -0.18, "us-west-2": -0.09,
               "eu-central-1": 0.04, "ap-south-1": 1.04},
    "instance_type": {"m5.large": 0.40, "c5.xlarge": 0.00, "r5.2xlarge": -0.54},
    "service_tier": {"free": -0.76, "pro": 0.00, "enterprise": 0.99},
}

# Latent correlation structure of the numeric block.
CORRELATIONS = {
    ("cpu_util", "mem_util"): 0.50,
    ("cpu_util", "request_rate"): 0.44,
    ("cpu_util", "queue_depth"): 0.30,
    ("request_rate", "active_connections"): 0.62,
    ("request_rate", "queue_depth"): 0.40,
    ("p99_latency_ms", "queue_depth"): 0.38,
    ("p99_latency_ms", "disk_io_wait_ms"): 0.34,
    ("error_rate_pct", "p99_latency_ms"): 0.30,
    ("cache_hit_ratio", "request_rate"): -0.28,
    ("mem_util", "gc_pause_ms"): 0.45,
    ("gc_pause_ms", "p99_latency_ms"): 0.25,
}

# --------------------------------------------------------------- data mess --
FRAC_MISSING = {"gc_pause_ms": 0.013, "cache_hit_ratio": 0.009}
FRAC_LATENCY_SENTINEL = 0.008       # p99_latency_ms == -1 means "no probe data"
N_DUPLICATE_ROWS = 60

# -------------------------------------------------------------- drift spec --
# Concept drift across the horizon (t runs 0 -> 1 over days 60..119).
#
# Fading a coefficient toward zero is not enough to make a model visibly decay:
# the surviving features carry the slack and AUC barely moves. What actually
# degrades a deployed model is a relationship that *reverses*. So the story is:
# a new caching layer ships, and a high cache-hit ratio stops being protective
# (it now masks a slow backend); a retry layer hides errors, so `error_rate_pct`
# loses its meaning; and a new runtime makes GC pauses the dominant failure.
#
# `cache_end` and `error_end` are absolute END coefficients, not multipliers,
# so the sign flip is explicit.
DRIFT = {
    "cache_end": 0.30,       # from -0.46 -> +0.30: the sign REVERSES
    "error_end": 0.12,       # from  0.70 -> +0.12: signal nearly vanishes
    "gc_growth": 3.4,        # gc coefficient multiplies by (1 + this)
    "traffic_growth": 0.45,  # latent mean shift on request rate / connections
    "latency_growth": 0.35,
}


def sigmoid(x):
    out = np.empty_like(x, dtype=float)
    pos = x >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
    e = np.exp(x[~pos])
    out[~pos] = e / (1.0 + e)
    return out


def solve_offset(logit, target_rate, lo=-8.0, hi=8.0, tol=1e-10):
    """Find delta with mean(sigmoid(logit + delta)) == target_rate."""
    for _ in range(200):
        mid = (lo + hi) / 2
        if sigmoid(logit + mid).mean() < target_rate:
            lo = mid
        else:
            hi = mid
        if hi - lo < tol:
            break
    return (lo + hi) / 2


def build_correlation():
    idx = {n: i for i, n in enumerate(NUMERIC)}
    corr = np.eye(len(NUMERIC))
    for (a, b), r in CORRELATIONS.items():
        corr[idx[a], idx[b]] = corr[idx[b], idx[a]] = r
    vals, vecs = np.linalg.eigh(corr)
    if vals.min() < 1e-6:
        corr = vecs @ np.diag(np.clip(vals, 1e-6, None)) @ vecs.T
        d = np.sqrt(np.diag(corr))
        corr = corr / np.outer(d, d)
        corr = (corr + corr.T) / 2
        np.fill_diagonal(corr, 1.0)
    return corr


# Tail heaviness of the skewed columns. These are a *calibration* knob, not
# cosmetics: the heavier the tail, the more a linear model on RAW values is
# dragged around by a few extreme rows, and the more a log transform pays in
# Part 4. Setting them too mild leaves the student nothing to earn.
TAILS = {
    "p99_latency_ms": 1.35,
    "gc_pause_ms": 1.25,
    "error_rate_pct": 0.55,     # gamma shape; smaller = more mass at 0, longer tail
    "disk_io_wait_ms": 1.10,    # gamma shape
}


def marginal(name, u):
    """Push uniforms through each column's inverse CDF."""
    if name == "cpu_util":
        return stats.beta.ppf(u, 4.2, 3.4)
    if name == "mem_util":
        return stats.beta.ppf(u, 5.0, 3.6)
    if name == "p99_latency_ms":
        return stats.lognorm.ppf(u, s=TAILS["p99_latency_ms"], scale=140.0)
    if name == "request_rate":
        return stats.lognorm.ppf(u, s=0.45, scale=900.0)
    if name == "error_rate_pct":
        return stats.gamma.ppf(u, a=TAILS["error_rate_pct"], scale=0.85)
    if name == "queue_depth":
        return stats.nbinom.ppf(u, n=4.0, p=0.30)
    if name == "cache_hit_ratio":
        return stats.beta.ppf(u, 14.0, 1.8)
    if name == "gc_pause_ms":
        return stats.lognorm.ppf(u, s=TAILS["gc_pause_ms"], scale=22.0)
    if name == "active_connections":
        return stats.lognorm.ppf(u, s=0.40, scale=520.0)
    if name == "disk_io_wait_ms":
        return stats.gamma.ppf(u, a=TAILS["disk_io_wait_ms"], scale=1.7)
    raise ValueError(name)


def model_frame(df):
    """The representation the true mechanism acts on."""
    out = pd.DataFrame(index=df.index)
    out["cpu_util"] = df["cpu_util"]
    out["mem_util"] = df["mem_util"]
    out["log_p99_latency"] = np.log(df["p99_latency_ms"])
    out["log_request_rate"] = np.log(df["request_rate"])
    out["log1p_error_rate"] = np.log1p(df["error_rate_pct"])
    out["queue_depth"] = df["queue_depth"].astype(float)
    out["cache_hit_ratio"] = df["cache_hit_ratio"]
    out["log_gc_pause"] = np.log(df["gc_pause_ms"])
    out["log_active_connections"] = np.log(df["active_connections"])
    out["log_disk_io_wait"] = np.log(df["disk_io_wait_ms"] + 0.05)
    return out


def generate(seed=SEED):
    rng = np.random.default_rng(seed)

    n_windows = N_DAYS * WINDOWS_PER_DAY
    n_rows = n_windows * N_INSTANCES

    # ---- timeline & instances -------------------------------------------
    w_index = np.tile(np.arange(n_windows), N_INSTANCES)
    inst_index = np.repeat(np.arange(N_INSTANCES), n_windows)
    timestamps = START + pd.to_timedelta(w_index * WINDOW_HOURS, unit="h")
    day = w_index // WINDOWS_PER_DAY
    hour = (w_index % WINDOWS_PER_DAY) * WINDOW_HOURS

    # 24 instances = 4 regions x 3 types x 2 replicas, so a category is never
    # just an alias for one machine.
    combos = [(r, t) for r in REGIONS for t in INSTANCE_TYPES]
    region = np.array([combos[i % 12][0] for i in inst_index])
    instance_type = np.array([combos[i % 12][1] for i in inst_index])
    instance_id = np.array([f"gw-{i:02d}" for i in inst_index])
    service_tier = rng.choice(SERVICE_TIERS, size=n_rows, p=[0.45, 0.35, 0.20])

    phase = np.full(n_rows, "E", dtype=object)
    bounds = [0] + DEPLOY_DAYS + [DEV_DAYS]
    for k in range(5):
        phase[(day >= bounds[k]) & (day < bounds[k + 1])] = PHASES[k]
    is_horizon = day >= DEV_DAYS
    t_drift = np.where(is_horizon, (day - DEV_DAYS) / (N_DAYS - DEV_DAYS - 1), 0.0)

    # ---- correlated latents ---------------------------------------------
    latent = rng.standard_normal((n_rows, len(NUMERIC))) @ np.linalg.cholesky(
        build_correlation()).T

    # Diurnal traffic: a real daily cycle, strongest mid-afternoon.
    diurnal = np.sin((hour - 4) / 24 * 2 * np.pi)
    latent[:, NUMERIC.index("request_rate")] += 0.85 * diurnal
    latent[:, NUMERIC.index("active_connections")] += 0.65 * diurnal
    latent[:, NUMERIC.index("cpu_util")] += 0.40 * diurnal

    # Covariate shift over the horizon: traffic and latency creep upward.
    latent[:, NUMERIC.index("request_rate")] += DRIFT["traffic_growth"] * t_drift
    latent[:, NUMERIC.index("active_connections")] += DRIFT["traffic_growth"] * t_drift
    latent[:, NUMERIC.index("p99_latency_ms")] += DRIFT["latency_growth"] * t_drift

    # Deployment C shipped a memory leak: GC pauses jump and stay high.
    latent[:, NUMERIC.index("gc_pause_ms")] += np.where(
        np.isin(phase, ["C"]), 0.55, 0.0)

    u = np.clip(stats.norm.cdf(latent), 1e-9, 1 - 1e-9)
    data = {name: marginal(name, u[:, i]) for i, name in enumerate(NUMERIC)}

    df = pd.DataFrame(data)
    df.insert(0, "timestamp", timestamps)
    df.insert(1, "instance_id", instance_id)
    df.insert(2, "region", region)
    df.insert(3, "instance_type", instance_type)
    df.insert(4, "service_tier", service_tier)
    df["deploy_phase"] = phase

    # ---- true logit -------------------------------------------------------
    mf = model_frame(df)
    stdz = {c: (mf[c].mean(), mf[c].std(ddof=0)) for c in mf.columns}
    z = {c: (mf[c].to_numpy() - m) / s for c, (m, s) in stdz.items()}

    logit = np.zeros(n_rows)
    for name, coef in BETA.items():
        if name == "cache_hit_ratio":
            # Linearly interpolate to an END coefficient of the opposite sign.
            eff = coef + (DRIFT["cache_end"] - coef) * t_drift
        elif name == "log1p_error_rate":
            eff = coef + (DRIFT["error_end"] - coef) * t_drift
        elif name == "log_gc_pause":
            eff = coef * (1 + DRIFT["gc_growth"] * t_drift)
        else:
            eff = coef
        logit += eff * z[name]
    for (a, b), coef in GAMMA.items():
        logit += coef * z[a] * z[b]

    for col, mapping in CAT_EFFECT.items():
        logit += np.array([mapping[v] for v in df[col]])

    # ---- lock the phase rates in by construction --------------------------
    offsets = {}
    for ph in PHASES:
        m = (phase == ph) & ~is_horizon
        offsets[ph] = solve_offset(logit[m], PHASE_RATES[ph])
        logit[m] += offsets[ph]
    # The horizon keeps running release E, so it inherits E's offset.
    logit[is_horizon] += offsets["E"]

    df["incident"] = rng.binomial(1, sigmoid(logit))

    # ---- realistic mess ---------------------------------------------------
    df = df.sort_values(["timestamp", "instance_id"]).reset_index(drop=True)

    for col, frac in FRAC_MISSING.items():
        df.loc[rng.choice(len(df), int(frac * len(df)), replace=False), col] = np.nan

    sentinel = rng.choice(len(df), int(FRAC_LATENCY_SENTINEL * len(df)), replace=False)
    df.loc[sentinel, "p99_latency_ms"] = -1.0

    dup_rows = df.iloc[rng.choice(len(df), N_DUPLICATE_ROWS, replace=False)]
    df = pd.concat([df, dup_rows], ignore_index=True)
    df = df.sort_values(["timestamp", "instance_id"], kind="mergesort").reset_index(drop=True)

    # ---- rounding ---------------------------------------------------------
    for col, nd in [("cpu_util", 4), ("mem_util", 4), ("p99_latency_ms", 2),
                    ("request_rate", 2), ("error_rate_pct", 4),
                    ("cache_hit_ratio", 4), ("gc_pause_ms", 3),
                    ("active_connections", 1), ("disk_io_wait_ms", 3)]:
        df[col] = df[col].round(nd)
    df["queue_depth"] = df["queue_depth"].astype(int)

    meta = {
        "seed": seed,
        "n_rows": int(len(df)),
        "dev_period_end": str(START + pd.Timedelta(days=DEV_DAYS)),
        "deploy_timestamps": [str(START + pd.Timedelta(days=d)) for d in DEPLOY_DAYS],
        "phase_target_rates": PHASE_RATES,
        "phase_intercept_offsets": {k: float(v) for k, v in offsets.items()},
        "beta": BETA,
        "gamma": {f"{a}*{b}": v for (a, b), v in GAMMA.items()},
        "categorical_effects": CAT_EFFECT,
        "drift": DRIFT,
        "n_duplicates": int(N_DUPLICATE_ROWS),
    }
    return df, meta


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir", type=Path,
                    default=Path(__file__).resolve().parents[1] / "data")
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    df, meta = generate()
    cols = (["timestamp", "instance_id", "region", "instance_type", "service_tier"]
            + NUMERIC + ["deploy_phase", "incident"])
    df[cols].to_csv(args.out_dir / "edge_gateway_logs.csv", index=False)
    with open(args.out_dir / "generator_metadata.json", "w") as fh:
        json.dump(meta, fh, indent=2)

    dev = df[df.timestamp < START + pd.Timedelta(days=DEV_DAYS)]
    hor = df[df.timestamp >= START + pd.Timedelta(days=DEV_DAYS)]
    print(f"wrote {len(df):,} rows -> {args.out_dir}")
    print(f"  dev period : {len(dev):,} rows, incident rate {dev.incident.mean():.4f}")
    print(f"  horizon    : {len(hor):,} rows, incident rate {hor.incident.mean():.4f}")
    print("\n  phase rates (dev only):")
    for ph in PHASES:
        m = dev[dev.deploy_phase == ph]
        print(f"    {ph}: n={len(m):5,}  rate={m.incident.mean():.4f}  "
              f"(target {PHASE_RATES[ph]:.3f})")
    print(f"\n  missing: {df[NUMERIC].isna().sum().sum():,} cells, "
          f"{int((df.p99_latency_ms == -1).sum()):,} latency sentinels, "
          f"{int(df.duplicated().sum()):,} duplicate rows")


if __name__ == "__main__":
    main()
