"""Tests for the generated dataset.

These guard the properties the lesson's narrative depends on. If a change to
the generator breaks one of these, some section of the notebook stops being
true, so each test names the section it protects.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import generate_dataset as G  # noqa: E402

DATA = ROOT / "data"


@pytest.fixture(scope="module")
def df():
    return pd.read_csv(DATA / "service_telemetry.csv", parse_dates=["window_start"])


@pytest.fixture(scope="module")
def truth():
    return pd.read_csv(DATA / "anomaly_ground_truth.csv")


@pytest.fixture(scope="module")
def meta():
    with open(DATA / "dgp_metadata.json") as fh:
        return json.load(fh)


# ==========================================================================
# integrity
# ==========================================================================


def test_files_exist():
    for name in ["service_telemetry.csv", "anomaly_ground_truth.csv",
                 "dgp_metadata.json"]:
        assert (DATA / name).exists(), f"missing {name} -- run src/generate_dataset.py"


def test_shape_and_no_missing(df):
    assert len(df) == 20_000
    assert df.isna().sum().sum() == 0
    for col in G.RAW_FEATURES + ["incident", "split", "phase", "row_id"]:
        assert col in df.columns


def test_row_ids_unique_and_ordered(df):
    assert df["row_id"].is_unique
    assert (df["row_id"].to_numpy() == np.arange(len(df))).all()


def test_ground_truth_aligns_with_data(df, truth):
    """§11 masks the truth file positionally; a reordering would corrupt it."""
    assert len(truth) == len(df)
    assert (truth["row_id"].to_numpy() == df["row_id"].to_numpy()).all()
    merged = df.merge(truth, on="row_id")
    assert (merged["row_id"].to_numpy() == df["row_id"].to_numpy()).all()


def test_target_is_binary(df):
    assert set(df["incident"].unique()) == {0, 1}


def test_generation_is_deterministic():
    """Re-running the generator must reproduce the shipped CSV exactly."""
    regenerated, _ = G.generate(G.SEED)
    on_disk = pd.read_csv(DATA / "service_telemetry.csv")
    cols = (["row_id", "phase"] + G.RAW_FEATURES + ["incident", "split"])
    for col in cols:
        left = regenerated[col].to_numpy()
        right = on_disk[col].to_numpy()
        if left.dtype.kind == "f":
            assert np.allclose(left, right, atol=1e-6), f"{col} differs"
        else:
            assert (left == right).all(), f"{col} differs"


# ==========================================================================
# §2 -- the distributions the EDA section describes
# ==========================================================================


def test_class_balance_is_imbalanced_but_workable(df):
    rate = df["incident"].mean()
    assert 0.20 < rate < 0.30, "the §6 accuracy-baseline argument assumes ~25%"


def test_splits_are_stratified_and_sized(df):
    counts = df["split"].value_counts().to_dict()
    assert counts == {"train": 12_000, "val": 4_000, "test": 4_000}
    rates = df.groupby("split")["incident"].mean()
    assert rates.max() - rates.min() < 0.01, "splits must be stratified"


def test_heavy_tailed_feature_is_heavy_tailed(df):
    """§2 motivates the log transform from this skew."""
    assert stats.skew(df["disk_latency_ms"]) > 8
    assert abs(stats.skew(np.log(df["disk_latency_ms"]))) < 2


def test_request_rate_is_bimodal(df):
    """§2 claims two traffic regimes; verify via a dip in the log histogram."""
    logs = np.log(df["request_rate_rps"])
    hist, edges = np.histogram(logs, bins=60)
    peak_left = hist[:25].max()
    peak_right = hist[35:].max()
    trough = hist[25:35].min()
    assert trough < 0.75 * min(peak_left, peak_right), "not visibly bimodal"


def test_marginal_shapes_are_varied(df):
    """The point of the EDA panel is that no two features look alike."""
    assert stats.skew(df["cache_hit_ratio"]) < -0.5      # left-skewed
    assert stats.skew(df["error_rate_pct"]) > 2          # right-skewed
    assert abs(stats.skew(df["temperature_c"])) < 0.2    # symmetric
    assert df["queue_depth"].dtype.kind in "iu"          # discrete


def test_features_covary(df):
    """§2's correlation heatmap needs genuine structure, not noise."""
    X = G.to_model_features(df)
    corr = X.corr().to_numpy()
    off_diag = corr[~np.eye(len(corr), dtype=bool)]
    assert np.abs(off_diag).max() > 0.5
    assert corr[0, 7] > 0.4     # cpu_util vs temperature_c


# ==========================================================================
# §1 -- the change that section 1 detects
# ==========================================================================


def test_incident_rate_really_changed_after_deploy(df):
    pre = df.loc[df.phase == "pre_deploy", "incident"]
    post = df.loc[df.phase == "post_deploy", "incident"]
    diff = post.mean() - pre.mean()
    assert diff > 0.015, "§1 needs a real, detectable increase"
    assert diff < 0.06, "...but subtle enough to require a test, not eyeballing"


def test_placebo_comparison_is_null(df):
    """§1's A/A control must NOT be significant, or the lesson inverts."""
    pre = df[df.phase == "pre_deploy"].sort_values("window_start")
    half = len(pre) // 2
    a, b = pre.iloc[:half]["incident"], pre.iloc[half:]["incident"]
    p_pool = pd.concat([a, b]).mean()
    se = np.sqrt(p_pool * (1 - p_pool) * (1 / len(a) + 1 / len(b)))
    z = (b.mean() - a.mean()) / se
    assert abs(z) < 2.0, f"placebo split is significant (z={z:.2f})"


# ==========================================================================
# §11 -- the injected anomalies
# ==========================================================================


def test_anomaly_cohorts_have_expected_sizes(df, truth):
    counts = truth["anomaly_kind"].value_counts().to_dict()
    assert counts["silent_failure"] == G.N_SILENT_FAILURE
    assert counts["sensor_glitch"] == G.N_SENSOR_GLITCH
    assert truth["is_anomaly"].sum() == G.N_SILENT_FAILURE + G.N_SENSOR_GLITCH


def test_anomalies_are_rare(df, truth):
    assert 0.005 < truth["is_anomaly"].mean() < 0.05


def test_anomalies_are_post_deploy_only(df, truth):
    merged = df.merge(truth, on="row_id")
    assert (merged.loc[merged.is_anomaly == 1, "phase"] == "post_deploy").all()


def test_silent_failures_are_incidents_with_benign_telemetry(df, truth):
    merged = df.merge(truth, on="row_id")
    silent = merged[merged.anomaly_kind == "silent_failure"]
    normal = merged[merged.anomaly_kind == "none"]
    assert silent["incident"].mean() > 0.90
    # "benign" means their telemetry sits below the ordinary median
    assert silent["cpu_util"].median() < normal["cpu_util"].median()
    assert silent["disk_latency_ms"].median() < normal["disk_latency_ms"].median()


def test_sensor_glitches_are_extreme_and_never_incidents(df, truth):
    merged = df.merge(truth, on="row_id")
    glitch = merged[merged.anomaly_kind == "sensor_glitch"]
    normal = merged[merged.anomaly_kind == "none"]
    assert (glitch["incident"] == 0).all()
    assert glitch["disk_latency_ms"].min() > normal["disk_latency_ms"].quantile(0.999)


def test_every_split_contains_some_anomalies(df, truth):
    """§11 pools val+test; both must actually contain anomalies."""
    merged = df.merge(truth, on="row_id")
    per_split = merged.groupby("split")["is_anomaly"].sum()
    assert (per_split > 0).all()
    heldout = merged[merged.split.isin(["val", "test"])]
    assert heldout["is_anomaly"].sum() >= 50, "too few anomalies to score §11"


# ==========================================================================
# §5 / §10 -- the interaction effect the model comparison relies on
# ==========================================================================


def test_metadata_records_a_reachable_ceiling(meta):
    assert 0.80 < meta["bayes_auc_clean_rows"] < 0.95
    assert meta["bayes_auc_clean_rows"] > meta["main_effects_only_auc_clean_rows"]


def test_interaction_terms_are_present_in_the_mechanism(meta):
    """§5's whole argument is that cross-features can recover something real."""
    assert len(meta["gamma"]) >= 4
    assert any(abs(v) > 0.2 for v in meta["gamma"].values())


def test_cross_features_beat_plain_features(df):
    """The §5 / §10 result, reproduced from scratch on the shipped CSV."""
    import itertools
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import roc_auc_score

    X = G.to_model_features(df)
    y = df["incident"].to_numpy()
    tr = (df["split"] == "train").to_numpy()
    te = (df["split"] == "test").to_numpy()

    raw = X[tr].to_numpy(float)
    mu, sd = raw.mean(0), raw.std(0)
    Xtr, Xte = (raw - mu) / sd, (X[te].to_numpy(float) - mu) / sd

    auc_a = roc_auc_score(y[te], LinearRegression().fit(Xtr, y[tr]).predict(Xte))

    pairs = list(itertools.combinations_with_replacement(range(Xtr.shape[1]), 2))
    cross = lambda M: np.hstack(
        [M, np.column_stack([M[:, i] * M[:, j] for i, j in pairs])])
    auc_b = roc_auc_score(
        y[te], LinearRegression().fit(cross(Xtr), y[tr]).predict(cross(Xte)))

    assert auc_b > auc_a, "cross-features must help, or §5 and §10 are wrong"
    gain = auc_b - auc_a
    assert 0.005 < gain < 0.06, (
        f"gain of {gain:.4f} is outside the 'slightly better' regime the "
        "lesson depends on")
