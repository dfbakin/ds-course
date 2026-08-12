"""Tests for the generated dataset and the calibrated reference ladder.

These guard the properties the lesson's narrative depends on.  If a change to
the generator breaks one of these, some section of the notebook stops being
true, so each test names the story it protects.

Fast by default: the ladder is asserted from data/reference_results.json (the
frozen output of src/calibrate_reference.py).  Refits of models 1 and 4 live
behind the `slow` marker.
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
    return pd.read_csv(DATA / "fleet_incidents.csv")


@pytest.fixture(scope="module")
def meta():
    with open(DATA / "dgp_metadata.json") as fh:
        return json.load(fh)


@pytest.fixture(scope="module")
def ref():
    with open(DATA / "reference_results.json") as fh:
        return json.load(fh)


# ==========================================================================
# integrity
# ==========================================================================


def test_files_exist():
    for name in ["fleet_incidents.csv", "dgp_metadata.json",
                 "reference_results.json", "DATA_CARD.md"]:
        assert (DATA / name).exists(), (
            f"missing {name} -- run src/generate_dataset.py and "
            "src/calibrate_reference.py")


def test_shape_and_no_missing(df):
    assert len(df) == 30_000
    assert df.isna().sum().sum() == 0
    for col in (["row_id"] + G.CATEGORICAL_FEATURES + G.RAW_FEATURES
                + ["power_draw_watts", "incident", "split"]):
        assert col in df.columns


def test_row_ids_unique_and_ordered(df):
    assert df["row_id"].is_unique
    assert (df["row_id"].to_numpy() == np.arange(len(df))).all()


def test_targets_are_well_typed(df):
    assert set(df["incident"].unique()) == {0, 1}
    assert df["power_draw_watts"].dtype.kind == "f"


def test_generation_is_deterministic():
    """Re-running the generator must reproduce the shipped CSV exactly."""
    regenerated, _ = G.generate(G.SEED)
    on_disk = pd.read_csv(DATA / "fleet_incidents.csv")
    for col in on_disk.columns:
        left = regenerated[col].to_numpy()
        right = on_disk[col].to_numpy()
        if left.dtype.kind == "f":
            assert np.allclose(left, right, atol=1e-9), f"{col} differs"
        else:
            assert (left.astype(str) == right.astype(str)).all(), f"{col} differs"


# ==========================================================================
# §1 -- the categorical structure the EDA section describes
# ==========================================================================


def test_categorical_cardinalities(df):
    cards = {c: df[c].nunique() for c in G.CATEGORICAL_FEATURES}
    assert cards["region"] == 10
    assert cards["instance_type"] == 6
    assert cards["deploy_channel"] == 3
    assert cards["os_image"] == 12
    assert cards["cluster_id"] == G.N_CLUSTERS  # ~180: the OHE pain point


def test_deploy_channel_mix(df):
    """§1 quotes the ~70/20/10 stable/beta/canary mix."""
    props = df["deploy_channel"].value_counts(normalize=True)
    assert abs(props["stable"] - 0.70) < 0.02
    assert abs(props["beta"] - 0.20) < 0.02
    assert abs(props["canary"] - 0.10) < 0.02


def test_clusters_are_nested_in_regions(df):
    """cluster_id -> region must be a function (strict nesting)."""
    assert df.groupby("cluster_id")["region"].nunique().max() == 1
    pairs = df[["cluster_id", "region"]].drop_duplicates()
    for cid, region in pairs.itertuples(index=False):
        assert cid.startswith(region + "-c"), (cid, region)


def test_cluster_sizes_are_long_tailed(df):
    """§1's 'you cannot eyeball 180 clusters' argument needs tiny clusters."""
    sizes = df["cluster_id"].value_counts()
    assert (sizes < 25).sum() >= 20, "need a real long tail of small clusters"
    assert sizes.min() >= G.CLUSTER_MIN_ROWS
    assert sizes.max() > 500, "and a few giants for contrast"


def test_unseen_clusters(df, meta):
    """§4's unknown-category problem: some clusters never appear in train."""
    train = set(df.loc[df.split == "train", "cluster_id"])
    heldout = set(df.loc[df.split != "train", "cluster_id"])
    unseen = heldout - train
    assert set(meta["unseen_clusters"]) <= unseen
    assert len(meta["unseen_clusters"]) >= 5
    rows = df[df.cluster_id.isin(meta["unseen_clusters"])]
    assert (rows["split"] != "train").all()
    # both val and test must contain some, or the OHE demo only half-works
    assert (rows["split"] == "val").sum() >= 20
    assert (rows["split"] == "test").sum() >= 20


# ==========================================================================
# splits and class balance
# ==========================================================================


def test_splits_are_sized_and_stratified(df):
    counts = df["split"].value_counts().to_dict()
    assert counts == {"train": 18_000, "val": 6_000, "test": 6_000}
    rates = df.groupby("split")["incident"].mean()
    assert rates.max() - rates.min() < 0.01, "splits must be stratified"


def test_incident_rate_in_calibrated_range(df):
    """The spec pins the base rate to 20-24%."""
    assert 0.20 <= df["incident"].mean() <= 0.24


# ==========================================================================
# §1 -- marginal shapes (the EDA panel needs variety)
# ==========================================================================


def test_heavy_tailed_latency(df):
    """Right-skewed raw, symmetric after log -- motivates the log transform.
    lognorm(s=0.55) has theoretical skew ~2.0."""
    assert stats.skew(df["disk_latency_ms"]) > 1.5
    assert abs(stats.skew(np.log(df["disk_latency_ms"]))) < 0.5


def test_request_rate_is_bimodal(df):
    logs = np.log(df["request_rate_rps"])
    hist, _ = np.histogram(logs, bins=60)
    peak_left = hist[:25].max()
    peak_right = hist[35:].max()
    trough = hist[25:35].min()
    assert trough < 0.75 * min(peak_left, peak_right), "not visibly bimodal"


def test_marginal_shapes_are_varied(df):
    assert stats.skew(df["cache_hit_ratio"]) < -0.5   # left-skewed
    assert stats.skew(df["error_rate_pct"]) > 2       # spike at 0, long tail
    assert df["queue_depth"].dtype.kind in "iu"       # discrete
    old = (df["node_age_days"] > 600).mean()          # young fleet + old tail
    assert 0.15 < old < 0.40


def test_collinear_pair_is_collinear(df):
    """§2/§3's ridge-vs-lasso story rests on corr(cpu, mem) being real."""
    assert df["cpu_util"].corr(df["mem_pressure"]) > 0.5


# ==========================================================================
# §2/§3 -- the regression target's sparse ground truth
# ==========================================================================


def test_power_true_coefficients_are_sparse(meta):
    coefs = meta["power_mechanism"]["coefficients"]
    support = [k for k, v in coefs.items() if v != 0.0]
    assert sorted(support) == sorted(meta["power_mechanism"]["true_support"])
    assert len(support) == 3
    assert coefs["mem_pressure"] == 0.0, "the collinear trap must be inert"


def test_ols_recovers_the_power_mechanism(df, meta):
    """Cheap end-to-end check: OLS on train separates real from null coefs."""
    from sklearn.linear_model import LinearRegression

    tr = df[df.split == "train"]
    X = tr[G.RAW_FEATURES].to_numpy(float)
    X = (X - X.mean(0)) / X.std(0)
    fit = LinearRegression().fit(X, tr["power_draw_watts"].to_numpy())
    coef = dict(zip(G.RAW_FEATURES, fit.coef_))
    true_support = set(meta["power_mechanism"]["true_support"])
    for name in G.RAW_FEATURES:
        if name in true_support:
            assert abs(coef[name]) > 10, f"{name} should carry >10 W/sd"
        else:
            assert abs(coef[name]) < 3, f"{name} should be ~0 (got {coef[name]:.2f})"


# ==========================================================================
# the calibrated reference ladder (read from reference_results.json)
# ==========================================================================


def test_reference_ladder_ordering(ref):
    """§9's table: strict (1) < (2) < (4) < (5), tree between (1) and (4)."""
    a = {k: v["test_auc"] for k, v in ref["ladder"].items()}
    assert a["logreg_numeric"] < a["logreg_ohe"] < a["catboost_default"] \
        < a["catboost_tuned"]
    assert a["logreg_numeric"] < a["tree"] < a["catboost_default"]
    for v in a.values():
        assert 0.60 < v < 0.95, "AUCs must stay in a plausible band"


def test_reference_ladder_gaps(ref):
    """The three calibrated margins the lesson's narrative quotes."""
    gaps = ref["gaps"]
    assert gaps["numeric_to_ohe"] >= 0.02, "categoricals must matter"
    assert gaps["ohe_to_catboost"] >= 0.02, "trees + native cats must matter"
    assert gaps["default_to_tuned"] >= 0.003, "tuning must be visible"
    assert gaps["default_to_tuned"] < 0.02, "...but modest (honest message)"


def test_reference_feature_explosion(ref):
    """§4 quotes the OHE feature count (~220 columns from 13)."""
    assert 200 <= ref["n_features_after_ohe"] <= 240
    assert ref["n_features_numeric"] == 8


def test_reference_matches_dataset_rate(ref, df):
    assert abs(ref["incident_rate"] - df["incident"].mean()) < 1e-9


def test_optuna_metadata(ref):
    tuned = ref["ladder"]["catboost_tuned"]
    assert tuned["n_trials"] == 30
    assert f"seed={G.SEED}" in tuned["sampler"]
    p = tuned["best_params"]
    assert 4 <= p["depth"] <= 8
    assert 0.02 <= p["learning_rate"] <= 0.3
    assert 1.0 <= p["l2_leaf_reg"] <= 30.0


def test_lasso_support_recovery(ref, meta):
    """§3's punchline: lasso finds the true support, and not by luck."""
    L = ref["lasso_support_recovery"]
    assert L["recovered"] is True
    assert sorted(L["true_support"]) == sorted(
        meta["power_mechanism"]["true_support"])
    # The recovering alpha window must be wide (>= one decade), otherwise
    # "at reasonable alpha" would be an overstatement.
    assert L["recovering_alpha_max"] / L["recovering_alpha_min"] >= 10
    zeroed = set(L["example_coefficients"]) - set(L["true_support"])
    for name in zeroed:
        assert L["example_coefficients"][name] == 0.0
    assert L["test_r2_at_example_alpha"] > 0.5


def test_ridge_shrinks_collinear_pair_together(ref):
    """§3: as the L2 penalty grows, |coef_cpu - coef_mem| must shrink."""
    R = ref["ridge_collinearity"]
    assert R["gap_shrinks_monotonically"] is True
    assert R["abs_gap_path"][0] > 5 * R["abs_gap_path"][-1]


# ==========================================================================
# slow -- refit the cheap and the expensive end of the ladder
# ==========================================================================


@pytest.mark.slow
def test_refit_logreg_numeric_matches_reference(df, ref):
    """Model 1 refit from the shipped CSV must land within +-0.01."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    tr, te = df[df.split == "train"], df[df.split == "test"]
    model = Pipeline([("scale", StandardScaler()),
                      ("clf", LogisticRegression(max_iter=1000))])
    model.fit(tr[G.RAW_FEATURES], tr["incident"])
    auc = roc_auc_score(te["incident"],
                        model.predict_proba(te[G.RAW_FEATURES])[:, 1])
    assert abs(auc - ref["ladder"]["logreg_numeric"]["test_auc"]) < 0.01


@pytest.mark.slow
def test_refit_catboost_default_matches_reference(df, ref):
    """Model 4 refit must land within +-0.01 (CatBoost is deterministic per
    machine, but we do not bet on cross-machine bit-equality)."""
    from catboost import CatBoostClassifier
    from sklearn.metrics import roc_auc_score

    d = df.copy()
    for c in G.CATEGORICAL_FEATURES:
        d[c] = d[c].astype(str)
    cols = G.RAW_FEATURES + G.CATEGORICAL_FEATURES
    tr, va, te = (d[d.split == s] for s in ("train", "val", "test"))
    model = CatBoostClassifier(iterations=500, random_seed=G.SEED, verbose=0,
                               cat_features=G.CATEGORICAL_FEATURES,
                               allow_writing_files=False)
    model.fit(tr[cols], tr["incident"],
              eval_set=(va[cols], va["incident"]),
              early_stopping_rounds=50, use_best_model=True)
    auc = roc_auc_score(te["incident"], model.predict_proba(te[cols])[:, 1])
    stored = ref["ladder"]["catboost_default"]["test_auc"]
    assert abs(auc - stored) < 0.01
    # and the big picture must survive a refit: CatBoost >> numeric logreg
    assert auc - ref["ladder"]["logreg_numeric"]["test_auc"] > 0.08
