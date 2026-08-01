"""Unit tests for every algorithm the lesson implements by hand.

The standard each implementation is held to: agreement with an *independent*
route to the same number -- a library, a different formula, or a simulation.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy import stats
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score,
                             roc_curve)
from sklearn.linear_model import LinearRegression

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import reference_impl as R  # noqa: E402


@pytest.fixture(scope="module")
def rng():
    return np.random.default_rng(12345)


def make_binary(rng, n=800, p=0.3, signal=0.9, ties=False):
    y = rng.binomial(1, p, n)
    while y.sum() < 5 or y.sum() > n - 5:
        y = rng.binomial(1, p, n)
    s = rng.normal(y * signal, 1.0, n)
    if ties:
        s = np.round(s, 1)
    return y, s


# ==========================================================================
# §1 rate comparison
# ==========================================================================


def test_two_proportion_matches_chi_square(rng):
    """z^2 from the two-proportion test must equal the 2x2 chi-square."""
    for _ in range(20):
        n1, n2 = int(rng.integers(200, 3000)), int(rng.integers(200, 3000))
        x1, x2 = int(rng.binomial(n1, 0.25)), int(rng.binomial(n2, 0.30))
        _, z, p, _ = R.two_proportion_test(x1, n1, x2, n2)
        table = np.array([[x1, n1 - x1], [x2, n2 - x2]])
        chi2, chi2_p, _, _ = stats.chi2_contingency(table, correction=False)
        assert np.isclose(z ** 2, chi2)
        assert np.isclose(p, chi2_p)


def test_wilson_interval_covers_and_is_bounded(rng):
    """Wilson intervals stay inside [0,1] even at extreme p, unlike the normal one."""
    lo, hi = R.wilson_interval(0, 50)
    assert 0 <= lo <= hi <= 1
    lo, hi = R.wilson_interval(50, 50)
    assert 0 <= lo <= hi <= 1

    # Coverage: a 95% interval should contain the truth about 95% of the time.
    p_true, n, hits, trials = 0.25, 400, 0, 2000
    for _ in range(trials):
        x = int(rng.binomial(n, p_true))
        lo, hi = R.wilson_interval(x, n)
        hits += lo <= p_true <= hi
    assert 0.93 <= hits / trials <= 0.97


# ==========================================================================
# §3 / §4 fitting
# ==========================================================================


def test_normal_equation_matches_sklearn(rng):
    X = rng.normal(size=(500, 6))
    y = X @ np.array([1.0, -2.0, 0.5, 0.0, 3.0, -1.5]) + rng.normal(0, 0.5, 500) + 4.0
    w, b = R.fit_normal_equation(X, y)
    sk = LinearRegression().fit(X, y)
    assert np.allclose(w, sk.coef_, atol=1e-9)
    assert np.isclose(b, sk.intercept_, atol=1e-9)


def test_normal_equation_raises_on_singular(rng):
    X = rng.normal(size=(200, 4))
    X = np.hstack([X, X[:, [0]]])  # exact duplicate column
    y = rng.normal(size=200)
    with pytest.raises(np.linalg.LinAlgError):
        R.fit_normal_equation(X, y)


def test_gradient_descent_reaches_closed_form(rng):
    X = rng.normal(size=(600, 5))
    y = X @ np.array([1.0, -1.0, 2.0, 0.3, -0.7]) + rng.normal(0, 0.3, 600) + 2.0
    w_exact, b_exact = R.fit_normal_equation(X, y)
    lr = 0.9 * R.critical_learning_rate(X)
    w_gd, b_gd, hist = R.fit_gradient_descent(X, y, lr=lr, n_iters=20000)
    assert np.allclose(w_gd, w_exact, atol=1e-6)
    assert np.isclose(b_gd, b_exact, atol=1e-6)
    # Loss must decrease monotonically for a stable step size on a convex loss.
    assert np.all(np.diff(hist) <= 1e-12)


def test_gradient_descent_diverges_above_critical_rate(rng):
    """The lr < 2/L_max theory from §4, tested rather than asserted."""
    X = rng.normal(size=(400, 4))
    y = rng.normal(size=400)
    lr_crit = R.critical_learning_rate(X)

    _, _, below = R.fit_gradient_descent(X, y, lr=0.9 * lr_crit, n_iters=300)
    assert np.isfinite(below[-1]) and below[-1] < 2.0

    _, _, above = R.fit_gradient_descent(X, y, lr=1.10 * lr_crit, n_iters=300)
    assert (not np.isfinite(above[-1])) or above[-1] > 1e3


# ==========================================================================
# §5 cross-features
# ==========================================================================


def test_cross_features_shape_and_values(rng):
    X = rng.normal(size=(50, 4))
    out, names, pairs = R.make_cross_features(X, ["a", "b", "c", "d"])
    assert out.shape == (50, 4 + 10)          # 4 + C(4+1, 2)
    assert len(names) == 14 and len(pairs) == 10
    assert names[4] == "a^2" and names[5] == "a*b"
    assert np.allclose(out[:, 4], X[:, 0] ** 2)
    assert np.allclose(out[:, 5], X[:, 0] * X[:, 1])
    assert np.allclose(out[:, :4], X)


# ==========================================================================
# §6 / §7 metrics
# ==========================================================================


def test_confusion_and_metrics_match_sklearn(rng):
    for _ in range(30):
        y, s = make_binary(rng, n=int(rng.integers(50, 1500)))
        t = float(rng.normal(0.3, 0.5))
        pred = (s >= t).astype(int)

        tn, fp, fn, tp = R.confusion_counts(y, pred)
        if len(np.unique(pred)) == 1 and len(np.unique(y)) == 1:
            continue
        sk = confusion_matrix(y, pred, labels=[0, 1]).ravel()
        assert (tn, fp, fn, tp) == tuple(int(v) for v in sk)

        assert np.isclose(R.accuracy_manual(y, pred), accuracy_score(y, pred))
        assert np.isclose(R.precision_manual(y, pred),
                          precision_score(y, pred, zero_division=0))
        assert np.isclose(R.recall_manual(y, pred),
                          recall_score(y, pred, zero_division=0))
        assert np.isclose(R.f1_manual(y, pred), f1_score(y, pred, zero_division=0))


def test_metrics_handle_degenerate_predictions():
    y = np.array([0, 1, 0, 1])
    none_flagged = np.zeros(4, dtype=int)
    assert R.precision_manual(y, none_flagged) == 0.0   # 0/0 -> 0, like sklearn
    assert R.recall_manual(y, none_flagged) == 0.0
    assert R.f1_manual(y, none_flagged) == 0.0

    all_flagged = np.ones(4, dtype=int)
    assert R.recall_manual(y, all_flagged) == 1.0
    assert np.isclose(R.precision_manual(y, all_flagged), 0.5)


def test_roc_curve_matches_sklearn_exactly(rng):
    for ties in (False, True):
        y, s = make_binary(rng, n=900, ties=ties)
        fpr, tpr, _ = R.roc_curve_manual(y, s)
        sk_fpr, sk_tpr, _ = roc_curve(y, s, drop_intermediate=False)
        assert fpr.shape == sk_fpr.shape
        assert np.allclose(fpr, sk_fpr)
        assert np.allclose(tpr, sk_tpr)


def test_three_auc_routes_agree(rng):
    """Trapezoid, Mann-Whitney ranks and sklearn must all coincide."""
    for ties in (False, True):
        y, s = make_binary(rng, n=1200, ties=ties)
        fpr, tpr, _ = R.roc_curve_manual(y, s)
        a_trapz = R.auc_trapezoid(fpr, tpr)
        a_rank = R.auc_via_ranks(y, s)
        a_sk = roc_auc_score(y, s)
        assert np.isclose(a_trapz, a_sk, atol=1e-12)
        assert np.isclose(a_rank, a_sk, atol=1e-12)


def test_auc_known_values():
    """Hand-checkable cases."""
    y = np.array([0, 0, 1, 1])
    assert np.isclose(R.auc_via_ranks(y, np.array([0.1, 0.2, 0.8, 0.9])), 1.0)
    assert np.isclose(R.auc_via_ranks(y, np.array([0.9, 0.8, 0.2, 0.1])), 0.0)
    # All scores tied -> every pair is a coin flip -> 0.5
    assert np.isclose(R.auc_via_ranks(y, np.array([0.5, 0.5, 0.5, 0.5])), 0.5)


# ==========================================================================
# §9 bootstrap
# ==========================================================================


def test_bootstrap_se_matches_analytic_for_a_mean(rng):
    """For a plain mean the bootstrap SE must reproduce sigma/sqrt(n)."""
    n = 2000
    values = rng.normal(5.0, 2.0, n)
    dummy_y = np.ones(n, dtype=int)
    boot = R.bootstrap_metric(dummy_y, values, lambda _y, v: v.mean(),
                              n_boot=2000, seed=1, stratified=False)
    analytic = values.std(ddof=1) / np.sqrt(n)
    assert abs(boot.std(ddof=1) - analytic) / analytic < 0.08


def test_bootstrap_ci_contains_point_estimate(rng):
    y, s = make_binary(rng, n=1500)
    boot = R.bootstrap_metric(y, s, roc_auc_score, n_boot=800, seed=3)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    assert lo < roc_auc_score(y, s) < hi


def test_bootstrap_is_deterministic_given_seed(rng):
    y, s = make_binary(rng, n=500)
    a = R.bootstrap_metric(y, s, roc_auc_score, n_boot=200, seed=7)
    b = R.bootstrap_metric(y, s, roc_auc_score, n_boot=200, seed=7)
    assert np.array_equal(a, b)


def test_stratified_bootstrap_preserves_class_balance(rng):
    """The reason we stratify: no resample may lose a class."""
    y = np.array([1] * 5 + [0] * 200)
    s = rng.normal(size=205)
    boot = R.bootstrap_metric(y, s, roc_auc_score, n_boot=300, seed=5,
                              stratified=True)
    assert np.all(np.isfinite(boot))   # AUC undefined if a class vanishes


# ==========================================================================
# §10 DeLong -- the highest-risk implementation
# ==========================================================================


def test_delong_auc_matches_sklearn(rng):
    for n, ties in [(500, False), (500, True), (2500, True), (77, True)]:
        y, s = make_binary(rng, n=n, ties=ties)
        aucs, _ = R.delong_auc_cov(s[None, :], y)
        assert np.isclose(aucs[0], roc_auc_score(y, s), atol=1e-12)


def test_delong_identical_models_give_p_one(rng):
    y, s = make_binary(rng, n=900)
    out = R.delong_test(y, s, s.copy())
    assert np.isclose(out["diff"], 0.0)
    assert out["p_value"] == 1.0


def test_delong_se_matches_paired_bootstrap(rng):
    """Closed-form variance vs a completely independent resampling estimate."""
    n = 2500
    y = rng.binomial(1, 0.3, n)
    signal = rng.normal(y * 1.0, 1.0, n)
    good = signal + rng.normal(0, 0.4, n)
    bad = signal * 0.5 + rng.normal(0, 1.0, n)

    out = R.delong_test(y, bad, good)
    boot = R.paired_bootstrap_auc_diff(y, bad, good, n_boot=2000, seed=11)
    assert abs(out["se"] - boot.std(ddof=1)) / out["se"] < 0.12


def test_delong_single_auc_variance_matches_bootstrap(rng):
    y, s = make_binary(rng, n=2000, signal=1.0)
    _, cov = R.delong_auc_cov(s[None, :], y)
    boot = R.bootstrap_metric(y, s, roc_auc_score, n_boot=2000, seed=13)
    assert abs(np.sqrt(cov[0, 0]) - boot.std(ddof=1)) / boot.std(ddof=1) < 0.12


@pytest.mark.slow
def test_delong_null_calibration(rng):
    """The decisive test: under H0 the p-values must be Uniform(0,1).

    Two different-but-equally-good correlated scorers. A test with a wrong
    variance formula produces plausible p-values that are not uniform, so this
    catches errors that no single-example check would.
    """
    pvals = []
    for _ in range(500):
        n = 700
        y = rng.binomial(1, 0.35, n)
        if y.sum() < 10 or y.sum() > n - 10:
            continue
        signal = rng.normal(y * 1.0, 1.0, n)
        a = signal + rng.normal(0, 0.5, n)
        b = signal + rng.normal(0, 0.5, n)
        pvals.append(R.delong_test(y, a, b)["p_value"])

    pvals = np.array(pvals)
    reject_rate = float((pvals < 0.05).mean())
    ks_p = stats.kstest(pvals, "uniform").pvalue
    assert 0.025 < reject_rate < 0.085, f"type-I error off: {reject_rate}"
    assert ks_p > 0.01, f"p-values not uniform under H0 (KS p={ks_p})"


def test_delong_detects_a_real_difference(rng):
    n = 3000
    y = rng.binomial(1, 0.3, n)
    good = rng.normal(y * 1.3, 1.0, n)
    bad = rng.normal(y * 0.5, 1.0, n)
    out = R.delong_test(y, bad, good)
    assert out["diff"] > 0
    assert out["p_value"] < 1e-6


def test_delong_is_symmetric(rng):
    y, s = make_binary(rng, n=800)
    s2 = s + rng.normal(0, 0.5, len(s))
    f = R.delong_test(y, s, s2)
    b = R.delong_test(y, s2, s)
    assert np.isclose(f["diff"], -b["diff"])
    assert np.isclose(f["p_value"], b["p_value"])
    assert np.isclose(f["se"], b["se"])


# ==========================================================================
# §11 anomalies
# ==========================================================================


class _DummyModel:
    def __init__(self, scores):
        self._scores = scores

    def predict(self, frame):
        return self._scores[: len(frame)]


def test_find_anomalies_percentile_flags_exact_fraction(rng):
    n = 2000
    frame = pd.DataFrame({"x": np.arange(n)})
    y = rng.binomial(1, 0.3, n)
    model = _DummyModel(rng.normal(0.3, 0.4, n))

    for pct, expected in [(99.0, 20), (95.0, 100), (90.0, 200)]:
        out, cutoff = R.find_anomalies(model, frame, y, percentile=pct)
        # Ties can push this slightly above the nominal count; never below.
        assert expected <= len(out) <= expected + 3
        assert (out["error"] >= cutoff).all()


def test_find_anomalies_sorted_and_consistent(rng):
    n = 500
    frame = pd.DataFrame({"x": np.arange(n)})
    y = rng.binomial(1, 0.3, n)
    scores = rng.normal(0.3, 0.4, n)
    out, _ = R.find_anomalies(_DummyModel(scores), frame, y, percentile=95.0)

    assert out["error"].is_monotonic_decreasing
    assert np.allclose(out["residual"], out["y_true"] - out["score"])
    assert np.allclose(out["error"], np.abs(out["residual"]))


def test_find_anomalies_absolute_threshold_is_informative(rng):
    """Unlike a percentile, an absolute threshold yields a data-dependent count."""
    n = 1000
    frame = pd.DataFrame({"x": np.arange(n)})
    y = np.zeros(n, dtype=int)

    clean, _ = R.find_anomalies(_DummyModel(np.full(n, 0.01)), frame, y,
                                abs_threshold=0.5)
    dirty_scores = np.full(n, 0.01)
    dirty_scores[:50] = 5.0
    dirty, _ = R.find_anomalies(_DummyModel(dirty_scores), frame, y,
                                abs_threshold=0.5)

    assert len(clean) == 0        # no anomalies -> flags nothing
    assert len(dirty) == 50       # 50 planted -> flags exactly those

    # ...whereas the percentile flags 1% of rows in both cases, regardless.
    p_clean, _ = R.find_anomalies(_DummyModel(np.full(n, 0.01)), frame, y,
                                  percentile=99.0)
    assert len(p_clean) == n      # all errors identical -> all >= the cutoff


def test_find_anomalies_signed_separates_tails(rng):
    n = 600
    frame = pd.DataFrame({"x": np.arange(n)})
    y = np.zeros(n, dtype=int)
    scores = np.zeros(n)
    scores[:10] = -3.0    # residual +3 (under-prediction)
    scores[10:20] = 3.0   # residual -3 (over-prediction)

    # 580 of the 600 residuals are exactly 0, so the cut must sit above that
    # mass: the 99th percentile of the signed residual is +3, the 97th
    # percentile of |residual| is 3.
    signed, cut = R.find_anomalies(_DummyModel(scores), frame, y, percentile=99.0,
                                   signed=True)
    assert cut == 3.0
    assert len(signed) == 10
    assert (signed["residual"] > 0).all(), "signed mode must keep one tail only"

    unsigned, _ = R.find_anomalies(_DummyModel(scores), frame, y, percentile=97.0)
    assert len(unsigned) == 20
    assert (unsigned["residual"] > 0).any() and (unsigned["residual"] < 0).any(), \
        "unsigned mode pools both tails -- the behaviour §11 warns about"
