"""Reference implementations of everything the lesson builds by hand.

Two jobs:

1. **Answer key.** When you delete an implementation from `lesson01.py` to write
   it live with students, the version you deleted is still here.
2. **Testable surface.** The notebook defines these functions inline (which is
   the point -- students should see them), but inline definitions cannot be
   unit-tested. These copies can, and `tests/test_notebook_sync.py` verifies
   that the notebook's inline copies still agree with these, so the two cannot
   silently drift apart.

Every function here is byte-for-byte the same logic as the notebook's version.
"""

from __future__ import annotations

import itertools

import numpy as np
import pandas as pd
from scipy import stats

# --------------------------------------------------------------------------
# §1 -- comparing two rates
# --------------------------------------------------------------------------


def two_proportion_test(x1, n1, x2, n2):
    """Compare two binomial rates. Returns (diff, z, p_value, se)."""
    p1, p2 = x1 / n1, x2 / n2
    p_pool = (x1 + x2) / (n1 + n2)
    se = np.sqrt(p_pool * (1 - p_pool) * (1 / n1 + 1 / n2))
    z = (p2 - p1) / se
    p_value = 2 * stats.norm.sf(abs(z))
    return p2 - p1, z, p_value, se


def wilson_interval(x, n, conf=0.95):
    """Wilson score interval for a binomial proportion."""
    zc = stats.norm.ppf(1 - (1 - conf) / 2)
    p = x / n
    denom = 1 + zc ** 2 / n
    centre = (p + zc ** 2 / (2 * n)) / denom
    halfwidth = zc * np.sqrt(p * (1 - p) / n + zc ** 2 / (4 * n ** 2)) / denom
    return centre - halfwidth, centre + halfwidth


# --------------------------------------------------------------------------
# §3 / §4 -- fitting a linear model
# --------------------------------------------------------------------------


def fit_normal_equation(X, y, fit_intercept: bool = True):
    """Least squares by the normal equations. Returns (weights, intercept)."""
    X = np.asarray(X, dtype=float)
    if fit_intercept:
        X = np.hstack([np.ones((X.shape[0], 1)), X])
    gram = X.T @ X
    rhs = X.T @ y
    beta = np.linalg.solve(gram, rhs)
    return (beta[1:], beta[0]) if fit_intercept else (beta, 0.0)


def mse_loss(X, y, w, b):
    resid = X @ w + b - y
    return float(np.mean(resid ** 2))


def fit_gradient_descent(X, y, lr=0.1, n_iters=2000, record_every=1):
    """Full-batch gradient descent on the mean squared error."""
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    n, d = X.shape

    w = np.zeros(d)
    b = 0.0
    history = []

    for it in range(n_iters):
        resid = X @ w + b - y
        grad_w = (2.0 / n) * (X.T @ resid)
        grad_b = (2.0 / n) * resid.sum()
        w -= lr * grad_w
        b -= lr * grad_b
        if it % record_every == 0:
            history.append(float(np.mean(resid ** 2)))
    return w, b, np.array(history)


def critical_learning_rate(X):
    """Largest stable step size for full-batch GD on the MSE."""
    hessian_max_eig = 2 * np.linalg.eigvalsh(X.T @ X / len(X)).max()
    return 2 / hessian_max_eig


# --------------------------------------------------------------------------
# §5 -- cross-features
# --------------------------------------------------------------------------


def make_cross_features(X, feature_names):
    """Append every product x_i * x_j (i <= j) to the design matrix."""
    pairs = list(itertools.combinations_with_replacement(range(X.shape[1]), 2))
    extra = np.column_stack([X[:, i] * X[:, j] for i, j in pairs])
    names = feature_names + [
        f"{feature_names[i]}^2" if i == j else f"{feature_names[i]}*{feature_names[j]}"
        for i, j in pairs
    ]
    return np.hstack([X, extra]), names, pairs


# --------------------------------------------------------------------------
# §6 / §7 -- metrics
# --------------------------------------------------------------------------


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
    return tp / (tp + fp) if (tp + fp) > 0 else 0.0


def recall_manual(y_true, y_pred):
    _, _, fn, tp = confusion_counts(y_true, y_pred)
    return tp / (tp + fn) if (tp + fn) > 0 else 0.0


def f1_manual(y_true, y_pred):
    p = precision_manual(y_true, y_pred)
    r = recall_manual(y_true, y_pred)
    return 2 * p * r / (p + r) if (p + r) > 0 else 0.0


def roc_curve_manual(y_true, score):
    """ROC curve without library help."""
    y_true = np.asarray(y_true)
    score = np.asarray(score, dtype=float)

    order = np.argsort(-score, kind="mergesort")
    s_sorted = score[order]
    y_sorted = y_true[order]

    tps = np.cumsum(y_sorted)
    fps = np.cumsum(1 - y_sorted)

    distinct = np.flatnonzero(np.diff(s_sorted))
    idx = np.r_[distinct, s_sorted.size - 1]

    tpr = np.r_[0.0, tps[idx] / tps[-1]]
    fpr = np.r_[0.0, fps[idx] / fps[-1]]
    thresholds = np.r_[np.inf, s_sorted[idx]]
    return fpr, tpr, thresholds


def auc_trapezoid(fpr, tpr):
    return float(np.trapezoid(tpr, fpr))


def auc_via_ranks(y_true, score):
    y_true = np.asarray(y_true)
    ranks = stats.rankdata(score)
    n_pos = int(y_true.sum())
    n_neg = len(y_true) - n_pos
    r_pos = ranks[y_true == 1].sum()
    u = r_pos - n_pos * (n_pos + 1) / 2
    return u / (n_pos * n_neg)


# --------------------------------------------------------------------------
# §9 -- bootstrap
# --------------------------------------------------------------------------


def bootstrap_metric(y_true, score, metric_fn, n_boot=2000, seed=0,
                     stratified=True):
    """Percentile bootstrap for a metric of the form metric(y_true, score)."""
    boot_rng = np.random.default_rng(seed)
    y_true = np.asarray(y_true)
    score = np.asarray(score)

    if stratified:
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


def paired_bootstrap_auc_diff(y_true, score_1, score_2, n_boot=4000, seed=0):
    """Bootstrap the AUC difference, resampling rows once for both models."""
    from sklearn.metrics import roc_auc_score

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


# --------------------------------------------------------------------------
# §10 -- DeLong
# --------------------------------------------------------------------------


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
    """Fast DeLong (Sun & Xu 2014). Returns (aucs, covariance matrix)."""
    score_matrix = np.atleast_2d(np.asarray(score_matrix, dtype=float))
    y_true = np.asarray(y_true).astype(int)

    pos = y_true == 1
    n_pos, n_neg = int(pos.sum()), int((~pos).sum())
    k = score_matrix.shape[0]

    ordered = np.hstack([score_matrix[:, pos], score_matrix[:, ~pos]])

    tx = np.empty((k, n_pos))
    ty = np.empty((k, n_neg))
    tz = np.empty((k, n_pos + n_neg))
    for r in range(k):
        tx[r] = midrank(ordered[r, :n_pos])
        ty[r] = midrank(ordered[r, n_pos:])
        tz[r] = midrank(ordered[r])

    aucs = tz[:, :n_pos].sum(axis=1) / n_pos / n_neg - (n_pos + 1.0) / 2.0 / n_neg

    v01 = (tz[:, :n_pos] - tx) / n_neg
    v10 = 1.0 - (tz[:, n_pos:] - ty) / n_pos

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


# --------------------------------------------------------------------------
# §11 -- anomalies
# --------------------------------------------------------------------------


def find_anomalies(model, frame, y_true, percentile=99.0, abs_threshold=None,
                   signed=False):
    """Flag rows whose prediction error is extreme. See the notebook for docs."""
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
