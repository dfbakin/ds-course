"""Guard against the notebook and the reference implementations drifting apart.

`lesson01.py` defines every algorithm inline, because students need to see them.
`src/reference_impl.py` holds the same code so it can be unit-tested and so it
survives as an answer key when implementations are deleted for a live session.

Two copies of anything will diverge unless something checks. This module parses
the notebook, extracts its top-level function definitions, executes only those,
and asserts they produce identical results to the reference versions.

If a test here fails, the two copies disagree -- decide which is right and sync
the other.
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import reference_impl as R  # noqa: E402

NOTEBOOK = ROOT / "lesson01.py"

# Functions defined inline in the notebook that must match reference_impl.
SHARED = [
    "two_proportion_test", "wilson_interval",
    "fit_normal_equation", "mse_loss", "fit_gradient_descent",
    "critical_learning_rate", "make_cross_features",
    "confusion_counts", "accuracy_manual", "precision_manual",
    "recall_manual", "f1_manual",
    "roc_curve_manual", "auc_trapezoid", "auc_via_ranks",
    "bootstrap_metric", "paired_bootstrap_auc_diff",
    "midrank", "delong_auc_cov", "delong_test",
    "find_anomalies",
]


@pytest.fixture(scope="module")
def nb():
    """Exec the notebook's top-level function definitions in isolation.

    A jupytext `py:percent` file is valid Python, so it parses directly. We
    execute only the FunctionDef nodes -- never the notebook's top-level code,
    which would load data and draw plots.
    """
    tree = ast.parse(NOTEBOOK.read_text())
    defs = {n.name: n for n in tree.body if isinstance(n, ast.FunctionDef)}

    ns: dict = {}
    exec(
        "from __future__ import annotations\n"
        "import itertools\n"
        "import numpy as np\n"
        "import pandas as pd\n"
        "from scipy import stats\n"
        "from sklearn.metrics import roc_auc_score\n",
        ns,
    )
    missing = [name for name in SHARED if name not in defs]
    assert not missing, f"notebook no longer defines: {missing}"

    for name in SHARED:
        module = ast.Module(body=[defs[name]], type_ignores=[])
        exec(compile(ast.fix_missing_locations(module), "<notebook>", "exec"), ns)
    return ns


def same(a, b):
    """Structural equality for the return shapes these functions use."""
    if isinstance(a, dict):
        assert set(a) == set(b)
        for k in a:
            same(a[k], b[k])
        return True
    if isinstance(a, (tuple, list)):
        assert len(a) == len(b)
        for x, y in zip(a, b):
            same(x, y)
        return True
    if isinstance(a, pd.DataFrame):
        pd.testing.assert_frame_equal(a, b)
        return True
    if isinstance(a, np.ndarray):
        assert np.allclose(a, b, equal_nan=True)
        return True
    if isinstance(a, str):
        assert a == b
        return True
    assert np.isclose(a, b, equal_nan=True), f"{a} != {b}"
    return True


@pytest.fixture(scope="module")
def sample():
    rng = np.random.default_rng(99)
    n = 600
    y = rng.binomial(1, 0.3, n)
    s = rng.normal(y * 0.9, 1.0, n)
    s2 = s + rng.normal(0, 0.4, n)
    X = rng.normal(size=(n, 4))
    target = X @ np.array([1.0, -0.5, 2.0, 0.3]) + rng.normal(0, 0.4, n)
    return dict(y=y, s=s, s2=s2, X=X, target=target,
                pred=(s >= 0.3).astype(int))


def test_all_shared_functions_exist_in_both(nb):
    for name in SHARED:
        assert name in nb, f"{name} missing from the notebook"
        assert hasattr(R, name), f"{name} missing from reference_impl"


def test_rate_comparison_functions_match(nb, sample):
    same(nb["two_proportion_test"](250, 1000, 300, 1100),
         R.two_proportion_test(250, 1000, 300, 1100))
    same(nb["wilson_interval"](37, 200), R.wilson_interval(37, 200))


def test_fitting_functions_match(nb, sample):
    X, t = sample["X"], sample["target"]
    same(nb["fit_normal_equation"](X, t), R.fit_normal_equation(X, t))
    same(nb["critical_learning_rate"](X), R.critical_learning_rate(X))

    w, b = R.fit_normal_equation(X, t)
    same(nb["mse_loss"](X, t, w, b), R.mse_loss(X, t, w, b))
    same(nb["fit_gradient_descent"](X, t, lr=0.05, n_iters=300),
         R.fit_gradient_descent(X, t, lr=0.05, n_iters=300))


def test_cross_features_match(nb, sample):
    names = ["a", "b", "c", "d"]
    same(nb["make_cross_features"](sample["X"], list(names)),
         R.make_cross_features(sample["X"], list(names)))


def test_metric_functions_match(nb, sample):
    y, pred, s = sample["y"], sample["pred"], sample["s"]
    same(nb["confusion_counts"](y, pred), R.confusion_counts(y, pred))
    for fn in ["accuracy_manual", "precision_manual", "recall_manual", "f1_manual"]:
        same(nb[fn](y, pred), getattr(R, fn)(y, pred))

    same(nb["roc_curve_manual"](y, s), R.roc_curve_manual(y, s))
    fpr, tpr, _ = R.roc_curve_manual(y, s)
    same(nb["auc_trapezoid"](fpr, tpr), R.auc_trapezoid(fpr, tpr))
    same(nb["auc_via_ranks"](y, s), R.auc_via_ranks(y, s))


def test_bootstrap_functions_match(nb, sample):
    y, s, s2 = sample["y"], sample["s"], sample["s2"]
    same(nb["bootstrap_metric"](y, s, roc_auc_score, n_boot=100, seed=4),
         R.bootstrap_metric(y, s, roc_auc_score, n_boot=100, seed=4))
    same(nb["paired_bootstrap_auc_diff"](y, s, s2, n_boot=100, seed=4),
         R.paired_bootstrap_auc_diff(y, s, s2, n_boot=100, seed=4))


def test_delong_functions_match(nb, sample):
    y, s, s2 = sample["y"], sample["s"], sample["s2"]
    same(nb["midrank"](s), R.midrank(s))
    same(nb["delong_auc_cov"](np.vstack([s, s2]), y),
         R.delong_auc_cov(np.vstack([s, s2]), y))
    same(nb["delong_test"](y, s, s2), R.delong_test(y, s, s2))


def test_find_anomalies_matches(nb, sample):
    class Dummy:
        def __init__(self, scores):
            self.scores = scores

        def predict(self, frame):
            return self.scores[: len(frame)]

    n = len(sample["y"])
    frame = pd.DataFrame({"x": np.arange(n)})
    model = Dummy(sample["s"])
    same(nb["find_anomalies"](model, frame, sample["y"], percentile=97.0),
         R.find_anomalies(model, frame, sample["y"], percentile=97.0))
    same(nb["find_anomalies"](model, frame, sample["y"], percentile=97.0, signed=True),
         R.find_anomalies(model, frame, sample["y"], percentile=97.0, signed=True))
    same(nb["find_anomalies"](model, frame, sample["y"], abs_threshold=1.0),
         R.find_anomalies(model, frame, sample["y"], abs_threshold=1.0))
