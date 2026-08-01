"""Tests for lesson 2.

Every implementation in the notebook is checked against an independent route to
the same number, plus the workshop variant is checked to be byte-identical to
the lesson outside the blanked bodies.

The notebook's functions are extracted straight out of `lesson02.py` with `ast`
(a jupytext py:percent file is valid Python), so there is no second copy to
drift.
"""

from __future__ import annotations

import ast
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from scipy import stats
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import make_workshop as W  # noqa: E402

LESSON = ROOT / "lesson02.py"
WORKSHOP = ROOT / "lesson02_workshop.py"
DATA = (ROOT / ".." / "lesson01" / "data" / "service_telemetry.csv").resolve()


@pytest.fixture(scope="module")
def nb():
    """Exec only the notebook's top-level function definitions."""
    tree = ast.parse(LESSON.read_text())
    defs = [n for n in tree.body if isinstance(n, ast.FunctionDef)]
    ns: dict = {}
    exec("import numpy as np\nfrom scipy import stats\n", ns)
    for node in defs:
        mod = ast.Module(body=[node], type_ignores=[])
        exec(compile(ast.fix_missing_locations(mod), "<nb>", "exec"), ns)
    return ns


@pytest.fixture(scope="module")
def data():
    X = np.loadtxt(DATA, delimiter=",", skiprows=1, usecols=range(4, 12))
    y = np.loadtxt(DATA, delimiter=",", skiprows=1, usecols=12, dtype=int)
    meta = np.loadtxt(DATA, delimiter=",", skiprows=1, usecols=(2, 3, 13), dtype=str)
    return X, y, meta[:, 0], meta[:, 1], meta[:, 2]


@pytest.fixture(scope="module")
def rng():
    return np.random.default_rng(4242)


# ==========================================================================
# §4 standardize
# ==========================================================================


def test_standardize_gives_zero_mean_unit_std(nb, data):
    X = data[0]
    Z, mean, std = nb["standardize"](X)
    assert np.allclose(Z.mean(axis=0), 0, atol=1e-10)
    assert np.allclose(Z.std(axis=0), 1, atol=1e-10)
    assert mean.shape == (X.shape[1],) and std.shape == (X.shape[1],)


def test_standardize_reuses_supplied_statistics(nb, data):
    """The leak-prevention path: val/test must use train's mean and std."""
    X = data[0]
    _, mean, std = nb["standardize"](X[:1000])
    Z, m2, s2 = nb["standardize"](X[1000:2000], mean, std)
    assert np.array_equal(mean, m2) and np.array_equal(std, s2)
    assert np.allclose(Z, (X[1000:2000] - mean) / std)
    # Applying foreign statistics must NOT re-centre to exactly zero.
    assert not np.allclose(Z.mean(axis=0), 0, atol=1e-6)


def test_standardize_survives_a_constant_column(nb):
    X = np.column_stack([np.arange(50.0), np.full(50, 3.0)])
    Z, _, std = nb["standardize"](X)
    assert np.all(np.isfinite(Z)), "a zero-variance column must not produce inf/nan"
    assert np.allclose(Z[:, 1], 0)


# ==========================================================================
# §5 vectorisation
# ==========================================================================


def test_loop_and_vectorised_scores_agree(nb, rng):
    X = rng.normal(size=(200, 6))
    w = rng.normal(size=6)
    assert np.allclose(nb["score_loop"](X, w), nb["score_vectorised"](X, w))
    assert np.allclose(nb["score_vectorised"](X, w), X @ w)


# ==========================================================================
# §6 sorting, percentiles, ranks
# ==========================================================================


def test_top_k_matches_a_full_sort(nb, rng):
    for k in [1, 5, 50]:
        v = rng.normal(size=500)
        idx = nb["top_k_indices"](v, k)
        assert idx.shape == (k,)
        assert np.allclose(v[idx], np.sort(v)[::-1][:k])


def test_percentile_matches_numpy(nb, data, rng):
    for q in [0, 1, 25, 50, 75, 99, 99.9, 100]:
        assert np.isclose(nb["percentile_manual"](data[0][:, 2], q),
                          np.percentile(data[0][:, 2], q))
    v = rng.normal(size=101)
    for q in [10, 33.3, 90]:
        assert np.isclose(nb["percentile_manual"](v, q), np.percentile(v, q))


def test_rank_average_matches_scipy(nb, data, rng):
    assert np.allclose(nb["rank_average"](data[0][:, 2]),
                       stats.rankdata(data[0][:, 2]))
    # heavy ties, which is where a naive argsort implementation fails
    tied = rng.integers(0, 5, size=300).astype(float)
    assert np.allclose(nb["rank_average"](tied), stats.rankdata(tied))


def test_rank_average_known_case(nb):
    got = nb["rank_average"](np.array([10.0, 20.0, 20.0, 20.0, 30.0, 10.0]))
    assert np.allclose(got, [1.5, 4.0, 4.0, 4.0, 6.0, 1.5])


# ==========================================================================
# §7 grouping
# ==========================================================================


def test_group_mean_matches_a_python_loop(nb, data):
    _, y, node, _, _ = data
    keys, means = nb["group_mean"](node, y.astype(float))
    for k, m in zip(keys, means):
        assert np.isclose(m, y[node == k].mean())
    assert keys.size == np.unique(node).size


def test_group_mean_weighted_average_recovers_the_global_mean(nb, data):
    _, y, node, _, _ = data
    keys, means = nb["group_mean"](node, y.astype(float))
    counts = np.array([(node == k).sum() for k in keys])
    assert np.isclose((counts * means).sum() / counts.sum(), y.mean())


def test_crosstab_matches_direct_masking(nb, data):
    _, y, node, phase, _ = data
    ka, kb, table = nb["crosstab_rate"](node, phase, y.astype(float))
    assert table.shape == (ka.size, kb.size)
    for i, a in enumerate(ka):
        for j, b in enumerate(kb):
            mask = (node == a) & (phase == b)
            assert np.isclose(table[i, j], y[mask].mean())


# ==========================================================================
# §8 design matrix
# ==========================================================================


def test_log_transform_does_not_mutate_its_input(nb, data):
    """The copy/view trap, as a regression test."""
    X = data[0].copy()
    before = X.copy()
    out = nb["apply_log_transform"](X, [2, 3])
    assert np.array_equal(X, before), "input array was modified in place"
    assert not np.shares_memory(out, X)
    assert np.allclose(out[:, 2], np.log1p(before[:, 2]))
    assert np.allclose(out[:, 5], before[:, 5]), "untouched columns must pass through"


def test_one_hot_is_a_valid_encoding(nb, data):
    node = data[2]
    M, keys = nb["one_hot"](node)
    assert M.shape == (node.size, keys.size)
    assert np.all(M.sum(axis=1) == 1)
    assert set(np.unique(M)) <= {0.0, 1.0}
    for j, k in enumerate(keys):
        assert np.array_equal(M[:, j] == 1, node == k)


def test_cross_features_are_the_right_products(nb, rng):
    X = rng.normal(size=(40, 5))
    out, pairs = nb["add_cross_features"](X)
    assert out.shape == (40, 5 + 15)          # 5 + C(5+1, 2)
    assert np.allclose(out[:, :5], X)
    for k, (i, j) in enumerate(pairs):
        assert np.allclose(out[:, 5 + k], X[:, i] * X[:, j])
    assert all(i <= j for i, j in pairs), "each product must be kept once"


# ==========================================================================
# §9 splitting
# ==========================================================================


def test_split_is_stratified_and_exhaustive(nb, data):
    y = data[1]
    a = nb["stratified_split"](y, seed=1)
    assert a.shape == y.shape
    assert set(np.unique(a)) == {0, 1, 2}
    rates = [y[a == p].mean() for p in (0, 1, 2)]
    assert max(rates) - min(rates) < 0.01, "class balance drifted across splits"
    sizes = [(a == p).sum() / y.size for p in (0, 1, 2)]
    assert np.allclose(sizes, [0.6, 0.2, 0.2], atol=0.01)


def test_split_is_deterministic_and_seed_sensitive(nb, data):
    y = data[1]
    assert np.array_equal(nb["stratified_split"](y, seed=7),
                          nb["stratified_split"](y, seed=7))
    assert not np.array_equal(nb["stratified_split"](y, seed=7),
                              nb["stratified_split"](y, seed=8))


# ==========================================================================
# §10 fitting
# ==========================================================================


def test_normal_equation_matches_sklearn(nb, rng):
    from sklearn.linear_model import LinearRegression
    X = rng.normal(size=(300, 5))
    y = X @ np.array([1.0, -2.0, 0.5, 0.0, 3.0]) + rng.normal(0, 0.3, 300) + 2.0
    w, b = nb["fit_normal_equation"](X, y)
    sk = LinearRegression().fit(X, y)
    assert np.allclose(w, sk.coef_, atol=1e-9)
    assert np.isclose(b, sk.intercept_, atol=1e-9)


def test_gradient_descent_reaches_the_closed_form(nb, rng):
    X = rng.normal(size=(400, 4))
    y = X @ np.array([1.0, -1.0, 2.0, 0.3]) + rng.normal(0, 0.2, 400) + 1.0
    w_exact, b_exact = nb["fit_normal_equation"](X, y)
    w, b, hist = nb["fit_gradient_descent"](X, y, lr=0.1, n_iters=8000)
    assert np.allclose(w, w_exact, atol=1e-6)
    assert np.isclose(b, b_exact, atol=1e-6)
    assert hist.shape == (8000,)
    assert np.all(np.diff(hist) <= 1e-12), "loss must decrease monotonically"


# ==========================================================================
# §11 metrics
# ==========================================================================


def test_confusion_matrix_matches_sklearn(nb, rng):
    for _ in range(20):
        n = int(rng.integers(50, 2000))
        y_true = rng.integers(0, 2, n)
        y_pred = rng.integers(0, 2, n)
        assert np.array_equal(nb["confusion_matrix_numpy"](y_true, y_pred),
                              confusion_matrix(y_true, y_pred, labels=[0, 1]))


def test_roc_curve_matches_sklearn(nb, rng):
    for ties in (False, True):
        n = 800
        y_true = rng.binomial(1, 0.3, n)
        score = rng.normal(y_true * 0.9, 1.0, n)
        if ties:
            score = np.round(score, 1)
        fpr, tpr = nb["roc_curve_numpy"](y_true, score)
        sk_fpr, sk_tpr, _ = roc_curve(y_true, score, drop_intermediate=False)
        assert np.allclose(fpr, sk_fpr)
        assert np.allclose(tpr, sk_tpr)


def test_auc_matches_sklearn_and_the_trapezoid(nb, rng):
    for ties in (False, True):
        n = 1200
        y_true = rng.binomial(1, 0.35, n)
        score = rng.normal(y_true * 1.0, 1.0, n)
        if ties:
            score = np.round(score, 1)
        fpr, tpr = nb["roc_curve_numpy"](y_true, score)
        expected = roc_auc_score(y_true, score)
        assert np.isclose(nb["auc_numpy"](y_true, score), expected, atol=1e-12)
        assert np.isclose(np.trapezoid(tpr, fpr), expected, atol=1e-12)


def test_auc_known_values(nb):
    y = np.array([0, 0, 1, 1])
    assert np.isclose(nb["auc_numpy"](y, np.array([0.1, 0.2, 0.8, 0.9])), 1.0)
    assert np.isclose(nb["auc_numpy"](y, np.array([0.9, 0.8, 0.2, 0.1])), 0.0)
    assert np.isclose(nb["auc_numpy"](y, np.array([0.5, 0.5, 0.5, 0.5])), 0.5)


# ==========================================================================
# workshop variant
# ==========================================================================


def test_workshop_is_up_to_date():
    assert WORKSHOP.exists(), "run src/make_workshop.py"
    assert W.transform(LESSON.read_text()) == WORKSHOP.read_text(), \
        "lesson02_workshop.py is stale -- rerun src/make_workshop.py"


def test_every_target_is_blanked_and_raises():
    text = WORKSHOP.read_text()
    funcs = {n.name: n for n in ast.parse(text).body
             if isinstance(n, ast.FunctionDef)}
    ns: dict = {}
    exec("import numpy as np\nfrom scipy import stats\n", ns)
    for name in W.TODOS:
        node = funcs[name]
        assert "NotImplementedError" in ast.get_source_segment(text, node)
        mod = ast.Module(body=[node], type_ignores=[])
        exec(compile(ast.fix_missing_locations(mod), "<wk>", "exec"), ns)
        n_required = len(node.args.args) - len(node.args.defaults)
        with pytest.raises(NotImplementedError):
            ns[name](*([None] * n_required))


def test_workshop_preserves_signatures_and_docstrings():
    src = {n.name: n for n in ast.parse(LESSON.read_text()).body
           if isinstance(n, ast.FunctionDef)}
    wk = {n.name: n for n in ast.parse(WORKSHOP.read_text()).body
          if isinstance(n, ast.FunctionDef)}
    for name in W.TODOS:
        assert ast.dump(src[name].args) == ast.dump(wk[name].args)
        assert ast.get_docstring(src[name]) == ast.get_docstring(wk[name])


def test_nothing_outside_the_blanked_bodies_changed():
    """Byte-identical everywhere except the replaced function bodies."""
    src_text, wk_text = LESSON.read_text(), WORKSHOP.read_text()
    src_tree, wk_tree = ast.parse(src_text), ast.parse(wk_text)
    assert len(src_tree.body) == len(wk_tree.body)

    src_lines, wk_lines = src_text.splitlines(), wk_text.splitlines()
    checked = 0
    for a, b in zip(src_tree.body, wk_tree.body):
        if isinstance(a, ast.FunctionDef) and a.name in W.TODOS:
            continue
        assert ast.dump(a) == ast.dump(b), f"statement at line {a.lineno} changed"
        assert (src_lines[a.lineno - 1:a.end_lineno]
                == wk_lines[b.lineno - 1:b.end_lineno])
        checked += 1
    assert checked > 30


def test_workshop_prose_is_identical():
    def prose(t):
        return [ln for ln in t.splitlines() if ln.startswith("#")]
    assert prose(LESSON.read_text()) == prose(WORKSHOP.read_text())


# ==========================================================================
# end to end
# ==========================================================================


@pytest.mark.slow
def test_notebook_executes_end_to_end(tmp_path):
    jupytext = Path(sys.executable).parent / "jupytext"
    jupyter = Path(sys.executable).parent / "jupyter"
    if not jupytext.exists():
        pytest.skip("jupytext not installed")

    work = tmp_path / "lesson02"
    work.mkdir()
    (work / "figures").mkdir()
    (work / "lesson01").mkdir()
    import shutil
    shutil.copytree(DATA.parent, work / "lesson01" / "data")
    shutil.copy(LESSON, work / "lesson02.py")

    subprocess.run([str(jupytext), "--to", "ipynb", "lesson02.py",
                    "-o", "lesson02.ipynb"], cwd=work, check=True,
                   capture_output=True)
    r = subprocess.run([str(jupyter), "nbconvert", "--to", "notebook", "--execute",
                        "--inplace", "--ExecutePreprocessor.timeout=900",
                        "lesson02.ipynb"], cwd=work, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[-3000:]

    import nbformat
    nb_out = nbformat.read(work / "lesson02.ipynb", as_version=4)
    errors = [(i, o.ename) for i, c in enumerate(nb_out.cells)
              for o in c.get("outputs", []) if o.output_type == "error"]
    assert not errors, f"cells raised: {errors}"
