"""Tests for home assignment 1.

Three jobs:

1. the **dataset** still has the properties every task's expected answer
   depends on (if the generator drifts, the marking scheme silently rots);
2. the **solution** runs end to end with every self-check passing;
3. the **assignment** is byte-identical to the solution outside the blanked
   task regions, and its stubs fail loudly rather than silently.
"""

from __future__ import annotations

import ast
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import generate_logs as G  # noqa: E402
import make_assignment as M  # noqa: E402

DATA = ROOT / "data" / "edge_gateway_logs.csv"
SOLUTION = ROOT / "solution.py"
ASSIGNMENT = ROOT / "assignment.py"
DEV_END = pd.Timestamp("2026-03-06")


@pytest.fixture(scope="module")
def raw():
    return pd.read_csv(DATA, parse_dates=["timestamp"])


@pytest.fixture(scope="module")
def clean(raw):
    c = raw.drop_duplicates().copy()
    c["p99_latency_ms"] = c["p99_latency_ms"].replace(-1.0, np.nan)
    return c


@pytest.fixture(scope="module")
def dev(clean):
    d = clean[clean.timestamp < DEV_END].copy()
    d[G.NUMERIC] = d[G.NUMERIC].fillna(d[G.NUMERIC].median())
    return d


# ==========================================================================
# the dataset the marking scheme assumes
# ==========================================================================


def test_dataset_exists_and_has_expected_shape(raw):
    assert DATA.exists(), "run src/generate_logs.py"
    assert len(raw) == 34_620
    for col in ["timestamp", "instance_id", "region", "instance_type",
                "service_tier", "deploy_phase", "incident"] + G.NUMERIC:
        assert col in raw.columns


def test_part1_has_exactly_three_data_problems(raw):
    """Task 1.2 tells the student there are three. There must be three."""
    assert raw.duplicated().sum() == G.N_DUPLICATE_ROWS
    assert raw[G.NUMERIC].isna().sum().sum() > 0
    assert (raw.p99_latency_ms == -1.0).sum() > 0
    # ...and no *other* impossible values, or the task statement is a lie.
    ok = raw[raw.p99_latency_ms != -1.0]
    assert (ok.p99_latency_ms > 0).all()
    assert ok.cpu_util.between(0, 1).all()
    assert ok.cache_hit_ratio.dropna().between(0, 1).all()
    assert (ok.queue_depth >= 0).all()


def test_dev_and_horizon_are_both_substantial(clean):
    dev = clean[clean.timestamp < DEV_END]
    hor = clean[clean.timestamp >= DEV_END]
    assert len(dev) == 17_280 and len(hor) == 17_280


def test_five_phases_of_equal_size(dev):
    counts = dev.deploy_phase.value_counts()
    assert set(counts.index) == set("ABCDE")
    assert counts.min() == counts.max() == 3456


def _two_prop(x1, n1, x2, n2):
    p = (x1 + x2) / (n1 + n2)
    se = np.sqrt(p * (1 - p) * (1 / n1 + 1 / n2))
    z = (x2 / n2 - x1 / n1) / se
    return z, 2 * stats.norm.sf(abs(z))


@pytest.fixture(scope="module")
def deploy_pvalues(dev):
    out = []
    for a, b in zip("ABCD", "BCDE"):
        ga = dev[dev.deploy_phase == a].incident
        gb = dev[dev.deploy_phase == b].incident
        out.append(_two_prop(ga.sum(), len(ga), gb.sum(), len(gb))[1])
    return np.array(out)


def test_deployment_effects_are_a_deliberate_mix(deploy_pvalues):
    """Two clearly real, one plainly null, one borderline. This is Part 2."""
    p1, p2, p3, p4 = deploy_pvalues
    assert p1 > 0.20, f"D1 should be plainly null, got p={p1:.4f}"
    assert p2 < 1e-4, f"D2 should be unmissable, got p={p2:.4g}"
    assert p3 < 1e-3, f"D3 should be unmissable, got p={p3:.4g}"
    assert 0.025 < p4 < 0.05, (
        f"D4 must be significant alone but not after correction; got p={p4:.4f}")


def test_multiple_testing_correction_changes_a_conclusion(deploy_pvalues):
    """The entire point of Task 2.3."""
    alpha, m = 0.05, len(deploy_pvalues)
    raw_sig = deploy_pvalues < alpha
    bonf = deploy_pvalues < alpha / m

    order = np.argsort(deploy_pvalues)
    holm = np.zeros(m, bool)
    for rank, i in enumerate(order):
        if deploy_pvalues[i] <= alpha / (m - rank):
            holm[i] = True
        else:
            break

    assert raw_sig.sum() == 3, "expected three raw-significant deployments"
    assert bonf.sum() == 2 and holm.sum() == 2
    assert (raw_sig & ~holm).sum() == 1, "exactly one verdict must flip"
    assert (raw_sig & ~bonf).sum() == 1


def test_categoricals_carry_signal_beyond_the_numerics(dev):
    """Task 4.3 must be able to earn its marks."""
    for col in ["region", "instance_type", "service_tier"]:
        rates = dev.groupby(col).incident.mean()
        assert rates.max() - rates.min() > 0.04, f"{col} barely matters"


def test_skewed_columns_are_actually_skewed(dev):
    """Task 4.2's log transform needs something to fix."""
    assert stats.skew(dev.p99_latency_ms) > 5
    assert stats.skew(dev.gc_pause_ms) > 3
    assert abs(stats.skew(np.log(dev.p99_latency_ms))) < 1.0


# ==========================================================================
# the model results each part reports
# ==========================================================================


@pytest.fixture(scope="module")
def model_results(clean, dev):
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import roc_auc_score

    def build_numeric(f):
        return pd.DataFrame({
            "cpu_util": f.cpu_util, "mem_util": f.mem_util,
            "log_p99_latency": np.log(f.p99_latency_ms),
            "log_request_rate": np.log(f.request_rate),
            "log1p_error_rate": np.log1p(f.error_rate_pct),
            "queue_depth": f.queue_depth.astype(float),
            "cache_hit_ratio": f.cache_hit_ratio,
            "log_gc_pause": np.log(f.gc_pause_ms),
            "log_active_connections": np.log(f.active_connections),
            "log_disk_io_wait": np.log(f.disk_io_wait_ms + 0.05),
        }, index=f.index)

    cut = dev.timestamp.quantile(0.75)
    tr = (dev.timestamp < cut).to_numpy()
    te = ~tr
    y = dev.incident.to_numpy()

    def auc(X):
        m = LinearRegression().fit(X[tr], y[tr])
        return roc_auc_score(y[te], m.predict(X[te]))

    raw_X = dev[G.NUMERIC].to_numpy(float)
    mu, sd = raw_X[tr].mean(0), raw_X[tr].std(0)
    log_X = build_numeric(dev).to_numpy(float)
    cats = pd.get_dummies(dev[["region", "instance_type", "service_tier"]],
                          drop_first=True).to_numpy(float)
    z = (log_X - log_X[tr].mean(0)) / log_X[tr].std(0)
    i, j = np.triu_indices(z.shape[1])
    full = np.hstack([z, (z[:, :, None] * z[:, None, :])[:, i, j], cats])

    return {
        "raw": auc(raw_X),
        "standardised": auc((raw_X - mu) / sd),
        "logs": auc(log_X),
        "cats": auc(np.hstack([log_X, cats])),
        "full": auc(full),
    }


def test_baseline_leaves_room_to_improve(model_results):
    assert 0.78 < model_results["raw"] < 0.83, (
        f"baseline {model_results['raw']:.4f} leaves the wrong amount of headroom")


def test_standardisation_changes_nothing_at_all(model_results):
    """Task 4.1's checkpoint. If this ever fails, the task's answer is wrong."""
    assert np.isclose(model_results["standardised"], model_results["raw"], atol=1e-9)


def test_each_feature_engineering_step_pays(model_results):
    r = model_results
    assert r["logs"] - r["raw"] > 0.008, "log transforms must visibly pay"
    assert r["cats"] - r["logs"] > 0.02, "one-hot must be the biggest single win"
    assert r["full"] - r["cats"] > 0.008, "interactions must pay"
    assert r["full"] >= 0.87, f"the stated 0.87 target must be reachable, got {r['full']:.4f}"


# ==========================================================================
# the assignment file
# ==========================================================================


def test_assignment_is_up_to_date():
    assert ASSIGNMENT.exists(), "run src/make_assignment.py"
    assert M.transform(SOLUTION.read_text()) == ASSIGNMENT.read_text(), \
        "assignment.py is stale -- rerun src/make_assignment.py"


def test_both_files_parse():
    ast.parse(SOLUTION.read_text())
    ast.parse(ASSIGNMENT.read_text())


def test_every_task_marker_is_paired():
    text = SOLUTION.read_text()
    opens = M.OPEN_RE.findall(text) if False else [
        m.group(2) for m in map(M.OPEN_RE.match, text.splitlines()) if m]
    closes = [m.group(2) for m in map(M.CLOSE_RE.match, text.splitlines()) if m]
    assert opens == closes, "task markers are not properly nested/paired"
    assert len(opens) == len(set(opens)), "duplicate task ids"
    assert set(opens) == set(M.TASKS), "TASKS dict and markers disagree"


def test_assignment_contains_no_solution_code():
    """The blanked regions must not leak an answer.

    These needles are fragments of solution *bodies*. Deliberately NOT checked:
    names like `np.linalg.solve` or `np.triu_indices`, which appear in the task
    hints on purpose -- pointing at the right tool is the help we intend to
    give; writing the line for them is not.
    """
    text = ASSIGNMENT.read_text()
    body_fragments = [
        "p_pool",                      # two-proportion test internals
        "z ** 2 / (2 * n)",            # the Wilson formula
        "1 - (1 - alpha)",             # the family-wise error rate
        "alpha / (m - rank)",          # the Holm step-down
        "drop_duplicates",             # the Task 1.2 fix
        "replace(-1.0",                # the sentinel fix
        "n_pos * (n_pos + 1) / 2",     # the AUC rank identity
        "np.sign(coef.dev)",           # the Task 5.2 sign-flip diagnosis
    ]
    leaked = [n for n in body_fragments if n in text]
    assert not leaked, f"assignment.py still contains solution code: {leaked}"
    assert "# ---- Task" not in text, "sentinel markers leaked into the assignment"


def test_assignment_keeps_the_self_checks():
    """Students must keep the PASS/FAIL feedback loop."""
    sol_checks = SOLUTION.read_text().count("check(")
    asn_checks = ASSIGNMENT.read_text().count("check(")
    assert asn_checks >= sol_checks - 2, (
        f"self-checks were blanked away: {asn_checks} vs {sol_checks}")
    assert asn_checks > 20


def test_nothing_outside_task_regions_is_lost():
    """Every line the solution keeps outside a task region survives verbatim.

    That covers all the task statements, the marking scheme, the imports, the
    plotting style and the self-checks -- i.e. everything the student is
    supposed to receive unchanged.
    """
    assignment_text = ASSIGNMENT.read_text()

    kept, inside = [], False
    for line in SOLUTION.read_text().splitlines():
        if M.OPEN_RE.match(line):
            inside = True
            continue
        if M.CLOSE_RE.match(line):
            inside = False
            continue
        if not inside:
            kept.append(line)

    missing = [l for l in kept if l.strip() and l not in assignment_text]
    assert not missing, f"lost from the assignment: {missing[:5]}"

    # ...and the task statements specifically, since those carry the marks.
    for part in ["Task 1.2", "Task 2.3", "Task 3.3", "Task 4.1", "Task 5.2"]:
        assert part in assignment_text, f"{part}'s statement is missing"


def test_assignment_stubs_raise_not_implemented():
    """A stub must fail loudly, not silently return None."""
    text = ASSIGNMENT.read_text()
    n = len(re.findall(r"raise NotImplementedError", text))
    assert n >= 15, f"only {n} stubs raise NotImplementedError"


# ==========================================================================
# end to end
# ==========================================================================


@pytest.mark.slow
def test_solution_runs_with_all_checks_passing(tmp_path):
    jupytext = Path(sys.executable).parent / "jupytext"
    jupyter = Path(sys.executable).parent / "jupyter"
    if not jupytext.exists():
        pytest.skip("jupytext not installed")

    import shutil
    work = tmp_path / "assignment01"
    work.mkdir()
    shutil.copytree(ROOT / "data", work / "data")
    (work / "figures").mkdir()
    shutil.copy(SOLUTION, work / "solution.py")

    subprocess.run([str(jupytext), "--to", "ipynb", "solution.py",
                    "-o", "solution.ipynb"], cwd=work, check=True, capture_output=True)
    r = subprocess.run([str(jupyter), "nbconvert", "--to", "notebook", "--execute",
                        "--inplace", "--ExecutePreprocessor.timeout=900",
                        "solution.ipynb"], cwd=work, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[-3000:]

    import nbformat
    nb = nbformat.read(work / "solution.ipynb", as_version=4)
    errors = [(i, o.ename) for i, c in enumerate(nb.cells)
              for o in c.get("outputs", []) if o.output_type == "error"]
    assert not errors, f"cells raised: {errors}"

    text = "".join(o.get("text", "") for c in nb.cells
                   for o in c.get("outputs", []) if o.output_type == "stream")
    assert "[FAIL]" not in text, "a self-check failed in the solution"
    assert text.count("[PASS]") >= 20


@pytest.mark.slow
def test_assignment_stops_at_the_first_task(tmp_path):
    """The student's copy must fail with a clear NotImplementedError, early."""
    jupytext = Path(sys.executable).parent / "jupytext"
    jupyter = Path(sys.executable).parent / "jupyter"
    if not jupytext.exists():
        pytest.skip("jupytext not installed")

    import shutil
    work = tmp_path / "assignment01"
    work.mkdir()
    shutil.copytree(ROOT / "data", work / "data")
    (work / "figures").mkdir()
    shutil.copy(ASSIGNMENT, work / "assignment.py")

    subprocess.run([str(jupytext), "--to", "ipynb", "assignment.py",
                    "-o", "assignment.ipynb"], cwd=work, check=True, capture_output=True)
    r = subprocess.run([str(jupyter), "nbconvert", "--to", "notebook", "--execute",
                        "--inplace", "--allow-errors",
                        "--ExecutePreprocessor.timeout=300",
                        "assignment.ipynb"], cwd=work, capture_output=True, text=True)
    assert r.returncode == 0

    import nbformat
    nb = nbformat.read(work / "assignment.ipynb", as_version=4)
    errors = [o.ename for c in nb.cells for o in c.get("outputs", [])
              if o.output_type == "error"]
    assert errors, "the assignment should not run to completion as shipped"
    assert errors[0] == "NotImplementedError", (
        f"first failure should be a clear NotImplementedError, got {errors[0]}")
