"""Generate `assignment.py` (the version students receive) from `solution.py`.

`solution.py` marks each answer with a sentinel pair:

    # ---- Task 3.2 ------------------------------------------------------
    <the answer>
    # ---- /Task 3.2 -----------------------------------------------------

This script replaces everything between the markers with a TODO block. Anything
outside the markers — the task statements, the marking scheme, the plotting
style, and crucially the `check(...)` self-tests — is copied through **byte for
byte**, so the assignment cannot drift from the solution.

Why sentinels here rather than the AST-based blanking used in the lessons: a
homework answer is not always a function. Several tasks are free-standing
analysis (build a table, run a sweep, draw a chart), and a marker pair blanks
those just as cleanly as it blanks a `def`.

Usage
-----
    python src/make_assignment.py            # write assignment.py
    python src/make_assignment.py --check    # verify it is up to date
"""

from __future__ import annotations

import argparse
import ast
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "solution.py"
TARGET = ROOT / "assignment.py"

OPEN_RE = re.compile(r"^(\s*)# ---- Task ([0-9.]+[a-z]?) -*\s*$")
CLOSE_RE = re.compile(r"^(\s*)# ---- /Task ([0-9.]+[a-z]?) -*\s*$")

# Per-task scaffolding for the student. `hint` is optional; `stub` lines are
# emitted as real code so the notebook still runs top to bottom (raising a
# clear NotImplementedError) instead of dying on a NameError three cells later.
TASKS: dict[str, dict] = {
    "1.1": dict(todo=["Load the CSV (parse `timestamp` as a datetime) into `logs`,",
                      "then print shape / period / dtypes and show `.head()`."],
                names=["logs"]),
    "1.2": dict(todo=["Report the count of each of the three problems, then build",
                      "`clean`: de-duplicated, with the -1 latency sentinel turned",
                      "into a proper NaN.",
                      "",
                      "Do NOT drop the rows with missing values -- that would throw",
                      "away ~1000 perfectly good rows. Impute them in Task 1.3."],
                names=["clean"]),
    "1.3": dict(todo=["Build `dev` (before DEV_END) and `horizon` (from DEV_END on),",
                      "then fill the missing numeric values.",
                      "",
                      "The imputation statistic must be computed on `dev` only, and",
                      "then applied to BOTH frames. Computing it over the whole file",
                      "leaks the future into your model.",
                      "",
                      "Store the values you used in `impute_values` -- Part 5 needs",
                      "them again."],
                names=["dev", "horizon", "impute_values"]),
    "1.4": dict(todo=["Three panels: p99_latency_ms raw, the same log-transformed,",
                      "and the incident rate by hour of day.",
                      "Write one sentence under the figure about each."]),
    "2.1": dict(todo=["Implement `wilson_interval(x, n, conf=0.95)` by hand, then",
                      "build `phase_stats`: one row per phase A-E with columns",
                      "phase / windows / incidents / rate / ci_low / ci_high."],
                names=["wilson_interval", "phase_stats"],
                stub=["def wilson_interval(x, n, conf=0.95):",
                      "    raise NotImplementedError('wilson_interval')",
                      "",
                      "",
                      "phase_stats = None"]),
    "2.2": dict(todo=["Implement `two_proportion_test(x1, n1, x2, n2)` returning",
                      "(difference, z, p_value), then apply it to all four",
                      "deployments and collect the answers in a DataFrame",
                      "`results` with a `p_value` and a `z` column.",
                      "",
                      "Then cross-check every one against",
                      "`stats.chi2_contingency(table, correction=False)` and collect",
                      "the booleans in `chi2_matches`."],
                names=["two_proportion_test", "results", "chi2_matches"],
                stub=["def two_proportion_test(x1, n1, x2, n2):",
                      "    raise NotImplementedError('two_proportion_test')",
                      "",
                      "",
                      "results = None",
                      "chi2_matches = []"]),
    "2.3": dict(todo=["Implement `holm_bonferroni(pvals, alpha=0.05)` -> boolean array.",
                      "",
                      "  sort the p-values ascending; the k-th smallest (k from 1) is",
                      "  compared against alpha / (m - k + 1); STOP at the first",
                      "  failure and reject nothing after it.",
                      "",
                      "Add `sig_raw`, `sig_bonferroni` and `sig_holm` columns to",
                      "`results`, print the family-wise error rate of 4 uncorrected",
                      "tests, and print which deployments change verdict."],
                names=["holm_bonferroni"],
                stub=["def holm_bonferroni(pvals, alpha=0.05):",
                      "    raise NotImplementedError('holm_bonferroni')"]),
    "2.4": dict(todo=["Plot the per-phase rate with its 95% interval and mark the",
                      "four deployments. The intervals are the point of the chart."]),
    "3.1": dict(todo=["Split `dev` by TIME at the 75% quantile of `timestamp`.",
                      "Define `tr_mask`, `te_mask` (boolean numpy arrays) and",
                      "`y_tr`, `y_te`."],
                names=["tr_mask", "te_mask", "y_tr", "y_te"],
                stub=["split_at = None",
                      "tr_mask = te_mask = None",
                      "y_tr = y_te = None"]),
    "3.2": dict(todo=["Implement `fit_ols(X, y)` -> (weights, intercept) using the",
                      "normal equation and `np.linalg.solve`.",
                      "Fit it on the raw numeric columns and print the weights.",
                      "Keep the design matrix in `X_raw` and the fit in w_raw/b_raw."],
                names=["fit_ols", "X_raw", "w_raw", "b_raw"],
                stub=["def fit_ols(X, y):",
                      "    raise NotImplementedError('fit_ols')",
                      "",
                      "",
                      "X_raw = dev[NUMERIC].to_numpy(float)",
                      "w_raw = b_raw = None"]),
    "3.3": dict(todo=["Implement `auc_score(y_true, score)` with the rank identity.",
                      "Store the baseline test AUC in `AUC_BASELINE`."],
                names=["auc_score", "AUC_BASELINE"],
                stub=["def auc_score(y_true, score):",
                      "    raise NotImplementedError('auc_score')",
                      "",
                      "",
                      "scores_te = None",
                      "AUC_BASELINE = None"]),
    "3.4": dict(todo=["Compute the AUC of random scores and of each single feature.",
                      "Put the per-feature AUCs in a Series `single` and the name of",
                      "the most informative one in `best_feature`."],
                names=["single", "best_feature"],
                stub=["random_auc = None", "single = None", "best_feature = None"]),
    "4.1": dict(todo=["Implement `standardize(X, mean=None, std=None)` returning",
                      "(X_scaled, mean, std). Standardise with TRAIN statistics,",
                      "refit, and put the result in `auc_standardised`.",
                      "",
                      "Write down your prediction BEFORE running it."],
                names=["standardize", "auc_standardised"],
                stub=["def standardize(X, mean=None, std=None):",
                      "    raise NotImplementedError('standardize')",
                      "",
                      "",
                      "auc_standardised = None"]),
    "4.2": dict(todo=["Write `build_numeric(frame)` -> DataFrame with the skewed",
                      "columns log-transformed (use log1p where a column can be",
                      "exactly zero). Refit, and store the AUC in `auc_logs`.",
                      "",
                      "Keep the numeric design matrix in `X_log` -- later tasks",
                      "reuse it."],
                names=["build_numeric", "X_log", "auc_logs"],
                stub=["def build_numeric(frame):",
                      "    raise NotImplementedError('build_numeric')",
                      "",
                      "",
                      "X_log = None",
                      "auc_logs = None"]),
    "4.3": dict(todo=["Implement `one_hot(labels, categories=None, drop_first=True)`",
                      "and `encode_categoricals(frame, cats=None)`.",
                      "",
                      "`encode_categoricals` must be able to reuse the categories it",
                      "learned on `dev` (return them as `CATS`), or Part 5 will build",
                      "a different number of columns and the weights will not line up.",
                      "",
                      "Store the AUC in `auc_cats` and the design in `X_cat`."],
                names=["one_hot", "encode_categoricals", "cat_matrix", "cat_names",
                       "CATS", "auc_cats"],
                stub=["def one_hot(labels, categories=None, drop_first=True):",
                      "    raise NotImplementedError('one_hot')",
                      "",
                      "",
                      "def encode_categoricals(frame, cats=None):",
                      "    raise NotImplementedError('encode_categoricals')",
                      "",
                      "",
                      "cat_matrix = cat_names = CATS = None",
                      "auc_cats = None"]),
    "4.4": dict(todo=["Implement `add_interactions(X)` -> (expanded, pairs), adding",
                      "every product x_i * x_j with i <= j.",
                      "`np.triu_indices` plus broadcasting is the whole trick.",
                      "",
                      "Standardise the numeric block FIRST, then expand, then stick",
                      "the (already 0/1) dummies on the end. Store `X_full`, `w_f`,",
                      "`b_f` and `auc_full` -- Part 5 reuses all four."],
                names=["add_interactions", "X_full", "w_f", "b_f", "auc_full"],
                stub=["def add_interactions(X):",
                      "    raise NotImplementedError('add_interactions')",
                      "",
                      "",
                      "X_full = w_f = b_f = None",
                      "auc_full = None"]),
    "4.5": dict(todo=["Express the same preprocessing as a sklearn Pipeline with a",
                      "ColumnTransformer (StandardScaler / FunctionTransformer for",
                      "the numerics, OneHotEncoder for the categoricals,",
                      "PolynomialFeatures for the interactions).",
                      "Store the AUC in `auc_pipeline`."],
                names=["pipe", "auc_pipeline"],
                stub=["pipe = None", "auc_pipeline = None"]),
    "4.6": dict(todo=["Add hour-of-day to your best model, as sin/cos and as dummies,",
                      "and report both AUCs. Then build the summary `ladder` table,",
                      "set `BEST_AUC`, and plot the ladder as a horizontal bar chart.",
                      "",
                      "Predict what hour-of-day will do BEFORE you run it, and write",
                      "your explanation of the result in the markdown cell below."],
                names=["ladder", "BEST_AUC"],
                stub=["auc_cyc = auc_hod = None", "ladder = None", "BEST_AUC = None"]),
    "5.1": dict(todo=["Write `build_design(frame, mu, sd, cats)` that reproduces the",
                      "Task 4.4 design matrix for ANY frame, and",
                      "`rolling_auc(timestamps, y, scores, window_days, step_days)`",
                      "returning (centres, aucs, counts).",
                      "",
                      "Score the horizon with the Task 4.4 model and compute the",
                      "rolling AUC. Store `centres` and `aucs`."],
                names=["build_design", "rolling_auc", "centres", "aucs",
                       "mu_log", "sd_log", "hor_scores", "y_hor"],
                stub=["def build_design(frame, mu, sd, cats):",
                      "    raise NotImplementedError('build_design')",
                      "",
                      "",
                      "def rolling_auc(timestamps, y_true, scores, window_days=14,",
                      "                step_days=3):",
                      "    raise NotImplementedError('rolling_auc')",
                      "",
                      "",
                      "mu_log = sd_log = None",
                      "X_hor = y_hor = hor_scores = None",
                      "centres = aucs = counts = None"]),
    "5.1b": dict(todo=["Plot the rolling AUC over time, with a horizontal line for the",
                       "development-period AUC and a fitted linear trend."]),
    "5.2": dict(todo=["Quantify BOTH kinds of shift:",
                      "",
                      "  covariate shift -- compare feature means in `dev` against the",
                      "    last 21 days of the horizon, expressed in dev standard",
                      "    deviations;",
                      "  concept drift  -- refit on the last 21 days and compare the",
                      "    main-effect coefficients with the dev-trained ones. Build",
                      "    a `coef` frame with a `flipped_sign` column.",
                      "",
                      "Then say in words which one is doing the damage, and why."],
                names=["late", "coef"],
                stub=["late = None", "shift = None", "coef = None"]),
    "5.3": dict(todo=["Hold out the last 21 days as `final`; train on the 21 days",
                      "immediately before (`recent`). Compare the stale dev-trained",
                      "model against the retrained one on `final`.",
                      "Store `auc_stale`, `auc_fresh`, `cut`, `X_final`, `y_final`."],
                names=["cut", "recent", "final", "X_final", "y_final",
                       "auc_stale", "auc_fresh"],
                stub=["cut = None", "recent = final = None",
                      "X_final = y_final = None",
                      "auc_stale = auc_fresh = None"]),
    "5.4": dict(todo=["Sweep the training-window length (7, 14, 21, 30, 45, 60, 90",
                      "days before `cut`), refit on each, and evaluate on `final`.",
                      "Build a `sweep` DataFrame with `window_days` and `auc`, then",
                      "plot AUC against window length with the stale model's AUC as a",
                      "reference line.",
                      "",
                      "Careful: the older windows reach back before DEV_END, so build",
                      "them from `clean` (and impute with `impute_values`), not from",
                      "`horizon`.",
                      "",
                      "Is more data better here? Explain the shape you get."],
                names=["sweep"],
                stub=["sweep = None"]),
}


def build_block(indent: str, task_id: str) -> list[str]:
    spec = TASKS.get(task_id, {})
    todo = spec.get("todo", ["Your code here."])
    width = 78 - len(indent)

    out = [f"{indent}# {'=' * (width - 2)}",
           f"{indent}# TASK {task_id} — YOUR CODE HERE",
           f"{indent}# {'=' * (width - 2)}"]
    out += [f"{indent}# {line}".rstrip() for line in todo]
    if spec.get("names"):
        out.append(f"{indent}#")
        out.append(f"{indent}# Define: {', '.join(spec['names'])}")
    out.append(f"{indent}# {'=' * (width - 2)}")
    out.append("")
    stub = spec.get("stub", [f"raise NotImplementedError('Task {task_id}')"])
    out += [f"{indent}{line}".rstrip() for line in stub]
    return out


def transform(source: str) -> str:
    lines = source.splitlines()
    out: list[str] = []
    i = 0
    seen: list[str] = []

    while i < len(lines):
        m = OPEN_RE.match(lines[i])
        if not m:
            out.append(lines[i])
            i += 1
            continue

        indent, task_id = m.group(1), m.group(2)
        j = i + 1
        while j < len(lines):
            c = CLOSE_RE.match(lines[j])
            if c:
                if c.group(2) != task_id:
                    raise SystemExit(
                        f"marker mismatch: '# ---- Task {task_id}' on line {i+1} "
                        f"closed by '/Task {c.group(2)}' on line {j+1}")
                break
            if OPEN_RE.match(lines[j]):
                raise SystemExit(f"Task {task_id} (line {i+1}) never closed")
            j += 1
        else:
            raise SystemExit(f"Task {task_id} (line {i+1}) never closed")

        out += build_block(indent, task_id)
        seen.append(task_id)
        i = j + 1

    unknown = [t for t in seen if t not in TASKS]
    if unknown:
        raise SystemExit(f"no TODO text defined for tasks: {unknown}")
    unused = [t for t in TASKS if t not in seen]
    if unused:
        raise SystemExit(f"TODO text defined for missing tasks: {unused}")

    return "\n".join(out) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()

    generated = transform(SOURCE.read_text())
    ast.parse(generated)      # never emit something that will not parse

    if args.check:
        current = TARGET.read_text() if TARGET.exists() else ""
        if current != generated:
            sys.exit(f"{TARGET.name} is out of date -- rerun {Path(__file__).name}")
        print(f"{TARGET.name} is up to date")
        return

    TARGET.write_text(generated)
    n = len(OPEN_RE.findall("\n".join([])) or []) or len(TASKS)
    print(f"wrote {TARGET.name}  ({n} task regions blanked)")


if __name__ == "__main__":
    main()
