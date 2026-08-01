"""Generate the fill-in-the-blanks variant of the lesson.

`lesson01_workshop.py` is `lesson01.py` with a chosen set of function *bodies*
replaced by TODO comments, for writing live with a student.

Selection rule: only implementations that are **plain syntax** -- arithmetic,
a loop, a comprehension, array indexing. Anything whose difficulty is a library
API or an intricate algorithm stays intact, because writing it live teaches
nothing. So DeLong's variance algebra (`midrank`, `delong_auc_cov`), the model
plumbing (`LinearProbabilityModel`, `train_model`), the scipy/sklearn calls and
every plotting cell are left exactly as they are.

The transformation is done with `ast`, so the signature and docstring of each
target are preserved byte for byte and **nothing outside the replaced bodies is
touched**. `tests/test_workshop.py` asserts exactly that.

Usage
-----
    python src/make_workshop.py            # writes lesson01_workshop.py
    python src/make_workshop.py --check    # verify it is up to date
"""

from __future__ import annotations

import argparse
import ast
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "lesson01.py"
TARGET = ROOT / "lesson01_workshop.py"

# Per-function TODO text. Each entry states the contract, not the solution --
# the derivation or definition is always in the markdown cell directly above.
TODOS: dict[str, list[str]] = {
    "two_proportion_test": [
        "TODO (§1)",
        "Pool the two samples to estimate one shared rate under H0, use it to",
        "build the standard error of the difference, then form z and a",
        "two-sided p-value.",
        "",
        "return: (difference, z, p_value, standard_error)",
        "hint: scipy's `stats.norm.sf` gives the upper tail.",
    ],
    "fit_normal_equation": [
        "TODO (§3)",
        "Implement w = (X^T X)^-1 X^T y, the formula derived just above.",
        "",
        "If `fit_intercept`, prepend a column of ones to X and split the first",
        "coefficient back out at the end.",
        "",
        "return: (weights, intercept)",
        "hint: use `np.linalg.solve(A, b)`, NOT `np.linalg.inv(A) @ b` --",
        "      forming the inverse explicitly is slower and less accurate.",
    ],
    "mse_loss": [
        "TODO (§4)",
        "Mean squared error of the predictions X @ w + b against y.",
    ],
    "fit_gradient_descent": [
        "TODO (§4)  -- the core exercise of this lesson",
        "",
        "Start from w = 0, b = 0. Repeat `n_iters` times:",
        "  1. residual = predictions - y",
        "  2. grad_w = (2/n) * X^T @ residual ;  grad_b = (2/n) * sum(residual)",
        "  3. step both parameters *against* the gradient, scaled by `lr`",
        "  4. every `record_every` iterations, append the current MSE to history",
        "",
        "return: (w, b, np.array(history))",
        "",
        "Note there is no matrix inverse anywhere, and no d x d matrix is ever",
        "built -- that is the whole point of the method.",
    ],
    "make_cross_features": [
        "TODO (§5)",
        "Append every product x_i * x_j with i <= j to the design matrix, so",
        "the model can express interactions while staying linear in its",
        "parameters.",
        "",
        "return: (expanded_matrix, names, pairs) where `pairs` is the list of",
        "(i, j) index pairs in the order the columns were added.",
        "",
        "Name a squared term f'{a}^2' and a product f'{a}*{b}'.",
        "hint: `itertools.combinations_with_replacement(range(d), 2)` gives",
        "      exactly the i <= j pairs, including i == j.",
    ],
    "confusion_counts": [
        "TODO (§6)",
        "Count the four cells of the confusion matrix, without library help.",
        "",
        "return: (tn, fp, fn, tp)   -- this order matches sklearn's",
        "                             `confusion_matrix(...).ravel()`",
        "hint: cast both arrays to bool and combine them with & and ~.",
    ],
    "accuracy_manual": [
        "TODO (§6)   accuracy = (TP + TN) / everything",
    ],
    "precision_manual": [
        "TODO (§6)   precision = TP / (TP + FP)",
        "",
        "Of the windows we flagged, what fraction were real?",
        "Careful: if nothing was flagged the denominator is 0. Return 0.0",
        "there, which is what sklearn does with `zero_division=0` and keeps a",
        "threshold sweep plottable.",
    ],
    "recall_manual": [
        "TODO (§6)   recall = TP / (TP + FN)",
        "",
        "Of the real incidents, what fraction did we catch?",
        "Same divide-by-zero care as precision.",
    ],
    "f1_manual": [
        "TODO (§7)   F1 = harmonic mean of precision and recall",
        "",
        "        F1 = 2 * p * r / (p + r)",
        "",
        "Reuse the two functions you just wrote. Return 0.0 when p + r == 0.",
    ],
    "roc_curve_manual": [
        "TODO (§7)",
        "Build the ROC curve from scratch.",
        "",
        "  1. sort the objects by DESCENDING score",
        "  2. sweeping the threshold down that sorted list moves one object at a",
        "     time from 'predicted 0' to 'predicted 1' -- so the running TP and",
        "     FP counts are just `np.cumsum` of the sorted labels",
        "  3. only thresholds strictly *between distinct scores* are real",
        "     operating points, so keep the last index of each run of equal",
        "     scores (tied objects must be committed to as a block)",
        "  4. divide by the totals to get rates, and prepend the (0, 0) origin",
        "",
        "return: (fpr, tpr, thresholds), matching",
        "        `sklearn.metrics.roc_curve(..., drop_intermediate=False)`",
        "hint: `np.argsort(-score, kind='mergesort')` sorts descending and is",
        "      stable; `np.flatnonzero(np.diff(sorted_scores))` finds the ends",
        "      of tied runs.",
    ],
    "auc_via_ranks": [
        "TODO (§7)",
        "Compute the same AUC a completely different way, via the",
        "Mann-Whitney U statistic:",
        "",
        "        U = R+ - n+ * (n+ + 1) / 2,      AUC = U / (n+ * n-)",
        "",
        "where R+ is the sum of the ranks of the positives in the pooled",
        "sample. This is the identity DeLong's test is built on in §10.",
        "",
        "hint: `stats.rankdata` already averages ranks across ties, which is",
        "      exactly the convention the formula needs.",
    ],
    "bootstrap_metric": [
        "TODO (§9)",
        "Percentile bootstrap for any metric of the form metric(y_true, score).",
        "",
        "Repeat `n_boot` times: draw a resample of the SAME SIZE with",
        "replacement, recompute the metric on it, store the result.",
        "",
        "If `stratified`, resample within each class separately so n+ and n-",
        "stay fixed -- otherwise a resample can contain no positives at all and",
        "AUC becomes undefined.",
        "",
        "return: np.array of `n_boot` metric values",
        "hint: seed with `np.random.default_rng(seed)` so the result is",
        "      reproducible; `rng.choice(idx, idx.size, replace=True)`.",
    ],
    "find_anomalies": [
        "TODO (§11)",
        "Score `frame`, form the residual y - prediction, and flag the extreme",
        "tail.",
        "",
        "  - `signed=False` (default): rank by |residual|",
        "  - `signed=True`           : rank by the raw residual, keeping one tail",
        "  - `abs_threshold` given   : use it directly as the cutoff",
        "  - otherwise               : cutoff = that percentile of the errors",
        "",
        "return: (flagged_rows_dataframe, cutoff) where the DataFrame is indexed",
        "        like `frame`, carries columns score / y_true / residual / error,",
        "        and is sorted by decreasing error.",
        "",
        "Remember why the `abs_threshold` branch exists: a percentile always",
        "flags (100 - p)% of the rows, so its count can never tell you whether",
        "anomalies are present.",
    ],
}


def build_stub(indent: str, name: str, lines: list[str]) -> list[str]:
    """Render a TODO block plus a loud failure, at the given indentation.

    The `raise` is what makes an unfinished stub fail immediately instead of
    silently returning None and producing a confusing error three cells later.
    """
    out = [f"{indent}# {line}".rstrip() for line in lines]
    out.append(f"{indent}raise NotImplementedError({name!r})")
    return out


def transform(source: str) -> str:
    tree = ast.parse(source)
    lines = source.splitlines()

    targets = {}
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name in TODOS:
            targets[node.name] = node

    missing = set(TODOS) - set(targets)
    if missing:
        raise SystemExit(f"functions not found at module level: {sorted(missing)}")

    # Replace from the back, so earlier line numbers stay valid.
    out = list(lines)
    for node in sorted(targets.values(), key=lambda n: n.lineno, reverse=True):
        first = node.body[0]
        # Keep the signature, and the docstring if there is one.
        keep_through = (first.end_lineno
                        if (isinstance(first, ast.Expr)
                            and isinstance(first.value, ast.Constant)
                            and isinstance(first.value.value, str))
                        else first.lineno - 1)
        indent = " " * first.col_offset
        stub = build_stub(indent, node.name, TODOS[node.name])
        out[keep_through:node.end_lineno] = stub

    return "\n".join(out) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true",
                        help="fail if the workshop file is stale")
    args = parser.parse_args()

    generated = transform(SOURCE.read_text())
    ast.parse(generated)  # never emit a file that does not parse

    if args.check:
        current = TARGET.read_text() if TARGET.exists() else ""
        if current != generated:
            sys.exit(f"{TARGET.name} is out of date -- rerun {Path(__file__).name}")
        print(f"{TARGET.name} is up to date")
        return

    TARGET.write_text(generated)
    print(f"wrote {TARGET.relative_to(ROOT)}  "
          f"({len(TODOS)} implementations replaced with TODOs)")
    for name in TODOS:
        print(f"  - {name}")


if __name__ == "__main__":
    main()
