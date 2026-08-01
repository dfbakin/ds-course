"""Generate the fill-in-the-blanks variant of lesson 2.

Same machinery as `lesson01/src/make_workshop.py`: replace a chosen set of
function *bodies* with TODO comments using `ast`, so signatures and docstrings
survive byte for byte and nothing outside the replaced bodies is touched.

Selection rule here is different from lesson 1, because the lesson is different.
Lesson 2 is *about* implementation, so almost every function is fair game --
the blanks are the point, not a subset of it. What stays intact is only the
scaffolding: plotting, the style block, and `run_pipeline` (which is just a
composition of the functions the student will have written by then).

Usage
-----
    python src/make_workshop.py            # writes lesson02_workshop.py
    python src/make_workshop.py --check    # verify it is up to date
"""

from __future__ import annotations

import argparse
import ast
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "lesson02.py"
TARGET = ROOT / "lesson02_workshop.py"

TODOS: dict[str, list[str]] = {
    "standardize": [
        "TODO (§4)   z = (x - mean) / std, per COLUMN",
        "",
        "Which axis disappears? You want one mean per feature, so the 20000",
        "must go: `axis=0`.",
        "",
        "If `mean`/`std` are passed in, use them instead of recomputing --",
        "that is how you apply TRAIN statistics to val/test without leaking.",
        "",
        "Guard against a constant column: std == 0 would give inf. Replace",
        "those with 1.0 (`np.where`) before dividing.",
        "",
        "return: (X_scaled, mean, std)",
    ],
    "score_loop": [
        "TODO (§5)   the deliberately slow version",
        "",
        "Two nested Python loops: for each row i, accumulate sum_j X[i,j]*w[j].",
        "Write it once so the vectorised version below has something to beat.",
        "",
        "return: 1-D array of length X.shape[0]",
    ],
    "score_vectorised": [
        "TODO (§5)   the same thing in one expression",
        "",
        "hint: this is exactly what the `@` operator does.",
    ],
    "top_k_indices": [
        "TODO (§6)   indices of the k largest values, largest first",
        "",
        "`np.argpartition(values, -k)[-k:]` gets you the k largest in O(n),",
        "but in ARBITRARY order. Sort just those k to order them, then reverse.",
        "",
        "return: array of k indices",
        "hint: `values[part]` then `np.argsort(...)[::-1]`, and remember to",
        "      index back into `part` -- not into `values`.",
    ],
    "percentile_manual": [
        "TODO (§6)   linear-interpolation percentile (NumPy's default method)",
        "",
        "  1. sort the values",
        "  2. the target position is (q/100) * (n - 1)  -- usually FRACTIONAL",
        "  3. interpolate between the neighbours either side of that position",
        "",
        "The fractional position is the whole exercise; `s[int(pos)]` is the",
        "off-by-a-bit bug this is designed to make you meet.",
    ],
    "rank_average": [
        "TODO (§6)   ranks 1..n, with ties sharing their average rank",
        "",
        "  1. `np.argsort` gives the order; scatter 1..n back through it to get",
        "     ordinal ranks (`ranks[order] = np.arange(1, n+1)`)",
        "  2. `np.unique(values, return_inverse=True, return_counts=True)`",
        "     gives each row a group code and each group its size",
        "  3. `np.bincount(inverse, weights=ranks)` sums ranks per group;",
        "     divide by the counts and index back out with `inverse`",
        "",
        "Step 1 alone is wrong whenever two values are equal -- which on real",
        "data is always.",
    ],
    "group_mean": [
        "TODO (§7)   pandas' groupby().mean(), in three lines",
        "",
        "  1. `np.unique(labels, return_inverse=True)` -> distinct keys, and an",
        "     integer code per row",
        "  2. `np.bincount(codes, weights=values)` sums values into buckets",
        "  3. `np.bincount(codes)` counts them; divide",
        "",
        "return: (keys, means)",
        "Pass `minlength=keys.size` so an empty trailing group is not dropped.",
    ],
    "crosstab_rate": [
        "TODO (§7)   the same idea for TWO grouping columns",
        "",
        "Get codes for both label arrays, then flatten the pair into a single",
        "integer: `flat = code_a * n_b + code_b`. That is row-major indexing by",
        "hand. bincount over `flat`, then `.reshape(n_a, n_b)`.",
        "",
        "return: (keys_a, keys_b, table)",
        "Use `np.maximum(counts, 1)` in the denominator so an empty cell gives",
        "0 rather than a divide-by-zero warning.",
    ],
    "apply_log_transform": [
        "TODO (§8)   log1p the listed columns, leave the others alone",
        "",
        "Start from `X.copy()`. Without the copy you modify the CALLER's array",
        "in place -- the view/copy trap from §2, in its most expensive form.",
        "",
        "Why log1p and not log: `error_rate_pct` is exactly 0 for many rows,",
        "and log(0) = -inf.",
    ],
    "one_hot": [
        "TODO (§8)   categorical labels -> 0/1 matrix",
        "",
        "`np.unique(labels, return_inverse=True)` gives an integer code per row.",
        "Then the trick: row k of `np.eye(n)` IS the one-hot vector for",
        "category k, so `np.eye(n)[codes]` encodes everything at once.",
        "",
        "return: (matrix of shape (len(labels), n_categories), keys)",
    ],
    "add_cross_features": [
        "TODO (§8)   append every product x_i * x_j with i <= j",
        "",
        "  1. `X[:, :, None] * X[:, None, :]` broadcasts (n,d,1) against (n,1,d)",
        "     to give (n, d, d): every pairwise product for every row",
        "  2. `np.triu_indices(d)` gives the upper-triangle (i, j) pairs,",
        "     including the diagonal, so each product is kept once",
        "  3. index with `outer[:, i, j]` and hstack onto the original",
        "",
        "return: (expanded, pairs) where pairs is list(zip(i, j))",
    ],
    "stratified_split": [
        "TODO (§9)   assign rows to train/val/test, preserving class balance",
        "",
        "For each class separately: take its row positions, shuffle them with a",
        "seeded generator, and cut at the cumulative fractions. Doing it per",
        "class is what makes it stratified.",
        "",
        "return: int array, 0 = train, 1 = val, 2 = test",
        "hint: `np.split(idx, cuts)` splits at a list of cut positions;",
        "      `(np.cumsum(fractions)[:-1] * idx.size).astype(int)` builds them.",
    ],
    "fit_normal_equation": [
        "TODO (§10)   w = (X^T X)^-1 X^T y",
        "",
        "Prepend a column of ones for the intercept (`np.hstack` with",
        "`np.ones((n, 1))`), solve, then split the intercept back off the front.",
        "",
        "return: (weights, intercept)",
        "hint: `np.linalg.solve(A, b)`, never `np.linalg.inv(A) @ b`.",
    ],
    "fit_gradient_descent": [
        "TODO (§10)   full-batch gradient descent on the MSE",
        "",
        "Start at w = 0, b = 0, then repeat n_iters times:",
        "  resid = X @ w + b - y            (n,)",
        "  record mean(resid**2)",
        "  w -= lr * (2/n) * (X.T @ resid)  (d,)",
        "  b -= lr * (2/n) * resid.sum()",
        "",
        "Watch the shapes: `X.T @ resid` is (d,n)@(n,) -> (d,), matching w.",
        "",
        "return: (w, b, history)",
    ],
    "confusion_matrix_numpy": [
        "TODO (§11)   the 2x2 matrix [[TN, FP], [FN, TP]] in two lines",
        "",
        "Encode each row as a 2-bit number: `2 * y_true + y_pred`. That sends",
        "  (0,0)->0   (0,1)->1   (1,0)->2   (1,1)->3",
        "so `np.bincount(codes, minlength=4).reshape(2, 2)` IS the confusion",
        "matrix, in exactly sklearn's layout.",
    ],
    "roc_curve_numpy": [
        "TODO (§11)   the ROC curve from argsort and cumsum",
        "",
        "  1. order by DESCENDING score (`np.argsort(-score, kind='mergesort')`)",
        "  2. walking down that list adds one object at a time to 'predicted",
        "     positive', so running TP/FP counts are `np.cumsum` of the sorted",
        "     labels and of (1 - labels)",
        "  3. tied scores must be committed to as a block: keep only the last",
        "     index of each run of equal scores",
        "     (`np.flatnonzero(np.diff(sorted_scores))`, plus the final index)",
        "  4. divide by the totals, and prepend the (0, 0) origin",
        "",
        "return: (fpr, tpr)",
    ],
    "auc_numpy": [
        "TODO (§11)   AUC as a rank statistic",
        "",
        "        U = R+ - n+ * (n+ + 1) / 2,      AUC = U / (n+ * n-)",
        "",
        "where R+ is the sum of the ranks of the positives. Reuse the",
        "`rank_average` you wrote in §6 -- the averaged ties are exactly the",
        "convention this formula needs.",
    ],
}


def build_stub(indent: str, name: str, lines: list[str]) -> list[str]:
    out = [f"{indent}# {line}".rstrip() for line in lines]
    out.append(f"{indent}raise NotImplementedError({name!r})")
    return out


def transform(source: str) -> str:
    tree = ast.parse(source)
    targets = {n.name: n for n in tree.body
               if isinstance(n, ast.FunctionDef) and n.name in TODOS}

    missing = set(TODOS) - set(targets)
    if missing:
        raise SystemExit(f"functions not found at module level: {sorted(missing)}")

    out = source.splitlines()
    for node in sorted(targets.values(), key=lambda n: n.lineno, reverse=True):
        first = node.body[0]
        keep_through = (first.end_lineno
                        if (isinstance(first, ast.Expr)
                            and isinstance(first.value, ast.Constant)
                            and isinstance(first.value.value, str))
                        else first.lineno - 1)
        out[keep_through:node.end_lineno] = build_stub(
            " " * first.col_offset, node.name, TODOS[node.name])

    return "\n".join(out) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()

    generated = transform(SOURCE.read_text())
    ast.parse(generated)

    if args.check:
        current = TARGET.read_text() if TARGET.exists() else ""
        if current != generated:
            sys.exit(f"{TARGET.name} is out of date -- rerun {Path(__file__).name}")
        print(f"{TARGET.name} is up to date")
        return

    TARGET.write_text(generated)
    print(f"wrote {TARGET.name}  ({len(TODOS)} implementations replaced)")
    for name in TODOS:
        print(f"  - {name}")


if __name__ == "__main__":
    main()
