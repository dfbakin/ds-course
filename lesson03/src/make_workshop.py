"""Generate the fill-in-the-blanks variant of lesson 3.

Same machinery as `lesson02/src/make_workshop.py`: replace a chosen set of
function *bodies* with TODO comments using `ast`, so signatures and docstrings
survive byte for byte and nothing outside the replaced bodies is touched.

Selection rule for this lesson: blank what the ladder is *about* -- the Wilson
interval, the coefficient paths, the five ladder-fitting functions, the
boosting-by-hand loop and the Optuna objective. What stays intact is the
scaffolding a student should read but not rewrite: `load_splits`, the plotting
helper `finish`, `rate_per_level` (a groupby around the interval you wrote)
and `support_of` (a one-line filter half the asserts depend on).

Usage
-----
    python src/make_workshop.py            # writes lesson03_workshop.py
    python src/make_workshop.py --check    # verify it is up to date
"""

from __future__ import annotations

import argparse
import ast
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "lesson03.py"
TARGET = ROOT / "lesson03_workshop.py"

TODOS: dict[str, list[str]] = {
    "wilson_ci": [
        "TODO (§1)   the Wilson score interval, vectorised",
        "",
        "With p = k/n and denom = 1 + z^2/n:",
        "  centre = (p + z^2 / (2n)) / denom",
        "  half   = (z / denom) * sqrt( p(1-p)/n + z^2 / (4 n^2) )",
        "",
        "Cast k and n to float arrays first (`np.asarray(..., dtype=float)`) --",
        "rate_per_level below calls this with whole columns, and integer n",
        "would make z**2/n integer-divide-adjacent nonsense on old habits.",
        "",
        "return: (centre - half, centre + half)",
    ],
    "lasso_path_coefs": [
        "TODO (§3)   refit Lasso once per alpha, stack the coefficient rows",
        "",
        "One list comprehension: `Lasso(alpha=a).fit(Z, y).coef_` for each a,",
        "wrapped in np.array. Deliberately transparent and slow -- sklearn's",
        "lasso_path does it faster, but you would not see the refits.",
        "",
        "return: array of shape (len(alphas), n_features)",
    ],
    "ridge_path_coefs": [
        "TODO (§3)   the same recipe with Ridge",
        "",
        "Identical loop, `Ridge(alpha=a)` instead of `Lasso(alpha=a)`. Note",
        "the RIDGE_ALPHAS grid below runs to 1e6 where lasso's stops at 100:",
        "the L2 penalty has no corners, so it needs far bigger alphas to move.",
        "",
        "return: array of shape (len(alphas), n_features)",
    ],
    "fit_logreg_numeric": [
        "TODO (§4)   ladder model 1 -- the baseline you always build first",
        "",
        "Pipeline: StandardScaler -> LogisticRegression(max_iter=1000), fitted",
        "on Xtr[numeric] ONLY -- this model must not see a single categorical.",
        "",
        "Score with `predict_proba(...)[:, 1]`: AUC ranks probabilities, and",
        ".predict() would throw the ranking away.",
        "",
        "return: dict with keys model, val_auc, test_auc, n_features",
    ],
    "build_ohe_pipeline": [
        "TODO (§4)   ColumnTransformer: scale the numerics, one-hot the cats",
        "",
        "Two named transformers: (\"num\", StandardScaler(), numeric) and",
        "(\"cat\", OneHotEncoder(...), categorical).",
        "",
        "handle_unknown=\"ignore\" is the load-bearing choice -- six clusters",
        "exist only in val/test, and a level the encoder never saw must",
        "become an all-zero block, not a crash. No drop=\"first\": with L2 the",
        "redundancy is harmless and coefficients stay readable per level.",
        "",
        "return: the (unfitted) ColumnTransformer",
    ],
    "fit_logreg_ohe": [
        "TODO (§4)   ladder model 2 -- numerics + all 5 categoricals",
        "",
        "Pipeline: build_ohe_pipeline(numeric, categorical) ->",
        "LogisticRegression(max_iter=1000), fitted on the full Xtr.",
        "",
        "n_features is the POST-encoding column count: transform a single row",
        "(`model[\"prep\"].transform(Xtr[:1]).shape[1]`). Expect ~213 -- the",
        "explosion is the point of the section, so report it honestly.",
        "",
        "return: dict with keys model, val_auc, test_auc, n_features",
    ],
    "tree_depth_sweep": [
        "TODO (§5)   train-vs-val AUC of a tree at every depth in `depths`",
        "",
        "For each d: fit DecisionTreeClassifier(max_depth=d,",
        "min_samples_leaf=min_samples_leaf, random_state=RANDOM_SEED) on the",
        "already-preprocessed (Ztr, ytr), then score BOTH matrices with",
        "predict_proba(...)[:, 1]. The train/val gap is the overfitting",
        "figure; forget the train score and the plot has nothing to say.",
        "",
        "return: (train_aucs, val_aucs), two lists aligned with depths",
    ],
    "fit_tree": [
        "TODO (§5)   ladder model 3 -- one tree, same 213 columns as model 2",
        "",
        "Pipeline: build_ohe_pipeline(numeric, categorical) ->",
        "DecisionTreeClassifier(max_depth=max_depth,",
        "min_samples_leaf=min_samples_leaf, random_state=RANDOM_SEED).",
        "Same protocol as models 1-2: fit on train, AUC on val and test,",
        "n_features from the fitted preprocessor.",
        "",
        "return: dict with keys model, val_auc, test_auc, n_features",
    ],
    "boost_by_hand": [
        "TODO (§6)   gradient boosting for squared loss, smallest honest form",
        "",
        "Stage 0: pred = the constant y.mean(). Then, n_rounds times:",
        "  residual = y - pred          <- recomputed EVERY round, not once",
        "  stump = DecisionTreeRegressor(max_depth=1,",
        "          random_state=RANDOM_SEED).fit(X, residual)",
        "  pred = pred + stump.predict(X)    (full weight -- no shrinkage yet)",
        "",
        "X is x.reshape(-1, 1): sklearn wants 2-D. Record pred and the MSE",
        "after stage 0 AND after every round, and append pred.copy() -- ",
        "without the copy every history entry aliases the same array.",
        "",
        "return: dict with keys f0, stumps, train_pred, train_mse",
        "        (train_pred and train_mse hold n_rounds + 1 entries)",
    ],
    "fit_catboost_default": [
        "TODO (§7)   ladder model 4 -- CatBoost defaults, native categoricals",
        "",
        "No encoding pipeline: pass the raw DataFrame and",
        "cat_features=categorical -- COLUMN NAMES, because Xtr is a DataFrame.",
        "CatBoostClassifier(iterations=500, random_seed=RANDOM_SEED,",
        "verbose=0, allow_writing_files=False); fit with eval_set=(Xva, yva),",
        "early_stopping_rounds=50, use_best_model=True. Touch nothing else --",
        "eval_metric stays Logloss; \"defaults\" is the whole rung.",
        "",
        "return: dict with keys model, val_auc, test_auc, best_iteration",
        "        (int(model.get_best_iteration())), n_features",
    ],
    "fit_catboost_eval_auc": [
        "TODO (§8)   the tuning protocol's fit: 800 iterations, AUC-watched",
        "",
        "Like model 4 but iterations=800, eval_metric=\"AUC\" (early stopping",
        "now monitors the thing we optimise, not Logloss), plus **params for",
        "depth / learning_rate / l2_leaf_reg. Unpack ONLY train and val from",
        "parts -- this function must never touch the test split.",
        "",
        "return: (model, val_auc)",
    ],
    "make_optuna_objective": [
        "TODO (§8)   build and return objective(trial) -> val AUC",
        "",
        "Inside the closure, sample the reference search space:",
        "  depth          trial.suggest_int(\"depth\", 4, 8)",
        "  learning_rate  trial.suggest_float(..., 0.02, 0.3, log=True)",
        "  l2_leaf_reg    trial.suggest_float(..., 1.0, 30.0, log=True)",
        "then fit via fit_catboost_eval_auc and return its val AUC.",
        "",
        "The objective returns VAL AUC -- never touch test inside it. Any",
        "split you optimise against stops measuring generalisation; test",
        "buys its meaning by being spent once, in the refit cell below.",
        "",
        "return: the objective function (a closure over parts / categorical)",
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
