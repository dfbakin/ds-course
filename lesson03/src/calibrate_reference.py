"""
Fit the lesson's five-model ladder on data/fleet_incidents.csv and write
data/reference_results.json.

The ladder (all models: fit on train, tune/early-stop on val, report test
ROC-AUC):

1. LogisticRegression, 8 numerics only, standardized.
2. LogisticRegression, numerics + OneHotEncoder(handle_unknown="ignore") on
   all 5 categoricals (~220 columns after encoding).
3. DecisionTreeClassifier(max_depth=6, min_samples_leaf=50) on the same
   numeric+OHE matrix as model 2.
4. CatBoostClassifier defaults (iterations=500, early stopping on val),
   native cat_features.
5. CatBoost tuned with Optuna: TPESampler(seed=SEED), 30 trials,
   depth in [4, 8], learning_rate loguniform [0.02, 0.3], l2_leaf_reg
   loguniform [1, 30], iterations <= 800 with early stopping on val;
   objective = val AUC; then refit best params and report test AUC.

Also verifies the regression story on power_draw_watts: lasso over an alpha
grid must recover the TRUE sparse support (cpu_util, request_rate_rps,
queue_depth -- everything else has true coefficient exactly 0), and ridge
must shrink the collinear pair (cpu_util, mem_pressure) toward each other.

The resulting JSON is the single source of truth for the lesson's §9 ladder
table, the slides, and tests/test_dataset.py.

Usage
-----
    python src/calibrate_reference.py [--skip-optuna] [--out-dir DATA_DIR]

--skip-optuna fits models 1-4 only and does NOT write the JSON; it exists
for fast iterations of the generator-calibration loop.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.linear_model import Lasso, LassoCV, LinearRegression, LogisticRegression, Ridge
from sklearn.metrics import r2_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

import generate_dataset as G

SEED = G.SEED
NUMERIC = G.RAW_FEATURES
CATEGORICAL = G.CATEGORICAL_FEATURES


def load_splits(data_dir: Path):
    df = pd.read_csv(data_dir / "fleet_incidents.csv",
                     dtype={c: str for c in CATEGORICAL})
    parts = {}
    for s in ("train", "val", "test"):
        d = df[df["split"] == s]
        parts[s] = (d[NUMERIC + CATEGORICAL], d["incident"].to_numpy(),
                    d["power_draw_watts"].to_numpy())
    return df, parts


def make_preprocessor() -> ColumnTransformer:
    """Numerics standardized, categoricals one-hot -- the model-2/3 matrix."""
    return ColumnTransformer([
        ("num", StandardScaler(), NUMERIC),
        ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),
    ])


def fit_ladder_linear_and_tree(parts) -> dict:
    """Models 1-3 (seconds each)."""
    (Xtr, ytr, _), (Xv, yv, _), (Xte, yte, _) = (
        parts["train"], parts["val"], parts["test"])
    out = {}

    m1 = Pipeline([("scale", StandardScaler()),
                   ("clf", LogisticRegression(max_iter=1000))])
    m1.fit(Xtr[NUMERIC], ytr)
    out["logreg_numeric"] = {
        "val_auc": float(roc_auc_score(yv, m1.predict_proba(Xv[NUMERIC])[:, 1])),
        "test_auc": float(roc_auc_score(yte, m1.predict_proba(Xte[NUMERIC])[:, 1])),
        "n_features": len(NUMERIC),
    }

    pre = make_preprocessor()
    m2 = Pipeline([("prep", pre), ("clf", LogisticRegression(max_iter=1000))])
    m2.fit(Xtr, ytr)
    n_ohe = m2["prep"].transform(Xtr[:1]).shape[1]
    out["logreg_ohe"] = {
        "val_auc": float(roc_auc_score(yv, m2.predict_proba(Xv)[:, 1])),
        "test_auc": float(roc_auc_score(yte, m2.predict_proba(Xte)[:, 1])),
        "n_features": int(n_ohe),
    }

    m3 = Pipeline([("prep", make_preprocessor()),
                   ("clf", DecisionTreeClassifier(max_depth=6, min_samples_leaf=50,
                                                  random_state=SEED))])
    m3.fit(Xtr, ytr)
    out["tree"] = {
        "val_auc": float(roc_auc_score(yv, m3.predict_proba(Xv)[:, 1])),
        "test_auc": float(roc_auc_score(yte, m3.predict_proba(Xte)[:, 1])),
        "n_features": int(n_ohe),
    }
    return out


def fit_catboost_default(parts) -> dict:
    """Model 4: CatBoost with native categoricals, otherwise defaults."""
    from catboost import CatBoostClassifier

    (Xtr, ytr, _), (Xv, yv, _), (Xte, yte, _) = (
        parts["train"], parts["val"], parts["test"])
    model = CatBoostClassifier(iterations=500, random_seed=SEED, verbose=0,
                               cat_features=CATEGORICAL,
                               allow_writing_files=False)
    model.fit(Xtr, ytr, eval_set=(Xv, yv), early_stopping_rounds=50,
              use_best_model=True)
    return {
        "val_auc": float(roc_auc_score(yv, model.predict_proba(Xv)[:, 1])),
        "test_auc": float(roc_auc_score(yte, model.predict_proba(Xte)[:, 1])),
        "best_iteration": int(model.get_best_iteration()),
        "n_features": len(NUMERIC) + len(CATEGORICAL),
    }


def fit_catboost_tuned(parts, n_trials: int = 30) -> dict:
    """Model 5: the Optuna study (TPESampler(seed=SEED), 30 trials)."""
    import optuna
    from catboost import CatBoostClassifier

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    (Xtr, ytr, _), (Xv, yv, _), (Xte, yte, _) = (
        parts["train"], parts["val"], parts["test"])

    def fit_with(params: dict) -> "CatBoostClassifier":
        model = CatBoostClassifier(
            iterations=800, random_seed=SEED, verbose=0,
            cat_features=CATEGORICAL, allow_writing_files=False,
            eval_metric="AUC", **params)
        model.fit(Xtr, ytr, eval_set=(Xv, yv), early_stopping_rounds=50,
                  use_best_model=True)
        return model

    def objective(trial: "optuna.Trial") -> float:
        params = {
            "depth": trial.suggest_int("depth", 4, 8),
            "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.3,
                                                 log=True),
            "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1.0, 30.0,
                                               log=True),
        }
        model = fit_with(params)
        return float(roc_auc_score(yv, model.predict_proba(Xv)[:, 1]))

    sampler = optuna.samplers.TPESampler(seed=SEED)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    study.optimize(objective, n_trials=n_trials)

    best = fit_with(study.best_params)
    return {
        "val_auc": float(study.best_value),
        "test_auc": float(roc_auc_score(yte, best.predict_proba(Xte)[:, 1])),
        "best_iteration": int(best.get_best_iteration()),
        "best_params": study.best_params,
        "n_trials": n_trials,
        "sampler": f"TPESampler(seed={SEED})",
    }


def check_lasso_support(parts, meta: dict) -> dict:
    """Does lasso recover the true sparse support of power_draw_watts?

    Fits the lasso path on standardized train numerics over a log-spaced
    alpha grid and finds the alphas whose active set is EXACTLY the true
    support.  A wide recovering interval is the honest version of the claim
    "lasso at reasonable alpha recovers the truth".
    """
    (Xtr, _, ptr), _, (Xte, _, pte) = (
        parts["train"], parts["val"], parts["test"])
    true_support = meta["power_mechanism"]["true_support"]

    scaler = StandardScaler().fit(Xtr[NUMERIC])
    Ztr = scaler.transform(Xtr[NUMERIC])
    Zte = scaler.transform(Xte[NUMERIC])

    alphas = np.logspace(-2, 2, 81)
    recovering = []
    for a in alphas:
        lasso = Lasso(alpha=a).fit(Ztr, ptr)
        support = [NUMERIC[i] for i in np.flatnonzero(np.abs(lasso.coef_) > 1e-6)]
        if sorted(support) == sorted(true_support):
            recovering.append(a)

    cv = LassoCV(alphas=alphas, cv=5).fit(Ztr, ptr)
    cv_support = [NUMERIC[i] for i in np.flatnonzero(np.abs(cv.coef_) > 1e-6)]

    result = {
        "true_support": true_support,
        "alpha_grid": "np.logspace(-2, 2, 81)",
        "n_alphas_recovering": len(recovering),
        "recovering_alpha_min": float(min(recovering)) if recovering else None,
        "recovering_alpha_max": float(max(recovering)) if recovering else None,
        "recovered": bool(recovering),
        "lasso_cv_alpha": float(cv.alpha_),
        "lasso_cv_support": cv_support,
        "lasso_cv_recovers": sorted(cv_support) == sorted(true_support),
    }
    if recovering:
        mid = recovering[len(recovering) // 2]
        lasso = Lasso(alpha=mid).fit(Ztr, ptr)
        result["example_alpha"] = float(mid)
        result["example_coefficients"] = {
            n: float(c) for n, c in zip(NUMERIC, lasso.coef_)}
        result["test_r2_at_example_alpha"] = float(
            r2_score(pte, lasso.predict(Zte)))
    return result


def check_ridge_collinearity(parts) -> dict:
    """Ridge shrinks the collinear pair (cpu_util, mem_pressure) together.

    mem_pressure has TRUE coefficient 0 but is correlated with cpu_util
    (rho ~ 0.6).  OLS keeps them apart; as the ridge penalty grows, weight
    is spread across the correlated pair, so the gap |cpu - mem| shrinks.
    """
    (Xtr, _, ptr), _, _ = parts["train"], parts["val"], parts["test"]
    scaler = StandardScaler().fit(Xtr[NUMERIC])
    Ztr = scaler.transform(Xtr[NUMERIC])
    i_cpu, i_mem = NUMERIC.index("cpu_util"), NUMERIC.index("mem_pressure")

    path = {}
    ols = LinearRegression().fit(Ztr, ptr)
    path["ols"] = {"cpu_util": float(ols.coef_[i_cpu]),
                   "mem_pressure": float(ols.coef_[i_mem])}
    for a in (1e3, 1e4, 1e5):
        ridge = Ridge(alpha=a).fit(Ztr, ptr)
        path[f"ridge_alpha_{a:.0e}"] = {
            "cpu_util": float(ridge.coef_[i_cpu]),
            "mem_pressure": float(ridge.coef_[i_mem])}
    gaps = [abs(v["cpu_util"] - v["mem_pressure"]) for v in path.values()]
    return {
        "coefficient_path": path,
        "abs_gap_path": [float(g) for g in gaps],
        "gap_shrinks_monotonically": bool(
            all(a > b for a, b in zip(gaps, gaps[1:]))),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path,
                        default=Path(__file__).resolve().parents[1] / "data")
    parser.add_argument("--skip-optuna", action="store_true",
                        help="models 1-4 only; do not write the JSON")
    args = parser.parse_args()

    with open(args.out_dir / "dgp_metadata.json") as fh:
        meta = json.load(fh)

    df, parts = load_splits(args.out_dir)
    t0 = time.time()

    ladder = fit_ladder_linear_and_tree(parts)
    print(f"[{time.time() - t0:6.1f}s] 1 logreg numeric  test AUC "
          f"{ladder['logreg_numeric']['test_auc']:.4f}")
    print(f"[{time.time() - t0:6.1f}s] 2 logreg + OHE    test AUC "
          f"{ladder['logreg_ohe']['test_auc']:.4f} "
          f"({ladder['logreg_ohe']['n_features']} features)")
    print(f"[{time.time() - t0:6.1f}s] 3 decision tree   test AUC "
          f"{ladder['tree']['test_auc']:.4f}")

    ladder["catboost_default"] = fit_catboost_default(parts)
    print(f"[{time.time() - t0:6.1f}s] 4 catboost        test AUC "
          f"{ladder['catboost_default']['test_auc']:.4f}")

    lasso = check_lasso_support(parts, meta)
    ridge = check_ridge_collinearity(parts)
    print(f"[{time.time() - t0:6.1f}s] lasso support recovered: "
          f"{lasso['recovered']} "
          f"(alphas {lasso['recovering_alpha_min']} .. "
          f"{lasso['recovering_alpha_max']})")
    print(f"[{time.time() - t0:6.1f}s] ridge gap path: "
          f"{[round(g, 2) for g in ridge['abs_gap_path']]}")

    if args.skip_optuna:
        a1 = ladder["logreg_numeric"]["test_auc"]
        a2 = ladder["logreg_ohe"]["test_auc"]
        a3 = ladder["tree"]["test_auc"]
        a4 = ladder["catboost_default"]["test_auc"]
        print("\ncalibration snapshot (no optuna, nothing written):")
        print(f"  gap 1->2: {a2 - a1:+.4f}   (need >= +0.02)")
        print(f"  gap 2->4: {a4 - a2:+.4f}   (need >= +0.02)")
        print(f"  tree between 1 and 4: {a1 < a3 < a4}")
        print(f"  incident rate: {df['incident'].mean():.4f}")
        return

    ladder["catboost_tuned"] = fit_catboost_tuned(parts)
    print(f"[{time.time() - t0:6.1f}s] 5 catboost tuned  test AUC "
          f"{ladder['catboost_tuned']['test_auc']:.4f} "
          f"params {ladder['catboost_tuned']['best_params']}")

    gaps = {
        "numeric_to_ohe": ladder["logreg_ohe"]["test_auc"]
        - ladder["logreg_numeric"]["test_auc"],
        "ohe_to_catboost": ladder["catboost_default"]["test_auc"]
        - ladder["logreg_ohe"]["test_auc"],
        "default_to_tuned": ladder["catboost_tuned"]["test_auc"]
        - ladder["catboost_default"]["test_auc"],
    }

    results = {
        "seed": SEED,
        "dataset": "fleet_incidents.csv",
        "incident_rate": float(df["incident"].mean()),
        "n_features_numeric": len(NUMERIC),
        "n_features_after_ohe": ladder["logreg_ohe"]["n_features"],
        "ladder": ladder,
        "gaps": {k: float(v) for k, v in gaps.items()},
        "lasso_support_recovery": lasso,
        "ridge_collinearity": ridge,
    }
    out_path = args.out_dir / "reference_results.json"
    with open(out_path, "w") as fh:
        json.dump(results, fh, indent=2)
    print(f"\nwrote {out_path}")
    for name in ("logreg_numeric", "logreg_ohe", "tree",
                 "catboost_default", "catboost_tuned"):
        print(f"  {name:18s} test AUC {ladder[name]['test_auc']:.4f}")
    for k, v in gaps.items():
        print(f"  gap {k:18s} {v:+.4f}")


if __name__ == "__main__":
    main()
