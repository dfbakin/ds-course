"""End-to-end reproducibility test: the notebook must execute cleanly.

Marked `slow` (about a minute). Run with `pytest -m slow` or plain `pytest`.

This is the test that matters most for a teaching notebook: every `assert` the
notebook makes about its own results runs here, so a broken claim anywhere in
the lesson fails this test.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.slow
def test_notebook_executes_end_to_end(tmp_path):
    jupytext = Path(sys.executable).parent / "jupytext"
    jupyter = Path(sys.executable).parent / "jupyter"
    if not jupytext.exists() or not jupyter.exists():
        pytest.skip("jupytext/jupyter not installed in this interpreter")

    # Execute a copy, so a test run never rewrites the shipped notebook.
    work = tmp_path / "lesson01"
    work.mkdir()
    shutil.copytree(ROOT / "data", work / "data")
    (work / "models").mkdir()
    (work / "figures").mkdir()
    shutil.copy(ROOT / "lesson01.py", work / "lesson01.py")

    subprocess.run(
        [str(jupytext), "--to", "ipynb", "lesson01.py", "-o", "lesson01.ipynb"],
        cwd=work, check=True, capture_output=True,
    )
    result = subprocess.run(
        [str(jupyter), "nbconvert", "--to", "notebook", "--execute", "--inplace",
         "--ExecutePreprocessor.timeout=900", "lesson01.ipynb"],
        cwd=work, capture_output=True, text=True,
    )
    assert result.returncode == 0, (
        "notebook failed to execute:\n" + result.stderr[-4000:]
    )

    import nbformat
    nb = nbformat.read(work / "lesson01.ipynb", as_version=4)
    errors = [
        (i, o.ename, o.evalue)
        for i, cell in enumerate(nb.cells)
        for o in cell.get("outputs", [])
        if o.output_type == "error"
    ]
    assert not errors, f"cells raised: {errors}"

    # The notebook is also responsible for producing the shipped artifacts.
    for name in ["model_a_plain.joblib", "model_b_cross.joblib", "metadata.json"]:
        assert (work / "models" / name).exists(), f"notebook did not write {name}"
    assert len(list((work / "figures").glob("*.png"))) >= 15


@pytest.mark.slow
def test_saved_models_reproduce_their_predictions():
    """The shipped .joblib backups must load without any notebook classes."""
    import itertools

    import joblib
    import numpy as np
    import pandas as pd

    sys.path.insert(0, str(ROOT / "src"))
    import generate_dataset as G

    models_dir = ROOT / "models"
    if not (models_dir / "model_b_cross.joblib").exists():
        pytest.skip("model binaries not built yet; run the notebook first")

    df = pd.read_csv(ROOT / "data" / "service_telemetry.csv")
    frame = G.to_model_features(df)

    for name in ["model_a_plain.joblib", "model_b_cross.joblib"]:
        bundle = joblib.load(models_dir / name)
        assert isinstance(bundle, dict), "backups must be plain dicts, not pickled objects"
        assert set(bundle) >= {"w", "b", "feat_mean", "feat_std", "use_cross",
                               "feature_names"}

        X = (frame[bundle["feature_names"]].to_numpy(float)
             - bundle["feat_mean"]) / bundle["feat_std"]
        if bundle["use_cross"]:
            pairs = list(itertools.combinations_with_replacement(range(X.shape[1]), 2))
            X = np.hstack([X, np.column_stack([X[:, i] * X[:, j] for i, j in pairs])])
        if bundle["design_mean"] is not None:
            X = (X - bundle["design_mean"]) / bundle["design_std"]
        scores = X @ bundle["w"] + bundle["b"]

        assert scores.shape == (len(df),)
        assert np.all(np.isfinite(scores))

        from sklearn.metrics import roc_auc_score
        test = (df["split"] == "test").to_numpy()
        auc = roc_auc_score(df.loc[test, "incident"], scores[test])
        assert auc > 0.80, f"{name} reloaded but scores badly (AUC {auc:.3f})"
