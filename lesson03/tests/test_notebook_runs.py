"""End-to-end reproducibility test: the notebook must execute cleanly.

Marked `slow` (about three minutes -- the Optuna study is most of it). Run
with `pytest -m slow` or plain `pytest`.

This is the test that matters most for a teaching notebook: every `assert`
the notebook makes about its own results runs here -- including the §9 block
that checks the whole ladder against data/reference_results.json -- so a
broken claim anywhere in the lesson fails this test.
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
    work = tmp_path / "lesson03"
    work.mkdir()
    shutil.copytree(ROOT / "data", work / "data")
    (work / "figures").mkdir()
    shutil.copy(ROOT / "lesson03.py", work / "lesson03.py")

    subprocess.run(
        [str(jupytext), "--to", "ipynb", "lesson03.py", "-o", "lesson03.ipynb"],
        cwd=work, check=True, capture_output=True,
    )
    result = subprocess.run(
        [str(jupyter), "nbconvert", "--to", "notebook", "--execute", "--inplace",
         "--ExecutePreprocessor.timeout=900", "lesson03.ipynb"],
        cwd=work, capture_output=True, text=True,
    )
    assert result.returncode == 0, (
        "notebook failed to execute:\n" + result.stderr[-4000:]
    )

    import nbformat
    nb = nbformat.read(work / "lesson03.ipynb", as_version=4)
    errors = [
        (i, o.ename, o.evalue)
        for i, cell in enumerate(nb.cells)
        for o in cell.get("outputs", [])
        if o.output_type == "error"
    ]
    assert not errors, f"cells raised: {errors}"

    # The notebook is also responsible for producing the shipped figures.
    assert len(list((work / "figures").glob("*.png"))) >= 8
