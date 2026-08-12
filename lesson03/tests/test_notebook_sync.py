"""Guard against a .py / .ipynb pair drifting apart.

Both the worked lesson and the workshop variant are jupytext-paired
(`formats: py:percent,ipynb`). The `.py` is the reviewable source of truth;
the `.ipynb` is what students actually open. Two copies of anything diverge
unless something checks: this module reads each pair through jupytext and
asserts the shipped notebook carries exactly the cells the script does, so
neither file can be edited without regenerating its partner.
"""

from __future__ import annotations

from pathlib import Path

import pytest

jupytext = pytest.importorskip("jupytext")

ROOT = Path(__file__).resolve().parents[1]
PAIRS = ["lesson03", "lesson03_workshop"]


@pytest.fixture(scope="module", params=PAIRS)
def pair(request):
    py = ROOT / f"{request.param}.py"
    ipynb = ROOT / f"{request.param}.ipynb"
    assert py.exists(), f"{py.name} missing"
    assert ipynb.exists(), (
        f"{ipynb.name} missing -- run jupytext --to notebook {py.name}")
    return jupytext.read(py), jupytext.read(ipynb), request.param


def test_same_number_and_kind_of_cells(pair):
    from_py, from_ipynb, name = pair
    kinds_py = [c.cell_type for c in from_py.cells]
    kinds_nb = [c.cell_type for c in from_ipynb.cells]
    assert kinds_py == kinds_nb, f"{name}: cell structure diverged"


def test_cell_sources_are_identical(pair):
    """The check that matters: every cell's text, in order, byte for byte."""
    from_py, from_ipynb, name = pair
    for i, (a, b) in enumerate(zip(from_py.cells, from_ipynb.cells)):
        assert a.source == b.source, (
            f"{name}: cell {i} differs between .py and .ipynb -- "
            "edit one side and sync with jupytext")


def test_notebook_declares_the_course_kernel(pair):
    """Every notebook in the course runs on the `ds-course` kernel."""
    _, from_ipynb, name = pair
    kernel = from_ipynb.metadata.get("kernelspec", {})
    assert kernel.get("name") == "ds-course", f"{name}: wrong kernel"
    assert kernel.get("display_name") == "DS Course", f"{name}: wrong kernel"


def test_pairing_header_is_declared(pair):
    """Both representations must claim the py:percent,ipynb pairing."""
    from_py, from_ipynb, name = pair
    for nb, side in [(from_py, ".py"), (from_ipynb, ".ipynb")]:
        formats = nb.metadata.get("jupytext", {}).get("formats")
        assert formats == "py:percent,ipynb", (
            f"{name}{side}: jupytext pairing header missing or wrong")
