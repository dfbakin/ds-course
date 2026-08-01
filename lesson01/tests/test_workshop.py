"""Tests for the fill-in-the-blanks variant.

The contract of `lesson01_workshop.py` is narrow and worth pinning down: it is
`lesson01.py` with 14 function *bodies* blanked, and **nothing else touched**.
The test that matters is the last one, which proves that claim by comparing
every other byte of the two files.
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import make_workshop as W  # noqa: E402

SOURCE = ROOT / "lesson01.py"
WORKSHOP = ROOT / "lesson01_workshop.py"


@pytest.fixture(scope="module")
def src_text():
    return SOURCE.read_text()


@pytest.fixture(scope="module")
def wk_text():
    if not WORKSHOP.exists():
        pytest.skip("workshop file not generated; run src/make_workshop.py")
    return WORKSHOP.read_text()


def module_functions(text):
    return {n.name: n for n in ast.parse(text).body if isinstance(n, ast.FunctionDef)}


def test_workshop_exists_and_parses(wk_text):
    ast.parse(wk_text)


def test_workshop_is_up_to_date(src_text, wk_text):
    """Regenerating from the current lesson must reproduce the shipped file."""
    assert W.transform(src_text) == wk_text, (
        "lesson01_workshop.py is stale -- rerun src/make_workshop.py")


def test_every_target_is_blanked(wk_text):
    funcs = module_functions(wk_text)
    for name in W.TODOS:
        assert name in funcs, f"{name} vanished from the workshop file"
        body_src = ast.get_source_segment(wk_text, funcs[name])
        assert "NotImplementedError" in body_src, f"{name} was not blanked"


def test_blanked_functions_raise_when_called(wk_text):
    """A stub must fail loudly, not return None and confuse everyone later."""
    ns: dict = {}
    exec("import itertools\nimport numpy as np\nimport pandas as pd\n"
         "from scipy import stats\n", ns)
    funcs = module_functions(wk_text)
    for name in W.TODOS:
        node = funcs[name]
        mod = ast.Module(body=[node], type_ignores=[])
        exec(compile(ast.fix_missing_locations(mod), "<wk>", "exec"), ns)
        n_required = len(node.args.args) - len(node.args.defaults)
        with pytest.raises(NotImplementedError):
            ns[name](*([None] * n_required))


def test_signatures_and_docstrings_are_preserved(src_text, wk_text):
    """The student must still see the exact contract they are implementing."""
    src_funcs, wk_funcs = module_functions(src_text), module_functions(wk_text)
    for name in W.TODOS:
        a, b = src_funcs[name], wk_funcs[name]
        assert ast.dump(a.args) == ast.dump(b.args), f"{name} signature changed"
        assert ast.get_docstring(a) == ast.get_docstring(b), \
            f"{name} docstring changed"


def test_complicated_implementations_are_left_intact(wk_text):
    """Only basic-syntax code is blanked; library-heavy code stays."""
    keep = ["midrank", "delong_auc_cov", "delong_test", "build_model_features",
            "train_model", "critical_learning_rate", "wilson_interval",
            "auc_trapezoid", "predict_from_bundle", "paired_bootstrap_auc_diff",
            "time_normal_equation", "sweep_thresholds", "finish", "save_fig"]
    funcs = module_functions(wk_text)
    for name in keep:
        assert name in funcs, f"{name} disappeared"
        assert "NotImplementedError" not in ast.get_source_segment(wk_text, funcs[name]), \
            f"{name} should NOT have been blanked -- it is not a basic-syntax exercise"


def test_nothing_outside_the_blanked_bodies_changed(src_text, wk_text):
    """The strict claim: byte-identical everywhere except the 14 bodies.

    Both files are cut into the same top-level statements. Every statement that
    is not a blanked target must match byte for byte -- markdown cells, plots,
    imports, asserts, the lot.
    """
    src_tree, wk_tree = ast.parse(src_text), ast.parse(wk_text)
    assert len(src_tree.body) == len(wk_tree.body), "statement count changed"

    src_lines, wk_lines = src_text.splitlines(), wk_text.splitlines()
    checked = 0
    for a, b in zip(src_tree.body, wk_tree.body):
        is_target = isinstance(a, ast.FunctionDef) and a.name in W.TODOS
        if is_target:
            assert isinstance(b, ast.FunctionDef) and b.name == a.name
            continue
        assert ast.dump(a) == ast.dump(b), f"statement at line {a.lineno} changed"
        assert (src_lines[a.lineno - 1:a.end_lineno]
                == wk_lines[b.lineno - 1:b.end_lineno]), \
            f"source text at line {a.lineno} changed"
        checked += 1
    assert checked > 50, "sanity: expected many untouched top-level statements"


def test_all_comment_and_markdown_lines_are_preserved(src_text, wk_text):
    """Markdown cells are comments in py:percent, so diff them directly.

    Guards the instruction that only code was substituted: every `# %%` cell
    marker and every prose line in the original must survive verbatim.
    """
    def prose(text):
        return [ln for ln in text.splitlines()
                if ln.startswith("#") or ln.strip().startswith("# %%")]

    src_prose, wk_prose = prose(src_text), prose(wk_text)
    missing = [ln for ln in src_prose if ln not in wk_prose]
    assert not missing, f"prose lines lost: {missing[:5]}"
    assert src_prose == wk_prose, "top-level prose/markdown changed"
