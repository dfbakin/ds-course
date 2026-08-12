"""Tests for the fill-in-the-blanks variant.

The contract of `lesson03_workshop.py` is narrow and worth pinning down: it is
`lesson03.py` with 12 function *bodies* blanked, and **nothing else touched**.
Signatures and docstrings survive byte for byte -- the student must still see
the exact contract they are implementing -- and the test that matters most is
the one that proves the "nothing else" claim by comparing every other byte of
the two files.
"""

from __future__ import annotations

import ast
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import make_workshop as W  # noqa: E402

SOURCE = ROOT / "lesson03.py"
WORKSHOP = ROOT / "lesson03_workshop.py"

# Scaffolding the generator must leave alone: students read these, not write.
KEEP_INTACT = ["load_splits", "rate_per_level", "support_of", "finish"]


@pytest.fixture(scope="module")
def src_text():
    return SOURCE.read_text()


@pytest.fixture(scope="module")
def wk_text():
    if not WORKSHOP.exists():
        pytest.skip("workshop file not generated; run src/make_workshop.py")
    return WORKSHOP.read_text()


def module_functions(text):
    return {n.name: n for n in ast.parse(text).body
            if isinstance(n, ast.FunctionDef)}


def test_make_workshop_check_passes():
    """The shipped CLI contract: `make_workshop.py --check` must exit 0."""
    result = subprocess.run(
        [sys.executable, str(ROOT / "src" / "make_workshop.py"), "--check"],
        capture_output=True, text=True)
    assert result.returncode == 0, (
        "make_workshop.py --check failed -- rerun src/make_workshop.py:\n"
        + result.stdout + result.stderr)


def test_workshop_is_up_to_date(src_text, wk_text):
    """Regenerating from the current lesson must reproduce the shipped file."""
    assert W.transform(src_text) == wk_text, (
        "lesson03_workshop.py is stale -- rerun src/make_workshop.py")


def test_workshop_parses_and_compiles(wk_text):
    """Blanked bodies must still be valid Python, down to bytecode."""
    ast.parse(wk_text)
    compile(wk_text, str(WORKSHOP), "exec")


def test_every_target_is_blanked_and_raises(wk_text):
    """A stub must fail loudly, not return None and confuse everyone later."""
    funcs = module_functions(wk_text)
    ns: dict = {}
    exec("import numpy as np\n", ns)
    for name in W.TODOS:
        assert name in funcs, f"{name} vanished from the workshop file"
        node = funcs[name]
        assert "NotImplementedError" in ast.get_source_segment(wk_text, node), \
            f"{name} was not blanked"
        mod = ast.Module(body=[node], type_ignores=[])
        exec(compile(ast.fix_missing_locations(mod), "<wk>", "exec"), ns)
        n_required = len(node.args.args) - len(node.args.defaults)
        with pytest.raises(NotImplementedError):
            ns[name](*([None] * n_required))


def test_signatures_and_docstrings_are_byte_identical(src_text, wk_text):
    """From `def` through the docstring, the two files must agree exactly."""
    src_funcs, wk_funcs = module_functions(src_text), module_functions(wk_text)
    src_lines, wk_lines = src_text.splitlines(), wk_text.splitlines()
    for name in W.TODOS:
        a, b = src_funcs[name], wk_funcs[name]
        assert ast.dump(a.args) == ast.dump(b.args), f"{name} signature changed"
        assert ast.get_docstring(a) == ast.get_docstring(b), \
            f"{name} docstring changed"
        # the strict version: header + docstring lines, byte for byte
        doc_a, doc_b = a.body[0], b.body[0]
        assert (src_lines[a.lineno - 1:doc_a.end_lineno]
                == wk_lines[b.lineno - 1:doc_b.end_lineno]), \
            f"{name}: def line or docstring text drifted"


def test_scaffolding_is_left_intact(wk_text):
    """Helpers the student reads, not writes, must keep their bodies."""
    funcs = module_functions(wk_text)
    for name in KEEP_INTACT:
        assert name in funcs, f"{name} disappeared"
        assert name not in W.TODOS, f"{name} must not be a blank target"
        assert "NotImplementedError" not in \
            ast.get_source_segment(wk_text, funcs[name]), \
            f"{name} should NOT have been blanked -- it is scaffolding"


def test_nothing_outside_the_blanked_bodies_changed(src_text, wk_text):
    """Byte-identical everywhere except the 12 replaced function bodies."""
    src_tree, wk_tree = ast.parse(src_text), ast.parse(wk_text)
    assert len(src_tree.body) == len(wk_tree.body), "statement count changed"

    src_lines, wk_lines = src_text.splitlines(), wk_text.splitlines()
    checked = 0
    for a, b in zip(src_tree.body, wk_tree.body):
        if isinstance(a, ast.FunctionDef) and a.name in W.TODOS:
            assert isinstance(b, ast.FunctionDef) and b.name == a.name
            continue
        assert ast.dump(a) == ast.dump(b), f"statement at line {a.lineno} changed"
        assert (src_lines[a.lineno - 1:a.end_lineno]
                == wk_lines[b.lineno - 1:b.end_lineno]), \
            f"source text at line {a.lineno} changed"
        checked += 1
    assert checked > 50, "sanity: expected many untouched top-level statements"


def test_all_comment_and_markdown_lines_are_preserved(src_text, wk_text):
    """Markdown cells are comments in py:percent -- every prose line survives.

    TODO comments live *inside* function bodies and are indented, so the
    column-0 comment lines of the two files must match exactly.
    """
    src_prose = [ln for ln in src_text.splitlines() if ln.startswith("#")]
    wk_prose = [ln for ln in wk_text.splitlines() if ln.startswith("#")]
    assert src_prose == wk_prose, "top-level prose/markdown changed"
