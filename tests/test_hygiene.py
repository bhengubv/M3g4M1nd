"""Rows D7, E2 and E4 — repository hygiene, enforced rather than documented.

These are the cheapest tests here and they guard three of MiniMind's defects.
All three were plainly visible to anyone who looked; nobody looked, because
nothing made them.
"""

from __future__ import annotations

import ast
import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
PACKAGE = ROOT / "m3g4m1nd"

# Our own source only. docs/minimind-recovered-tests/ holds MiniMind's files
# verbatim as provenance — they are evidence, not code we are responsible for,
# and scanning them would charge us with their imports.
_EXCLUDED_DIRS = {"minimind-recovered-tests", "__pycache__", "site-packages"}


def _is_ours(path: Path) -> bool:
    """Our own source only.

    Virtual environments are matched by shape, not by one hardcoded name. The
    first version excluded the literal ".venv", so creating a second
    environment called ".venv-xpu" made this scan walk thousands of torch and
    Intel files and report their imports as undeclared. The same wrong
    assumption — that there is exactly one venv and it is called .venv — also
    let that directory slip past .gitignore.
    """
    for part in path.parts:
        if part in _EXCLUDED_DIRS:
            return False
        if part.startswith((".venv", "venv")) or part == "env":
            return False
    return True


SOURCES = sorted(p for p in ROOT.rglob("*.py") if _is_ours(p))


# ---------------------------------------------------------------- D7
# MiniMind called torch.load without weights_only at roughly eight sites while
# its README told users to download weights from the internet. On torch below
# 2.6 that lets a downloaded file execute code.

def test_every_torch_load_refuses_to_execute_code():
    offenders = []
    for path in SOURCES:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            target = node.func
            name = getattr(target, "attr", None) or getattr(target, "id", None)
            if name != "load":
                continue
            owner = getattr(getattr(target, "value", None), "id", None)
            if owner not in {"torch", None}:
                continue
            if owner != "torch":
                continue
            keywords = {k.arg for k in node.keywords}
            if "weights_only" not in keywords:
                offenders.append(f"{path.relative_to(ROOT)}:{node.lineno}")
    assert not offenders, (
        "torch.load without weights_only=True at: " + ", ".join(offenders)
    )


def test_torch_floor_is_at_least_the_version_that_defaults_to_safe_loading():
    text = (ROOT / "requirements.txt").read_text(encoding="utf-8")
    match = re.search(r"^torch>=(\d+)\.(\d+)", text, re.MULTILINE)
    assert match, "requirements.txt must pin torch with a floor"
    major, minor = int(match.group(1)), int(match.group(2))
    assert (major, minor) >= (2, 6), (
        f"torch>={major}.{minor} is below 2.6, where weights_only became the default"
    )


# ---------------------------------------------------------------- E2
# MiniMind commented torch out of requirements.txt, so the documented setup
# produced an environment that cannot start. 22 of its 34 pins were never
# imported, and 2 imports were never declared.

STDLIB = set(sys.stdlib_module_names)
# Declared distributions whose import name differs, or which arrive as a
# transitive dependency we rely on directly.
DISTRIBUTION_TO_MODULE = {
    "torch": "torch",
    "transformers": "transformers",
    "tokenizers": "tokenizers",
    "numpy": "numpy",
    "fastapi": "fastapi",
    "uvicorn": "uvicorn",
    "pydantic": "pydantic",
    "requests": "requests",
    "pytest": "pytest",
    "httpx": "httpx",
}


def _declared() -> set[str]:
    text = (ROOT / "requirements.txt").read_text(encoding="utf-8")
    pins = set()
    for line in text.splitlines():
        line = line.split("#")[0].strip()
        if line:
            pins.add(re.split(r"[<>=!\[]", line)[0].strip())
    return pins


def _imported() -> set[str]:
    modules = set()
    for path in SOURCES:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                modules.update(a.name.split(".")[0] for a in node.names)
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                modules.add(node.module.split(".")[0])
    return {m for m in modules if m not in STDLIB and m not in {"m3g4m1nd", "tests"}}


def test_torch_is_declared_and_not_commented_out():
    assert "torch" in _declared(), (
        "torch is the one library this repository cannot run without. MiniMind "
        "had it commented out, so its documented setup produced a broken env."
    )


def test_every_pin_is_actually_imported():
    declared, imported = _declared(), _imported()
    expected = {DISTRIBUTION_TO_MODULE.get(d, d) for d in declared}
    unused = {d for d in declared if DISTRIBUTION_TO_MODULE.get(d, d) not in imported}
    assert not unused, (
        f"declared but never imported: {sorted(unused)}. Either use them or drop them."
    )


def test_every_import_is_declared():
    declared = {DISTRIBUTION_TO_MODULE.get(d, d) for d in _declared()}
    undeclared = _imported() - declared
    assert not undeclared, (
        f"imported but not declared: {sorted(undeclared)}. Relying on a transitive "
        f"dependency is how a working checkout breaks on someone else's machine."
    )


# ---------------------------------------------------------------- E4
def test_attribution_files_exist_and_name_the_source():
    for filename in ("LICENSE", "NOTICE", "CREDITS.md"):
        assert (ROOT / filename).exists(), f"{filename} is missing"

    notice = (ROOT / "NOTICE").read_text(encoding="utf-8")
    assert "minimind" in notice.lower()
    assert "Apache" in notice
    assert "cc9629e" in notice, "NOTICE should pin the commit we read"

    credits = (ROOT / "CREDITS.md").read_text(encoding="utf-8")
    assert "jingyaogong" in credits
    for pr in ("#907", "#909", "#912", "#919", "#883", "#905"):
        assert pr in credits, f"recovered test from PR {pr} is not credited"


def test_the_recovered_minimind_tests_are_kept_verbatim():
    """The rule this repository exists to invert. These are the six tests
    MiniMind deleted, held unmodified as provenance. If the count or the line
    total moves, someone edited the evidence."""
    recovered = sorted((ROOT / "docs" / "minimind-recovered-tests").glob("test_*.py"))
    assert len(recovered) == 6, f"expected 6 recovered MiniMind tests, found {len(recovered)}"
    total = sum(len(p.read_text(encoding="utf-8").splitlines()) for p in recovered)
    assert total == 303, f"recovered tests are {total} lines; they were 303 when restored"


def test_every_recovered_test_is_either_ported_or_openly_listed_as_not_yet():
    """No silent gaps. Each original either has a live port or is named in the
    provenance README as outstanding, with the reason."""
    readme = (ROOT / "docs" / "minimind-recovered-tests" / "README.md").read_text(encoding="utf-8")
    for original in sorted((ROOT / "docs" / "minimind-recovered-tests").glob("test_*.py")):
        assert original.name in readme, f"{original.name} is not accounted for in the README"


def test_every_test_file_the_readme_names_actually_exists():
    """Added after the README's twelve-defect table spent a day naming twelve
    guard files that did not exist — a false claim in our own documentation,
    which is the defect class this whole repository was built to answer."""
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    named = set(re.findall(r"`(test_[a-z_]+\.py)`", readme))
    assert named, "the README no longer names any test file; the table has been gutted"

    missing = sorted(name for name in named if not (ROOT / "tests" / name).exists())
    assert not missing, f"README names test files that do not exist: {missing}"


def test_no_test_in_this_repository_is_skipped_or_expected_to_fail():
    """Walks decorators, not text — a substring scan matches its own source."""
    banned = {"skip", "skipif", "xfail"}
    offenders = []

    for path in sorted((ROOT / "tests").rglob("test_*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                continue
            for decorator in node.decorator_list:
                target = decorator.func if isinstance(decorator, ast.Call) else decorator
                parts = []
                while isinstance(target, ast.Attribute):
                    parts.append(target.attr)
                    target = target.value
                if isinstance(target, ast.Name):
                    parts.append(target.id)
                dotted = ".".join(reversed(parts))
                if dotted.split(".")[-1] in banned:
                    offenders.append(f"{path.relative_to(ROOT)}:{node.lineno} @{dotted}")

    assert not offenders, (
        "a test here is never skipped or expected to fail: " + "; ".join(offenders)
    )


def test_every_name_the_package_promises_actually_exists():
    """`__all__` is a claim about the public API, and a claim can be wrong."""
    import m3g4m1nd

    missing = [name for name in m3g4m1nd.__all__ if not hasattr(m3g4m1nd, name)]
    assert not missing, f"__all__ promises names that do not exist: {missing}"


# Modules deliberately left out of the top-level import, with the reason.
# `import m3g4m1nd` should not construct a web application or pull FastAPI and
# Starlette into every process that wanted a tensor.
IMPORT_ON_DEMAND = {"serve", "webui"}


def test_every_module_is_either_exported_or_listed_as_import_on_demand():
    """A module nobody can reach is a module nobody will use — unless there is
    a stated reason, in which case it is a decision rather than an oversight."""
    import m3g4m1nd

    modules = {
        path.stem for path in (ROOT / "m3g4m1nd").glob("*.py")
        if path.stem not in {"__init__", "__main__"}
    }
    unaccounted = sorted(
        name for name in modules
        if not hasattr(m3g4m1nd, name) and name not in IMPORT_ON_DEMAND
    )
    assert not unaccounted, (
        f"neither exported by __init__.py nor listed as import-on-demand: "
        f"{unaccounted}"
    )


def test_importing_the_package_does_not_build_a_web_application():
    """The reason `serve` is import-on-demand, asserted rather than asserted-to."""
    import subprocess
    import sys

    probe = (
        "import sys, m3g4m1nd; "
        "print(any(m == 'fastapi' or m.startswith('fastapi.') for m in sys.modules))"
    )
    result = subprocess.run(
        [sys.executable, "-c", probe], capture_output=True, text=True, cwd=ROOT, check=True
    )
    assert result.stdout.strip() == "False", (
        "importing m3g4m1nd pulled in FastAPI, so every process that wanted a "
        "tensor now pays for a web framework"
    )


def test_the_import_on_demand_list_is_not_a_dumping_ground():
    """It is two modules for one reason. If it grows, that needs arguing for."""
    assert IMPORT_ON_DEMAND == {"serve", "webui"}


def test_the_source_scan_ignores_any_virtual_environment_not_just_dot_venv():
    """Regression: a second environment named `.venv-xpu` made the scan walk
    torch's own source and fail. Matched by shape now, not by one name."""
    assert not _is_ours(Path(".venv/Lib/site-packages/torch/_init_.py"))
    assert not _is_ours(Path(".venv-xpu/Lib/site-packages/torch/_init_.py"))
    assert not _is_ours(Path("venv311/lib/x.py"))
    assert _is_ours(Path("m3g4m1nd/model.py"))
    assert _is_ours(Path("tests/test_hygiene.py"))


def test_the_scan_only_covers_our_own_files():
    assert 20 < len(SOURCES) < 80, (
        f"{len(SOURCES)} files scanned; that is not this repository's own source"
    )
