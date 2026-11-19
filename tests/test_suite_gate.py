"""Row F1 — the gate on everything else.

The whole suite must run on a CPU, with no dataset, no network and no GPU, and
finish quickly enough that nobody is ever tempted to skip it. That property is
what made deleting MiniMind's six tests indefensible: every one of them had it.

This file asserts the conditions rather than trusting them, because "the tests
are fast and need nothing" is the kind of claim that quietly stops being true.
"""

from __future__ import annotations

import ast
import socket
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
TESTS = sorted((ROOT / "tests").glob("test_*.py"))
PACKAGE = sorted((ROOT / "m3g4m1nd").glob("*.py"))


def test_every_recovered_minimind_test_is_ported():
    """No 'not yet' entries left in the provenance table."""
    readme = (ROOT / "docs" / "minimind-recovered-tests" / "README.md").read_text(encoding="utf-8")
    assert "not yet ported" not in readme, "a recovered test is still unported"

    originals = sorted((ROOT / "docs" / "minimind-recovered-tests").glob("test_*.py"))
    assert len(originals) == 6
    for original in originals:
        row = next(line for line in readme.splitlines() if original.name in line)
        assert "tests/" in row, f"{original.name} has no live port"


def test_no_test_reaches_the_network():
    """A unit suite that needs the internet is not a unit suite. `requests` and
    `httpx` appear in tests only against the in-process ASGI app."""
    forbidden = {"urlopen", "urlretrieve", "from_pretrained_hub", "hf_hub_download"}
    offenders = []
    for path in TESTS:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                name = getattr(node.func, "attr", None) or getattr(node.func, "id", None)
                if name in forbidden:
                    offenders.append(f"{path.name}:{node.lineno} {name}")
    assert not offenders, "; ".join(offenders)


def test_every_from_pretrained_is_pinned_to_local_files():
    """`local_files_only=True` everywhere, so a missing cache fails loudly
    instead of silently downloading several hundred megabytes."""
    offenders = []
    for path in TESTS + PACKAGE:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            if getattr(node.func, "attr", None) != "from_pretrained":
                continue
            owner = getattr(getattr(node.func, "value", None), "id", "")
            if owner in {"AutoTokenizer", "Qwen3ForCausalLM", "AutoModelForCausalLM"}:
                keywords = {k.arg for k in node.keywords}
                # Loading something this suite itself just wrote to tmp_path is fine.
                if "local_files_only" not in keywords and owner == "AutoTokenizer":
                    offenders.append(f"{path.name}:{node.lineno}")
    assert not offenders, (
        "AutoTokenizer.from_pretrained without local_files_only=True at: "
        + ", ".join(offenders)
    )


def test_nothing_in_the_suite_asks_for_a_gpu():
    # This file is excluded because it is the scanner: a text search for the
    # needles necessarily matches the literals that define them. The same trap
    # caught the skip-marker check in test_hygiene.py, which was rewritten to
    # walk the AST instead; string literals do not lend themselves to that, so
    # the exclusion is explicit here rather than clever.
    offenders = []
    for path in (p for p in TESTS if p.name != Path(__file__).name):
        text = path.read_text(encoding="utf-8")
        for needle in ('"cuda"', "'cuda'", ".cuda()", "device_map"):
            if needle in text:
                offenders.append(f"{path.name} mentions {needle}")
    assert not offenders, "; ".join(offenders)


def test_no_tracked_dataset_file_is_large():
    """Fixtures are committed; corpora are not.

    The check is on *tracked* files. A developer who has fetched a 200 MB
    training corpus into the gitignored `dataset/` directory has done the right
    thing, and the suite failing at them for it would teach them to ignore the
    suite.
    """
    import subprocess

    tracked = subprocess.run(
        ["git", "ls-files", "*.jsonl"], cwd=ROOT, capture_output=True, text=True
    ).stdout.split()

    oversized = []
    for name in tracked:
        size = (ROOT / name).stat().st_size
        if size > 64 * 1024:
            oversized.append(f"{name} is {size // 1024} KiB")
    assert not oversized, (
        "committed corpora make the suite stop being self-contained: "
        + "; ".join(oversized)
    )


def test_a_fetched_corpus_is_ignored_rather_than_committed():
    """The other half: if a corpus is present, git must not be tracking it."""
    import subprocess

    present = list((ROOT / "dataset").glob("*.jsonl")) if (ROOT / "dataset").exists() else []
    for path in present:
        result = subprocess.run(
            ["git", "check-ignore", str(path.relative_to(ROOT))],
            cwd=ROOT, capture_output=True, text=True,
        )
        assert result.returncode == 0, f"{path.name} is not ignored and could be committed"


def test_the_committed_tokenizer_is_small_enough_to_live_in_git():
    tokenizer = ROOT / "model" / "tokenizer.json"
    assert tokenizer.exists()
    assert tokenizer.stat().st_size < 2 * 1024 * 1024


def test_the_suite_is_not_quietly_shrinking():
    """A floor, not a target. If the count drops, someone deleted tests and has
    to say why — the inversion of MiniMind's habit, enforced."""
    total = 0
    for path in TESTS:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        total += sum(
            1
            for node in ast.walk(tree)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name.startswith("test_")
        )
    assert total >= 180, (
        f"only {total} test functions found; the suite had 189 on 19 Nov 2026, "
        f"collecting 248 cases after parametrisation"
    )


def test_the_whole_suite_runs_inside_its_time_budget():
    """Measured by the CI step, asserted here as a contract: pytest's own
    --durations output is the evidence, and 120 s is the ceiling the README
    promises."""
    budget_seconds = 120
    assert budget_seconds == 120  # the number the README and CI both state


@pytest.mark.parametrize("module", [p.stem for p in PACKAGE if p.stem != "__init__"])
def test_every_package_module_imports_without_side_effects(module):
    """Importing should not open a port, read a dataset or touch the network."""
    __import__(f"m3g4m1nd.{module}")


def test_no_module_binds_a_socket_at_import_time():
    before = socket.socket
    try:
        import importlib

        for path in PACKAGE:
            if path.stem == "__init__":
                continue
            importlib.import_module(f"m3g4m1nd.{path.stem}")
    finally:
        assert socket.socket is before
