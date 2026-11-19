"""The CI workflow, validated.

CI has never actually run — there is no remote, so GitHub has never executed
this file. That means the workflow is the one piece of the repository with no
evidence behind it, and the only honest thing to do short of a remote is to
check it here.

The sharpest test is the last one. The first version of this workflow had a
step titled "Refuse a suite that lost tests" which printed a count and exited
0. A check that reports a problem and passes is worse than no check, because
it occupies the place where a real one would go.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parent.parent
WORKFLOW = ROOT / ".github" / "workflows" / "tests.yml"


@pytest.fixture(scope="module")
def workflow():
    assert WORKFLOW.exists(), "the CI workflow is missing"
    return yaml.safe_load(WORKFLOW.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def job(workflow):
    jobs = workflow["jobs"]
    assert len(jobs) == 1, f"expected one job, found {sorted(jobs)}"
    return next(iter(jobs.values()))


def test_the_workflow_is_valid_yaml(workflow):
    assert isinstance(workflow, dict)
    assert workflow["name"] == "tests"


def test_it_runs_on_push_and_on_pull_request(workflow):
    # PyYAML reads a bare `on:` as the boolean True, which is a YAML 1.1 quirk
    # worth knowing about rather than being surprised by.
    triggers = workflow.get("on", workflow.get(True))
    assert triggers is not None, "the workflow has no triggers"
    assert set(triggers) >= {"push", "pull_request"}


def test_the_matrix_covers_the_python_this_repository_is_developed_on(job):
    versions = {str(v) for v in job["strategy"]["matrix"]["python-version"]}
    assert "3.14" in versions, (
        "this repository is developed and verified on 3.14; CI not testing it "
        "means CI is not testing what ships"
    )
    assert len(versions) >= 2, "one version is not a matrix"


def test_torch_is_installed_from_the_cpu_index(job):
    """A GitHub runner has no GPU. Installing a CUDA build would download two
    gigabytes to run on a CPU."""
    commands = " ".join(step.get("run", "") for step in job["steps"])
    assert "download.pytorch.org/whl/cpu" in commands


def test_the_requirements_file_is_installed(job):
    commands = " ".join(step.get("run", "") for step in job["steps"])
    assert "-r requirements.txt" in commands


def test_the_suite_is_actually_run(job):
    commands = " ".join(step.get("run", "") for step in job["steps"])
    assert "pytest" in commands


def test_the_job_cannot_hang_forever(job):
    assert "timeout-minutes" in job, "a job with no timeout can occupy a runner"
    assert 1 <= job["timeout-minutes"] <= 60


def test_the_matrix_does_not_stop_at_the_first_failure(job):
    assert job["strategy"].get("fail-fast") is False, (
        "fail-fast hides whether a failure is version-specific"
    )


def test_no_step_reports_a_problem_and_then_passes():
    """The regression guard for a defect this workflow actually had.

    A step whose body only echoes cannot fail, so a step *named* as a check
    while only echoing is a check that always passes. Any step claiming to
    refuse, verify, fail or enforce something must contain a command capable
    of returning non-zero.
    """
    text = WORKFLOW.read_text(encoding="utf-8")
    config = yaml.safe_load(text)
    job = next(iter(config["jobs"].values()))

    claims = ("refuse", "verify", "enforce", "fail", "check", "assert", "must")
    offenders = []
    for step in job["steps"]:
        name = (step.get("name") or "").lower()
        body = step.get("run")
        if body is None or not any(word in name for word in claims):
            continue
        lines = [
            line.strip() for line in body.splitlines()
            if line.strip() and not line.strip().startswith("#")
        ]
        if lines and all(line.startswith("echo ") for line in lines):
            offenders.append(step.get("name"))

    assert not offenders, (
        f"these steps are named as checks but can only ever pass: {offenders}"
    )


def test_every_step_either_runs_something_or_uses_an_action(job):
    for step in job["steps"]:
        assert "run" in step or "uses" in step, f"inert step: {step}"


def test_pinned_actions_are_major_versioned(job):
    for step in job["steps"]:
        action = step.get("uses")
        if action:
            assert "@" in action, f"{action} is unpinned"
