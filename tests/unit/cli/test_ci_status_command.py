"""The `ci-status` wrapper's own work: which workflows it asks about, and what it exits.

This gate exists because a check that reports is not a gate (docs/todo.md TASK-0313), so
every assertion here is about the exit code rather than the wording. The three ways it can
fail to learn anything - no repository, no workflows, no answer - must each exit non-zero,
because a check that passes because it could not run is the defect it replaces.
"""

import json
from pathlib import Path

import pytest

from fasterrag.cli.commands import diagnostics
from fasterrag.cli.main import main
from fasterrag.cli.output import ExitCode
from fasterrag.services.ci_status import (
    BranchVerdict,
    JobSummary,
    Remote,
    RunSummary,
    WorkflowVerdict,
)

REMOTE = Remote(owner="balrajsingh165", repo="FasterRag")
RAN = JobSummary(name="quality", conclusion="success")


def summary(conclusion: str, workflow: str = "ci.yml") -> RunSummary:
    """Return a run summary with the given conclusion and one job that actually ran."""
    return RunSummary(
        workflow_file=workflow,
        branch="main",
        run_number=163,
        sha="6be46f9ec87be32b427505979385c9dab2d604fe",
        conclusion=conclusion,
        url="https://github.com/balrajsingh165/FasterRag/actions/runs/42522906097",
        jobs=(RAN,),
    )


def verdict(*workflows: WorkflowVerdict) -> BranchVerdict:
    """Return a branch verdict over the given workflow verdicts."""
    return BranchVerdict(slug=REMOTE.slug, branch="main", workflows=workflows)


@pytest.fixture
def checkout(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Run the command from a directory that looks like a checkout with three workflows."""
    directory = tmp_path / ".github" / "workflows"
    directory.mkdir(parents=True)
    for name in ("ci.yml", "nightly.yml", "security.yml"):
        (directory / name).write_text("on: push\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    return tmp_path


@pytest.fixture(autouse=True)
def origin(monkeypatch: pytest.MonkeyPatch) -> None:
    """Answer the origin lookup with this repository, so no test shells out to git."""

    async def remote(*args: object, **kwargs: object) -> Remote:
        return REMOTE

    monkeypatch.setattr(diagnostics, "origin_remote", remote)


def answer(
    monkeypatch: pytest.MonkeyPatch, result: BranchVerdict, seen: dict[str, object] | None = None
) -> None:
    """Make the service return ``result`` and record the call."""

    async def verdicts(remote: Remote, workflows: list[str], **kwargs: object) -> BranchVerdict:
        if seen is not None:
            seen["workflows"] = list(workflows)
            seen.update(kwargs)
        return result

    monkeypatch.setattr(diagnostics, "branch_verdict", verdicts)


def test_a_red_workflow_exits_non_zero(
    checkout: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """The whole point: a red conclusion on main must fail, not be reported."""
    answer(
        monkeypatch,
        verdict(
            WorkflowVerdict("ci.yml", run=summary("success")),
            WorkflowVerdict("nightly.yml", run=summary("failure", "nightly.yml")),
        ),
    )

    code = main(["ci-status"])

    assert code == ExitCode.FAILURE
    assert "nightly.yml" in capsys.readouterr().err


def test_a_green_branch_exits_zero(checkout: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    answer(monkeypatch, verdict(WorkflowVerdict("ci.yml", run=summary("success"))))

    assert main(["ci-status"]) == ExitCode.SUCCESS


def test_a_cancelled_run_is_not_a_pass(checkout: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """cancel-in-progress left 21 commits with no verdict; no verdict is not a green one."""
    answer(monkeypatch, verdict(WorkflowVerdict("ci.yml", run=summary("cancelled"))))

    assert main(["ci-status"]) == ExitCode.FAILURE


def test_a_workflow_that_never_ran_is_not_a_pass(
    checkout: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    answer(monkeypatch, verdict(WorkflowVerdict("nightly.yml")))

    assert main(["ci-status"]) == ExitCode.FAILURE


def test_an_unreadable_api_exits_three_not_zero(
    checkout: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A check that went green because it could not reach GitHub would be the same bug."""
    answer(monkeypatch, verdict(WorkflowVerdict("ci.yml", unreachable="no network")))

    assert main(["ci-status"]) == ExitCode.UNREACHABLE


def test_a_red_workflow_beside_an_unreadable_one_still_exits_one(
    checkout: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """3 means only could-not-ask; a real red must not be downgraded to it."""
    answer(
        monkeypatch,
        verdict(
            WorkflowVerdict("ci.yml", unreachable="no network"),
            WorkflowVerdict("nightly.yml", run=summary("failure", "nightly.yml")),
        ),
    )

    assert main(["ci-status"]) == ExitCode.FAILURE


def test_checking_nothing_exits_non_zero(
    checkout: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Excluding every workflow must fail rather than report an empty success."""
    answer(monkeypatch, verdict())

    code = main(
        [
            "ci-status",
            "--exclude",
            "ci.yml",
            "--exclude",
            "nightly.yml",
            "--exclude",
            "security.yml",
        ]
    )

    assert code == ExitCode.FAILURE
    assert "no workflows" in capsys.readouterr().err


def test_a_checkout_without_workflows_exits_non_zero(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A renamed or missing .github/workflows is the silent-vacuity shape."""
    monkeypatch.chdir(tmp_path)
    answer(monkeypatch, verdict())

    assert main(["ci-status"]) == ExitCode.FAILURE


def test_without_a_github_remote_it_exits_two(
    checkout: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def nowhere(*args: object, **kwargs: object) -> Remote | None:
        return None

    monkeypatch.setattr(diagnostics, "origin_remote", nowhere)
    answer(monkeypatch, verdict())

    assert main(["ci-status"]) == ExitCode.USAGE


def test_every_discovered_workflow_is_asked_about(
    checkout: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Discovered from the checkout, so a fourth workflow is gated the day it lands."""
    seen: dict[str, object] = {}
    answer(monkeypatch, verdict(WorkflowVerdict("ci.yml", run=summary("success"))), seen)

    main(["ci-status"])

    assert seen["workflows"] == ["ci.yml", "nightly.yml", "security.yml"]


def test_exclude_removes_only_what_it_names(
    checkout: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The CI step excludes the workflow it runs inside; the rest must survive."""
    seen: dict[str, object] = {}
    answer(monkeypatch, verdict(WorkflowVerdict("ci.yml", run=summary("success"))), seen)

    main(["ci-status", "--exclude", "ci.yml"])

    assert seen["workflows"] == ["nightly.yml", "security.yml"]


def test_workflow_overrides_discovery(checkout: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    seen: dict[str, object] = {}
    answer(monkeypatch, verdict(WorkflowVerdict("ci.yml", run=summary("success"))), seen)

    main(["ci-status", "--workflow", "nightly.yml", "--branch", "release"])

    assert seen["workflows"] == ["nightly.yml"]
    assert seen["branch"] == "release"


def test_the_token_comes_from_the_named_variable_only(
    checkout: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Named explicitly, never picked up: GITHUB_TOKEN is ignored unless it is the one named."""
    monkeypatch.setenv("GITHUB_TOKEN", "ghp_ambient")
    monkeypatch.setenv("CI_READ_TOKEN", "ghs_named")
    seen: dict[str, object] = {}
    answer(monkeypatch, verdict(WorkflowVerdict("ci.yml", run=summary("success"))), seen)

    main(["ci-status", "--token-env", "CI_READ_TOKEN"])

    assert seen["token"] == "ghs_named"


def test_no_token_flag_means_no_token(checkout: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GITHUB_TOKEN", "ghp_ambient")
    seen: dict[str, object] = {}
    answer(monkeypatch, verdict(WorkflowVerdict("ci.yml", run=summary("success"))), seen)

    main(["ci-status"])

    assert seen["token"] is None


def test_a_token_env_that_is_unset_exits_two(
    checkout: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Asked to authenticate and could not: that is a usage error, not a silent downgrade."""
    monkeypatch.delenv("ABSENT_TOKEN", raising=False)
    answer(monkeypatch, verdict(WorkflowVerdict("ci.yml", run=summary("success"))))

    assert main(["ci-status", "--token-env", "ABSENT_TOKEN"]) == ExitCode.USAGE


def test_json_stays_one_parseable_document(
    checkout: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    answer(
        monkeypatch,
        verdict(WorkflowVerdict("nightly.yml", run=summary("failure", "nightly.yml"))),
    )

    code = main(["ci-status", "--json"])

    payload = json.loads(capsys.readouterr().out)
    assert code == ExitCode.FAILURE
    assert payload["green"] is False
    assert payload["workflows"][0]["conclusion"] == "failure"
    assert payload["workflows"][0]["url"].startswith("https://github.com/")


def test_no_token_value_ever_reaches_the_output(
    checkout: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """The flag names a variable; the value it holds is a credential and stays one."""
    monkeypatch.setenv("CI_READ_TOKEN", "ghs_must_not_appear")
    answer(monkeypatch, verdict(WorkflowVerdict("ci.yml", run=summary("failure"))))

    main(["ci-status", "--json", "--token-env", "CI_READ_TOKEN"])

    captured = capsys.readouterr()
    assert "ghs_must_not_appear" not in captured.out
    assert "ghs_must_not_appear" not in captured.err
