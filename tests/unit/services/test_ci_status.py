"""The Actions API client: what it asks, what it never sends, and how every failure is typed.

The payloads here mirror what api.github.com returned for this repository on 2026-09-13
(run #120 for `995eb84c`, `integration` failing at "Adapter contract suite against real
backends"), so the fake is shaped like the real thing rather than like the parser.
"""

import subprocess
from collections.abc import Callable
from pathlib import Path
from typing import Any

import httpx
import pytest

from fasterrag.errors import ErrorCode, ProvisioningError
from fasterrag.services import ci_status
from fasterrag.services.ci_status import (
    JobSummary,
    Remote,
    RunSummary,
    latest_run,
    origin_remote,
    parse_remote,
)

REMOTE = Remote(owner="balrajsingh165", repo="FasterRag")
SHA = "995eb84c4efc5649ce33ddf3a05e2fe2087f284e"
RUN_ID = 33406200616
RUN_URL = f"https://github.com/balrajsingh165/FasterRag/actions/runs/{RUN_ID}"
CONTRACT_STEP = "Adapter contract suite against real backends"

Handler = Callable[[httpx.Request], httpx.Response]


def serve(handler: Handler, monkeypatch: pytest.MonkeyPatch) -> list[httpx.Request]:
    """Route every AsyncClient the module builds through an in-memory transport.

    Returns the requests as they reach the transport — after httpx has applied any auth it
    was going to apply, which is what the no-token tests need to see. The real class is
    captured before patching (``ci_status.httpx`` is the global module, so the patched
    name is the one this helper would otherwise resolve), and the fake takes exactly the
    keyword arguments production passes so a new one fails here rather than diverging.
    """
    real_client = httpx.AsyncClient
    seen: list[httpx.Request] = []

    def recording(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return handler(request)

    def client(
        *, base_url: str, headers: dict[str, str], timeout: float, trust_env: bool
    ) -> httpx.AsyncClient:
        return real_client(
            transport=httpx.MockTransport(recording),
            base_url=base_url,
            headers=headers,
            timeout=timeout,
            trust_env=trust_env,
        )

    monkeypatch.setattr("fasterrag.services.ci_status.httpx.AsyncClient", client)
    return seen


def run(conclusion: str = "success", *, number: int = 120, sha: str = SHA) -> dict[str, Any]:
    """Return one workflow run in the API's shape."""
    return {
        "id": RUN_ID,
        "name": "ci",
        "head_branch": "main",
        "head_sha": sha,
        "status": "completed",
        "conclusion": conclusion,
        "run_number": number,
        "html_url": RUN_URL,
    }


def job(name: str, conclusion: str, *steps: tuple[str, str]) -> dict[str, Any]:
    """Return one job in the API's shape, with ``(step name, conclusion)`` pairs."""
    return {
        "name": name,
        "status": "completed",
        "conclusion": conclusion,
        "steps": [
            {"name": step, "status": "completed", "conclusion": outcome, "number": index}
            for index, (step, outcome) in enumerate(steps, start=1)
        ],
    }


GREEN_JOBS = [
    job("quality", "success", ("Install", "success"), ("Unit tests", "success")),
    job("windows", "success", ("Unit tests", "success")),
    job("commit-messages", "skipped"),
]

RED_JOBS = [
    job("models", "success", ("Verify against real embedding and reranking models", "success")),
    job("integration", "failure", ("Install", "success"), (CONTRACT_STEP, "failure")),
    job("quality", "success", ("Typecheck", "success")),
    job("windows", "success", ("Unit tests", "success")),
    job("supply-chain", "success", ("Lockfile is current", "success")),
    job("commit-messages", "skipped"),
]


def github(runs: list[dict[str, Any]], jobs: list[dict[str, Any]]) -> Handler:
    """Return a handler answering the two endpoints the client reads."""

    def handle(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/runs"):
            return httpx.Response(200, json={"total_count": len(runs), "workflow_runs": runs})
        if request.url.path.endswith("/jobs"):
            return httpx.Response(200, json={"total_count": len(jobs), "jobs": jobs})
        return httpx.Response(404, json={"message": "Not Found"})

    return handle


async def test_a_green_run_is_summarised(monkeypatch: pytest.MonkeyPatch) -> None:
    serve(github([run("success")], GREEN_JOBS), monkeypatch)

    summary = await latest_run(REMOTE)

    assert summary == RunSummary(
        workflow_file="ci.yml",
        branch="main",
        run_number=120,
        sha=SHA,
        conclusion="success",
        url=RUN_URL,
        jobs=(
            JobSummary(name="quality", conclusion="success"),
            JobSummary(name="windows", conclusion="success"),
            JobSummary(name="commit-messages", conclusion="skipped"),
        ),
    )
    assert summary.passed is True
    assert summary.failing_jobs == ()
    assert summary.short_sha == "995eb84c"


async def test_a_red_run_names_the_failing_jobs_and_their_failing_steps(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """What the API returned for this repository on 2026-09-13, reduced to what matters."""
    serve(github([run("failure")], RED_JOBS), monkeypatch)

    summary = await latest_run(REMOTE)

    assert summary is not None
    assert summary.passed is False
    assert summary.failing_jobs == (
        JobSummary(name="integration", conclusion="failure", failing_steps=(CONTRACT_STEP,)),
    )
    assert [job.name for job in summary.jobs] == [
        "models",
        "integration",
        "quality",
        "windows",
        "supply-chain",
        "commit-messages",
    ]


async def test_a_skipped_job_is_not_a_failing_one(monkeypatch: pytest.MonkeyPatch) -> None:
    """`commit-messages` is skipped on every push to main; that is not a red job."""
    serve(github([run("success")], GREEN_JOBS), monkeypatch)

    summary = await latest_run(REMOTE)

    assert summary is not None
    assert all(not job.failed for job in summary.jobs)


async def test_a_cancelled_run_is_not_passed(monkeypatch: pytest.MonkeyPatch) -> None:
    """cancel-in-progress leaves cancelled runs on main; no verdict is not a green one."""
    serve(github([run("cancelled")], [job("quality", "cancelled")]), monkeypatch)

    summary = await latest_run(REMOTE)

    assert summary is not None
    assert summary.passed is False
    assert summary.conclusion == "cancelled"


async def test_the_two_requests_are_the_documented_ones(monkeypatch: pytest.MonkeyPatch) -> None:
    """Runs filtered to the branch's latest completed one, then that run's jobs."""
    seen = serve(github([run()], GREEN_JOBS), monkeypatch)

    await latest_run(REMOTE, branch="release", workflow_file="nightly.yml")

    assert [request.url.path for request in seen] == [
        "/repos/balrajsingh165/FasterRag/actions/workflows/nightly.yml/runs",
        f"/repos/balrajsingh165/FasterRag/actions/runs/{RUN_ID}/jobs",
    ]
    assert dict(seen[0].url.params) == {"branch": "release", "status": "completed", "per_page": "1"}
    assert seen[0].headers["Accept"] == "application/vnd.github+json"
    assert seen[0].headers["X-GitHub-Api-Version"] == "2022-11-28"
    assert seen[0].headers["User-Agent"].startswith("fasterrag/")


async def test_a_workflow_that_never_completed_is_none_not_an_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen = serve(github([], []), monkeypatch)

    assert await latest_run(REMOTE) is None
    assert len(seen) == 1


async def test_no_token_is_sent_even_when_one_is_in_the_environment(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """The client must never authenticate: not from GITHUB_TOKEN, GH_TOKEN, or a .netrc.

    A ``.netrc`` is the one httpx would pick up on its own under ``trust_env``, so the
    test plants one for api.github.com and then looks at the request as it reached the
    transport, after httpx applied whatever auth it was going to.
    """
    netrc = tmp_path / "netrc"
    netrc.write_text("machine api.github.com login octocat password netrc-secret\n")
    monkeypatch.setenv("NETRC", str(netrc))
    monkeypatch.setenv("GITHUB_TOKEN", "ghp_environment-secret")
    monkeypatch.setenv("GH_TOKEN", "gho_environment-secret")
    seen = serve(github([run("failure")], RED_JOBS), monkeypatch)

    with caplog.at_level("DEBUG"):
        await latest_run(REMOTE)

    assert len(seen) == 2
    for request in seen:
        assert "authorization" not in request.headers
        assert "cookie" not in request.headers
        for value in request.headers.values():
            assert "secret" not in value
    assert "secret" not in caplog.text


async def test_the_client_opts_out_of_environment_trust(monkeypatch: pytest.MonkeyPatch) -> None:
    """`trust_env=False` is the mechanism behind the no-token guarantee; pin it by name."""
    real_client = httpx.AsyncClient
    captured: dict[str, Any] = {}

    def client(**kwargs: Any) -> httpx.AsyncClient:
        captured.update(kwargs)
        return real_client(transport=httpx.MockTransport(github([run()], GREEN_JOBS)), **kwargs)

    monkeypatch.setattr("fasterrag.services.ci_status.httpx.AsyncClient", client)

    await latest_run(REMOTE)

    assert captured["trust_env"] is False
    assert "auth" not in captured


async def test_no_network_is_a_typed_error_with_the_cause_chained(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def unreachable(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("[Errno 11001] getaddrinfo failed", request=request)

    serve(unreachable, monkeypatch)

    with pytest.raises(ProvisioningError) as caught:
        await latest_run(REMOTE)

    assert caught.value.code is ErrorCode.PROVISIONING_FAILED
    assert "api.github.com" in caught.value.detail
    assert "ConnectError" in caught.value.detail
    assert isinstance(caught.value.__cause__, httpx.ConnectError)
    assert caught.value.fix


async def test_a_timeout_is_a_typed_error(monkeypatch: pytest.MonkeyPatch) -> None:
    def slow(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("timed out", request=request)

    serve(slow, monkeypatch)

    with pytest.raises(ProvisioningError, match="ReadTimeout"):
        await latest_run(REMOTE)


async def test_a_spent_rate_limit_is_rate_limited_with_the_reset_time(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """GitHub's 403 with X-RateLimit-Remaining: 0 is the anonymous budget running out."""

    def limited(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            403,
            headers={
                "X-RateLimit-Limit": "60",
                "X-RateLimit-Remaining": "0",
                "X-RateLimit-Reset": "1789309119",
            },
            json={"message": "API rate limit exceeded for 203.0.113.9."},
        )

    serve(limited, monkeypatch)

    with pytest.raises(ProvisioningError) as caught:
        await latest_run(REMOTE)

    assert caught.value.code is ErrorCode.RATE_LIMITED
    assert caught.value.retryable is True
    assert "rate limit" in caught.value.detail
    assert caught.value.fix is not None
    assert "14:18:39 UTC" in caught.value.fix


async def test_a_429_is_rate_limited_without_the_header(monkeypatch: pytest.MonkeyPatch) -> None:
    serve(lambda request: httpx.Response(429, json={"message": "slow down"}), monkeypatch)

    with pytest.raises(ProvisioningError) as caught:
        await latest_run(REMOTE)

    assert caught.value.code is ErrorCode.RATE_LIMITED
    assert caught.value.fix is not None
    assert "60 requests an hour" in caught.value.fix


async def test_a_403_with_budget_left_is_not_a_rate_limit(monkeypatch: pytest.MonkeyPatch) -> None:
    def forbidden(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            403, headers={"X-RateLimit-Remaining": "42"}, json={"message": "Forbidden"}
        )

    serve(forbidden, monkeypatch)

    with pytest.raises(ProvisioningError) as caught:
        await latest_run(REMOTE)

    assert caught.value.code is ErrorCode.PROVISIONING_FAILED
    assert "HTTP 403" in caught.value.detail


async def test_a_missing_workflow_is_not_found(monkeypatch: pytest.MonkeyPatch) -> None:
    """A private repository and a renamed workflow both answer 404 to an anonymous call."""
    serve(lambda request: httpx.Response(404, json={"message": "Not Found"}), monkeypatch)

    with pytest.raises(ProvisioningError) as caught:
        await latest_run(REMOTE, workflow_file="renamed.yml")

    assert caught.value.code is ErrorCode.NOT_FOUND
    assert "renamed.yml" in caught.value.detail
    assert caught.value.fix is not None
    assert "public" in caught.value.fix


async def test_a_non_json_body_is_a_typed_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """A captive portal answers 200 with HTML; that must not surface as a JSONDecodeError."""
    serve(
        lambda request: httpx.Response(
            200, headers={"Content-Type": "text/html"}, text="<html>sign in</html>"
        ),
        monkeypatch,
    )

    with pytest.raises(ProvisioningError) as caught:
        await latest_run(REMOTE)

    assert caught.value.code is ErrorCode.PROVISIONING_FAILED
    assert "not JSON" in caught.value.detail
    assert isinstance(caught.value.__cause__, ValueError)


async def test_a_server_error_is_a_typed_error(monkeypatch: pytest.MonkeyPatch) -> None:
    serve(lambda request: httpx.Response(502, text="bad gateway"), monkeypatch)

    with pytest.raises(ProvisioningError, match="HTTP 502"):
        await latest_run(REMOTE)


@pytest.mark.parametrize(
    ("body", "expected"),
    [
        ({"total_count": 1}, "no 'workflow_runs' field"),
        ([], "is not a JSON object"),
        ({"workflow_runs": {"id": 1}}, "is not a JSON array"),
        ({"workflow_runs": [{"conclusion": "success"}]}, "no 'id' field"),
        ({"workflow_runs": ["not a run"]}, "is not a JSON object"),
    ],
)
async def test_a_payload_missing_what_is_read_is_a_typed_error(
    body: Any, expected: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Every KeyError and TypeError the payload could raise is a ProvisioningError instead."""
    serve(lambda request: httpx.Response(200, json=body), monkeypatch)

    with pytest.raises(ProvisioningError, match=expected):
        await latest_run(REMOTE)


async def test_a_run_with_a_non_numeric_number_is_a_typed_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    broken = dict(run(), run_number="one hundred and twenty")
    serve(github([broken], GREEN_JOBS), monkeypatch)

    with pytest.raises(ProvisioningError, match="non-numeric"):
        await latest_run(REMOTE)


async def test_a_job_payload_without_steps_is_summarised(monkeypatch: pytest.MonkeyPatch) -> None:
    """A queued-then-cancelled job carries `steps: null`; that is a job, not a payload bug."""
    stepless = {"name": "windows", "status": "completed", "conclusion": "cancelled", "steps": None}
    serve(github([run("cancelled")], [stepless]), monkeypatch)

    summary = await latest_run(REMOTE)

    assert summary is not None
    assert summary.jobs == (JobSummary(name="windows", conclusion="cancelled"),)


@pytest.mark.parametrize(
    "url",
    [
        "https://github.com/acme/widgets.git",
        "https://github.com/acme/widgets",
        "https://github.com/acme/widgets/",
        "http://github.com/acme/widgets.git",
        "https://x-access-token:ghs_secret@github.com/acme/widgets.git",
        "git@github.com:acme/widgets.git",
        "git@github.com:acme/widgets",
        "ssh://git@github.com/acme/widgets.git",
        "  https://github.com/acme/widgets.git\n",
    ],
)
def test_every_remote_form_git_writes_is_parsed(url: str) -> None:
    assert parse_remote(url) == Remote(owner="acme", repo="widgets")


@pytest.mark.parametrize(
    "url",
    [
        "https://gitlab.com/acme/widgets.git",
        "git@bitbucket.org:acme/widgets.git",
        "https://github.com/acme",
        "https://github.com/acme/widgets/tree/main",
        "/srv/git/widgets.git",
        "",
    ],
)
def test_a_remote_that_is_not_a_github_repository_is_none(url: str) -> None:
    assert parse_remote(url) is None


def test_the_slug_is_owner_slash_repo() -> None:
    assert Remote(owner="acme", repo="widgets").slug == "acme/widgets"


def git(*args: str, cwd: Path) -> None:
    """Run one git command in ``cwd``, failing the test if it fails."""
    subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True, timeout=30)


async def test_origin_is_read_from_a_real_checkout(tmp_path: Path) -> None:
    git("init", "-q", cwd=tmp_path)
    git("remote", "add", "origin", "git@github.com:acme/widgets.git", cwd=tmp_path)

    assert await origin_remote(tmp_path) == Remote(owner="acme", repo="widgets")


async def test_a_checkout_without_an_origin_is_none(tmp_path: Path) -> None:
    git("init", "-q", cwd=tmp_path)

    assert await origin_remote(tmp_path) is None


async def test_a_directory_that_is_not_a_checkout_is_none(tmp_path: Path) -> None:
    assert await origin_remote(tmp_path) is None


async def test_a_non_github_origin_is_none(tmp_path: Path) -> None:
    git("init", "-q", cwd=tmp_path)
    git("remote", "add", "origin", "https://gitlab.com/acme/widgets.git", cwd=tmp_path)

    assert await origin_remote(tmp_path) is None


async def test_a_credential_in_the_remote_url_never_reaches_the_log(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """`git remote get-url` happily prints an embedded token; this module must not."""
    git("init", "-q", cwd=tmp_path)
    git(
        "remote",
        "add",
        "origin",
        "https://x-access-token:ghs_embedded-secret@github.com/acme/widgets.git",
        cwd=tmp_path,
    )

    with caplog.at_level("DEBUG", logger=ci_status.__name__):
        remote = await origin_remote(tmp_path)

    assert remote == Remote(owner="acme", repo="widgets")
    assert "ghs_embedded-secret" not in caplog.text
    assert "x-access-token" not in caplog.text


async def test_a_missing_git_executable_is_none(monkeypatch: pytest.MonkeyPatch) -> None:
    async def absent(*args: object, **kwargs: object) -> object:
        raise FileNotFoundError("git")

    monkeypatch.setattr("fasterrag.services.ci_status.asyncio.create_subprocess_exec", absent)

    assert await origin_remote() is None


REPO_ROOT = Path(__file__).resolve().parents[3]

RED_163 = {
    "id": 42522906097,
    "name": "ci",
    "head_branch": "main",
    "head_sha": "6be46f9ec87be32b427505979385c9dab2d604fe",
    "status": "completed",
    "conclusion": "failure",
    "run_number": 163,
    "event": "push",
    "html_url": "https://github.com/balrajsingh165/FasterRag/actions/runs/42522906097",
}
RED_163_JOBS = [
    job("supply-chain", "success"),
    job("models", "success"),
    job("packaging", "success"),
    job("quality", "failure", ("Install", "success"), ("Typecheck", "failure")),
    job("windows", "success"),
    job("integration", "success"),
    job("eval-gate", "success"),
    job("commit-messages", "skipped"),
]

GREEN_164 = dict(run("success", number=164, sha="882a07ea" + "0" * 32), event="push")


def workflows(answers: dict[str, tuple[list[dict[str, Any]], list[dict[str, Any]]]]) -> Handler:
    """Return a handler serving a different run and job list per workflow file."""

    def handle(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        for workflow, (runs, _) in answers.items():
            if path.endswith(f"/workflows/{workflow}/runs"):
                return httpx.Response(200, json={"total_count": len(runs), "workflow_runs": runs})
        if path.endswith("/jobs"):
            run_id = int(path.split("/runs/")[1].split("/")[0])
            for runs, jobs in answers.values():
                if runs and runs[0]["id"] == run_id:
                    return httpx.Response(200, json={"total_count": len(jobs), "jobs": jobs})
        return httpx.Response(404, json={"message": "Not Found"})

    return handle


async def test_the_real_red_run_of_2026_11_08_is_not_green(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Run #163 as api.github.com returned it: main red on quality / Typecheck.

    This is the run that was red for three days while the signal a human reads said green,
    and it is the thing the gate has to turn into a non-zero exit.
    """
    serve(github([RED_163], RED_163_JOBS), monkeypatch)

    verdict = await ci_status.branch_verdict(REMOTE, ["ci.yml"])

    assert verdict.green is False
    assert verdict.slug == "balrajsingh165/FasterRag"
    assert len(verdict.red) == 1
    assert verdict.red[0].run is not None
    assert verdict.red[0].run.conclusion == "failure"
    assert verdict.red[0].run.failing_jobs == (
        JobSummary(name="quality", conclusion="failure", failing_steps=("Typecheck",)),
    )
    assert "quality" in verdict.red[0].reason


async def test_a_red_scheduled_workflow_fails_a_green_push_workflow(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The trap TASK-0313 names: ci green on push, nightly red on a schedule.

    A check that read only ci.yml - the run attached to the commit, the one a human opens
    - would call this branch green and reproduce the defect it exists to catch.
    """
    nightly_red = dict(RED_163, id=99, name="nightly", run_number=99, event="schedule")
    serve(
        workflows(
            {
                "ci.yml": ([GREEN_164], GREEN_JOBS),
                "nightly.yml": ([nightly_red], RED_163_JOBS),
                "security.yml": ([dict(GREEN_164, id=23, run_number=23)], GREEN_JOBS),
            }
        ),
        monkeypatch,
    )

    verdict = await ci_status.branch_verdict(
        REMOTE, ["ci.yml", "nightly.yml", "security.yml"], branch="main"
    )

    assert [workflow.workflow_file for workflow in verdict.workflows] == [
        "ci.yml",
        "nightly.yml",
        "security.yml",
    ]
    assert verdict.workflows[0].green is True
    assert verdict.green is False
    assert [workflow.workflow_file for workflow in verdict.red] == ["nightly.yml"]


async def test_the_run_listing_is_never_filtered_by_trigger(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """No event parameter, ever: filtering to push is exactly how the bug hides."""
    seen = serve(github([GREEN_164], GREEN_JOBS), monkeypatch)

    await ci_status.branch_verdict(REMOTE, ["ci.yml"])

    assert "event" not in dict(seen[0].url.params)


async def test_a_workflow_with_no_completed_run_is_not_green(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Zero runs is the vacuity shape: nothing was measured, so nothing is green."""
    serve(github([], []), monkeypatch)

    verdict = await ci_status.branch_verdict(REMOTE, ["ci.yml"])

    assert verdict.green is False
    assert verdict.workflows[0].run is None
    assert "no completed run" in verdict.workflows[0].reason


async def test_an_unreachable_api_is_not_green(monkeypatch: pytest.MonkeyPatch) -> None:
    """The other vacuity shape: the check could not ask, so it must not answer green."""

    def unreachable(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("[Errno 11001] getaddrinfo failed", request=request)

    serve(unreachable, monkeypatch)

    verdict = await ci_status.branch_verdict(REMOTE, ["ci.yml", "nightly.yml"])

    assert verdict.green is False
    assert len(verdict.red) == 2
    assert all(workflow.unreachable is not None for workflow in verdict.red)
    assert "could not be read" in verdict.red[0].reason


async def test_one_unreadable_workflow_does_not_hide_another_workflows_red(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A 404 on the first workflow must not short-circuit the second one's verdict."""
    serve(workflows({"nightly.yml": ([RED_163], RED_163_JOBS)}), monkeypatch)

    verdict = await ci_status.branch_verdict(REMOTE, ["renamed.yml", "nightly.yml"])

    assert verdict.green is False
    assert verdict.workflows[0].unreachable is not None
    assert verdict.workflows[1].run is not None
    assert verdict.workflows[1].run.conclusion == "failure"


def test_no_workflows_at_all_is_not_green() -> None:
    """An empty check is the defect, not a pass; this is asserted, not assumed."""
    assert ci_status.BranchVerdict(slug="acme/widgets", branch="main").green is False


async def test_every_workflow_green_is_green(monkeypatch: pytest.MonkeyPatch) -> None:
    serve(github([GREEN_164], GREEN_JOBS), monkeypatch)

    verdict = await ci_status.branch_verdict(REMOTE, ["ci.yml"])

    assert verdict.green is True
    assert verdict.red == ()
    assert verdict.as_dict()["green"] is True


async def test_an_explicit_token_is_sent_as_a_bearer(monkeypatch: pytest.MonkeyPatch) -> None:
    """CI shares the anonymous 60-an-hour budget with the runner's neighbours, so it opts in."""
    seen = serve(github([GREEN_164], GREEN_JOBS), monkeypatch)

    await latest_run(REMOTE, token="ghs_explicit")

    assert seen[0].headers["Authorization"] == "Bearer ghs_explicit"


async def test_a_token_in_the_environment_is_still_not_sent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Opt-in means a caller passes it; the environment is never a caller."""
    monkeypatch.setenv("GITHUB_TOKEN", "ghp_ambient")
    monkeypatch.setenv("GH_TOKEN", "gho_ambient")
    seen = serve(github([GREEN_164], GREEN_JOBS), monkeypatch)

    await ci_status.branch_verdict(REMOTE, ["ci.yml"])

    for request in seen:
        assert "authorization" not in request.headers


def test_the_workflow_set_is_discovered_from_the_checkout(tmp_path: Path) -> None:
    """Discovered, so a fourth workflow is gated the day it is added."""
    directory = tmp_path / ".github" / "workflows"
    directory.mkdir(parents=True)
    for name in ("nightly.yml", "ci.yml", "security.yaml", "notes.md"):
        (directory / name).write_text("on: push\n", encoding="utf-8")

    assert ci_status.workflow_files(tmp_path) == ("ci.yml", "nightly.yml", "security.yaml")


def test_a_checkout_without_a_workflow_directory_discovers_nothing(tmp_path: Path) -> None:
    """Which the caller must treat as a failure, not as nothing to do."""
    assert ci_status.workflow_files(tmp_path) == ()


def test_this_repository_has_the_three_workflows_the_ledger_names() -> None:
    """Pins the premise of the CI step: the gate is asked about every workflow that exists."""
    assert set(ci_status.workflow_files(REPO_ROOT)) >= {"ci.yml", "nightly.yml", "security.yml"}


QUALITY_163 = job(
    "quality",
    "failure",
    ("Set up job", "success"),
    ("Install", "success"),
    ("Lint", "success"),
    ("Format", "success"),
    ("Typecheck", "failure"),
    ("Docs match the repository", "skipped"),
    ("API matches its reference", "skipped"),
    ("Blockers view matches the ledger", "skipped"),
    ("Denials still cite open work", "skipped"),
    ("Denials match the tree, not just the ledger", "skipped"),
    ("Task ids cited in code resolve", "skipped"),
    ("Typed error taxonomy holds", "skipped"),
    ("Only permitted comments exist", "skipped"),
    ("Version is coherent across the tree", "skipped"),
    ("Unit tests", "skipped"),
    ("Coverage gate (core, adapters, workers)", "skipped"),
    ("Coverage gate (append Chroma's serverless contract suite)", "skipped"),
    ("Doctor smoke test", "skipped"),
    ("Post Run actions/setup-python@v5", "skipped"),
    ("Complete job", "success"),
)


async def test_a_failing_step_hides_the_thirteen_that_never_ran(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Run 42522906097's `quality` job, step for step, as api.github.com returned it.

    One `Typecheck` failure skipped thirteen later steps, nine of them named gates. The
    run is red for one stated reason and silent about thirteen checks that never objected
    to anything, which is why `skipped` must not read here as `fine`.
    """
    serve(github([RED_163], [QUALITY_163]), monkeypatch)

    verdict = await ci_status.branch_verdict(REMOTE, ["ci.yml"])

    assert verdict.green is False
    run = verdict.workflows[0].run
    assert run is not None
    assert run.failing_jobs[0].failing_steps == ("Typecheck",)
    assert len(run.unrun_steps) == 13
    assert "quality/Denials match the tree, not just the ledger" in run.unrun_steps
    assert "13 later step(s) never ran" in verdict.workflows[0].reason


async def test_runner_bookkeeping_is_not_counted_as_a_check_that_never_ran(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`Post Run ...` and `Complete job` are the runner's, not a gate somebody wrote."""
    serve(github([RED_163], [QUALITY_163]), monkeypatch)

    verdict = await ci_status.branch_verdict(REMOTE, ["ci.yml"])

    run = verdict.workflows[0].run
    assert run is not None
    assert not any(step.startswith("quality/Post ") for step in run.unrun_steps)
    assert "quality/Set up job" not in run.unrun_steps


async def test_a_green_job_reports_no_unrun_steps(monkeypatch: pytest.MonkeyPatch) -> None:
    """A step skipped by its own `if:` in a job that passed is a decision, not a gap."""
    conditional = job("quality", "success", ("Unit tests", "success"), ("Nightly only", "skipped"))
    serve(github([GREEN_164], [conditional]), monkeypatch)

    verdict = await ci_status.branch_verdict(REMOTE, ["ci.yml"])

    assert verdict.green is True
    run = verdict.workflows[0].run
    assert run is not None
    assert run.unrun_steps == ()


async def test_a_run_whose_every_job_skipped_is_not_green(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """GitHub concludes such a run `success`; a workflow that ran nothing found nothing."""
    serve(github([GREEN_164], [job("quality", "skipped"), job("windows", "skipped")]), monkeypatch)

    verdict = await ci_status.branch_verdict(REMOTE, ["ci.yml"])

    assert verdict.workflows[0].run is not None
    assert verdict.workflows[0].run.conclusion == "success"
    assert verdict.green is False


async def test_a_run_with_no_jobs_at_all_is_not_green(monkeypatch: pytest.MonkeyPatch) -> None:
    serve(github([GREEN_164], []), monkeypatch)

    verdict = await ci_status.branch_verdict(REMOTE, ["ci.yml"])

    assert verdict.green is False


async def test_the_unrun_steps_reach_the_machine_readable_report(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    serve(github([RED_163], [QUALITY_163]), monkeypatch)

    payload = (await ci_status.branch_verdict(REMOTE, ["ci.yml"])).as_dict()

    assert len(payload["workflows"][0]["unrun_steps"]) == 13
    assert payload["workflows"][0]["failing_jobs"][0]["steps"] == ["Typecheck"]
    assert len(payload["workflows"][0]["failing_jobs"][0]["skipped_steps"]) == 13


async def test_the_report_lines_name_every_check_that_never_ran(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    serve(github([RED_163], [QUALITY_163]), monkeypatch)

    lines = ci_status.format_branch_verdict(await ci_status.branch_verdict(REMOTE, ["ci.yml"]))

    assert sum(1 for line in lines if "never ran:" in line) == 13
    assert any("quality/Typecheck" in line for line in lines)
