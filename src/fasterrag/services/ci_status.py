"""Read the latest CI verdict for a branch from the GitHub Actions API.

CI on ``main`` was red for eight days and nobody noticed (``docs/todo.md`` TASK-0261), then
again from 2026-11-08 to 2026-11-11 (TASK-0313): the pre-commit hook passes locally, so
every commit looked green to the person making it, and nothing read the Actions API. This
module is the thing that reads it. ``fasterrag ci-status`` is its consumer, and it is kept
free of the CLI's types so anything else — a release check, an admin endpoint — can ask the
same question and get the same typed answer.

``branch_verdict`` is the gate-shaped entry point: it asks every workflow and is green only
when every one of them has a completed run that concluded ``success``. It is deliberately
**not** filtered by trigger. The whole defect this module exists for is that the *push* run
a human looks at can be green while a *scheduled* run of the same commit is red, so reading
only the visible trigger reproduces the bug instead of catching it.

Authentication is opt-in and explicit, never ambient. The client builds its own headers,
passes no ``auth``, and sets ``trust_env=False`` so a ``.netrc`` entry cannot smuggle a
credential in; nothing here reads ``GITHUB_TOKEN`` or ``GH_TOKEN`` on its own. A caller that
*wants* to authenticate — CI, where the anonymous budget of 60 requests an hour is shared
with every other job on the runner's address — passes ``token=`` and says so. Nothing logs a
URL either: the origin remote may embed a credential, so only the owner and repository
parsed out of it survive.

Every failure is a ``ProvisioningError`` carrying a fix hint — no network, a rate limit
(``RATE_LIMITED``, with the reset time in the hint), a missing repository or workflow
(``NOT_FOUND``), a non-JSON body, a payload missing a field this module reads. A caller
never sees a raw ``httpx`` or ``OSError``, so a consumer can treat "could not ask" as one
case and never mistake it for a verdict.
"""

from __future__ import annotations

import asyncio
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Final

import httpx

from fasterrag import __version__
from fasterrag.errors import ErrorCode, ProvisioningError
from fasterrag.observability.logging import get_logger

__all__ = [
    "DEFAULT_API_BASE",
    "DEFAULT_BRANCH",
    "DEFAULT_WORKFLOW_DIR",
    "DEFAULT_WORKFLOW_FILE",
    "BranchVerdict",
    "JobSummary",
    "Remote",
    "RunSummary",
    "WorkflowVerdict",
    "branch_verdict",
    "format_branch_verdict",
    "latest_run",
    "origin_remote",
    "parse_remote",
    "workflow_files",
]

DEFAULT_API_BASE: Final = "https://api.github.com"
DEFAULT_BRANCH: Final = "main"
DEFAULT_WORKFLOW_FILE: Final = "ci.yml"
DEFAULT_WORKFLOW_DIR: Final = Path(".github/workflows")

_REQUEST_TIMEOUT_SECONDS: Final = 15.0
_GIT_TIMEOUT_SECONDS: Final = 10.0
_ANONYMOUS_LIMIT: Final = "the anonymous limit (60 requests an hour per address)"
_PAYLOAD_FIX: Final = "GitHub may have changed the payload; report this with the fasterRag version."
_WORKFLOW_SUFFIXES: Final[frozenset[str]] = frozenset({".yml", ".yaml"})

# CRITICAL: exactly these headers unless a caller passes an explicit token. No Authorization
# is ever derived from the environment, and the client is built with trust_env=False so
# httpx cannot add one from a .netrc file either.
_HEADERS: Final[dict[str, str]] = {
    "Accept": "application/vnd.github+json",
    "X-GitHub-Api-Version": "2022-11-28",
    "User-Agent": f"fasterrag/{__version__} (ci-status)",
}

_REMOTE_FORMS: Final[tuple[re.Pattern[str], ...]] = (
    re.compile(r"^(?:ssh://)?git@github\.com[:/](?P<owner>[^/]+)/(?P<repo>[^/]+?)(?:\.git)?/?$"),
    re.compile(r"^https?://(?:[^@/]+@)?github\.com/(?P<owner>[^/]+)/(?P<repo>[^/]+?)(?:\.git)?/?$"),
)

_HARMLESS_CONCLUSIONS: Final[frozenset[str]] = frozenset({"success", "skipped", "neutral"})
_STEP_FAILURES: Final[frozenset[str]] = frozenset({"failure", "timed_out"})
_RUNNER_STEPS: Final[frozenset[str]] = frozenset({"Set up job", "Complete job"})


def _is_runner_step(name: object) -> bool:
    """Return whether a step is runner bookkeeping rather than a check somebody wrote.

    ``Post Run …`` cleanups skip whenever the job stopped early, so counting them would
    pad the number this gate reports with steps nobody was relying on.
    """
    text = str(name or "")
    return text in _RUNNER_STEPS or text.startswith("Post ")


_logger = get_logger(__name__)


@dataclass(frozen=True, slots=True)
class Remote:
    """A GitHub repository, as parsed out of a remote URL."""

    owner: str
    repo: str

    @property
    def slug(self) -> str:
        """Return ``owner/repo``."""
        return f"{self.owner}/{self.repo}"


@dataclass(frozen=True, slots=True)
class JobSummary:
    """One job of a run, the steps that failed inside it, and the steps that never ran."""

    name: str
    conclusion: str
    failing_steps: tuple[str, ...] = ()
    skipped_steps: tuple[str, ...] = ()

    @property
    def failed(self) -> bool:
        """Return whether this job is a reason the run is not green."""
        return self.conclusion not in _HARMLESS_CONCLUSIONS

    @property
    def unrun_steps(self) -> tuple[str, ...]:
        """Return the steps this job skipped after it had already gone wrong."""
        return self.skipped_steps if self.failed else ()


@dataclass(frozen=True, slots=True)
class RunSummary:
    """The latest completed run of one workflow on one branch."""

    workflow_file: str
    branch: str
    run_number: int
    sha: str
    conclusion: str
    url: str
    jobs: tuple[JobSummary, ...] = ()

    @property
    def passed(self) -> bool:
        """Return whether the run concluded green and at least one job actually ran."""
        return self.conclusion == "success" and any(
            job.conclusion == "success" for job in self.jobs
        )

    @property
    def failing_jobs(self) -> tuple[JobSummary, ...]:
        """Return the jobs whose conclusion is not green."""
        return tuple(job for job in self.jobs if job.failed)

    @property
    def unrun_steps(self) -> tuple[str, ...]:
        """Return every step a failing job skipped, qualified by its job name."""
        return tuple(f"{job.name}/{step}" for job in self.failing_jobs for step in job.unrun_steps)

    @property
    def short_sha(self) -> str:
        """Return the first eight characters of the commit, as the ledger writes them."""
        return self.sha[:8]


def parse_remote(url: str) -> Remote | None:
    """Return the repository a remote URL points at, or ``None`` for a non-GitHub remote."""
    for form in _REMOTE_FORMS:
        match = form.match(url.strip())
        if match is not None:
            return Remote(owner=match.group("owner"), repo=match.group("repo"))
    return None


async def origin_remote(cwd: Path | None = None) -> Remote | None:
    """Return the GitHub repository ``origin`` points at, or ``None`` when there is none."""
    try:
        process = await asyncio.create_subprocess_exec(
            "git",
            "remote",
            "get-url",
            "origin",
            cwd=cwd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
    except (FileNotFoundError, OSError) as exc:
        _logger.debug("git is not available", extra={"reason": type(exc).__name__})
        return None

    try:
        stdout, _ = await asyncio.wait_for(process.communicate(), timeout=_GIT_TIMEOUT_SECONDS)
    except TimeoutError:
        process.kill()
        await process.wait()
        _logger.debug("git remote get-url did not finish", extra={"timeout": _GIT_TIMEOUT_SECONDS})
        return None

    if process.returncode != 0:
        _logger.debug("no origin remote", extra={"returncode": process.returncode})
        return None

    remote = parse_remote(stdout.decode(errors="replace"))
    if remote is None:
        _logger.debug("origin is not a github.com remote")
    return remote


def _object(value: Any, *, what: str) -> Mapping[str, Any]:
    """Return ``value`` as a JSON object, or raise the typed error naming what it should be."""
    if not isinstance(value, Mapping):
        raise ProvisioningError(
            f"{what} from the Actions API is not a JSON object", fix=_PAYLOAD_FIX
        )
    return value


def _list(value: Any, *, what: str) -> Sequence[Any]:
    """Return ``value`` as a JSON array, or raise the typed error naming what it should be."""
    if not isinstance(value, Sequence) or isinstance(value, str):
        raise ProvisioningError(
            f"{what} from the Actions API is not a JSON array", fix=_PAYLOAD_FIX
        )
    return value


def _field(payload: Mapping[str, Any], key: str, *, where: str) -> Any:
    """Return ``payload[key]``, or raise the typed error naming what was missing."""
    if key not in payload:
        raise ProvisioningError(
            f"{where} from the Actions API has no {key!r} field", fix=_PAYLOAD_FIX
        )
    return payload[key]


def _rate_limit_fix(reset: str | None) -> str:
    """Return the fix hint for a spent rate limit, naming the reset time when known."""
    if reset is None or not reset.isdigit():
        return f"Wait for {_ANONYMOUS_LIMIT} to reset, then run the check again."
    when = datetime.fromtimestamp(int(reset), tz=UTC)
    return f"Wait until {when:%H:%M:%S} UTC, when {_ANONYMOUS_LIMIT} resets."


async def _get_json(client: httpx.AsyncClient, path: str, *, params: Mapping[str, Any]) -> Any:
    """Fetch one endpoint and decode it, mapping every failure onto the taxonomy."""
    try:
        response = await client.get(path, params=params)
    except httpx.HTTPError as exc:
        raise ProvisioningError(
            f"could not reach {client.base_url.host}: {type(exc).__name__}",
            fix=f"Check the network connection; the check needs {client.base_url.host}.",
        ) from exc

    status = response.status_code
    rate_limited = status == 429 or (
        status == 403 and response.headers.get("X-RateLimit-Remaining") == "0"
    )
    if rate_limited:
        raise ProvisioningError(
            f"the anonymous GitHub API rate limit is exhausted (HTTP {status})",
            fix=_rate_limit_fix(response.headers.get("X-RateLimit-Reset")),
            code=ErrorCode.RATE_LIMITED,
        )
    if status == 404:
        raise ProvisioningError(
            f"GitHub has no public resource at {path} (HTTP 404)",
            fix=(
                "Check that the repository is public and the workflow file name matches "
                ".github/workflows; a private repository answers 404 to an anonymous call."
            ),
            code=ErrorCode.NOT_FOUND,
        )
    if status != 200:
        raise ProvisioningError(
            f"GitHub answered HTTP {status} for {path}",
            fix="Check https://www.githubstatus.com and try again.",
        )

    try:
        return response.json()
    except ValueError as exc:
        raise ProvisioningError(
            f"GitHub answered {path} with a body that is not JSON",
            fix="A proxy or captive portal may be intercepting the request; check the network.",
        ) from exc


def _job_summary(job: Mapping[str, Any]) -> JobSummary:
    """Reduce one job payload to its name, conclusion, and failing step names."""
    steps = [step for step in job.get("steps") or () if isinstance(step, Mapping)]
    return JobSummary(
        name=str(_field(job, "name", where="a job")),
        conclusion=str(job.get("conclusion") or "unknown"),
        failing_steps=tuple(
            str(step.get("name") or "unnamed step")
            for step in steps
            if step.get("conclusion") in _STEP_FAILURES
        ),
        skipped_steps=tuple(
            str(step.get("name") or "unnamed step")
            for step in steps
            if step.get("conclusion") == "skipped" and not _is_runner_step(step.get("name"))
        ),
    )


def _run_summary(
    run: Mapping[str, Any], jobs: Sequence[Any], *, workflow_file: str, branch: str
) -> RunSummary:
    """Reduce a run payload and its job listing to the summary."""
    try:
        run_number = int(_field(run, "run_number", where="the run"))
    except (TypeError, ValueError) as exc:
        raise ProvisioningError(
            "the run from the Actions API has a non-numeric 'run_number'", fix=_PAYLOAD_FIX
        ) from exc

    return RunSummary(
        workflow_file=workflow_file,
        branch=branch,
        run_number=run_number,
        sha=str(_field(run, "head_sha", where="the run")),
        conclusion=str(run.get("conclusion") or "unknown"),
        url=str(_field(run, "html_url", where="the run")),
        jobs=tuple(_job_summary(_object(job, what="a job")) for job in jobs),
    )


async def latest_run(
    remote: Remote,
    *,
    branch: str = DEFAULT_BRANCH,
    workflow_file: str = DEFAULT_WORKFLOW_FILE,
    api_base: str = DEFAULT_API_BASE,
    timeout: float = _REQUEST_TIMEOUT_SECONDS,
    token: str | None = None,
) -> RunSummary | None:
    """Return the latest *completed* run of a workflow on a branch, with its jobs."""
    repository = f"/repos/{remote.owner}/{remote.repo}"
    # CRITICAL: no auth argument and trust_env=False. The only Authorization this client can
    # send is one a caller passed in by hand; it must never pick a token up from the
    # environment or a .netrc (docs/todo.md TASK-0284, TASK-0313).
    headers = dict(_HEADERS) if token is None else {**_HEADERS, "Authorization": f"Bearer {token}"}
    async with httpx.AsyncClient(
        base_url=api_base, headers=headers, timeout=timeout, trust_env=False
    ) as client:
        listing = _object(
            await _get_json(
                client,
                f"{repository}/actions/workflows/{workflow_file}/runs",
                params={"branch": branch, "status": "completed", "per_page": 1},
            ),
            what="the run listing",
        )
        runs = _list(
            _field(listing, "workflow_runs", where="the run listing"), what="workflow_runs"
        )
        if not runs:
            return None
        run = _object(runs[0], what="the run")

        run_id = _field(run, "id", where="the run")
        jobs_payload = _object(
            await _get_json(
                client, f"{repository}/actions/runs/{run_id}/jobs", params={"per_page": 100}
            ),
            what="the job listing",
        )
        jobs = _list(_field(jobs_payload, "jobs", where="the job listing"), what="jobs")

    return _run_summary(run, jobs, workflow_file=workflow_file, branch=branch)


@dataclass(frozen=True, slots=True)
class WorkflowVerdict:
    """One workflow's last word on a branch: a run, nothing, or why it could not be asked."""

    workflow_file: str
    run: RunSummary | None = None
    unreachable: str | None = None

    @property
    def green(self) -> bool:
        """Return whether this workflow is known to have concluded green."""
        return self.run is not None and self.run.passed

    @property
    def reason(self) -> str:
        """Return one line saying what this workflow's state is and why."""
        if self.unreachable is not None:
            return f"{self.workflow_file}: could not be read -- {self.unreachable}"
        if self.run is None:
            return f"{self.workflow_file}: no completed run on this branch"
        jobs = ", ".join(
            f"{job.name}/{step}" for job in self.run.failing_jobs for step in job.failing_steps
        ) or ", ".join(job.name for job in self.run.failing_jobs)
        detail = f"{self.run.conclusion} (#{self.run.run_number} {self.run.short_sha})"
        unrun = len(self.run.unrun_steps)
        never = f" -- {unrun} later step(s) never ran" if unrun else ""
        return f"{self.workflow_file}: {detail}{f' -- {jobs}' if jobs else ''}{never}"

    def as_dict(self) -> dict[str, Any]:
        """Return the machine-readable form used by ``--json``."""
        return {
            "workflow": self.workflow_file,
            "green": self.green,
            "conclusion": None if self.run is None else self.run.conclusion,
            "run_number": None if self.run is None else self.run.run_number,
            "sha": None if self.run is None else self.run.sha,
            "url": None if self.run is None else self.run.url,
            "failing_jobs": (
                []
                if self.run is None
                else [
                    {
                        "name": job.name,
                        "conclusion": job.conclusion,
                        "steps": list(job.failing_steps),
                        "skipped_steps": list(job.unrun_steps),
                    }
                    for job in self.run.failing_jobs
                ]
            ),
            "unrun_steps": [] if self.run is None else list(self.run.unrun_steps),
            "unreachable": self.unreachable,
            "reason": self.reason,
        }


@dataclass(frozen=True, slots=True)
class BranchVerdict:
    """Every workflow's last conclusion on one branch, and whether that is green."""

    slug: str
    branch: str
    workflows: tuple[WorkflowVerdict, ...] = ()

    @property
    def green(self) -> bool:
        """Return whether every workflow asked about concluded green."""
        return bool(self.workflows) and all(workflow.green for workflow in self.workflows)

    @property
    def red(self) -> tuple[WorkflowVerdict, ...]:
        """Return the workflows that are not green."""
        return tuple(workflow for workflow in self.workflows if not workflow.green)

    def as_dict(self) -> dict[str, Any]:
        """Return the machine-readable report."""
        return {
            "repository": self.slug,
            "branch": self.branch,
            "green": self.green,
            "workflows": [workflow.as_dict() for workflow in self.workflows],
        }


def workflow_files(
    root: Path = Path(), *, workflow_dir: Path = DEFAULT_WORKFLOW_DIR
) -> tuple[str, ...]:
    """Return the workflow file names in a checkout, sorted.

    Discovered rather than listed, so a fourth workflow is gated the day it is added; a
    hard-coded triple is a second copy of ``.github/workflows`` that rots in silence.
    """
    directory = root / workflow_dir
    if not directory.is_dir():
        return ()
    return tuple(
        sorted(
            entry.name
            for entry in directory.iterdir()
            if entry.is_file() and entry.suffix in _WORKFLOW_SUFFIXES
        )
    )


async def branch_verdict(
    remote: Remote,
    workflows: Sequence[str],
    *,
    branch: str = DEFAULT_BRANCH,
    api_base: str = DEFAULT_API_BASE,
    timeout: float = _REQUEST_TIMEOUT_SECONDS,
    token: str | None = None,
) -> BranchVerdict:
    """Return every named workflow's last conclusion on a branch."""
    verdicts: list[WorkflowVerdict] = []
    for workflow in workflows:
        try:
            run = await latest_run(
                remote,
                branch=branch,
                workflow_file=workflow,
                api_base=api_base,
                timeout=timeout,
                token=token,
            )
        except ProvisioningError as exc:
            _logger.warning(
                "could not read a workflow's status",
                extra={"workflow": workflow, "code": exc.code.value, "detail": exc.detail},
            )
            verdicts.append(WorkflowVerdict(workflow_file=workflow, unreachable=exc.detail))
            continue
        verdicts.append(WorkflowVerdict(workflow_file=workflow, run=run))
    return BranchVerdict(slug=remote.slug, branch=branch, workflows=tuple(verdicts))


def format_branch_verdict(verdict: BranchVerdict) -> list[str]:
    """Return the human-readable report lines for a branch verdict."""
    lines = [f"{verdict.slug} @ {verdict.branch}"]
    if not verdict.workflows:
        lines.append("  no workflows were checked")
        return lines
    for workflow in verdict.workflows:
        marker = "ok  " if workflow.green else "RED "
        lines.append(f"  {marker}{workflow.reason}")
        if workflow.run is None or workflow.green:
            continue
        lines.append(f"      {workflow.run.url}")
        lines.extend(f"      never ran: {step}" for step in workflow.run.unrun_steps)
    return lines
