"""Test-suite-wide configuration.

Four concerns: Hypothesis's per-example deadline, the provenance header that says which
tree a run's counts describe, the process-wide cost governors reset around every test so a
rolling spend window cannot leak between them, and the ``must_run`` execution report that
``scripts/check_must_run.py`` reads across CI jobs.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from collections.abc import Iterator
from pathlib import Path
from typing import Final

import pytest
from hypothesis import HealthCheck, settings

import fasterrag
from fasterrag.core.breaker import reset_shared_breakers
from fasterrag.core.rerank.cross_encoder import reset_shared_cross_encoders
from fasterrag.services.budget import reset_shared_governors

REPO_ROOT: Final = Path(__file__).resolve().parent.parent

_GIT_TIMEOUT_SECONDS: Final = 10

# CRITICAL: git exports these to a hook, and the pre-commit hook runs this suite. In a linked
# worktree GIT_DIR names the real repository, so a test that ran `git init` in a temporary
# directory acted on this repository instead: five tests in test_ci_status.py failed in every
# worktree, and `git init` set core.bare in the shared config, which broke the main checkout
# until it was reset by hand. They are removed for the whole test process, at import, so that
# every git call a test makes resolves from its own working directory.
GIT_HOOK_VARIABLES: Final = (
    "GIT_DIR",
    "GIT_WORK_TREE",
    "GIT_INDEX_FILE",
    "GIT_COMMON_DIR",
    "GIT_PREFIX",
    "GIT_OBJECT_DIRECTORY",
    "GIT_ALTERNATE_OBJECT_DIRECTORIES",
)
for _name in GIT_HOOK_VARIABLES:
    os.environ.pop(_name, None)

# CRITICAL: the deadline is disabled, not merely raised. Hypothesis fails an example that
# takes longer than 200 ms by default, which makes every property test in this repo
# timing-dependent — and they assert *behaviour*, never latency. Under load (a machine
# running the Docker fleet, or CI on a shared runner) that produced intermittent failures in
# a different property test on each run: tenant isolation once, name scoping the next, both
# passing in isolation moments later.
#
# A correctness suite that fails for reasons unrelated to correctness is worse than a slower
# one. It teaches whoever sees red to re-run rather than investigate, which is exactly the
# habit that lets a real isolation failure through. Performance belongs in the benchmark
# ledger (docs/benchmarks.md), measured on isolated hardware, not asserted as a side effect
# of a property test.
settings.register_profile(
    "fasterrag",
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow],
)
settings.load_profile("fasterrag")


def _git(*args: str) -> str:
    """Return a git command's output, or a reason string when git cannot answer."""
    try:
        completed = subprocess.run(
            ["git", *args],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            timeout=_GIT_TIMEOUT_SECONDS,
            check=True,
        )
    except FileNotFoundError:
        return "unavailable (git is not on PATH)"
    except subprocess.TimeoutExpired:
        return f"unavailable (git did not answer within {_GIT_TIMEOUT_SECONDS}s)"
    except subprocess.CalledProcessError as exc:
        return f"unavailable (git exited {exc.returncode})"
    except OSError as exc:
        return f"unavailable ({exc.strerror or exc})"
    return completed.stdout.strip()


def working_tree_state() -> str:
    """Describe whether the checkout under test carries uncommitted changes."""
    status = _git("status", "--porcelain")
    if status.startswith("unavailable"):
        return status
    changed = [line for line in status.splitlines() if line.strip()]
    if not changed:
        return "clean"
    return f"{len(changed)} uncommitted path(s) - A NUMBER MEASURED HERE IS NOT REPRODUCIBLE"


def provenance_lines() -> list[str]:
    """Return the facts that make a run's counts attributable to a specific tree."""
    return [
        f"checkout: {REPO_ROOT}",
        f"commit: {_git('rev-parse', 'HEAD')}",
        f"working tree: {working_tree_state()}",
        f"fasterrag imported from: {Path(fasterrag.__file__).resolve().parent}",
    ]


def provenance_summary() -> str:
    """Return the one-line form printed beside the counts, which ``-q`` still shows."""
    return (
        f"measured: {_git('rev-parse', '--short', 'HEAD')} | "
        f"working tree: {working_tree_state()} | {REPO_ROOT}"
    )


# CRITICAL: this header is the record that TASK-0322 cost two days to reconstruct without.
# A pass/fail count is evidence about one tree at one commit; printed without them it is an
# anecdote, and a dirty tree silently invalidates it. Keep it in the header, where every
# run carries it, rather than in a doc nobody re-reads.
def pytest_report_header() -> list[str]:
    """Name the tree under test so a recorded count can never lose its provenance."""
    return provenance_lines()


# CRITICAL: the header above is suppressed by `-q`, which is the flag every measurement run
# in this repo actually uses. The summary line is therefore not a duplicate: it is the only
# copy a `-q` run prints, and it lands beside the counts someone is about to paste.
def pytest_terminal_summary(terminalreporter: pytest.TerminalReporter) -> None:
    """Repeat the provenance beside the counts, where ``-q`` cannot hide it."""
    terminalreporter.write_line(provenance_summary())


# CRITICAL: `services.budget.shared_governor` is a process-wide cache, and a rolling spend
# window that outlives the test which filled it is cross-test state leakage — the exact
# class of defect TASK-0322 spent a session ruling out of this suite. Resetting it around
# every test, autouse, is what keeps the singleton an implementation detail: before, so a
# test never inherits another's spend, and after, so a test that deliberately exhausts a
# budget cannot hand a refusal to whatever runs next.
@pytest.fixture(autouse=True)
def _unspent_governors() -> Iterator[None]:
    """Give every test an empty shared-governor cache, and leave one behind it."""
    reset_shared_governors()
    yield
    reset_shared_governors()


# CRITICAL: the shared circuit breakers and the shared cross-encoder cache are the same kind
# of process-wide state for the same reason, and leak across tests the same way: a test that
# opens a circuit would hand `CIRCUIT_OPEN` to whatever runs next, and a test that installs a
# fake model would hand it to every later reranker of that name.
@pytest.fixture(autouse=True)
def _closed_circuits_and_no_loaded_models() -> Iterator[None]:
    """Give every test closed shared breakers and an empty reranker cache."""
    reset_shared_breakers()
    reset_shared_cross_encoders()
    yield
    reset_shared_breakers()
    reset_shared_cross_encoders()


MUST_RUN_MARKER: Final = "must_run"
MUST_RUN_REPORT_VAR: Final = "FASTERRAG_MUST_RUN_REPORT"
MUST_RUN_REPORT_VERSION: Final = 1
MUST_RUN_RAN: Final = "ran"
MUST_RUN_SKIPPED: Final = "skipped"

_must_run_collected: set[str] = set()
_must_run_outcomes: dict[str, str] = {}

# CRITICAL: this plugin is the only evidence scripts/check_must_run.py has that a test named
# as must-run actually executed, and it is deliberately inert unless FASTERRAG_MUST_RUN_REPORT
# names a destination. A lane that stops setting the variable stops producing a report, which
# the gate reads as "ran nowhere" and fails on — never as a pass. Do not make the absent
# variable write an empty document somewhere the gate might find it.


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    """Record which collected tests carry the marker, before anything can skip them."""
    if not os.environ.get(MUST_RUN_REPORT_VAR):
        return
    for item in items:
        if item.get_closest_marker(MUST_RUN_MARKER) is not None:
            _must_run_collected.add(item.nodeid)


def pytest_runtest_logreport(report: pytest.TestReport) -> None:
    """Record whether a marked test executed; a skip raised in the body overwrites a run.

    A failure counts as having run. This gate judges execution, not verdicts, and a
    failing test fails its own job already.
    """
    if report.when == "teardown" or report.nodeid not in _must_run_collected:
        return
    _must_run_outcomes[report.nodeid] = MUST_RUN_SKIPPED if report.skipped else MUST_RUN_RAN


def pytest_sessionfinish(session: pytest.Session, exitstatus: int) -> None:
    """Write this lane's must-run report, saying whether it ran tests or only collected."""
    destination = os.environ.get(MUST_RUN_REPORT_VAR)
    if not destination:
        return
    path = Path(destination)
    path.parent.mkdir(parents=True, exist_ok=True)
    document = {
        "report_version": MUST_RUN_REPORT_VERSION,
        "mode": "collect" if bool(session.config.getoption("--collect-only")) else "run",
        "label": path.stem,
        "platform": sys.platform,
        "collected": sorted(_must_run_collected),
        "outcomes": dict(sorted(_must_run_outcomes.items())),
    }
    path.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8")
