"""Test-suite-wide configuration.

Three concerns: Hypothesis's per-example deadline, the provenance header that says which
tree a run's counts describe, and the process-wide cost governors, which are reset around
every test so a rolling spend window cannot leak between them.
"""

from __future__ import annotations

import subprocess
from collections.abc import Iterator
from pathlib import Path
from typing import Final

import pytest
from hypothesis import HealthCheck, settings

import fasterrag
from fasterrag.services.budget import reset_shared_governors

REPO_ROOT: Final = Path(__file__).resolve().parent.parent

_GIT_TIMEOUT_SECONDS: Final = 10

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
