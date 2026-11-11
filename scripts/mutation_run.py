"""Mutation-test the modules whose correctness this repository argues about.

Mutation checking has been this project's habit rather than its infrastructure: several
slices (TASK-0193, TASK-0208, TASK-0211, TASK-0229, TASK-0232) record a maintainer breaking
a line by hand and confirming exactly one test went red. A habit protects whatever somebody
remembered to break. This automates it for a named target set, which is the mutation layer
of TASK-0243 — the load and soak layers still need reference hardware and remain open.

**Why a purpose-built runner rather than mutmut or cosmic-ray.** Both were evaluated here on
2026-11-04 against Python 3.14.7 on Windows, which is the machine this repository is
developed on:

* ``mutmut`` 3.8.0 installs and then refuses to run at all — it prints "To run mutmut on
  Windows, please use the WSL" and exits. A nightly survivor nobody can reproduce locally is
  a report, not a gate.
* ``cosmic-ray`` 8.7.0 installs cleanly on 3.14, but it parses with ``parso`` 0.8.7, which
  reports a false ``SyntaxError`` on the parenthesized ``with`` statements in
  ``src/fasterrag/services/archive.py`` — valid since Python 3.10 and accepted by
  ``ast.parse`` here. It also carries one global ``test-command`` per session, so running
  only the tests that cover a module means one session, one SQLite database and one report
  per target, stitched back together by a script of our own.

Both facts point the same way: this repository already has hand-rolled AST tooling
(``scripts/trim_docstrings.py``, with its before/after structure comparison), the mutation
budget is only affordable if each mutant runs *its own module's* tests, and the output has
to be a survivor list rather than a score. ``ast`` parses exactly what the interpreter
parses, so the parso class of defect cannot occur.

**A surviving mutant is a claim, not a verdict.** It says some line can be changed without
any test noticing. That is usually a missing test and sometimes an equivalent mutant, so
this prints every survivor with its file, line, operator and the exact text substitution,
and never prints a bare score. Exit codes: ``0`` every mutant killed, ``1`` survivors
found, ``2`` the run could not be performed. ``2`` is the vacuity case this repository
treats as a bug — a baseline suite that was already red kills every mutant for free, so it
is never allowed to look like a pass.

**Nothing under ``src/`` is ever written.** The tree is copied once per run and the copy is
mutated; pytest is pointed at it with ``-o pythonpath=`` (which empties the ini setting that
would otherwise put the real ``src`` first) plus ``PYTHONPATH``.

**The tests are snapshotted too, and no bytecode crosses into the run** (TASK-0303). The
runner used to copy ``src/`` and then read the tests live from the working tree, so a test
file saved mid-run silently changed the verdict of every mutant executed afterwards — a
``core/breaker.py`` mutant was flipped from survived to killed with ``src/`` byte-identical,
purely by editing its test between two runs. ``tests/`` and ``pyproject.toml`` are now copied
into the same workspace and pytest is pointed at the copies. The whole tree is copied rather
than only the paths a target names, because a test that reaches past itself —
``test_comparability.py`` reads ``tests/eval/datasets`` through ``__file__`` — must find the
snapshot's copy rather than nothing. ``__pycache__`` is excluded from both copies and the
child runs with ``PYTHONDONTWRITEBYTECODE``, because CPython validates a cached ``.pyc`` on
``(int source mtime seconds, source size)`` alone: a length-preserving mutant such as
``'0' -> '1'`` whose mtime second collides with the cached one is never executed, and the
runner would then report a verdict for a mutant it never ran — the vacuity class this script
exists to detect. The price is roughly a second of copying per run and a recompile of the
mutated package per mutant; the breaker target's verdict was unchanged across the fix (34
mutants, 32 killed, the same two survivors), and no wall-clock comparison taken on a machine
running ten agents at once is worth quoting.

**A pathological mutant is a verdict, not a dead runner** (TASK-0321). A mutated comparison
can turn a bounded loop into an unbounded one, and ``--timeout`` bounds a mutant's seconds
rather than its resident set: the nightly job was SIGTERMed twice with no output at all, so
nobody could name the mutant that did it. Four things answer that. Every mutant is announced
on a flushed stdout line *before* it runs, so the last line printed before a kill names the
mutant in flight. ``--memory-mb`` caps the child's address space through ``RLIMIT_AS``, which
converts a runaway into a ``MemoryError`` inside its own pytest run — recorded as killed,
campaign continuing — rather than into a dead runner; it is POSIX-only and refused rather
than silently ignored where ``resource`` does not exist. The artifact carries the peak
resident size of the worst child reaped so far, so a run that came close to the ceiling says
so. And ``--json-out`` is rewritten after every target, so a killed run leaves the targets it
finished instead of nothing, with ``complete: false`` saying it is a partial.

**The peak is carried per target as well as per run** (TASK-0363), because the question an
operator has while calibrating ``--memory-mb`` is *which* target approached the cap, and one
cumulative number cannot answer it. ``getrusage(RUSAGE_CHILDREN)`` is a high-water mark over
every child reaped so far and nothing resets it, so each target records the mark as it stood
when that target finished: the series only rises, and the target where it rises is the one
that set it. A target that did not raise the mark has an upper bound and no measurement, and
the reports say which target first reached the figure rather than implying six readings.

**A timeout is counted as a kill and named as a timeout.** It stays inside ``killed`` because
``executed == killed + survived`` is what every consumer of this artifact adds up, and
because the alternative — calling it a survivor — would publish "no test defends this line"
about a line whose test never finished. But a timeout is not evidence that a test caught
anything, so each timed-out mutant is now listed by file, line and substitution exactly as a
survivor is: a run reporting 214 kills of which 4 were timeouts now says which 4.

**The cap is calibrated, and it is not the resident figure** (TASK-0362). ``RLIMIT_AS`` bounds
address space while ``peak_child_rss_bytes`` reports resident size, and for these six targets
the two differ by roughly a factor of three — so a cap read off the 337 MiB the nightly last
reported would have turned every baseline into ``not-run``. Measured 2026-11-11 under WSL
Ubuntu-24.04 (Python 3.12.3, 16 cores) by running each target's real suite under ``ulimit
-v``, which is the same limit this script sets: all six pass at 896 MiB and above, all six
fail below it, and between 640 and 768 MiB the child does not even fail cleanly — a corrupted
``ast`` import, a ``SIGABRT``, a ``SIGINT``. The nightly therefore runs at 2048 MiB, 2.3x that
knee. The figure is an upper bound for the hosted runner rather than a reading from it: glibc
reserves address space per arena and the runner has four cores to this machine's sixteen.

**A child killed by a signal is the same class and is named too** (TASK-0362). ``RLIMIT_AS``
converts a runaway allocation into a ``MemoryError`` inside the child's own pytest run, which
is a real kill; what it cannot convert is a kill arriving from outside the process — a cgroup
OOM kill, which is ``SIGKILL`` to a process that never asked for the memory itself. POSIX
reports that as a negative return code, and a negative return code is not zero, so until now
the one memory pathology the cap cannot catch was also the one it could not report. A
negative code is counted as a kill, for the arithmetic reason above, and listed by signal
name exactly as a timeout is. On Windows an abnormal termination is a large positive status
rather than a negative one, so this separation is POSIX-only and is not claimed elsewhere.
The calibration run above produced one of these without being asked to: at a 768 MiB cap the
``rate-limit`` child aborted with ``SIGABRT`` rather than raising ``MemoryError``, which the
runner would previously have recorded as a mutant some test caught.

**The survivor count is ratcheted against a measured baseline** (TASK-0382). Reporting
survivors as a warning made the job one that cannot fail, which is the class of defect that
let the nightly benchmark measure nothing for 83 days. ``--check-ratchet ARTIFACT`` reads a
written ``--json-out`` document and holds each target to ``CEILING``: more survivors than the
ceiling, or any kill left unproven by the clock or a signal, exits 1. The ceiling is
**per-target**, because a total lets one target rot behind another's improvement, and it is
**counts rather than mutant identities**, because a seeded sample is drawn from the mutants a
file generates and any edit to that file redraws it — an identity ratchet would fire on every
refactor and be deleted within the week.

``CEILING`` was measured, not chosen: four consecutive nightly runs — 42419161973 (2026-11-07,
``0c7f81a``), 42468563494 (11-08, ``d1a0d4a``), 42603353729 (11-09) and 42800995556 (11-10,
both ``6be46f9``) — all at ``--budget 10 --seed 0`` report the identical per-target split
``retrieval 1, chunking 5, identity 2, breaker 1, regression 0, rate-limit 2`` of 60 executed,
with zero timeouts. Each entry records the **sample** it was measured on as well as the count,
and a run whose sample differs in either direction is refused rather than judged: a smaller
budget would satisfy the ceiling for free, and an exhaustive run would fail it for nothing.
Three commits, four nights, no movement, so the ceiling carries no slack: slack is a raise
granted in advance to a number nobody has yet had to justify. The counts even survived a
redraw — ``chunking`` generated 234 mutants on the first night and 238 on the other three, so
its ten were a different ten, and five of them still survived. The ±1 noise the ledger
records (TASK-0330, TASK-0331) belongs to the archive suite's ``tracemalloc`` assertions,
which defend no module in ``DEFAULT_TARGETS`` and so cannot move these counts.

**The gate lowers itself and never raises itself.** A run below its ceiling prints the exact
replacement block to paste, so tightening is mechanical and the gate asks for it every night;
a run above its ceiling prints the diagnosis and no block at all, because a tool that hands
back a loosened ceiling is a rubber stamp. Raising a number is a hand edit in a reviewed diff.
And the gate is vacuity-guarded (trap T12) by reusing ``decide()`` rather than re-deriving it:
a missing or unparseable artifact, a partial one, a target of the ceiling that did not run, a
target the ceiling does not cover, a stored verdict that disagrees with the recomputed one,
and a run that executed nothing all exit 2 instead of reporting a held ratchet.
"""

from __future__ import annotations

import argparse
import ast
import importlib
import json
import os
import random
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from types import ModuleType
from typing import Any, Final

__all__ = [
    "ARITHMETIC_SWAPS",
    "BOOLEAN_SWAPS",
    "CEILING",
    "COMPARISON_SWAPS",
    "DEFAULT_TARGETS",
    "EXIT_CODES",
    "STAGED_ROOTS",
    "Ceiling",
    "Mutation",
    "RunReport",
    "Target",
    "TargetResult",
    "Verdict",
    "apply",
    "child_peak_rss",
    "decide",
    "gate",
    "generate",
    "load_targets",
    "main",
    "markdown",
    "memory_limiter",
    "ratchet",
    "ratchet_block",
    "read_report",
    "render",
    "resource_module",
    "restrict",
    "run",
    "select",
    "source_files",
    "stage_suite",
    "staged_tests",
]

REPOSITORY_ROOT: Final = Path(__file__).resolve().parents[1]

DEFAULT_TIMEOUT_S: Final = 180.0
DEFAULT_SEED: Final = 0

# CRITICAL: no ``__pycache__`` may enter the workspace. A cached ``.pyc`` is validated on
# the source's mtime seconds and byte size only, so one carried in beside a mutated file can
# shadow the mutation entirely and the run reports a verdict for a mutant it never executed.
_IGNORED: Final = shutil.ignore_patterns("__pycache__")

# CRITICAL: every repository directory a staged test may read through ``__file__``. ``tests``
# carries the fixtures and eval datasets a test reaches for sideways; ``docs`` is here
# because ``tests/unit/services/test_archive_format_spec.py`` reads
# ``docs/archive-format.md`` and the whole archive suite reported `not-run` without it
# (TASK-0331). Everything else at the repository root is deliberately absent — a target
# whose tests read `config.yaml`, `.github/` or `src/` under that name must extend this
# tuple, and will see a hard failure rather than a silent wrong verdict if it does not.
STAGED_ROOTS: Final[tuple[str, ...]] = ("tests", "docs")

_MEGABYTE: Final = 1024 * 1024

# CRITICAL: ``ru_maxrss`` is kilobytes on Linux and bytes on macOS. Reporting the raw number
# would misstate a Linux peak by three orders of magnitude, which is the direction that makes
# a memory pathology look harmless.
_RSS_SCALE: Final[dict[str, int]] = {"darwin": 1}
_RSS_SCALE_DEFAULT: Final = 1024

# CRITICAL: SIGKILL is what a cgroup OOM kill arrives as, and it is the one name this
# repository must never fail to print. Windows has no SIGKILL in `signal.Signals`, so a
# POSIX artifact read on a development machine would otherwise describe the same kill by a
# bare number while CI described it by a name.
_POSIX_SIGNALS: Final[dict[int, str]] = {
    6: "SIGABRT",
    9: "SIGKILL",
    11: "SIGSEGV",
    15: "SIGTERM",
}


class Verdict(StrEnum):
    """What a mutation run concluded."""

    CLEAN = "clean"
    SURVIVORS = "survivors"
    NOT_RUN = "not-run"


EXIT_CODES: Final[dict[Verdict, int]] = {
    Verdict.CLEAN: 0,
    Verdict.SURVIVORS: 1,
    Verdict.NOT_RUN: 2,
}


def resource_module() -> ModuleType | None:
    """Import POSIX ``resource``, or report that this platform has none.

    Imported by name rather than at module scope so the rest of the runner keeps
    working on Windows, where the module does not exist at all.
    """
    try:
        return importlib.import_module("resource")
    except ImportError:
        return None


def memory_limiter(megabytes: int | None, limits: ModuleType | None) -> Callable[[], None] | None:
    """Build the ``preexec_fn`` capping a child's address space, or ``None`` for no cap.

    A cap asked for on a platform without ``resource`` is refused here rather than
    dropped, because a limit silently not applied is the pathology it was meant to catch.
    """
    if megabytes is None or megabytes <= 0:
        return None
    if limits is None:
        raise ValueError(
            "--memory-mb needs POSIX resource limits, which this platform does not provide"
        )
    ceiling = megabytes * _MEGABYTE

    def apply_limit() -> None:
        """Run in the forked child, before ``exec``, so the cap binds the pytest process."""
        limits.setrlimit(limits.RLIMIT_AS, (ceiling, ceiling))

    return apply_limit


def child_peak_rss(limits: ModuleType | None, platform: str = sys.platform) -> int | None:
    """Peak resident bytes of the worst child reaped so far, or ``None`` off POSIX.

    ``getrusage`` reports a high-water mark across every reaped child, so the number
    only ever rises and says how close the campaign came to the ceiling.
    """
    if limits is None:
        return None
    usage = limits.getrusage(limits.RUSAGE_CHILDREN)
    return int(usage.ru_maxrss) * _RSS_SCALE.get(platform, _RSS_SCALE_DEFAULT)


@dataclass(frozen=True)
class Target:
    """One mutated source path and the tests that are supposed to defend it."""

    name: str
    source: str
    tests: tuple[str, ...]


# CRITICAL: this set is deliberately small. Mutating all 27k lines produces a run that never
# finishes and a report nobody reads; these are the modules whose behaviour the docs argue
# about — chunk boundaries, retrieval scoring and fusion, the index identity digest, the
# breaker's state machine, the eval regression gate and the rate limiter. Widen it with
# --targets-file rather than by quietly adding paths nobody budgeted for.
DEFAULT_TARGETS: Final[tuple[Target, ...]] = (
    Target(
        name="retrieval",
        source="src/fasterrag/core/retrieval",
        tests=("tests/unit/core/retrieval",),
    ),
    Target(
        name="chunking",
        source="src/fasterrag/core/chunking",
        tests=("tests/unit/core/chunking",),
    ),
    Target(
        name="identity",
        source="src/fasterrag/core/identity.py",
        tests=(
            "tests/unit/core/test_identity.py",
            "tests/unit/core/test_identity_properties.py",
            "tests/unit/services/test_identity_drift.py",
        ),
    ),
    Target(
        name="breaker",
        source="src/fasterrag/core/breaker.py",
        tests=("tests/unit/core/test_breaker.py",),
    ),
    Target(
        name="regression",
        source="src/fasterrag/services/regression.py",
        tests=(
            "tests/unit/services/test_regression.py",
            "tests/unit/services/test_comparability.py",
        ),
    ),
    Target(
        name="rate-limit",
        source="src/fasterrag/core/rate_limit.py",
        tests=("tests/unit/core/test_rate_limit.py",),
    ),
)


@dataclass(frozen=True)
class Ceiling:
    """One target's measured limits: the sample they were measured on, and the two counts.

    ``sample`` is part of the measurement, not decoration: a ceiling drawn from ten
    mutants says nothing about a run of two hundred, in either direction.
    """

    survived: int
    sample: int
    unproven: int = 0


# CRITICAL: raising a number here is the one edit this file's own tooling will never make.
# --check-ratchet prints a tightened block when a target improves and never prints a loosened
# one, so a raise is a hand edit in a reviewed diff — and it means a line this repository used
# to defend is no longer defended. The numbers are the measured split of four consecutive
# nightly runs at --budget 10 --seed 0 (see the module docstring); `unproven` is zero
# everywhere because no night has yet produced a timeout or a signalled child.
CEILING: Final[dict[str, Ceiling]] = {
    "retrieval": Ceiling(survived=1, sample=10),
    "chunking": Ceiling(survived=5, sample=10),
    "identity": Ceiling(survived=2, sample=10),
    "breaker": Ceiling(survived=1, sample=10),
    "regression": Ceiling(survived=0, sample=10),
    "rate-limit": Ceiling(survived=2, sample=10),
}

COMPARISON_SWAPS: Final[dict[type[ast.cmpop], tuple[str, str]]] = {
    ast.Lt: ("<", "<="),
    ast.LtE: ("<=", "<"),
    ast.Gt: (">", ">="),
    ast.GtE: (">=", ">"),
    ast.Eq: ("==", "!="),
    ast.NotEq: ("!=", "=="),
    ast.Is: ("is", "is not"),
    ast.IsNot: ("is not", "is"),
    ast.In: ("in", "not in"),
    ast.NotIn: ("not in", "in"),
}

ARITHMETIC_SWAPS: Final[dict[type[ast.operator], tuple[str, str]]] = {
    ast.Add: ("+", "-"),
    ast.Sub: ("-", "+"),
    ast.Mult: ("*", "/"),
    ast.Div: ("/", "*"),
    ast.FloorDiv: ("//", "/"),
    ast.Mod: ("%", "//"),
}

BOOLEAN_SWAPS: Final[dict[type[ast.boolop], tuple[str, str]]] = {
    ast.And: ("and", "or"),
    ast.Or: ("or", "and"),
}

Candidate = tuple[int, int, str, str, str]
"""One substitution: start offset, end offset, operator name, text before, text after."""


@dataclass(frozen=True)
class Mutation:
    """One text substitution, located precisely enough for a reader to act on it."""

    path: str
    line: int
    column: int
    operator: str
    before: str
    after: str
    start: int
    end: int

    def describe(self) -> str:
        """Render the mutation the way a survivor report has to read."""
        return (
            f"{self.path}:{self.line}:{self.column} [{self.operator}] "
            f"{self.before!r} -> {self.after!r}"
        )

    def as_dict(self) -> dict[str, Any]:
        """Serialise for the JSON artifact."""
        return {
            "path": self.path,
            "line": self.line,
            "column": self.column,
            "operator": self.operator,
            "before": self.before,
            "after": self.after,
        }


def _signal_name(code: int) -> str:
    """Name the signal a negative return code encodes, or say which number it carried.

    The handful below are pinned because the artifact is written on a POSIX runner and
    read anywhere, and ``signal.Signals`` on Windows carries no SIGKILL to name.
    """
    pinned = _POSIX_SIGNALS.get(-code)
    if pinned is not None:
        return pinned
    try:
        return signal.Signals(-code).name
    except ValueError:
        return f"signal {-code}"


def _exit_description(code: int | None) -> str:
    """Say how a child ended, separating a signal from an exit status a test chose."""
    if code is None:
        return "did not finish inside the per-mutant timeout"
    if code < 0:
        return (
            f"was killed by {_signal_name(code)} from outside the process, which a "
            "--memory-mb cap cannot convert into a MemoryError"
        )
    return f"exited {code}"


@dataclass
class TargetResult:
    """What one target's mutants did."""

    name: str
    generated: int = 0
    executed: int = 0
    killed: int = 0
    survived: int = 0
    timed_out: int = 0
    signalled: int = 0
    duration_s: float = 0.0
    peak_child_rss_bytes: int | None = None
    survivors: list[Mutation] = field(default_factory=list)
    timeouts: list[Mutation] = field(default_factory=list)
    signals: list[tuple[Mutation, int]] = field(default_factory=list)
    problems: list[str] = field(default_factory=list)

    @property
    def unproven(self) -> int:
        """Kills that no test proved: the clock ended them, or something outside did."""
        return self.timed_out + self.signalled

    def as_dict(self) -> dict[str, Any]:
        """Serialise for the JSON artifact."""
        return {
            "name": self.name,
            "generated": self.generated,
            "executed": self.executed,
            "killed": self.killed,
            "survived": self.survived,
            "timed_out": self.timed_out,
            "signalled": self.signalled,
            "duration_s": round(self.duration_s, 1),
            "peak_child_rss_bytes": self.peak_child_rss_bytes,
            "survivors": [mutation.as_dict() for mutation in self.survivors],
            "timeouts": [mutation.as_dict() for mutation in self.timeouts],
            "signals": [
                {**mutation.as_dict(), "signal": code, "signal_name": _signal_name(code)}
                for mutation, code in self.signals
            ],
            "problems": list(self.problems),
        }


@dataclass
class RunReport:
    """Every target's result plus the verdict they add up to."""

    verdict: Verdict
    results: list[TargetResult]
    planned: int = 0
    peak_child_rss_bytes: int | None = None

    @property
    def complete(self) -> bool:
        """Whether every target the run was asked for actually produced a result."""
        return self.planned <= len(self.results)

    @property
    def timed_out(self) -> int:
        """Total mutants the clock ended rather than a test."""
        return sum(result.timed_out for result in self.results)

    @property
    def signalled(self) -> int:
        """Total mutants something outside the process ended rather than a test."""
        return sum(result.signalled for result in self.results)

    @property
    def executed(self) -> int:
        """Total mutants actually run."""
        return sum(result.executed for result in self.results)

    @property
    def survived(self) -> int:
        """Total mutants no test noticed."""
        return sum(result.survived for result in self.results)

    @property
    def killed(self) -> int:
        """Total mutants some test caught."""
        return sum(result.killed for result in self.results)

    def as_dict(self) -> dict[str, Any]:
        """Serialise for the JSON artifact."""
        return {
            "verdict": self.verdict.value,
            "executed": self.executed,
            "killed": self.killed,
            "survived": self.survived,
            "timed_out": self.timed_out,
            "signalled": self.signalled,
            "complete": self.complete,
            "planned": max(self.planned, len(self.results)),
            "peak_child_rss_bytes": self.peak_child_rss_bytes,
            "targets": [result.as_dict() for result in self.results],
        }


def _line_offsets(source: str) -> list[int]:
    """Character offset at which each 1-based line starts."""
    offsets = [0, 0]
    for line in source.splitlines(keepends=True):
        offsets.append(offsets[-1] + len(line))
    return offsets


def _offset(offsets: list[int], line: int, column: int) -> int:
    """Translate an ``ast`` (line, column) position into a character offset."""
    return offsets[line] + column


def _token_span(source: str, start: int, end: int, token: str) -> tuple[int, int] | None:
    """Locate ``token`` in the gap between two child nodes, tolerating odd whitespace."""
    pattern = r"\s+".join(re.escape(part) for part in token.split())
    found = re.search(pattern, source[start:end])
    if found is None:
        return None
    return start + found.start(), start + found.end()


def _position(offsets: list[int], offset: int) -> tuple[int, int]:
    """Translate a character offset back into a 1-based line and 0-based column."""
    line = 1
    while line + 1 < len(offsets) and offsets[line + 1] <= offset:
        line += 1
    return line, offset - offsets[line]


def _number_replacement(value: int | float) -> str:
    """Perturb a numeric literal so a boundary test has something to catch."""
    if isinstance(value, bool):
        return "False" if value else "True"
    if value == 0:
        return "1"
    if value == 1:
        return "0"
    return repr(value + 1)


def _gap(offsets: list[int], before: ast.expr, after: ast.expr) -> tuple[int, int] | None:
    """Offsets of the text between two sibling expressions, where their operator sits."""
    if before.end_lineno is None or before.end_col_offset is None:
        return None
    return (
        _offset(offsets, before.end_lineno, before.end_col_offset),
        _offset(offsets, after.lineno, after.col_offset),
    )


def _infix(
    source: str, offsets: list[int], left: ast.expr, right: ast.expr, swap: tuple[str, str]
) -> tuple[int, int] | None:
    """Locate an infix operator's own characters between the operands it joins."""
    gap = _gap(offsets, left, right)
    if gap is None:
        return None
    return _token_span(source, gap[0], gap[1], swap[0])


def _candidates(source: str, tree: ast.Module, offsets: list[int]) -> list[Candidate]:
    """Collect every (start, end, operator, before, after) substitution the catalogue allows."""
    found: list[Candidate] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Compare):
            left: ast.expr = node.left
            for operation, comparator in zip(node.ops, node.comparators, strict=True):
                swap = COMPARISON_SWAPS.get(type(operation))
                span = None if swap is None else _infix(source, offsets, left, comparator, swap)
                if swap is not None and span is not None:
                    found.append((span[0], span[1], "comparison", swap[0], swap[1]))
                left = comparator
        elif isinstance(node, ast.BinOp):
            swap = ARITHMETIC_SWAPS.get(type(node.op))
            span = None if swap is None else _infix(source, offsets, node.left, node.right, swap)
            if swap is not None and span is not None:
                found.append((span[0], span[1], "arithmetic", swap[0], swap[1]))
        elif isinstance(node, ast.BoolOp):
            swap = BOOLEAN_SWAPS[type(node.op)]
            for first, second in zip(node.values, node.values[1:], strict=False):
                span = _infix(source, offsets, first, second, swap)
                if span is not None:
                    found.append((span[0], span[1], "boolean-op", swap[0], swap[1]))
        elif isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not):
            start = _offset(offsets, node.lineno, node.col_offset)
            end = _offset(offsets, node.operand.lineno, node.operand.col_offset)
            if source[start:end].strip() == "not":
                found.append((start, end, "unary-not", source[start:end], ""))
        elif isinstance(node, ast.Constant) and isinstance(node.value, (bool, int, float)):
            if node.end_lineno is None or node.end_col_offset is None:
                continue
            start = _offset(offsets, node.lineno, node.col_offset)
            end = _offset(offsets, node.end_lineno, node.end_col_offset)
            replacement = _number_replacement(node.value)
            kind = "boolean-constant" if isinstance(node.value, bool) else "number"
            found.append((start, end, kind, source[start:end], replacement))
    return found


def generate(source: str, path: str) -> list[Mutation]:
    """Generate every viable mutation of one source file, guards included.

    Each candidate is spliced and kept only if the result still parses, differs
    structurally from the original, and occupies the same number of lines.
    """
    tree = ast.parse(source)
    original = ast.dump(tree)
    offsets = _line_offsets(source)
    line_count = source.count("\n")

    mutations: list[Mutation] = []
    for start, end, operator, before, after in _candidates(source, tree, offsets):
        if before == after:
            continue
        mutated = source[:start] + after + source[end:]
        if mutated.count("\n") != line_count:
            continue
        try:
            candidate = ast.parse(mutated)
        except SyntaxError:
            continue
        if ast.dump(candidate) == original:
            continue
        line, column = _position(offsets, start)
        mutations.append(
            Mutation(
                path=path,
                line=line,
                column=column,
                operator=operator,
                before=before,
                after=after,
                start=start,
                end=end,
            )
        )
    mutations.sort(key=lambda item: (item.line, item.column, item.operator, item.after))
    return mutations


def apply(source: str, mutation: Mutation) -> str:
    """Splice one mutation into the source it was generated from."""
    return source[: mutation.start] + mutation.after + source[mutation.end :]


def select(mutations: list[Mutation], budget: int | None, seed: int) -> list[Mutation]:
    """Take a reproducible sample when the full set exceeds the budget.

    Sampling is seeded and the result re-sorted, so a bounded nightly run is as
    repeatable as a full one and two runs of the same seed report the same mutants.
    """
    if budget is None or budget <= 0 or len(mutations) <= budget:
        return list(mutations)
    chosen = random.Random(seed).sample(range(len(mutations)), budget)
    return [mutations[index] for index in sorted(chosen)]


def source_files(root: Path, source: str) -> list[Path]:
    """Every Python file a target names, whether it points at a module or a package."""
    path = root / source
    if path.is_dir():
        return sorted(path.rglob("*.py"))
    return [path] if path.is_file() else []


def stage_suite(workspace: Path) -> None:
    """Copy ``STAGED_ROOTS`` and pytest's configuration beside the mutated sources.

    Each root is copied whole rather than only the paths a target names, so a test
    that reaches past itself into one of them reads the snapshot. No other root is
    staged, and a test that reaches for one of those reads the live working tree.
    """
    for root in STAGED_ROOTS:
        source = REPOSITORY_ROOT / root
        if source.is_dir():
            shutil.copytree(source, workspace / root, dirs_exist_ok=True, ignore=_IGNORED)
    configuration = REPOSITORY_ROOT / "pyproject.toml"
    if configuration.is_file():
        shutil.copy2(configuration, workspace / configuration.name)


def staged_tests(tests: tuple[str, ...], workspace: Path) -> tuple[str, ...]:
    """Point one target's test paths at their copies, refusing one the snapshot lacks.

    Reading the tests live from the working tree makes every verdict a function of
    whatever was saved while the run was in flight, which is no verdict at all.
    """
    resolved: list[str] = []
    root = workspace.resolve()
    for test in tests:
        copy = (workspace / test).resolve()
        if not copy.is_relative_to(root):
            raise ValueError(f"{test} points outside the snapshot")
        if not copy.exists():
            raise FileNotFoundError(f"{test} matched no test in the snapshot")
        resolved.append(str(copy))
    return tuple(resolved)


def load_targets(document: Any) -> list[Target]:
    """Read a --targets-file document, refusing a shape that cannot be run."""
    if not isinstance(document, list):
        raise ValueError("a targets file must hold a list of targets")
    targets: list[Target] = []
    for entry in document:
        if not isinstance(entry, dict):
            raise ValueError("each target must be an object with name, source and tests")
        name = entry.get("name")
        source = entry.get("source")
        tests = entry.get("tests")
        if not isinstance(name, str) or not isinstance(source, str):
            raise ValueError("each target needs a string name and a string source path")
        if not isinstance(tests, list) or not tests or not all(isinstance(t, str) for t in tests):
            raise ValueError(f"target {name!r} needs a non-empty list of test paths")
        targets.append(Target(name=name, source=source, tests=tuple(tests)))
    return targets


def restrict(targets: tuple[Target, ...] | list[Target], only: list[str]) -> list[Target]:
    """Keep the named targets, refusing a name that does not exist rather than running nothing."""
    if not only:
        return list(targets)
    known = {target.name: target for target in targets}
    missing = sorted(set(only) - set(known))
    if missing:
        raise ValueError(f"unknown target(s): {', '.join(missing)}")
    return [known[name] for name in only]


def decide(results: list[TargetResult]) -> Verdict:
    """Decide the run's verdict; a run that could not be performed is never clean."""
    if not results or any(result.problems for result in results):
        return Verdict.NOT_RUN
    if not any(result.executed for result in results):
        return Verdict.NOT_RUN
    if any(result.survivors for result in results):
        return Verdict.SURVIVORS
    return Verdict.CLEAN


def _peak_setter(report: RunReport) -> str | None:
    """Name the target whose children first reached the run's peak, if one can be named."""
    reached: int | None = None
    setter: str | None = None
    for result in report.results:
        peak = result.peak_child_rss_bytes
        if peak is not None and (reached is None or peak > reached):
            reached = peak
            setter = result.name
    return setter


def _text(entry: Mapping[str, Any], key: str, where: str) -> str:
    """Read one string the gate needs, refusing a document that does not carry it."""
    value = entry.get(key)
    if not isinstance(value, str):
        raise ValueError(f"{where}: {key} is missing or not a string")
    return value


def _integer(entry: Mapping[str, Any], key: str, where: str) -> int:
    """Read one count the gate has to add up, refusing anything it cannot."""
    value = entry.get(key)
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"{where}: {key} is missing or not an integer")
    return value


def _mutation(entry: Any, where: str) -> Mutation:
    """Rebuild one named mutant from the artifact; its offsets are never used again."""
    if not isinstance(entry, dict):
        raise ValueError(f"{where}: a listed mutant is not an object")
    return Mutation(
        path=_text(entry, "path", where),
        line=_integer(entry, "line", where),
        column=_integer(entry, "column", where),
        operator=_text(entry, "operator", where),
        before=_text(entry, "before", where),
        after=_text(entry, "after", where),
        start=0,
        end=0,
    )


def _mutants(entry: Mapping[str, Any], key: str, where: str) -> list[Mutation]:
    """Rebuild one of a target's named mutant lists."""
    listed = entry.get(key, [])
    if not isinstance(listed, list):
        raise ValueError(f"{where}: {key} is not a list")
    return [_mutation(item, where) for item in listed]


def _target_result(entry: Any) -> TargetResult:
    """Rebuild one target's counts, refusing a shape or an arithmetic no gate can judge.

    ``signalled`` is required rather than defaulted: an artifact without it was written
    by a runner that could not tell a signalled child from a test failure.
    """
    if not isinstance(entry, dict):
        raise ValueError("a listed target is not an object")
    name = _text(entry, "name", "target")
    where = f"target {name}"
    problems = entry.get("problems", [])
    if not isinstance(problems, list) or not all(isinstance(item, str) for item in problems):
        raise ValueError(f"{where}: problems is not a list of strings")
    result = TargetResult(
        name=name,
        generated=_integer(entry, "generated", where),
        executed=_integer(entry, "executed", where),
        killed=_integer(entry, "killed", where),
        survived=_integer(entry, "survived", where),
        timed_out=_integer(entry, "timed_out", where),
        signalled=_integer(entry, "signalled", where),
        survivors=_mutants(entry, "survivors", where),
        timeouts=_mutants(entry, "timeouts", where),
        problems=list(problems),
    )
    if result.executed != result.killed + result.survived:
        raise ValueError(
            f"{where}: {result.executed} executed is not {result.killed} killed "
            f"plus {result.survived} survived"
        )
    if result.survived != len(result.survivors):
        raise ValueError(
            f"{where}: {result.survived} survived but {len(result.survivors)} are named"
        )
    if result.unproven > result.killed:
        raise ValueError(f"{where}: {result.unproven} unproven kills of {result.killed}")
    return result


def read_report(document: Any) -> RunReport:
    """Rebuild a RunReport from a written artifact, refusing one that cannot be judged.

    Every guard here is a vacuity guard: the gate must never read a document it does
    not understand as a run that went well.
    """
    if not isinstance(document, dict):
        raise ValueError("the artifact is not a JSON object")
    targets = document.get("targets")
    if not isinstance(targets, list) or not targets:
        raise ValueError("the artifact lists no target at all")
    verdict = _text(document, "verdict", "artifact")
    if verdict not in {member.value for member in Verdict}:
        raise ValueError(f"the artifact carries an unknown verdict {verdict!r}")
    complete = document.get("complete")
    if not isinstance(complete, bool):
        raise ValueError("the artifact does not say whether it is complete")
    peak = document.get("peak_child_rss_bytes")
    if peak is not None and (not isinstance(peak, int) or isinstance(peak, bool)):
        raise ValueError("the artifact's peak child RSS is neither a number nor absent")
    report = RunReport(
        Verdict(verdict),
        [_target_result(entry) for entry in targets],
        planned=_integer(document, "planned", "artifact"),
        peak_child_rss_bytes=peak,
    )
    if report.complete != complete:
        raise ValueError(
            f"the artifact calls itself complete={complete} while carrying "
            f"{len(report.results)} of {report.planned} targets"
        )
    return report


def ratchet(report: RunReport, ceiling: Mapping[str, Ceiling] = CEILING) -> list[str]:
    """Every way this run exceeds the committed ceiling, worst overshoot first.

    An empty list is the only result that means held. Callers guarantee the ceiling
    covers every target in the report, which ``gate`` establishes before it asks.
    """
    exceeded: list[tuple[int, str]] = []
    for result in report.results:
        limit = ceiling[result.name]
        if result.survived > limit.survived:
            over = result.survived - limit.survived
            named = "; ".join(mutation.describe() for mutation in result.survivors)
            exceeded.append(
                (
                    over,
                    f"{result.name}: {result.survived} mutants survived against a ceiling of "
                    f"{limit.survived} ({over} more). The ceiling records a count rather than "
                    f"an identity, so compare these against the list in the baseline run: "
                    f"{named}",
                )
            )
        if result.unproven > limit.unproven:
            over = result.unproven - limit.unproven
            exceeded.append(
                (
                    over,
                    f"{result.name}: {result.timed_out} kills were the clock and "
                    f"{result.signalled} were a signal, against a ceiling of {limit.unproven}. "
                    "An unproven kill is not evidence that a test caught anything",
                )
            )
    return [text for _, text in sorted(exceeded, key=lambda item: -item[0])]


def ratchet_block(report: RunReport, ceiling: Mapping[str, Ceiling] = CEILING) -> str:
    """Render the CEILING literal this run's counts justify, refusing to loosen one.

    The gate prints this only for a run that held, which is what makes tightening
    mechanical and raising a hand edit somebody has to defend in review.
    """
    lines = ["CEILING: Final[dict[str, Ceiling]] = {"]
    for result in report.results:
        limit = ceiling[result.name]
        if result.survived > limit.survived or result.unproven > limit.unproven:
            raise ValueError(
                f"{result.name} is above its ceiling, and a block is never printed to raise one"
            )
        unproven = "" if result.unproven == 0 else f", unproven={result.unproven}"
        lines.append(
            f'    "{result.name}": Ceiling(survived={result.survived}, '
            f"sample={result.executed}{unproven}),"
        )
    lines.append("}")
    return "\n".join(lines)


def _vacuous(report: RunReport, ceiling: Mapping[str, Ceiling]) -> str | None:
    """Why this artifact cannot be judged at all, or ``None`` when it can."""
    recomputed = decide(report.results)
    if recomputed is Verdict.NOT_RUN:
        problems = "; ".join(p for result in report.results for p in result.problems)
        return "the mutation run could not be performed, so there is nothing to ratchet: " + (
            problems or "no target executed a single mutant"
        )
    if recomputed is not report.verdict:
        return (
            f"the artifact calls itself {report.verdict.value!r} while its own counts say "
            f"{recomputed.value!r}, so one of the two was written by something else"
        )
    if not report.complete:
        return (
            f"the artifact is a partial: {len(report.results)} of {report.planned} targets "
            "finished, and a ratchet judged on part of a run is not a ratchet"
        )
    ran = {result.name for result in report.results}
    missing = sorted(set(ceiling) - ran)
    if missing:
        return (
            f"the ceiling covers {', '.join(missing)}, which this run did not execute — a "
            "ratchet a run can satisfy by skipping a target is not a ratchet"
        )
    unknown = sorted(ran - set(ceiling))
    if unknown:
        return (
            f"{', '.join(unknown)} ran with no ceiling to be held to; give each one a measured "
            "entry in CEILING rather than leaving it ungated"
        )
    resampled = [
        f"{result.name} ran {result.executed} of its mutants against a ceiling measured on "
        f"{ceiling[result.name].sample}"
        for result in report.results
        if result.executed != ceiling[result.name].sample
    ]
    if resampled:
        return (
            "the ceiling cannot judge a different sample, in either direction — a smaller one "
            "would satisfy it for free and a larger one would fail it for nothing: "
            + "; ".join(resampled)
        )
    return None


def gate(path: Path, ceiling: Mapping[str, Ceiling] = CEILING) -> tuple[Verdict, str]:
    """Judge a written artifact against the ceiling: held, regressed, or never really run.

    Exit codes keep their meaning from a live run — 0 held, 1 regressed, 2 the run
    cannot be judged — so the workflow treats both failures the way it always has.
    """
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        return Verdict.NOT_RUN, f"the mutation artifact {path} could not be read: {exc}"
    try:
        report = read_report(document)
    except ValueError as exc:
        return Verdict.NOT_RUN, f"the mutation artifact {path} cannot be judged: {exc}"

    refusal = _vacuous(report, ceiling)
    if refusal is not None:
        return Verdict.NOT_RUN, refusal

    exceeded = ratchet(report, ceiling)
    if exceeded:
        return Verdict.SURVIVORS, "\n".join(
            ["the mutation ratchet moved the wrong way:", *(f"  {line}" for line in exceeded)]
        )

    lines = [
        f"the mutation ratchet holds: {report.survived} survived of {report.executed} executed "
        f"over {len(report.results)} targets, none above its ceiling"
    ]
    if any(
        result.survived < ceiling[result.name].survived
        or result.unproven < ceiling[result.name].unproven
        for result in report.results
    ):
        lines.extend(
            [
                "",
                "a target improved; tighten the ceiling in scripts/mutation_run.py to:",
                "",
                ratchet_block(report, ceiling),
            ]
        )
    return Verdict.CLEAN, "\n".join(lines)


def render(report: RunReport) -> str:
    """Render the human report: counts per target, then every survivor in full."""
    lines = [f"mutation run: {report.verdict.value}"]
    if not report.complete:
        lines.append(
            f"  PARTIAL: {len(report.results)} of {report.planned} targets ran before the end"
        )
    for result in report.results:
        lines.append(
            f"  {result.name}: {result.executed}/{result.generated} run, "
            f"{result.killed} killed, {result.survived} survived, "
            f"{result.timed_out} timed out, {result.signalled} signalled, "
            f"{result.duration_s:.0f}s"
        )
        for problem in result.problems:
            lines.append(f"    [not-run] {problem}")
    if report.peak_child_rss_bytes is not None:
        setter = _peak_setter(report)
        reached = "" if setter is None else f", first reached in {setter}"
        lines.append(f"  peak child RSS: {report.peak_child_rss_bytes // _MEGABYTE} MiB{reached}")
    survivors = [(result.name, m) for result in report.results for m in result.survivors]
    if survivors:
        lines.append("")
        lines.append("surviving mutants (each one is a line no test defends):")
        lines.extend(f"  [{name}] {mutation.describe()}" for name, mutation in survivors)
    timeouts = [(result.name, m) for result in report.results for m in result.timeouts]
    if timeouts:
        lines.append("")
        lines.append("timed-out mutants (counted as killed; the clock ended them, not a test):")
        lines.extend(f"  [{name}] {mutation.describe()}" for name, mutation in timeouts)
    signals = [(result.name, m, c) for result in report.results for m, c in result.signals]
    if signals:
        lines.append("")
        lines.append("signalled mutants (counted as killed; something outside the process was):")
        lines.extend(
            f"  [{name}] {_signal_name(code)} {mutation.describe()}"
            for name, mutation, code in signals
        )
    return "\n".join(lines)


def _peak_mib(result: TargetResult) -> str:
    """Render one target's high-water mark, or a dash where the platform reports none."""
    if result.peak_child_rss_bytes is None:
        return "-"
    return str(result.peak_child_rss_bytes // _MEGABYTE)


def markdown(report: RunReport) -> str:
    """Render the same report for the job summary, where a reader cannot avoid it."""
    lines = [
        "## Mutation testing",
        "",
        f"**Verdict: `{report.verdict.value}`** — "
        f"{report.killed} killed, {report.survived} survived of {report.executed} run.",
        "",
        "| Target | Run | Killed | Survived | Timed out | Signalled | Peak MiB | Seconds |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    lines.extend(
        f"| {result.name} | {result.executed} | {result.killed} | {result.survived} "
        f"| {result.timed_out} | {result.signalled} | {_peak_mib(result)} "
        f"| {result.duration_s:.0f} |"
        for result in report.results
    )
    if not report.complete:
        lines.extend(
            ["", f"**Partial:** {len(report.results)} of {report.planned} targets ran.", ""]
        )
    if report.peak_child_rss_bytes is not None:
        setter = _peak_setter(report)
        reached = "" if setter is None else f", first reached in `{setter}`"
        lines.extend(
            ["", f"Peak child RSS: {report.peak_child_rss_bytes // _MEGABYTE} MiB{reached}.", ""]
        )
    problems = [p for result in report.results for p in result.problems]
    if problems:
        lines.extend(["", "**Could not run:**", ""])
        lines.extend(f"- {problem}" for problem in problems)
    survivors = [(result.name, m) for result in report.results for m in result.survivors]
    if survivors:
        lines.extend(["", "**Surviving mutants** — each is a line no test defends:", ""])
        lines.extend(f"- `{name}` {mutation.describe()}" for name, mutation in survivors)
    timeouts = [(result.name, m) for result in report.results for m in result.timeouts]
    if timeouts:
        lines.extend(
            [
                "",
                f"**Timed out** — {report.timed_out} of the {report.killed} kills were the "
                "clock rather than a test:",
                "",
            ]
        )
        lines.extend(f"- `{name}` {mutation.describe()}" for name, mutation in timeouts)
    signals = [(result.name, m, c) for result in report.results for m, c in result.signals]
    if signals:
        lines.extend(
            [
                "",
                f"**Signalled** — {report.signalled} child process(es) were ended from "
                "outside rather than by a test, which no memory cap can convert into a kill:",
                "",
            ]
        )
        lines.extend(
            f"- `{name}` {_signal_name(code)} {mutation.describe()}"
            for name, mutation, code in signals
        )
    return "\n".join(lines) + "\n"


def _progress(line: str) -> None:
    """Announce one step on stdout and flush it, so a killed run still names it.

    Flushing here rather than trusting ``PYTHONUNBUFFERED`` is deliberate: the report
    is written only at the end, so without this a SIGTERM leaves no record at all.
    """
    sys.stdout.write(f"{line}\n")
    sys.stdout.flush()


def _pytest(
    tests: tuple[str, ...],
    workspace: Path,
    timeout: float,
    limiter: Callable[[], None] | None = None,
) -> int | None:
    """Run one target's staged tests against the mutated copy; ``None`` means it timed out.

    ``-o pythonpath=`` empties the ini setting that would otherwise put the real ``src``
    ahead of the copy; ``PYTHONDONTWRITEBYTECODE`` stops a ``.pyc`` outliving its mutant.
    """
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(workspace)
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    command = [
        sys.executable,
        "-m",
        "pytest",
        *tests,
        "-x",
        "-q",
        "--no-header",
        "-p",
        "no:cacheprovider",
        "-o",
        "pythonpath=",
    ]
    try:
        completed = subprocess.run(
            command,
            cwd=REPOSITORY_ROOT,
            env=environment,
            capture_output=True,
            timeout=timeout,
            check=False,
            preexec_fn=limiter,
        )
    except subprocess.TimeoutExpired:
        return None
    return completed.returncode


def _run_target(
    target: Target,
    workspace: Path,
    *,
    budget: int | None,
    seed: int,
    timeout: float,
    limiter: Callable[[], None] | None = None,
) -> TargetResult:
    """Mutate one target in the workspace copy and record what its tests caught.

    Mutations are generated from the copy and the tests are read from the copy, so
    neither a mutation's offsets nor its verdict depend on the live working tree.
    """
    result = TargetResult(name=target.name)
    started = time.monotonic()

    files = source_files(REPOSITORY_ROOT, target.source)
    if not files:
        result.problems.append(f"{target.name}: {target.source} matched no Python file")
        return result

    try:
        tests = staged_tests(target.tests, workspace)
    except (OSError, ValueError) as exc:
        result.problems.append(f"{target.name}: its tests could not be snapshotted: {exc}")
        return result

    _progress(f"[{target.name}] baseline: {len(tests)} staged test path(s)")
    baseline = _pytest(tests, workspace, timeout, limiter)
    if baseline != 0:
        result.problems.append(
            f"{target.name}: the unmutated suite {_exit_description(baseline)}, so every "
            "mutant would have died for free"
        )
        return result

    mutations: list[Mutation] = []
    for file in files:
        relative = file.relative_to(REPOSITORY_ROOT)
        copy = workspace / relative.relative_to("src")
        mutations.extend(generate(copy.read_text(encoding="utf-8"), relative.as_posix()))
    result.generated = len(mutations)
    if not mutations:
        result.problems.append(f"{target.name}: no mutant could be generated from {target.source}")
        return result

    chosen = select(mutations, budget, seed)
    for index, mutation in enumerate(chosen, start=1):
        _progress(f"[{target.name}] {index}/{len(chosen)} {mutation.describe()}")
        copied = workspace / Path(mutation.path).relative_to("src")
        original = copied.read_text(encoding="utf-8")
        copied.write_text(apply(original, mutation), encoding="utf-8")
        try:
            code = _pytest(tests, workspace, timeout, limiter)
        finally:
            copied.write_text(original, encoding="utf-8")
        result.executed += 1
        if code is None:
            result.timed_out += 1
            result.killed += 1
            result.timeouts.append(mutation)
        elif code == 0:
            result.survived += 1
            result.survivors.append(mutation)
        elif code < 0:
            result.signalled += 1
            result.killed += 1
            result.signals.append((mutation, code))
        else:
            result.killed += 1

    result.duration_s = time.monotonic() - started
    _progress(
        f"[{target.name}] done: {result.executed}/{result.generated} run, {result.killed} killed "
        f"({result.timed_out} by the clock, {result.signalled} by a signal), "
        f"{result.survived} survived, {result.duration_s:.0f}s"
    )
    return result


def run(
    targets: list[Target],
    *,
    budget: int | None,
    seed: int,
    timeout: float,
    workspace: Path,
    limiter: Callable[[], None] | None = None,
    limits: ModuleType | None = None,
    after_target: Callable[[RunReport], None] | None = None,
) -> RunReport:
    """Snapshot ``src/``, the staged roots and the ini, then run every target against the copy.

    ``after_target`` sees the report as it grows, one call per finished target, so a
    campaign that is killed mid-run still leaves the targets it completed.
    """
    shutil.copytree(REPOSITORY_ROOT / "src", workspace, dirs_exist_ok=True, ignore=_IGNORED)
    stage_suite(workspace)
    results: list[TargetResult] = []
    report = RunReport(Verdict.NOT_RUN, results, planned=len(targets))
    for target in targets:
        outcome = _run_target(
            target, workspace, budget=budget, seed=seed, timeout=timeout, limiter=limiter
        )
        results.append(outcome)
        report.verdict = decide(results)
        report.peak_child_rss_bytes = child_peak_rss(limits)
        outcome.peak_child_rss_bytes = report.peak_child_rss_bytes
        if after_target is not None:
            after_target(report)
    return report


def _write(path: Path, body: str, *, append: bool) -> None:
    """Write a report file, reporting a write failure without altering the verdict."""
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a" if append else "w", encoding="utf-8") as handle:
            handle.write(body)
    except OSError as exc:
        sys.stderr.write(f"could not write {path}: {exc}\n")


def _write_json(path: Path, report: RunReport) -> None:
    """Replace the JSON artifact atomically, so a kill mid-write cannot truncate it.

    The artifact is rewritten once per target, which is the whole point of it: a run
    that dies leaves the targets it finished, and a half-written file would leave none.
    """
    payload = json.dumps(report.as_dict(), indent=2, sort_keys=True)
    staging = path.with_name(f"{path.name}.partial")
    _write(staging, f"{payload}\n", append=False)
    try:
        os.replace(staging, path)
    except OSError as exc:
        sys.stderr.write(f"could not replace {path}: {exc}\n")


def _parser() -> argparse.ArgumentParser:
    """Build the command line."""
    parser = argparse.ArgumentParser(
        description="Mutation-test the targeted modules and report every surviving mutant."
    )
    parser.add_argument(
        "--only", action="append", default=[], help="run one named target (repeatable)"
    )
    parser.add_argument(
        "--targets-file", type=Path, default=None, help="JSON target set replacing the default"
    )
    parser.add_argument(
        "--budget",
        type=int,
        default=None,
        help="maximum mutants per target; a seeded sample is taken above it",
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="sampling seed")
    parser.add_argument(
        "--timeout", type=float, default=DEFAULT_TIMEOUT_S, help="seconds allowed per mutant"
    )
    parser.add_argument(
        "--memory-mb",
        type=int,
        default=None,
        help="cap each mutant's address space in MiB (POSIX only; a runaway dies as a kill)",
    )
    parser.add_argument(
        "--check-ratchet",
        type=Path,
        default=None,
        help="judge an existing --json-out artifact against CEILING and run no mutant",
    )
    parser.add_argument("--list", action="store_true", help="count the mutants and run none")
    parser.add_argument("--json-out", type=Path, default=None, help="write the report as JSON")
    parser.add_argument(
        "--summary", type=Path, default=None, help="append a markdown summary to this file"
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the mutation layer and exit with the verdict's code."""
    args = _parser().parse_args(argv)

    if args.check_ratchet is not None:
        verdict, text = gate(args.check_ratchet)
        stream = sys.stdout if verdict is Verdict.CLEAN else sys.stderr
        stream.write(f"{text}\n")
        return EXIT_CODES[verdict]

    try:
        configured: list[Target] = (
            load_targets(json.loads(args.targets_file.read_text(encoding="utf-8")))
            if args.targets_file is not None
            else list(DEFAULT_TARGETS)
        )
        targets = restrict(configured, args.only)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        sys.stderr.write(f"target set unusable: {exc}\n")
        return EXIT_CODES[Verdict.NOT_RUN]

    if args.list:
        for target in targets:
            total = sum(
                len(generate(file.read_text(encoding="utf-8"), file.as_posix()))
                for file in source_files(REPOSITORY_ROOT, target.source)
            )
            sys.stdout.write(f"{target.name}: {total} mutants from {target.source}\n")
        return EXIT_CODES[Verdict.CLEAN]

    limits = resource_module()
    try:
        limiter = memory_limiter(args.memory_mb, limits)
    except ValueError as exc:
        sys.stderr.write(f"{exc}\n")
        return EXIT_CODES[Verdict.NOT_RUN]

    def snapshot(partial: RunReport) -> None:
        """Rewrite the JSON artifact after every target, so a kill leaves what finished."""
        if args.json_out is not None:
            _write_json(args.json_out, partial)

    with tempfile.TemporaryDirectory(prefix="fasterrag-mutation-") as directory:
        report = run(
            targets,
            budget=args.budget,
            seed=args.seed,
            timeout=args.timeout,
            workspace=Path(directory),
            limiter=limiter,
            limits=limits,
            after_target=snapshot,
        )

    sys.stdout.write(f"{render(report)}\n")
    if args.json_out is not None:
        _write_json(args.json_out, report)
    if args.summary is not None:
        _write(args.summary, markdown(report), append=True)
    return EXIT_CODES[report.verdict]


if __name__ == "__main__":
    raise SystemExit(main())
