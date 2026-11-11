"""Fail when a contract suite's JUnit report shows it skipped, or shows nothing at all.

**The defect this exists to stop is one level up from the suites.** `pytest` exits **0**
when every selected test skips. The `integration` job has concluded success on every push
for as long as the Milvus and Weaviate adapters have existed, while 98 of its 380 selected
cases skipped entire for want of a server (`docs/testing-strategy.md` §3.1). A job step
that reads only the exit code cannot tell "the contract held" from "the contract was never
tested", so provisioning a backend and then trusting the exit code would reproduce the same
defect in the job that was added to fix it — rebuild-playbook trap T12.

So the stack's job writes `--junitxml` and this reader judges the report. The rules, each
of which exits 1:

* ``skipped`` — a case in the report carries a ``<skipped>`` child. The whole point: a
  contract suite pointed at a provisioned server has no business skipping, and the skip
  message is printed so the reason is in the log beside the verdict.
* ``empty`` — the report parses and holds zero test cases. A suite that collected nothing
  is the failure mode a bad path argument or a collection error produces, and it is
  indistinguishable from a clean run by exit code alone.
* ``missing`` — no report at the named path. `pytest` writes the file even when it fails,
  so an absent one means the step did not reach pytest at all.
* ``unreadable`` — the file exists and is not parseable JUnit XML, or its root is neither
  ``<testsuites>`` nor ``<testsuite>``. A truncated upload must not read as a pass.

**Failures and errors are NOT findings here, deliberately.** A contract case that ran and
failed is a *result*: it executed against the server, which is all this gate asks, and the
job's own pytest step goes red for it. Counting it twice would make this gate's message the
one a reader debugs, and the pytest output is the better message. The counts are printed so
the summary line is honest about what ran, but only a skip, an empty report, or a missing
one sets the exit code.

**Why a skip is fatal rather than warned about.** Both conftests skip with the connection
error when nothing answers, which is correct behaviour for a developer running the suite
with no daemon and exactly wrong for a CI job that just started the server: there, a skip
means the stack did not come up, or came up too late, or published the wrong port, and the
suite silently measured nothing. `--wait` is supposed to prevent the race; this is what
notices when it did not.

What this gate CANNOT do: it reads a report, so it cannot tell a suite that skipped from
one that was never selected — ``empty`` is the answer to both, and the remedy differs. It
does not know which suites *should* appear in a report, so pointing it at the report of a
run that selected one suite instead of two passes; ``--expect-at-least`` is the blunt
instrument for that, and it is a count rather than a roster.

Exit codes: ``0`` when every case in every report executed, ``1`` otherwise.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final
from xml.etree import ElementTree

SKIPPED: Final = "skipped"
EMPTY: Final = "empty"
MISSING: Final = "missing"
UNREADABLE: Final = "unreadable"
TOO_FEW: Final = "too-few"

_SUITE_ROOTS: Final = frozenset({"testsuites", "testsuite"})

REMEDY: Final = (
    "A contract suite that skips has measured nothing, and pytest exits 0 when every "
    "selected case skips. Bring the backend up before the suite runs and confirm the "
    "published ports match the conftest defaults; do not widen this gate to accept a "
    "skip, and do not drop it and read the exit code instead.\n"
)


@dataclass(frozen=True, slots=True)
class Finding:
    """One way a JUnit report fails to show that a contract suite executed."""

    rule: str
    subject: str
    detail: str

    def __str__(self) -> str:
        """Return the one-line report form."""
        return f"[{self.rule}] {self.subject}: {self.detail}"


@dataclass(frozen=True, slots=True)
class Tally:
    """What one report says ran, and how."""

    cases: int
    skipped: int
    failed: int
    errored: int

    @property
    def executed(self) -> int:
        """Return the cases that reached the server, verdict aside."""
        return self.cases - self.skipped


def _case_name(case: ElementTree.Element) -> str:
    """Return a readable node id for one ``<testcase>``."""
    classname = case.get("classname", "")
    name = case.get("name", "(unnamed)")
    return f"{classname}::{name}" if classname else name


def _skip_reason(case: ElementTree.Element) -> str:
    """Return why one case skipped, from whichever attribute carries it."""
    skip = case.find("skipped")
    if skip is None:
        return ""
    message = skip.get("message") or (skip.text or "").strip()
    return " ".join(message.split())[:300]


def read_report(path: Path) -> list[ElementTree.Element] | str:
    """Return the ``<testcase>`` elements in the report at ``path``, or why it cannot be read."""
    try:
        root = ElementTree.parse(path).getroot()
    except OSError as error:
        return f"cannot be read ({error.strerror or error})"
    except ElementTree.ParseError as error:
        return f"is not parseable XML ({error})"
    if root.tag not in _SUITE_ROOTS:
        return (
            f"has root element <{root.tag}>, and a JUnit report's root is "
            "<testsuites> or <testsuite>; this is not a JUnit report"
        )
    return list(root.iter("testcase"))


def _tally(cases: Sequence[ElementTree.Element]) -> Tally:
    """Return the counts for one report's cases."""
    return Tally(
        cases=len(cases),
        skipped=sum(1 for case in cases if case.find("skipped") is not None),
        failed=sum(1 for case in cases if case.find("failure") is not None),
        errored=sum(1 for case in cases if case.find("error") is not None),
    )


def check_report(path: Path) -> tuple[Tally, list[Finding]]:
    """Return what one report shows, with a finding for each way it shows nothing."""
    where = path.as_posix()
    if not path.is_file():
        return Tally(0, 0, 0, 0), [
            Finding(
                MISSING,
                where,
                "no report at this path; pytest writes --junitxml even when it fails, so "
                "an absent report means the step never reached pytest",
            )
        ]

    loaded = read_report(path)
    if isinstance(loaded, str):
        return Tally(0, 0, 0, 0), [Finding(UNREADABLE, where, f"this report {loaded}")]

    tally = _tally(loaded)
    if not tally.cases:
        return tally, [
            Finding(
                EMPTY,
                where,
                "the report holds zero test cases, so the suite selected nothing and this "
                "run proves nothing about the contract",
            )
        ]

    findings = [
        Finding(
            SKIPPED,
            where,
            f"{_case_name(case)} skipped: {_skip_reason(case) or '(no reason)'}",
        )
        for case in loaded
        if case.find("skipped") is not None
    ]
    return tally, findings


def check(paths: Sequence[Path], expect_at_least: int = 1) -> tuple[Tally, list[Finding]]:
    """Return the combined tally and every finding across ``paths``."""
    if not paths:
        return Tally(0, 0, 0, 0), [
            Finding(MISSING, "(no path given)", "this gate was run with no report to read")
        ]

    totals = Tally(0, 0, 0, 0)
    findings: list[Finding] = []
    for path in paths:
        tally, found = check_report(path)
        totals = Tally(
            totals.cases + tally.cases,
            totals.skipped + tally.skipped,
            totals.failed + tally.failed,
            totals.errored + tally.errored,
        )
        findings.extend(found)

    if totals.executed < expect_at_least:
        findings.append(
            Finding(
                TOO_FEW,
                ", ".join(path.as_posix() for path in paths),
                f"{totals.executed} cases executed and at least {expect_at_least} were "
                "expected; the run selected less than it was asked to prove",
            )
        )
    return totals, findings


def _report(write: Callable[[str], object], tally: Tally, findings: list[Finding]) -> None:
    """Write the findings, then the counts, so a reader sees both."""
    if findings:
        write("contract runs this report cannot vouch for:\n")
        for finding in findings:
            write(f"  {finding}\n")
        write("\n")
    write(
        f"{tally.executed} of {tally.cases} cases executed "
        f"({tally.skipped} skipped, {tally.failed} failed, {tally.errored} errored)\n"
    )


def main(argv: list[str] | None = None) -> int:
    """Run the gate over the JUnit reports a contract job produced."""
    parser = argparse.ArgumentParser(
        description=(
            "Check that a contract suite actually executed, since pytest exits 0 when "
            "every selected case skips."
        )
    )
    parser.add_argument("reports", type=Path, nargs="*", help="JUnit XML reports to read")
    parser.add_argument(
        "--expect-at-least",
        type=int,
        default=1,
        help="the fewest executed cases this run is allowed to show",
    )
    arguments = parser.parse_args(argv)

    tally, findings = check(arguments.reports, arguments.expect_at_least)
    _report(sys.stderr.write if findings else sys.stdout.write, tally, findings)
    if findings:
        sys.stderr.write(REMEDY)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
