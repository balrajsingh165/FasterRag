"""Fail when a test named as must-run skipped in every CI lane that could have run it.

``-rs`` (TASK-0328) prints every skip with its reason, which makes a skip legible to
somebody who reads the log. It does not make one impossible to ignore: the census that
found "all seven unit-lane skips skip on both platforms" was produced by diffing progress
dots against a local collection count, by hand, once. A test that skips on every platform
is indistinguishable from a test that does not exist, and the suite is green either way.

This gate closes the half ``-rs`` cannot. ``@pytest.mark.must_run`` is a claim about one
test — *this one must have executed somewhere in CI* — and this script is what makes the
claim cost something.

**What "skipped everywhere" means, and why it is not flattened.** The platforms legitimately
differ. ``test_a_symlink_out_of_an_allowed_root_is_refused`` creates a symlink, which a
Windows account without ``SeCreateSymbolicLinkPrivilege`` cannot do; skipping there and
running on Linux is the correct outcome and must stay green. Skipping on *both* is the bug,
because the confinement guard's symlink leg is then unverified everywhere. So the rule is
**ran at least once**, never "ran everywhere": each lane reports its own verdict, the
verdicts are kept apart, and a finding prints all of them so the asymmetry stays readable.

**Where the evidence comes from.** ``tests/conftest.py`` writes one JSON document per
pytest invocation when ``FASTERRAG_MUST_RUN_REPORT`` names a destination: which marked
tests that invocation collected, and whether each one ran or skipped. The ubuntu and
windows unit lanes each upload theirs as an artifact, and the gate job downloads both.
A document also records whether its invocation RAN tests or only collected them, so a
``--collect-only`` roster can never be mistaken for a lane's evidence or the reverse.

**The roster is separate on purpose.** A lane can only report on tests it collected, so a
test marked ``must_run`` and placed in a lane CI never runs would appear in no report at
all — and "absent from every report" must not read as "nothing to check". The roster is a
``--collect-only`` pass with the marker filter and no lane restriction, so it is the
authoritative set of tests that *claim* the marker, and every one of them has to be
accounted for by some lane.

The rules, each of which exits 1:

* ``ran-nowhere`` — a roster entry that no lane reported as ``ran``. The finding names
  every lane and what it said, including the lanes that never collected it.
* ``unknown-test`` — a lane reported a node id the roster does not contain. The two
  disagree about what exists, so the evidence belongs to a different tree than the roster
  and nothing here can be trusted.
* ``missing-roster`` / ``unreadable`` / ``wrong-mode`` — the inputs are not what they
  claim to be.
* ``empty-roster`` — the roster holds zero marked tests. The marker is registered and
  nothing carries it, so this gate proves nothing.
* ``no-reports`` — no lane report was found at all.
* ``no-evidence`` — reports exist and between them record not one outcome.

The last four are the vacuity guard, and they are the point: a gate that cannot tell
"clean" from "never ran" is worse than no gate (rebuild-playbook trap T12), and this one
has the obvious hole — an artifact that failed to upload would otherwise leave an empty
directory and a green build.

What this gate CANNOT do, said plainly:

* it cannot decide which tests deserve the marker, and a marker on every test is a marker
  that means nothing. The set is meant to stay small enough to read: see
  ``docs/testing-strategy.md`` §3.2;
* it judges execution, never verdicts. A marked test that ran and failed is ``ran`` here,
  and its own job is what goes red;
* it sees only the lanes that set ``FASTERRAG_MUST_RUN_REPORT``. Marking a test in a lane
  that does not set it is caught — as ``ran-nowhere``, which is the right answer and the
  wrong-looking message — and the fix is to wire that lane, not to unmark the test.

Exit codes: ``0`` when every roster entry ran somewhere, ``1`` otherwise.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final

REPOSITORY_ROOT: Final = Path(__file__).resolve().parents[1]

RAN_NOWHERE: Final = "ran-nowhere"
UNKNOWN_TEST: Final = "unknown-test"
MISSING_ROSTER: Final = "missing-roster"
UNREADABLE: Final = "unreadable"
WRONG_MODE: Final = "wrong-mode"
EMPTY_ROSTER: Final = "empty-roster"
NO_REPORTS: Final = "no-reports"
NO_EVIDENCE: Final = "no-evidence"

SUPPORTED_VERSION: Final = 1
COLLECT_MODE: Final = "collect"
RUN_MODE: Final = "run"
RAN: Final = "ran"
SKIPPED: Final = "skipped"
NOT_COLLECTED: Final = "not-collected"

REMEDY: Final = (
    "A test carrying @pytest.mark.must_run claims it executes somewhere in CI. Either make "
    "a lane run it, or take the marker off and say in docs/testing-strategy.md why this "
    "test is allowed to skip everywhere. Do not widen the gate to accept a skip.\n"
)


@dataclass(frozen=True, slots=True)
class Finding:
    """One way the must-run claim is not backed by evidence."""

    rule: str
    subject: str
    detail: str

    def __str__(self) -> str:
        """Return the one-line report form."""
        return f"[{self.rule}] {self.subject}: {self.detail}"


@dataclass(frozen=True, slots=True)
class Report:
    """One pytest invocation's account of the marked tests it saw."""

    label: str
    platform: str
    mode: str
    collected: frozenset[str]
    outcomes: dict[str, str]

    def verdict(self, nodeid: str) -> str:
        """Return what this invocation said about one test."""
        return self.outcomes.get(nodeid, NOT_COLLECTED)


def _strings(raw: object) -> list[str]:
    """Return the string members of a value that should have been a list of them."""
    return [item for item in raw if isinstance(item, str)] if isinstance(raw, list) else []


def load_report(path: Path) -> Report | str:
    """Return the report at ``path``, or the reason it cannot be read as one."""
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except OSError as error:
        return f"cannot be read ({error.strerror or error})"
    except ValueError as error:
        return f"is not JSON ({error})"
    if not isinstance(document, dict):
        return "is JSON but not an object"
    version = document.get("report_version")
    if version != SUPPORTED_VERSION:
        return (
            f"declares report_version {version!r}, and this gate reads "
            f"{SUPPORTED_VERSION}; the writer in tests/conftest.py and this reader "
            "have drifted apart"
        )
    raw_outcomes = document.get("outcomes")
    outcomes = (
        {key: value for key, value in raw_outcomes.items() if isinstance(value, str)}
        if isinstance(raw_outcomes, dict)
        else {}
    )
    mode = document.get("mode")
    return Report(
        label=str(document.get("label") or path.stem),
        platform=str(document.get("platform") or "unknown"),
        mode=mode if isinstance(mode, str) else "unknown",
        collected=frozenset(_strings(document.get("collected"))),
        outcomes=outcomes,
    )


def _roster(path: Path) -> tuple[frozenset[str], list[Finding]]:
    """Return the node ids claiming the marker, or the reason the roster cannot be used."""
    if not path.is_file():
        return frozenset(), [
            Finding(
                MISSING_ROSTER,
                path.as_posix(),
                "no roster document; the gate never learned which tests claim the marker, "
                "so it cannot have checked any of them",
            )
        ]
    loaded = load_report(path)
    if isinstance(loaded, str):
        return frozenset(), [Finding(UNREADABLE, path.as_posix(), f"the roster {loaded}")]
    if loaded.mode != COLLECT_MODE:
        return frozenset(), [
            Finding(
                WRONG_MODE,
                path.as_posix(),
                f"the roster was written by a {loaded.mode!r} invocation; it must come from "
                "`pytest -m must_run --collect-only`, which alone sees every lane's tests",
            )
        ]
    if not loaded.collected:
        return frozenset(), [
            Finding(
                EMPTY_ROSTER,
                path.as_posix(),
                "no test carries @pytest.mark.must_run. The marker is registered and unused, "
                "so this gate passes without examining anything",
            )
        ]
    return loaded.collected, []


def _lanes(paths: Sequence[Path]) -> tuple[list[Report], list[Finding]]:
    """Return the usable lane reports among ``paths``, with a finding for each that is not."""
    lanes: list[Report] = []
    findings: list[Finding] = []
    for path in sorted(paths):
        loaded = load_report(path)
        if isinstance(loaded, str):
            findings.append(Finding(UNREADABLE, path.as_posix(), f"this report {loaded}"))
            continue
        if loaded.mode != RUN_MODE:
            findings.append(
                Finding(
                    WRONG_MODE,
                    path.as_posix(),
                    f"written by a {loaded.mode!r} invocation, so it records what was "
                    "collected and not what executed; a collect-only document is not "
                    "evidence that anything ran",
                )
            )
            continue
        lanes.append(loaded)
    return lanes, findings


def reports_under(root: Path) -> list[Path]:
    """Return the JSON documents at or under ``root``, in a stable order."""
    if root.is_file():
        return [root]
    if not root.is_dir():
        return []
    return sorted(path for path in root.rglob("*.json"))


def check(roster_path: Path, report_roots: Sequence[Path]) -> list[Finding]:
    """Return every finding, the vacuity guard's own included."""
    expected, refused = _roster(roster_path)
    if refused:
        return refused

    paths = [path for root in report_roots for path in reports_under(root)]
    where = ", ".join(root.as_posix() for root in report_roots) or "(no path given)"
    if not paths:
        return [
            Finding(
                NO_REPORTS,
                where,
                f"{len(expected)} tests claim the marker and no lane report was found; the "
                "artifacts did not arrive, so nothing is known about any of them",
            )
        ]

    lanes, findings = _lanes(paths)
    if not lanes:
        findings.append(
            Finding(
                NO_REPORTS,
                where,
                f"{len(paths)} documents were found and none of them is a lane report, so "
                f"the {len(expected)} marked tests are unaccounted for",
            )
        )
        return findings
    if not any(lane.outcomes for lane in lanes):
        findings.append(
            Finding(
                NO_EVIDENCE,
                ", ".join(lane.label for lane in lanes),
                f"{len(lanes)} lane reports between them record not one outcome; either no "
                "lane collected a marked test, or the reporting plugin in tests/conftest.py "
                "is no longer writing them",
            )
        )
        return findings

    for lane in lanes:
        for nodeid in sorted(set(lane.outcomes) - expected):
            findings.append(
                Finding(
                    UNKNOWN_TEST,
                    nodeid,
                    f"lane {lane.label!r} reports a test the roster does not contain, so the "
                    "two were produced from different trees and neither can vouch for the "
                    "other",
                )
            )

    for nodeid in sorted(expected):
        verdicts = {lane.label: lane.verdict(nodeid) for lane in lanes}
        if RAN in verdicts.values():
            continue
        findings.append(
            Finding(
                RAN_NOWHERE,
                nodeid,
                "marked must_run and executed in no lane - "
                + "; ".join(f"{label}: {verdict}" for label, verdict in sorted(verdicts.items())),
            )
        )
    return findings


def _report(write: Callable[[str], int], heading: str, findings: list[Finding]) -> None:
    """Write one headed block of findings, or nothing when the block is empty."""
    if not findings:
        return
    write(f"{heading}\n")
    for finding in findings:
        write(f"  {finding}\n")
    write("\n")


def main(argv: list[str] | None = None) -> int:
    """Run the gate over one roster and the lane reports CI downloaded."""
    parser = argparse.ArgumentParser(
        description="Check that every @pytest.mark.must_run test ran in at least one CI lane."
    )
    parser.add_argument(
        "reports",
        type=Path,
        nargs="*",
        default=[Path("must-run-reports")],
        help="lane report files, or directories holding them",
    )
    parser.add_argument(
        "--roster",
        type=Path,
        required=True,
        help="the --collect-only document listing every test that claims the marker",
    )
    arguments = parser.parse_args(argv)

    findings = check(arguments.roster, arguments.reports or [Path("must-run-reports")])
    _report(sys.stderr.write, "must-run claims this build cannot back:", findings)
    if findings:
        sys.stderr.write(REMEDY)
        return 1
    sys.stdout.write("every must_run test ran in at least one lane\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
