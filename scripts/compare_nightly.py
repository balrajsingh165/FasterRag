"""Compare tonight's benchmark document with the previous night's.

``.github/workflows/nightly.yml`` runs the benchmark suite and uploads the result, which
made a night-over-night move *recordable* and left it *undetected*: nobody downloads thirty
JSON files to diff them by eye. ``docs/performance.md`` §3 described a regression check that
no step performed, which is the defect TASK-0246 names.

This is the retrieval regression gate's rule set (``src/fasterrag/services/regression.py``)
applied to latency and throughput rather than to recall, and the vocabulary is deliberately
the same rather than a second, inconsistent notion of "regression":

* **A tolerance, not an equality.** Only a move *worse* than ``--tolerance`` counts.
  Improvements never fail, exactly as ``check_regression`` never blocks on a rise.
* **A baseline is only comparable to a run that could have produced it.** That gate refuses
  a baseline recorded under a different embedding model, because comparing across it
  measures the model rather than the change. The analogue here is the machine and the
  dataset: a number taken on a different runner, or over a different corpus, measures the
  runner or the corpus.
* **Nothing compared is never a pass.** That gate blocks on a missing baseline. A nightly
  job cannot block the first time it ever runs, so rather than collapse the case into a
  green tick this reports a third verdict, ``not-comparable``, with its own exit code. A
  gate that reports green because it compared nothing is worse than no gate.

What this deliberately does NOT treat as incomparable is a change in the fasterRag or Python
version recorded in the fingerprint: the code changing is the thing being watched for, so
refusing to compare across it would disable the check on exactly the nights it matters.

The numbers compared here are not ledger-citable, and comparing them does not make them so
— a shared GitHub runner has neighbours, and ``docs/performance.md`` requires an isolated,
documented machine. A large night-over-night move is a signal to go and measure on real
hardware, which is the whole of what this claims to be.

Exit codes: ``0`` within tolerance, ``1`` regressed, ``2`` not comparable. ``2`` fails the
nightly job as of 2026-11-08: it was mapped to a warning there while ``ci.yml`` exited 1 on
the same verdict, and a benchmark that had run nothing since August reported success 95
times out of 95 behind that warning (TASK-0314). The script never collapsed the case; its
caller did.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any, Final

__all__ = [
    "EXIT_CODES",
    "MACHINE_FIELDS",
    "TOLERANCE",
    "TRACKED_METRICS",
    "ComparisonReport",
    "MetricComparison",
    "Verdict",
    "compare",
    "compare_files",
    "load_document",
    "machine_mismatch",
    "main",
]

TOLERANCE: Final = 0.10

# CRITICAL: these are the fingerprint fields that describe the *machine*. The fingerprint
# also carries `python` and `fasterrag`, and their absence here is deliberate: a version
# change is the change under test, so refusing to compare across one would switch the check
# off on exactly the nights something moved.
MACHINE_FIELDS: Final[tuple[str, ...]] = ("cpu", "cores", "ram_gb", "gpu", "os")

TRACKED_METRICS: Final[tuple[tuple[str, bool], ...]] = (
    ("p50_ms", True),
    ("p95_ms", True),
    ("p99_ms", True),
    ("throughput_per_sec", False),
)


class Verdict(StrEnum):
    """What a comparison concluded, keeping "nothing was compared" distinct."""

    WITHIN_TOLERANCE = "within-tolerance"
    REGRESSED = "regressed"
    NOT_COMPARABLE = "not-comparable"


EXIT_CODES: Final[dict[Verdict, int]] = {
    Verdict.WITHIN_TOLERANCE: 0,
    Verdict.REGRESSED: 1,
    Verdict.NOT_COMPARABLE: 2,
}


@dataclass(frozen=True, slots=True)
class MetricComparison:
    """One tracked metric of one suite, judged against the previous run."""

    suite: str
    metric: str
    before: float | None
    after: float | None
    change: float | None
    verdict: Verdict
    reason: str

    def as_dict(self) -> dict[str, Any]:
        """Return the machine-readable form."""
        return {
            "suite": self.suite,
            "metric": self.metric,
            "before": self.before,
            "after": self.after,
            "change": round(self.change, 6) if self.change is not None else None,
            "verdict": self.verdict.value,
            "reason": self.reason,
        }


@dataclass(frozen=True, slots=True)
class ComparisonReport:
    """The verdict for one night, and the evidence behind it."""

    verdict: Verdict
    tolerance: float
    comparisons: list[MetricComparison] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    @property
    def regressions(self) -> list[MetricComparison]:
        """Return the metrics that moved beyond tolerance."""
        return [entry for entry in self.comparisons if entry.verdict is Verdict.REGRESSED]

    @property
    def obstacles(self) -> list[str]:
        """Return every reason a comparison could not be made."""
        skipped = [
            entry.reason for entry in self.comparisons if entry.verdict is Verdict.NOT_COMPARABLE
        ]
        return [*self.notes, *skipped]

    def as_dict(self) -> dict[str, Any]:
        """Return the machine-readable report the artifact carries."""
        checked = [
            entry for entry in self.comparisons if entry.verdict is not Verdict.NOT_COMPARABLE
        ]
        return {
            "verdict": self.verdict.value,
            "tolerance": self.tolerance,
            "compared": len(checked),
            "comparisons": [entry.as_dict() for entry in self.comparisons],
            "notes": self.notes,
        }

    def summary(self) -> str:
        """Return the one-line verdict a log line and a job annotation carry."""
        if self.verdict is Verdict.REGRESSED:
            reasons = "; ".join(entry.reason for entry in self.regressions)
            return f"nightly benchmark REGRESSED: {reasons}"
        if self.verdict is Verdict.NOT_COMPARABLE:
            reasons = "; ".join(self.obstacles) or "nothing was compared"
            return f"nightly benchmark NOT COMPARABLE, no regression ruled out: {reasons}"
        return (
            f"nightly benchmark within tolerance: {len(self.comparisons)} metrics compared, "
            f"none worse than {self.tolerance:.1%}"
        )

    def markdown(self) -> str:
        """Return the job-summary section, verdict first."""
        rows = [
            f"| `{entry.suite}` | `{entry.metric}` | {_cell(entry.before)} "
            f"| {_cell(entry.after)} | {_change(entry.change)} | {entry.verdict.value} |"
            for entry in self.comparisons
        ]
        table = (
            [
                "",
                "| Suite | Metric | Previous | Tonight | Change (worse is +) | Verdict |",
                "|---|---|---|---|---|---|",
                *rows,
            ]
            if rows
            else []
        )
        obstacles = ["", *(f"- {reason}" for reason in self.obstacles)] if self.obstacles else []
        return "\n".join(
            [
                "### Nightly benchmark comparison",
                "",
                self.summary(),
                "",
                "A regression signal only: a shared runner has neighbours, so these numbers "
                "are never citable in `docs/benchmarks.md`.",
                *table,
                *obstacles,
                "",
            ]
        )


def _cell(value: float | None) -> str:
    """Render a metric value for the summary table."""
    return "n/a" if value is None else f"{value:g}"


def _change(value: float | None) -> str:
    """Render a proportional change for the summary table."""
    return "n/a" if value is None else f"{value:+.1%}"


def _mapping(value: Any) -> dict[str, Any]:
    """Return ``value`` when it is a JSON object, an empty mapping otherwise."""
    return value if isinstance(value, dict) else {}


def _number(value: Any) -> float | None:
    """Return ``value`` as a float when it is a real number, None otherwise."""
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    return float(value)


def load_document(path: Path) -> tuple[dict[str, Any] | None, str]:
    """Read a ``fasterrag benchmark --json`` document, or say why it cannot be read.

    The benchmark step carries ``continue-on-error``, so an empty or truncated file still
    arrives here and has to be reported rather than crash the comparison.
    """
    if not path.is_file():
        return None, f"{path} does not exist, so there is nothing to compare against"

    try:
        payload: Any = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        return None, f"{path} is not a readable benchmark document ({exc})"

    if not isinstance(payload, dict):
        return None, f"{path} holds a JSON {type(payload).__name__}, not a benchmark document"
    return payload, ""


def _suites(document: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Index a document's suite results by suite name."""
    raw = document.get("suites")
    entries: list[Any] = raw if isinstance(raw, list) else []

    indexed: dict[str, dict[str, Any]] = {}
    for entry in entries:
        result = _mapping(entry)
        name = result.get("suite")
        if isinstance(name, str) and name:
            indexed[name] = result
    return indexed


def machine_mismatch(previous: dict[str, Any], current: dict[str, Any]) -> str:
    """Return why two runs' machines are not comparable, empty when they are."""
    before = _mapping(previous.get("hardware"))
    after = _mapping(current.get("hardware"))
    if not before or not after:
        return (
            "one of the two runs recorded no hardware fingerprint, so there is no evidence "
            "the two numbers came from the same machine"
        )
    differing = [
        f"{name} {before.get(name)!r} became {after.get(name)!r}"
        for name in MACHINE_FIELDS
        if before.get(name) != after.get(name)
    ]
    if not differing:
        return ""
    joined = "; ".join(differing)
    return (
        "the two runs were measured on different machines, which measures the machine rather "
        f"than the change ({joined})"
    )


def _compare_metric(
    suite: str,
    metric: str,
    before: float | None,
    after: float | None,
    *,
    lower_is_better: bool,
    tolerance: float,
) -> MetricComparison:
    """Judge one metric, treating an absent or zero baseline as not comparable."""
    if before is None or after is None:
        if before is None and after is None:
            missing = "both runs"
        else:
            missing = "the previous run" if before is None else "tonight's run"
        return MetricComparison(
            suite=suite,
            metric=metric,
            before=before,
            after=after,
            change=None,
            verdict=Verdict.NOT_COMPARABLE,
            reason=f"{suite}.{metric} is absent from {missing}, so it was not checked",
        )

    if before <= 0:
        return MetricComparison(
            suite=suite,
            metric=metric,
            before=before,
            after=after,
            change=None,
            verdict=Verdict.NOT_COMPARABLE,
            reason=(
                f"{suite}.{metric} was {before:g} in the previous run, and a proportional "
                "change against zero says nothing"
            ),
        )

    change = (after - before) / before if lower_is_better else (before - after) / before
    if change > tolerance:
        return MetricComparison(
            suite=suite,
            metric=metric,
            before=before,
            after=after,
            change=change,
            verdict=Verdict.REGRESSED,
            reason=(
                f"{suite}.{metric} moved from {before:g} to {after:g}, {change:.1%} worse, "
                f"beyond the {tolerance:.1%} tolerance"
            ),
        )

    return MetricComparison(
        suite=suite,
        metric=metric,
        before=before,
        after=after,
        change=change,
        verdict=Verdict.WITHIN_TOLERANCE,
        reason=f"{suite}.{metric} moved from {before:g} to {after:g} ({change:+.1%})",
    )


def _compare_suite(
    suite: str, previous: dict[str, Any], current: dict[str, Any], tolerance: float
) -> list[MetricComparison]:
    """Compare one suite's tracked metrics, warmed medians only.

    Cold-start figures are excluded deliberately: the methodology reports warm and cold
    separately, and a cold start is dominated by a model load that varies by minutes.
    """
    before = _mapping(previous.get("median"))
    after = _mapping(current.get("median"))
    return [
        _compare_metric(
            suite,
            metric,
            _number(before.get(metric)),
            _number(after.get(metric)),
            lower_is_better=lower_is_better,
            tolerance=tolerance,
        )
        for metric, lower_is_better in TRACKED_METRICS
    ]


def _verdict(comparisons: list[MetricComparison], notes: list[str]) -> Verdict:
    """Decide the night's verdict; anything left unchecked is never within tolerance."""
    if any(entry.verdict is Verdict.REGRESSED for entry in comparisons):
        return Verdict.REGRESSED
    if notes or not comparisons:
        return Verdict.NOT_COMPARABLE
    if any(entry.verdict is Verdict.NOT_COMPARABLE for entry in comparisons):
        return Verdict.NOT_COMPARABLE
    return Verdict.WITHIN_TOLERANCE


def compare(
    previous: dict[str, Any], current: dict[str, Any], *, tolerance: float = TOLERANCE
) -> ComparisonReport:
    """Compare every tracked metric of every suite the two documents share."""
    mismatch = machine_mismatch(previous, current)
    if mismatch:
        return ComparisonReport(Verdict.NOT_COMPARABLE, tolerance, [], [mismatch])

    before_suites = _suites(previous)
    after_suites = _suites(current)

    notes: list[str] = []
    if not after_suites:
        notes.append("tonight's run recorded no suite results, so the benchmark did not complete")
    if not before_suites:
        notes.append("the previous run recorded no suite results, so there is no baseline")

    comparisons: list[MetricComparison] = []
    for name in sorted(after_suites):
        if name not in before_suites:
            notes.append(f"the {name} suite has no counterpart in the previous run")
            continue
        was = before_suites[name].get("dataset")
        now = after_suites[name].get("dataset")
        if was != now:
            notes.append(
                f"the {name} suite measured {was!r} previously and {now!r} tonight, which "
                "compares corpora rather than runs"
            )
            continue
        comparisons.extend(_compare_suite(name, before_suites[name], after_suites[name], tolerance))

    notes.extend(
        f"the {name} suite ran previously but not tonight"
        for name in sorted(set(before_suites) - set(after_suites))
    )

    return ComparisonReport(_verdict(comparisons, notes), tolerance, comparisons, notes)


def compare_files(
    previous: Path, current: Path, *, tolerance: float = TOLERANCE
) -> ComparisonReport:
    """Load both documents and compare them; an unreadable one is not comparable."""
    current_document, current_problem = load_document(current)
    previous_document, previous_problem = load_document(previous)

    if current_document is None or previous_document is None:
        problems = [problem for problem in (previous_problem, current_problem) if problem]
        return ComparisonReport(Verdict.NOT_COMPARABLE, tolerance, [], problems)

    return compare(previous_document, current_document, tolerance=tolerance)


def _tolerance(value: str) -> float:
    """Parse ``--tolerance`` as a fraction, refusing a negative one."""
    parsed = float(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError(
            "a negative tolerance would fail a run that improved; use 0 for an exact gate"
        )
    return parsed


def _write(path: Path, body: str, *, append: bool) -> None:
    """Write a report file, reporting a write failure without altering the verdict."""
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a" if append else "w", encoding="utf-8") as handle:
            handle.write(body)
    except OSError as exc:
        sys.stderr.write(f"could not write {path}: {exc}\n")


def main(argv: list[str] | None = None) -> int:
    """Compare two benchmark documents and exit with the verdict's code."""
    parser = argparse.ArgumentParser(
        description="Compare two `fasterrag benchmark --json` documents for a regression."
    )
    parser.add_argument("--current", required=True, type=Path, help="tonight's benchmark.json")
    parser.add_argument(
        "--previous", required=True, type=Path, help="the previous run's benchmark.json"
    )
    parser.add_argument(
        "--tolerance",
        type=_tolerance,
        default=TOLERANCE,
        help="proportional worsening allowed per metric (default 0.10, i.e. 10%%)",
    )
    parser.add_argument("--json-out", type=Path, default=None, help="write the report as JSON")
    parser.add_argument(
        "--summary", type=Path, default=None, help="append a markdown summary to this file"
    )
    args = parser.parse_args(argv)

    report = compare_files(args.previous, args.current, tolerance=args.tolerance)

    sys.stdout.write(f"{report.summary()}\n")
    for entry in report.comparisons:
        sys.stdout.write(f"  [{entry.verdict.value}] {entry.reason}\n")
    for note in report.notes:
        sys.stdout.write(f"  [not-comparable] {note}\n")

    if args.json_out is not None:
        payload = json.dumps(report.as_dict(), indent=2, sort_keys=True)
        _write(args.json_out, f"{payload}\n", append=False)
    if args.summary is not None:
        _write(args.summary, report.markdown(), append=True)

    return EXIT_CODES[report.verdict]


if __name__ == "__main__":
    raise SystemExit(main())
