"""The load harness's dependency-free core: configuration, parsing, SLIs, and the guards.

Nothing here imports Locust, httpx, or fasterRag. That is deliberate three times over. It
keeps the one module that *does* touch Locust (``locustfile.py``) thin enough to read in a
sitting; it means the guards below are unit-tested in the ordinary CI lane by
``test_load_harness.py``, on a machine with no load generator installed; and it keeps the
pieces that decide whether a run counts separate from the pieces that generate traffic, so
the decision cannot be quietly influenced by the generator.

**Why there is so much machinery around refusing.** A load harness has one failure mode that
matters more than every other combined: reporting a number when it measured nothing. A dead
endpoint, a dataset with no queries, a rate limiter shaping the whole run, a percentile
computed from an empty sample — each of these produces a tidy report full of zeros that
looks exactly like a fast system. This repository calls that the T12 vacuity class and
treats it as a bug, and the eval regression gate already applies the same rule: a gate that
goes green having compared nothing is worse than no gate.

So the type system carries the distinction. An SLI's value is ``float | None`` and ``None``
means *not measured*; there is no path that turns an absent measurement into ``0.0``.
:func:`EndpointStats.percentile` returns ``None`` on an empty sample rather than the ``0``
Locust's own percentile helper returns. Every refusal maps to a non-zero exit code, so a
caller who reads nothing but ``$?`` still cannot mistake an empty run for a fast one.

**Why the methodology lives in code.** ``docs/performance.md`` §3 fixes four rules that a
run either followed or did not: at least 1,000 samples per latency figure, three warmed
repetitions with the median reported, cold and warmed kept apart, and no co-tenant load.
Three of the four are mechanical and are enforced here —
:data:`MINIMUM_SAMPLES`, :data:`REPETITIONS` and the separate cold run. The fourth cannot be
detected from inside the process: no program can tell whether the machine under test was
busy with something else. It is therefore an assertion the operator makes with
``--isolated``, and a report without it stamps its own ledger draft ``NOT CITABLE``. A run
on a developer laptop cannot accidentally become a benchmark entry.

**Percentile provenance.** Locust records response times into a histogram whose keys are
rounded — exact below 100 ms, two significant figures above — and the percentiles here are
computed from that histogram with the same nearest-rank walk Locust uses, so the numbers
match what ``locust --csv`` would print. The rounding is reported alongside the figures
rather than hidden, because a p95 quoted to the millisecond from a bucket rounded to the
nearest 10 ms is a fabricated precision.
"""

from __future__ import annotations

import json
import re
import statistics
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from enum import IntEnum
from pathlib import Path
from typing import Any, Final, Literal

__all__ = [
    "API_KEY_VARIABLE",
    "ENVIRONMENT_PREFIX",
    "MINIMUM_SAMPLES",
    "REPETITIONS",
    "EndpointStats",
    "ExitCode",
    "LoadConfig",
    "LoadHarnessError",
    "LoadReport",
    "Problem",
    "RunStats",
    "Scenario",
    "Series",
    "Sli",
    "audit",
    "config_from_env",
    "derive_slis",
    "exit_code_for",
    "load_corpus",
    "load_queries",
    "median_slis",
    "parse_exposition",
    "parse_locust_stats",
    "series_total",
]

# CRITICAL: docs/performance.md §3 rule 4 — "minimum 1,000 samples per latency figure". A
# run below it is reported as insufficient rather than as a fast p95, because the whole
# point of the rule is that a p95 over 40 samples is not a p95.
MINIMUM_SAMPLES: Final = 1000

# CRITICAL: docs/performance.md §3 rule 5 and services/benchmark.REPETITIONS both fix this
# at three. A ledger draft produced under a different count is not comparable with the
# entries already committed, so the draft records the count it actually ran.
REPETITIONS: Final = 3

Scenario = Literal["query", "ingest", "mixed"]
SCENARIOS: Final[tuple[Scenario, ...]] = ("query", "ingest", "mixed")

RETRIEVE_ENDPOINT: Final = "/v1/retrieve"
QUERY_ENDPOINT: Final = "/v1/query"
STREAM_ENDPOINT: Final = "/v1/query [stream]"
INGEST_ENDPOINT: Final = "/v1/ingest"
JOB_ENDPOINT: Final = "/v1/ingest/{job_id}"

RETRIEVAL_SLI_ENTRY: Final = "SLI retrieval_ms (server timings_ms)"
TTFT_SLI_ENTRY: Final = "SLI ttft_ms (first token event)"
STATUS_2XX_ENTRY: Final = "SLI status 2xx"
STATUS_4XX_ENTRY: Final = "SLI status 4xx"
STATUS_429_ENTRY: Final = "SLI status 429"
STATUS_5XX_ENTRY: Final = "SLI status 5xx"
TRANSPORT_ERROR_ENTRY: Final = "SLI transport error"

QUERY_ENTRIES: Final = (RETRIEVE_ENDPOINT, QUERY_ENDPOINT, STREAM_ENDPOINT)
INGEST_ENTRIES: Final = (INGEST_ENDPOINT, JOB_ENDPOINT)

# CRITICAL: slo.md defines ingestion throughput over a ten-minute sustained window. A
# shorter repetition measures a burst, which is a different quantity, so the SLI is reported
# not-measured below this duration rather than reported from too short a window.
THROUGHPUT_WINDOW_S: Final = 600

# A run in which the rate limiter rejected this share of requests measured the limiter
# rather than the system, so it is refused instead of reported.
MAXIMUM_THROTTLED_RATIO: Final = 0.05

_LABEL_PATTERN: Final = re.compile(r'(\w+)="((?:[^"\\]|\\.)*)"')
_GOLDEN_SET_NAME: Final = "golden.jsonl"


class ExitCode(IntEnum):
    """Process exit codes. Every non-zero value means this run is not a measurement."""

    OK = 0
    USAGE = 1
    PREFLIGHT_FAILED = 2
    VACUOUS = 3
    TOO_MANY_FAILURES = 4
    TOOL_MISSING = 5


# CRITICAL: order matters. A run that produced no usable sample is reported as vacuous even
# when it also failed requests, because "nothing was measured" is the finding an operator
# must act on first; swapping these hides an empty run behind an error rate.
_EXIT_PRECEDENCE: Final = (
    ExitCode.TOOL_MISSING,
    ExitCode.PREFLIGHT_FAILED,
    ExitCode.USAGE,
    ExitCode.VACUOUS,
    ExitCode.TOO_MANY_FAILURES,
)


class LoadHarnessError(Exception):
    """A load run that cannot produce a valid sample, carrying the exit code it earns."""

    def __init__(self, detail: str, exit_code: ExitCode) -> None:
        """Record the operator-facing detail and the exit code this failure maps to."""
        super().__init__(detail)
        self.detail = detail
        self.exit_code = exit_code


@dataclass(frozen=True, slots=True)
class LoadConfig:
    """One load run's parameters, every one of which changes what the numbers mean."""

    target: str
    scenario: Scenario
    users: int
    spawn_rate: float
    duration_s: int
    dataset: Path
    sut_hardware: str
    warmup_s: int = 60
    repetitions: int = REPETITIONS
    collection: str | None = None
    tenant: str | None = None
    stream_weight: int = 1
    isolated: bool = False
    min_samples: int = MINIMUM_SAMPLES
    max_failure_ratio: float = 0.01

    @property
    def required_entries(self) -> tuple[str, ...]:
        """Return the stat entries this scenario must have produced to be a valid run."""
        reads = tuple(
            name for name in QUERY_ENTRIES if name != STREAM_ENDPOINT or self.stream_weight > 0
        )
        if self.scenario == "query":
            return reads
        if self.scenario == "ingest":
            return INGEST_ENTRIES
        return reads + INGEST_ENTRIES

    def as_dict(self) -> dict[str, Any]:
        """Return the parameter echo that every report and ledger draft carries."""
        return {
            "target": self.target,
            "scenario": self.scenario,
            "users": self.users,
            "spawn_rate": self.spawn_rate,
            "duration_s": self.duration_s,
            "warmup_s": self.warmup_s,
            "repetitions": self.repetitions,
            "dataset": str(self.dataset),
            "collection": self.collection,
            "tenant": self.tenant,
            "sut_hardware": self.sut_hardware,
            "isolated": self.isolated,
            "min_samples": self.min_samples,
            "max_failure_ratio": self.max_failure_ratio,
        }


ENVIRONMENT_PREFIX: Final = "FASTERRAG_LOAD_"
API_KEY_VARIABLE: Final = "FASTERRAG_LOAD_API_KEY"


def _required(environ: Mapping[str, str], name: str) -> str:
    """Read one mandatory environment variable, naming it when it is missing."""
    value = environ.get(f"{ENVIRONMENT_PREFIX}{name}", "").strip()
    if not value:
        raise LoadHarnessError(
            f"{ENVIRONMENT_PREFIX}{name} is unset. The locustfile is configured by "
            "tests/load/run_load.py; running `locust -f` directly means setting these "
            "variables by hand",
            ExitCode.USAGE,
        )
    return value


def config_from_env(environ: Mapping[str, str]) -> LoadConfig:
    """Rebuild the run's configuration inside the Locust process, refusing a partial one."""
    scenario = _required(environ, "SCENARIO")
    if scenario not in SCENARIOS:
        raise LoadHarnessError(
            f"{ENVIRONMENT_PREFIX}SCENARIO is {scenario!r}; expected one of {', '.join(SCENARIOS)}",
            ExitCode.USAGE,
        )
    optional = environ.get(f"{ENVIRONMENT_PREFIX}COLLECTION", "").strip()
    tenant = environ.get(f"{ENVIRONMENT_PREFIX}TENANT", "").strip()
    return LoadConfig(
        target=_required(environ, "TARGET"),
        scenario=scenario,
        users=int(_required(environ, "USERS")),
        spawn_rate=float(_required(environ, "SPAWN_RATE")),
        duration_s=int(_required(environ, "DURATION")),
        dataset=Path(_required(environ, "DATASET")),
        sut_hardware=_required(environ, "SUT_HARDWARE"),
        collection=optional or None,
        tenant=tenant or None,
        stream_weight=int(environ.get(f"{ENVIRONMENT_PREFIX}STREAM_WEIGHT", "1")),
        min_samples=int(environ.get(f"{ENVIRONMENT_PREFIX}MIN_SAMPLES", str(MINIMUM_SAMPLES))),
        max_failure_ratio=float(environ.get(f"{ENVIRONMENT_PREFIX}MAX_FAILURE_RATIO", "0.01")),
    )


def load_queries(dataset: Path) -> tuple[str, ...]:
    """Return a dataset's queries, refusing a set that holds none."""
    path = dataset if dataset.is_file() else dataset / _GOLDEN_SET_NAME
    if not path.is_file():
        raise LoadHarnessError(
            f"no golden set at {path}: --dataset must name a directory holding "
            f"{_GOLDEN_SET_NAME}, or that file directly",
            ExitCode.USAGE,
        )

    queries: list[str] = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise LoadHarnessError(
                f"{path} line {number} is not JSON: {exc}", ExitCode.USAGE
            ) from exc
        if not isinstance(record, dict):
            raise LoadHarnessError(f"{path} line {number} is not an object", ExitCode.USAGE)
        text = str(record.get("query", "")).strip()
        if text:
            queries.append(text)

    if not queries:
        raise LoadHarnessError(
            f"{path} yielded no non-empty `query` field; a load run against an empty query "
            "set would report latencies for requests it never sent",
            ExitCode.USAGE,
        )
    return tuple(queries)


def load_corpus(dataset: Path) -> tuple[Path, ...]:
    """Return a dataset's corpus files, which the ingest scenarios replay."""
    root = dataset / "corpus"
    if not root.is_dir():
        raise LoadHarnessError(
            f"no corpus directory at {root}; the ingest and mixed scenarios need documents "
            "to ingest and will not invent them",
            ExitCode.USAGE,
        )
    documents = tuple(sorted(path for path in root.rglob("*") if path.is_file()))
    if not documents:
        raise LoadHarnessError(f"{root} holds no files", ExitCode.USAGE)
    return documents


@dataclass(frozen=True, slots=True)
class Series:
    """One Prometheus exposition line: a metric name, its labels, and its value."""

    name: str
    labels: tuple[tuple[str, str], ...]
    value: float

    def matches(self, name: str, labels: Mapping[str, str]) -> bool:
        """Return whether this series has ``name`` and carries every label in ``labels``."""
        if self.name != name:
            return False
        own = dict(self.labels)
        return all(own.get(key) == value for key, value in labels.items())


def parse_exposition(text: str) -> tuple[Series, ...]:
    """Parse a Prometheus text exposition body, refusing one that holds no series."""
    parsed: list[Series] = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        head, _, raw_value = stripped.rpartition(" ")
        if not head:
            continue
        try:
            value = float(raw_value)
        except ValueError:
            continue
        name, _, label_part = head.partition("{")
        labels = tuple(_LABEL_PATTERN.findall(label_part)) if label_part else ()
        parsed.append(Series(name=name.strip(), labels=labels, value=value))

    if not parsed:
        raise LoadHarnessError(
            "the target's /metrics body parsed to zero series, so no metric-derived SLI "
            "could be read from it",
            ExitCode.PREFLIGHT_FAILED,
        )
    return tuple(parsed)


def series_total(series: Iterable[Series], name: str, **labels: str) -> float:
    """Return the summed value of every series matching ``name`` and ``labels``."""
    return sum(item.value for item in series if item.matches(name, labels))


def _counter_delta(
    before: Sequence[Series], after: Sequence[Series], name: str, **labels: str
) -> float:
    """Return a counter's increase across a run, or 0.0 when it did not move."""
    return max(series_total(after, name, **labels) - series_total(before, name, **labels), 0.0)


@dataclass(frozen=True, slots=True)
class EndpointStats:
    """One Locust stat entry: how many requests, how many failed, and their histogram."""

    name: str
    method: str
    num_requests: int
    num_failures: int
    response_times: Mapping[float, int]

    @property
    def failure_ratio(self) -> float:
        """Return the share of requests Locust marked failed, or 0.0 when none were sent."""
        return self.num_failures / self.num_requests if self.num_requests else 0.0

    def percentile(self, fraction: float) -> float | None:
        """Return a nearest-rank percentile in ms, or ``None`` when nothing was sampled."""
        if self.num_requests <= 0 or not self.response_times:
            return None
        wanted = int(self.num_requests * fraction)
        seen = 0
        for bucket in sorted(self.response_times, reverse=True):
            seen += self.response_times[bucket]
            if self.num_requests - seen <= wanted:
                return bucket
        return None


def parse_locust_stats(payload: object) -> tuple[EndpointStats, ...]:
    """Parse Locust's ``--json-file`` document, refusing one that recorded no entry."""
    if not isinstance(payload, list) or not payload:
        raise LoadHarnessError(
            "Locust wrote no stat entries: the run started and recorded nothing, which is "
            "an empty sample rather than a fast one",
            ExitCode.VACUOUS,
        )

    entries: list[EndpointStats] = []
    for item in payload:
        if not isinstance(item, dict) or "name" not in item:
            raise LoadHarnessError(
                f"unrecognised Locust stat entry {item!r}; the harness will not guess at a "
                "shape it does not know",
                ExitCode.VACUOUS,
            )
        raw_times = item.get("response_times") or {}
        times = {float(key): int(value) for key, value in dict(raw_times).items()}
        entries.append(
            EndpointStats(
                name=str(item["name"]),
                method=str(item.get("method") or ""),
                num_requests=int(item.get("num_requests") or 0),
                num_failures=int(item.get("num_failures") or 0),
                response_times=times,
            )
        )
    return tuple(entries)


@dataclass(frozen=True, slots=True)
class RunStats:
    """One repetition: its label, its duration, its Locust entries, and metric deltas."""

    label: str
    duration_s: float
    endpoints: tuple[EndpointStats, ...]
    metrics_before: tuple[Series, ...]
    metrics_after: tuple[Series, ...]

    def entry(self, name: str) -> EndpointStats | None:
        """Return the stat entry with ``name``, or ``None`` when the run produced none."""
        for item in self.endpoints:
            if item.name == name:
                return item
        return None

    def count(self, name: str) -> int:
        """Return how many requests a named entry recorded."""
        entry = self.entry(name)
        return entry.num_requests if entry else 0

    def delta(self, name: str, **labels: str) -> float:
        """Return how far a counter moved across this run."""
        return _counter_delta(self.metrics_before, self.metrics_after, name, **labels)


@dataclass(frozen=True, slots=True)
class Sli:
    """One service-level indicator. ``value is None`` means not measured, never zero."""

    key: str
    title: str
    source: str
    unit: str
    value: float | None = None
    note: str = ""

    @property
    def measured(self) -> bool:
        """Return whether this run produced a value for the indicator."""
        return self.value is not None

    def render(self) -> str:
        """Return the one-line report form, which never prints a number it does not have."""
        shown = "not measured" if self.value is None else f"{self.value:g} {self.unit}"
        suffix = f" — {self.note}" if self.note else ""
        return f"  {self.title}: {shown}{suffix}"

    def as_dict(self) -> dict[str, Any]:
        """Return the serialized form, preserving the measured/not-measured distinction."""
        return {
            "key": self.key,
            "title": self.title,
            "source": self.source,
            "unit": self.unit,
            "status": "measured" if self.measured else "not-measured",
            "value": self.value,
            "note": self.note,
        }


def _availability(run: RunStats) -> Sli:
    """Derive the query-availability SLI from the status-class entries."""
    successes = run.count(STATUS_2XX_ENTRY)
    server_errors = run.count(STATUS_5XX_ENTRY)
    transport = run.count(TRANSPORT_ERROR_ENTRY)
    considered = successes + server_errors + transport
    note = (
        "this run's window only; slo.md defines the SLI over rolling 30 days, so this is an "
        "input to the SLO and not the SLO itself. 4xx excluded per the definition"
    )
    if considered <= 0:
        return Sli(
            key="query_availability",
            title="Query availability",
            source="slo.md §1 row 1",
            unit="ratio",
            note="no 2xx, 5xx or transport outcome was recorded",
        )
    return Sli(
        key="query_availability",
        title="Query availability",
        source="slo.md §1 row 1",
        unit="ratio",
        value=round(successes / considered, 6),
        note=note,
    )


def _retrieval_latency(run: RunStats) -> Sli:
    """Derive p95 retrieval latency from the server-reported pre-generate stage sum."""
    entry = run.entry(RETRIEVAL_SLI_ENTRY)
    value = entry.percentile(0.95) if entry else None
    note = (
        "sum of the non-generate keys of the response's timings_ms. The shipped code emits "
        "retrieve and assemble only; the embed/fuse/rerank split slo.md names is not "
        "separately instrumented, so the total is measurable and the breakdown is not"
    )
    if value is None:
        return Sli(
            key="retrieval_p95",
            title="p95 retrieval latency (excl. generation)",
            source="slo.md §1 row 2",
            unit="ms",
            note="no query response carried a usable timings_ms",
        )
    return Sli(
        key="retrieval_p95",
        title="p95 retrieval latency (excl. generation)",
        source="slo.md §1 row 2",
        unit="ms",
        value=value,
        note=note,
    )


def _ttft(run: RunStats) -> Sli:
    """Derive p95 time-to-first-token from the streamed queries."""
    entry = run.entry(TTFT_SLI_ENTRY)
    value = entry.percentile(0.95) if entry else None
    if value is None:
        return Sli(
            key="ttft_p95",
            title="p95 time-to-first-token",
            source="slo.md §1 row 3",
            unit="ms",
            note="no streamed query reached a token event",
        )
    return Sli(
        key="ttft_p95",
        title="p95 time-to-first-token",
        source="slo.md §1 row 3",
        unit="ms",
        value=value,
        note="client-observed: request sent to first SSE `token` frame",
    )


def _ingest_throughput(run: RunStats) -> Sli:
    """Derive sustained ingestion throughput, refusing a window shorter than the SLI's."""
    title = "Ingestion throughput"
    source = "slo.md §1 row 4"
    if run.duration_s < THROUGHPUT_WINDOW_S:
        return Sli(
            key="ingest_throughput",
            title=title,
            source=source,
            unit="docs/min",
            note=(
                f"window was {run.duration_s:.0f}s; the SLI is defined over "
                f"{THROUGHPUT_WINDOW_S}s sustained, and a shorter window measures a burst"
            ),
        )
    documents = run.delta("fasterrag_ingest_documents_total", status="succeeded")
    if documents <= 0:
        return Sli(
            key="ingest_throughput",
            title=title,
            source=source,
            unit="docs/min",
            note="fasterrag_ingest_documents_total{status=succeeded} did not move",
        )
    return Sli(
        key="ingest_throughput",
        title=title,
        source=source,
        unit="docs/min",
        value=round(documents / run.duration_s * 60, 3),
        note=(
            "docs/min only. slo.md also names tokens/sec; fasterrag_ingest_throughput "
            "carries documents_per_second and chunks_per_second and no token unit, so "
            "tokens/sec cannot be read from the shipped metrics"
        ),
    )


def _error_rate(run: RunStats) -> Sli:
    """Derive the 5xx error rate over every request the run sent."""
    total = sum(
        run.count(name)
        for name in (STATUS_2XX_ENTRY, STATUS_4XX_ENTRY, STATUS_5XX_ENTRY, TRANSPORT_ERROR_ENTRY)
    )
    if total <= 0:
        return Sli(
            key="error_rate",
            title="Error rate (5xx problem codes)",
            source="slo.md §1 row 5",
            unit="ratio",
            note="no request recorded an outcome",
        )
    return Sli(
        key="error_rate",
        title="Error rate (5xx problem codes)",
        source="slo.md §1 row 5",
        unit="ratio",
        value=round(run.count(STATUS_5XX_ENTRY) / total, 6),
        note="client-observed statuses; cross-check against fasterrag_errors_total",
    )


def _cache_hit_rate(run: RunStats) -> Sli:
    """Derive the cache hit rate from the counter's movement across the run."""
    hits = run.delta("fasterrag_cache_events_total", result="hit")
    misses = run.delta("fasterrag_cache_events_total", result="miss")
    lookups = hits + misses
    if lookups <= 0:
        return Sli(
            key="cache_hit_rate",
            title="Cache hit rate",
            source="slo.md §1 row 6",
            unit="ratio",
            note="fasterrag_cache_events_total did not move during the run",
        )
    return Sli(
        key="cache_hit_rate",
        title="Cache hit rate",
        source="slo.md §1 row 6",
        unit="ratio",
        value=round(hits / lookups, 6),
        note=(
            "a load run replaying a fixed query set inflates this; the figure describes the "
            "dataset's repetition as much as the cache, and slo.md defines it over 24 h"
        ),
    )


def _dlq_depth(run: RunStats) -> Sli:
    """Read the dead-letter gauge after the run."""
    present = any(item.name == "fasterrag_dlq_depth" for item in run.metrics_after)
    if not present:
        return Sli(
            key="dlq_depth",
            title="DLQ depth",
            source="slo.md §1 row 7",
            unit="documents",
            note="the target exposed no fasterrag_dlq_depth series",
        )
    return Sli(
        key="dlq_depth",
        title="DLQ depth",
        source="slo.md §1 row 7",
        unit="documents",
        value=series_total(run.metrics_after, "fasterrag_dlq_depth"),
        note="gauge read after the run, summed across collections",
    )


def derive_slis(run: RunStats) -> tuple[Sli, ...]:
    """Derive every slo.md §1 indicator this run can support, in that file's order."""
    return (
        _availability(run),
        _retrieval_latency(run),
        _ttft(run),
        _ingest_throughput(run),
        _error_rate(run),
        _cache_hit_rate(run),
        _dlq_depth(run),
    )


def median_slis(per_run: Sequence[tuple[Sli, ...]]) -> tuple[Sli, ...]:
    """Reduce repetitions to the median of each indicator, per the methodology's rule 5."""
    if not per_run:
        return ()
    headline: list[Sli] = []
    for index, template in enumerate(per_run[0]):
        values: list[float] = []
        for run in per_run:
            candidate = run[index].value
            if candidate is not None:
                values.append(candidate)
        if not values:
            headline.append(template)
            continue
        note = template.note
        if len(per_run) > 1:
            note = f"{note}; median of {len(values)} of {len(per_run)} repetitions"
        headline.append(
            Sli(
                key=template.key,
                title=template.title,
                source=template.source,
                unit=template.unit,
                value=statistics.median(values),
                note=note,
            )
        )
    return tuple(headline)


@dataclass(frozen=True, slots=True)
class Problem:
    """One reason this run is not a measurement, and the exit code it earns."""

    code: ExitCode
    detail: str


def _latency_entries(config: LoadConfig) -> tuple[str, ...]:
    """Return the entries whose percentiles the sample-count rule applies to."""
    if config.scenario == "ingest":
        return ()
    if config.stream_weight <= 0:
        return (RETRIEVAL_SLI_ENTRY,)
    return (RETRIEVAL_SLI_ENTRY, TTFT_SLI_ENTRY)


def audit(run: RunStats, config: LoadConfig) -> tuple[Problem, ...]:
    """Return every reason this repetition fails to be a valid sample."""
    problems: list[Problem] = []
    total = sum(entry.num_requests for entry in run.endpoints)
    if total <= 0:
        problems.append(
            Problem(
                ExitCode.VACUOUS,
                f"{run.label}: zero requests were recorded against {config.target}",
            )
        )

    for name in config.required_entries:
        if run.count(name) <= 0:
            problems.append(
                Problem(
                    ExitCode.VACUOUS,
                    f"{run.label}: the {config.scenario} scenario recorded no {name} request",
                )
            )

    for name in _latency_entries(config):
        entry = run.entry(name)
        if entry is None:
            continue
        if entry.num_requests < config.min_samples:
            problems.append(
                Problem(
                    ExitCode.VACUOUS,
                    f"{run.label}: {name} has {entry.num_requests} samples, below the "
                    f"{config.min_samples} docs/performance.md §3 rule 4 requires",
                )
            )
        measured = entry.percentile(0.95)
        if measured is not None and measured <= 0:
            problems.append(
                Problem(
                    ExitCode.VACUOUS,
                    f"{run.label}: {name} p95 is {measured}, which no real HTTP round trip "
                    "produces; the entry is recording something other than a response time",
                )
            )

    for entry in run.endpoints:
        if entry.name.startswith("SLI "):
            continue
        if entry.failure_ratio > config.max_failure_ratio:
            problems.append(
                Problem(
                    ExitCode.TOO_MANY_FAILURES,
                    f"{run.label}: {entry.name} failed {entry.num_failures} of "
                    f"{entry.num_requests} requests "
                    f"({entry.failure_ratio:.1%} > {config.max_failure_ratio:.1%})",
                )
            )

    throttled = run.count(STATUS_429_ENTRY)
    if total > 0 and throttled / total > MAXIMUM_THROTTLED_RATIO:
        problems.append(
            Problem(
                ExitCode.TOO_MANY_FAILURES,
                f"{run.label}: {throttled} of {total} requests were rate-limited "
                f"({throttled / total:.1%}), so this run measured the rate limiter; raise "
                "security.rate_limit on the target or lower --users",
            )
        )

    return tuple(problems)


def exit_code_for(problems: Sequence[Problem]) -> ExitCode:
    """Return the most serious problem's exit code, or OK when there are none."""
    for code in _EXIT_PRECEDENCE:
        if any(problem.code is code for problem in problems):
            return code
    return ExitCode.OK


@dataclass(frozen=True, slots=True)
class LoadReport:
    """A whole load run: its parameters, its repetitions, its SLIs, and its verdict."""

    config: LoadConfig
    generator: str
    started_at: str
    commit: str
    cold: RunStats | None
    runs: tuple[RunStats, ...]
    slis: tuple[Sli, ...]
    problems: tuple[Problem, ...]

    @property
    def verdict(self) -> str:
        """Return ``valid-sample`` or ``invalid``. It is never ``pass``."""
        return "valid-sample" if not self.problems else "invalid"

    @property
    def citable(self) -> bool:
        """Return whether this run may become a benchmark-ledger entry."""
        return (
            self.verdict == "valid-sample"
            and self.config.isolated
            and len(self.runs) >= REPETITIONS
        )

    def render(self) -> str:
        """Return the operator-facing report."""
        lines = [
            f"load run — {self.config.scenario} scenario against {self.config.target}",
            f"  verdict: {self.verdict}",
            f"  started: {self.started_at}  commit: {self.commit}",
            f"  generator: {self.generator}",
            f"  system under test: {self.config.sut_hardware}",
            f"  concurrency: {self.config.users} users at {self.config.spawn_rate}/s, "
            f"{self.config.duration_s}s x {len(self.runs)} warmed repetitions "
            f"(+{self.config.warmup_s}s cold)",
            f"  dataset: {self.config.dataset}",
            "",
            "SLIs (docs/slo.md §1):"
            if self.verdict == "valid-sample"
            else "SLIs (docs/slo.md §1) — DIAGNOSTIC ONLY, this run is invalid:",
            *(sli.render() for sli in self.slis),
            "",
            "percentiles come from Locust's rounded response-time histogram: exact below "
            "100 ms, two significant figures above.",
        ]
        if self.cold is not None:
            cold_entry = self.cold.entry(RETRIEVAL_SLI_ENTRY)
            cold_p95 = cold_entry.percentile(0.95) if cold_entry else None
            shown = "not measured" if cold_p95 is None else f"{cold_p95:g} ms"
            lines.append(f"cold-start repetition (discarded from the headline): p95 {shown}")
        if self.problems:
            lines.extend(["", "this run is not a measurement:"])
            lines.extend(f"  [{problem.code.name}] {problem.detail}" for problem in self.problems)
        return "\n".join(lines)

    def as_dict(self) -> dict[str, Any]:
        """Return the machine-readable report."""
        return {
            "verdict": self.verdict,
            "citable": self.citable,
            "started_at": self.started_at,
            "commit": self.commit,
            "generator": self.generator,
            "config": self.config.as_dict(),
            "slis": [sli.as_dict() for sli in self.slis],
            "problems": [
                {"code": problem.code.name, "detail": problem.detail} for problem in self.problems
            ],
            "repetitions": [
                {
                    "label": run.label,
                    "duration_s": run.duration_s,
                    "entries": [
                        {
                            "name": entry.name,
                            "num_requests": entry.num_requests,
                            "num_failures": entry.num_failures,
                            "p50_ms": entry.percentile(0.50),
                            "p95_ms": entry.percentile(0.95),
                            "p99_ms": entry.percentile(0.99),
                        }
                        for entry in run.endpoints
                    ],
                }
                for run in self.runs
            ],
        }

    def ledger_draft(self) -> str:
        """Return a ledger entry draft, or the reason this run may not produce one."""
        if not self.citable:
            return "\n".join(
                [
                    "NOT CITABLE — no ledger entry may be written from this run:",
                    *(f"  - {reason}" for reason in self._citation_blockers()),
                ]
            )
        measured = [sli for sli in self.slis if sli.measured]
        numbers = "; ".join(f"{sli.title} {sli.value:g} {sli.unit}" for sli in measured)
        return "\n".join(
            [
                "### BENCH-XXXX — <claim, as a goal unless a named baseline was measured>",
                "- Claim: <what this number licenses, and nothing more>",
                f"- Method: tests/load/run_load.py --scenario {self.config.scenario} "
                f"--users {self.config.users} --duration {self.config.duration_s} "
                f"--repetitions {len(self.runs)}; median of the warmed repetitions, "
                f"{self.config.warmup_s}s cold start discarded, nearest-rank percentiles "
                "over Locust's rounded histogram",
                f"- Dataset: {self.config.dataset}",
                f"- Hardware: SUT {self.config.sut_hardware}; generator {self.generator}",
                f"- Date: {self.started_at[:10]}",
                f"- Numbers: {numbers}",
                f"- Commit: {self.commit}",
                "- Supersedes: none",
            ]
        )

    def _citation_blockers(self) -> tuple[str, ...]:
        """Return why this run cannot be cited, naming every reason rather than the first."""
        reasons: list[str] = []
        if self.problems:
            reasons.append(f"the run is {self.verdict}: {len(self.problems)} problem(s) above")
        if not self.config.isolated:
            reasons.append(
                "the operator did not pass --isolated, so nothing asserts the system under "
                "test was free of co-tenant load (docs/performance.md §3 rule 5)"
            )
        if len(self.runs) < REPETITIONS:
            reasons.append(
                f"{len(self.runs)} warmed repetition(s) were run; the methodology fixes "
                f"{REPETITIONS} with the median reported"
            )
        return tuple(reasons)
