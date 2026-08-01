"""The benchmark suite and ledger emitter.

Implements the methodology of ``docs/performance.md`` §Methodology, which exists so that a
number in this repository means something. Four rules it enforces mechanically rather than by
convention:

* **Hardware is part of the number.** A latency without a machine attached is not a
  measurement, so the fingerprint is captured automatically and cannot be omitted.
* **Warm and cold are reported separately.** A model load or an empty cache belongs to
  cold-start; folding it into a warmed percentile makes a system look slower than it is, and
  hiding it makes a first request look faster than it will ever be.
* **Three repetitions, median reported, every run attached.** One run is an anecdote. The
  median resists the single slow run every machine occasionally produces, and attaching all
  three means a reader can see the spread rather than trust the summary.
* **The commit is recorded.** A measurement that cannot be tied to the code that produced it
  cannot be reproduced or superseded.

Nothing here decides whether a number is *good*. The suite measures; the ledger records;
whether a claim may be made from it is a human judgement made against the ledger rules.
"""

from __future__ import annotations

import platform
import shutil
import statistics
import subprocess
import time
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any, Final

from fasterrag import __version__
from fasterrag.observability.logging import get_logger

__all__ = [
    "REPETITIONS",
    "BenchmarkRun",
    "Fingerprint",
    "SuiteResult",
    "fingerprint",
    "ledger_entry",
    "measure",
    "percentile",
]

# CRITICAL: the methodology fixes this at three. Changing it changes what every recorded
# number means, so a ledger entry produced under a different count is not comparable with the
# entries already committed.
REPETITIONS: Final = 3

_logger = get_logger(__name__)


@dataclass(frozen=True, slots=True)
class Fingerprint:
    """The machine a measurement was taken on."""

    cpu: str
    cores: int
    ram_gb: float
    gpu: str
    storage_gb: float
    os: str
    python: str
    fasterrag: str

    def as_dict(self) -> dict[str, Any]:
        """Return the serialized fingerprint."""
        return {
            "cpu": self.cpu,
            "cores": self.cores,
            "ram_gb": self.ram_gb,
            "gpu": self.gpu,
            "storage_gb": self.storage_gb,
            "os": self.os,
            "python": self.python,
            "fasterrag": self.fasterrag,
        }

    def describe(self) -> str:
        """Return the one-line form a ledger entry's Hardware field carries."""
        return (
            f"{self.cpu} ({self.cores} cores), {self.ram_gb} GB RAM, GPU: {self.gpu}, "
            f"{self.storage_gb} GB storage, {self.os}, Python {self.python}"
        )


def _gpu() -> str:
    """Return the GPU description, or a plain statement that there is none.

    Never guesses. "unknown" and "none detected" are different facts, and a benchmark that
    reported a CPU-only run as though a GPU might have been involved would be unreproducible.
    """
    if shutil.which("nvidia-smi") is None:
        return "none detected (no nvidia-smi)"

    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return "unknown (nvidia-smi failed)"

    output = result.stdout.strip()
    return output.replace("\n", "; ") if output else "none detected"


def fingerprint() -> Fingerprint:
    """Capture the machine this run is happening on."""
    import psutil

    usage = shutil.disk_usage(".")
    return Fingerprint(
        cpu=platform.processor() or platform.machine() or "unknown",
        cores=psutil.cpu_count(logical=True) or 0,
        ram_gb=round(psutil.virtual_memory().total / 1024**3, 1),
        gpu=_gpu(),
        storage_gb=round(usage.total / 1024**3, 1),
        os=f"{platform.system()} {platform.release()}",
        python=platform.python_version(),
        fasterrag=__version__,
    )


def commit_hash() -> str:
    """Return the current commit, or a plain statement that it is unknown.

    A ledger entry without a commit is unreproducible, so this never silently returns an
    empty string that would render as a blank field nobody notices.
    """
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, timeout=10, check=False
        )
    except (OSError, subprocess.SubprocessError):
        return "unknown (git unavailable)"

    revision = result.stdout.strip()
    if not revision:
        return "unknown (not a git checkout)"

    dirty = subprocess.run(
        ["git", "status", "--porcelain"], capture_output=True, text=True, timeout=10, check=False
    )
    # CRITICAL: a dirty tree is recorded, not hidden. A number measured against uncommitted
    # changes cannot be reproduced from the commit alone, and a reader has to know that.
    return f"{revision}-dirty" if dirty.stdout.strip() else revision


def percentile(values: Sequence[float], fraction: float) -> float:
    """Return the nearest-rank percentile of ``values``.

    Nearest-rank rather than interpolated: an interpolated p95 reports a latency no request
    actually had, and with the small sample counts a benchmark run produces that invention
    is large enough to matter.
    """
    if not values:
        return 0.0

    ordered = sorted(values)
    rank = max(1, min(len(ordered), round(fraction * len(ordered) + 0.5)))
    return ordered[rank - 1]


@dataclass(frozen=True, slots=True)
class BenchmarkRun:
    """One repetition of one suite."""

    samples: list[float] = field(default_factory=list)
    operations: int = 0
    seconds: float = 0.0

    @property
    def throughput(self) -> float:
        """Return operations per second."""
        return self.operations / self.seconds if self.seconds else 0.0

    def as_dict(self) -> dict[str, Any]:
        """Return the serialized run."""
        return {
            "operations": self.operations,
            "seconds": round(self.seconds, 4),
            "throughput_per_sec": round(self.throughput, 3),
            "p50_ms": round(percentile(self.samples, 0.50) * 1000, 2),
            "p95_ms": round(percentile(self.samples, 0.95) * 1000, 2),
            "p99_ms": round(percentile(self.samples, 0.99) * 1000, 2),
            "max_ms": round(max(self.samples) * 1000, 2) if self.samples else 0.0,
        }


@dataclass(frozen=True, slots=True)
class SuiteResult:
    """Every repetition of one suite, and the median across them."""

    suite: str
    dataset: str
    cold: BenchmarkRun | None = None
    runs: list[BenchmarkRun] = field(default_factory=list)
    notes: str = ""

    @property
    def median(self) -> dict[str, Any]:
        """Return the median of each statistic across the warmed repetitions."""
        if not self.runs:
            return {}

        keys = self.runs[0].as_dict().keys()
        return {
            key: round(statistics.median(run.as_dict()[key] for run in self.runs), 3)
            for key in keys
        }

    def as_dict(self) -> dict[str, Any]:
        """Return the serialized suite result, all runs attached."""
        return {
            "suite": self.suite,
            "dataset": self.dataset,
            "repetitions": len(self.runs),
            "cold": self.cold.as_dict() if self.cold else None,
            "median": self.median,
            "runs": [run.as_dict() for run in self.runs],
            "notes": self.notes,
        }


async def measure(
    operation: Callable[[], Awaitable[Any]],
    *,
    iterations: int,
    repetitions: int = REPETITIONS,
) -> tuple[BenchmarkRun | None, list[BenchmarkRun]]:
    """Time an operation cold once, then warmed for each repetition.

    Args:
        operation: The awaitable being measured. Called once before the warmed runs begin,
            and that first call is reported separately as the cold-start sample.
        iterations: Calls per warmed repetition.
        repetitions: How many warmed repetitions to run.

    Returns:
        The cold run and the warmed runs. The cold run holds exactly one sample by
        construction: cold-start happens once and averaging it with itself would be
        meaningless.
    """
    started = time.perf_counter()
    await operation()
    cold = BenchmarkRun(
        samples=[time.perf_counter() - started],
        operations=1,
        seconds=time.perf_counter() - started,
    )

    runs: list[BenchmarkRun] = []
    for repetition in range(repetitions):
        samples: list[float] = []
        began = time.perf_counter()
        for _ in range(iterations):
            call_started = time.perf_counter()
            await operation()
            samples.append(time.perf_counter() - call_started)
        elapsed = time.perf_counter() - began

        runs.append(BenchmarkRun(samples=samples, operations=iterations, seconds=elapsed))
        _logger.info(
            "benchmark repetition complete",
            extra={
                "repetition": repetition + 1,
                "of": repetitions,
                "iterations": iterations,
                "p95_ms": round(percentile(samples, 0.95) * 1000, 2),
            },
        )

    return cold, runs


def ledger_entry(
    identifier: str,
    claim: str,
    result: SuiteResult,
    *,
    machine: Fingerprint | None = None,
    method: str = "",
    supersedes: str = "none",
) -> str:
    """Render a ready-to-commit ledger entry.

    Every one of the seven mandatory fields is emitted, populated from what was actually
    measured. A template with a blank to fill in by hand is a template someone eventually
    commits with the blank still in it.
    """
    machine = machine or fingerprint()
    median = result.median
    iterations = result.runs[0].operations if result.runs else 0
    default_method = f"fasterrag benchmark --suite {result.suite} --ledger"
    cold = result.cold.as_dict() if result.cold else {}

    numbers = (
        f"warmed median of {len(result.runs)} repetitions — "
        f"p50 {median.get('p50_ms', 0)} ms, p95 {median.get('p95_ms', 0)} ms, "
        f"p99 {median.get('p99_ms', 0)} ms, max {median.get('max_ms', 0)} ms, "
        f"throughput {median.get('throughput_per_sec', 0)}/s; "
        f"cold start {cold.get('p50_ms', 0)} ms"
    )

    return "\n".join(
        [
            f"### {identifier} — {claim}",
            f"- Claim: {claim}",
            f"- Method: {method or default_method}; {REPETITIONS} warmed repetitions of "
            f"{iterations} iterations, median reported, cold start measured separately, "
            "nearest-rank percentiles",
            f"- Dataset: {result.dataset}",
            f"- Hardware: {machine.describe()}",
            f"- Date: {datetime.now(tz=UTC).date().isoformat()}",
            f"- Numbers: {numbers}",
            f"- Commit: {commit_hash()}",
            f"- Supersedes: {supersedes}",
            *([f"- Notes: {result.notes}"] if result.notes else []),
        ]
    )
