"""Measure what a tenant's rolling spend window costs to hold and to read — TASK-0366.

``CostGovernor`` became process-lifetime under TASK-0337, which is what turned its
per-tenant window from a structure that died with the request into one that accumulates for
a full 24 hours. The ledger inferred "~1M tuples at 1M queries/day" and inferred nothing
about bytes or microseconds. This script replaces both inferences with a measurement.

**Two designs, one process, one machine.** The shipped governor is measured beside
``ExactWindow`` — a transcription of the structure ``services/budget.py`` held before
TASK-0366, one ``(timestamp, tokens)`` tuple per charge in a per-tenant ``deque``. Measuring
them in the same run makes the comparison paired: the same interpreter, the same allocator
state, the same co-tenant load. A number from one tree and a number from another tree are
not comparable, and this is the reason the superseded design is transcribed here rather
than read from git history.

**Memory is reported twice, by two instruments that can disagree.** ``tracemalloc`` reports
what the interpreter allocated and still holds while the structure was built, which includes
allocator slack and anything incidental; a recursive ``sys.getsizeof`` walk attributes bytes
to the structure's own objects and misses the slack. Both are printed. An estimate of the
form "8 bytes x N" is neither, and is what this script exists to retire.

**The structures are seeded, not charged, and the reason is itself a finding.** The
superseded ``record`` prunes by calling ``_spent_since``, which also *sums* the whole
window and discards the total — so one charge costs O(entries) and filling a day's traffic
is quadratic. Filling 1M entries through it does not finish. Seeding therefore appends
directly, and the writer's real per-call cost is measured separately against a structure
already at that fill, which is the number that matters and the one a build loop hides.

**``seed_s`` is not a writer cost and must not be read as one.** The fill runs with
``tracemalloc`` active so the memory figure covers it, and tracing every allocation inflates
it several-fold. ``write_p50_us`` is sampled afterwards, with tracing stopped, against a
structure already at that fill; that is the per-charge cost.

**``spent_tokens`` is the cross-check that makes the comparison mean anything.** Both
designs report it, and it must be identical: a bucketed window approximates *when* spend is
forgotten, never *how much* is counted inside the window. A run where the two differ is a
bug in the bucketing, not a trade-off.

**It changes nothing.** It writes no files unless ``--json`` names one, touches no vector
database, no model and no network, and recommends nothing: it prints numbers for
``docs/benchmarks.md``.

Usage::

    python scripts/measure_cost_governor.py
    python scripts/measure_cost_governor.py --charges 10000,100000,1000000 --json out.json
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
import time
import tracemalloc
from collections import deque
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final, Protocol

# CRITICAL: imported for its side effect, before any fasterrag import: it puts this
# checkout's src first on sys.path, so a worktree measures its own code (TASK-0522).
import _checkout  # noqa: F401

from fasterrag.services.benchmark import commit_hash, fingerprint, percentile
from fasterrag.services.budget import ROLLING_WINDOW_SECONDS, CostGovernor

__all__ = [
    "DEFAULT_CHARGES",
    "ExactWindow",
    "Measurement",
    "Report",
    "Window",
    "deep_size",
    "main",
    "measure",
    "measure_one",
    "parse_args",
    "render",
    "seed",
]

DEFAULT_CHARGES: Final[tuple[int, ...]] = (10_000, 100_000, 1_000_000)
DEFAULT_TENANTS: Final = 1
DEFAULT_SAMPLES: Final = 20
DEFAULT_BUDGET: Final = 1_000_000_000_000

_TOKEN_SPREAD: Final = 4096
_SHIPPED: Final = "shipped (bucketed)"
_SUPERSEDED: Final = "superseded (tuple per charge)"


class Window(Protocol):
    """The two methods this script exercises, so both designs can be measured by one path."""

    def record(self, tokens: int, *, tenant: str | None = ..., now: float | None = ...) -> None:
        """Charge tokens to a tenant at a moment."""

    def spent(self, tenant: str | None = ..., *, now: float | None = ...) -> int:
        """Return the tenant's spend inside the window at a moment."""


class ExactWindow:
    """The structure ``services/budget.py`` held before TASK-0366, transcribed for comparison.

    One ``(timestamp, tokens)`` tuple per charge, per tenant, evicted from the left on read.
    Kept here and not in ``src`` because it is a measurement baseline, not a shipped design.
    """

    def __init__(self, per_tenant: int, window_seconds: float = ROLLING_WINDOW_SECONDS) -> None:
        """Build an empty window with the budget and period the comparison runs at."""
        self.per_tenant = per_tenant
        self.window_seconds = window_seconds
        self._spent: dict[str, deque[tuple[float, int]]] = {}

    def record(self, tokens: int, *, tenant: str | None = None, now: float | None = None) -> None:
        """Charge a completed request's token usage, exactly as the superseded design did."""
        moment = time.monotonic() if now is None else now
        key = tenant or ""
        self._spent_since(key, moment)
        self._spent.setdefault(key, deque()).append((moment, tokens))

    def append(self, tokens: int, *, tenant: str, now: float) -> None:
        """Seed one entry without the read ``record`` performs, so filling stays linear."""
        self._spent.setdefault(tenant, deque()).append((now, tokens))

    def spent(self, tenant: str | None = None, *, now: float | None = None) -> int:
        """Return a tenant's spend inside the current window."""
        moment = time.monotonic() if now is None else now
        return self._spent_since(tenant or "", moment)

    def _spent_since(self, key: str, moment: float) -> int:
        """Drop everything older than the window and sum what is left."""
        window = self._spent.get(key)
        if window is None:
            return 0
        while window and moment - window[0][0] >= self.window_seconds:
            window.popleft()
        if not window:
            del self._spent[key]
            return 0
        return sum(tokens for _, tokens in window)


def deep_size(value: object, seen: set[int] | None = None) -> int:
    """Return the bytes the object graph under ``value`` occupies, counting each object once.

    Shared objects — interned small integers above all — are counted once, which is what
    the process actually pays, and is why this is reported beside the ``tracemalloc`` figure
    rather than instead of it.
    """
    visited = set() if seen is None else seen
    marker = id(value)
    if marker in visited:
        return 0
    visited.add(marker)

    total = sys.getsizeof(value)
    if isinstance(value, dict):
        for key, item in value.items():
            total += deep_size(key, visited) + deep_size(item, visited)
    elif isinstance(value, (list, tuple, set, frozenset, deque)):
        for item in value:
            total += deep_size(item, visited)
    return total


@dataclass(frozen=True, slots=True)
class Measurement:
    """What one design cost at one number of charges."""

    design: str
    charges: int
    tenants: int
    seed_s: float
    traced_current_bytes: int
    traced_peak_bytes: int
    structure_bytes: int
    retained_entries: int
    read_p50_us: float
    read_p95_us: float
    read_worst_us: float
    write_p50_us: float
    write_p95_us: float
    expiry_walk_ms: float
    spent_tokens: int

    @property
    def bytes_per_charge(self) -> float:
        """Return the structure's own bytes divided by the charges it absorbed."""
        return self.structure_bytes / self.charges if self.charges else 0.0

    def as_dict(self) -> dict[str, Any]:
        """Return the JSON form."""
        return {
            "design": self.design,
            "charges": self.charges,
            "tenants": self.tenants,
            "seed_s": round(self.seed_s, 3),
            "traced_current_bytes": self.traced_current_bytes,
            "traced_peak_bytes": self.traced_peak_bytes,
            "structure_bytes": self.structure_bytes,
            "bytes_per_charge": round(self.bytes_per_charge, 2),
            "retained_entries": self.retained_entries,
            "read_p50_us": round(self.read_p50_us, 2),
            "read_p95_us": round(self.read_p95_us, 2),
            "read_worst_us": round(self.read_worst_us, 2),
            "write_p50_us": round(self.write_p50_us, 2),
            "write_p95_us": round(self.write_p95_us, 2),
            "expiry_walk_ms": round(self.expiry_walk_ms, 3),
            "spent_tokens": self.spent_tokens,
        }


@dataclass
class Report:
    """Everything one run measured."""

    window_seconds: float = ROLLING_WINDOW_SECONDS
    samples: int = DEFAULT_SAMPLES
    measurements: list[Measurement] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        """Return the JSON form."""
        return {
            "window_seconds": self.window_seconds,
            "samples": self.samples,
            "measurements": [item.as_dict() for item in self.measurements],
            "hardware": fingerprint().as_dict(),
            "commit": commit_hash(),
        }


def _tenant(index: int, tenants: int) -> str:
    """Return the tenant a charge belongs to, spreading charges round-robin."""
    return f"tenant-{index % tenants:06d}"


def seed(window: Window, *, charges: int, tenants: int, window_seconds: float) -> float:
    """Fill with ``charges`` entries spread evenly across one window, and time the fill.

    Spreading matters: charges piled onto one instant would land in one bucket and flatter
    the bucketed design, while the traffic this sizes is spread across the day.
    """
    step = window_seconds / charges
    appender = getattr(window, "append", None)
    started = time.perf_counter()
    for index in range(charges):
        tokens = 1 + index % _TOKEN_SPREAD
        name = _tenant(index, tenants)
        if appender is None:
            window.record(tokens, tenant=name, now=index * step)
        else:
            appender(tokens, tenant=name, now=index * step)
    return time.perf_counter() - started


def _retained(window: Window) -> int:
    """Return how many stored entries the structure holds across every tenant."""
    held = getattr(window, "_spent", {})
    return sum(len(entries) for entries in held.values())


def measure_one(
    design: str,
    factory: Window,
    *,
    charges: int,
    tenants: int,
    samples: int,
    window_seconds: float,
) -> Measurement:
    """Fill one design at one scale, then price holding it, reading it and charging it."""
    gc.collect()
    tracemalloc.start(1)
    seed_s = seed(factory, charges=charges, tenants=tenants, window_seconds=window_seconds)
    gc.collect()
    traced_current, traced_peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    last = (charges - 1) * (window_seconds / charges)
    probe = _tenant(0, tenants)

    reads: list[float] = []
    for _ in range(samples):
        started = time.perf_counter()
        total = factory.spent(probe, now=last)
        reads.append((time.perf_counter() - started) * 1_000_000)

    writes: list[float] = []
    for index in range(samples):
        started = time.perf_counter()
        factory.record(1, tenant=probe, now=last + index * 1e-6)
        writes.append((time.perf_counter() - started) * 1_000_000)

    structure_bytes = deep_size(getattr(factory, "_spent", {}))
    retained = _retained(factory)

    started = time.perf_counter()
    for index in range(tenants):
        factory.spent(_tenant(index, tenants), now=last + window_seconds + 1)
    expiry_walk_ms = (time.perf_counter() - started) * 1000

    return Measurement(
        design=design,
        charges=charges,
        tenants=tenants,
        seed_s=seed_s,
        traced_current_bytes=traced_current,
        traced_peak_bytes=traced_peak,
        structure_bytes=structure_bytes,
        retained_entries=retained,
        read_p50_us=percentile(reads, 0.50),
        read_p95_us=percentile(reads, 0.95),
        read_worst_us=max(reads),
        write_p50_us=percentile(writes, 0.50),
        write_p95_us=percentile(writes, 0.95),
        expiry_walk_ms=expiry_walk_ms,
        spent_tokens=total,
    )


def measure(
    *,
    charges: Sequence[int],
    tenants: int,
    samples: int,
    window_seconds: float = ROLLING_WINDOW_SECONDS,
) -> Report:
    """Measure both designs at every requested scale."""
    report = Report(window_seconds=window_seconds, samples=samples)
    for count in charges:
        for design in (_SUPERSEDED, _SHIPPED):
            window: Window = (
                ExactWindow(DEFAULT_BUDGET, window_seconds)
                if design == _SUPERSEDED
                else CostGovernor(
                    per_tenant=DEFAULT_BUDGET,
                    window_seconds=window_seconds,
                )
            )
            report.measurements.append(
                measure_one(
                    design,
                    window,
                    charges=count,
                    tenants=tenants,
                    samples=samples,
                    window_seconds=window_seconds,
                )
            )
            del window
            gc.collect()
    return report


def _mib(value: int) -> str:
    """Render bytes as MiB, which is the unit the answer is in."""
    return f"{value / 1024**2:,.2f} MiB"


def render(report: Report) -> str:
    """Return the table a reader pastes into the ledger."""
    lines = [
        f"window: {report.window_seconds:,.0f} s | read samples per scale: {report.samples}",
        "",
        f"{'design':<30}{'charges':>10}{'held':>10}{'structure':>14}"
        f"{'traced':>14}{'read p50':>13}{'write p50':>13}{'expiry ms':>12}",
    ]
    for item in report.measurements:
        lines.append(
            f"{item.design:<30}{item.charges:>10,}{item.retained_entries:>10,}"
            f"{_mib(item.structure_bytes):>14}{_mib(item.traced_current_bytes):>14}"
            f"{item.read_p50_us:>12,.1f}u{item.write_p50_us:>12,.1f}u"
            f"{item.expiry_walk_ms:>12,.2f}"
        )
    return "\n".join(lines)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse the command line."""
    parser = argparse.ArgumentParser(description=__doc__ or "")
    parser.add_argument(
        "--charges",
        default=",".join(str(value) for value in DEFAULT_CHARGES),
        help="comma-separated charge counts to measure",
    )
    parser.add_argument("--tenants", type=int, default=DEFAULT_TENANTS)
    parser.add_argument("--samples", type=int, default=DEFAULT_SAMPLES)
    parser.add_argument("--window-seconds", type=float, default=ROLLING_WINDOW_SECONDS)
    parser.add_argument("--json", dest="json_path", default=None)
    parsed = parser.parse_args(argv)
    parsed.charge_counts = [int(part) for part in parsed.charges.split(",") if part.strip()]
    return parsed


def main(argv: Sequence[str] | None = None) -> int:
    """Run the measurement and print the table."""
    args = parse_args(argv)
    report = measure(
        charges=args.charge_counts,
        tenants=args.tenants,
        samples=args.samples,
        window_seconds=args.window_seconds,
    )
    sys.stdout.write(render(report) + "\n")
    if args.json_path:
        Path(args.json_path).write_text(
            json.dumps(report.as_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        sys.stdout.write(f"wrote {args.json_path}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
