"""Measure what reclaiming idempotency records costs — evidence for TASK-0365 and TASK-0358.

Two questions, one directory layout, so they share a harness.

**TASK-0365** asks what it costs to stop ``FileIdempotencyStore`` leaking a file per keyed
request. Three designs were on the table — an opportunistic prune of the tenant bucket on
every write, a background sweeper with its own lifecycle, and the amortized pass that
shipped — and they differ only in *who pays* a cost this script measures once: a full walk
of the record directory. It therefore reports the pass itself (the cost the shipped design
charges to one request per interval, and the cost the prune design would charge to **every**
request), and separately what a concurrent request pays while a pass is running, which is
the number that decides whether the pass needs to leave the request path at all.

**TASK-0358** asks whether ``Journal.find_by_idempotency_key`` — a glob over every job
record, on the deprecated ingest body-field path — is worth indexing before the field is
removed at 1.0. The answer is a latency at a realistic journal size, so this script measures
the miss (the full scan, which is what submitting a *new* key does) and the hit. **TASK-0437**
indexed the lookup under ``job-keys/``; the jobs here are still written without the index, as
an operator's journal from before that change was, so the script also times the first lookup,
which builds the index from a full scan once.

**TASK-0399** asks what that scan — and every other journal write — costs the *other*
requests on the worker, because ``IngestionService.accept`` used to make it from inside an
``async def``. The question is not how long a journal call takes but how long the event
loop stops serving anyone, so this script runs a heartbeat coroutine and reports the lag it
sees while one ingest is journalled: once with the blocking call made on the loop, as the
code did before, and once awaited through the journal's async face. It also measures the
per-document pair, because a thread handoff on something that runs millions of times can
cost more than the write it moves (the trap TASK-0365 recorded).

**It changes nothing.** It writes only into a temporary directory of its own, which it
removes, and it recommends nothing: it prints numbers for ``docs/benchmarks.md``.

The records and jobs it writes are produced by the shipped writers — ``write_record`` and
``Journal.save_job`` — so the files are the ones production reads, checksum envelope and
all, rather than a convenient approximation of them.

Usage::

    python scripts/measure_idempotency_reclaim.py
    python scripts/measure_idempotency_reclaim.py --records 50000 --jobs 20000 --json out.json
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import shutil
import sys
import tempfile
import threading
import time
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Final

# CRITICAL: imported for its side effect, before any fasterrag import: it puts this
# checkout's src first on sys.path, so a worktree measures its own code (TASK-0522).
import _checkout  # noqa: F401

from fasterrag.services.benchmark import commit_hash, fingerprint, percentile
from fasterrag.services.idempotency import (
    RECLAIM_GRACE_SECONDS,
    FileIdempotencyStore,
    Reclamation,
    request_fingerprint,
)
from fasterrag.services.journal import DocumentRecord, JobRecord, Journal

__all__ = [
    "Latency",
    "Report",
    "idle_lag",
    "loop_lag",
    "main",
    "measure",
    "parse_args",
    "per_document",
]

DEFAULT_RECORDS: Final = 10_000
DEFAULT_EXPIRED: Final = 2_000
DEFAULT_JOBS: Final = 10_000
DEFAULT_SAMPLES: Final = 200
DEFAULT_LAG_ROUNDS: Final = 20

_HEARTBEAT_SECONDS: Final = 0.005

_TENANT: Final = "acme"
_STORED_BODY: Final = b'{"id":"col_00000000","name":"handbook","status":"created"}'


@dataclass(frozen=True, slots=True)
class Latency:
    """A latency sample set, in milliseconds."""

    name: str
    samples: int
    p50: float
    p95: float
    worst: float

    def as_dict(self) -> dict[str, Any]:
        """Return the JSON form."""
        return asdict(self)


@dataclass
class Report:
    """Everything one run measured."""

    records: int = 0
    expired: int = 0
    jobs: int = 0
    build_records_s: float = 0.0
    build_jobs_s: float = 0.0
    reclaim_first: dict[str, Any] = field(default_factory=dict)
    reclaim_steady: dict[str, Any] = field(default_factory=dict)
    latencies: list[Latency] = field(default_factory=list)
    journal: list[Latency] = field(default_factory=list)
    loop: list[Latency] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        """Return the JSON form."""
        return {
            "records": self.records,
            "expired": self.expired,
            "jobs": self.jobs,
            "build_records_s": round(self.build_records_s, 2),
            "build_jobs_s": round(self.build_jobs_s, 2),
            "reclaim_first": self.reclaim_first,
            "reclaim_steady": self.reclaim_steady,
            "latencies": [item.as_dict() for item in self.latencies],
            "journal": [item.as_dict() for item in self.journal],
            "loop": [item.as_dict() for item in self.loop],
            "hardware": fingerprint().as_dict(),
            "commit": commit_hash(),
        }


def _pass_as_dict(outcome: Reclamation) -> dict[str, Any]:
    """Return a reclamation pass in the JSON form the report carries."""
    return {
        "scanned": outcome.scanned,
        "opened": outcome.opened,
        "removed": outcome.removed,
        "failed": outcome.failed,
        "duration_ms": round(outcome.duration_ms, 2),
    }


def _latency(name: str, samples: Sequence[float]) -> Latency:
    """Summarise one set of millisecond samples."""
    return Latency(
        name=name,
        samples=len(samples),
        p50=percentile(samples, 0.50),
        p95=percentile(samples, 0.95),
        worst=max(samples),
    )


def populate(store: FileIdempotencyStore, *, records: int, expired: int) -> float:
    """Write ``records`` completed records, ``expired`` of them old enough to reclaim."""
    started = time.perf_counter()
    stale = time.time() - store.ttl_seconds - RECLAIM_GRACE_SECONDS - 1
    for index in range(records):
        created = stale if index < expired else time.time()
        path = store.path_for(f"key-{index}", _TENANT)
        store.write_record(
            path,
            {
                "tenant": _TENANT,
                "fingerprint": request_fingerprint("POST", "/v1/collections", "", b"{}"),
                "owner": f"{index:016x}",
                "state": "completed",
                "created_at": created,
                "response": {
                    "status": 201,
                    "media_type": "application/json",
                    "body": _STORED_BODY.decode("ascii"),
                },
            },
        )
        if index < expired:
            os.utime(path, (created, created))
    return time.perf_counter() - started


def claim_samples(store: FileIdempotencyStore, *, samples: int, offset: int) -> list[float]:
    """Time ``samples`` fresh claims against a populated store, in milliseconds."""
    digest = request_fingerprint("POST", "/v1/collections", "", b"{}")
    timings: list[float] = []
    for index in range(samples):
        key = f"probe-{offset}-{index}"
        started = time.perf_counter()
        store.claim(key, tenant=_TENANT, fingerprint=digest)
        timings.append((time.perf_counter() - started) * 1000)
    return timings


def claims_under_reclamation(store: FileIdempotencyStore, *, samples: int) -> list[float]:
    """Time claims issued while a reclamation pass walks the same directory."""
    timings: list[float] = []
    finished = threading.Event()

    def sweep() -> None:
        while not finished.is_set():
            store.reclaim()

    worker = threading.Thread(target=sweep, name="reclaim-probe", daemon=True)
    worker.start()
    try:
        timings = claim_samples(store, samples=samples, offset=2)
    finally:
        finished.set()
        worker.join(timeout=60)
    return timings


def populate_journal(journal: Journal, *, jobs: int) -> float:
    """Write ``jobs`` job records, each carrying its own body idempotency key.

    ``save_job_blocking`` rather than ``create_job_blocking``: the jobs are written as a
    journal from before the idempotency index was, so the first lookup has to build it.
    """
    started = time.perf_counter()
    for index in range(jobs):
        journal.save_job_blocking(
            JobRecord(
                job_id=f"job_{index:016x}",
                collection="handbook",
                sources=[{"type": "path", "value": f"corpus/{index}"}],
                idempotency_key=f"body-key-{index}",
                tenant=_TENANT,
            )
        )
    return time.perf_counter() - started


def first_lookup(journal: Journal) -> list[float]:
    """Time the first keyed lookup on a journal whose jobs were written without an index.

    That is the one-time cost an operator's existing journal pays after upgrading: the
    lookup finds no ``job-keys/`` index and builds it from a full scan (TASK-0437).
    """
    started = time.perf_counter()
    journal.find_by_idempotency_key_blocking("first-lookup", tenant=_TENANT)
    return [(time.perf_counter() - started) * 1000]


def journal_samples(journal: Journal, *, jobs: int, samples: int, hit: bool) -> list[float]:
    """Time ``find_by_idempotency_key`` for keys that are present, or for ones that are not.

    Present keys are spread across the scan order rather than taken from its start, so the
    hit figure is an average position and not a best case.
    """
    timings: list[float] = []
    stride = max(1, jobs // max(1, samples))
    journal.find_by_idempotency_key_blocking("warm-up", tenant=_TENANT)
    for index in range(samples):
        key = f"body-key-{(index * stride) % jobs}" if hit else f"absent-key-{index}"
        started = time.perf_counter()
        journal.find_by_idempotency_key_blocking(key, tenant=_TENANT)
        timings.append((time.perf_counter() - started) * 1000)
    return timings


async def _heartbeat(lags: list[float], stop: asyncio.Event, interval: float) -> None:
    """Record how far past ``interval`` each wake-up lands, which is the loop's lag."""
    while not stop.is_set():
        started = time.perf_counter()
        await asyncio.sleep(interval)
        lags.append((time.perf_counter() - started - interval) * 1000)


async def _under_load(
    work: Callable[[int], Awaitable[None]], *, rounds: int, interval: float
) -> list[float]:
    """Return the lag a heartbeat coroutine sees while ``work`` runs ``rounds`` times."""
    lags: list[float] = []
    stop = asyncio.Event()
    beat = asyncio.create_task(_heartbeat(lags, stop, interval))
    await asyncio.sleep(interval * 10)
    lags.clear()
    for index in range(rounds):
        await work(index)
        await asyncio.sleep(interval * 4)
    stop.set()
    await beat
    return lags


async def idle_lag(*, rounds: int) -> list[float]:
    """Return the lag a heartbeat sees with nothing else on the loop.

    The floor, not a result: a 5 ms sleep on Windows wakes on a ~15.6 ms timer, so every
    other row here has to be read against this one.
    """

    async def work(_: int) -> None:
        return None

    return await _under_load(work, rounds=rounds, interval=_HEARTBEAT_SECONDS)


async def loop_lag(journal: Journal, *, on_loop: bool, rounds: int) -> list[float]:
    """Time the loop's lag while a keyed job is accepted, with and without the offload.

    ``on_loop`` calls the blocking twin directly, which is what ``IngestionService.accept``
    did before TASK-0399; the other arm awaits the shipped async method.
    """
    sources = [{"type": "path", "value": "corpus/probe"}]

    async def work(index: int) -> None:
        key = f"lag-{int(on_loop)}-{index}"
        if on_loop:
            journal.create_job_blocking("handbook", sources, idempotency_key=key, tenant=_TENANT)
        else:
            await journal.create_job("handbook", sources, idempotency_key=key, tenant=_TENANT)

    return await _under_load(work, rounds=rounds, interval=_HEARTBEAT_SECONDS)


async def per_document(journal: Journal, *, offload: bool, samples: int) -> list[float]:
    """Time the per-document journal pair, with and without the thread handoff.

    This is the tight loop TASK-0365 warned about: if the handoff costs more than the two
    appends it moves, the offload is a regression dressed as a fix.
    """
    timings: list[float] = []
    for index in range(samples):
        record = DocumentRecord(
            document_id=f"doc-{index:08x}",
            source=f"corpus/{index}",
            status="indexed",
            content_hash=f"{index:064x}",
        )
        started = time.perf_counter()
        if offload:
            await journal.record_indexed("job_probe", "handbook", record, dedup=True)
        else:
            journal.record_indexed_blocking("job_probe", "handbook", record, dedup=True)
        timings.append((time.perf_counter() - started) * 1000)
    return timings


def measure(root: Path, args: argparse.Namespace) -> Report:
    """Run every measurement against a store and a journal under ``root``."""
    report = Report(records=args.records, expired=args.expired, jobs=args.jobs)
    store = FileIdempotencyStore(root / "store")

    _progress(f"writing {args.records} idempotency records")
    report.build_records_s = populate(store, records=args.records, expired=args.expired)

    _progress("timing claims against a populated store")
    report.latencies.append(
        _latency("claim (idle)", claim_samples(store, samples=args.samples, offset=1))
    )

    _progress("timing the first reclamation pass")
    report.reclaim_first = _pass_as_dict(store.reclaim())

    _progress("timing a steady-state reclamation pass")
    report.reclaim_steady = _pass_as_dict(store.reclaim())

    _progress("timing claims while a pass walks the same directory")
    report.latencies.append(
        _latency("claim (pass running)", claims_under_reclamation(store, samples=args.samples))
    )

    if args.jobs:
        journal = Journal(root / "journal")
        _progress(f"writing {args.jobs} job records")
        report.build_jobs_s = populate_journal(journal, jobs=args.jobs)
        _progress("timing the first lookup, which builds the idempotency index")
        report.journal.append(_latency("first lookup (builds index)", first_lookup(journal)))
        _progress("timing find_by_idempotency_key")
        report.journal.append(
            _latency(
                "find_by_idempotency_key (miss)",
                journal_samples(journal, jobs=args.jobs, samples=args.journal_samples, hit=False),
            )
        )
        report.journal.append(
            _latency(
                "find_by_idempotency_key (hit)",
                journal_samples(journal, jobs=args.jobs, samples=args.journal_samples, hit=True),
            )
        )

        _progress("timing an idle event loop, for the heartbeat floor")
        report.loop.append(
            _latency("loop lag, idle (floor)", asyncio.run(idle_lag(rounds=args.lag_rounds)))
        )
        _progress("timing the event loop while a job is accepted on it")
        report.loop.append(
            _latency(
                "loop lag, accept on the loop",
                asyncio.run(loop_lag(journal, on_loop=True, rounds=args.lag_rounds)),
            )
        )
        _progress("timing the event loop while a job is accepted off it")
        report.loop.append(
            _latency(
                "loop lag, accept off the loop",
                asyncio.run(loop_lag(journal, on_loop=False, rounds=args.lag_rounds)),
            )
        )

        _progress("timing the per-document journal pair")
        report.loop.append(
            _latency(
                "record_indexed, blocking",
                asyncio.run(per_document(journal, offload=False, samples=args.samples)),
            )
        )
        report.loop.append(
            _latency(
                "record_indexed, off the loop",
                asyncio.run(per_document(journal, offload=True, samples=args.samples)),
            )
        )

    return report


def _progress(message: str) -> None:
    """Write progress to stderr, so stdout carries only the report."""
    sys.stderr.write(f"... {message}\n")
    sys.stderr.flush()


def table(report: Report) -> str:
    """Render the report as the table the ledger entry is written from."""
    lines = [
        f"records {report.records} ({report.expired} expired), "
        f"written in {report.build_records_s:.1f}s",
        "",
        "reclamation pass           scanned  opened  removed   duration",
        f"  first (with garbage)   {report.reclaim_first['scanned']:8d}"
        f"{report.reclaim_first['opened']:8d}{report.reclaim_first['removed']:9d}"
        f"{report.reclaim_first['duration_ms']:10.2f} ms",
        f"  steady state           {report.reclaim_steady['scanned']:8d}"
        f"{report.reclaim_steady['opened']:8d}{report.reclaim_steady['removed']:9d}"
        f"{report.reclaim_steady['duration_ms']:10.2f} ms",
        "",
        "latency                        n       p50       p95     worst",
    ]
    lines.extend(_row(item) for item in report.latencies)
    if report.journal:
        lines.extend(["", f"journal of {report.jobs} jobs, written in {report.build_jobs_s:.1f}s"])
        lines.extend(_row(item) for item in report.journal)
    return "\n".join(lines)


def _row(item: Latency) -> str:
    """Render one latency row."""
    return (
        f"  {item.name:<28}{item.samples:4d}{item.p50:10.3f}{item.p95:10.3f}{item.worst:10.3f} ms"
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse the command line."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--records", type=int, default=DEFAULT_RECORDS)
    parser.add_argument("--expired", type=int, default=DEFAULT_EXPIRED)
    parser.add_argument("--jobs", type=int, default=DEFAULT_JOBS)
    parser.add_argument("--samples", type=int, default=DEFAULT_SAMPLES)
    parser.add_argument("--journal-samples", type=int, default=20)
    parser.add_argument("--lag-rounds", type=int, default=DEFAULT_LAG_ROUNDS)
    parser.add_argument("--json", dest="json_path", default=None)
    parser.add_argument(
        "--keep",
        action="store_true",
        help="leave the temporary tree in place instead of removing it",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Run the measurement and print the report."""
    args = parse_args(argv)
    root = Path(tempfile.mkdtemp(prefix="fasterrag-reclaim-"))
    try:
        report = measure(root, args)
    finally:
        if not args.keep:
            shutil.rmtree(root, ignore_errors=True)

    sys.stdout.write(table(report) + "\n")
    if args.json_path:
        Path(args.json_path).write_text(
            json.dumps(report.as_dict(), indent=2) + "\n", encoding="utf-8"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
