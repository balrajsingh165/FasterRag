"""The ingestion pipeline through the production ``ProcessPoolExecutor``.

Every other pool and pipeline test injects a ``ThreadPoolExecutor``, so for fourteen slices
nothing pickled a ``Settings`` object, a ``DocumentTask`` or a ``ParseOutcome`` across a
process boundary, started a real worker, or saw a real ``BrokenProcessPool`` (TASK-0455).
These tests pass no ``executor_factory`` at all: what runs is what a deployment runs.

They are marked ``must_run`` because a test of the real pool that is skipped in every lane
proves nothing. Chunking is ``fixed`` with the estimating token counter so a worker loads no
model, and the stand-in parse targets live in ``process_targets.py`` because a spawned child
has to be able to import them.

The parse-timeout cases (TASK-0473) are here rather than beside the other pool tests for
the same reason. A thread cannot be killed, so a thread-backed pool can show that an overdue
document is dead-lettered and its neighbours resubmitted, and nothing about the half that
matters: that the stuck *worker* is gone and the pool can parse again. Their timings are in
whole seconds on purpose — every margin is at least a second, because the machine running
them is usually running other suites too.
"""

import asyncio
import logging
import multiprocessing
import os
import threading
import time
from collections.abc import Callable, Iterator
from concurrent.futures import Executor, ProcessPoolExecutor
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import pytest

from fasterrag.adapters.embeddings.tiering import TieringRouter
from fasterrag.config.schema import Settings
from fasterrag.errors import ErrorCode, IngestionError
from fasterrag.services.ingestion import IngestionService
from fasterrag.services.jobs import INTERRUPTED, JobRegistry
from fasterrag.services.journal import Journal
from fasterrag.workers import cpu_pool
from fasterrag.workers.cpu_pool import CpuWorkerPool
from fasterrag.workers.queues import BoundedQueue, ChunkPayload
from tests.unit.services.test_ingestion import KeywordLegAdapter, StubEmbedder
from tests.unit.workers import process_targets

pytestmark = pytest.mark.must_run

NAMES = ("expenses.md", "leave.md", "travel.md")

HUNG = f"{process_targets.HANG}.md"
PAUSED = f"{process_targets.PAUSE}.md"
STALLED = f"{process_targets.STALL_ONCE}.md"

PATIENCE = 10.0
"""Seconds a test waits for something that takes well under one on an idle machine."""


def settings(**overrides: Any) -> Settings:
    payload: dict[str, Any] = {
        "embeddings": {"batch_size": 8},
        "workers": {"cpu_pool_size": 2, "embedding_pool_size": 1, "queue_depth": 200},
        "chunking": {
            "strategy": "fixed",
            "token_counter": "estimate",
            "chunk_size": 64,
            "overlap": 8,
        },
        "reliability": {"retries": {"max_attempts": 0, "backoff_base_ms": 1, "jitter": False}},
    }
    payload.update(overrides)
    return Settings.model_validate(payload)


@pytest.fixture
def corpus(tmp_path: Path) -> Path:
    root = tmp_path / "corpus"
    root.mkdir()
    for name in (*NAMES, HUNG, PAUSED, STALLED):
        (root / name).write_text(
            f"# {name}\n\n" + f"The policy in {name} applies to every employee. " * 20,
            encoding="utf-8",
        )
    return root


def build(tmp_path: Path, configured: Settings) -> tuple[IngestionService, KeywordLegAdapter]:
    """Return a service that builds its own executor, and the adapter it writes through."""
    adapter = KeywordLegAdapter(configured)
    service = IngestionService(
        configured,
        journal=Journal(tmp_path / "journal", checkpoint_every=1),
        adapter=adapter,
        router=TieringRouter(StubEmbedder(configured)),
    )
    return service, adapter


async def test_three_documents_ingest_through_the_default_executor(
    tmp_path: Path, corpus: Path
) -> None:
    """No test ran the real process pool, so nothing had ever crossed its pickle boundary."""
    service, adapter = build(tmp_path, settings())

    record = await service.accept([str(corpus / name) for name in NAMES])
    settled = await service.run(record)

    assert settled.status == "completed"
    assert settled.counts["parsed"] == 3
    assert settled.counts["dead_lettered"] == 0
    assert settled.counts["indexed"] == len(adapter.points) > 0
    indexed = service.journal.documents_blocking(record.job_id, status="indexed")
    assert sorted(Path(entry.source).name for entry in indexed) == sorted(NAMES)


async def test_a_failed_document_keeps_its_reason_code_across_the_process_boundary(
    tmp_path: Path, corpus: Path
) -> None:
    """A typed error raised in a worker is pickled back; its code must survive the trip."""
    service, _ = build(tmp_path, settings())

    record = await service.accept([str(corpus / NAMES[0]), str(corpus / "absent.md")])
    settled = await service.run(record)

    dead = service.journal.dead_lettered_blocking(record.job_id)
    assert settled.status == "partial"
    assert [entry.reason_code for entry in dead] == ["PARSE_FAILED"]
    assert "absent.md" in (dead[0].detail or "")


async def test_a_worker_failure_is_filed_under_the_trace_id_of_its_job(
    tmp_path: Path, corpus: Path
) -> None:
    """An error raised in a worker carried the worker's own trace id, never the job's."""
    service, _ = build(tmp_path, settings())

    record = await service.accept([str(corpus / NAMES[0]), str(corpus / "absent.md")])
    await service.run(record)

    indexed = service.journal.documents_blocking(record.job_id, status="indexed")
    dead = service.journal.dead_lettered_blocking(record.job_id)
    assert indexed[0].trace_id is not None
    assert dead[0].trace_id == indexed[0].trace_id


async def test_a_killed_worker_fails_the_job_with_a_typed_error_and_settles_it(
    tmp_path: Path, corpus: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A real BrokenProcessPool had never been raised in a test, only its message built."""
    monkeypatch.setattr(cpu_pool, "parse_and_chunk", process_targets.die)
    service, adapter = build(tmp_path, settings())

    record = await service.accept([str(corpus / name) for name in NAMES])
    with pytest.raises(IngestionError) as raised:
        await service.run(record)

    assert raised.value.code is ErrorCode.CHUNK_FAILED
    assert not raised.value.retryable
    assert service.journal.load_job_blocking(record.job_id).status == "failed"
    assert service.journal.load_job_blocking(record.job_id).finished_at is not None
    assert adapter.points == []


def worker_pids() -> set[int]:
    """Return the pid of every live child process of this interpreter."""
    return {child.pid for child in multiprocessing.active_children() if child.pid is not None}


async def surviving_workers(before: set[int], *, patience: float = 10.0) -> set[int]:
    """Return the workers started since ``before`` that are still alive once patience runs out."""
    deadline = asyncio.get_running_loop().time() + patience
    while (alive := worker_pids() - before) and asyncio.get_running_loop().time() < deadline:
        await asyncio.sleep(0.05)
    return alive


def outcomes(service: IngestionService, job: str, status: str) -> list[tuple[str, str | None]]:
    """Return the (file name, reason code) of a job's documents in one status, by name."""
    found = service.journal.documents_blocking(job, status=status)
    return sorted((Path(entry.source).name, entry.reason_code) for entry in found)


async def test_a_hung_parser_costs_one_document_and_its_worker_is_recovered(
    tmp_path: Path, corpus: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A hung parser stalled its job for ever: the pool future was awaited with no timeout.

    The pool holds one worker, so the two documents after the hung one can only have been
    parsed by a worker that replaced it.
    """
    monkeypatch.setattr(cpu_pool, "parse_and_chunk", process_targets.parse_after_a_delay)
    before = worker_pids()
    configured = settings(
        workers={"cpu_pool_size": 1, "embedding_pool_size": 1, "queue_depth": 200},
        ingestion={"parse_timeout_seconds": 3},
    )
    service, adapter = build(tmp_path, configured)

    record = await service.accept(
        [str(corpus / name) for name in (NAMES[0], HUNG, NAMES[1], NAMES[2])]
    )
    settled = await service.run(record)

    dead = service.journal.dead_lettered_blocking(record.job_id)
    assert outcomes(service, record.job_id, "dead_lettered") == [(HUNG, "PARSE_TIMEOUT")]
    assert "ingestion.parse_timeout_seconds" in (dead[0].detail or "")
    assert outcomes(service, record.job_id, "indexed") == sorted((name, None) for name in NAMES)
    assert settled.status == "partial"
    assert settled.counts["parsed"] == 3
    assert settled.counts["dead_lettered"] == 1
    assert {Path(point.payload["source_uri"]).name for point in adapter.points} == set(NAMES)
    assert await surviving_workers(before) == set()


async def test_documents_in_flight_beside_a_hung_one_are_parsed_again_not_lost(
    tmp_path: Path,
    corpus: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Recycling kills every worker, so a document mid-parse beside the hung one would vanish.

    The stalled document is submitted when the paused one returns, two seconds after the hung
    one, and its first parse blocks, so it is mid-parse and not overdue when the hung one's
    deadline passes. Only a second parse of it returns, so it is indexed only if resubmitted.
    """
    monkeypatch.setattr(cpu_pool, "parse_and_chunk", process_targets.parse_after_a_delay)
    before = worker_pids()
    configured = settings(
        workers={"cpu_pool_size": 2, "embedding_pool_size": 1, "queue_depth": 200},
        ingestion={"parse_timeout_seconds": 4},
    )
    service, adapter = build(tmp_path, configured)
    order = (PAUSED, HUNG, STALLED, NAMES[0], NAMES[1])

    record = await service.accept([str(corpus / name) for name in order])
    with caplog.at_level(logging.WARNING, logger="fasterrag.workers.cpu_pool"):
        settled = await service.run(record)

    recycles = [entry for entry in caplog.records if "recycled" in entry.getMessage()]
    survivors = sorted((name, None) for name in order if name != HUNG)
    assert outcomes(service, record.job_id, "dead_lettered") == [(HUNG, "PARSE_TIMEOUT")]
    assert outcomes(service, record.job_id, "indexed") == survivors
    assert [getattr(entry, "resubmitted", None) for entry in recycles] == [1]
    assert process_targets.marker(corpus / STALLED, "started").exists()
    assert settled.status == "partial"
    assert settled.counts["parsed"] == 4
    assert len({point.point_id for point in adapter.points}) == len(adapter.points)
    assert await surviving_workers(before) == set()


@contextmanager
def released_after(source: Path, seconds: float) -> Iterator[None]:
    """Release a blocked parse of ``source`` after ``seconds``, from a thread.

    A pool that waits for the parse blocks the event loop, so only a thread can end the wait;
    without this a regression would hang the run for an hour instead of failing it.
    """
    timer = threading.Timer(seconds, process_targets.release, args=(source,))
    timer.start()
    try:
        yield
    finally:
        timer.cancel()


async def parse_started(source: Path) -> None:
    """Return once a blocking parse of ``source`` has begun in a worker."""
    deadline = time.monotonic() + PATIENCE * 6
    while not process_targets.marker(source, "started").exists():
        assert time.monotonic() < deadline, f"the parse of {source.name} never started"
        await asyncio.sleep(0.05)


async def test_a_job_cancelled_beside_a_hung_parser_settles_within_the_grace_period(
    tmp_path: Path, corpus: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """TASK-0525: a drain waited for the hung parse, because the pool's exit joined its workers."""
    monkeypatch.setattr(cpu_pool, "parse_and_chunk", process_targets.parse_after_a_delay)
    before = worker_pids()
    configured = settings(
        reliability={
            "retries": {"max_attempts": 0, "backoff_base_ms": 1, "jitter": False},
            "shutdown_grace_seconds": 1,
        }
    )
    grace = configured.reliability.shutdown_grace_seconds
    service, _ = build(tmp_path, configured)
    registry = JobRegistry()
    record = await service.accept([str(corpus / name) for name in (NAMES[0], HUNG, NAMES[1])])

    async def work() -> None:
        await service.run(record)

    await registry.submit(record.job_id, work())
    await parse_started(corpus / HUNG)
    with released_after(corpus / HUNG, PATIENCE * 3):
        began = time.monotonic()
        drained = await registry.drain(grace)
        took = time.monotonic() - began

    settled = service.journal.load_job_blocking(record.job_id)
    assert drained.cancelled == 1
    assert took < grace + PATIENCE
    assert (settled.status, settled.reason_code) == ("failed", INTERRUPTED)
    assert await surviving_workers(before) == set()


async def test_a_pool_with_no_parse_timeout_is_left_without_waiting_for_a_hung_parse(
    corpus: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """TASK-0509: with parse_timeout_seconds 0, leaving the pool joined a hung worker for good."""
    monkeypatch.setattr(cpu_pool, "parse_and_chunk", process_targets.parse_after_a_delay)
    before = worker_pids()
    configured = settings(ingestion={"parse_timeout_seconds": 0})
    queue: BoundedQueue[ChunkPayload] = BoundedQueue(1000)

    with released_after(corpus / HUNG, PATIENCE * 3):
        async with CpuWorkerPool(configured) as pool:
            tasks = pool.tasks_for([str(corpus / name) for name in (NAMES[0], HUNG)])
            processing = asyncio.create_task(pool.process(tasks, queue))
            await parse_started(corpus / HUNG)
            began = time.monotonic()
            processing.cancel()
            await asyncio.gather(processing, return_exceptions=True)
        took = time.monotonic() - began

    assert processing.cancelled()
    assert took < PATIENCE
    assert await surviving_workers(before) == set()


async def test_a_worker_killed_mid_job_is_not_blamed_on_a_missing_main_guard(
    tmp_path: Path, corpus: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """TASK-0509: a worker the OS killed mid-job was reported as a script with no main guard."""
    monkeypatch.setattr(cpu_pool, "parse_and_chunk", process_targets.die)
    service, _ = build(tmp_path, settings())

    record = await service.accept([str(corpus / name) for name in NAMES])
    with pytest.raises(IngestionError) as raised:
        await service.run(record)

    assert raised.value.code is ErrorCode.CHUNK_FAILED
    assert "__main__" not in raised.value.detail
    assert "out-of-memory" in raised.value.detail
    assert NAMES[0] in raised.value.detail


def semantic(**overrides: Any) -> Settings:
    """Return settings for the semantic strategy over the default local model."""
    chunking = {"strategy": "semantic", "token_counter": "estimate", "chunk_size": 64, "overlap": 8}
    return settings(chunking=chunking, **overrides)


def recording_executor(log_dir: Path, load_seconds: float) -> Callable[[int], Executor]:
    """Return a real process pool whose workers build a recording stand-in embedder."""

    def factory(workers: int) -> Executor:
        return ProcessPoolExecutor(
            max_workers=workers,
            initializer=process_targets.record_sentence_embedders,
            initargs=(str(log_dir), load_seconds),
        )

    return factory


def builds(log_dir: Path) -> dict[int, list[str]]:
    """Return, per process id, the model names that process built an embedder for."""
    return {int(log.stem): log.read_text(encoding="utf-8").split() for log in log_dir.glob("*.log")}


async def test_each_parse_worker_builds_its_own_sentence_embedder_once(
    tmp_path: Path, corpus: Path
) -> None:
    """The parse worker had no embedder, so a semantic job could not run in the pipeline."""
    log_dir = tmp_path / "builds"
    log_dir.mkdir()
    configured = semantic()
    queue: BoundedQueue[ChunkPayload] = BoundedQueue(1000)

    async with CpuWorkerPool(configured, executor_factory=recording_executor(log_dir, 0)) as pool:
        report = await pool.process(pool.tasks_for([str(corpus / name) for name in NAMES]), queue)

    chunks = await queue.drain()
    built = builds(log_dir)
    assert report.parsed == 3
    assert report.dead_lettered == 0
    assert {payload.chunk.strategy for payload in chunks} == {"semantic"}
    assert len(built) == configured.workers.cpu_pool_size
    assert os.getpid() not in built
    assert all(models == [configured.embeddings.model] for models in built.values())
    assert cpu_pool._sentence_embedders == {}


async def test_a_model_load_is_never_billed_to_a_documents_parse_deadline(
    tmp_path: Path, corpus: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A worker loading its model inside a parse would be killed as a hung parser.

    Loading takes three seconds against a two-second deadline, in the first worker and again
    in the one that replaces it after the hung document, and no healthy document times out.
    """
    monkeypatch.setattr(cpu_pool, "parse_and_chunk", process_targets.parse_after_a_delay)
    log_dir = tmp_path / "builds"
    log_dir.mkdir()
    configured = semantic(
        workers={"cpu_pool_size": 1, "embedding_pool_size": 1, "queue_depth": 200},
        ingestion={"parse_timeout_seconds": 2},
    )
    queue: BoundedQueue[ChunkPayload] = BoundedQueue(1000)
    order = (NAMES[0], HUNG, NAMES[1])

    async with CpuWorkerPool(configured, executor_factory=recording_executor(log_dir, 3)) as pool:
        report = await pool.process(pool.tasks_for([str(corpus / name) for name in order]), queue)

    chunks = await queue.drain()
    assert report.dead_lettered == 1
    assert report.parsed == 2
    assert {Path(payload.source).name for payload in chunks} == {NAMES[0], NAMES[1]}
    assert len(builds(log_dir)) == 2
