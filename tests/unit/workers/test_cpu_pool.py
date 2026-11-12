import asyncio
import threading
import time
from collections.abc import Callable, Iterator
from concurrent.futures import Executor, ThreadPoolExecutor
from concurrent.futures.process import BrokenProcessPool
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from fasterrag.config.schema import Settings
from fasterrag.core.identity import content_hash
from fasterrag.errors import ConfigError, ErrorCode, IngestionError
from fasterrag.services.journal import Journal
from fasterrag.workers import cpu_pool
from fasterrag.workers.cpu_pool import (
    EMPTY_DOCUMENT,
    CpuWorkerPool,
    parse_and_chunk,
    resolve_pool_size,
)
from fasterrag.workers.ledger import IndexLedger
from fasterrag.workers.queues import BoundedQueue, ChunkPayload

BODY = "# Title\n\n" + ("sentence body text. " * 30)


def settings(**overrides: Any) -> Settings:
    payload: dict[str, Any] = {"chunking": {"chunk_size": 64, "overlap": 8}}
    payload.update(overrides)
    return Settings.model_validate(payload)


@pytest.fixture
def corpus(tmp_path: Path) -> Path:
    for name in ("a.md", "b.md", "c.md"):
        (tmp_path / name).write_text(f"# {name}\n\n{BODY}", encoding="utf-8")
    return tmp_path


@pytest.fixture
def journal(tmp_path: Path) -> Journal:
    return Journal(tmp_path / "journal", checkpoint_every=2)


def threads(workers: int) -> Executor:
    """Run parse tasks in threads so tests stay fast and deterministic."""
    return ThreadPoolExecutor(max_workers=workers)


def sources(corpus: Path, *names: str) -> list[str]:
    return [str(corpus / name) for name in names]


@pytest.mark.parametrize(("configured", "expected_at_least"), [(1, 1), (4, 4)])
def test_an_explicit_pool_size_is_honoured(configured: int, expected_at_least: int) -> None:
    assert resolve_pool_size(configured) == expected_at_least


def test_zero_expands_to_the_cpu_count() -> None:
    assert resolve_pool_size(0) >= 1


async def test_using_the_pool_without_starting_it_is_an_error(corpus: Path) -> None:
    pool = CpuWorkerPool(settings(), executor_factory=threads)
    queue: BoundedQueue[ChunkPayload] = BoundedQueue(10)

    with pytest.raises(Exception, match="not running"):
        await pool.process(pool.tasks_for(sources(corpus, "a.md")), queue)


async def test_documents_are_parsed_chunked_and_streamed(corpus: Path) -> None:
    queue: BoundedQueue[ChunkPayload] = BoundedQueue(200)

    async with CpuWorkerPool(settings(), executor_factory=threads) as pool:
        report = await pool.process(pool.tasks_for(sources(corpus, "a.md", "b.md")), queue)

    assert report.parsed == 2
    assert report.chunked > 2
    assert queue.depth == report.chunked


async def test_streamed_chunks_carry_deterministic_ids_and_context(corpus: Path) -> None:
    queue: BoundedQueue[ChunkPayload] = BoundedQueue(200)

    async with CpuWorkerPool(settings(), executor_factory=threads) as pool:
        await pool.process(pool.tasks_for(sources(corpus, "a.md")), queue)

    chunks = list(await queue.drain())
    assert chunks
    first = chunks[0]
    assert first.chunk_id.startswith("c_")
    assert first.document_id.startswith("d_")
    assert first.source.endswith("a.md")
    assert len(first.content_hash) == 64
    assert first.chunk.strategy == "recursive"


async def test_chunk_ids_are_stable_across_runs(corpus: Path) -> None:
    async def run() -> list[str]:
        queue: BoundedQueue[ChunkPayload] = BoundedQueue(200)
        async with CpuWorkerPool(settings(), executor_factory=threads) as pool:
            await pool.process(pool.tasks_for(sources(corpus, "a.md")), queue)
        return [payload.chunk_id for payload in await queue.drain()]

    assert await run() == await run()


async def test_a_duplicate_document_is_skipped_on_the_second_run(
    corpus: Path, journal: Journal
) -> None:
    first_queue: BoundedQueue[ChunkPayload] = BoundedQueue(200)
    second_queue: BoundedQueue[ChunkPayload] = BoundedQueue(200)
    tasks = CpuWorkerPool.tasks_for(sources(corpus, "a.md"))

    async with CpuWorkerPool(settings(), journal=journal, executor_factory=threads) as pool:
        first = await pool.process(tasks, first_queue, job="job_1")
        second = await pool.process(tasks, second_queue, job="job_2")

    assert first.parsed == 1
    assert second.parsed == 0
    assert second.deduplicated == 1
    assert second_queue.depth == 0


async def test_deduplication_can_be_switched_off(corpus: Path, journal: Journal) -> None:
    tasks = CpuWorkerPool.tasks_for(sources(corpus, "a.md"))
    configured = settings(ingestion={"dedup": False})

    async with CpuWorkerPool(configured, journal=journal, executor_factory=threads) as pool:
        await pool.process(tasks, BoundedQueue(200), job="job_1")
        second = await pool.process(tasks, BoundedQueue(200), job="job_2")

    assert second.parsed == 1
    assert second.deduplicated == 0


async def test_an_unreadable_document_is_dead_lettered_and_the_pass_continues(
    corpus: Path, journal: Journal
) -> None:
    (corpus / "broken.zip").write_bytes(b"PK\x03\x04not a document")
    order = sources(corpus, "broken.zip", "a.md")
    queue: BoundedQueue[ChunkPayload] = BoundedQueue(200)

    async with CpuWorkerPool(settings(), journal=journal, executor_factory=threads) as pool:
        report = await pool.process(CpuWorkerPool.tasks_for(order), queue, job="job_1")

    assert report.dead_lettered == 1
    assert report.parsed == 1

    entries = journal.dead_lettered_blocking("job_1")
    assert len(entries) == 1
    assert entries[0].reason_code == "PARSE_FAILED"
    assert entries[0].source.endswith("broken.zip")


async def test_an_oversized_document_is_refused_before_it_is_read(
    corpus: Path, journal: Journal
) -> None:
    oversized = "word " * 300_000
    (corpus / "big.md").write_text(oversized, encoding="utf-8")
    configured = settings(ingestion={"max_document_mb": 1})

    async with CpuWorkerPool(configured, journal=journal, executor_factory=threads) as pool:
        report = await pool.process(
            CpuWorkerPool.tasks_for(sources(corpus, "big.md")), BoundedQueue(200), job="job_1"
        )

    assert report.dead_lettered == 1
    assert report.parsed == 0
    assert journal.dead_lettered_blocking("job_1")[0].reason_code == "PAYLOAD_TOO_LARGE"


async def test_a_missing_file_is_dead_lettered(corpus: Path, journal: Journal) -> None:
    async with CpuWorkerPool(settings(), journal=journal, executor_factory=threads) as pool:
        report = await pool.process(
            CpuWorkerPool.tasks_for([str(corpus / "absent.md")]),
            BoundedQueue(200),
            job="job_1",
        )

    assert report.dead_lettered == 1
    assert journal.dead_lettered_blocking("job_1")[0].reason_code == "PARSE_FAILED"


async def test_resuming_skips_documents_a_checkpoint_already_covered(corpus: Path) -> None:
    queue: BoundedQueue[ChunkPayload] = BoundedQueue(400)

    async with CpuWorkerPool(settings(), executor_factory=threads) as pool:
        report = await pool.process(
            pool.tasks_for(sources(corpus, "a.md", "b.md", "c.md")), queue, resume_from=2
        )

    assert report.skipped == 2
    assert report.parsed == 1


async def test_the_pool_loads_the_tokenizer_before_it_accepts_work(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Startup pays the tokenizer load, so the first document does not."""
    counted: list[str] = []

    class RecordingCounter:
        chars_per_token = 4

        def count(self, text: str) -> int:
            counted.append(text)
            return len(text)

    monkeypatch.setattr(
        "fasterrag.workers.cpu_pool.create_token_counter", lambda _settings: RecordingCounter()
    )

    async with CpuWorkerPool(settings(), executor_factory=threads):
        assert counted, "the pool accepted work without loading the tokenizer"


async def test_a_full_queue_makes_the_parser_wait(corpus: Path) -> None:
    queue: BoundedQueue[ChunkPayload] = BoundedQueue(2)

    async with CpuWorkerPool(settings(), executor_factory=threads) as pool:
        parsing = asyncio.create_task(
            pool.process(pool.tasks_for(sources(corpus, "a.md", "b.md")), queue)
        )
        await asyncio.sleep(0.05)

        assert parsing.done() is False
        assert queue.full is True

        while not parsing.done():
            await queue.get()
            queue.task_done()
            await asyncio.sleep(0)

        await parsing


async def test_document_outcomes_are_journalled(corpus: Path, journal: Journal) -> None:
    async with CpuWorkerPool(settings(), journal=journal, executor_factory=threads) as pool:
        await pool.process(
            pool.tasks_for(sources(corpus, "a.md", "b.md")), BoundedQueue(400), job="job_1"
        )

    assert journal.counts_blocking("job_1") == {"indexed": 2, "total": 2}


async def test_parse_flags_are_counted(corpus: Path) -> None:
    (corpus / "table.md").write_text(
        "| Term | Days |\n| --- | --- |\n| Notice | 30 |\n", encoding="utf-8"
    )

    async with CpuWorkerPool(settings(), executor_factory=threads) as pool:
        report = await pool.process(pool.tasks_for(sources(corpus, "table.md")), BoundedQueue(50))

    assert report.flags.get("tables_detected") == 1


def test_tasks_are_ordered_and_carry_metadata(corpus: Path) -> None:
    tasks = CpuWorkerPool.tasks_for(
        sources(corpus, "a.md", "b.md"), tenant="acme", metadata={"department": "legal"}
    )

    assert [task.index for task in tasks] == [0, 1]
    assert all(task.tenant == "acme" for task in tasks)
    assert all(task.metadata["department"] == "legal" for task in tasks)


def test_tenants_get_distinct_document_ids(corpus: Path) -> None:
    first = CpuWorkerPool.tasks_for(sources(corpus, "a.md"), tenant="acme")[0]
    second = CpuWorkerPool.tasks_for(sources(corpus, "a.md"), tenant="globex")[0]

    assert first.document_id != second.document_id


async def test_the_executor_is_shut_down_on_exit(corpus: Path) -> None:
    created: list[Executor] = []

    def recording(workers: int) -> Executor:
        executor = ThreadPoolExecutor(max_workers=workers)
        created.append(executor)
        return executor

    async with CpuWorkerPool(settings(), executor_factory=recording):
        pass

    assert len(created) == 1
    with pytest.raises(RuntimeError, match="shutdown"):
        created[0].submit(int, "1")


def test_a_broken_pool_is_reported_as_a_missing_main_guard() -> None:
    """A raw BrokenProcessPool names neither the cause nor the fix.

    It is also the first thing a library user embedding fasterRag on Windows hits.
    """
    from fasterrag.workers.cpu_pool import _unguarded_entry_point_error

    error = _unguarded_entry_point_error()

    assert "__main__" in error.detail
    assert not error.retryable


async def test_documents_are_parsed_concurrently_up_to_the_pool_size(
    corpus: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Awaiting each document before submitting the next keeps exactly one worker busy."""
    real = parse_and_chunk
    lock = threading.Lock()
    running = 0
    peak = 0

    def slow(task: Any, configured: Settings) -> Any:
        nonlocal running, peak
        with lock:
            running += 1
            peak = max(peak, running)
        time.sleep(0.05)
        try:
            return real(task, configured)
        finally:
            with lock:
                running -= 1

    monkeypatch.setattr(cpu_pool, "parse_and_chunk", slow)
    queue: BoundedQueue[ChunkPayload] = BoundedQueue(500)

    async with CpuWorkerPool(
        settings(workers={"cpu_pool_size": 3}), executor_factory=threads
    ) as pool:
        report = await pool.process(pool.tasks_for(sources(corpus, "a.md", "b.md", "c.md")), queue)

    assert report.parsed == 3
    assert peak > 1


async def test_parallel_parsing_keeps_submission_order(corpus: Path) -> None:
    queue: BoundedQueue[ChunkPayload] = BoundedQueue(500)

    async with CpuWorkerPool(
        settings(workers={"cpu_pool_size": 3}), executor_factory=threads
    ) as pool:
        tasks = pool.tasks_for(sources(corpus, "a.md", "b.md", "c.md"))
        await pool.process(tasks, queue)

    seen: list[str] = []
    for payload in await queue.drain():
        if payload.document_id not in seen:
            seen.append(payload.document_id)

    assert seen == [task.document_id for task in tasks]


async def test_an_untyped_parser_failure_costs_one_document_not_the_job(
    corpus: Path, journal: Journal, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A parser library raising its own exception type used to abort the whole pass."""
    real = cpu_pool._parse_and_chunk

    def flaky(task: Any, configured: Settings) -> Any:
        if task.source.endswith("b.md"):
            raise RecursionError("maximum recursion depth exceeded")
        return real(task, configured)

    monkeypatch.setattr(cpu_pool, "_parse_and_chunk", flaky)
    queue: BoundedQueue[ChunkPayload] = BoundedQueue(500)

    async with CpuWorkerPool(settings(), journal=journal, executor_factory=threads) as pool:
        job = journal.create_job_blocking("default", [])
        report = await pool.process(
            pool.tasks_for(sources(corpus, "a.md", "b.md", "c.md")), queue, job=job.job_id
        )

    dead = journal.dead_lettered_blocking(job.job_id)
    assert report.parsed == 2
    assert report.dead_lettered == 1
    assert [entry.reason_code for entry in dead] == ["PARSE_FAILED"]
    assert "RecursionError" in (dead[0].detail or "")


async def test_with_a_ledger_nothing_is_journalled_until_the_index_side_reports(
    corpus: Path, journal: Journal
) -> None:
    """The ordering defect: 'indexed' was written when the chunks were merely enqueued."""
    queue: BoundedQueue[ChunkPayload] = BoundedQueue(500)
    ledger = IndexLedger()

    async with CpuWorkerPool(settings(), journal=journal, executor_factory=threads) as pool:
        job = journal.create_job_blocking("default", [])
        await pool.process(
            pool.tasks_for(sources(corpus, "a.md")), queue, job=job.job_id, ledger=ledger
        )

    assert journal.documents_blocking(job.job_id, status="indexed") == []
    assert journal.known_content_blocking("default") == {}

    await ledger.written(list(await queue.drain()))

    assert len(journal.documents_blocking(job.job_id, status="indexed")) == 1
    assert len(journal.known_content_blocking("default")) == 1


async def test_a_document_with_nothing_to_index_is_skipped_with_a_reason(
    corpus: Path, journal: Journal
) -> None:
    """A document that produced no chunks was journalled as indexed and kept for dedup."""
    (corpus / "blank.md").write_bytes(b"")
    queue: BoundedQueue[ChunkPayload] = BoundedQueue(500)

    async with CpuWorkerPool(settings(), journal=journal, executor_factory=threads) as pool:
        job = journal.create_job_blocking("default", [])
        tasks = pool.tasks_for(sources(corpus, "blank.md", "a.md"))
        report = await pool.process(tasks, queue, job=job.job_id)

    skipped = journal.documents_blocking(job.job_id, status="skipped")
    assert report.empty == 1
    assert report.parsed == 1
    assert [(entry.document_id, entry.reason_code) for entry in skipped] == [
        (tasks[0].document_id, EMPTY_DOCUMENT)
    ]
    assert "blank.md" in (skipped[0].detail or "")
    assert list(journal.known_content_blocking("default").values()) == [tasks[1].document_id]
    assert {payload.document_id for payload in await queue.drain()} == {tasks[1].document_id}


async def test_an_empty_document_is_skipped_even_where_an_older_run_remembered_its_hash(
    corpus: Path, journal: Journal
) -> None:
    """A dedup file written before the fix holds the hash of nothing; it must not match."""
    (corpus / "blank.md").write_bytes(b"")
    journal.remember_content_blocking("default", content_hash(b""), "d_from_an_older_run")
    queue: BoundedQueue[ChunkPayload] = BoundedQueue(500)

    async with CpuWorkerPool(settings(), journal=journal, executor_factory=threads) as pool:
        job = journal.create_job_blocking("default", [])
        report = await pool.process(
            pool.tasks_for(sources(corpus, "blank.md")), queue, job=job.job_id
        )

    assert report.empty == 1
    assert report.deduplicated == 0
    assert journal.documents_blocking(job.job_id, status="deduplicated") == []


async def test_a_skipped_document_lets_the_checkpoint_pass_it(corpus: Path) -> None:
    """A skipped document that never settled would pin the resume point behind it for ever."""
    (corpus / "blank.md").write_bytes(b"")
    queue: BoundedQueue[ChunkPayload] = BoundedQueue(500)
    reached: list[int] = []

    async def on_progress(index: int) -> None:
        reached.append(index)

    async with CpuWorkerPool(settings(), executor_factory=threads) as pool:
        await pool.process(
            pool.tasks_for(sources(corpus, "a.md", "blank.md", "b.md")),
            queue,
            on_progress=on_progress,
        )

    assert reached == [0, 1, 2]


class Stalling:
    """A parse stand-in that holds chosen documents until the test lets them go.

    A thread cannot be killed, so a held parse outlives the pool that was abandoned around
    it; ``release`` is what lets those threads end when the test does.
    """

    def __init__(
        self,
        *,
        hang: tuple[str, ...] = (),
        hang_once: tuple[str, ...] = (),
        delay: dict[str, float] | None = None,
    ) -> None:
        self.hang = hang
        self.hang_once = hang_once
        self.delay = delay or {}
        self.attempts: dict[str, int] = {}
        self.release = threading.Event()
        self._lock = threading.Lock()

    def __call__(self, task: Any, configured: Settings) -> Any:
        name = Path(task.source).name
        with self._lock:
            self.attempts[name] = attempt = self.attempts.get(name, 0) + 1
        if name in self.hang or (name in self.hang_once and attempt == 1):
            self.release.wait(30)
        time.sleep(self.delay.get(name, 0.0))
        return parse_and_chunk(task, configured)


@pytest.fixture
def stalling(monkeypatch: pytest.MonkeyPatch) -> Iterator[Callable[..., Stalling]]:
    """Install a :class:`Stalling` parser, and release whatever it still holds afterwards."""
    installed: list[Stalling] = []

    def install(**kwargs: Any) -> Stalling:
        parser = Stalling(**kwargs)
        monkeypatch.setattr(cpu_pool, "parse_and_chunk", parser)
        installed.append(parser)
        return parser

    yield install
    for parser in installed:
        parser.release.set()


class Factory:
    """Builds thread-backed executors and remembers each one it built."""

    def __init__(self) -> None:
        self.built: list[ThreadPoolExecutor] = []

    def __call__(self, workers: int) -> Executor:
        self.built.append(ThreadPoolExecutor(max_workers=workers))
        return self.built[-1]


def timed(seconds: float, *, workers: int = 2) -> Settings:
    return settings(
        workers={"cpu_pool_size": workers}, ingestion={"parse_timeout_seconds": seconds}
    )


async def test_a_parse_that_outlives_the_timeout_is_dead_lettered_and_the_pass_continues(
    corpus: Path, journal: Journal, stalling: Callable[..., Stalling]
) -> None:
    """A hung parser stalled its job for ever: the pool future was awaited with no timeout."""
    stalling(hang=("b.md",))
    factory = Factory()
    queue: BoundedQueue[ChunkPayload] = BoundedQueue(500)

    async with CpuWorkerPool(timed(0.3), journal=journal, executor_factory=factory) as pool:
        job = journal.create_job_blocking("default", [])
        tasks = pool.tasks_for(sources(corpus, "a.md", "b.md", "c.md"))
        report = await pool.process(tasks, queue, job=job.job_id)

    dead = journal.dead_lettered_blocking(job.job_id)
    assert report.parsed == 2
    assert report.dead_lettered == 1
    assert [(entry.document_id, entry.reason_code) for entry in dead] == [
        (tasks[1].document_id, "PARSE_TIMEOUT")
    ]
    assert "ingestion.parse_timeout_seconds" in (dead[0].detail or "")
    assert dead[0].trace_id is not None
    assert len(factory.built) == 2
    assert {payload.document_id for payload in await queue.drain()} == {
        tasks[0].document_id,
        tasks[2].document_id,
    }


async def test_a_document_in_flight_beside_an_overdue_one_is_submitted_again(
    corpus: Path, stalling: Callable[..., Stalling]
) -> None:
    """Recycling the pool abandons whatever else it was parsing; that work must be redone.

    ``a.md`` takes long enough that ``c.md`` is submitted well after ``b.md``, so when
    ``b.md`` runs out of time ``c.md`` is mid-parse and still has time of its own left.
    """
    parser = stalling(hang=("b.md",), hang_once=("c.md",), delay={"a.md": 0.5})
    queue: BoundedQueue[ChunkPayload] = BoundedQueue(500)

    async with CpuWorkerPool(timed(1.0), executor_factory=threads) as pool:
        tasks = pool.tasks_for(sources(corpus, "a.md", "b.md", "c.md"))
        report = await pool.process(tasks, queue)

    streamed = [payload.chunk_id for payload in await queue.drain()]
    assert report.parsed == 2
    assert report.dead_lettered == 1
    assert parser.attempts == {"a.md": 1, "b.md": 1, "c.md": 2}
    assert len(set(streamed)) == len(streamed)


async def test_a_document_that_finished_beside_an_overdue_one_is_not_parsed_twice(
    corpus: Path, stalling: Callable[..., Stalling]
) -> None:
    """A finished parse waiting its turn behind a hung one keeps the result it already has."""
    parser = stalling(hang=("a.md",))
    queue: BoundedQueue[ChunkPayload] = BoundedQueue(500)

    async with CpuWorkerPool(timed(0.3), executor_factory=threads) as pool:
        report = await pool.process(pool.tasks_for(sources(corpus, "a.md", "b.md")), queue)

    assert report.parsed == 1
    assert report.dead_lettered == 1
    assert parser.attempts == {"a.md": 1, "b.md": 1}


async def test_two_documents_overdue_together_cost_one_timeout_not_two(
    corpus: Path, stalling: Callable[..., Stalling]
) -> None:
    """A hung neighbour that was resubmitted would hang through a second whole deadline."""
    parser = stalling(hang=("a.md", "b.md"))
    factory = Factory()
    queue: BoundedQueue[ChunkPayload] = BoundedQueue(500)

    async with CpuWorkerPool(timed(0.3), executor_factory=factory) as pool:
        report = await pool.process(pool.tasks_for(sources(corpus, "a.md", "b.md", "c.md")), queue)

    assert report.dead_lettered == 2
    assert report.parsed == 1
    assert parser.attempts == {"a.md": 1, "b.md": 1, "c.md": 1}
    assert len(factory.built) == 2


async def test_a_timeout_of_zero_waits_for_as_long_as_the_parse_takes(
    corpus: Path, stalling: Callable[..., Stalling]
) -> None:
    stalling(delay={"a.md": 0.3})
    factory = Factory()
    queue: BoundedQueue[ChunkPayload] = BoundedQueue(500)

    async with CpuWorkerPool(timed(0, workers=1), executor_factory=factory) as pool:
        report = await pool.process(pool.tasks_for(sources(corpus, "a.md", "b.md")), queue)

    assert report.parsed == 2
    assert report.dead_lettered == 0
    assert len(factory.built) == 1


async def test_a_parse_inside_its_timeout_is_left_alone(
    corpus: Path, stalling: Callable[..., Stalling]
) -> None:
    stalling(delay={"a.md": 0.1})
    factory = Factory()
    queue: BoundedQueue[ChunkPayload] = BoundedQueue(500)

    async with CpuWorkerPool(timed(30), executor_factory=factory) as pool:
        report = await pool.process(pool.tasks_for(sources(corpus, "a.md", "b.md")), queue)

    assert report.parsed == 2
    assert len(factory.built) == 1


class Unstartable(ThreadPoolExecutor):
    """An executor whose workers die before running anything, as an unguarded script's do."""

    def submit(self, fn: Any, /, *args: Any, **kwargs: Any) -> Any:
        raise BrokenProcessPool("a child process terminated abruptly")


async def test_a_pool_that_cannot_start_a_worker_says_so_and_is_not_left_running() -> None:
    """A worker that never starts is reported when the pool starts, not at its first document."""
    built: list[Unstartable] = []

    def factory(workers: int) -> Executor:
        built.append(Unstartable(max_workers=workers))
        return built[-1]

    pool = CpuWorkerPool(settings(), executor_factory=factory)
    with pytest.raises(IngestionError, match="__main__"):
        await pool.__aenter__()

    assert pool._executor is None
    assert built[0]._shutdown


class BreaksAfter(ThreadPoolExecutor):
    """An executor that accepts ``healthy`` tasks and then refuses, as a broken pool does."""

    def __init__(self, max_workers: int, healthy: int) -> None:
        super().__init__(max_workers=max_workers)
        self.healthy = healthy

    def submit(self, fn: Any, /, *args: Any, **kwargs: Any) -> Any:
        if self.healthy <= 0:
            raise BrokenProcessPool("A child process terminated abruptly")
        self.healthy -= 1
        return super().submit(fn, *args, **kwargs)


async def test_a_pool_that_breaks_between_documents_fails_with_the_typed_error(
    corpus: Path,
) -> None:
    """TASK-0509: a pool broken before the next submit raised a raw BrokenProcessPool."""
    configured = settings(workers={"cpu_pool_size": 1})
    queue: BoundedQueue[ChunkPayload] = BoundedQueue(500)

    def factory(workers: int) -> Executor:
        return BreaksAfter(workers, healthy=2)

    async with CpuWorkerPool(configured, executor_factory=factory) as pool:
        with pytest.raises(IngestionError) as raised:
            await pool.process(pool.tasks_for(sources(corpus, "a.md", "b.md")), queue)

    assert raised.value.code is ErrorCode.CHUNK_FAILED
    assert "b.md" in raised.value.detail
    assert "__main__" not in raised.value.detail


def test_the_parse_timeout_is_on_by_default_and_cannot_be_negative() -> None:
    assert Settings().ingestion.parse_timeout_seconds == 600
    assert Settings.model_validate({"ingestion": {"parse_timeout_seconds": 0}})

    with pytest.raises(ValidationError):
        Settings.model_validate({"ingestion": {"parse_timeout_seconds": -1}})


async def test_a_semantic_pool_refuses_a_hosted_embedding_provider() -> None:
    """A hosted provider's client cannot run in a parse worker, so none may try to build one."""
    configured = settings(
        chunking={"strategy": "semantic", "chunk_size": 64, "overlap": 8},
        embeddings={"provider": "openai", "model": "text-embedding-3-small", "api_key_env": "K"},
    )

    with pytest.raises(ConfigError) as refused:
        async with CpuWorkerPool(configured, executor_factory=threads):
            pass

    assert "needs a local embedding model" in refused.value.detail
    assert "embeddings.provider" in refused.value.detail
    assert cpu_pool._sentence_embedders == {}
