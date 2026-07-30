import asyncio
from concurrent.futures import Executor, ThreadPoolExecutor
from pathlib import Path
from typing import Any

import pytest

from fasterrag.config.schema import Settings
from fasterrag.services.journal import Journal
from fasterrag.workers.cpu_pool import CpuWorkerPool, resolve_pool_size
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

    entries = journal.dead_lettered("job_1")
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
    assert journal.dead_lettered("job_1")[0].reason_code == "PAYLOAD_TOO_LARGE"


async def test_a_missing_file_is_dead_lettered(corpus: Path, journal: Journal) -> None:
    async with CpuWorkerPool(settings(), journal=journal, executor_factory=threads) as pool:
        report = await pool.process(
            CpuWorkerPool.tasks_for([str(corpus / "absent.md")]),
            BoundedQueue(200),
            job="job_1",
        )

    assert report.dead_lettered == 1
    assert journal.dead_lettered("job_1")[0].reason_code == "PARSE_FAILED"


async def test_resuming_skips_documents_a_checkpoint_already_covered(corpus: Path) -> None:
    queue: BoundedQueue[ChunkPayload] = BoundedQueue(400)

    async with CpuWorkerPool(settings(), executor_factory=threads) as pool:
        report = await pool.process(
            pool.tasks_for(sources(corpus, "a.md", "b.md", "c.md")), queue, resume_from=2
        )

    assert report.skipped == 2
    assert report.parsed == 1


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

    assert journal.counts("job_1") == {"indexed": 2, "total": 2}


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
