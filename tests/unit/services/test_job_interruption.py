"""A job the process stops working on is settled, never left ``running`` (TASK-0436).

``asyncio.CancelledError`` is a ``BaseException``. Every handler that settled a failed job
caught ``Exception``, so cancelling one — which is what a shutdown does — settled nothing,
left both halves of the pipeline running, and could not be told apart from slow progress.
"""

import asyncio
import threading
from collections.abc import Sequence
from pathlib import Path

import pytest

from fasterrag.adapters.embeddings.base import EmbeddingResult
from fasterrag.adapters.embeddings.tiering import TieringRouter
from fasterrag.config.schema import Settings
from fasterrag.services.ingestion import IngestionService
from fasterrag.services.jobs import INTERRUPTED, settle_interrupted
from fasterrag.services.journal import JobRecord, Journal
from tests.unit.services.test_ingestion import KeywordLegAdapter, StubEmbedder, settings, threads

WAIT_SECONDS = 10
QUEUE_DEPTH = 10


class HeldEmbedder(StubEmbedder):
    """Embeds normally until it meets ``marker``, then waits to be released."""

    def __init__(self, configured: Settings, marker: str = "") -> None:
        super().__init__(configured)
        self.marker = marker
        self.entered = asyncio.Event()
        self.release = asyncio.Event()

    async def embed_documents(self, texts: Sequence[str]) -> EmbeddingResult:
        if any(self.marker in text for text in texts):
            self.entered.set()
            await self.release.wait()
        return await super().embed_documents(texts)


class SlowCheckpointJournal(Journal):
    """A journal whose checkpoint write can be held open inside its worker thread."""

    def __init__(self, root: Path) -> None:
        super().__init__(root, checkpoint_every=1)
        self.writing = threading.Event()
        self.finish = threading.Event()
        self.hold = True

    def checkpoint_blocking(self, record: JobRecord, document_index: int) -> JobRecord:
        if self.hold:
            self.hold = False
            self.writing.set()
            assert self.finish.wait(WAIT_SECONDS), "the test never released the checkpoint"
        return super().checkpoint_blocking(record, document_index)


def one_at_a_time(**workers: int) -> Settings:
    """Return settings under which every chunk is its own embedding batch."""
    return settings(
        embeddings={"batch_size": 1},
        workers={"embedding_pool_size": 1, "queue_depth": 200, **workers},
    )


def write_corpus(root: Path, words: Sequence[str]) -> list[str]:
    """Write one single-chunk document per word and return the paths in ingestion order."""
    root.mkdir()
    paths: list[str] = []
    for index, word in enumerate(words):
        path = root / f"{index:02d}-{word}.md"
        path.write_text(f"# {word}\n\nThe {word} policy allows {index} days.\n", encoding="utf-8")
        paths.append(str(path))
    return paths


@pytest.fixture
def corpus(tmp_path: Path) -> list[str]:
    """Return four documents, enough to interrupt a job part of the way through."""
    return write_corpus(tmp_path / "corpus", ("alpha", "bravo", "charlie", "delta"))


def service_for(
    tmp_path: Path,
    configured: Settings,
    embedder: StubEmbedder,
    journal: Journal | None = None,
) -> tuple[IngestionService, KeywordLegAdapter]:
    adapter = KeywordLegAdapter(configured)
    service = IngestionService(
        configured,
        journal=journal or Journal(tmp_path / "journal", checkpoint_every=1),
        adapter=adapter,
        router=TieringRouter(embedder),
        executor_factory=threads,
    )
    return service, adapter


async def cancelled(task: "asyncio.Task[JobRecord]") -> None:
    """Cancel a running job and wait for it to finish unwinding."""
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task


async def test_a_cancelled_job_is_settled_failed_and_interrupted(
    tmp_path: Path, corpus: list[str]
) -> None:
    """Cancellation settled nothing, so the record stayed ``running`` for good."""
    configured = one_at_a_time()
    embedder = HeldEmbedder(configured, marker="alpha")
    service, _ = service_for(tmp_path, configured, embedder)
    record = await service.accept(corpus)

    task = asyncio.ensure_future(service.run(record))
    await asyncio.wait_for(embedder.entered.wait(), WAIT_SECONDS)
    await cancelled(task)

    settled = await service.journal.load_job(record.job_id)
    assert settled.status == "failed"
    assert settled.reason_code == INTERRUPTED
    assert settled.finished_at is not None


async def test_an_interrupted_job_keeps_the_checkpoint_it_reached(
    tmp_path: Path, corpus: list[str]
) -> None:
    """Settling from the accepted record would have thrown the progress away."""
    configured = one_at_a_time()
    embedder = HeldEmbedder(configured, marker="charlie")
    service, _ = service_for(tmp_path, configured, embedder)
    record = await service.accept(corpus)

    task = asyncio.ensure_future(service.run(record))
    await asyncio.wait_for(embedder.entered.wait(), WAIT_SECONDS)
    await cancelled(task)

    settled = await service.journal.load_job(record.job_id)
    assert settled.checkpoint is not None
    assert settled.checkpoint.last_document_index == 1
    assert service.journal.resume_index(settled) == 2


async def test_nothing_keeps_working_on_a_cancelled_job(tmp_path: Path) -> None:
    """Both halves of the pipeline outlived the job and wrote ``running`` over its record.

    The corpus is larger than the queue, so the parse side is still blocked on a full queue
    when the job is cancelled — the case in which neither half was being awaited.
    """
    configured = one_at_a_time(queue_depth=QUEUE_DEPTH)
    embedder = HeldEmbedder(configured, marker="alpha")
    service, adapter = service_for(tmp_path, configured, embedder)
    words = ["alpha", *(f"filler{index}" for index in range(QUEUE_DEPTH * 2))]
    record = await service.accept(write_corpus(tmp_path / "large", words))

    task = asyncio.ensure_future(service.run(record))
    await asyncio.wait_for(embedder.entered.wait(), WAIT_SECONDS)
    await asyncio.sleep(0.05)
    await cancelled(task)
    survivors = [other for other in asyncio.all_tasks() if other is not asyncio.current_task()]
    embedder.release.set()
    await asyncio.sleep(0.05)

    settled = await service.journal.load_job(record.job_id)
    assert survivors == []
    assert adapter.points == []
    assert settled.status == "failed"
    assert settled.reason_code == INTERRUPTED


async def test_a_checkpoint_in_flight_lands_before_the_job_is_settled(
    tmp_path: Path, corpus: list[str]
) -> None:
    """A write still in its thread finished after the settle and put ``running`` back."""
    configured = one_at_a_time()
    journal = SlowCheckpointJournal(tmp_path / "journal")
    service, _ = service_for(tmp_path, configured, StubEmbedder(configured), journal)
    record = await service.accept(corpus)

    task = asyncio.ensure_future(service.run(record))
    assert await asyncio.to_thread(journal.writing.wait, WAIT_SECONDS)
    asyncio.get_running_loop().call_later(0.1, journal.finish.set)
    await cancelled(task)
    await asyncio.sleep(0.05)

    settled = await journal.load_job(record.job_id)
    assert settled.status == "failed"
    assert settled.reason_code == INTERRUPTED
    assert settled.checkpoint is not None


async def test_an_interrupted_job_runs_to_completion_when_it_is_run_again(
    tmp_path: Path, corpus: list[str]
) -> None:
    """The checkpoint was written and nothing ever read it after an interruption."""
    configured = one_at_a_time()
    embedder = HeldEmbedder(configured, marker="charlie")
    service, adapter = service_for(tmp_path, configured, embedder)
    record = await service.accept(corpus)
    task = asyncio.ensure_future(service.run(record))
    await asyncio.wait_for(embedder.entered.wait(), WAIT_SECONDS)
    await cancelled(task)
    embedder.release.set()

    resumed = await service.run(await service.journal.load_job(record.job_id))

    assert resumed.status == "completed"
    assert resumed.reason_code is None
    assert resumed.counts["parsed"] == 2
    assert len({point.payload["document_id"] for point in adapter.points}) == len(corpus)


async def test_settling_an_interrupted_job_leaves_a_finished_one_alone(tmp_path: Path) -> None:
    """A cancel arriving after the job completed must not relabel a success as a failure."""
    journal = Journal(tmp_path / "journal")
    record = await journal.create_job("default", [{"type": "path", "value": "corpus/"}])
    await journal.save_job(JobRecord(record.job_id, "default", status="completed"))

    await settle_interrupted(journal, record)

    assert (await journal.load_job(record.job_id)).status == "completed"


async def test_a_record_written_before_the_reason_existed_still_loads(tmp_path: Path) -> None:
    """Every job file on disk predates the field; none of them may become unreadable."""
    payload = JobRecord("job_old", "default", status="completed").as_dict()
    del payload["reason_code"]

    assert JobRecord.from_dict(payload).reason_code is None
