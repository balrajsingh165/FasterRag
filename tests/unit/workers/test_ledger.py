"""A document is indexed when the index side says so, not when its chunks are enqueued.

These pin the three properties ``workers/ledger.py`` exists for: a commit waits for every
chunk, a failed chunk prevents the commit and is reported once per document, and the
checkpoint watermark never passes a document that is still in flight.
"""

import asyncio
from collections.abc import Awaitable, Callable

from fasterrag.core.chunking.models import TextChunk
from fasterrag.workers.ledger import IndexLedger
from fasterrag.workers.queues import ChunkPayload


def chunk(document: str, index: int) -> ChunkPayload:
    return ChunkPayload(
        chunk_id=f"{document}_{index}",
        document_id=document,
        source=f"{document}.md",
        content_hash="hash",
        chunk=TextChunk(
            text="text", start=0, end=4, chunk_index=index, token_count=1, strategy="recursive"
        ),
    )


class Recorder:
    """Records commits and progress in the order the ledger issued them."""

    def __init__(self) -> None:
        self.commits: list[str] = []
        self.progress: list[int] = []

    def commit(self, document: str) -> Callable[[], Awaitable[None]]:
        async def run() -> None:
            self.commits.append(document)

        return run

    async def on_progress(self, index: int) -> None:
        self.progress.append(index)


async def register(ledger: IndexLedger, recorder: Recorder, document: str, index: int) -> None:
    await ledger.expect(
        document, index=index, source=f"{document}.md", chunks=2, commit=recorder.commit(document)
    )


async def test_a_document_is_not_committed_while_a_chunk_is_outstanding() -> None:
    recorder = Recorder()
    ledger = IndexLedger(recorder.on_progress)
    await register(ledger, recorder, "a", 0)

    await ledger.written([chunk("a", 0)])

    assert recorder.commits == []
    assert ledger.in_flight == 1


async def test_a_document_is_committed_when_its_last_chunk_is_written() -> None:
    recorder = Recorder()
    ledger = IndexLedger(recorder.on_progress)
    await register(ledger, recorder, "a", 0)

    await ledger.written([chunk("a", 0)])
    await ledger.written([chunk("a", 1)])

    assert recorder.commits == ["a"]
    assert ledger.committed == 1
    assert ledger.in_flight == 0


async def test_a_failed_chunk_prevents_the_commit() -> None:
    """The defect this module closes: a failed document remembered as indexed."""
    recorder = Recorder()
    ledger = IndexLedger(recorder.on_progress)
    await register(ledger, recorder, "a", 0)

    await ledger.written([chunk("a", 0)])
    await ledger.failed([chunk("a", 1)], "EMBED_PROVIDER_ERROR")

    assert recorder.commits == []
    assert ledger.committed == 0


async def test_a_failed_document_is_reported_once_however_many_chunks_failed() -> None:
    recorder = Recorder()
    ledger = IndexLedger(recorder.on_progress)
    await register(ledger, recorder, "a", 0)

    await ledger.failed([chunk("a", 0), chunk("a", 1)], "EMBED_PROVIDER_ERROR")

    failed = ledger.failed_documents
    assert [(entry.document_id, entry.source, entry.code) for entry in failed] == [
        ("a", "a.md", "EMBED_PROVIDER_ERROR")
    ]


async def test_a_document_with_no_chunks_is_committed_at_once() -> None:
    recorder = Recorder()
    ledger = IndexLedger(recorder.on_progress)

    await ledger.expect("empty", index=0, source="empty.md", chunks=0, commit=recorder.commit("e"))

    assert recorder.commits == ["e"]
    assert recorder.progress == [0]


async def test_one_batch_can_finish_several_documents() -> None:
    recorder = Recorder()
    ledger = IndexLedger(recorder.on_progress)
    await register(ledger, recorder, "a", 0)
    await register(ledger, recorder, "b", 1)

    await ledger.written([chunk("a", 0), chunk("b", 0), chunk("a", 1), chunk("b", 1)])

    assert recorder.commits == ["a", "b"]


async def test_the_watermark_does_not_pass_a_document_still_in_flight() -> None:
    """A checkpoint past an unwritten document is a document a resumed job would skip."""
    recorder = Recorder()
    ledger = IndexLedger(recorder.on_progress)
    await register(ledger, recorder, "a", 0)
    await register(ledger, recorder, "b", 1)

    await ledger.written([chunk("b", 0), chunk("b", 1)])

    assert recorder.commits == ["b"]
    assert recorder.progress == []


async def test_the_watermark_catches_up_once_the_gap_closes() -> None:
    recorder = Recorder()
    ledger = IndexLedger(recorder.on_progress)
    await register(ledger, recorder, "a", 0)
    await register(ledger, recorder, "b", 1)
    await ledger.written([chunk("b", 0), chunk("b", 1)])

    await ledger.written([chunk("a", 0), chunk("a", 1)])

    assert recorder.progress == [0, 1]


async def test_a_settled_document_lets_the_watermark_pass() -> None:
    """Dead-lettered and deduplicated documents never enter the queue, yet are finished."""
    recorder = Recorder()
    ledger = IndexLedger(recorder.on_progress)

    await ledger.settle(0)
    await ledger.settle(1)

    assert recorder.progress == [0, 1]


async def test_a_failed_document_still_lets_the_watermark_pass() -> None:
    """It is in the dead-letter queue, which is where a retry looks for it."""
    recorder = Recorder()
    ledger = IndexLedger(recorder.on_progress)
    await register(ledger, recorder, "a", 0)

    await ledger.failed([chunk("a", 0), chunk("a", 1)], "EMBED_PROVIDER_ERROR")

    assert recorder.progress == [0]


async def test_the_watermark_starts_where_a_resumed_job_does() -> None:
    recorder = Recorder()
    ledger = IndexLedger(recorder.on_progress, start=5)

    await ledger.settle(5)

    assert recorder.progress == [5]


async def test_chunks_of_an_unregistered_document_are_ignored() -> None:
    ledger = IndexLedger()

    await ledger.written([chunk("stranger", 0)])

    assert ledger.committed == 0
    assert ledger.failed_documents == []


async def test_progress_reports_never_overlap() -> None:
    """Two sides settle documents at once, and a checkpoint rewrites one file.

    The parse side settles a dead-lettered document while the embedding side finishes
    another. Overlapping reports meant two writers on the journal's single temporary file,
    which failed the first real-backend run of the ledger with a missing-file rename.
    """
    running = 0
    peak = 0

    async def slow_progress(index: int) -> None:
        nonlocal running, peak
        running += 1
        peak = max(peak, running)
        await asyncio.sleep(0.01)
        running -= 1

    ledger = IndexLedger(slow_progress)
    recorder = Recorder()
    await register(ledger, recorder, "b", 1)

    await asyncio.gather(
        ledger.settle(0),
        ledger.written([chunk("b", 0), chunk("b", 1)]),
        ledger.settle(2),
        ledger.settle(3),
    )

    assert peak == 1


async def test_commits_and_progress_reports_share_one_lock() -> None:
    """A commit and a checkpoint both write the journal, so neither may run inside the other."""
    active: list[str] = []
    overlaps: list[tuple[str, ...]] = []

    async def enter(name: str) -> None:
        active.append(name)
        if len(active) > 1:
            overlaps.append(tuple(active))
        await asyncio.sleep(0.01)
        active.remove(name)

    async def on_progress(index: int) -> None:
        await enter("progress")

    async def commit() -> None:
        await enter("commit")

    ledger = IndexLedger(on_progress)
    await ledger.expect("a", index=1, source="a.md", chunks=1, commit=commit)

    await asyncio.gather(ledger.settle(0), ledger.written([chunk("a", 0)]), ledger.settle(2))

    assert overlaps == []


async def test_a_caller_holding_the_journal_lock_excludes_the_ledger() -> None:
    """The parse side takes the same lock for its own dead-letter and dedup records."""
    order: list[str] = []

    async def on_progress(index: int) -> None:
        order.append("progress")

    ledger = IndexLedger(on_progress)

    async with ledger.journal_lock:
        settling = asyncio.create_task(ledger.settle(0))
        await asyncio.sleep(0.01)
        order.append("caller finished")

    await settling

    assert order == ["caller finished", "progress"]
