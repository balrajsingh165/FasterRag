"""Per-document completion tracking between the parse side and the index side.

A document's chunks leave the CPU pool long before they are embedded and written: they sit
in a bounded queue, are batched with other documents' chunks, and reach the vector database
from whichever embedding worker picked them up. The journal therefore cannot learn that a
document is indexed from the parse side, and for the first fourteen slices it did exactly
that — ``record_indexed`` and the dedup hash were written the moment the chunks were
*enqueued*. Three defects followed from that one ordering:

* a document whose embedding later failed was dead-lettered **and** remembered as indexed,
  so ``retry-dlq`` found its content hash, reported it ``deduplicated`` and never indexed it;
* a crash between enqueue and upsert left the journal claiming documents the collection did
  not hold, permanently, because the dedup file is append-only;
* the checkpoint advanced at parse time, so a resumed job would have skipped documents
  whose chunks were still queued.

The ledger closes all three by making "indexed" a statement about the vector database. The
parse side registers each document with the number of chunks it produced and a commit
callback; the embedding pool reports every chunk it wrote or gave up on; the callback runs
only when every chunk of the document has been written. A document with any failed chunk is
never committed — it is reported once, as one failed *document*, for the dead-letter queue.

The checkpoint is a contiguous watermark over document indexes rather than "the last index
seen", because documents now settle out of order. It advances only across documents that
are settled — committed, deduplicated, or dead-lettered — so resuming from it can never
skip one that is still in flight.

Every journal write of a job goes through one lock, :attr:`IndexLedger.journal_lock`:
commits, checkpoints, and the parse side's own dead-letter and dedup records. The journal
writes plain files from worker threads, and before this module existed those writes were
sequential by construction — they all came from the parse loop. They no longer are. A
checkpoint can now be triggered from the parse side (a dead-lettered document settles) and
from the embedding side (a document finishes) at the same moment, and the journal rewrites
a job record through one fixed temporary file name, so two such writes collide: the first
CI run of this module against a real backend failed on exactly that, with
``FileNotFoundError`` on the rename. Serialising here costs nothing measurable — each write
is already a thread hop — and makes the ordering a property instead of a coincidence.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass

from fasterrag.workers.queues import ChunkPayload

__all__ = ["FailedDocument", "IndexLedger"]

Commit = Callable[[], Awaitable[None]]
Progress = Callable[[int], Awaitable[None]]


@dataclass(frozen=True, slots=True)
class FailedDocument:
    """A document that had at least one chunk the pipeline could not index."""

    document_id: str
    source: str
    code: str


@dataclass(slots=True)
class _Pending:
    """One registered document that still has chunks in flight."""

    index: int
    source: str
    remaining: int
    commit: Commit
    failure: str | None = None


class IndexLedger:
    """Decides when a document is indexed, from what the index side actually wrote."""

    def __init__(self, on_progress: Progress | None = None, *, start: int = 0) -> None:
        """Build a ledger whose watermark begins at the document index ``start``."""
        self._on_progress = on_progress
        self._next = start
        self._settled: set[int] = set()
        self._pending: dict[str, _Pending] = {}
        self._failed: dict[str, FailedDocument] = {}
        self.journal_lock = asyncio.Lock()
        self.committed = 0

    @property
    def failed_documents(self) -> list[FailedDocument]:
        """Return every document with a failed chunk, one entry per document."""
        return list(self._failed.values())

    @property
    def in_flight(self) -> int:
        """Return how many registered documents still have chunks unaccounted for."""
        return len(self._pending)

    async def expect(
        self, document_id: str, *, index: int, source: str, chunks: int, commit: Commit
    ) -> None:
        """Register a document before its chunks are enqueued.

        A document that produced no chunks has nothing to wait for and is committed at once.
        """
        self._pending[document_id] = _Pending(
            index=index, source=source, remaining=chunks, commit=commit
        )
        if chunks <= 0:
            await self._finish(document_id)

    async def settle(self, index: int) -> None:
        """Mark a document that never entered the queue as settled.

        The parse side calls this for a dead-lettered or deduplicated document, so the
        watermark can pass it.
        """
        await self._advance(index)

    async def written(self, payloads: Sequence[ChunkPayload]) -> None:
        """Account for chunks the sink has durably written."""
        await self._account(payloads, failure=None)

    async def failed(self, payloads: Sequence[ChunkPayload], code: str) -> None:
        """Account for chunks the pipeline gave up on, with the reason code."""
        await self._account(payloads, failure=code)

    async def _account(self, payloads: Sequence[ChunkPayload], *, failure: str | None) -> None:
        """Decrement each document's outstanding chunks and finish the ones that reach zero."""
        finished: list[str] = []
        for payload in payloads:
            pending = self._pending.get(payload.document_id)
            if pending is None:
                continue
            pending.remaining -= 1
            if failure is not None and pending.failure is None:
                pending.failure = failure
            if pending.remaining <= 0 and payload.document_id not in finished:
                finished.append(payload.document_id)

        for document_id in finished:
            await self._finish(document_id)

    async def _finish(self, document_id: str) -> None:
        """Commit a fully written document, or record it as failed, then settle its index."""
        pending = self._pending.pop(document_id, None)
        if pending is None:
            return

        if pending.failure is not None:
            self._failed[document_id] = FailedDocument(
                document_id=document_id, source=pending.source, code=pending.failure
            )
        else:
            async with self.journal_lock:
                await pending.commit()
            self.committed += 1

        await self._advance(pending.index)

    async def _advance(self, index: int) -> None:
        """Move the contiguous watermark past every settled index, reporting each one."""
        self._settled.add(index)
        async with self.journal_lock:
            await self._report_settled()

    async def _report_settled(self) -> None:
        """Advance the watermark and report it; the caller holds the journal lock."""
        while self._next in self._settled:
            self._settled.discard(self._next)
            reached = self._next
            self._next += 1
            if self._on_progress is not None:
                await self._on_progress(reached)
