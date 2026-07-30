"""Bounded queues, backpressure, and the payloads that cross the pool boundary.

The queue between the pools is bounded on purpose. If embedding falls behind, CPU workers
block on enqueue instead of filling memory with chunks nobody is consuming, and a
throttling provider is absorbed by the queue rather than hammered
(``docs/architecture.md`` §2).

Backpressure has two faces, and they are deliberately different calls:

* Inside the pipeline, producers **wait**: falling behind should slow ingestion down, not
  fail it.
* At the API boundary, producers are **rejected** with ``QUEUE_FULL`` and a
  ``Retry-After``: accepting unbounded work would trade a fast failure for an
  out-of-memory one (``docs/reliability.md`` §2).
"""

from __future__ import annotations

import asyncio
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from fasterrag.core.chunking.models import TextChunk
from fasterrag.errors import ErrorCode, IngestionError

__all__ = [
    "BoundedQueue",
    "ChunkPayload",
    "DocumentTask",
    "EmbeddedBatch",
    "ParseOutcome",
]


@dataclass(frozen=True, slots=True)
class DocumentTask:
    """One source to load, parse, and chunk.

    ``index`` is the document's position in its job, which is what a checkpoint records
    and what a resumed job counts from.
    """

    document_id: str
    source: str
    index: int
    metadata: Mapping[str, Any] = field(default_factory=dict)
    tenant: str | None = None


@dataclass(frozen=True, slots=True)
class ChunkPayload:
    """A chunk plus the document context the indexer needs to write it."""

    chunk_id: str
    document_id: str
    source: str
    content_hash: str
    chunk: TextChunk
    metadata: Mapping[str, Any] = field(default_factory=dict)
    tenant: str | None = None

    @property
    def text(self) -> str:
        """Return the text that will be embedded."""
        return self.chunk.text


@dataclass(frozen=True, slots=True)
class ParseOutcome:
    """What a CPU worker produced for one document."""

    task: DocumentTask
    chunks: list[ChunkPayload]
    content_hash: str
    parser: str
    mime_type: str
    parse_flags: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class EmbeddedBatch:
    """Chunks with their vectors, ready to index."""

    chunks: list[ChunkPayload]
    vectors: list[list[float]]
    model: str
    model_version: str

    def __post_init__(self) -> None:
        """Reject a batch whose vectors do not line up with its chunks.

        A silent misalignment here would attach each chunk to the wrong vector — the kind
        of corruption that produces plausible but wrong answers forever.
        """
        if len(self.chunks) != len(self.vectors):
            raise IngestionError(
                f"an embedded batch has {len(self.chunks)} chunks but {len(self.vectors)} vectors",
                code=ErrorCode.CHUNK_FAILED,
                retryable=False,
            )


class BoundedQueue[Item]:
    """An asyncio queue with an explicit capacity and a reject-instead-of-grow path."""

    def __init__(self, capacity: int) -> None:
        """Build a queue holding at most ``capacity`` items."""
        self.capacity = capacity
        self._queue: asyncio.Queue[Item | None] = asyncio.Queue(maxsize=capacity)

    @property
    def depth(self) -> int:
        """Return the current occupancy, the source of the queue-depth metric."""
        return self._queue.qsize()

    @property
    def full(self) -> bool:
        """Return whether the queue is at capacity."""
        return self._queue.full()

    async def put(self, item: Item) -> None:
        """Enqueue an item, waiting while the queue is full."""
        await self._queue.put(item)

    def offer(self, item: Item) -> None:
        """Enqueue without waiting, rejecting the item if the queue is full.

        Raises:
            IngestionError: With ``QUEUE_FULL`` when at capacity, which the API renders as
                ``429`` with a ``Retry-After`` header.
        """
        try:
            self._queue.put_nowait(item)
        except asyncio.QueueFull as exc:
            raise IngestionError(
                f"the queue is at its capacity of {self.capacity} items",
                code=ErrorCode.QUEUE_FULL,
            ) from exc

    async def get(self) -> Item | None:
        """Dequeue an item, or ``None`` once the queue has been closed."""
        return await self._queue.get()

    def task_done(self) -> None:
        """Mark the most recently dequeued item as handled."""
        self._queue.task_done()

    async def join(self) -> None:
        """Wait until every enqueued item has been handled."""
        await self._queue.join()

    async def close(self, consumers: int) -> None:
        """Signal end-of-stream to each consumer.

        One sentinel per consumer, so every worker stops exactly once and none is left
        waiting on a queue that will never fill again.
        """
        for _ in range(consumers):
            await self._queue.put(None)

    async def drain(self) -> Sequence[Item]:
        """Remove and return everything currently queued, ignoring sentinels."""
        items: list[Item] = []
        while not self._queue.empty():
            item = self._queue.get_nowait()
            self._queue.task_done()
            if item is not None:
                items.append(item)
        return items
