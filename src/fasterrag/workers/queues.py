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
from fasterrag.observability import metrics

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

    ``source`` is the canonical URI — what the document *is*, and what its id derives from.
    ``location`` is where the bytes currently sit, which differs only for a staged URL or
    inline payload.
    """

    document_id: str
    source: str
    index: int
    metadata: Mapping[str, Any] = field(default_factory=dict)
    tenant: str | None = None
    location: str | None = None

    @property
    def readable(self) -> str:
        """Return the path the bytes are actually read from.

        # CRITICAL: identity and location are separate fields, and conflating them breaks
        # deduplication silently. A URL or inline document is staged to a temp file whose
        # name differs on every run, so a document id derived from that path would be new
        # every time and the same document would be re-indexed forever. ``source`` stays the
        # canonical URI the id is hashed from; this is only where the bytes live.
        """
        return self.location or self.source


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
    document_text: str = ""
    """The whole document, carried only for late chunking.

    Every chunk of one document references the *same* string object, attached on the
    parent side after the worker returns, so this costs a pointer per chunk rather than a
    copy. Late chunking needs it because a chunk's vector is pooled out of a pass over the
    document, not computed from the chunk alone.
    """

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
    document_text: str = ""
    """The parsed document, carried back for contextual enrichment (P2).

    # CRITICAL: enrichment cannot run inside the worker. Parsing happens in a *process*
    # pool, and an LLM adapter holds a client that cannot be pickled across that boundary —
    # so the text comes back and the provider call happens on the async side. Carrying it
    # costs roughly one extra copy of the document through IPC, which is the price of the
    # feature working at all.
    """


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

    def __init__(self, capacity: int, *, name: str = "chunks") -> None:
        """Build a queue holding at most ``capacity`` items.

        Args:
            capacity: The largest number of items the queue will hold before producers
                block or are rejected.
            name: The ``queue`` label its depth is published under. Named rather than
                anonymous because a single number for "the queue depth" cannot tell an
                operator which stage is the one falling behind.
        """
        self.capacity = capacity
        self.name = name
        self._queue: asyncio.Queue[Item | None] = asyncio.Queue(maxsize=capacity)
        self._publish()

    @property
    def depth(self) -> int:
        """Return the current occupancy, the source of the queue-depth metric."""
        return self._queue.qsize()

    def _publish(self) -> None:
        """Publish the current depth.

        Called after every mutation rather than sampled on a timer: the depth this gauge
        exists to show is a transient backlog, and a sampler is most likely to miss exactly
        the spike an operator went looking for.
        """
        metrics.QUEUE_DEPTH.set(float(self._queue.qsize()), queue=self.name)

    @property
    def full(self) -> bool:
        """Return whether the queue is at capacity."""
        return self._queue.full()

    async def put(self, item: Item) -> None:
        """Enqueue an item, waiting while the queue is full."""
        await self._queue.put(item)
        self._publish()

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
        self._publish()

    async def get(self) -> Item | None:
        """Dequeue an item, or ``None`` once the queue has been closed."""
        item = await self._queue.get()
        self._publish()
        return item

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
        self._publish()

    async def drain(self) -> Sequence[Item]:
        """Remove and return everything currently queued, ignoring sentinels."""
        items: list[Item] = []
        while not self._queue.empty():
            item = self._queue.get_nowait()
            self._queue.task_done()
            if item is not None:
                items.append(item)
        self._publish()
        return items
