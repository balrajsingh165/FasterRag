"""Indexer: writes embedded chunks to the vector database.

The end of the ingestion chain, and the ``ChunkSink`` the embedding pool writes to. Each
batch becomes one upsert carrying three things per chunk: the dense vector, the BM25 sparse
vector for the keyword leg, and the metadata payload that citations and filters are built
from.

Writes are idempotent because chunk ids are deterministic (``docs/data-model.md``). Replaying
a batch after a crash overwrites the same points rather than adding duplicates, which is what
turns an at-least-once pipeline into exactly-once index effects (D3).

The payload mirrors the ``Chunk`` entity rather than inventing fields. Anything a citation,
a filter, or drift detection needs later has to be written now — a chunk that reaches the
index without its span or its embedding model cannot be cited or checked for drift
afterwards.

Deterministic ids make a re-ingested document overwrite itself, but only as far as its new
length: a version with fewer chunks leaves the old tail in place. :meth:`Indexer.trim`
removes it, by the ``document_id`` and ``chunk_index`` this module writes into every payload
(TASK-0443). It is called from the document's commit callback, so it runs after the new
chunks are in — trimming first would leave a window in which the document has no chunks —
and only for a document the journal already knew, because a filtered delete per document on
a first bulk load would be paid for tails that cannot exist.
"""

from __future__ import annotations

import asyncio
from typing import Any

from fasterrag.adapters.vectordb.base import (
    CollectionSpec,
    Point,
    PointSelector,
    VectorDBAdapter,
)
from fasterrag.config.schema import Settings
from fasterrag.core.retrieval.bm25 import encode_document
from fasterrag.errors import ErrorCode, FasterRagError
from fasterrag.observability.logging import get_logger
from fasterrag.services.tenancy import TENANT_PAYLOAD_KEY
from fasterrag.workers.embed_pool import backoff_delay
from fasterrag.workers.queues import ChunkPayload, EmbeddedBatch

__all__ = ["Indexer", "chunk_payload"]

_logger = get_logger(__name__)


def chunk_payload(payload: ChunkPayload, *, model: str, model_version: str) -> dict[str, Any]:
    """Build the stored metadata for one chunk."""
    chunk = payload.chunk
    stored: dict[str, Any] = {
        "document_id": payload.document_id,
        "source_uri": payload.source,
        "content_hash": payload.content_hash,
        "text": chunk.text,
        "span": {"start": chunk.start, "end": chunk.end},
        "chunk_index": chunk.chunk_index,
        "token_count": chunk.token_count,
        "chunker_strategy": chunk.strategy,
        "embedding_model": model,
        "embedding_model_version": model_version,
    }

    if chunk.page is not None:
        stored["page"] = chunk.page
    if chunk.section is not None:
        stored["section"] = chunk.section
    stored.update(dict(payload.metadata))

    # CRITICAL: written *after* the caller's metadata, not before. The tenant is the field
    # `security.tenant_scoped_retrieval` filters on, so a job carrying
    # `metadata={"tenant": ...}` could otherwise relabel its own chunks into another
    # tenant's filtered view. The credential the job was created with wins (TASK-0339).
    if payload.tenant is not None:
        stored[TENANT_PAYLOAD_KEY] = payload.tenant

    return stored


class Indexer:
    """Writes embedded batches into a collection."""

    def __init__(
        self,
        settings: Settings,
        adapter: VectorDBAdapter,
        *,
        collection: str | None = None,
    ) -> None:
        """Build the indexer."""
        self.settings = settings
        self.adapter = adapter
        self.collection = collection or settings.vector_db.collection.default_name
        self.hybrid = settings.retrieval.hybrid
        self.written = 0
        self.trimmed = 0
        self._ready = False
        self._creating = asyncio.Lock()

    async def ensure_collection(self, dimensions: int) -> None:
        """Create the target collection if it does not exist yet."""
        await self.adapter.create_collection(
            CollectionSpec(
                name=self.collection,
                dimensions=dimensions,
                distance=self.settings.vector_db.collection.distance,
                shard_number=self.settings.vector_db.collection.shard_number,
                replication_factor=self.settings.vector_db.collection.replication_factor,
                sparse=self.hybrid,
            )
        )
        self._ready = True
        _logger.info(
            "collection ready",
            extra={
                "collection": self.collection,
                "dimensions": dimensions,
                "sparse": self.hybrid,
            },
        )

    async def write(self, batch: EmbeddedBatch) -> None:
        """Write one embedded batch, dense and sparse together."""
        if not batch.chunks:
            return

        # CRITICAL: several embedding workers reach their first batch at once, so the
        # lock is what stops them racing to create the same collection. Without it the
        # losers of the race fail with a conflict on a collection that is perfectly fine.
        if not self._ready:
            async with self._creating:
                if not self._ready:
                    await self.ensure_collection(len(batch.vectors[0]))

        points = [
            Point(
                point_id=payload.chunk_id,
                collection=self.collection,
                vector=vector,
                payload=chunk_payload(
                    payload, model=batch.model, model_version=batch.model_version
                ),
                sparse=(
                    encode_document(
                        payload.text,
                        k1=self.settings.retrieval.bm25_k1,
                        b=self.settings.retrieval.bm25_b,
                    )
                    if self.hybrid
                    else None
                ),
            )
            for payload, vector in zip(batch.chunks, batch.vectors, strict=True)
        ]

        result = await self.adapter.upsert(points)
        self.written += result.upserted
        _logger.info(
            "indexed a batch",
            extra={
                "collection": self.collection,
                "points": result.upserted,
                "model": batch.model,
            },
        )

    async def trim(self, document_id: str, keep: int) -> None:
        """Delete a document's chunks at index ``keep`` and beyond: a longer version's tail.

        Retries what is retryable and then raises, so an update is never recorded as
        indexed while chunks of the version it replaced are still retrievable (TASK-0443).
        """
        selector = PointSelector(
            collection=self.collection,
            filters={"document_id": document_id, "chunk_index": {"$gte": keep}},
        )
        attempts = self.settings.reliability.retries.max_attempts
        for attempt in range(attempts + 1):
            try:
                await self.adapter.delete(selector)
            except FasterRagError as exc:
                if exc.code is ErrorCode.NOT_FOUND:
                    _logger.info(
                        "no collection to trim; a collection that does not exist holds no tail",
                        extra={"collection": self.collection, "trace_id": exc.trace_id},
                    )
                    return
                if not exc.retryable or attempt == attempts:
                    raise
                delay = backoff_delay(attempt + 1, self.settings)
                _logger.warning(
                    "retrying a document trim",
                    extra={
                        "document_id": document_id,
                        "attempt": attempt + 1,
                        "delay_seconds": round(delay, 3),
                        "code": exc.code.value,
                        "trace_id": exc.trace_id,
                    },
                )
                await asyncio.sleep(delay)
                continue
            break

        self.trimmed += 1
        _logger.info(
            "trimmed a re-ingested document",
            extra={"collection": self.collection, "document_id": document_id, "keep": keep},
        )
