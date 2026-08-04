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
"""

from __future__ import annotations

import asyncio
from typing import Any

from fasterrag.adapters.vectordb.base import (
    CollectionSpec,
    Point,
    VectorDBAdapter,
)
from fasterrag.config.schema import Settings
from fasterrag.core.retrieval.bm25 import encode_document
from fasterrag.observability.logging import get_logger
from fasterrag.workers.queues import ChunkPayload, EmbeddedBatch

__all__ = ["Indexer", "chunk_payload"]

_logger = get_logger(__name__)


def chunk_payload(payload: ChunkPayload, *, model: str, model_version: str) -> dict[str, Any]:
    """Build the stored metadata for one chunk.

    Args:
        payload: The chunk and its document context.
        model: Embedding model that produced the vector.
        model_version: Its version, the anchor drift detection compares against.

    Returns:
        The payload written alongside the vector.
    """
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
    if payload.tenant is not None:
        stored["tenant"] = payload.tenant

    stored.update(dict(payload.metadata))
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
        """Build the indexer.

        Args:
            settings: Validated configuration. ``retrieval.hybrid`` decides whether a
                sparse vector is produced and stored at all.
            adapter: The vector database to write to.
            collection: Target collection; defaults to the configured one.
        """
        self.settings = settings
        self.adapter = adapter
        self.collection = collection or settings.vector_db.collection.default_name
        self.hybrid = settings.retrieval.hybrid
        self.written = 0
        self._ready = False
        self._creating = asyncio.Lock()

    async def ensure_collection(self, dimensions: int) -> None:
        """Create the target collection if it does not exist yet.

        The sparse index is created with the collection or not at all: Qdrant cannot add a
        sparse vector to a dense-only collection afterwards, so a corpus indexed without
        hybrid enabled must be rebuilt to gain a keyword leg.
        """
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
        """Write one embedded batch, dense and sparse together.

        The collection is created on the first batch, sized from the vectors themselves.
        Creating it earlier would mean either asking the provider for a dimension it may
        only know after loading a model, or spending an embedding call on a probe.
        """
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
