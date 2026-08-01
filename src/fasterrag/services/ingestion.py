"""Ingestion: accept a job, run the pipeline, own its lifecycle.

The service that turns the two pools into an ingest. It creates the job record, runs the
CPU pool and the embedding pool **concurrently against one bounded queue**, checkpoints as
documents complete, and settles the job as completed, partial, or failed.

Concurrency is the point. Parsing and embedding overlap rather than running in phases, so
the embedding workers start on the first document's chunks while the parser is still on the
second — which is what the bounded queue between them exists to make safe
(``docs/architecture.md`` §2).

The API accepts ingestion asynchronously and returns a job id immediately
(``docs/api-reference.md``). This service is what a background runner drives; nothing here
blocks an HTTP request.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Sequence
from concurrent.futures import Executor
from dataclasses import replace
from datetime import UTC, datetime
from typing import Any

from fasterrag.adapters.embeddings.tiering import TieringRouter, create_embedding_router
from fasterrag.adapters.vectordb.base import VectorDBAdapter
from fasterrag.adapters.vectordb.factory import create_vector_db_adapter
from fasterrag.config.schema import Settings
from fasterrag.core.cache.semantic import SemanticCache
from fasterrag.errors import ErrorCode
from fasterrag.observability.logging import get_logger, use_trace_id
from fasterrag.services.journal import JobRecord, Journal
from fasterrag.services.lockfile import LockStore, build_lock
from fasterrag.workers.cpu_pool import CpuWorkerPool
from fasterrag.workers.embed_pool import EmbeddingWorkerPool
from fasterrag.workers.indexer import Indexer
from fasterrag.workers.queues import BoundedQueue, ChunkPayload

__all__ = ["IngestionService"]

_logger = get_logger(__name__)


class IngestionService:
    """Owns an ingestion job from acceptance to settled status."""

    def __init__(
        self,
        settings: Settings,
        *,
        journal: Journal,
        adapter: VectorDBAdapter | None = None,
        router: TieringRouter | None = None,
        cache: SemanticCache | None = None,
        locks: LockStore | None = None,
        executor_factory: Callable[[int], Executor] | None = None,
    ) -> None:
        """Build the service.

        Args:
            settings: Validated configuration.
            journal: Durable job, dedup, and dead-letter state.
            adapter: Vector database to index into; built from configuration when omitted.
            router: Embedding router; built from configuration when omitted.
            locks: Records what produced the index, so drift is detectable (D1).
            cache: Semantic response cache, invalidated when a job changes the corpus. A
                cached answer describes the corpus as it was, and this is the event that
                makes that description potentially wrong.
            executor_factory: Passed to the CPU pool, so tests can avoid process spawning.
        """
        self.settings = settings
        self.journal = journal
        self.adapter = adapter or create_vector_db_adapter(settings)
        self.router = router or create_embedding_router(settings)
        self.cache = cache
        self.locks = locks
        self._executor_factory = executor_factory

    async def accept(
        self,
        sources: Sequence[str],
        *,
        collection: str | None = None,
        tenant: str | None = None,
        idempotency_key: str | None = None,
    ) -> JobRecord:
        """Record a job without doing any work.

        Returns immediately so the API can answer ``202 Accepted`` with a job id. Replaying
        an idempotency key returns the original job rather than starting a second ingest.
        """
        return self.journal.create_job(
            collection or self.settings.vector_db.collection.default_name,
            [{"type": "path", "value": source} for source in sources],
            idempotency_key=idempotency_key,
            tenant=tenant,
        )

    async def run(
        self,
        record: JobRecord,
        *,
        metadata: dict[str, Any] | None = None,
    ) -> JobRecord:
        """Run an accepted job to completion and return its settled record.

        Resumes from the job's checkpoint, so re-running after a crash continues rather
        than restarting. Deduplication makes any documents that were replayed anyway into
        no-ops (D3).
        """
        sources = [source["value"] for source in record.sources]
        tasks = CpuWorkerPool.tasks_for(sources, tenant=record.tenant, metadata=metadata)
        resume_from = self.journal.resume_index(record)

        indexer = Indexer(self.settings, self.adapter, collection=record.collection)
        pool = EmbeddingWorkerPool(self.settings, self.router, indexer)
        queue: BoundedQueue[ChunkPayload] = BoundedQueue(self.settings.workers.queue_depth)

        running = replace(record, status="running")
        self.journal.save_job(running)
        progress = _Checkpointer(self.journal, running)

        with use_trace_id():
            _logger.info(
                "ingest started",
                extra={
                    "job_id": record.job_id,
                    "collection": record.collection,
                    "documents": len(tasks),
                    "resume_from": resume_from,
                },
            )

            async with CpuWorkerPool(
                self.settings,
                journal=self.journal,
                executor_factory=self._executor_factory,
            ) as cpu:
                consumer = asyncio.create_task(pool.run(queue))
                try:
                    parse_report = await cpu.process(
                        tasks,
                        queue,
                        job=record.job_id,
                        collection=record.collection,
                        resume_from=resume_from,
                        on_progress=progress,
                    )
                finally:
                    await queue.close(consumers=pool.size)
                embed_report = await consumer

        for document, code in embed_report.failures:
            self.journal.dead_letter(
                record.job_id,
                document=document,
                source=next(
                    (task.source for task in tasks if task.document_id == document), document
                ),
                reason_code=ErrorCode(code),
                detail="the embedding provider could not be reached after its retries",
                attempts=self.settings.reliability.retries.max_attempts,
            )

        counts = {
            "total": len(tasks),
            "parsed": parse_report.parsed,
            "chunked": parse_report.chunked,
            "embedded": embed_report.embedded,
            "indexed": indexer.written,
            "deduplicated": parse_report.deduplicated,
            "dead_lettered": parse_report.dead_lettered + len(embed_report.failures),
        }
        settled = replace(
            progress.record,
            status=_status_for(counts),
            counts=counts,
            finished_at=_finished_at(),
        )
        self.journal.save_job(settled)

        if indexer.written:
            self._write_lockfile(record.collection)

        if self.cache is not None and indexer.written:
            await self.cache.invalidate(f"ingest job {record.job_id} indexed {indexer.written}")

        _logger.info("ingest finished", extra={"job_id": record.job_id, **counts})
        return settled

    def _write_lockfile(self, collection: str) -> None:
        """Record what produced this index, so drift against it is detectable (D1).

        Written after the job settles rather than before it runs: a lockfile describing an
        index that failed halfway would claim a corpus the collection does not hold.
        """
        if self.locks is None or not self.locks.enabled:
            return

        embedder = self.router.default
        self.locks.write(
            build_lock(
                collection,
                self.settings,
                embedding_model=embedder.model,
                embedding_model_version=embedder.model_version,
                dimensions=embedder.dimensions,
                document_hashes=self.journal.document_hashes(collection),
            )
        )

    async def ingest(
        self,
        sources: Sequence[str],
        *,
        collection: str | None = None,
        metadata: dict[str, Any] | None = None,
        tenant: str | None = None,
        idempotency_key: str | None = None,
    ) -> JobRecord:
        """Accept a job and run it, the path the CLI and the library take."""
        record = await self.accept(
            sources,
            collection=collection,
            tenant=tenant,
            idempotency_key=idempotency_key,
        )
        return await self.run(record, metadata=metadata)

    async def close(self) -> None:
        """Release the adapters this service owns."""
        await self.router.close()
        await self.adapter.close()


class _Checkpointer:
    """Records job progress at the configured interval as documents complete."""

    def __init__(self, journal: Journal, record: JobRecord) -> None:
        """Start from an accepted job."""
        self.journal = journal
        self.record = record

    def __call__(self, index: int) -> None:
        """Checkpoint after the document at ``index``."""
        self.record = self.journal.checkpoint(self.record, index)


def _status_for(counts: dict[str, int]) -> str:
    """Return the terminal status implied by a job's counts.

    ``partial`` exists so a job that indexed most of a corpus is not reported as a plain
    failure: the difference decides whether an operator retries the dead letters or the
    whole run.
    """
    failed = counts["dead_lettered"]
    succeeded = counts["indexed"] + counts["deduplicated"]

    if not failed:
        return "completed"
    if succeeded:
        return "partial"
    return "failed"


def _finished_at() -> str:
    """Return the completion timestamp."""
    return datetime.now(tz=UTC).isoformat()
