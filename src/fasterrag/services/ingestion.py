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
import time
from collections.abc import Callable, Mapping, Sequence
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
from fasterrag.observability import metrics
from fasterrag.observability.logging import get_logger, use_trace_id
from fasterrag.services.journal import JobRecord, Journal
from fasterrag.services.lockfile import LockStore, build_lock
from fasterrag.services.sources import resolve_sources, typed_source
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
        sources: Sequence[str] | Sequence[Mapping[str, Any]],
        *,
        collection: str | None = None,
        tenant: str | None = None,
        idempotency_key: str | None = None,
    ) -> JobRecord:
        """Record a job without doing any work.

        Returns immediately so the API can answer ``202 Accepted`` with a job id. Replaying
        an idempotency key returns the original job rather than starting a second ingest.

        Args:
            sources: Typed ``{"type", "value"}`` mappings, or plain strings — the CLI's and
                the facade's shape — classified by :func:`~fasterrag.services.sources.typed_source`
                (an explicit http(s) scheme is a URL, everything else a path). Stored as
                given; nothing is fetched or decoded until the job runs, so accepting a job
                stays a constant-time operation whatever it points at.
            collection: Collection to index into; defaults to the configured one.
            tenant: Tenant the documents belong to.
            idempotency_key: Replaying one returns the original job.
        """
        typed = [
            typed_source(source) if isinstance(source, str) else dict(source) for source in sources
        ]
        return self.journal.create_job(
            collection or self.settings.vector_db.collection.default_name,
            typed,
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
        resolved = await resolve_sources(record.sources, self.settings)
        tasks = CpuWorkerPool.tasks_for(
            resolved.sources,
            tenant=record.tenant,
            metadata=metadata,
            locations=resolved.locations,
        )
        resume_from = self.journal.resume_index(record)

        indexer = Indexer(self.settings, self.adapter, collection=record.collection)
        pool = EmbeddingWorkerPool(self.settings, self.router, indexer)
        queue: BoundedQueue[ChunkPayload] = BoundedQueue(self.settings.workers.queue_depth)

        running = replace(record, status="running")
        self.journal.save_job(running)
        progress = _Checkpointer(self.journal, running)
        started = time.monotonic()

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
        self._publish_metrics(settled, counts, elapsed=time.monotonic() - started)

        if indexer.written:
            self._write_lockfile(record.collection)

        if self.cache is not None and indexer.written:
            await self.cache.invalidate(f"ingest job {record.job_id} indexed {indexer.written}")

        resolved.cleanup()

        _logger.info("ingest finished", extra={"job_id": record.job_id, **counts})
        return settled

    def _publish_metrics(
        self, record: JobRecord, counts: dict[str, int], *, elapsed: float
    ) -> None:
        """Publish the ingestion half of the metrics catalogue for a settled job.

        Published once, at settle, rather than per document: throughput measured over a
        whole job is the number an operator can compare between runs, while a per-document
        rate mostly reports how large the last document happened to be.

        Dead-letter depth is read back from the journal rather than taken from this job's
        counts. The gauge describes a collection's *standing* backlog, and a job that
        dead-lettered nothing does not mean the collection has nothing waiting.
        """
        indexed = counts["indexed"]
        failed = counts["dead_lettered"]
        succeeded = counts["total"] - failed - counts["deduplicated"]

        metrics.INGEST_DOCUMENTS.increment(float(max(succeeded, 0)), status="succeeded")
        metrics.INGEST_DOCUMENTS.increment(float(counts["deduplicated"]), status="deduplicated")
        metrics.INGEST_DOCUMENTS.increment(float(failed), status="dead_lettered")

        if elapsed > 0:
            metrics.INGEST_THROUGHPUT.set(counts["total"] / elapsed, unit="documents_per_second")
            metrics.INGEST_THROUGHPUT.set(indexed / elapsed, unit="chunks_per_second")

        metrics.DLQ_DEPTH.set(
            float(len(self.journal.dead_lettered(record.job_id))),
            collection=record.collection,
        )

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
        sources: Sequence[str] | Sequence[Mapping[str, Any]],
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
