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

**Acceptance refuses a corpus the live index cannot hold** (TASK-0119, TASK-0320).
:func:`~fasterrag.services.reindex.require_keyword_leg` runs in :meth:`IngestionService.accept`,
before a job record exists, so configuring ``retrieval.hybrid`` over a collection indexed
dense-only is a named refusal rather than a job that is accepted and then dies at its first
upsert under ``EMBED_PROVIDER_ERROR`` — a code naming the embedding provider for a mistake
the embedding provider did not make. ``accept`` is the one place that covers the REST route,
the CLI and a direct library caller at once, which is why the gate is here rather than in
each entry point. The cost is one metadata round trip **per job**, noise beside a corpus
that takes minutes to hours; the same probe inside :meth:`run` would charge every document
for a mistake made once, so nothing on the per-document path repeats it.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Callable, Coroutine, Mapping, Sequence
from concurrent.futures import Executor
from dataclasses import replace
from datetime import UTC, datetime
from typing import Any

from fasterrag.adapters.embeddings.tiering import TieringRouter
from fasterrag.adapters.llm.factory import create_llm_adapter
from fasterrag.adapters.vectordb.base import VectorDBAdapter
from fasterrag.adapters.vectordb.factory import create_vector_db_adapter
from fasterrag.config.schema import Settings
from fasterrag.core.cache.semantic import SemanticCache
from fasterrag.errors import ConfigError, ErrorCode, FasterRagError, IngestionError
from fasterrag.observability import metrics
from fasterrag.observability.logging import get_logger, use_trace_id
from fasterrag.services.composition import create_embedding_router
from fasterrag.services.journal import JobRecord, Journal
from fasterrag.services.lockfile import LockStore, build_lock
from fasterrag.services.reindex import require_keyword_leg
from fasterrag.services.sources import resolve_sources, typed_source
from fasterrag.workers.cpu_pool import CpuWorkerPool, PoolReport
from fasterrag.workers.embed_pool import EmbeddingWorkerPool, EmbedReport
from fasterrag.workers.indexer import Indexer
from fasterrag.workers.ledger import IndexLedger
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
        """Build the service."""
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

        Refuses before the record exists if the live index cannot serve the configured
        keyword leg — one metadata round trip per job, covering every caller (TASK-0320).
        """
        target = collection or self.settings.vector_db.collection.default_name
        _require_pipeline_chunker(self.settings)
        await require_keyword_leg(target, self.settings, self.adapter)

        typed = [
            typed_source(source) if isinstance(source, str) else dict(source) for source in sources
        ]
        return await self.journal.create_job(
            target,
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

        Any failure settles the job as ``failed`` before it propagates, so a job can never
        be left ``running`` by an exception — typed or not — on any of the three surfaces.
        """
        with use_trace_id():
            try:
                return await self._run(record, metadata)
            except FasterRagError as exc:
                await self._settle_failed(record, exc.code.value, exc.detail)
                raise
            except Exception as exc:
                await self._settle_failed(record, ErrorCode.INTERNAL.value, repr(exc))
                raise IngestionError(
                    f"ingest job {record.job_id} stopped on an unexpected error: "
                    f"{type(exc).__name__}: {exc}",
                    code=ErrorCode.INTERNAL,
                    retryable=False,
                ) from exc

    async def _settle_failed(self, record: JobRecord, code: str, detail: str) -> None:
        """Persist a terminal ``failed`` status, keeping whatever checkpoint was reached."""
        try:
            current = await self.journal.load_job(record.job_id)
        except FasterRagError:
            current = record
        await self.journal.save_job(replace(current, status="failed", finished_at=_finished_at()))
        _logger.error(
            "ingest job failed",
            extra={"job_id": record.job_id, "code": code, "detail": detail},
        )

    async def _run(self, record: JobRecord, metadata: dict[str, Any] | None) -> JobRecord:
        """Resolve, parse, embed and index a job, then settle its record."""
        resolved = await resolve_sources(record.sources, self.settings)
        enricher = None
        try:
            tasks = CpuWorkerPool.tasks_for(
                resolved.sources,
                tenant=record.tenant,
                metadata=metadata,
                locations=resolved.locations,
            )
            resume_from = self.journal.resume_index(record)

            running = replace(record, status="running")
            await self.journal.save_job(running)
            progress = _Checkpointer(self.journal, running)
            ledger = IndexLedger(progress, start=resume_from)

            indexer = Indexer(self.settings, self.adapter, collection=record.collection)
            pool = EmbeddingWorkerPool(
                self.settings, self.router, indexer, tenant=record.tenant, ledger=ledger
            )
            queue: BoundedQueue[ChunkPayload] = BoundedQueue(self.settings.workers.queue_depth)
            started = time.monotonic()

            _logger.info(
                "ingest started",
                extra={
                    "job_id": record.job_id,
                    "collection": record.collection,
                    "documents": len(tasks),
                    "resume_from": resume_from,
                },
            )

            if self.settings.chunking.contextual_enrichment:
                enricher = create_llm_adapter(self.settings)

            async with CpuWorkerPool(
                self.settings,
                journal=self.journal,
                executor_factory=self._executor_factory,
                llm=enricher,
            ) as cpu:
                parse_report, embed_report = await _pipe(
                    cpu.process(
                        tasks,
                        queue,
                        job=record.job_id,
                        collection=record.collection,
                        resume_from=resume_from,
                        ledger=ledger,
                    ),
                    pool,
                    queue,
                )
        finally:
            if enricher is not None:
                await enricher.close()
            resolved.cleanup()

        failed = ledger.failed_documents
        for document in failed:
            await self.journal.dead_letter(
                record.job_id,
                document=document.document_id,
                source=document.source,
                reason_code=ErrorCode(document.code),
                detail="the document could not be embedded and indexed after its retries",
                attempts=self.settings.reliability.retries.max_attempts,
            )

        counts = {
            "total": len(tasks),
            "parsed": parse_report.parsed,
            "chunked": parse_report.chunked,
            "embedded": embed_report.embedded,
            "indexed": indexer.written,
            "deduplicated": parse_report.deduplicated,
            "skipped": parse_report.empty,
            "dead_lettered": parse_report.dead_lettered + len(failed),
        }
        settled = replace(
            progress.record,
            status=_status_for(counts),
            counts=counts,
            finished_at=_finished_at(),
        )
        await self.journal.save_job(settled)
        await self._publish_metrics(settled, counts, elapsed=time.monotonic() - started)

        if indexer.written:
            await self._write_lockfile(record.collection)

        if self.cache is not None and indexer.written:
            await self.cache.invalidate(f"ingest job {record.job_id} indexed {indexer.written}")

        _logger.info("ingest finished", extra={"job_id": record.job_id, **counts})
        _logger.info(
            "embedding usage reconciled",
            extra={
                "job_id": record.job_id,
                "tokens_counted": embed_report.tokens_counted,
                "tokens_reported": embed_report.tokens_reported,
                "token_drift": embed_report.token_drift,
                "batches_without_usage": embed_report.batches_without_usage,
            },
        )
        return settled

    async def _publish_metrics(
        self, record: JobRecord, counts: dict[str, int], *, elapsed: float
    ) -> None:
        """Publish the ingestion half of the metrics catalogue for a settled job."""
        indexed = counts["indexed"]
        failed = counts["dead_lettered"]
        succeeded = counts["total"] - failed - counts["deduplicated"] - counts["skipped"]

        metrics.INGEST_DOCUMENTS.increment(float(max(succeeded, 0)), status="succeeded")
        metrics.INGEST_DOCUMENTS.increment(float(counts["deduplicated"]), status="deduplicated")
        metrics.INGEST_DOCUMENTS.increment(float(counts["skipped"]), status="skipped")
        metrics.INGEST_DOCUMENTS.increment(float(failed), status="dead_lettered")

        if elapsed > 0:
            metrics.INGEST_THROUGHPUT.set(counts["total"] / elapsed, unit="documents_per_second")
            metrics.INGEST_THROUGHPUT.set(indexed / elapsed, unit="chunks_per_second")

        metrics.DLQ_DEPTH.set(
            float(len(await self.journal.dead_lettered(record.job_id))),
            collection=record.collection,
        )

    async def _write_lockfile(self, collection: str) -> None:
        """Record what produced this index, so drift against it is detectable (D1).

        Written after the job settles rather than before it runs: a lockfile describing an
        index that failed halfway would claim a corpus the collection does not hold.
        """
        if self.locks is None or not self.locks.enabled:
            return

        embedder = self.router.default
        lock = build_lock(
            collection,
            self.settings,
            embedding_model=embedder.model,
            embedding_model_version=embedder.model_version,
            dimensions=embedder.dimensions,
            document_hashes=await self.journal.document_hashes(collection),
        )
        # TODO: offload belongs inside LockStore, as it now does inside Journal. LockStore
        # still has synchronous callers (`facade.index_lock`, the CLI, `archive_import`),
        # so it needs the same two-faced treatment as a change of its own — NEW-1.
        await asyncio.to_thread(self.locks.write, lock)

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

    async def __call__(self, index: int) -> None:
        """Checkpoint after the document at ``index``, off the event loop."""
        self.record = await self.journal.checkpoint(self.record, index)


async def _pipe(
    producing: Coroutine[Any, Any, PoolReport],
    pool: EmbeddingWorkerPool,
    queue: BoundedQueue[ChunkPayload],
) -> tuple[PoolReport, EmbedReport]:
    """Run the parse side and the embedding side together, failing both if either fails."""
    producer = asyncio.ensure_future(producing)
    consumer = asyncio.ensure_future(pool.run(queue))

    await asyncio.wait({producer, consumer}, return_when=asyncio.FIRST_COMPLETED)

    if consumer.done() and not producer.done():
        producer.cancel()
        await asyncio.gather(producer, return_exceptions=True)
        embed_report = consumer.result()
        raise IngestionError(
            "the embedding pool stopped before the parser finished, having processed "
            f"{embed_report.embedded} chunks",
            code=ErrorCode.INTERNAL,
            retryable=False,
        )

    try:
        parse_report = producer.result()
    except BaseException:
        consumer.cancel()
        await asyncio.gather(consumer, return_exceptions=True)
        raise

    await queue.close(consumers=pool.size)
    return parse_report, await consumer


def _require_pipeline_chunker(settings: Settings) -> None:
    """Refuse a job whose chunking strategy the ingestion pipeline cannot run."""
    if settings.chunking.strategy == "semantic":
        raise ConfigError(
            "chunking.strategy is 'semantic', which the ingestion pipeline cannot run yet: "
            "the parse workers have no embedding model to find sentence boundaries with. "
            "Choose 'recursive', 'layout', 'fixed' or 'late'; SemanticChunker remains usable "
            "as a standalone component (fasterrag.chunking)"
        )


def _status_for(counts: dict[str, int]) -> str:
    """Return the terminal status implied by a job's counts."""
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
