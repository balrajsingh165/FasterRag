"""Embedding worker pool: stateful, batched, retryable.

Workers consume the chunk queue and embed in batches. The model is loaded once per
configured model and reused across every batch for the pool's whole lifetime — reloading
per task is the single largest avoidable cost in a naive pipeline and is prohibited by
design (``docs/architecture.md`` §2). The tiering router owns the adapters, so a document
routed to a cheap model reuses that model's single loaded instance too.

Retries live here rather than in the adapters. An adapter classifies a failure and says
whether it is worth repeating; this pool decides how often and how long to wait, with
exponential backoff and jitter from ``reliability.retries`` (``docs/reliability.md`` §2).
A batch that exhausts its attempts dead-letters its documents with a reason code instead of
stalling the pipeline.

The pool is also where embedding spend is accounted. Every successful batch is priced
through ``services.estimation`` and charged to the shared ``CostGovernor``, using the
*provider's* reported token count where the adapter surfaced one and the local count only
where it did not — so a deployment can reconcile ``fasterrag_cost_usd_total`` against an
invoice instead of against its own tokenizer. Where both numbers exist and disagree, the
difference is published as ``EmbedReport.token_drift`` rather than quietly discarded. The
price table is imported inside ``_charge`` rather than at module scope because the estimator
imports this package, and the two deadlock on import otherwise.
"""

from __future__ import annotations

import asyncio
import random
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Protocol, cast, runtime_checkable

from fasterrag.adapters.embeddings.base import EmbeddingAdapter, EmbeddingResult
from fasterrag.adapters.embeddings.tiering import TieringRouter
from fasterrag.config.schema import Settings
from fasterrag.core.breaker import Admission, CircuitBreaker, CircuitOpenError, shared_breaker
from fasterrag.core.chunking.models import Segment
from fasterrag.errors import ErrorCode, FasterRagError, IngestionError
from fasterrag.observability import metrics
from fasterrag.observability.logging import get_logger
from fasterrag.services.budget import CostGovernor, shared_governor
from fasterrag.workers.ledger import IndexLedger
from fasterrag.workers.queues import BoundedQueue, ChunkPayload, EmbeddedBatch

__all__ = [
    "ChunkSink",
    "EmbedReport",
    "EmbeddingWorkerPool",
    "PoolingAdapter",
    "backoff_delay",
]

_logger = get_logger(__name__)


class ChunkSink(Protocol):
    """Receives embedded batches. The indexer implements this."""

    async def write(self, batch: EmbeddedBatch) -> None:
        """Persist a batch of embedded chunks."""
        ...


@runtime_checkable
class PoolingAdapter(Protocol):
    """An embedding adapter that can late-chunk a document."""

    async def embed_pooled(self, document: str, spans: Sequence[Segment]) -> EmbeddingResult:
        """Return one vector per span, pooled from a pass over the whole document."""
        ...


@dataclass
class EmbedReport:
    """What an embedding pass produced."""

    embedded: int = 0
    batches: int = 0
    retries: int = 0
    failed: int = 0
    failures: list[tuple[str, str]] = field(default_factory=list)
    tokens_counted: int = 0
    """Tokens this pass counted locally, over every batch it embedded."""

    tokens_reported: int = 0
    """Tokens the providers said they billed, over the batches that stated a figure."""

    tokens_counted_where_reported: int = 0
    """The local count for exactly those batches, so ``token_drift`` compares like with like."""

    batches_without_usage: int = 0
    """Batches whose provider stated no usage, so their spend is priced from the local count."""

    @property
    def token_drift(self) -> int:
        """Return provider-reported minus locally-counted tokens, over reporting batches.

        Published rather than reconciled away: a non-zero drift means our tokenizer and the
        invoice disagree, and silently preferring either one hides that.
        """
        return self.tokens_reported - self.tokens_counted_where_reported


def backoff_delay(attempt: int, settings: Settings) -> float:
    """Return the seconds to wait before an attempt.

    Exponential with a ceiling, and jittered when configured so a fleet of workers that
    failed together does not retry in lockstep and re-create the spike that caused it.
    """
    retries = settings.reliability.retries
    exponential = retries.backoff_base_ms * (2 ** max(attempt - 1, 0))
    capped = float(min(exponential, retries.backoff_max_ms))
    if retries.jitter:
        capped = random.uniform(capped / 2, capped)
    return capped / 1000


class EmbeddingWorkerPool:
    """Embeds queued chunks in batches and hands them to a sink."""

    def __init__(
        self,
        settings: Settings,
        router: TieringRouter,
        sink: ChunkSink,
        breaker: CircuitBreaker | None = None,
        *,
        tenant: str | None = None,
        governor: CostGovernor | None = None,
        ledger: IndexLedger | None = None,
    ) -> None:
        """Build the pool."""
        self.settings = settings
        self.router = router
        self.sink = sink
        self.tenant = tenant
        self.ledger = ledger
        # CRITICAL: the shared governor, not a fresh one. A per-job instance would charge a
        # rolling tenant window that is discarded with the job, which looks like accounting
        # and enforces nothing.
        self.governor = governor or shared_governor(settings)
        # CRITICAL: the process-wide breaker, not a fresh one. This object is built per
        # request on the REST path and per job on the ingest path, and a breaker rebuilt
        # per unit of work counts to one forever and can never open.
        self.breaker = breaker or shared_breaker("embeddings", settings)
        self.size = settings.workers.embedding_pool_size
        self.batch_size = settings.embeddings.batch_size
        self.report = EmbedReport()

    async def run(self, source: BoundedQueue[ChunkPayload]) -> EmbedReport:
        """Consume ``source`` until it closes, embedding and writing every chunk.

        The caller closes the queue with one sentinel per worker once the producer is
        finished, so shutdown needs no timeout and no polling.
        """
        self.report = EmbedReport()
        workers = [asyncio.create_task(self._worker(source, index)) for index in range(self.size)]
        try:
            await asyncio.gather(*workers)
        except FasterRagError:
            await self._stop(workers)
            raise
        except Exception as exc:
            # CRITICAL: one worker raising something untyped must end the pool loudly. Left
            # alone, the surviving workers keep draining while the dead one's share of the
            # queue backs up, and a producer blocked on a full queue then waits forever.
            await self._stop(workers)
            raise IngestionError(
                f"an embedding worker stopped unexpectedly: {type(exc).__name__}: {exc}",
                code=ErrorCode.INTERNAL,
                retryable=False,
            ) from exc
        return self.report

    @staticmethod
    async def _stop(workers: Sequence[asyncio.Task[None]]) -> None:
        """Cancel every worker still running and wait for them to finish unwinding."""
        for worker in workers:
            worker.cancel()
        await asyncio.gather(*workers, return_exceptions=True)

    async def _worker(self, source: BoundedQueue[ChunkPayload], index: int) -> None:
        """Drain the queue into batches, embedding each one."""
        batch: list[ChunkPayload] = []

        while True:
            payload = await source.get()
            if payload is None:
                source.task_done()
                break

            batch.append(payload)
            source.task_done()

            if len(batch) >= self.batch_size:
                await self._flush(batch, index)
                batch = []

        if batch:
            await self._flush(batch, index)

    async def _flush(self, batch: Sequence[ChunkPayload], worker: int) -> None:
        """Embed one batch, grouping by the model its documents route to."""
        for adapter, grouped in self._group(batch).items():
            await self._embed_group(adapter, grouped, worker)

    def _group(self, batch: Sequence[ChunkPayload]) -> dict[EmbeddingAdapter, list[ChunkPayload]]:
        """Split a batch by the adapter each chunk's metadata routes it to."""
        grouped: dict[EmbeddingAdapter, list[ChunkPayload]] = {}
        for payload in batch:
            adapter = self.router.select(payload.metadata)
            grouped.setdefault(adapter, []).append(payload)
        return grouped

    def _pooled_documents(
        self, adapter: EmbeddingAdapter, group: Sequence[ChunkPayload]
    ) -> dict[str, list[ChunkPayload]] | None:
        """Return the group split by document when it should be embedded by pooling."""
        if self.settings.chunking.strategy != "late":
            return None
        if not isinstance(adapter, PoolingAdapter):
            return None
        if not all(payload.document_text for payload in group):
            return None

        documents: dict[str, list[ChunkPayload]] = {}
        for payload in group:
            documents.setdefault(payload.document_id, []).append(payload)
        return documents

    async def _embed_pooled(
        self, adapter: PoolingAdapter, documents: dict[str, list[ChunkPayload]]
    ) -> EmbeddingResult:
        """Pool every document in the group, returning vectors in the group's own order."""
        vectors: list[list[float]] = []
        model = ""
        version = ""

        for chunks in documents.values():
            spans = [(payload.chunk.start, payload.chunk.end) for payload in chunks]
            result = await adapter.embed_pooled(chunks[0].document_text, spans)
            vectors.extend(result.vectors)
            model, version = result.model, result.model_version

        return EmbeddingResult(vectors=vectors, model=model, model_version=version)

    async def _embed_group(
        self, adapter: EmbeddingAdapter, group: Sequence[ChunkPayload], worker: int
    ) -> None:
        """Embed one same-model group, retrying only what the provider said to retry."""
        attempts = self.settings.reliability.retries.max_attempts
        last: FasterRagError | None = None
        documents = self._pooled_documents(adapter, group)

        # CRITICAL: when pooling, the group is reordered to match the per-document batches
        # the vectors come back in. Writing the original order against pooled vectors would
        # pair every chunk with another chunk's vector — a corpus that indexes cleanly and
        # retrieves nonsense.
        ordered = (
            [payload for chunks in documents.values() for payload in chunks]
            if documents
            else list(group)
        )

        for attempt in range(attempts + 1):
            admission: Admission | None = None
            try:
                # CRITICAL: checked inside the retry loop, not before it. A breaker that
                # opens mid-batch must stop the *remaining* attempts — checking once at the
                # top would spend the whole retry budget on a provider already known dead.
                admission = self.breaker.check()
                result = (
                    await self._embed_pooled(cast(PoolingAdapter, adapter), documents)
                    if documents
                    else await adapter.embed_documents([payload.text for payload in group])
                )
            except FasterRagError as exc:
                last = exc
                if not isinstance(exc, CircuitOpenError):
                    self.breaker.record_failure(admission, exc)
                if not exc.retryable or attempt == attempts or self.breaker.is_open:
                    break

                self.report.retries += 1
                delay = backoff_delay(attempt + 1, self.settings)
                _logger.warning(
                    "retrying an embedding batch",
                    extra={
                        "worker": worker,
                        "attempt": attempt + 1,
                        "delay_seconds": round(delay, 3),
                        "code": exc.code.value,
                        "trace_id": exc.trace_id,
                    },
                )
                await asyncio.sleep(delay)
                continue

            self.breaker.record_success(admission)
            self._charge(adapter, ordered, result)
            self.report.embedded += len(group)
            self.report.batches += 1

            failure = await self._write(
                EmbeddedBatch(
                    chunks=ordered,
                    vectors=result.vectors,
                    model=result.model,
                    model_version=result.model_version,
                ),
                worker,
            )
            if failure is not None:
                await self._give_up(group, failure)
            elif self.ledger is not None:
                await self.ledger.written(ordered)
            return

        await self._give_up(group, last)

    async def _write(self, batch: EmbeddedBatch, worker: int) -> FasterRagError | None:
        """Hand a batch to the sink, retrying what is retryable, and return the last failure."""
        attempts = self.settings.reliability.retries.max_attempts
        last: FasterRagError | None = None

        for attempt in range(attempts + 1):
            try:
                await self.sink.write(batch)
            except FasterRagError as exc:
                last = exc
                if not exc.retryable or attempt == attempts:
                    break

                self.report.retries += 1
                delay = backoff_delay(attempt + 1, self.settings)
                _logger.warning(
                    "retrying an index write",
                    extra={
                        "worker": worker,
                        "attempt": attempt + 1,
                        "delay_seconds": round(delay, 3),
                        "code": exc.code.value,
                        "trace_id": exc.trace_id,
                    },
                )
                await asyncio.sleep(delay)
                continue
            return None

        return last

    def _charge(
        self, adapter: EmbeddingAdapter, group: Sequence[ChunkPayload], result: EmbeddingResult
    ) -> None:
        """Account for one embedded batch, in tokens and in money.

        The provider's figure is billed where it stated one and the local count where it
        did not; the gap between them is recorded rather than resolved.
        """
        from fasterrag.services.estimation import price_for

        counted = sum(payload.chunk.token_count for payload in group)
        reported = result.total_tokens
        self.report.tokens_counted += counted

        if reported is None:
            self.report.batches_without_usage += 1
        else:
            self.report.tokens_reported += reported
            self.report.tokens_counted_where_reported += counted

        billed = counted if reported is None else reported
        provider = adapter.provider
        metrics.TOKENS.increment(
            float(billed), kind="embedding", provider=provider, tenant=self.tenant or "none"
        )

        cost, _ = price_for(provider, result.model, billed)
        if cost is None:
            # CRITICAL: an unpriced model must be visible rather than absent. Contributing
            # nothing to the cost total makes a dashboard understate spend while looking
            # healthy, which reads as "this was free" instead of "this is unknown".
            metrics.UNPRICED_TOKENS.increment(float(billed), provider=provider, model=result.model)
        elif cost:
            metrics.COST.increment(cost, provider=provider, tenant=self.tenant or "none")

        self.governor.record(billed, tenant=self.tenant)

    async def _give_up(self, group: Sequence[ChunkPayload], error: FasterRagError | None) -> None:
        """Record a batch that could not be embedded or written, without stopping the pool."""
        code = error.code.value if error else ErrorCode.EMBED_PROVIDER_ERROR.value
        detail = error.detail if error else "the embedding provider could not be reached"

        self.report.failed += len(group)
        for payload in group:
            self.report.failures.append((payload.document_id, code))

        _logger.error(
            "giving up on an embedding batch",
            extra={
                "chunks": len(group),
                "code": code,
                "detail": detail,
                "trace_id": error.trace_id if error else None,
            },
        )
        if self.ledger is not None:
            await self.ledger.failed(group, code)
