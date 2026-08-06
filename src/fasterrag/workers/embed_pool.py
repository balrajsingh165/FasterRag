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
from fasterrag.core.breaker import CircuitBreaker, CircuitOpenError
from fasterrag.core.chunking.models import Segment
from fasterrag.errors import ErrorCode, FasterRagError
from fasterrag.observability.logging import get_logger
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
    """An embedding adapter that can late-chunk a document.

    Kept separate from ``EmbeddingAdapter`` because only a locally loaded model can satisfy
    it: pooling needs token-level output, and an embedding API returns one vector per input
    with no way to ask for more. Making it part of the base contract would force every
    hosted adapter to implement something it cannot do.
    """

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
    ) -> None:
        """Build the pool.

        Args:
            settings: Validated configuration; ``workers.embedding_pool_size`` sizes the
                pool and ``embeddings.batch_size`` sizes each request.
            router: Owns the adapters, so each model is loaded exactly once.
            sink: Receives embedded batches.
            breaker: Circuit breaker for the embedding provider. Built from
                ``reliability.circuit_breaker`` when omitted; injected by tests.
        """
        self.settings = settings
        self.router = router
        self.sink = sink
        self.breaker = breaker or CircuitBreaker(
            provider="embeddings",
            failure_threshold=settings.reliability.circuit_breaker.failure_threshold,
            reset_timeout_ms=settings.reliability.circuit_breaker.reset_timeout_ms,
            enabled=settings.reliability.circuit_breaker.enabled,
        )
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
        await asyncio.gather(*workers)
        return self.report

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
        """Return the group split by document when it should be embedded by pooling.

        ``None`` means embed normally, which covers every case pooling cannot serve: a
        strategy other than ``late``, an adapter with no token-level output (any hosted
        provider), or chunks that reached here without their document. Falling back rather
        than failing is deliberate — a slightly worse vector beats a dead-lettered corpus,
        and the alternative would make ``strategy: late`` unusable on a hosted model
        instead of merely no better than ``recursive``.
        """
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
        """Pool every document in the group, returning vectors in the group's own order.

        One pass per document rather than one per chunk: that single pass is the whole
        point, and it is what makes each chunk's vector carry context from beyond its own
        boundaries.
        """
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
            try:
                # CRITICAL: checked inside the retry loop, not before it. A breaker that
                # opens mid-batch must stop the *remaining* attempts — checking once at the
                # top would spend the whole retry budget on a provider already known dead.
                self.breaker.check()
                result = (
                    await self._embed_pooled(cast(PoolingAdapter, adapter), documents)
                    if documents
                    else await adapter.embed_documents([payload.text for payload in group])
                )
            except FasterRagError as exc:
                last = exc
                if not isinstance(exc, CircuitOpenError):
                    self.breaker.record_failure(exc)
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

            await self.sink.write(
                EmbeddedBatch(
                    chunks=ordered,
                    vectors=result.vectors,
                    model=result.model,
                    model_version=result.model_version,
                )
            )
            self.breaker.record_success()
            self.report.embedded += len(group)
            self.report.batches += 1
            return

        self._give_up(group, last)

    def _give_up(self, group: Sequence[ChunkPayload], error: FasterRagError | None) -> None:
        """Record a batch that could not be embedded, without stopping the pool."""
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
