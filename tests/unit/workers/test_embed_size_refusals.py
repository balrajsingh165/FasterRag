"""The embedding pool adapting to what a provider refuses (TASK-0478).

Batching was a fixed slice. A provider that refused a batch for its size had the same batch
sent back until the retry budget ran out, and every chunk in it was dead-lettered, so one
over-long chunk took its whole batch with it; a ``Retry-After`` was ignored in favour of the
pool's own backoff.
"""

import asyncio
import time
from collections.abc import Sequence
from typing import Any

import pytest

from fasterrag.adapters.embeddings.base import (
    EmbeddingResult,
    EmbedThrottledError,
    EmbedTooLargeError,
)
from fasterrag.adapters.embeddings.tiering import TieringRouter
from fasterrag.config.schema import Settings
from fasterrag.core.breaker import CircuitBreaker, CircuitState
from fasterrag.errors import ErrorCode
from fasterrag.workers import embed_pool
from fasterrag.workers.embed_pool import EmbeddingWorkerPool
from fasterrag.workers.ledger import IndexLedger
from fasterrag.workers.queues import BoundedQueue, ChunkPayload
from tests.unit.workers.test_embed_pool import (
    DIMENSIONS,
    CollectingSink,
    RecordingEmbedder,
    feed,
    settings,
)


class SizeLimitedEmbedder(RecordingEmbedder):
    """Refuses any batch larger than ``limit``, and any text listed in ``oversized``."""

    def __init__(self, configured: Settings, *, limit: int, oversized: Sequence[str] = ()) -> None:
        super().__init__(configured)
        self.limit = limit
        self.oversized = set(oversized)
        self.refused: list[int] = []

    async def embed_documents(self, texts: Sequence[str]) -> EmbeddingResult:
        if len(texts) > self.limit or self.oversized.intersection(texts):
            self.refused.append(len(texts))
            raise EmbedTooLargeError("recording refused the request as too large")
        return await super().embed_documents(texts)


class ThrottlingEmbedder(RecordingEmbedder):
    """Answers the first call with a rate limit that names a pause."""

    def __init__(self, configured: Settings, *, pause: float) -> None:
        super().__init__(configured)
        self.pause = pause
        self.throttled = False

    async def embed_documents(self, texts: Sequence[str]) -> EmbeddingResult:
        if not self.throttled:
            self.throttled = True
            raise EmbedThrottledError(
                "recording returned status 429", retry_after_seconds=self.pause
            )
        return await super().embed_documents(texts)


async def run(pool: EmbeddingWorkerPool, count: int) -> embed_pool.EmbedReport:
    queue: BoundedQueue[ChunkPayload] = BoundedQueue(50)
    await feed(queue, count, workers=pool.size)
    return await pool.run(queue)


async def test_a_batch_refused_for_its_size_is_halved_until_it_fits() -> None:
    configured = settings(embeddings={"batch_size": 8})
    embedder = SizeLimitedEmbedder(configured, limit=2)
    sink = CollectingSink()

    report = await run(EmbeddingWorkerPool(configured, TieringRouter(embedder), sink), 8)

    assert report.failed == 0
    assert report.embedded == 8
    assert sorted(sink.chunk_ids) == sorted(f"c_{index}" for index in range(8))
    assert embedder.refused == [8, 4, 4]
    assert all(len(call) <= 2 for call in embedder.calls)


async def test_vectors_stay_with_their_chunks_after_a_split() -> None:
    configured = settings(embeddings={"batch_size": 4})
    embedder = SizeLimitedEmbedder(configured, limit=1)
    sink = CollectingSink()

    await run(EmbeddingWorkerPool(configured, TieringRouter(embedder), sink), 4)

    for batch in sink.batches:
        assert len(batch.chunks) == len(batch.vectors) == 1
        assert len(batch.vectors[0]) == DIMENSIONS


async def test_an_input_too_large_on_its_own_fails_alone() -> None:
    """The whole batch used to be dead-lettered with the one chunk that could never fit."""
    configured = settings(embeddings={"batch_size": 4})
    embedder = SizeLimitedEmbedder(configured, limit=4, oversized=["chunk 2"])
    sink = CollectingSink()
    ledger = IndexLedger()
    for index in range(4):
        await ledger.expect(
            f"d_{index}", index=index, source=f"{index}.md", chunks=1, commit=_nothing
        )

    report = await run(
        EmbeddingWorkerPool(configured, TieringRouter(embedder), sink, ledger=ledger), 4
    )

    assert report.embedded == 3
    assert report.failed == 1
    assert report.failures == [("d_2", ErrorCode.EMBED_PROVIDER_ERROR.value)]
    assert sorted(sink.chunk_ids) == ["c_0", "c_1", "c_3"]
    assert [entry.document_id for entry in ledger.failed_documents] == ["d_2"]


async def test_a_size_refusal_spends_no_retries() -> None:
    configured = settings(embeddings={"batch_size": 4})
    embedder = SizeLimitedEmbedder(configured, limit=2)

    report = await run(
        EmbeddingWorkerPool(configured, TieringRouter(embedder), CollectingSink()), 4
    )

    assert report.retries == 0


async def test_a_size_refusal_does_not_count_against_the_breaker() -> None:
    """The caller sent too much; the provider is healthy and must not be shed."""
    configured = settings(embeddings={"batch_size": 8})
    breaker = CircuitBreaker("embeddings", failure_threshold=1)
    embedder = SizeLimitedEmbedder(configured, limit=1)

    report = await run(
        EmbeddingWorkerPool(configured, TieringRouter(embedder), CollectingSink(), breaker), 8
    )

    assert breaker.state is CircuitState.CLOSED
    assert report.embedded == 8


async def test_a_size_refusal_answers_a_half_open_probe() -> None:
    """The provider answered, so the probe's lease must not be left held."""
    configured = settings(embeddings={"batch_size": 2})
    breaker = CircuitBreaker("embeddings", failure_threshold=1, reset_timeout_ms=50)
    breaker.record_failure(breaker.check())
    time.sleep(0.06)
    embedder = SizeLimitedEmbedder(configured, limit=1)

    report = await run(
        EmbeddingWorkerPool(configured, TieringRouter(embedder), CollectingSink(), breaker), 2
    )

    assert report.embedded == 2
    assert breaker.state is CircuitState.CLOSED


async def test_the_pause_a_provider_asks_for_is_honoured(monkeypatch: pytest.MonkeyPatch) -> None:
    configured = settings()
    delays = _record_sleeps(monkeypatch)
    embedder = ThrottlingEmbedder(configured, pause=0.75)

    report = await run(
        EmbeddingWorkerPool(configured, TieringRouter(embedder), CollectingSink()), 2
    )

    assert report.embedded == 2
    assert delays == [0.75]


async def test_the_pause_is_bounded_by_the_backoff_ceiling(monkeypatch: pytest.MonkeyPatch) -> None:
    configured = settings(
        reliability={
            "retries": {
                "max_attempts": 2,
                "backoff_base_ms": 1,
                "backoff_max_ms": 2000,
                "jitter": False,
            }
        }
    )
    delays = _record_sleeps(monkeypatch)
    embedder = ThrottlingEmbedder(configured, pause=600.0)

    await run(EmbeddingWorkerPool(configured, TieringRouter(embedder), CollectingSink()), 2)

    assert delays == [2.0]


async def test_a_shorter_pause_never_undercuts_the_backoff(monkeypatch: pytest.MonkeyPatch) -> None:
    configured = settings(
        reliability={"retries": {"max_attempts": 2, "backoff_base_ms": 500, "jitter": False}}
    )
    delays = _record_sleeps(monkeypatch)
    embedder = ThrottlingEmbedder(configured, pause=0.1)

    await run(EmbeddingWorkerPool(configured, TieringRouter(embedder), CollectingSink()), 2)

    assert delays == [0.5]


async def _nothing() -> None:
    return None


def _record_sleeps(monkeypatch: pytest.MonkeyPatch) -> list[float]:
    delays: list[float] = []

    async def sleep(delay: float, *args: Any) -> None:
        delays.append(delay)

    monkeypatch.setattr(asyncio, "sleep", sleep)
    return delays
