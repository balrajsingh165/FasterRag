from collections.abc import Sequence
from typing import Any

import pytest

from fasterrag.adapters.embeddings.base import EmbeddingAdapter, EmbeddingResult
from fasterrag.adapters.embeddings.tiering import TieringRouter
from fasterrag.adapters.vectordb.base import HealthStatus
from fasterrag.config.schema import Settings
from fasterrag.core.chunking.models import TextChunk
from fasterrag.errors import EmbedError, ErrorCode
from fasterrag.workers.embed_pool import EmbeddingWorkerPool, backoff_delay
from fasterrag.workers.queues import BoundedQueue, ChunkPayload, EmbeddedBatch

DIMENSIONS = 3


class RecordingEmbedder(EmbeddingAdapter):
    """Counts calls and can be made to fail a fixed number of times."""

    provider = "recording"

    def __init__(self, settings: Settings, name: str = "test-model") -> None:
        super().__init__(settings)
        self._name = name
        self.calls: list[list[str]] = []
        self.failures = 0
        self.error: EmbedError | None = None
        self.load_count = 0

    @property
    def model(self) -> str:
        return self._name

    @property
    def model_version(self) -> str:
        return f"{self._name}-v1"

    @property
    def dimensions(self) -> int | None:
        return DIMENSIONS

    async def embed_documents(self, texts: Sequence[str]) -> EmbeddingResult:
        if self.failures > 0:
            self.failures -= 1
            raise self.error or EmbedError("provider unavailable", retryable=True)

        self.load_count = 1
        self.calls.append(list(texts))
        return EmbeddingResult(
            vectors=[[float(index)] * DIMENSIONS for index in range(len(texts))],
            model=self.model,
            model_version=self.model_version,
        )

    async def embed_query(self, text: str) -> list[float]:
        return [0.0] * DIMENSIONS

    async def health(self) -> HealthStatus:
        return HealthStatus(healthy=True)

    async def close(self) -> None:
        return None


class CollectingSink:
    """Stands in for the indexer."""

    def __init__(self) -> None:
        self.batches: list[EmbeddedBatch] = []

    async def write(self, batch: EmbeddedBatch) -> None:
        self.batches.append(batch)

    @property
    def chunk_ids(self) -> list[str]:
        return [payload.chunk_id for batch in self.batches for payload in batch.chunks]


def settings(**overrides: Any) -> Settings:
    payload: dict[str, Any] = {
        "workers": {"embedding_pool_size": 1},
        "embeddings": {"batch_size": 2},
        "reliability": {"retries": {"max_attempts": 2, "backoff_base_ms": 1, "jitter": False}},
    }
    payload.update(overrides)
    return Settings.model_validate(payload)


def payload(index: int, metadata: dict[str, Any] | None = None) -> ChunkPayload:
    return ChunkPayload(
        chunk_id=f"c_{index}",
        document_id=f"d_{index}",
        source="a.md",
        content_hash="hash",
        chunk=TextChunk(
            text=f"chunk {index}",
            start=0,
            end=7,
            chunk_index=index,
            token_count=2,
            strategy="recursive",
        ),
        metadata=metadata or {},
    )


async def feed(queue: BoundedQueue[ChunkPayload], count: int, workers: int = 1) -> None:
    for index in range(count):
        await queue.put(payload(index))
    await queue.close(consumers=workers)


async def test_every_chunk_is_embedded_and_written() -> None:
    configured = settings()
    embedder = RecordingEmbedder(configured)
    sink = CollectingSink()
    queue: BoundedQueue[ChunkPayload] = BoundedQueue(20)
    await feed(queue, 5)

    report = await EmbeddingWorkerPool(configured, TieringRouter(embedder), sink).run(queue)

    assert report.embedded == 5
    assert sink.chunk_ids == [f"c_{index}" for index in range(5)]


async def test_requests_are_batched_to_the_configured_size() -> None:
    configured = settings()
    embedder = RecordingEmbedder(configured)
    queue: BoundedQueue[ChunkPayload] = BoundedQueue(20)
    await feed(queue, 5)

    await EmbeddingWorkerPool(configured, TieringRouter(embedder), CollectingSink()).run(queue)

    assert [len(call) for call in embedder.calls] == [2, 2, 1]


async def test_the_model_is_never_reloaded_between_batches() -> None:
    configured = settings()
    embedder = RecordingEmbedder(configured)
    queue: BoundedQueue[ChunkPayload] = BoundedQueue(20)
    await feed(queue, 6)

    await EmbeddingWorkerPool(configured, TieringRouter(embedder), CollectingSink()).run(queue)

    assert embedder.load_count == 1
    assert len(embedder.calls) == 3


async def test_vectors_stay_aligned_with_their_chunks() -> None:
    configured = settings()
    sink = CollectingSink()
    queue: BoundedQueue[ChunkPayload] = BoundedQueue(20)
    await feed(queue, 4)

    await EmbeddingWorkerPool(configured, TieringRouter(RecordingEmbedder(configured)), sink).run(
        queue
    )

    for batch in sink.batches:
        assert len(batch.chunks) == len(batch.vectors)
        assert all(len(vector) == DIMENSIONS for vector in batch.vectors)


async def test_the_batch_records_the_model_that_produced_it() -> None:
    configured = settings()
    sink = CollectingSink()
    queue: BoundedQueue[ChunkPayload] = BoundedQueue(20)
    await feed(queue, 2)

    await EmbeddingWorkerPool(configured, TieringRouter(RecordingEmbedder(configured)), sink).run(
        queue
    )

    assert sink.batches[0].model == "test-model"
    assert sink.batches[0].model_version == "test-model-v1"


async def test_several_workers_share_the_queue() -> None:
    configured = settings(workers={"embedding_pool_size": 3}, embeddings={"batch_size": 1})
    sink = CollectingSink()
    queue: BoundedQueue[ChunkPayload] = BoundedQueue(20)
    await feed(queue, 6, workers=3)

    report = await EmbeddingWorkerPool(
        configured, TieringRouter(RecordingEmbedder(configured)), sink
    ).run(queue)

    assert report.embedded == 6
    assert sorted(sink.chunk_ids) == sorted(f"c_{index}" for index in range(6))


async def test_a_retryable_failure_is_retried_then_succeeds() -> None:
    configured = settings()
    embedder = RecordingEmbedder(configured)
    embedder.failures = 1
    embedder.error = EmbedError("rate limited", retryable=True)
    queue: BoundedQueue[ChunkPayload] = BoundedQueue(20)
    await feed(queue, 2)

    report = await EmbeddingWorkerPool(configured, TieringRouter(embedder), CollectingSink()).run(
        queue
    )

    assert report.retries == 1
    assert report.embedded == 2
    assert report.failed == 0


async def test_a_non_retryable_failure_is_not_retried() -> None:
    configured = settings()
    embedder = RecordingEmbedder(configured)
    embedder.failures = 99
    embedder.error = EmbedError("bad credentials", retryable=False)
    queue: BoundedQueue[ChunkPayload] = BoundedQueue(20)
    await feed(queue, 2)

    report = await EmbeddingWorkerPool(configured, TieringRouter(embedder), CollectingSink()).run(
        queue
    )

    assert report.retries == 0
    assert report.failed == 2


async def test_attempts_are_bounded_and_the_pool_keeps_running() -> None:
    configured = settings()
    embedder = RecordingEmbedder(configured)
    embedder.failures = 99
    embedder.error = EmbedError("provider down", retryable=True)
    sink = CollectingSink()
    queue: BoundedQueue[ChunkPayload] = BoundedQueue(20)
    await feed(queue, 4)

    report = await EmbeddingWorkerPool(configured, TieringRouter(embedder), sink).run(queue)

    assert report.retries == 4
    assert report.failed == 4
    assert sink.batches == []
    assert {code for _, code in report.failures} == {ErrorCode.EMBED_PROVIDER_ERROR.value}


async def test_a_failed_batch_does_not_block_later_ones() -> None:
    configured = settings()
    embedder = RecordingEmbedder(configured)
    embedder.failures = 3
    embedder.error = EmbedError("transient", retryable=True)
    sink = CollectingSink()
    queue: BoundedQueue[ChunkPayload] = BoundedQueue(20)
    await feed(queue, 4)

    report = await EmbeddingWorkerPool(configured, TieringRouter(embedder), sink).run(queue)

    assert report.failed == 2
    assert report.embedded == 2


async def test_tiered_documents_are_grouped_by_their_model() -> None:
    configured = settings(
        embeddings={"batch_size": 10},
        workers={"embedding_pool_size": 1},
    )
    default = RecordingEmbedder(configured, "default-model")
    archive = RecordingEmbedder(configured, "cheap-model")
    router = TieringRouter(default, [({"priority_class": "archive"}, archive)])
    sink = CollectingSink()

    queue: BoundedQueue[ChunkPayload] = BoundedQueue(20)
    await queue.put(payload(0))
    await queue.put(payload(1, {"priority_class": "archive"}))
    await queue.put(payload(2, {"priority_class": "archive"}))
    await queue.close(consumers=1)

    report = await EmbeddingWorkerPool(configured, router, sink).run(queue)

    assert report.embedded == 3
    assert len(default.calls) == 1
    assert len(archive.calls) == 1
    assert len(archive.calls[0]) == 2
    assert {batch.model for batch in sink.batches} == {"default-model", "cheap-model"}


async def test_an_empty_queue_produces_no_work() -> None:
    configured = settings()
    queue: BoundedQueue[ChunkPayload] = BoundedQueue(5)
    await queue.close(consumers=1)

    report = await EmbeddingWorkerPool(
        configured, TieringRouter(RecordingEmbedder(configured)), CollectingSink()
    ).run(queue)

    assert report.embedded == 0
    assert report.batches == 0


def test_backoff_grows_and_is_capped() -> None:
    configured = Settings.model_validate(
        {
            "reliability": {
                "retries": {"backoff_base_ms": 100, "backoff_max_ms": 400, "jitter": False}
            }
        }
    )

    assert backoff_delay(1, configured) == pytest.approx(0.1)
    assert backoff_delay(2, configured) == pytest.approx(0.2)
    assert backoff_delay(3, configured) == pytest.approx(0.4)
    assert backoff_delay(9, configured) == pytest.approx(0.4)


def test_jitter_keeps_delays_inside_half_the_window() -> None:
    configured = Settings.model_validate(
        {
            "reliability": {
                "retries": {"backoff_base_ms": 200, "backoff_max_ms": 1000, "jitter": True}
            }
        }
    )

    delays = [backoff_delay(2, configured) for _ in range(30)]

    assert all(0.2 <= delay <= 0.4 for delay in delays)
    assert len(set(delays)) > 1
