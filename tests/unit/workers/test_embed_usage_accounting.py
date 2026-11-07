"""What the embedding pool does with the token usage a provider reports.

The gap this covers: an adapter that extracts billed units but hands them to nothing has
changed a dataclass, not a bill. These tests pin the other half — that the reported figure
is what gets priced and charged, that the local count is used only where there is no
receipt, and that a disagreement between the two is published rather than resolved away.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from fasterrag.adapters.embeddings.base import EmbeddingAdapter, EmbeddingResult
from fasterrag.adapters.embeddings.tiering import TieringRouter
from fasterrag.adapters.vectordb.base import HealthStatus
from fasterrag.config.schema import Settings
from fasterrag.core.chunking.models import TextChunk
from fasterrag.services.budget import CostGovernor
from fasterrag.workers.embed_pool import EmbeddingWorkerPool
from fasterrag.workers.queues import BoundedQueue, ChunkPayload, EmbeddedBatch

DIMENSIONS = 3
LOCAL_TOKENS_PER_CHUNK = 2
PRICED_MODEL = "embed-english-v3.0"


class UsageEmbedder(EmbeddingAdapter):
    """Reports whatever billed usage the test primes it with, as a priced provider."""

    provider = "cohere"

    def __init__(
        self,
        settings: Settings,
        *,
        model: str = PRICED_MODEL,
        tokens: int | None = None,
    ) -> None:
        super().__init__(settings)
        self._model = model
        self._tokens = tokens

    @property
    def model(self) -> str:
        return self._model

    @property
    def model_version(self) -> str:
        return f"{self._model}-v1"

    @property
    def dimensions(self) -> int | None:
        return DIMENSIONS

    async def embed_documents(self, texts: Sequence[str]) -> EmbeddingResult:
        return EmbeddingResult(
            vectors=[[0.0] * DIMENSIONS for _ in texts],
            model=self.model,
            model_version=self.model_version,
            total_tokens=self._tokens,
        )

    async def embed_query(self, text: str) -> list[float]:
        return [0.0] * DIMENSIONS

    async def health(self) -> HealthStatus:
        return HealthStatus(healthy=True)

    async def close(self) -> None:
        return None


class UnpricedEmbedder(UsageEmbedder):
    """A provider the price tables have never heard of."""

    provider = "mystery"


class NullSink:
    """Accepts embedded batches and keeps none of them."""

    async def write(self, batch: EmbeddedBatch) -> None:
        return None


def settings(**overrides: Any) -> Settings:
    payload: dict[str, Any] = {
        "workers": {"embedding_pool_size": 1},
        "embeddings": {"batch_size": 10},
        "reliability": {"retries": {"max_attempts": 0, "backoff_base_ms": 1, "jitter": False}},
    }
    payload.update(overrides)
    return Settings.model_validate(payload)


def payload(index: int) -> ChunkPayload:
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
            token_count=LOCAL_TOKENS_PER_CHUNK,
            strategy="recursive",
        ),
    )


async def run_pool(
    embedder: EmbeddingAdapter,
    configured: Settings,
    *,
    chunks: int = 3,
    governor: CostGovernor | None = None,
    tenant: str | None = None,
) -> Any:
    """Drive one pass of the pool over ``chunks`` payloads and return its report."""
    queue: BoundedQueue[ChunkPayload] = BoundedQueue(chunks + 2)
    for index in range(chunks):
        await queue.put(payload(index))
    await queue.close(consumers=1)

    pool = EmbeddingWorkerPool(
        configured,
        TieringRouter(embedder),
        NullSink(),
        tenant=tenant,
        governor=governor or CostGovernor(),
    )
    return await pool.run(queue)


async def test_the_reported_total_is_what_the_pass_records() -> None:
    configured = settings()
    report = await run_pool(UsageEmbedder(configured, tokens=41), configured)

    assert report.tokens_reported == 41
    assert report.batches_without_usage == 0


async def test_the_local_count_is_recorded_alongside_it() -> None:
    configured = settings()
    report = await run_pool(UsageEmbedder(configured, tokens=41), configured)

    assert report.tokens_counted == 3 * LOCAL_TOKENS_PER_CHUNK


async def test_a_disagreement_is_published_rather_than_resolved() -> None:
    """Provider and tokenizer may differ; the pass reports the gap instead of picking one."""
    configured = settings()
    report = await run_pool(UsageEmbedder(configured, tokens=41), configured)

    assert report.token_drift == 41 - 3 * LOCAL_TOKENS_PER_CHUNK


async def test_a_provider_that_reports_nothing_is_counted_as_unreported() -> None:
    configured = settings()
    report = await run_pool(UsageEmbedder(configured, tokens=None), configured)

    assert report.tokens_reported == 0
    assert report.batches_without_usage == 1
    assert report.tokens_counted == 3 * LOCAL_TOKENS_PER_CHUNK
    assert report.token_drift == 0


async def test_an_unreported_batch_does_not_pollute_the_drift() -> None:
    """Drift compares only batches that reported; otherwise it reads as a huge disagreement."""
    configured = settings(embeddings={"batch_size": 2})
    report = await run_pool(UsageEmbedder(configured, tokens=None), configured, chunks=5)

    assert report.tokens_counted_where_reported == 0
    assert report.token_drift == 0


async def test_the_governor_is_charged_the_reported_figure() -> None:
    configured = settings()
    governor = CostGovernor(per_tenant=10_000)

    await run_pool(
        UsageEmbedder(configured, tokens=41), configured, governor=governor, tenant="acme"
    )

    assert governor.spent("acme") == 41


async def test_the_governor_falls_back_to_the_local_count_without_a_receipt() -> None:
    configured = settings()
    governor = CostGovernor(per_tenant=10_000)

    await run_pool(
        UsageEmbedder(configured, tokens=None), configured, governor=governor, tenant="acme"
    )

    assert governor.spent("acme") == 3 * LOCAL_TOKENS_PER_CHUNK


async def test_ingestion_spend_is_never_refused_mid_job() -> None:
    """A cap that abandons a half-indexed corpus is worse than one that bites the next query."""
    configured = settings()
    governor = CostGovernor(per_query=1, per_tenant=1)

    report = await run_pool(
        UsageEmbedder(configured, tokens=41), configured, governor=governor, tenant="acme"
    )

    assert report.embedded == 3
    assert report.failed == 0


async def test_a_priced_model_reaches_the_cost_counter() -> None:
    from fasterrag.observability import metrics

    configured = settings()
    before = _counter(metrics, "fasterrag_cost_usd_total")

    await run_pool(UsageEmbedder(configured, tokens=1_000_000), configured)

    assert _counter(metrics, "fasterrag_cost_usd_total") > before


async def test_an_unpriced_model_is_counted_rather_than_dropped() -> None:
    from fasterrag.observability import metrics

    configured = settings()
    before = _counter(metrics, "fasterrag_unpriced_tokens_total")

    await run_pool(UnpricedEmbedder(configured, model="unknown-model", tokens=77), configured)

    assert _counter(metrics, "fasterrag_unpriced_tokens_total") == before + 77


def _counter(metrics: Any, name: str) -> float:
    """Return one counter's total across every label set."""
    return sum(float(value) for _, value in metrics.REGISTRY.series(name))
