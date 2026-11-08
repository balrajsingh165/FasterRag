"""The per-tenant budget has to survive the request that spent against it (TASK-0337).

``api/query.py`` builds a :class:`GenerationService` per request, so every test here builds
one per query too. A governor constructed inside that service is therefore a window that is
filled and thrown away within one request, which is how ``cost.per_tenant_token_budget``
became a ceiling nothing could reach — the cases below fail against that implementation and
pass against the shared one.
"""

from collections.abc import AsyncIterator, Sequence
from dataclasses import fields
from typing import Any

import pytest

from fasterrag.adapters.embeddings.base import EmbeddingAdapter, EmbeddingResult
from fasterrag.adapters.embeddings.tiering import TieringRouter
from fasterrag.adapters.llm.base import Completion, LLMAdapter
from fasterrag.adapters.vectordb.base import HealthStatus
from fasterrag.config.schema import Settings
from fasterrag.core.cache.semantic import SemanticCache
from fasterrag.core.cache.store import MemoryStore
from fasterrag.core.chunking.models import TextChunk
from fasterrag.core.retrieval.models import ScoredChunk
from fasterrag.errors import ErrorCode, FasterRagError
from fasterrag.services.budget import (
    CostGovernor,
    create_governor,
    reset_shared_governors,
    shared_governor,
)
from fasterrag.services.generation import GenerationService
from fasterrag.services.querying import FULL_MODE, Retrieval
from fasterrag.workers.embed_pool import EmbeddingWorkerPool
from fasterrag.workers.queues import BoundedQueue, ChunkPayload, EmbeddedBatch

TENANT = "acme"
PROMPT_TOKENS = 1000
COMPLETION_TOKENS = 500
SPEND_PER_QUERY = PROMPT_TOKENS + COMPLETION_TOKENS

# A budget far above anything one query can reach, so a refusal below is attributable to
# accumulated spend rather than to the query itself being oversized.
TENANT_BUDGET = 100_000
CHUNK_TOKENS = 60_000
DIMENSIONS = 3


class StubLLM(LLMAdapter):
    """Answers with fixed text and usage, and counts calls so "never reached" is checkable."""

    provider = "openai"

    def __init__(self, settings: Settings) -> None:
        super().__init__(settings)
        self.calls = 0

    async def complete(self, prompt: str, *, system: str | None = None) -> Completion:
        self.calls += 1
        return Completion(
            text="Either party may terminate [^c_a].",
            model=self.model,
            prompt_tokens=PROMPT_TOKENS,
            completion_tokens=COMPLETION_TOKENS,
            finish_reason="stop",
        )

    async def stream(self, prompt: str, *, system: str | None = None) -> AsyncIterator[str]:
        self.calls += 1
        yield "Either party may terminate [^c_a]."

    async def health(self) -> HealthStatus:
        return HealthStatus(healthy=True)

    async def close(self) -> None:
        return None


class StubRetrieval:
    """Returns one fixed chunk, so a query reaches the generator."""

    async def search(self, text: str, **kwargs: Any) -> Retrieval:
        chunk = ScoredChunk(
            chunk_id="c_a", text="Either party may terminate.", payload={}, rrf_score=0.5
        )
        return Retrieval(chunks=[chunk], mode=FULL_MODE)


class StubEmbedder(EmbeddingAdapter):
    """Embeds without a provider and reports no usage, so the counted tokens are billed."""

    provider = "recording"

    @property
    def model(self) -> str:
        return "test-model"

    @property
    def model_version(self) -> str:
        return "test-model-v1"

    @property
    def dimensions(self) -> int | None:
        return DIMENSIONS

    async def embed_documents(self, texts: Sequence[str]) -> EmbeddingResult:
        return EmbeddingResult(
            vectors=[[0.0] * DIMENSIONS for _ in texts],
            model=self.model,
            model_version=self.model_version,
        )

    async def embed_query(self, text: str) -> list[float]:
        """Return a non-zero vector, so a repeated question is a certain cache hit."""
        return [1.0, *([0.0] * (DIMENSIONS - 1))]

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


def configured(per_tenant: int = TENANT_BUDGET) -> Settings:
    return Settings.model_validate(
        {
            "cost": {"per_tenant_token_budget": per_tenant},
            "workers": {"embedding_pool_size": 1},
            "embeddings": {"batch_size": 2},
        }
    )


def a_request(settings: Settings) -> GenerationService:
    """Build the service the way a REST request does: freshly, for this request alone."""
    return GenerationService(
        settings,
        StubRetrieval(),  # type: ignore[arg-type]
        StubLLM(settings),
        tenant=TENANT,
    )


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
            token_count=CHUNK_TOKENS,
            strategy="recursive",
        ),
        metadata={},
    )


async def ingest(settings: Settings, chunks: int) -> None:
    """Run the real embedding pool, which is what charges embedding spend."""
    queue: BoundedQueue[ChunkPayload] = BoundedQueue(20)
    for index in range(chunks):
        await queue.put(payload(index))
    await queue.close(consumers=1)

    pool = EmbeddingWorkerPool(
        settings, TieringRouter(StubEmbedder(settings)), CollectingSink(), tenant=TENANT
    )
    await pool.run(queue)


async def test_the_tenant_budget_can_be_reached_through_repeated_requests() -> None:
    """The headline regression: a per-request service must not reset the rolling window."""
    settings = configured()
    served = 0
    refusal: FasterRagError | None = None

    for _ in range(200):
        try:
            await a_request(settings).answer("When may either party terminate?")
        except FasterRagError as exc:
            refusal = exc
            break
        served += 1

    assert refusal is not None, "200 requests never reached cost.per_tenant_token_budget"
    assert refusal.code is ErrorCode.BUDGET_EXCEEDED
    assert refusal.retryable is False
    assert served >= 2, "the cap bit before any request was served, so nothing accumulated"
    assert shared_governor(settings).spent(TENANT) == served * SPEND_PER_QUERY


async def test_one_request_alone_is_nowhere_near_the_budget() -> None:
    """Pins that the refusal above is accumulated spend, not an oversized single query."""
    settings = configured()

    await a_request(settings).answer("When may either party terminate?")

    assert shared_governor(settings).spent(TENANT) == SPEND_PER_QUERY
    assert SPEND_PER_QUERY < TENANT_BUDGET


async def test_embedding_spend_charged_at_ingest_is_visible_to_a_later_query() -> None:
    """The whole point of one shared window: ingest and query must count into the same one."""
    settings = configured()
    await ingest(settings, chunks=2)

    assert shared_governor(settings).spent(TENANT) == 2 * CHUNK_TOKENS

    with pytest.raises(FasterRagError) as caught:
        await a_request(settings).answer("When may either party terminate?")

    assert caught.value.code is ErrorCode.BUDGET_EXCEEDED


async def test_the_same_query_is_served_when_no_ingest_has_spent_anything() -> None:
    """The control for the case above: without the ingest, the identical query gets through."""
    settings = configured()

    await a_request(settings).answer("When may either party terminate?")


async def test_a_query_tenant_cannot_be_refused_by_another_tenants_ingest() -> None:
    """Sharing one instance must not become sharing one bucket."""
    settings = configured()
    await ingest(settings, chunks=2)

    other = GenerationService(
        settings,
        StubRetrieval(),  # type: ignore[arg-type]
        StubLLM(settings),
        tenant="globex",
    )
    await other.answer("When may either party terminate?")


async def test_the_per_query_budget_was_never_affected_by_the_defect() -> None:
    """It is stateless, so a window rebuilt per request enforced it exactly as configured."""
    settings = Settings.model_validate({"cost": {"per_query_token_budget": 10}})

    with pytest.raises(FasterRagError, match="per_query_token_budget") as caught:
        await a_request(settings).answer("When may either party terminate?")

    assert caught.value.code is ErrorCode.BUDGET_EXCEEDED
    assert create_governor(settings).per_query == shared_governor(settings).per_query


def test_the_service_takes_the_instance_the_embedding_pool_records_into() -> None:
    settings = configured()

    assert a_request(settings).governor is shared_governor(settings)
    assert a_request(settings).governor is a_request(settings).governor


def test_an_injected_governor_wins_over_the_shared_one() -> None:
    """Injection is what lets a test bound its own window without touching the singleton."""
    settings = configured()
    isolated = create_governor(settings)

    service = GenerationService(
        settings,
        StubRetrieval(),  # type: ignore[arg-type]
        StubLLM(settings),
        governor=isolated,
    )

    assert service.governor is isolated
    assert service.governor is not shared_governor(settings)


def test_configurations_differing_only_outside_the_governor_share_one_window() -> None:
    """Splitting the window over an unrelated field would grant the budget twice."""
    estimating = Settings.model_validate(
        {"cost": {"per_tenant_token_budget": TENANT_BUDGET, "estimator": True}}
    )
    not_estimating = Settings.model_validate(
        {"cost": {"per_tenant_token_budget": TENANT_BUDGET, "estimator": False}}
    )

    assert shared_governor(estimating) is shared_governor(not_estimating)


def test_different_budgets_never_share_a_window() -> None:
    assert shared_governor(configured(per_tenant=10)) is not shared_governor(
        configured(per_tenant=20)
    )


def test_the_cache_key_covers_every_configured_field() -> None:
    """A key over a subset of the inputs is the bug this guard exists to prevent.

    The key is read off the dataclass, so a configuration field added to ``CostGovernor``
    joins it automatically; this fails if one is ever added to the excluded set by mistake.
    """
    declared = {field.name for field in fields(CostGovernor)}
    state = {"_spent"}

    assert len(CostGovernor().identity) == len(declared - state)
    assert declared - state == {"per_query", "per_tenant", "window_seconds"}


def test_resetting_drops_every_shared_window() -> None:
    """What keeps the singleton from leaking spend between tests."""
    settings = configured()
    first = shared_governor(settings)
    first.record(500, tenant=TENANT)

    reset_shared_governors()

    second = shared_governor(settings)
    assert second is not first
    assert second.spent(TENANT) == 0


async def test_a_refused_query_never_reaches_the_provider() -> None:
    """The point of a preflight budget: the refusal lands while the tokens are unbought."""
    settings = Settings.model_validate({"cost": {"per_query_token_budget": 10}})
    llm = StubLLM(settings)
    service = GenerationService(settings, StubRetrieval(), llm, tenant=TENANT)  # type: ignore[arg-type]

    with pytest.raises(FasterRagError) as caught:
        await service.answer("When may either party terminate?")

    assert caught.value.status == 402
    assert llm.calls == 0


async def test_the_budget_counts_the_completion_ceiling_not_only_the_prompt() -> None:
    """A cap that ignored ``llm.max_tokens`` would be passed by every query, then exceeded."""
    generous = Settings.model_validate(
        {"cost": {"per_query_token_budget": 2000}, "llm": {"max_tokens": 64}}
    )
    strict = Settings.model_validate(
        {"cost": {"per_query_token_budget": 2000}, "llm": {"max_tokens": 8192}}
    )
    permitted = StubLLM(generous)
    refused = StubLLM(strict)

    await GenerationService(generous, StubRetrieval(), permitted).answer("terminate?")  # type: ignore[arg-type]
    with pytest.raises(FasterRagError) as caught:
        await GenerationService(strict, StubRetrieval(), refused).answer("terminate?")  # type: ignore[arg-type]

    assert permitted.calls == 1
    assert refused.calls == 0
    assert "llm.max_tokens" in caught.value.detail


async def test_the_streamed_path_is_refused_as_well() -> None:
    """Streaming must not be the way around a cap the JSON path enforces."""
    settings = Settings.model_validate({"cost": {"per_query_token_budget": 10}})
    llm = StubLLM(settings)
    service = GenerationService(settings, StubRetrieval(), llm, tenant=TENANT)  # type: ignore[arg-type]

    with pytest.raises(FasterRagError) as caught:
        [event async for event in service.stream("When may either party terminate?")]

    assert caught.value.code is ErrorCode.BUDGET_EXCEEDED
    assert llm.calls == 0


async def test_a_cache_hit_is_never_refused_because_it_buys_nothing() -> None:
    """A cap that refused free answers would punish the mechanism that saves the money."""
    cache = SemanticCache(Settings.model_validate({"cache": {"semantic": True}}), MemoryStore())
    question = "When may either party terminate?"

    generous = Settings.model_validate(
        {"cost": {"per_query_token_budget": 100_000}, "cache": {"semantic": True}}
    )
    served = StubLLM(generous)
    await GenerationService(
        generous,
        StubRetrieval(),  # type: ignore[arg-type]
        served,
        cache=cache,
        embedder=StubEmbedder(generous),
    ).answer(question)

    strict = Settings.model_validate(
        {"cost": {"per_query_token_budget": 1}, "cache": {"semantic": True}}
    )
    refused = StubLLM(strict)
    result = await GenerationService(
        strict,
        StubRetrieval(),  # type: ignore[arg-type]
        refused,
        cache=cache,
        embedder=StubEmbedder(strict),
    ).answer(question)

    assert served.calls == 1
    assert refused.calls == 0
    assert result.cache["semantic_hit"] is True
