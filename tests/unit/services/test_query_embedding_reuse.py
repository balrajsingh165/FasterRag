"""A cache-enabled query embeds its question once (TASK-0515)."""

from typing import Any

from fasterrag.adapters.embeddings.tiering import TieringRouter
from fasterrag.config.schema import Settings
from fasterrag.core.cache.semantic import SemanticCache
from fasterrag.core.cache.store import MemoryStore
from fasterrag.services.generation import GenerationService
from fasterrag.services.querying import RetrievalService
from tests.unit.services.test_generation import FAST_RETRIES, ScriptedEmbedder, ScriptedLLM
from tests.unit.services.test_querying import ScriptedAdapter, hit


def build(
    *, semantic: bool = True, embedder: ScriptedEmbedder | None = None
) -> tuple[GenerationService, ScriptedEmbedder, ScriptedEmbedder]:
    """Return a service over real retrieval, its cache embedder and its retrieval embedder."""
    settings = Settings.model_validate(
        {
            "cache": {"semantic": semantic},
            "reliability": FAST_RETRIES,
        }
    )
    retrieval_embedder = ScriptedEmbedder(settings)
    adapter = ScriptedAdapter(settings)
    adapter.dense = [hit("c_a", 0.9, "Either party may terminate.")]
    router = TieringRouter(retrieval_embedder)
    retrieval = RetrievalService(settings, adapter, router)
    cache_embedder = embedder or router.default
    assert isinstance(cache_embedder, ScriptedEmbedder)
    service = GenerationService(
        settings,
        retrieval,
        ScriptedLLM(settings, text="thirty days [^c_a]"),
        cache=SemanticCache(settings, MemoryStore()),
        embedder=cache_embedder,
    )
    return service, cache_embedder, retrieval_embedder


async def test_a_cache_enabled_query_embeds_the_question_once() -> None:
    """The semantic cache and retrieval each embedded the question, so a miss paid twice."""
    service, embedder, _ = build()

    result = await service.answer("what is the notice period?")

    assert result.cache["semantic_hit"] is False
    assert embedder.calls == 1


async def test_a_streamed_cache_enabled_query_embeds_the_question_once() -> None:
    """The streamed path shares the pipeline, so it must reuse the vector too."""
    service, embedder, _ = build()

    events: list[Any] = [event async for event in service.stream("what is the notice period?")]

    assert events[-1].type == "done"
    assert embedder.calls == 1


async def test_a_query_without_the_semantic_cache_still_embeds_once() -> None:
    """Nothing to reuse: retrieval embeds the question itself."""
    service, embedder, _ = build(semantic=False)

    await service.answer("what is the notice period?")

    assert embedder.calls == 1


async def test_a_cache_embedder_retrieval_does_not_use_is_not_reused() -> None:
    """A vector from another model would search the index in the wrong space."""
    other = ScriptedEmbedder(Settings.model_validate({}))
    service, cache_embedder, retrieval_embedder = build(embedder=other)

    await service.answer("what is the notice period?")

    assert cache_embedder.calls == 1
    assert retrieval_embedder.calls == 1
