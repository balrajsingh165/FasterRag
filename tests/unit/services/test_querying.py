from typing import Any

import pytest

from fasterrag.adapters.embeddings.base import EmbeddingAdapter, EmbeddingResult
from fasterrag.adapters.embeddings.tiering import TieringRouter
from fasterrag.adapters.vectordb.base import (
    CollectionInfo,
    CollectionSpec,
    HealthStatus,
    Point,
    PointSelector,
    PointUpdate,
    ScoredPoint,
    SearchQuery,
    UpsertResult,
    VectorDBAdapter,
)
from fasterrag.config.schema import Settings
from fasterrag.core.retrieval.models import ScoredChunk
from fasterrag.errors import ErrorCode, FasterRagError, RetrievalError
from fasterrag.services.querying import FULL_MODE, HYBRID_ONLY_MODE, RetrievalService


class StubEmbedder(EmbeddingAdapter):
    """Returns a fixed query vector."""

    provider = "stub"

    @property
    def model(self) -> str:
        return "stub"

    @property
    def model_version(self) -> str:
        return "stub-v1"

    @property
    def dimensions(self) -> int:
        return 3

    async def embed_documents(self, texts: Any) -> EmbeddingResult:
        return EmbeddingResult(
            vectors=[[0.1, 0.2, 0.3] for _ in texts], model="stub", model_version="stub-v1"
        )

    async def embed_query(self, text: str) -> list[float]:
        return [0.1, 0.2, 0.3]

    async def health(self) -> HealthStatus:
        return HealthStatus(healthy=True)

    async def close(self) -> None:
        return None


class ScriptedAdapter(VectorDBAdapter):
    """Returns a scripted result per leg and records the queries it received."""

    def __init__(self, settings: Settings) -> None:
        super().__init__(settings)
        self.dense: list[ScoredPoint] = []
        self.sparse: list[ScoredPoint] = []
        self.queries: list[SearchQuery] = []

    async def create_collection(self, spec: CollectionSpec) -> None:
        return None

    async def list_collections(self) -> list[CollectionInfo]:
        return []

    async def drop_collection(self, name: str) -> bool:
        return False

    async def upsert(self, points: list[Point]) -> UpsertResult:
        return UpsertResult(upserted=len(points))

    async def search(self, query: SearchQuery) -> list[ScoredPoint]:
        self.queries.append(query)
        return list(self.sparse if query.sparse is not None else self.dense)

    async def update(self, updates: list[PointUpdate]) -> None:
        return None

    async def delete(self, selector: PointSelector) -> None:
        return None

    async def health(self) -> HealthStatus:
        return HealthStatus(healthy=True)

    async def close(self) -> None:
        return None


def hit(point_id: str, score: float, text: str = "body") -> ScoredPoint:
    return ScoredPoint(
        point_id=point_id,
        score=score,
        payload={"text": text, "document_id": "d_1", "source_uri": "a.md", "point_id": point_id},
    )


def settings(**overrides: Any) -> Settings:
    return Settings.model_validate(overrides)


def build(configured: Settings | None = None) -> tuple[RetrievalService, ScriptedAdapter]:
    resolved = configured or settings()
    adapter = ScriptedAdapter(resolved)
    service = RetrievalService(resolved, adapter, TieringRouter(StubEmbedder(resolved)))
    return service, adapter


async def test_both_legs_are_searched_when_hybrid() -> None:
    service, adapter = build()
    adapter.dense = [hit("c_a", 0.9)]
    adapter.sparse = [hit("c_b", 4.2)]

    await service.retrieve("termination notice")

    assert len(adapter.queries) == 2
    assert any(query.vector is not None for query in adapter.queries)
    assert any(query.sparse is not None for query in adapter.queries)


async def test_only_the_dense_leg_runs_when_hybrid_is_off() -> None:
    service, adapter = build(settings(retrieval={"hybrid": False}))
    adapter.dense = [hit("c_a", 0.9)]

    await service.retrieve("termination")

    assert len(adapter.queries) == 1
    assert adapter.queries[0].sparse is None


async def test_a_query_with_no_indexable_terms_skips_the_keyword_leg() -> None:
    service, adapter = build()
    adapter.dense = [hit("c_a", 0.9)]

    results = await service.retrieve("the and of")

    assert len(adapter.queries) == 1
    assert results[0].bm25_rank is None


async def test_the_filter_is_pushed_down_to_every_leg() -> None:
    service, adapter = build()
    adapter.dense = [hit("c_a", 0.9)]
    adapter.sparse = [hit("c_a", 4.2)]

    await service.retrieve("notice", filters={"department": "legal"})

    assert len(adapter.queries) == 2
    assert all(query.filters == {"department": "legal"} for query in adapter.queries)


async def test_an_unsupported_filter_is_refused_before_any_search() -> None:
    service, adapter = build()

    with pytest.raises(FasterRagError, match="unsupported operators"):
        await service.retrieve("notice", filters={"year": {"$regex": ".*"}})

    assert adapter.queries == []


async def test_results_carry_the_rank_and_score_of_each_leg() -> None:
    service, adapter = build()
    adapter.dense = [hit("c_a", 0.91), hit("c_b", 0.80)]
    adapter.sparse = [hit("c_b", 5.1), hit("c_a", 3.0)]

    results = await service.retrieve("notice")
    by_id = {result.chunk_id: result for result in results}

    assert by_id["c_a"].dense_rank == 1
    assert by_id["c_a"].dense_score == pytest.approx(0.91)
    assert by_id["c_a"].bm25_rank == 2
    assert by_id["c_a"].bm25_score == pytest.approx(3.0)
    assert by_id["c_a"].found_by_both_legs is True


async def test_a_chunk_found_by_one_leg_only_reports_that() -> None:
    service, adapter = build()
    adapter.dense = [hit("c_a", 0.9)]
    adapter.sparse = [hit("c_b", 5.0)]

    results = await service.retrieve("notice")
    by_id = {result.chunk_id: result for result in results}

    assert by_id["c_a"].bm25_rank is None
    assert by_id["c_b"].dense_rank is None
    assert by_id["c_a"].found_by_both_legs is False


async def test_agreement_between_legs_wins_the_top_position() -> None:
    service, adapter = build()
    adapter.dense = [hit("c_dense_only", 0.99), hit("c_agreed", 0.5)]
    adapter.sparse = [hit("c_sparse_only", 9.9), hit("c_agreed", 1.0)]

    results = await service.retrieve("notice")

    assert results[0].chunk_id == "c_agreed"
    assert results[0].final_rank == 1


async def test_results_are_truncated_to_top_k() -> None:
    service, adapter = build(settings(retrieval={"top_k": 2}))
    adapter.dense = [hit(f"c_{index}", 1.0 - index / 10) for index in range(5)]
    adapter.sparse = []

    results = await service.retrieve("notice")

    assert len(results) == 2
    assert [result.final_rank for result in results] == [1, 2]


async def test_an_explicit_top_k_overrides_the_configured_one() -> None:
    service, adapter = build(settings(retrieval={"top_k": 10}))
    adapter.dense = [hit(f"c_{index}", 1.0) for index in range(5)]
    adapter.sparse = []

    assert len(await service.retrieve("notice", top_k=3)) == 3


async def test_candidates_widen_to_the_rerank_depth_when_reranking() -> None:
    configured = settings(retrieval={"top_k": 5, "rerank_top_n": 50})
    adapter = ScriptedAdapter(configured)
    adapter.dense = [hit("c_a", 0.9)]
    adapter.sparse = [hit("c_a", 1.0)]
    service = RetrievalService(
        configured, adapter, TieringRouter(StubEmbedder(configured)), ScriptedReranker()
    )

    await service.retrieve("notice")

    assert all(query.limit == 50 for query in adapter.queries)


async def test_no_reranker_means_no_wasted_candidates() -> None:
    service, adapter = build(settings(retrieval={"top_k": 5, "rerank_top_n": 50}))
    adapter.dense = [hit("c_a", 0.9)]
    adapter.sparse = [hit("c_a", 1.0)]

    await service.retrieve("notice")

    assert all(query.limit == 5 for query in adapter.queries)


async def test_the_reserved_payload_key_is_not_leaked_to_callers() -> None:
    service, adapter = build()
    adapter.dense = [hit("c_a", 0.9)]
    adapter.sparse = []

    results = await service.retrieve("notice")

    assert "point_id" not in results[0].payload
    assert results[0].payload["document_id"] == "d_1"


async def test_the_chunk_text_and_source_are_available_for_citation() -> None:
    service, adapter = build()
    adapter.dense = [hit("c_a", 0.9, text="Either party may terminate.")]
    adapter.sparse = []

    results = await service.retrieve("notice")

    assert results[0].text == "Either party may terminate."
    assert results[0].source == "a.md"
    assert results[0].document_id == "d_1"


async def test_an_empty_index_returns_nothing() -> None:
    service, _ = build()

    assert await service.retrieve("notice") == []


async def test_the_configured_collection_is_searched() -> None:
    configured = settings(vector_db={"collection": {"default_name": "contracts"}})
    service, adapter = build(configured)
    adapter.dense = [hit("c_a", 0.9)]

    await service.retrieve("notice")

    assert all(query.collection == "contracts" for query in adapter.queries)


async def test_weights_change_which_leg_wins() -> None:
    dense_heavy = settings(retrieval={"dense_weight": 1.0, "bm25_weight": 0.0})
    sparse_heavy = settings(retrieval={"dense_weight": 0.0, "bm25_weight": 1.0})

    for configured, expected in ((dense_heavy, "c_dense"), (sparse_heavy, "c_sparse")):
        service, adapter = build(configured)
        adapter.dense = [hit("c_dense", 0.9)]
        adapter.sparse = [hit("c_sparse", 5.0)]

        results = await service.retrieve("notice")

        assert results[0].chunk_id == expected


class ScriptedReranker:
    """Reverses the shortlist, or fails on demand."""

    def __init__(self, error: Exception | None = None) -> None:
        self.error = error
        self.calls: list[tuple[str, int]] = []

    async def rerank(self, query: str, chunks: Any) -> list[ScoredChunk]:
        self.calls.append((query, len(chunks)))
        if self.error is not None:
            raise self.error
        from dataclasses import replace

        return [
            replace(chunk, rerank_score=float(position), final_rank=position)
            for position, chunk in enumerate(reversed(list(chunks)), start=1)
        ]


def build_with_reranker(
    reranker: ScriptedReranker, configured: Settings | None = None
) -> tuple[RetrievalService, ScriptedAdapter]:
    resolved = configured or settings()
    adapter = ScriptedAdapter(resolved)
    service = RetrievalService(resolved, adapter, TieringRouter(StubEmbedder(resolved)), reranker)
    return service, adapter


async def test_the_reranker_reorders_the_shortlist() -> None:
    reranker = ScriptedReranker()
    service, adapter = build_with_reranker(reranker)
    adapter.dense = [hit("c_a", 0.9), hit("c_b", 0.8), hit("c_c", 0.7)]
    adapter.sparse = []

    results = await service.retrieve("notice")

    assert [result.chunk_id for result in results] == ["c_c", "c_b", "c_a"]
    assert results[0].rerank_score is not None
    assert [result.final_rank for result in results] == [1, 2, 3]


async def test_the_reranker_sees_the_whole_shortlist_not_just_top_k() -> None:
    reranker = ScriptedReranker()
    configured = settings(retrieval={"top_k": 2, "rerank_top_n": 20})
    service, adapter = build_with_reranker(reranker, configured)
    adapter.dense = [hit(f"c_{index}", 1.0 - index / 10) for index in range(6)]
    adapter.sparse = []

    results = await service.retrieve("notice")

    assert reranker.calls[0][1] == 6
    assert len(results) == 2


async def test_a_failing_reranker_degrades_instead_of_failing_the_query() -> None:
    reranker = ScriptedReranker(
        RetrievalError("model would not load", code=ErrorCode.RERANK_FAILED)
    )
    service, adapter = build_with_reranker(reranker)
    adapter.dense = [hit("c_a", 0.9), hit("c_b", 0.8)]
    adapter.sparse = []

    result = await service.search("notice")

    assert result.mode == HYBRID_ONLY_MODE
    assert result.degraded is True
    assert [chunk.chunk_id for chunk in result.chunks] == ["c_a", "c_b"]


async def test_a_successful_rerank_reports_full_mode() -> None:
    service, adapter = build_with_reranker(ScriptedReranker())
    adapter.dense = [hit("c_a", 0.9)]
    adapter.sparse = []

    result = await service.search("notice")

    assert result.mode == FULL_MODE
    assert result.degraded is False


async def test_reranking_is_skipped_when_the_config_disables_it() -> None:
    reranker = ScriptedReranker()
    configured = settings(retrieval={"rerank": False})
    service, adapter = build_with_reranker(reranker, configured)
    adapter.dense = [hit("c_a", 0.9), hit("c_b", 0.8)]
    adapter.sparse = []

    result = await service.search("notice")

    assert reranker.calls == []
    assert result.mode == FULL_MODE
    assert [chunk.chunk_id for chunk in result.chunks] == ["c_a", "c_b"]
