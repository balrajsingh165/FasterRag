"""Stage timings localise a slow request (TASK-0487).

A query's trace must show every stage it ran, each with a duration, and the stages must add
up to the whole: a span that starts where another stage started, or a stage nobody timed,
leaves a slow request with nowhere to point.
"""

import asyncio
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pytest

from fasterrag.adapters.embeddings.base import EmbeddingResult
from fasterrag.adapters.embeddings.tiering import TieringRouter
from fasterrag.adapters.vectordb.base import Point, ScoredPoint, SearchQuery, UpsertResult
from fasterrag.config.schema import Settings
from fasterrag.core.cache.semantic import SemanticCache
from fasterrag.core.cache.store import MemoryStore
from fasterrag.core.retrieval.models import ScoredChunk
from fasterrag.core.tracing import Span, Trace
from fasterrag.observability import metrics
from fasterrag.observability.logging import use_trace_id
from fasterrag.services.generation import GenerationService
from fasterrag.services.querying import HYBRID_ONLY_MODE, RetrievalService
from fasterrag.services.traces import TraceStore
from tests.unit.services.test_generation import FAST_RETRIES, ScriptedEmbedder, ScriptedLLM
from tests.unit.services.test_ingestion import KeywordLegAdapter, StubEmbedder, corpus
from tests.unit.services.test_ingestion import build as build_ingestion
from tests.unit.services.test_ingestion import settings as ingest_settings
from tests.unit.services.test_ingestion import sources as ingest_sources
from tests.unit.services.test_querying import ScriptedAdapter, ScriptedReranker, hit

PAUSE = 0.03
"""How long each slowed stage takes, in seconds: long enough to dwarf scheduling noise."""

__all__ = ["corpus"]

TOLERANCE_MS = 20.0
"""How far the stages may fall short of the whole, for the code between two awaits."""


class SlowEmbedder(ScriptedEmbedder):
    """Takes ``PAUSE`` to embed a question."""

    async def embed_query(self, text: str) -> list[float]:
        await asyncio.sleep(PAUSE)
        return await super().embed_query(text)


class SlowAdapter(ScriptedAdapter):
    """Takes ``PAUSE`` per search leg."""

    async def search(self, query: SearchQuery) -> list[ScoredPoint]:
        await asyncio.sleep(PAUSE)
        return await super().search(query)


class SlowReranker(ScriptedReranker):
    """Takes ``pause`` to rerank, then reverses the shortlist."""

    def __init__(self, pause: float = PAUSE) -> None:
        super().__init__()
        self.pause = pause

    async def rerank(self, query: str, chunks: Any) -> list[ScoredChunk]:
        await asyncio.sleep(self.pause)
        return await super().rerank(query, chunks)


def build(
    tmp_path: Path,
    *,
    semantic: bool = False,
    rerank_pause: float = PAUSE,
    rerank_ms: int = 30_000,
) -> tuple[GenerationService, TraceStore]:
    """Return a service over slowed real retrieval that stores its traces under ``tmp_path``."""
    settings = Settings.model_validate(
        {
            "retrieval": {"rerank": True},
            "cache": {"semantic": semantic},
            "reliability": {**FAST_RETRIES, "timeouts": {"rerank_ms": rerank_ms}},
        }
    )
    adapter = SlowAdapter(settings)
    adapter.dense = [hit("c_a", 0.9, "Either party may terminate."), hit("c_b", 0.8)]
    adapter.sparse = [hit("c_b", 4.2)]
    router = TieringRouter(SlowEmbedder(settings))
    retrieval_service = RetrievalService(settings, adapter, router, SlowReranker(rerank_pause))
    store = TraceStore(tmp_path)
    service = GenerationService(
        settings,
        retrieval_service,
        ScriptedLLM(settings, text="thirty days [^c_a]"),
        cache=SemanticCache(settings, MemoryStore()) if semantic else None,
        embedder=router.default,
        traces=store,
    )
    return service, store


async def answered(service: GenerationService, store: TraceStore, *, stream: bool) -> Trace:
    """Answer one question and return the trace it stored."""
    with use_trace_id() as trace_id:
        if stream:
            async for _ in service.stream("what is the notice period?"):
                pass
        else:
            await service.answer("what is the notice period?")
    trace = store.load(trace_id)
    assert trace is not None
    return trace


def by_name(trace: Trace) -> dict[str, Span]:
    """Return the trace's spans keyed by name, refusing a name recorded twice."""
    names = [span.name for span in trace.spans]
    assert len(names) == len(set(names)), names
    return {span.name: span for span in trace.spans}


def covered_ms(spans: list[Span]) -> float:
    """Return the milliseconds the union of ``spans`` covers, counting an overlap once."""
    total = 0.0
    reach = float("-inf")
    for span in sorted(spans, key=lambda span: span.start_ms):
        start = max(span.start_ms, reach)
        if span.end_ms > start:
            total += span.end_ms - start
        reach = max(reach, span.end_ms)
    return total


@pytest.mark.parametrize("stream", [False, True])
async def test_a_query_trace_shows_every_stage_and_they_sum_to_the_whole(
    tmp_path: Path, stream: bool
) -> None:
    """The trace had no embed, search or fuse spans, and no generation span when streamed."""
    service, store = build(tmp_path)

    spans = by_name(await answered(service, store, stream=stream))

    legs = ["query-embedding", "dense-search", "keyword-search", "fusion", "reranker"]
    top = ["retrieval", "context-assembly", "generation"]
    assert set(legs + top) <= set(spans)
    for name in ("query-embedding", "dense-search", "keyword-search", "reranker"):
        assert spans[name].duration_ms >= PAUSE * 1000 * 0.9, name

    retrieval = spans["retrieval"]
    inside = [spans[name] for name in legs]
    for span in inside:
        assert retrieval.start_ms <= span.start_ms <= span.end_ms <= retrieval.end_ms, span.name
    assert covered_ms(inside) == pytest.approx(retrieval.duration_ms, abs=TOLERANCE_MS)

    whole = spans["generation"].end_ms - retrieval.start_ms
    assert sum(spans[name].duration_ms for name in top) == pytest.approx(whole, abs=TOLERANCE_MS)


async def test_the_reranker_span_starts_when_reranking_starts(tmp_path: Path) -> None:
    """It started at the retrieval start, so it reported retrieval's duration as reranking's."""
    service, store = build(tmp_path)

    spans = by_name(await answered(service, store, stream=False))

    assert spans["reranker"].start_ms >= spans["fusion"].end_ms
    assert spans["reranker"].start_ms > spans["retrieval"].start_ms + PAUSE * 1000


async def test_a_cached_query_s_embedding_and_lookup_are_stages_of_its_trace(
    tmp_path: Path,
) -> None:
    """With the semantic cache on, the question is embedded before retrieval, once."""
    service, store = build(tmp_path, semantic=True)

    spans = by_name(await answered(service, store, stream=False))

    assert spans["query-embedding"].end_ms <= spans["semantic-cache"].start_ms
    assert spans["semantic-cache"].end_ms <= spans["retrieval"].start_ms
    assert spans["query-embedding"].duration_ms >= PAUSE * 1000 * 0.9


async def test_reranking_is_its_own_stage_metric(tmp_path: Path) -> None:
    """FMEA row 19's alarm reads `stage=rerank`, which was never written."""
    service, store = build(tmp_path)
    before = {
        stage: metrics.STAGE_DURATION.count(stage=stage)
        for stage in ("embed_query", "retrieve_dense", "retrieve_bm25", "fuse", "rerank")
    }

    await answered(service, store, stream=False)

    for stage, count in before.items():
        assert metrics.STAGE_DURATION.count(stage=stage) == count + 1, stage


async def test_a_reranker_that_times_out_is_still_timed(tmp_path: Path) -> None:
    """The slow rerank is the one the alarm exists for, so a timeout must reach the metric."""
    service, store = build(tmp_path, rerank_pause=5.0, rerank_ms=100)
    before = metrics.STAGE_DURATION.total(stage="rerank")

    trace = await answered(service, store, stream=False)

    spans = by_name(trace)
    assert trace.result["mode"] == HYBRID_ONLY_MODE
    assert spans["reranker"].attributes["code"] == "RERANK_FAILED"
    assert spans["reranker"].duration_ms >= 95
    assert metrics.STAGE_DURATION.total(stage="rerank") - before >= 0.095


class SlowDocumentEmbedder(StubEmbedder):
    """Takes ``PAUSE`` per embedding batch."""

    async def embed_documents(self, texts: Sequence[str]) -> EmbeddingResult:
        await asyncio.sleep(PAUSE)
        return await super().embed_documents(texts)


class SlowIndex(KeywordLegAdapter):
    """Takes ``PAUSE`` per upsert."""

    async def upsert(self, points: list[Point]) -> UpsertResult:
        await asyncio.sleep(PAUSE)
        return await super().upsert(points)


async def test_an_ingest_trace_shows_every_stage_and_they_sum_to_the_whole(
    tmp_path: Path, corpus: Path
) -> None:
    """Ingestion exported no spans at all, so a slow ingest could not be localised."""
    configured = ingest_settings()
    store = TraceStore(tmp_path / "traces")
    service, _ = build_ingestion(
        tmp_path,
        configured,
        embedder=SlowDocumentEmbedder(configured),
        adapter=SlowIndex(configured),
        traces=store,
    )
    before = {
        stage: metrics.STAGE_DURATION.count(stage=stage)
        for stage in ("parse", "chunk", "embed", "upsert")
    }

    record = await service.ingest(ingest_sources(corpus))

    stored = store.recent()
    assert len(stored) == 1
    trace = store.load(stored[0])
    assert trace is not None
    assert trace.kind == "ingest"
    assert trace.result["job_id"] == record.job_id
    spans = by_name(trace)

    top = ["resolve", "setup", "pipeline", "settle"]
    stages = ["parse", "chunk", "embed", "upsert"]
    assert set(top + stages) <= set(spans)
    whole = spans["settle"].end_ms - spans["resolve"].start_ms
    assert sum(spans[name].duration_ms for name in top) == pytest.approx(whole, abs=TOLERANCE_MS)

    pipeline = spans["pipeline"]
    for name in stages:
        span = spans[name]
        assert pipeline.start_ms <= span.start_ms <= span.end_ms <= pipeline.end_ms, name
        assert span.attributes["count"] >= 1, name
        assert 0 <= span.attributes["busy_ms"] <= span.attributes["count"] * pipeline.duration_ms
    assert spans["embed"].attributes["busy_ms"] >= PAUSE * 1000 * 0.9
    assert spans["upsert"].attributes["busy_ms"] >= PAUSE * 1000 * 0.9
    for stage, count in before.items():
        assert metrics.STAGE_DURATION.count(stage=stage) > count, stage
