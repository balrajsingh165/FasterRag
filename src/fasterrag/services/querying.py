"""Retrieval: hybrid dense and keyword legs, fused.

The highest-impact upgrade over pure vector search. Dense retrieval misses exact
identifiers and rare terms; BM25 catches those and misses paraphrases. Running both and
fusing the rankings is what covers each one's blind spot (``docs/adr/ADR-0004``).

Three properties the documentation asks for, and how they hold here:

* **The legs run in parallel**, not in sequence, so hybrid retrieval costs roughly one leg's
  latency rather than two.
* **Both legs receive the same pushed-down filter**, so they search the same subset of the
  corpus. A filter applied to only one leg would silently return an inconsistent candidate
  set.
* **Fusion happens here, with the configured ``retrieval.rrf_k``**, rather than in the
  backend, because a backend's built-in fusion does not expose that constant.

Reranking runs last, over the fused shortlist, and truncates to ``retrieval.top_k``. A
reranker that cannot load or score degrades the response to ``hybrid_only`` rather than
failing the query, because unranked results beat no answer (D4).
"""

from __future__ import annotations

import asyncio
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from fasterrag.adapters.embeddings.tiering import TieringRouter
from fasterrag.adapters.vectordb.base import (
    Filter,
    ScoredPoint,
    SearchQuery,
    VectorDBAdapter,
    validate_filter,
)
from fasterrag.adapters.vectordb.qdrant import POINT_ID_PAYLOAD_KEY
from fasterrag.config.schema import Settings
from fasterrag.core.breaker import CircuitBreaker
from fasterrag.core.rerank import Reranker
from fasterrag.core.retrieval.bm25 import encode_query
from fasterrag.core.retrieval.fusion import Ranking, rrf_fuse
from fasterrag.core.retrieval.models import DENSE_LEG, SPARSE_LEG, ScoredChunk
from fasterrag.errors import FasterRagError
from fasterrag.observability.logging import get_logger

FULL_MODE = "full"
HYBRID_ONLY_MODE = "hybrid_only"

__all__ = ["FULL_MODE", "HYBRID_ONLY_MODE", "Retrieval", "RetrievalService"]

_logger = get_logger(__name__)


@dataclass(frozen=True, slots=True)
class Retrieval:
    """Retrieved chunks and how completely the pipeline ran to produce them."""

    chunks: list[ScoredChunk] = field(default_factory=list)
    mode: str = FULL_MODE

    @property
    def degraded(self) -> bool:
        """Return whether a stage was skipped, which callers must surface to the user."""
        return self.mode != FULL_MODE


class RetrievalService:
    """Retrieves chunks for a query, hybrid when configured."""

    def __init__(
        self,
        settings: Settings,
        adapter: VectorDBAdapter,
        router: TieringRouter,
        reranker: Reranker | None = None,
        breaker: CircuitBreaker | None = None,
    ) -> None:
        """Build the service.

        Args:
            settings: Validated configuration supplying ``retrieval.*``.
            adapter: The vector database to search.
            router: Supplies the query embedding through its default adapter; queries are
                not tier-routed, because a query has no document metadata to route on and
                must be embedded by the same model that embedded the corpus.
            reranker: Reorders the shortlist. Omitted when ``retrieval.rerank`` is off, and
                a failing one degrades the response rather than failing the query.
            breaker: Circuit breaker for the vector database. Built from
                ``reliability.circuit_breaker`` when omitted; injected by tests.
        """
        self.settings = settings
        self.adapter = adapter
        self.router = router
        self.reranker = reranker
        # CRITICAL: one breaker for the life of the service, as in GenerationService and
        # EmbeddingPool. A breaker rebuilt per query counts to one forever and never opens.
        self.breaker = breaker or CircuitBreaker(
            provider="vector_db",
            failure_threshold=settings.reliability.circuit_breaker.failure_threshold,
            reset_timeout_ms=settings.reliability.circuit_breaker.reset_timeout_ms,
            enabled=settings.reliability.circuit_breaker.enabled,
        )

    async def retrieve(
        self,
        text: str,
        *,
        collection: str | None = None,
        top_k: int | None = None,
        filters: Filter | None = None,
    ) -> list[ScoredChunk]:
        """Return the chunks most relevant to ``text``.

        Args:
            text: The query.
            collection: Collection to search; defaults to the configured one.
            top_k: Final result count; defaults to ``retrieval.top_k``.
            filters: Metadata filter, pushed down to every leg.

        Returns:
            Chunks ordered best first, each carrying the rank and score every leg gave it.
        """
        return (await self.search(text, collection=collection, top_k=top_k, filters=filters)).chunks

    async def search(
        self,
        text: str,
        *,
        collection: str | None = None,
        top_k: int | None = None,
        filters: Filter | None = None,
    ) -> Retrieval:
        """Retrieve, and report which stages actually ran.

        The detailed form. Generation needs the mode so a degraded answer can be labelled
        as one rather than passed off as a full-quality result (D4).
        """
        validate_filter(filters)
        retrieval = self.settings.retrieval
        target = collection or self.settings.vector_db.collection.default_name
        limit = top_k or retrieval.top_k
        candidates = max(retrieval.rerank_top_n, limit) if self._reranking else limit

        legs = await self._run_legs(text, target, candidates, filters)
        fused = self._fuse(legs)
        shortlist = self._assemble(fused, legs, candidates)

        results, mode = await self._rerank(text, shortlist, limit)

        _logger.info(
            "retrieved",
            extra={
                "collection": target,
                "hybrid": retrieval.hybrid,
                "mode": mode,
                "candidates": {name: len(points) for name, points in legs.items()},
                "returned": len(results),
            },
        )
        return Retrieval(chunks=results, mode=mode)

    @property
    def _reranking(self) -> bool:
        """Return whether a reranker will run, which decides how wide to retrieve."""
        return self.settings.retrieval.rerank and self.reranker is not None

    async def _rerank(
        self, text: str, shortlist: list[ScoredChunk], limit: int
    ) -> tuple[list[ScoredChunk], str]:
        """Rerank the shortlist, degrading to the fused order if the reranker fails."""
        reranker = self.reranker
        if reranker is None or not self.settings.retrieval.rerank:
            return shortlist[:limit], FULL_MODE

        try:
            reordered = await reranker.rerank(text, shortlist)
        except FasterRagError as exc:
            _logger.warning(
                "reranking failed, serving fused results instead",
                extra={"code": exc.code.value, "trace_id": exc.trace_id, "mode": HYBRID_ONLY_MODE},
            )
            return shortlist[:limit], HYBRID_ONLY_MODE

        return reordered[:limit], FULL_MODE

    async def _run_legs(
        self,
        text: str,
        collection: str,
        candidates: int,
        filters: Filter | None,
    ) -> dict[str, list[ScoredPoint]]:
        """Run the legs behind the vector-database breaker.

        Retrieval is the one outbound call on the query path with nowhere to degrade to: a
        failure here means there is nothing to answer *from*, so it surfaces rather than
        dropping a rung the way a failed reranker or a failed generation does. That is
        exactly why the breaker matters here — without it a backend that is down is retried
        on every request, and the retries are the load keeping it down.

        ``CircuitOpenError`` carries ``CIRCUIT_OPEN``, which the problem table already maps
        to a retryable 503, so shedding is reported as the temporary condition it is.
        """
        # CRITICAL: the query is embedded *outside* the guarded region. It is an embedding
        # provider call sitting on the retrieval path, and counting its failures here would
        # open the vector-database breaker for an outage the vector database had no part in
        # — shedding reads against a backend that is perfectly healthy, and hiding the real
        # provider behind the wrong gauge.
        vector = await self.router.default.embed_query(text)

        self.breaker.check()
        try:
            legs = await self._execute_legs(vector, text, collection, candidates, filters)
        except FasterRagError as exc:
            self.breaker.record_failure(exc)
            raise

        self.breaker.record_success()
        return legs

    async def _execute_legs(
        self,
        vector: Sequence[float],
        text: str,
        collection: str,
        candidates: int,
        filters: Filter | None,
    ) -> dict[str, list[ScoredPoint]]:
        """Run every configured leg concurrently and return their results by name."""
        dense = self.adapter.search(
            SearchQuery(
                collection=collection,
                vector=vector,
                limit=candidates,
                filters=filters,
            )
        )

        if not self.settings.retrieval.hybrid:
            return {DENSE_LEG: await dense}

        sparse_query = encode_query(text)
        if sparse_query.empty:
            _logger.info("query has no indexable terms, so the keyword leg is skipped")
            return {DENSE_LEG: await dense}

        sparse = self.adapter.search(
            SearchQuery(
                collection=collection,
                sparse=sparse_query,
                limit=candidates,
                filters=filters,
            )
        )
        dense_hits, sparse_hits = await asyncio.gather(dense, sparse)
        return {DENSE_LEG: dense_hits, SPARSE_LEG: sparse_hits}

    def _fuse(self, legs: Mapping[str, list[ScoredPoint]]) -> list[Any]:
        """Fuse the legs' rankings with the configured constant and weights."""
        retrieval = self.settings.retrieval
        weights = {DENSE_LEG: retrieval.dense_weight, SPARSE_LEG: retrieval.bm25_weight}

        rankings = [
            Ranking(
                name=name,
                ids=[point.point_id for point in points],
                weight=weights.get(name, 1.0),
            )
            for name, points in legs.items()
        ]
        return rrf_fuse(*rankings, k=retrieval.rrf_k)

    @staticmethod
    def _assemble(
        fused: list[Any],
        legs: Mapping[str, list[ScoredPoint]],
        limit: int,
    ) -> list[ScoredChunk]:
        """Build results carrying each leg's rank and score, truncated to ``limit``."""
        scores = {
            name: {point.point_id: point.score for point in points} for name, points in legs.items()
        }
        payloads: dict[str, Mapping[str, Any]] = {}
        for points in legs.values():
            for point in points:
                payloads.setdefault(point.point_id, point.payload)

        results: list[ScoredChunk] = []
        for position, entry in enumerate(fused[:limit], start=1):
            payload = dict(payloads.get(entry.id, {}))
            payload.pop(POINT_ID_PAYLOAD_KEY, None)
            text = payload.get("text", "")

            results.append(
                ScoredChunk(
                    chunk_id=entry.id,
                    text=text if isinstance(text, str) else "",
                    payload=payload,
                    dense_rank=entry.rank_in(DENSE_LEG),
                    dense_score=scores.get(DENSE_LEG, {}).get(entry.id),
                    bm25_rank=entry.rank_in(SPARSE_LEG),
                    bm25_score=scores.get(SPARSE_LEG, {}).get(entry.id),
                    rrf_score=entry.score,
                    final_rank=position,
                )
            )
        return results
