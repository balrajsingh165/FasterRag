"""Cross-encoder reranking — the single biggest quality lever in the stack.

Retrieval scores a query and a chunk *separately* and compares the results, which is what
makes it fast enough to search millions of chunks. A cross-encoder instead reads the query
and the chunk **together**, so it can judge whether this passage actually answers this
question rather than whether they occupy similar regions of a vector space. That is why it
reorders so effectively, and why it can only run over a shortlist: scoring every chunk in a
corpus this way is computationally impossible.

Hence the shape the architecture prescribes: retrieve a wide candidate set, rerank it, then
truncate to ``retrieval.top_k`` (``docs/architecture.md`` §6). The stage costs roughly
100 to 300 ms per query — a documented trade, not a measurement of this implementation.

Failure here degrades rather than fails. A reranker that cannot load leaves the fused
ranking in place and the response is flagged ``hybrid_only``, because unranked results are
far better than no answer (D4).
"""

from __future__ import annotations

import asyncio
from collections.abc import Sequence
from dataclasses import replace
from typing import Any, Protocol

from fasterrag.config.schema import Settings
from fasterrag.core.retrieval.models import ScoredChunk
from fasterrag.errors import ConfigError, ErrorCode, RetrievalError
from fasterrag.observability.logging import get_logger

__all__ = ["CrossEncoderReranker", "Reranker", "load_cross_encoder"]

_logger = get_logger(__name__)


class Reranker(Protocol):
    """Reorders retrieved chunks by how well each answers the query."""

    async def rerank(self, query: str, chunks: Sequence[ScoredChunk]) -> list[ScoredChunk]:
        """Return the chunks reordered, each carrying its rerank score."""
        ...


def load_cross_encoder(model: str) -> Any:
    """Load a cross-encoder model.

    Imported lazily so that a deployment with reranking switched off never pays for the
    dependency.

    Raises:
        ConfigError: If the reranking extra is not installed.
    """
    try:
        from sentence_transformers import CrossEncoder
    except ImportError as exc:
        raise ConfigError(
            "retrieval.rerank is enabled, which needs a cross-encoder model; install it "
            "with 'pip install \"fasterrag[rerank]\"' or set retrieval.rerank to false"
        ) from exc

    return CrossEncoder(model)


class CrossEncoderReranker:
    """Reranks with a locally-loaded cross-encoder."""

    def __init__(self, settings: Settings) -> None:
        """Build the reranker without loading the model."""
        self.model_name = settings.retrieval.reranker_model
        self._model: Any | None = None

    def _loaded(self) -> Any:
        """Return the model, loading it once on first use."""
        if self._model is None:
            _logger.info("loading reranker model", extra={"model": self.model_name})
            self._model = load_cross_encoder(self.model_name)
        return self._model

    def score(self, query: str, chunks: Sequence[ScoredChunk]) -> list[float]:
        """Score every query and chunk pair together.

        Raises:
            RetrievalError: If the model fails to score the batch, which the caller turns
                into a degraded response rather than a failed query.
        """
        model = self._loaded()
        pairs = [(query, chunk.text) for chunk in chunks]

        try:
            scores = model.predict(pairs)
        except (RuntimeError, ValueError, OSError) as exc:
            raise RetrievalError(
                f"the reranker failed to score {len(pairs)} candidates: {type(exc).__name__}",
                code=ErrorCode.RERANK_FAILED,
                retryable=False,
            ) from exc

        return [float(score) for score in scores]

    async def rerank(self, query: str, chunks: Sequence[ScoredChunk]) -> list[ScoredChunk]:
        """Reorder chunks by cross-encoder score, best first.

        Scoring is CPU- or GPU-bound, so it runs in a worker thread and never blocks the
        event loop serving other queries.
        """
        if not chunks:
            return []

        scores = await asyncio.to_thread(self.score, query, chunks)
        scored = [
            replace(chunk, rerank_score=score) for chunk, score in zip(chunks, scores, strict=True)
        ]
        ordered = sorted(scored, key=lambda chunk: -(chunk.rerank_score or 0.0))

        return [
            replace(chunk, final_rank=position) for position, chunk in enumerate(ordered, start=1)
        ]
