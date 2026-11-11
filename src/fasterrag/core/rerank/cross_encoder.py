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
import threading
import time
from collections.abc import Sequence
from dataclasses import replace
from typing import Any, Final, Protocol

from fasterrag.config.schema import Settings
from fasterrag.core.retrieval.models import ScoredChunk
from fasterrag.errors import ConfigError, ErrorCode, RetrievalError
from fasterrag.observability.logging import get_logger

__all__ = [
    "CrossEncoderReranker",
    "Reranker",
    "load_cross_encoder",
    "reset_shared_cross_encoders",
]

_logger = get_logger(__name__)


class Reranker(Protocol):
    """Reorders retrieved chunks by how well each answers the query."""

    async def rerank(self, query: str, chunks: Sequence[ScoredChunk]) -> list[ScoredChunk]:
        """Return the chunks reordered, each carrying its rerank score."""
        ...


def load_cross_encoder(model: str) -> Any:
    """Load a cross-encoder model."""
    try:
        from sentence_transformers import CrossEncoder
    except ImportError as exc:
        raise ConfigError(
            "retrieval.rerank is enabled, which needs a cross-encoder model; install it "
            "with 'pip install \"fasterrag[rerank]\"' or set retrieval.rerank to false"
        ) from exc

    return CrossEncoder(model)


_MODELS: Final[dict[str, Any]] = {}
_LOADING: Final = threading.Lock()


def reset_shared_cross_encoders() -> None:
    """Drop every loaded model, so a test or a reload starts from an empty cache."""
    with _LOADING:
        _MODELS.clear()


class CrossEncoderReranker:
    """Reranks with a locally-loaded cross-encoder."""

    def __init__(self, settings: Settings) -> None:
        """Build the reranker without loading the model."""
        self.model_name = settings.retrieval.reranker_model
        self._model: Any | None = None

    def _loaded(self) -> Any:
        """Return the model, loading it once per process on first use."""
        if self._model is None:
            with _LOADING:
                self._model = _MODELS.get(self.model_name)
                if self._model is None:
                    self._model = self._load()
                    _MODELS[self.model_name] = self._model
        return self._model

    def _load(self) -> Any:
        """Load the model from disk or the hub, reporting how long it took."""
        if self._model is None:
            # CRITICAL: warning, not info. Nothing configures logging in a plain script or a
            # notebook, so an info line is invisible and the default reranker —
            # BAAI/bge-reranker-v2-m3, 2.2 GB — downloads and loads in total silence.
            # BENCH-0003 measured that first run on a developer laptop as a 357 s download
            # followed by an 8.3-9.3 s load from the populated cache: minutes of silence,
            # which reads as a hang rather than as work. The earlier "over 400 s to load"
            # figure conflated the two. Python's last-resort handler emits warnings without
            # setup, so this is the only level that reaches the person waiting.
            _logger.warning(
                "loading the reranker model; the first run downloads the weights, which for "
                "a large cross-encoder is several gigabytes and can take minutes. Set "
                "retrieval.rerank to false, or choose a smaller retrieval.reranker_model, "
                "to skip this stage",
                extra={"model": self.model_name},
            )
            started = time.perf_counter()
            try:
                self._model = load_cross_encoder(self.model_name)
            except ConfigError:
                raise
            # CRITICAL: deliberately broad. A model load traverses huggingface_hub, safetensors,
            # transformers, and torch, each raising its own exception types, and any of them
            # leaking untyped defeats the degradation ladder — which is the failure this
            # rung exists to absorb. Nothing is swallowed: the cause is chained and the
            # result is a classified error the caller degrades on.
            except Exception as exc:
                raise RetrievalError(
                    f"the reranker model {self.model_name!r} could not be loaded: "
                    f"{type(exc).__name__}: {exc}",
                    code=ErrorCode.RERANK_FAILED,
                    retryable=False,
                ) from exc

            _logger.info(
                "reranker model ready",
                extra={
                    "model": self.model_name,
                    "load_seconds": round(time.perf_counter() - started, 1),
                },
            )
        return self._model

    def score(self, query: str, chunks: Sequence[ScoredChunk]) -> list[float]:
        """Score every query and chunk pair together."""
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
