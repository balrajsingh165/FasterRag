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
import contextvars
import threading
import time
from collections.abc import Sequence
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import replace
from typing import Any, Final, Protocol

from fasterrag.config.schema import Settings
from fasterrag.core.retrieval.models import ScoredChunk
from fasterrag.errors import ConfigError, ErrorCode, RetrievalError
from fasterrag.observability.logging import get_logger

__all__ = [
    "MAX_RERANKS_IN_FLIGHT",
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

MAX_RERANKS_IN_FLIGHT: Final = 4
"""How many scoring calls may hold a worker thread at once (TASK-0449).

A rerank abandoned at ``reliability.timeouts.rerank_ms`` keeps its thread until the model
returns, because a thread cannot be cancelled. This bounds how many threads a hung model can
take; it is a cap on damage, not a tuned or measured throughput figure.
"""


class _ScoringSlots:
    """The threads scoring runs on, and the slots a call holds until its thread finishes."""

    def __init__(self) -> None:
        """Start with every slot free and no thread started."""
        self.free = threading.BoundedSemaphore(MAX_RERANKS_IN_FLIGHT)
        self._pool: ThreadPoolExecutor | None = None

    def pool(self) -> ThreadPoolExecutor:
        """Return the scoring threads, starting them on first use."""
        if self._pool is None:
            self._pool = ThreadPoolExecutor(
                max_workers=MAX_RERANKS_IN_FLIGHT, thread_name_prefix="fasterrag-rerank"
            )
        return self._pool


_SLOTS: Final = _ScoringSlots()


def reset_shared_cross_encoders() -> None:
    """Drop every loaded model and free every slot, so a test or a reload starts afresh."""
    with _LOADING:
        _MODELS.clear()
    _SLOTS.free = threading.BoundedSemaphore(MAX_RERANKS_IN_FLIGHT)


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

    def _submit(self, query: str, chunks: Sequence[ScoredChunk]) -> Future[list[float]]:
        """Start scoring on a worker thread, holding a slot until that thread is done.

        The slot is freed by the thread finishing, not by the caller giving up, so an
        abandoned call counts against ``MAX_RERANKS_IN_FLIGHT`` until the model returns.
        """
        slots = _SLOTS.free
        if not slots.acquire(blocking=False):
            raise RetrievalError(
                f"all {MAX_RERANKS_IN_FLIGHT} reranking slots are held by scoring calls that "
                "have not returned, so this query is served the fused order instead of "
                "queueing behind them",
                code=ErrorCode.RERANK_FAILED,
                retryable=True,
            )
        try:
            context = contextvars.copy_context()
            future = _SLOTS.pool().submit(context.run, self.score, query, chunks)
        except BaseException:
            slots.release()
            raise
        future.add_done_callback(lambda _finished: slots.release())
        return future

    async def rerank(self, query: str, chunks: Sequence[ScoredChunk]) -> list[ScoredChunk]:
        """Reorder chunks by cross-encoder score, best first.

        Scoring is CPU- or GPU-bound, so it runs in a worker thread and never blocks the
        event loop serving other queries; at most ``MAX_RERANKS_IN_FLIGHT`` at once.
        """
        if not chunks:
            return []

        scores = await asyncio.wrap_future(self._submit(query, chunks))
        scored = [
            replace(chunk, rerank_score=score) for chunk, score in zip(chunks, scores, strict=True)
        ]
        ordered = sorted(scored, key=lambda chunk: -(chunk.rerank_score or 0.0))

        return [
            replace(chunk, final_rank=position) for position, chunk in enumerate(ordered, start=1)
        ]
