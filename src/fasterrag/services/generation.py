"""Answer generation: retrieve, assemble, generate, cite.

Produces the event sequence the SSE contract specifies —
``meta`` → ``token``* → ``citations`` → ``usage`` → ``done`` — as typed events rather than
wire bytes, so the sequence is testable without HTTP and the router only has to serialize
(``docs/api-reference.md``).

Ordering is a contract, not a convenience. ``meta`` arrives first so a client knows the
trace id and whether the response is degraded *before* any text; ``citations`` arrives after
the answer because a citation is only real once the model has actually referenced it; and
``usage`` is last because a streamed call cannot know its token count until it ends.

A mid-stream failure emits ``error`` and stops without ``done``. Clients are told to treat a
missing ``done`` as an incomplete answer, so the absence carries meaning and must never be
papered over by emitting it anyway.

When generation fails outright the degradation ladder serves an ``extractive`` answer — the
retrieved passages themselves — rather than nothing (D4). Retrieval already succeeded at that
point, so the user gets the material even though the model could not summarize it.

Only a **retryable** failure takes that rung. Degradation absorbs a provider that is *down* —
rate-limited, timing out, returning 5xx. A rejected credential, an unknown model name, or a
missing provider extra is not down, it is misconfigured, and no retry will change that;
degrading one would answer every query extractively forever while reporting nothing more
specific than ``degraded: true``. Those surface as the typed error they already are.

Grounded-or-refuse (D5) adds a P3 grading call after generation. Below
``generation.faithfulness_threshold`` the answer is withheld and the caller gets the
retrieved candidates instead of a guess. An ungraded answer — grader down, unparseable
response — is always returned: a grader outage is not evidence of hallucination.

Refusal and streaming genuinely conflict, because a token cannot be unsaid once it has been
sent. With ``grounded_or_refuse`` enabled the stream therefore generates into a buffer and
grades before emitting any ``token`` event, trading time-to-first-token for the guarantee the
flag exists to provide. The flag defaults to ``false``, so that trade is only ever made by an
operator who asked for it.
"""

from __future__ import annotations

import time
from collections.abc import AsyncIterator, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Final

from fasterrag.adapters.embeddings.base import EmbeddingAdapter
from fasterrag.adapters.llm.base import LLMAdapter
from fasterrag.config.schema import Settings
from fasterrag.core.cache.semantic import MISS, CacheHit, SemanticCache
from fasterrag.core.chunking.models import EstimatingTokenCounter, TokenCounter
from fasterrag.core.context import AssembledContext, Citation, Span, assemble_context
from fasterrag.core.faithfulness import UNGRADED, FaithfulnessVerdict, grade
from fasterrag.core.generation import (
    P1_SYSTEM_PROMPT,
    P1_TEMPLATE_VERSION,
    build_prompt,
    resolve_citations,
)
from fasterrag.core.retrieval.models import ScoredChunk
from fasterrag.errors import ErrorCode, FasterRagError
from fasterrag.observability.logging import current_trace_id, get_logger
from fasterrag.services.querying import FULL_MODE, RetrievalService

__all__ = [
    "DEFAULT_CONTEXT_BUDGET_TOKENS",
    "EXTRACTIVE_MODE",
    "MAXIMUM_BEST_CANDIDATES",
    "Answer",
    "GenerationService",
    "QueryEvent",
]

EXTRACTIVE_MODE: Final = "extractive"

MAXIMUM_BEST_CANDIDATES: Final = 5

# CRITICAL: no configuration key carries a provider's context-window size, so this is the
# budget used when a caller supplies none. It is deliberately conservative: overflowing a
# window is silent truncation, and losing the passage that mattered is worse than sending
# fewer chunks.
DEFAULT_CONTEXT_BUDGET_TOKENS: Final = 4000

_logger = get_logger(__name__)


@dataclass(frozen=True, slots=True)
class QueryEvent:
    """One event in the response stream."""

    type: str
    data: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class Answer:
    """A complete, non-streamed answer."""

    answer: str | None
    citations: list[Citation] = field(default_factory=list)
    mode: str = FULL_MODE
    trace_id: str = ""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    timings_ms: dict[str, int] = field(default_factory=dict)
    faithfulness: float | None = None
    threshold: float | None = None
    best_candidates: list[dict[str, Any]] = field(default_factory=list)
    cache: dict[str, Any] = field(default_factory=lambda: dict(MISS))

    @property
    def degraded(self) -> bool:
        """Return whether any stage was skipped or substituted."""
        return self.mode != FULL_MODE

    @property
    def insufficient_evidence(self) -> bool:
        """Return whether grounded-or-refuse withheld the answer (D5)."""
        return self.answer is None

    def as_dict(self) -> dict[str, Any]:
        """Return the non-streaming response body.

        A refusal is a different document from an answer: it carries the machine-readable
        code, the score and threshold that produced it, and the candidates the caller can
        inspect. Both shapes are HTTP 200 — declining to guess is a correct outcome, not a
        transport error.
        """
        if self.insufficient_evidence:
            return {
                "code": ErrorCode.INSUFFICIENT_EVIDENCE.value,
                "answer": None,
                "best_candidates": self.best_candidates,
                "faithfulness": self.faithfulness,
                "threshold": self.threshold,
                "trace_id": self.trace_id,
            }

        return {
            "answer": self.answer,
            "citations": [citation.as_dict() for citation in self.citations],
            "usage": {
                "prompt_tokens": self.prompt_tokens,
                "completion_tokens": self.completion_tokens,
            },
            "timings_ms": self.timings_ms,
            "degraded": self.degraded,
            "mode": self.mode,
            "faithfulness": self.faithfulness,
            "cache": self.cache,
            "trace_id": self.trace_id,
        }


def _extractive_answer(context: AssembledContext) -> str:
    """Return the retrieved passages as the answer of last resort."""
    return context.text


def _citation_from(payload: Mapping[str, Any]) -> Citation:
    """Rebuild a citation from its serialized form in a cached response."""
    span = payload.get("span")
    return Citation(
        chunk_id=str(payload.get("chunk_id", "")),
        source=payload.get("source"),
        page=payload.get("page"),
        span=Span(start=int(span["start"]), end=int(span["end"]))
        if isinstance(span, Mapping)
        else None,
        score=payload.get("score"),
    )


def _cached_answer(hit: CacheHit, trace_id: str, elapsed_ms: int) -> Answer:
    """Rebuild an ``Answer`` from a cached response body.

    The stored body is replayed as-is apart from three fields: the trace id becomes this
    query's, the timings become what this query actually spent, and ``cache`` records the hit
    and its similarity. Replaying the original trace id would attribute this request to a
    different one, and replaying the original timings would report a latency nobody paid.
    """
    stored = hit.response
    usage = stored.get("usage") or {}
    citations = stored.get("citations") or []
    return Answer(
        answer=stored.get("answer"),
        citations=[_citation_from(payload) for payload in citations],
        mode=str(stored.get("mode", FULL_MODE)),
        trace_id=trace_id,
        prompt_tokens=int(usage.get("prompt_tokens", 0)),
        completion_tokens=int(usage.get("completion_tokens", 0)),
        timings_ms={"cache": elapsed_ms},
        faithfulness=stored.get("faithfulness"),
        cache=hit.as_dict(),
    )


def _cached_events(answer: Answer) -> list[QueryEvent]:
    """Return the event sequence a cache hit replays.

    The same shape as a generated response — the whole answer simply arrives as one
    ``token`` event, since there is nothing left to stream incrementally. ``meta`` carries
    the cache member so a client learns it is being served from cache before the text.
    """
    return [
        QueryEvent(
            type="meta",
            data={
                "trace_id": answer.trace_id,
                "mode": answer.mode,
                "degraded": answer.degraded,
                "cache": answer.cache,
            },
        ),
        QueryEvent(type="token", data={"text": answer.answer or ""}),
        QueryEvent(
            type="citations",
            data={"citations": [citation.as_dict() for citation in answer.citations]},
        ),
        QueryEvent(
            type="usage",
            data={
                "usage": {
                    "prompt_tokens": answer.prompt_tokens,
                    "completion_tokens": answer.completion_tokens,
                },
                "timings_ms": answer.timings_ms,
                "faithfulness": answer.faithfulness,
                "template_version": P1_TEMPLATE_VERSION,
            },
        ),
        QueryEvent(type="done", data={}),
    ]


def _best_candidates(chunks: Sequence[ScoredChunk]) -> list[dict[str, Any]]:
    """Return what a refusal offers instead of an answer.

    The top-ranked chunks with their sources and scores, so a caller told "not enough
    evidence" can see what evidence there was and judge the refusal for themselves.
    """
    return [
        {
            "chunk_id": chunk.chunk_id,
            "source": chunk.source,
            "score": round(
                chunk.rerank_score if chunk.rerank_score is not None else chunk.rrf_score, 6
            ),
        }
        for chunk in chunks[:MAXIMUM_BEST_CANDIDATES]
    ]


@dataclass(frozen=True, slots=True)
class _Prepared:
    """Everything retrieval and assembly produced, before a model has seen it."""

    context: AssembledContext
    texts: list[str]
    prompt: str
    mode: str
    chunks: list[ScoredChunk]
    timings: dict[str, int]


class GenerationService:
    """Answers a question from the corpus, streaming or whole."""

    def __init__(
        self,
        settings: Settings,
        retrieval: RetrievalService,
        llm: LLMAdapter,
        *,
        grader: LLMAdapter | None = None,
        cache: SemanticCache | None = None,
        embedder: EmbeddingAdapter | None = None,
        counter: TokenCounter | None = None,
        context_budget_tokens: int = DEFAULT_CONTEXT_BUDGET_TOKENS,
    ) -> None:
        """Build the service.

        Args:
            settings: Validated configuration.
            retrieval: Supplies the chunks an answer is built from.
            llm: The generation provider.
            grader: The P3 faithfulness provider. Defaults to ``llm``; passing a separate
                adapter is how a cheaper grading model is selected. It is a distinct call
                either way, so the grader never sees P1's instructions.
            cache: The semantic response cache. Consulted only when ``embedder`` is also
                supplied, since a similarity cache cannot be keyed without a vector.
            embedder: Embeds the question for the cache lookup. The same query is embedded
                again inside retrieval, which costs nothing extra: the embedding cache is on
                by default and the second call is a hit on the first.
            counter: Token counter used for context budgeting.
            context_budget_tokens: Tokens available for context, which the caller sizes
                from the model's window minus room for the answer.
        """
        self.settings = settings
        self.retrieval = retrieval
        self.llm = llm
        self.grader = grader or llm
        self.cache = cache
        self.embedder = embedder
        self.counter = counter or EstimatingTokenCounter()
        self.context_budget_tokens = context_budget_tokens

    @property
    def grounded_or_refuse(self) -> bool:
        """Return whether low-faithfulness answers are withheld (D5)."""
        return self.settings.generation.grounded_or_refuse

    @property
    def caching(self) -> bool:
        """Return whether a semantic cache lookup is possible at all."""
        return self.cache is not None and self.cache.enabled and self.embedder is not None

    async def _cache_vector(self, question: str) -> list[float] | None:
        """Return the question's vector for cache use, or ``None`` if caching is off.

        An embedding failure here yields ``None`` rather than propagating: the cache is an
        optimization, and refusing to answer because the *cache key* could not be computed
        would be the cache taking down the pipeline it exists to accelerate.
        """
        if not self.caching or self.embedder is None:
            return None

        try:
            return await self.embedder.embed_query(question)
        except FasterRagError as exc:
            _logger.warning(
                "could not embed the question for the semantic cache; skipping the lookup",
                extra={"code": exc.code.value, "trace_id": exc.trace_id},
            )
            return None

    async def _prepare(
        self,
        question: str,
        *,
        collection: str | None,
        top_k: int | None,
        filters: Mapping[str, Any] | None,
    ) -> _Prepared:
        """Retrieve and assemble everything generation and grading both need."""
        started = time.perf_counter()
        retrieved = await self.retrieval.search(
            question, collection=collection, top_k=top_k, filters=filters
        )
        retrieve_ms = int((time.perf_counter() - started) * 1000)

        started = time.perf_counter()
        context = assemble_context(
            retrieved.chunks,
            budget_tokens=self.context_budget_tokens,
            counter=self.counter,
        )
        texts = [chunk.text for chunk in retrieved.chunks[: context.used]]
        prompt = build_prompt(question, context, texts)
        assemble_ms = int((time.perf_counter() - started) * 1000)

        return _Prepared(
            context=context,
            texts=texts,
            prompt=prompt,
            mode=retrieved.mode,
            chunks=retrieved.chunks,
            timings={"retrieve": retrieve_ms, "assemble": assemble_ms},
        )

    async def _grade(
        self, question: str, answer: str, prepared: _Prepared, timings: dict[str, int]
    ) -> FaithfulnessVerdict:
        """Score an answer when D5 is on, recording how long it took."""
        if not self.grounded_or_refuse:
            return UNGRADED

        started = time.perf_counter()
        verdict = await grade(self.grader, question, answer, prepared.context, prepared.texts)
        timings["grade"] = int((time.perf_counter() - started) * 1000)
        return verdict

    def _refusal(
        self,
        prepared: _Prepared,
        verdict: FaithfulnessVerdict,
        trace_id: str,
        timings: dict[str, int],
    ) -> Answer:
        """Build the INSUFFICIENT_EVIDENCE response for a withheld answer."""
        threshold = self.settings.generation.faithfulness_threshold
        _logger.info(
            "answer withheld below the faithfulness threshold",
            extra={
                "trace_id": trace_id,
                "faithfulness": verdict.score,
                "threshold": threshold,
                "unsupported_claims": len(verdict.unsupported_claims),
            },
        )
        return Answer(
            answer=None,
            mode=prepared.mode,
            trace_id=trace_id,
            timings_ms=timings,
            faithfulness=verdict.score,
            threshold=threshold,
            best_candidates=_best_candidates(prepared.chunks),
        )

    async def answer(
        self,
        question: str,
        *,
        collection: str | None = None,
        top_k: int | None = None,
        filters: Mapping[str, Any] | None = None,
    ) -> Answer:
        """Answer a question completely, without streaming.

        A semantic cache hit short-circuits everything below it — retrieval, assembly,
        generation, and grading — and returns the stored answer with ``cache.semantic_hit``
        set, so a caller can always tell a fresh answer from a reused one.
        """
        trace_id = current_trace_id()
        started = time.perf_counter()
        vector = await self._cache_vector(question)
        cache_ms = int((time.perf_counter() - started) * 1000)

        if vector is not None and self.cache is not None:
            hit = await self.cache.lookup(vector)
            if hit is not None:
                return _cached_answer(hit, trace_id, cache_ms)

        prepared = await self._prepare(
            question, collection=collection, top_k=top_k, filters=filters
        )
        timings = prepared.timings
        if vector is not None:
            timings["cache"] = cache_ms

        started = time.perf_counter()
        try:
            completion = await self.llm.complete(prepared.prompt, system=P1_SYSTEM_PROMPT)
        except FasterRagError as exc:
            # CRITICAL: only a retryable failure is a rung on the degradation ladder. A
            # rejected key, an unknown model, or a missing provider extra will never succeed
            # on a retry, so degrading one would answer every query extractively forever
            # while reporting nothing more specific than `degraded: true`. The ladder
            # absorbs a provider that is *down*, not one that is misconfigured.
            if not exc.retryable:
                raise
            timings["generate"] = int((time.perf_counter() - started) * 1000)
            _logger.warning(
                "generation failed, serving the retrieved passages instead",
                extra={
                    "code": exc.code.value,
                    # CRITICAL: the detail belongs in the log. A degraded response reports
                    # only `mode`, so without the provider's own reason here there is no
                    # record anywhere of why the answer was extractive.
                    "detail": exc.detail,
                    "retryable": exc.retryable,
                    "trace_id": trace_id,
                    "mode": EXTRACTIVE_MODE,
                },
            )
            return Answer(
                answer=_extractive_answer(prepared.context),
                citations=list(prepared.context.citations),
                mode=EXTRACTIVE_MODE,
                trace_id=trace_id,
                timings_ms=timings,
            )

        timings["generate"] = int((time.perf_counter() - started) * 1000)
        verdict = await self._grade(question, completion.text, prepared, timings)
        if verdict.withholds(self.settings.generation.faithfulness_threshold):
            return self._refusal(prepared, verdict, trace_id, timings)

        answer = Answer(
            answer=completion.text,
            citations=resolve_citations(completion.text, prepared.context.citations),
            mode=prepared.mode,
            trace_id=trace_id,
            prompt_tokens=completion.prompt_tokens,
            completion_tokens=completion.completion_tokens,
            timings_ms=timings,
            faithfulness=verdict.score,
        )

        if vector is not None and self.cache is not None and prepared.mode == FULL_MODE:
            await self.cache.store_response(question, vector, answer.as_dict())

        return answer

    async def stream(
        self,
        question: str,
        *,
        collection: str | None = None,
        top_k: int | None = None,
        filters: Mapping[str, Any] | None = None,
    ) -> AsyncIterator[QueryEvent]:
        """Answer a question as a stream of events in the documented order.

        With ``generation.grounded_or_refuse`` enabled, no ``token`` event is emitted until
        the whole answer has been graded, because a refusal after the text has left is not
        a refusal. A withheld answer produces an ``insufficient_evidence`` event in place of
        the token, citation, and usage events, then ``done`` — the query completed, so the
        stream is not truncated.

        A semantic cache hit still streams. The whole answer is already known, so it arrives
        as a single ``token`` event after ``meta`` — the event contract holds and a client
        needs no separate code path for a cached response.
        """
        trace_id = current_trace_id()
        started = time.perf_counter()
        vector = await self._cache_vector(question)
        cache_ms = int((time.perf_counter() - started) * 1000)

        if vector is not None and self.cache is not None:
            hit = await self.cache.lookup(vector)
            if hit is not None:
                for event in _cached_events(_cached_answer(hit, trace_id, cache_ms)):
                    yield event
                return

        prepared = await self._prepare(
            question, collection=collection, top_k=top_k, filters=filters
        )
        timings = prepared.timings
        if vector is not None:
            timings["cache"] = cache_ms

        yield QueryEvent(
            type="meta",
            data={
                "trace_id": trace_id,
                "mode": prepared.mode,
                "degraded": prepared.mode != FULL_MODE,
            },
        )

        started = time.perf_counter()
        parts: list[str] = []
        gated = self.grounded_or_refuse
        try:
            async for delta in self.llm.stream(prepared.prompt, system=P1_SYSTEM_PROMPT):
                parts.append(delta)
                if not gated:
                    yield QueryEvent(type="token", data={"text": delta})
        except FasterRagError as exc:
            if not exc.retryable:
                raise
            _logger.warning(
                "generation stream failed",
                extra={
                    "code": exc.code.value,
                    "detail": exc.detail,
                    "retryable": exc.retryable,
                    "trace_id": trace_id,
                },
            )
            yield QueryEvent(
                type="error",
                data={
                    "code": exc.code.value,
                    "detail": exc.detail,
                    "trace_id": exc.trace_id,
                    "retryable": exc.retryable,
                },
            )
            return

        answer = "".join(parts)
        timings["generate"] = int((time.perf_counter() - started) * 1000)

        verdict = await self._grade(question, answer, prepared, timings)
        if verdict.withholds(self.settings.generation.faithfulness_threshold):
            refusal = self._refusal(prepared, verdict, trace_id, timings)
            yield QueryEvent(type="insufficient_evidence", data=refusal.as_dict())
            yield QueryEvent(type="done", data={})
            return

        if gated:
            yield QueryEvent(type="token", data={"text": answer})

        citations = resolve_citations(answer, prepared.context.citations)
        usage = {
            "prompt_tokens": self.counter.count(prepared.prompt),
            "completion_tokens": self.counter.count(answer),
        }

        yield QueryEvent(
            type="citations",
            data={"citations": [citation.as_dict() for citation in citations]},
        )
        yield QueryEvent(
            type="usage",
            data={
                "usage": usage,
                "timings_ms": timings,
                "faithfulness": verdict.score,
                "template_version": P1_TEMPLATE_VERSION,
            },
        )
        yield QueryEvent(type="done", data={})

        if vector is not None and self.cache is not None and prepared.mode == FULL_MODE:
            await self.cache.store_response(
                question,
                vector,
                Answer(
                    answer=answer,
                    citations=citations,
                    mode=prepared.mode,
                    trace_id=trace_id,
                    prompt_tokens=usage["prompt_tokens"],
                    completion_tokens=usage["completion_tokens"],
                    timings_ms=timings,
                    faithfulness=verdict.score,
                ).as_dict(),
            )

    async def close(self) -> None:
        """Release the generation provider.

        # CRITICAL: the cache is deliberately not closed here. It is shared across queries
        # and outlives any one of them — a per-query service closing it would tear down a
        # backend connection other in-flight queries are still using. Whoever constructed
        # the cache closes it.
        """
        await self.llm.close()
