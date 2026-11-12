"""Answer generation: retrieve, assemble, generate, cite.

Produces the event sequence the SSE contract specifies —
``meta`` → ``token``* → ``citations`` → ``usage`` → ``done`` — as typed events rather than
wire bytes, so the sequence is testable without HTTP and the router only has to serialize
(``docs/api-reference.md``).

One pipeline serves both shapes (TASK-0438). ``GenerationService._pipeline`` is the only
place a query runs: it yields typed stages — admitted, text, refused, completed, interrupted
— and owns the budget check, breaker accounting, metrics, spans, the cache write and the
trace. ``answer()`` collects those stages into an ``Answer`` and ``stream()`` maps each one to
events, so a degradation, refusal, flag, metric or trace cannot exist on one surface and not
the other. The two used to be hand-kept copies, and had drifted apart on every one of those.

What the surfaces cannot share is the provider call. A whole answer comes from ``complete``,
which reports its own token counts and finish reason; a stream carries text only, so a
streamed answer's counts are this process's estimate and ``truncated`` is inferred from that
estimate reaching ``llm.max_tokens``.

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

import hashlib
import json
import time
from collections.abc import AsyncIterator, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Final

from fasterrag.adapters.embeddings.base import EmbeddingAdapter
from fasterrag.adapters.llm.base import Completion, LLMAdapter
from fasterrag.config.schema import GenerationSettings, Settings
from fasterrag.core.breaker import CircuitBreaker, shared_breaker
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
from fasterrag.core.tracing import SpanRecorder, Trace, config_snapshot, record_chunk
from fasterrag.errors import ErrorCode, FasterRagError, GenerationError
from fasterrag.observability import metrics
from fasterrag.observability.logging import current_trace_id, get_logger
from fasterrag.services.budget import CostGovernor, shared_governor
from fasterrag.services.degradation import DegradationMode, serving
from fasterrag.services.estimation import price_generation
from fasterrag.services.querying import FULL_MODE, RetrievalService
from fasterrag.services.traces import TraceStore, now

__all__ = [
    "DEFAULT_CONTEXT_BUDGET_TOKENS",
    "EXTRACTIVE_MODE",
    "MAXIMUM_BEST_CANDIDATES",
    "Answer",
    "GenerationService",
    "QueryEvent",
    "extractive_answer",
]

EXTRACTIVE_MODE: Final = DegradationMode.EXTRACTIVE.value

MAXIMUM_BEST_CANDIDATES: Final = 5

# CRITICAL: read from the schema rather than restated, so the constant and
# `generation.context_budget_tokens` cannot drift apart. The value itself is deliberately
# conservative: no configuration key carries a provider's context-window size, overflowing
# a window is silent truncation, and losing the passage that mattered is worse than sending
# fewer chunks.
DEFAULT_CONTEXT_BUDGET_TOKENS: Final = GenerationSettings().context_budget_tokens

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
    truncated: bool = False
    """Whether the model stopped at ``llm.max_tokens`` rather than finishing.

    A cut-off answer is served, not withheld — a partial answer with a flag beats no answer
    — but serving one *silently* means a caller cannot tell a complete answer from one that
    ran out of budget mid-sentence. A reasoning model can also spend the whole ceiling before
    emitting anything, which arrives here as an empty answer flagged truncated.
    """

    estimated_cost_usd: float | None = None
    """List-price estimate for this query's generation call, or ``None`` when unpriced.

    An *estimate*, never a measurement: it multiplies the call's token counts by the dated
    published rates in ``services.estimation``. A model with no recorded rate contributes
    ``None`` rather than ``0.0`` — a zero would read as "this query was free", which is a
    fabricated bill, and the whole point of the per-entry provenance in those tables is that
    a number without a source is not published. Local providers are genuinely ``0.0``.

    Emitted per response as well as counted into ``fasterrag_cost_usd_total``, because a
    caller holding one answer cannot read a process-wide counter to find out what it cost.
    """

    evidence_dropped: int = 0
    """Chunks that were retrieved and reranked but did not fit the context budget.

    Distinct from ``truncated``: nothing was cut off, but the answer is grounded in a subset
    of the evidence the retriever chose. Raising ``chunking.chunk_size`` or lowering
    ``retrieval.top_k`` is what an operator does about it, and neither is discoverable if the
    number is never reported.
    """

    @property
    def degraded(self) -> bool:
        """Return whether any stage was skipped or substituted."""
        return self.mode != FULL_MODE

    @property
    def insufficient_evidence(self) -> bool:
        """Return whether grounded-or-refuse withheld the answer (D5)."""
        return self.answer is None

    def as_dict(self) -> dict[str, Any]:
        """Return the non-streaming response body."""
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
                "estimated_cost_usd": self.estimated_cost_usd,
            },
            "timings_ms": self.timings_ms,
            "degraded": self.degraded,
            "mode": self.mode,
            "truncated": self.truncated,
            "evidence_dropped": self.evidence_dropped,
            "faithfulness": self.faithfulness,
            "cache": self.cache,
            "trace_id": self.trace_id,
        }


def extractive_answer(context: AssembledContext) -> str:
    """Return the retrieved passages as the answer of last resort.

    Registered as the ``extractive`` rung's implementation in ``services.degradation``,
    which is how the tree can say the rung is built (D4, TASK-0394).
    """
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
    """Rebuild an ``Answer`` from a cached response body."""
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
    spans: SpanRecorder | None = None


@dataclass(frozen=True, slots=True)
class _Started:
    """The query was admitted: retrieval and the budget check passed, nothing is generated."""

    trace_id: str
    mode: str
    cache: dict[str, Any]


@dataclass(frozen=True, slots=True)
class _Delta:
    """A piece of answer text the caller may be shown."""

    text: str


@dataclass(frozen=True, slots=True)
class _Completed:
    """The finished answer, yielded after every record of producing it has been written."""

    answer: Answer


@dataclass(frozen=True, slots=True)
class _Refused:
    """Grounded-or-refuse withheld the answer (D5); the refusal is the response."""

    answer: Answer


@dataclass(frozen=True, slots=True)
class _Interrupted:
    """The provider failed and no answer can be served; a stream ends here without `done`."""

    error: FasterRagError


type _Stage = _Started | _Delta | _Completed | _Refused | _Interrupted


def _events(stage: _Stage) -> list[QueryEvent]:
    """Return the events of the SSE contract that one pipeline stage becomes."""
    if isinstance(stage, _Started):
        return [
            QueryEvent(
                type="meta",
                data={
                    "trace_id": stage.trace_id,
                    "mode": stage.mode,
                    "degraded": stage.mode != FULL_MODE,
                    "cache": stage.cache,
                },
            )
        ]

    if isinstance(stage, _Delta):
        return [QueryEvent(type="token", data={"text": stage.text})]

    if isinstance(stage, _Interrupted):
        return [
            QueryEvent(
                type="error",
                data={
                    "code": stage.error.code.value,
                    "detail": stage.error.detail,
                    "trace_id": stage.error.trace_id,
                    "retryable": stage.error.retryable,
                    "status": stage.error.status,
                },
            )
        ]

    if isinstance(stage, _Refused):
        return [
            QueryEvent(type="insufficient_evidence", data=stage.answer.as_dict()),
            QueryEvent(type="done", data={}),
        ]

    answer = stage.answer
    return [
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
                    "estimated_cost_usd": answer.estimated_cost_usd,
                },
                "timings_ms": answer.timings_ms,
                "faithfulness": answer.faithfulness,
                "template_version": P1_TEMPLATE_VERSION,
                "mode": answer.mode,
                "degraded": answer.degraded,
                "truncated": answer.truncated,
                "evidence_dropped": answer.evidence_dropped,
            },
        ),
        QueryEvent(type="done", data={}),
    ]


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
        traces: TraceStore | None = None,
        tenant: str | None = None,
        counter: TokenCounter | None = None,
        context_budget_tokens: int | None = None,
        breaker: CircuitBreaker | None = None,
        governor: CostGovernor | None = None,
    ) -> None:
        """Build the service."""
        self.settings = settings
        self.retrieval = retrieval
        self.llm = llm
        self.grader = grader or llm
        self.cache = cache
        self.embedder = embedder
        self.traces = traces
        self.tenant = tenant
        self.counter = counter or EstimatingTokenCounter()
        self.context_budget_tokens = (
            settings.generation.context_budget_tokens
            if context_budget_tokens is None
            else context_budget_tokens
        )
        # CRITICAL: the process-wide governor, not a fresh one (TASK-0337). `api/query.py`
        # builds this service per request, so a `create_governor` call here rebuilt and
        # discarded the rolling 24-hour window on every request and made
        # `cost.per_tenant_token_budget` unreachable through the REST API. It is also the
        # instance the embedding pool records ingest spend into, which is the only way a
        # query's check can see what ingestion already spent.
        self.governor = governor or shared_governor(settings)
        # CRITICAL: the process-wide breaker, not a fresh one. This object is built per
        # request on the REST path and per job on the ingest path, and a breaker rebuilt
        # per unit of work counts to one forever and can never open.
        self.breaker = breaker or shared_breaker("llm", settings)
        self._last_candidates: list[ScoredChunk] = []

    @property
    def grounded_or_refuse(self) -> bool:
        """Return whether low-faithfulness answers are withheld (D5)."""
        return self.settings.generation.grounded_or_refuse

    @property
    def caching(self) -> bool:
        """Return whether a semantic cache lookup is possible at all."""
        return self.cache is not None and self.cache.enabled and self.embedder is not None

    def _cache_scope(
        self, collection: str | None, top_k: int | None, filters: Mapping[str, Any] | None
    ) -> str:
        """Return the digest of everything besides the question that decides an answer.

        Two queries with the same words are the same question only against the same
        collection, under the same filters, for the same number of chunks.
        """
        target = collection or self.settings.vector_db.collection.default_name
        scoped = {
            "collection": target,
            "top_k": top_k or self.settings.retrieval.top_k,
            "filters": dict(filters or {}),
        }
        canonical = json.dumps(scoped, sort_keys=True, default=str, separators=(",", ":"))
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:24]

    async def _cache_vector(self, question: str) -> list[float] | None:
        """Return the question's vector for cache use, or ``None`` if caching is off."""
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
        recorder: SpanRecorder | None = None,
    ) -> _Prepared:
        """Retrieve and assemble everything generation and grading both need."""
        spans = recorder or SpanRecorder()

        retrieval_started = spans.elapsed_ms
        started = time.perf_counter()
        retrieved = await self.retrieval.search(
            question, collection=collection, top_k=top_k, filters=filters
        )
        retrieve_ms = int((time.perf_counter() - started) * 1000)
        spans.record(
            "retrieval",
            retrieval_started,
            candidates=len(retrieved.chunks),
            mode=retrieved.mode,
            hybrid=self.settings.retrieval.hybrid,
        )

        # CRITICAL: reranking is reported as its own span only when it actually ran. A
        # zero-duration span for a skipped stage is indistinguishable from a stage that ran
        # instantly, and D4's whole point is telling those apart.
        if self.settings.retrieval.rerank and retrieved.mode == FULL_MODE:
            spans.record(
                "reranker",
                retrieval_started,
                model=self.settings.retrieval.reranker_model,
                reranked=sum(1 for chunk in retrieved.chunks if chunk.rerank_score is not None),
            )

        assemble_started = spans.elapsed_ms
        started = time.perf_counter()
        context = assemble_context(
            retrieved.chunks,
            budget_tokens=self.context_budget_tokens,
            counter=self.counter,
        )
        texts = context.packed
        prompt = build_prompt(question, context, texts)
        assemble_ms = int((time.perf_counter() - started) * 1000)
        spans.record(
            "context-assembly",
            assemble_started,
            packed=context.used,
            tokens=context.tokens,
            dropped_duplicate=context.dropped_duplicate,
            dropped_budget=context.dropped_budget,
        )

        metrics.STAGE_DURATION.observe(retrieve_ms / 1000, stage="retrieve")
        metrics.STAGE_DURATION.observe(assemble_ms / 1000, stage="assemble")

        self._last_candidates = list(retrieved.chunks)

        return _Prepared(
            context=context,
            texts=texts,
            prompt=prompt,
            mode=retrieved.mode,
            chunks=retrieved.chunks,
            timings={"retrieve": retrieve_ms, "assemble": assemble_ms},
            spans=spans,
        )

    def _store_trace(
        self,
        question: str,
        collection: str | None,
        filters: Mapping[str, Any] | None,
        prepared: _Prepared,
        response: str,
        answer: Answer,
    ) -> None:
        """Persist everything this query did, for later inspection and replay (D8).

        Records the *full* candidate set rather than the packed subset: replay explains why
        an ordering changed, and a chunk that lost its place is exactly the evidence needed.
        """
        if self.traces is None or not self.traces.enabled:
            return

        self.traces.store(
            Trace(
                trace_id=answer.trace_id,
                query=question,
                collection=collection,
                tenant=self.tenant,
                filters=dict(filters) if filters else None,
                config_snapshot=config_snapshot(self.settings),
                retrieved=[record_chunk(chunk) for chunk in prepared.chunks],
                prompt=prepared.prompt,
                response=response,
                result=answer.as_dict(),
                spans=prepared.spans.spans if prepared.spans else [],
                created_at=now(),
            )
        )

    def _charge_budget(self, prepared: _Prepared) -> None:
        """Refuse the call when it cannot fit inside the configured token budgets (D9)."""
        if not self.governor.enabled:
            return

        worst_case = self.counter.count(prepared.prompt) + self.settings.llm.max_tokens
        self.governor.check(worst_case, tenant=self.tenant)

    def _record_generation(self, elapsed_ms: int, prompt: int, completion: int) -> None:
        """Record what one generation call cost, in time, in tokens, and in money."""
        # CRITICAL: the tenant is charged here, at the one point both the streaming
        # and non-streaming paths pass through, and with the *actual* counts rather
        # than the ceiling the request was admitted against. Charging at the check
        # would bill every short answer for a long one and exhaust a budget that was
        # never spent.
        self.governor.record(prompt + completion, tenant=self.tenant)

        provider = self.llm.provider
        metrics.STAGE_DURATION.observe(elapsed_ms / 1000, stage="generate")
        if prompt:
            metrics.TOKENS.increment(float(prompt), kind="prompt", provider=provider, tenant="none")
        if completion:
            metrics.TOKENS.increment(
                float(completion), kind="completion", provider=provider, tenant="none"
            )

        estimated = price_generation(provider, self.llm.model, prompt, completion)
        if estimated is None:
            # CRITICAL: an unpriced model must be *visible*, not simply absent. Silently
            # contributing nothing to the cost total makes the dashboard understate spend
            # while looking healthy, which is worse than reporting no cost at all — the
            # panel cannot say "some traffic is unpriced" unless something counts it.
            metrics.UNPRICED_TOKENS.increment(
                float(prompt + completion), provider=provider, model=self.llm.model
            )
        elif estimated:
            metrics.COST.increment(estimated, provider=provider, tenant="none")

    async def _grade(
        self, question: str, answer: str, prepared: _Prepared, timings: dict[str, int]
    ) -> FaithfulnessVerdict:
        """Score an answer when D5 is on, recording how long it took."""
        if not self.grounded_or_refuse:
            return UNGRADED

        started = time.perf_counter()
        verdict = await grade(self.grader, question, answer, prepared.context, prepared.texts)
        timings["grade"] = int((time.perf_counter() - started) * 1000)
        if verdict.score is not None:
            metrics.FAITHFULNESS.observe(verdict.score)
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

    async def answer_with_candidates(
        self,
        question: str,
        *,
        collection: str | None = None,
        top_k: int | None = None,
        filters: Mapping[str, Any] | None = None,
    ) -> tuple[Answer, list[ScoredChunk]]:
        """Answer a question and return the full candidate set it retrieved."""
        self._last_candidates = []
        answer = await self.answer(question, collection=collection, top_k=top_k, filters=filters)
        return answer, self._last_candidates

    async def answer(
        self,
        question: str,
        *,
        collection: str | None = None,
        top_k: int | None = None,
        filters: Mapping[str, Any] | None = None,
    ) -> Answer:
        """Answer a question completely, without streaming."""
        final: Answer | None = None
        async for stage in self._pipeline(
            question, collection=collection, top_k=top_k, filters=filters, streaming=False
        ):
            if isinstance(stage, _Interrupted):
                raise stage.error
            if isinstance(stage, _Completed | _Refused):
                final = stage.answer

        if final is None:
            raise GenerationError(
                "the query pipeline ended without producing an answer", retryable=False
            )
        return final

    async def stream(
        self,
        question: str,
        *,
        collection: str | None = None,
        top_k: int | None = None,
        filters: Mapping[str, Any] | None = None,
    ) -> AsyncIterator[QueryEvent]:
        """Answer a question as a stream of events in the documented order."""
        async for stage in self._pipeline(
            question, collection=collection, top_k=top_k, filters=filters, streaming=True
        ):
            for event in _events(stage):
                yield event

    def _streamed_completion(self, prompt: str, text: str) -> Completion:
        """Describe a finished stream the way ``complete`` describes a whole call.

        A stream carries text only, so the counts and the finish reason are this process's
        own estimate: truncation is inferred from the count reaching ``llm.max_tokens``.
        """
        completion_tokens = self.counter.count(text)
        return Completion(
            text=text,
            model=self.llm.model,
            prompt_tokens=self.counter.count(prompt),
            completion_tokens=completion_tokens,
            finish_reason="length" if completion_tokens >= self.settings.llm.max_tokens else None,
        )

    def _degrade(
        self, prepared: _Prepared, exc: FasterRagError, trace_id: str, timings: dict[str, int]
    ) -> Answer:
        """Serve the retrieved passages in place of an answer the provider could not give."""
        # CRITICAL: the mode comes from the ladder's registry, never from a literal, so
        # a rung cannot be served without registering itself and the documentation's
        # claim about which rungs exist is decidable against the tree.
        mode = serving(DegradationMode.EXTRACTIVE)
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
                "mode": mode,
            },
        )
        metrics.DEGRADED_RESPONSES.increment(mode=mode)
        return Answer(
            answer=extractive_answer(prepared.context),
            citations=list(prepared.context.citations),
            mode=mode,
            trace_id=trace_id,
            timings_ms=timings,
        )

    async def _pipeline(
        self,
        question: str,
        *,
        collection: str | None,
        top_k: int | None,
        filters: Mapping[str, Any] | None,
        streaming: bool,
    ) -> AsyncIterator[_Stage]:
        """Run one query, yielding each stage as it concludes (TASK-0438).

        The only place a query is executed. ``answer()`` collects these stages and
        ``stream()`` maps them to events, so the two surfaces cannot disagree.
        """
        trace_id = current_trace_id()
        started = time.perf_counter()
        vector = await self._cache_vector(question)
        cache_ms = int((time.perf_counter() - started) * 1000)

        if vector is not None and self.cache is not None:
            hit = await self.cache.lookup(
                vector,
                tenant=self.tenant,
                scope=self._cache_scope(collection, top_k, filters),
            )
            if hit is not None:
                cached = _cached_answer(hit, trace_id, cache_ms)
                yield _Started(trace_id=trace_id, mode=cached.mode, cache=cached.cache)
                yield _Delta(text=cached.answer or "")
                yield _Completed(answer=cached)
                return

        prepared = await self._prepare(
            question, collection=collection, top_k=top_k, filters=filters
        )
        timings = prepared.timings
        if vector is not None:
            timings["cache"] = cache_ms

        # CRITICAL: *outside* the try below, and *above* the first stage, both deliberately.
        # That block absorbs a retryable failure into an extractive answer, and a budget
        # refusal must reach the caller as the 402 it is — degrading it would turn a spend
        # control into a silent quality regression. BUDGET_EXCEEDED is non-retryable in the
        # problem table, so it would re-raise anyway; keeping the call out here means that
        # stays true even if the ladder's conditions are ever loosened. And once a stream
        # has sent `meta` the HTTP status is committed to 200, so a refusal raised after it
        # could only reach a REST client as an SSE `error` event (TASK-0438).
        self._charge_budget(prepared)

        yield _Started(trace_id=trace_id, mode=prepared.mode, cache=dict(MISS))

        # CRITICAL: text is held back, not yielded as it arrives, in the two cases where it
        # may still be replaced by something else. A whole-answer call has no deltas, and
        # with grounded-or-refuse a token cannot be unsaid once sent, so the answer is
        # graded before a caller sees any of it.
        held = not streaming or self.grounded_or_refuse
        parts: list[str] = []
        started = time.perf_counter()
        try:
            # CRITICAL: inside the try, so an open circuit takes the same route a live
            # provider failure takes. `CircuitOpenError` is retryable by construction, which
            # makes shedding load a rung on the degradation ladder rather than a 503 — the
            # behaviour FMEA row 20 describes as "retries → breaker opens → extractive".
            self.breaker.check()
            if streaming:
                async for delta in self.llm.stream(prepared.prompt, system=P1_SYSTEM_PROMPT):
                    if not parts:
                        # CRITICAL: time to first token is measured from the request
                        # arriving, not from the provider call. What a user perceives
                        # includes retrieval and assembly, and a TTFT that excluded them
                        # would flatter every change that made retrieval slower.
                        metrics.TTFT.observe(
                            (timings.get("retrieve", 0) + timings.get("assemble", 0)) / 1000
                            + (time.perf_counter() - started)
                        )
                    parts.append(delta)
                    if not held:
                        yield _Delta(text=delta)
                completion = self._streamed_completion(prepared.prompt, "".join(parts))
            else:
                completion = await self.llm.complete(prepared.prompt, system=P1_SYSTEM_PROMPT)
        except FasterRagError as exc:
            self.breaker.record_failure(exc)
            # CRITICAL: only a retryable failure is a rung on the degradation ladder. A
            # rejected key, an unknown model, or a missing provider extra will never succeed
            # on a retry, so degrading one would answer every query extractively forever
            # while reporting nothing more specific than `degraded: true`. The ladder
            # absorbs a provider that is *down*, not one that is misconfigured.
            if not exc.retryable:
                raise
            timings["generate"] = int((time.perf_counter() - started) * 1000)
            if streaming:
                _logger.warning(
                    "generation stream failed",
                    extra={
                        "code": exc.code.value,
                        "detail": exc.detail,
                        "retryable": exc.retryable,
                        "trace_id": trace_id,
                    },
                )
                yield _Interrupted(error=exc)
                return
            yield _Completed(answer=self._degrade(prepared, exc, trace_id, timings))
            return

        self.breaker.record_success()
        generate_ms = int((time.perf_counter() - started) * 1000)
        timings["generate"] = generate_ms
        self._record_generation(generate_ms, completion.prompt_tokens, completion.completion_tokens)
        if prepared.spans is not None:
            prepared.spans.record(
                "generation",
                prepared.spans.elapsed_ms - generate_ms,
                model=completion.model,
                prompt_tokens=completion.prompt_tokens,
                completion_tokens=completion.completion_tokens,
            )

        verdict = await self._grade(question, completion.text, prepared, timings)
        if verdict.withholds(self.settings.generation.faithfulness_threshold):
            refusal = self._refusal(prepared, verdict, trace_id, timings)
            self._store_trace(question, collection, filters, prepared, completion.text, refusal)
            yield _Refused(answer=refusal)
            return

        if held:
            yield _Delta(text=completion.text)

        if completion.truncated:
            _logger.warning(
                "the model stopped at the token ceiling; the answer may be cut off",
                extra={
                    "trace_id": trace_id,
                    "max_tokens": self.settings.llm.max_tokens,
                    "completion_tokens": completion.completion_tokens,
                    "answer_empty": not completion.text.strip(),
                },
            )

        if prepared.context.truncated:
            _logger.info(
                "evidence was dropped to fit the context budget; the answer is grounded in "
                "a subset of what retrieval returned",
                extra={
                    "trace_id": trace_id,
                    "dropped": prepared.context.dropped_budget,
                    "kept": len(prepared.context.citations),
                },
            )

        answer = Answer(
            answer=completion.text,
            citations=resolve_citations(completion.text, prepared.context.citations),
            mode=prepared.mode,
            trace_id=trace_id,
            prompt_tokens=completion.prompt_tokens,
            completion_tokens=completion.completion_tokens,
            timings_ms=timings,
            faithfulness=verdict.score,
            truncated=completion.truncated,
            evidence_dropped=prepared.context.dropped_budget,
            estimated_cost_usd=price_generation(
                self.llm.provider,
                completion.model,
                completion.prompt_tokens,
                completion.completion_tokens,
            ),
        )

        if vector is not None and self.cache is not None and prepared.mode == FULL_MODE:
            await self.cache.store_response(
                question,
                vector,
                answer.as_dict(),
                tenant=self.tenant,
                scope=self._cache_scope(collection, top_k, filters),
            )

        # CRITICAL: stored *before* the final stage is yielded, not after it. A stream's
        # consumer is entitled to stop reading at `done`, and a generator abandoned there
        # never runs the code after its last `yield` — which is where this used to live,
        # so a consumer that stopped at `done` left no trace and no cache entry.
        # Streaming is the default path, and replay (D8) is only as complete as the
        # traces it has.
        self._store_trace(question, collection, filters, prepared, completion.text, answer)
        yield _Completed(answer=answer)

    async def close(self) -> None:
        """Release the generation provider."""
        await self.llm.close()
