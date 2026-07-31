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

from fasterrag.adapters.llm.base import LLMAdapter
from fasterrag.config.schema import Settings
from fasterrag.core.chunking.models import EstimatingTokenCounter, TokenCounter
from fasterrag.core.context import AssembledContext, Citation, assemble_context
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
            "trace_id": self.trace_id,
        }


def _extractive_answer(context: AssembledContext) -> str:
    """Return the retrieved passages as the answer of last resort."""
    return context.text


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
            counter: Token counter used for context budgeting.
            context_budget_tokens: Tokens available for context, which the caller sizes
                from the model's window minus room for the answer.
        """
        self.settings = settings
        self.retrieval = retrieval
        self.llm = llm
        self.grader = grader or llm
        self.counter = counter or EstimatingTokenCounter()
        self.context_budget_tokens = context_budget_tokens

    @property
    def grounded_or_refuse(self) -> bool:
        """Return whether low-faithfulness answers are withheld (D5)."""
        return self.settings.generation.grounded_or_refuse

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
        """Answer a question completely, without streaming."""
        trace_id = current_trace_id()
        prepared = await self._prepare(
            question, collection=collection, top_k=top_k, filters=filters
        )
        timings = prepared.timings

        started = time.perf_counter()
        try:
            completion = await self.llm.complete(prepared.prompt, system=P1_SYSTEM_PROMPT)
        except FasterRagError as exc:
            timings["generate"] = int((time.perf_counter() - started) * 1000)
            _logger.warning(
                "generation failed, serving the retrieved passages instead",
                extra={"code": exc.code.value, "trace_id": trace_id, "mode": EXTRACTIVE_MODE},
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

        return Answer(
            answer=completion.text,
            citations=resolve_citations(completion.text, prepared.context.citations),
            mode=prepared.mode,
            trace_id=trace_id,
            prompt_tokens=completion.prompt_tokens,
            completion_tokens=completion.completion_tokens,
            timings_ms=timings,
            faithfulness=verdict.score,
        )

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
        """
        trace_id = current_trace_id()
        prepared = await self._prepare(
            question, collection=collection, top_k=top_k, filters=filters
        )
        timings = prepared.timings

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
            _logger.warning(
                "generation stream failed",
                extra={"code": exc.code.value, "trace_id": trace_id},
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

        yield QueryEvent(
            type="citations",
            data={
                "citations": [
                    citation.as_dict()
                    for citation in resolve_citations(answer, prepared.context.citations)
                ]
            },
        )
        yield QueryEvent(
            type="usage",
            data={
                "usage": {
                    "prompt_tokens": self.counter.count(prepared.prompt),
                    "completion_tokens": self.counter.count(answer),
                },
                "timings_ms": timings,
                "faithfulness": verdict.score,
                "template_version": P1_TEMPLATE_VERSION,
            },
        )
        yield QueryEvent(type="done", data={})

    async def close(self) -> None:
        """Release the generation provider."""
        await self.llm.close()
