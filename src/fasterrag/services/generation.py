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
"""

from __future__ import annotations

import time
from collections.abc import AsyncIterator, Mapping
from dataclasses import dataclass, field
from typing import Any, Final

from fasterrag.adapters.llm.base import LLMAdapter
from fasterrag.config.schema import Settings
from fasterrag.core.chunking.models import EstimatingTokenCounter, TokenCounter
from fasterrag.core.context import AssembledContext, Citation, assemble_context
from fasterrag.core.generation import (
    P1_SYSTEM_PROMPT,
    P1_TEMPLATE_VERSION,
    build_prompt,
    resolve_citations,
)
from fasterrag.errors import FasterRagError
from fasterrag.observability.logging import current_trace_id, get_logger
from fasterrag.services.querying import FULL_MODE, RetrievalService

__all__ = [
    "DEFAULT_CONTEXT_BUDGET_TOKENS",
    "EXTRACTIVE_MODE",
    "Answer",
    "GenerationService",
    "QueryEvent",
]

EXTRACTIVE_MODE: Final = "extractive"

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

    @property
    def degraded(self) -> bool:
        """Return whether any stage was skipped or substituted."""
        return self.mode != FULL_MODE

    def as_dict(self) -> dict[str, Any]:
        """Return the non-streaming response body."""
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
            "trace_id": self.trace_id,
        }


def _extractive_answer(context: AssembledContext) -> str:
    """Return the retrieved passages as the answer of last resort."""
    return context.text


class GenerationService:
    """Answers a question from the corpus, streaming or whole."""

    def __init__(
        self,
        settings: Settings,
        retrieval: RetrievalService,
        llm: LLMAdapter,
        *,
        counter: TokenCounter | None = None,
        context_budget_tokens: int = DEFAULT_CONTEXT_BUDGET_TOKENS,
    ) -> None:
        """Build the service.

        Args:
            settings: Validated configuration.
            retrieval: Supplies the chunks an answer is built from.
            llm: The generation provider.
            counter: Token counter used for context budgeting.
            context_budget_tokens: Tokens available for context, which the caller sizes
                from the model's window minus room for the answer.
        """
        self.settings = settings
        self.retrieval = retrieval
        self.llm = llm
        self.counter = counter or EstimatingTokenCounter()
        self.context_budget_tokens = context_budget_tokens

    async def _prepare(
        self,
        question: str,
        *,
        collection: str | None,
        top_k: int | None,
        filters: Mapping[str, Any] | None,
    ) -> tuple[AssembledContext, str, str, dict[str, int]]:
        """Retrieve and assemble, returning the context, prompt, mode, and timings."""
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

        return context, prompt, retrieved.mode, {"retrieve": retrieve_ms, "assemble": assemble_ms}

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
        context, prompt, mode, timings = await self._prepare(
            question, collection=collection, top_k=top_k, filters=filters
        )

        started = time.perf_counter()
        try:
            completion = await self.llm.complete(prompt, system=P1_SYSTEM_PROMPT)
        except FasterRagError as exc:
            timings["generate"] = int((time.perf_counter() - started) * 1000)
            _logger.warning(
                "generation failed, serving the retrieved passages instead",
                extra={"code": exc.code.value, "trace_id": trace_id, "mode": EXTRACTIVE_MODE},
            )
            return Answer(
                answer=_extractive_answer(context),
                citations=list(context.citations),
                mode=EXTRACTIVE_MODE,
                trace_id=trace_id,
                timings_ms=timings,
            )

        timings["generate"] = int((time.perf_counter() - started) * 1000)
        return Answer(
            answer=completion.text,
            citations=resolve_citations(completion.text, context.citations),
            mode=mode,
            trace_id=trace_id,
            prompt_tokens=completion.prompt_tokens,
            completion_tokens=completion.completion_tokens,
            timings_ms=timings,
        )

    async def stream(
        self,
        question: str,
        *,
        collection: str | None = None,
        top_k: int | None = None,
        filters: Mapping[str, Any] | None = None,
    ) -> AsyncIterator[QueryEvent]:
        """Answer a question as a stream of events in the documented order."""
        trace_id = current_trace_id()
        context, prompt, mode, timings = await self._prepare(
            question, collection=collection, top_k=top_k, filters=filters
        )

        yield QueryEvent(
            type="meta",
            data={"trace_id": trace_id, "mode": mode, "degraded": mode != FULL_MODE},
        )

        started = time.perf_counter()
        parts: list[str] = []
        try:
            async for delta in self.llm.stream(prompt, system=P1_SYSTEM_PROMPT):
                parts.append(delta)
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

        yield QueryEvent(
            type="citations",
            data={
                "citations": [
                    citation.as_dict() for citation in resolve_citations(answer, context.citations)
                ]
            },
        )
        yield QueryEvent(
            type="usage",
            data={
                "usage": {
                    "prompt_tokens": self.counter.count(prompt),
                    "completion_tokens": self.counter.count(answer),
                },
                "timings_ms": timings,
                "template_version": P1_TEMPLATE_VERSION,
            },
        )
        yield QueryEvent(type="done", data={})

    async def close(self) -> None:
        """Release the generation provider."""
        await self.llm.close()
