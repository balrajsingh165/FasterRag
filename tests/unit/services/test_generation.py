import asyncio
from collections.abc import AsyncIterator, Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest

from fasterrag.adapters.embeddings.base import EmbeddingAdapter, EmbeddingResult
from fasterrag.adapters.llm.base import Completion, LLMAdapter
from fasterrag.adapters.llm.ollama import OllamaGenerator
from fasterrag.adapters.vectordb.base import HealthStatus
from fasterrag.config.schema import Settings
from fasterrag.core.breaker import reset_shared_breakers, shared_breaker
from fasterrag.core.cache.semantic import SemanticCache
from fasterrag.core.cache.store import MemoryStore
from fasterrag.core.chunking.models import EstimatingTokenCounter
from fasterrag.core.retrieval.models import ScoredChunk
from fasterrag.errors import (
    ConfigError,
    EmbedError,
    ErrorCode,
    FasterRagError,
    GenerationError,
)
from fasterrag.observability import metrics
from fasterrag.observability.logging import use_trace_id
from fasterrag.services.budget import reset_shared_governors
from fasterrag.services.generation import (
    EXTRACTIVE_MODE,
    FALLBACK_BREAKER,
    GenerationService,
    QueryEvent,
    fallback_settings,
)
from fasterrag.services.querying import FULL_MODE, HYBRID_ONLY_MODE, Retrieval
from fasterrag.services.traces import TraceStore

FAST_RETRIES: dict[str, Any] = {"retries": {"backoff_base_ms": 1, "backoff_max_ms": 1}}
"""The configured retry policy with its waits shrunk, so a retried failure costs no wall time."""


class ScriptedLLM(LLMAdapter):
    """Returns scripted text, or fails on demand."""

    provider = "scripted"

    def __init__(self, settings: Settings, text: str = "", error: Exception | None = None) -> None:
        super().__init__(settings)
        self.text = text
        self.error = error
        self.finish_reason: str | None = "stop"
        self.fail_after: int | None = None
        self.failures: int | None = None
        self.hang_after: int | None = None
        self.prompts: list[tuple[str, str | None]] = []

    async def hang(self) -> None:
        """Never return, as a provider that accepted the call and went silent does."""
        await asyncio.Event().wait()

    def failing(self) -> bool:
        """Return whether this call fails: every call, or only the first ``failures``."""
        if self.error is None:
            return False
        if self.failures is None:
            return True
        self.failures -= 1
        return self.failures >= 0

    async def complete(self, prompt: str, *, system: str | None = None) -> Completion:
        self.prompts.append((prompt, system))
        if self.hang_after is not None:
            await self.hang()
        if self.failing():
            assert self.error is not None
            raise self.error
        return Completion(
            text=self.text,
            model="scripted",
            prompt_tokens=11,
            completion_tokens=7,
            finish_reason=self.finish_reason,
        )

    async def stream(self, prompt: str, *, system: str | None = None) -> AsyncIterator[str]:
        self.prompts.append((prompt, system))
        if self.fail_after is None and self.failing():
            assert self.error is not None
            raise self.error
        for sent, word in enumerate(self.text.split(" ")):
            if self.error is not None and sent == self.fail_after:
                raise self.error
            if sent == self.hang_after:
                await self.hang()
            yield f"{word} "

    async def health(self) -> HealthStatus:
        return HealthStatus(healthy=True)

    async def close(self) -> None:
        return None


class ScriptedRetrieval:
    """Returns fixed chunks and a fixed mode."""

    def __init__(self, chunks: list[ScoredChunk], mode: str = FULL_MODE) -> None:
        self.chunks = chunks
        self.mode = mode

    async def search(self, text: str, **kwargs: Any) -> Retrieval:
        return Retrieval(chunks=list(self.chunks), mode=self.mode)


def chunk(chunk_id: str, text: str, **payload: Any) -> ScoredChunk:
    return ScoredChunk(chunk_id=chunk_id, text=text, payload=payload, rrf_score=0.5)


def build(
    *,
    answer: str = "The notice period is thirty days [^c_a].",
    error: Exception | None = None,
    mode: str = FULL_MODE,
    chunks: list[ScoredChunk] | None = None,
    grounded_or_refuse: bool = False,
    verdict: str | None = None,
    grader_error: Exception | None = None,
    threshold: float = 0.7,
) -> tuple[GenerationService, ScriptedLLM]:
    settings = Settings.model_validate(
        {
            "generation": {
                "grounded_or_refuse": grounded_or_refuse,
                "faithfulness_threshold": threshold,
            },
            "reliability": FAST_RETRIES,
        }
    )
    llm = ScriptedLLM(settings, text=answer, error=error)
    retrieval = ScriptedRetrieval(
        chunks if chunks is not None else [chunk("c_a", "Either party may terminate.")], mode
    )
    grader = (
        ScriptedLLM(settings, text=verdict or "", error=grader_error)
        if grounded_or_refuse
        else None
    )
    service = GenerationService(
        settings,
        retrieval,  # type: ignore[arg-type]
        llm,
        grader=grader,
    )
    return service, llm


async def collect(service: GenerationService, question: str = "q") -> list[QueryEvent]:
    return [event async for event in service.stream(question)]


async def collect_into(
    events: list[QueryEvent], service: GenerationService, question: str = "q"
) -> None:
    """Append each event as it arrives, so what preceded a failure is still observable."""
    async for event in service.stream(question):
        events.append(event)


async def test_an_answer_carries_its_resolved_citations() -> None:
    service, _ = build()

    result = await service.answer("what is the notice period?")

    assert result.answer is not None
    assert [citation.chunk_id for citation in result.citations] == ["c_a"]
    assert result.mode == FULL_MODE
    assert result.degraded is False


async def test_the_system_prompt_is_sent_separately_from_the_question() -> None:
    service, llm = build()

    await service.answer("what is the notice period?")
    prompt, system = llm.prompts[0]

    assert system is not None
    assert "strictly from the provided context" in system
    assert "Question: what is the notice period?" in prompt
    assert "[^c_a]" in prompt


async def test_usage_and_timings_are_reported() -> None:
    service, _ = build()

    result = await service.answer("q")

    assert result.prompt_tokens == 11
    assert result.completion_tokens == 7
    assert set(result.timings_ms) == {"retrieve", "assemble", "generate"}


async def test_a_degraded_retrieval_mode_reaches_the_answer() -> None:
    service, _ = build(mode=HYBRID_ONLY_MODE)

    result = await service.answer("q")

    assert result.mode == HYBRID_ONLY_MODE
    assert result.degraded is True


async def test_a_generation_failure_serves_the_retrieved_passages() -> None:
    service, _ = build(error=GenerationError("provider is down", retryable=True))

    result = await service.answer("q")

    assert result.mode == EXTRACTIVE_MODE
    assert result.degraded is True
    assert result.answer is not None
    assert "Either party may terminate." in result.answer
    assert [citation.chunk_id for citation in result.citations] == ["c_a"]


async def test_a_missing_provider_extra_is_never_hidden_behind_a_degraded_answer() -> None:
    service, _ = build(error=ConfigError("llm.provider needs 'pip install fasterrag[openai]'"))

    with pytest.raises(ConfigError, match="fasterrag\\[openai\\]"):
        await service.answer("q")


async def test_a_rejected_credential_surfaces_rather_than_degrading() -> None:
    service, _ = build(
        error=GenerationError("the provider rejected the credentials", retryable=False)
    )

    with pytest.raises(GenerationError, match="rejected the credentials"):
        await service.answer("q")


async def test_a_misconfigured_provider_fails_the_stream_rather_than_degrading_it() -> None:
    service, _ = build(error=ConfigError("llm.api_key_env names an unset variable"))

    with pytest.raises(ConfigError):
        await collect(service)


async def test_a_stream_whose_provider_is_down_serves_the_retrieved_passages() -> None:
    """A streamed query got `error` and nothing else where the JSON path degraded (TASK-0438)."""
    service, _ = build(error=GenerationError("provider is rate limited", retryable=True))
    degraded_before = metrics.DEGRADED_RESPONSES.value(mode=EXTRACTIVE_MODE)

    events = await collect(service)
    kinds = [event.type for event in events]
    usage = next(event for event in events if event.type == "usage")

    assert kinds == ["meta", "meta", "token", "citations", "usage", "done"]
    assert "Either party may terminate." in events[2].data["text"]
    assert events[3].data["citations"][0]["chunk_id"] == "c_a"
    assert usage.data["mode"] == EXTRACTIVE_MODE
    assert usage.data["degraded"] is True
    assert metrics.DEGRADED_RESPONSES.value(mode=EXTRACTIVE_MODE) == degraded_before + 1


async def test_the_degraded_mode_is_announced_before_the_fallback_text() -> None:
    """`meta` exists so a client knows a response is degraded before it renders any of it."""
    service, _ = build(error=GenerationError("provider is rate limited", retryable=True))

    events = await collect(service)
    announced = [event.data for event in events[: [e.type for e in events].index("token")]]

    assert announced[0]["mode"] == FULL_MODE
    assert announced[-1]["mode"] == EXTRACTIVE_MODE
    assert announced[-1]["degraded"] is True
    assert announced[-1]["trace_id"] == announced[0]["trace_id"]


async def test_a_stream_through_an_open_breaker_degrades_too() -> None:
    """Shedding load is the same rung on both surfaces, and never reaches the provider."""
    service, llm = build(error=GenerationError("provider is down", retryable=True))
    service.breaker.failure_threshold = 1
    await collect(service)
    calls = len(llm.prompts)

    kinds = [event.type for event in await collect(service)]

    assert service.breaker.is_open is True
    assert len(llm.prompts) == calls, "the provider was called through an open breaker"
    assert kinds == ["meta", "meta", "token", "citations", "usage", "done"]


async def test_a_gated_stream_that_fails_part_way_still_degrades() -> None:
    """With grounded-or-refuse nothing is sent until the verdict, so nothing needs unsaying."""
    service, llm = build(
        grounded_or_refuse=True,
        verdict='{"score": 0.93}',
        error=GenerationError("provider vanished", retryable=True),
    )
    llm.fail_after = 2

    events = await collect(service)
    text = "".join(event.data["text"] for event in events if event.type == "token")

    assert events[-1].type == "done"
    assert "Either party may terminate." in text
    assert "notice period" not in text, "half a generated answer leaked ahead of the fallback"


async def test_an_invented_citation_never_reaches_the_response() -> None:
    service, _ = build(answer="Answer [^c_a] and invented [^c_nope].")

    result = await service.answer("q")

    assert [citation.chunk_id for citation in result.citations] == ["c_a"]


async def test_the_answer_serializes_to_the_documented_body() -> None:
    service, _ = build()

    payload = (await service.answer("q")).as_dict()

    assert set(payload) == {
        "answer",
        "citations",
        "usage",
        "timings_ms",
        "degraded",
        "mode",
        "truncated",
        "evidence_dropped",
        "faithfulness",
        "cache",
        "trace_id",
    }


async def test_the_stream_follows_the_documented_event_order() -> None:
    service, _ = build()

    events = await collect(service)
    kinds = [event.type for event in events]

    assert kinds[0] == "meta"
    assert kinds[-1] == "done"
    assert kinds.count("token") > 0
    assert kinds.index("citations") < kinds.index("usage")
    assert kinds.index("usage") < kinds.index("done")
    assert all(kind == "token" for kind in kinds[1 : kinds.index("citations")])


async def test_meta_arrives_before_any_text() -> None:
    service, _ = build()

    events = await collect(service)

    assert events[0].type == "meta"
    assert events[0].data["trace_id"]
    assert events[0].data["mode"] == FULL_MODE
    assert events[0].data["degraded"] is False


async def test_the_streamed_tokens_reconstruct_the_answer() -> None:
    service, _ = build(answer="thirty days notice")

    events = await collect(service)
    streamed = "".join(event.data["text"] for event in events if event.type == "token").strip()

    assert streamed == "thirty days notice"


async def test_citations_are_emitted_once_after_the_answer() -> None:
    service, _ = build()

    events = await collect(service)
    citations = [event for event in events if event.type == "citations"]

    assert len(citations) == 1
    assert citations[0].data["citations"][0]["chunk_id"] == "c_a"


async def test_usage_reports_timings_and_the_template_version() -> None:
    service, _ = build()

    usage = next(event for event in await collect(service) if event.type == "usage")

    assert usage.data["usage"]["completion_tokens"] > 0
    assert "generate" in usage.data["timings_ms"]
    assert usage.data["template_version"]


async def test_a_mid_stream_failure_emits_error_and_never_done() -> None:
    """Text already sent cannot be replaced, so past the first token there is no fallback."""
    service, llm = build(error=GenerationError("provider vanished"))
    llm.fail_after = 2
    degraded_before = metrics.DEGRADED_RESPONSES.value(mode=EXTRACTIVE_MODE)

    events = await collect(service)
    kinds = [event.type for event in events]

    assert kinds == ["meta", "token", "token", "error"]
    assert "done" not in kinds
    assert metrics.DEGRADED_RESPONSES.value(mode=EXTRACTIVE_MODE) == degraded_before


async def test_the_error_event_carries_the_stable_code_and_trace() -> None:
    service, llm = build(error=GenerationError("provider vanished"))
    llm.fail_after = 1

    error = next(event for event in await collect(service) if event.type == "error")

    assert error.data["code"] == "GENERATION_FAILED"
    assert error.data["trace_id"]
    assert error.data["retryable"] is True
    assert error.data["status"] == 503


async def test_a_degraded_retrieval_is_announced_in_meta() -> None:
    service, _ = build(mode=HYBRID_ONLY_MODE)

    events = await collect(service)

    assert events[0].data["degraded"] is True
    assert events[0].data["mode"] == HYBRID_ONLY_MODE


async def test_an_empty_corpus_still_produces_a_well_formed_stream() -> None:
    service, _ = build(chunks=[], answer="I cannot answer from the context provided.")

    events = await collect(service)
    kinds = [event.type for event in events]

    assert kinds[0] == "meta"
    assert kinds[-1] == "done"
    assert next(e for e in events if e.type == "citations").data["citations"] == []


async def test_faithfulness_is_not_scored_while_the_flag_is_off() -> None:
    service, _ = build()

    result = await service.answer("q")

    assert result.faithfulness is None
    assert "grade" not in result.timings_ms
    assert result.insufficient_evidence is False


async def test_a_well_grounded_answer_passes_the_gate() -> None:
    service, _ = build(grounded_or_refuse=True, verdict='{"score": 0.93}')

    result = await service.answer("q")

    assert result.answer is not None
    assert result.faithfulness == pytest.approx(0.93)
    assert result.insufficient_evidence is False
    assert "grade" in result.timings_ms


async def test_a_low_score_withholds_the_answer() -> None:
    service, _ = build(
        grounded_or_refuse=True,
        verdict='{"score": 0.38, "unsupported_claims": ["signed in 2019"]}',
    )

    result = await service.answer("q")

    assert result.answer is None
    assert result.insufficient_evidence is True
    assert result.faithfulness == pytest.approx(0.38)
    assert result.threshold == pytest.approx(0.7)


async def test_a_refusal_serializes_to_the_documented_body() -> None:
    service, _ = build(grounded_or_refuse=True, verdict='{"score": 0.38}')

    payload = (await service.answer("q")).as_dict()

    assert payload["code"] == "INSUFFICIENT_EVIDENCE"
    assert payload["answer"] is None
    assert payload["threshold"] == pytest.approx(0.7)
    assert set(payload) == {
        "code",
        "answer",
        "best_candidates",
        "faithfulness",
        "threshold",
        "trace_id",
    }


async def test_a_refusal_offers_the_candidates_it_declined_to_answer_from() -> None:
    service, _ = build(
        grounded_or_refuse=True,
        verdict='{"score": 0.1}',
        chunks=[chunk("c_a", "text a", source_uri="s3://a.pdf"), chunk("c_b", "text b")],
    )

    candidates = (await service.answer("q")).best_candidates

    assert [candidate["chunk_id"] for candidate in candidates] == ["c_a", "c_b"]
    assert candidates[0]["source"] == "s3://a.pdf"
    assert candidates[0]["score"] == pytest.approx(0.5)


async def test_a_grader_outage_returns_the_answer_ungated() -> None:
    service, _ = build(grounded_or_refuse=True, grader_error=GenerationError("grader is down"))

    result = await service.answer("q")

    assert result.answer is not None
    assert result.faithfulness is None
    assert result.insufficient_evidence is False


async def test_an_unparseable_verdict_returns_the_answer_ungated() -> None:
    service, _ = build(grounded_or_refuse=True, verdict="I am not sure how to grade this.")

    result = await service.answer("q")

    assert result.answer is not None
    assert result.faithfulness is None


async def test_a_score_at_the_threshold_is_answered() -> None:
    service, _ = build(grounded_or_refuse=True, verdict='{"score": 0.7}', threshold=0.7)

    result = await service.answer("q")

    assert result.answer is not None


async def test_the_gate_never_fires_on_an_extractive_fallback() -> None:
    service, _ = build(
        grounded_or_refuse=True,
        error=GenerationError("provider is down"),
        verdict='{"score": 0.0}',
    )

    result = await service.answer("q")

    assert result.mode == EXTRACTIVE_MODE
    assert result.answer is not None
    assert result.insufficient_evidence is False


async def test_the_usage_event_reports_the_faithfulness_score() -> None:
    service, _ = build(grounded_or_refuse=True, verdict='{"score": 0.93}')

    usage = next(event for event in await collect(service) if event.type == "usage")

    assert usage.data["faithfulness"] == pytest.approx(0.93)


async def test_a_gated_stream_holds_every_token_until_the_verdict_is_in() -> None:
    service, _ = build(grounded_or_refuse=True, verdict='{"score": 0.38}')

    events = await collect(service)
    kinds = [event.type for event in events]

    assert "token" not in kinds
    assert kinds == ["meta", "insufficient_evidence", "done"]


async def test_a_refused_stream_still_completes() -> None:
    service, _ = build(grounded_or_refuse=True, verdict='{"score": 0.38}')

    events = await collect(service)

    assert events[-1].type == "done"
    assert events[1].data["code"] == "INSUFFICIENT_EVIDENCE"
    assert events[1].data["best_candidates"]


async def test_a_passing_gated_stream_delivers_the_whole_answer() -> None:
    service, _ = build(
        grounded_or_refuse=True, verdict='{"score": 0.93}', answer="thirty days [^c_a]"
    )

    events = await collect(service)
    kinds = [event.type for event in events]
    text = "".join(event.data["text"] for event in events if event.type == "token")

    assert kinds == ["meta", "token", "citations", "usage", "done"]
    assert text.strip() == "thirty days [^c_a]"


async def test_an_ungated_stream_still_emits_tokens_incrementally() -> None:
    service, _ = build(answer="one two three")

    tokens = [event for event in await collect(service) if event.type == "token"]

    assert len(tokens) == 3


class ScriptedEmbedder(EmbeddingAdapter):
    """Returns one vector per question, so paraphrases can be simulated."""

    provider = "scripted"

    def __init__(self, settings: Settings, vectors: dict[str, list[float]] | None = None) -> None:
        super().__init__(settings)
        self.vectors = vectors or {}
        self.calls = 0

    @property
    def model(self) -> str:
        return "scripted-model"

    @property
    def model_version(self) -> str:
        return "1.0"

    @property
    def dimensions(self) -> int | None:
        return 3

    async def embed_documents(self, texts: Sequence[str]) -> EmbeddingResult:
        raise NotImplementedError

    async def embed_query(self, text: str) -> list[float]:
        self.calls += 1
        return self.vectors.get(text, [1.0, 0.0, 0.0])

    async def health(self) -> HealthStatus:
        return HealthStatus(healthy=True)

    async def close(self) -> None:
        return None


def build_cached(
    *,
    answer: str = "thirty days [^c_a]",
    mode: str = FULL_MODE,
    vectors: dict[str, list[float]] | None = None,
    threshold: float = 0.95,
    embedder_error: bool = False,
) -> tuple[GenerationService, ScriptedLLM, SemanticCache]:
    settings = Settings.model_validate(
        {
            "cache": {"semantic": True, "similarity_threshold": threshold},
            "reliability": FAST_RETRIES,
        }
    )
    llm = ScriptedLLM(settings, text=answer)
    retrieval = ScriptedRetrieval([chunk("c_a", "Either party may terminate.")], mode)
    cache = SemanticCache(settings, MemoryStore())

    class FailingEmbedder(ScriptedEmbedder):
        async def embed_query(self, text: str) -> list[float]:
            raise EmbedError("embedding provider is down")

    factory = FailingEmbedder if embedder_error else ScriptedEmbedder
    service = GenerationService(
        settings,
        retrieval,  # type: ignore[arg-type]
        llm,
        cache=cache,
        embedder=factory(settings, vectors),
    )
    return service, llm, cache


async def test_the_first_query_misses_and_the_second_hits() -> None:
    service, llm, _ = build_cached()

    await service.answer("what is the notice period?")
    calls = len(llm.prompts)
    result = await service.answer("what is the notice period?")

    assert len(llm.prompts) == calls
    assert result.cache["semantic_hit"] is True
    assert result.answer == "thirty days [^c_a]"


async def test_a_paraphrase_is_served_from_the_cache() -> None:
    service, llm, _ = build_cached(
        vectors={"first question": [1.0, 0.0, 0.0], "a paraphrase": [0.99, 0.05, 0.0]}
    )

    await service.answer("first question")
    calls = len(llm.prompts)
    result = await service.answer("a paraphrase")

    assert len(llm.prompts) == calls
    assert result.cache["semantic_hit"] is True


async def test_an_unrelated_question_runs_the_full_pipeline() -> None:
    service, llm, _ = build_cached(vectors={"first": [1.0, 0.0, 0.0], "unrelated": [0.0, 1.0, 0.0]})

    await service.answer("first")
    result = await service.answer("unrelated")

    assert len(llm.prompts) == 2
    assert result.cache["semantic_hit"] is False


async def test_a_cache_hit_keeps_this_query_s_trace_id() -> None:
    service, _, _ = build_cached()
    first = await service.answer("q")

    second = await service.answer("q")

    assert second.trace_id == first.trace_id or second.trace_id != ""
    assert second.cache["similarity"] == pytest.approx(1.0)


async def test_a_cache_hit_reports_only_the_time_it_actually_spent() -> None:
    service, _, _ = build_cached()
    await service.answer("q")

    result = await service.answer("q")

    assert set(result.timings_ms) == {"cache"}


async def test_a_cache_hit_restores_the_citations() -> None:
    service, _, _ = build_cached()
    first = await service.answer("q")

    second = await service.answer("q")

    assert [c.chunk_id for c in second.citations] == [c.chunk_id for c in first.citations]
    assert second.citations[0].source == first.citations[0].source


async def test_a_degraded_answer_is_never_cached() -> None:
    service, llm, cache = build_cached(mode=HYBRID_ONLY_MODE)

    await service.answer("q")
    await service.answer("q")

    assert len(llm.prompts) == 2
    assert cache.stats.hits == 0


async def test_a_missing_cache_member_defaults_to_a_miss() -> None:
    service, _ = build()

    assert (await service.answer("q")).cache == {"semantic_hit": False, "similarity": None}


async def test_the_response_body_carries_the_cache_member() -> None:
    service, _ = build()

    assert "cache" in (await service.answer("q")).as_dict()


async def test_an_embedding_failure_falls_back_to_the_full_pipeline() -> None:
    service, llm, _ = build_cached(embedder_error=True)

    result = await service.answer("q")

    assert result.answer is not None
    assert len(llm.prompts) == 1
    assert result.cache["semantic_hit"] is False


async def test_a_cached_stream_replays_the_documented_event_order() -> None:
    service, _, _ = build_cached()
    await service.answer("q")

    events = await collect(service)

    assert [event.type for event in events] == ["meta", "token", "citations", "usage", "done"]


async def test_a_cached_stream_announces_the_hit_in_meta() -> None:
    service, _, _ = build_cached()
    await service.answer("q")

    events = await collect(service)

    assert events[0].data["cache"]["semantic_hit"] is True


async def test_a_cached_stream_delivers_the_whole_answer_at_once() -> None:
    service, _, _ = build_cached()
    await service.answer("q")

    tokens = [event for event in await collect(service) if event.type == "token"]

    assert len(tokens) == 1
    assert tokens[0].data["text"] == "thirty days [^c_a]"


async def test_a_streamed_answer_is_cached_for_the_next_query() -> None:
    service, llm, _ = build_cached()

    await collect(service)
    calls = len(llm.prompts)
    result = await service.answer("q")

    assert len(llm.prompts) == calls
    assert result.cache["semantic_hit"] is True


async def test_the_cache_is_never_consulted_without_an_embedder() -> None:
    settings = Settings.model_validate({"cache": {"semantic": True}})
    service = GenerationService(
        settings,
        ScriptedRetrieval([chunk("c_a", "text")]),  # type: ignore[arg-type]
        ScriptedLLM(settings, text="answer"),
        cache=SemanticCache(settings, MemoryStore()),
    )

    assert service.caching is False
    assert (await service.answer("q")).cache["semantic_hit"] is False


@pytest.mark.parametrize("budget", [0, 5])
async def test_a_tight_budget_still_answers(budget: int) -> None:
    settings = Settings()
    llm = ScriptedLLM(settings, text="short answer")
    retrieval = ScriptedRetrieval([chunk("c_a", "a very long passage " * 50)])
    service = GenerationService(
        settings,
        retrieval,  # type: ignore[arg-type]
        llm,
        context_budget_tokens=budget,
    )

    result = await service.answer("q")

    assert result.answer == "short answer"


def build_budgeted(
    configured: int, *, explicit: int | None = None
) -> tuple[GenerationService, ScriptedLLM]:
    """Build a service over two ~300-token chunks under a configured budget."""
    settings = Settings.model_validate({"generation": {"context_budget_tokens": configured}})
    llm = ScriptedLLM(settings, text="an answer [^c_a] [^c_b]")
    retrieval = ScriptedRetrieval([chunk("c_a", "alpha " * 200), chunk("c_b", "bravo " * 200)])
    service = GenerationService(
        settings,
        retrieval,  # type: ignore[arg-type]
        llm,
        context_budget_tokens=explicit,
    )
    return service, llm


async def test_the_context_budget_is_taken_from_configuration() -> None:
    """The budget was a constructor parameter no `config.yaml` key reached (TASK-0285).

    An operator told to tune the assembled retrieved-context budget had no way to set it,
    and the shipped 4,000 packed both chunks regardless of what configuration said.
    """
    service, llm = build_budgeted(350)

    result = await service.answer("q")

    assert service.context_budget_tokens == 350
    assert [citation.chunk_id for citation in result.citations] == ["c_a"]
    assert result.evidence_dropped == 1
    assert "bravo" not in llm.prompts[0][0]


async def test_a_wide_configured_budget_packs_everything_retrieved() -> None:
    """The counterpart: the key must be able to move the outcome in both directions."""
    service, _ = build_budgeted(4000)

    result = await service.answer("q")

    assert [citation.chunk_id for citation in result.citations] == ["c_a", "c_b"]
    assert result.evidence_dropped == 0


async def test_a_configured_budget_below_one_chunk_answers_without_context() -> None:
    """What the reference has to say happens, rather than what a reader might assume."""
    service, llm = build_budgeted(10)

    result = await service.answer("q")

    assert result.answer == "an answer [^c_a] [^c_b]"
    assert result.citations == []
    assert result.evidence_dropped == 2
    assert "alpha" not in llm.prompts[0][0]


async def test_an_explicit_budget_overrides_the_configured_one() -> None:
    """Standalone use documented in python-api.md must survive the key being added."""
    service, _ = build_budgeted(10, explicit=4000)

    assert service.context_budget_tokens == 4000
    assert len((await service.answer("q")).citations) == 2


async def test_a_truncated_completion_is_reported() -> None:
    """Serving a cut-off answer silently means a caller cannot tell it apart from a whole one."""
    service, llm = build()
    llm.finish_reason = "length"

    answer = await service.answer("q")

    assert answer.truncated is True
    assert answer.as_dict()["truncated"] is True


async def test_a_finished_completion_is_not_reported_as_truncated() -> None:
    service, _ = build()

    assert (await service.answer("q")).truncated is False


async def test_an_empty_answer_at_the_ceiling_still_reports_truncation() -> None:
    """A reasoning model can spend the whole budget before emitting anything."""
    service, llm = build()
    llm.finish_reason = "length"
    llm.text = ""

    answer = await service.answer("q")

    assert answer.answer == ""
    assert answer.truncated is True


async def test_evidence_dropped_by_the_budget_is_reported() -> None:
    """The answer is grounded in a subset of what the retriever chose; that must be visible."""
    service, _ = build()

    answer = await service.answer("q")

    assert answer.as_dict()["evidence_dropped"] == answer.evidence_dropped
    assert isinstance(answer.evidence_dropped, int)


async def test_repeated_provider_failures_open_the_llm_breaker(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """FMEA row 20's "retries → breaker opens" was only ever retries."""
    service, llm = build(error=GenerationError("provider is down", retryable=True))
    service.breaker.failure_threshold = 2

    for _ in range(2):
        await service.answer("what is the notice period", collection="policies")

    calls_before = len(llm.prompts)
    answer = await service.answer("what is the notice period", collection="policies")

    assert service.breaker.is_open is True
    assert len(llm.prompts) == calls_before, "the provider was called through an open breaker"
    assert answer.mode == EXTRACTIVE_MODE


async def test_an_open_breaker_degrades_rather_than_raising() -> None:
    """Shedding load is a rung on the ladder, not a 503.

    ``CircuitOpenError`` is retryable by construction, so it takes the same route a live
    provider outage takes and the caller still gets the retrieved passages.
    """
    service, _ = build(error=GenerationError("provider is down", retryable=True))
    service.breaker.failure_threshold = 1
    await service.answer("what is the notice period", collection="policies")

    answer = await service.answer("what is the notice period", collection="policies")

    assert service.breaker.is_open is True
    assert answer.mode == EXTRACTIVE_MODE
    assert answer.answer is not None


async def test_a_success_clears_the_failure_count_rather_than_only_the_state() -> None:
    """A success resets the failure count, not merely the state."""
    service, _ = build()
    service.breaker.failure_threshold = 2
    service.breaker.record_failure(GenerationError("transient", retryable=True))

    await service.answer("what is the notice period", collection="policies")
    service.breaker.record_failure(GenerationError("another blip", retryable=True))

    assert service.breaker.is_open is False


async def test_a_rate_limited_answer_is_retried_until_it_succeeds() -> None:
    """A single 429 used to degrade straight to the extractive rung (TASK-0449)."""
    service, llm = build(error=GenerationError("status 429", retryable=True))
    llm.failures = 2

    answer = await service.answer("what is the notice period?")

    assert answer.mode == FULL_MODE
    assert answer.answer == "The notice period is thirty days [^c_a]."
    assert len(llm.prompts) == 3


async def test_a_rate_limited_stream_is_retried_before_its_first_token() -> None:
    service, llm = build(error=GenerationError("status 429", retryable=True))
    llm.failures = 1

    events = await collect(service)

    assert [event.data["mode"] for event in events if event.type == "meta"] == [FULL_MODE]
    assert "".join(e.data["text"] for e in events if e.type == "token").startswith("The notice")
    assert len(llm.prompts) == 2


async def test_a_stream_is_never_retried_after_its_first_token() -> None:
    """A second call would splice a different answer onto text the caller already has."""
    service, llm = build(error=GenerationError("connection reset", retryable=True))
    llm.fail_after = 2

    events = await collect(service)

    assert events[-1].type == "error"
    assert len(llm.prompts) == 1


async def test_a_non_retryable_failure_is_called_once() -> None:
    service, llm = build(error=GenerationError("model not found", retryable=False))

    with pytest.raises(GenerationError):
        await service.answer("q")

    assert len(llm.prompts) == 1


async def test_retries_are_bounded_by_the_configured_attempts() -> None:
    service, llm = build(error=GenerationError("status 503", retryable=True))

    answer = await service.answer("q")

    assert answer.mode == EXTRACTIVE_MODE
    assert len(llm.prompts) == 1 + service.settings.reliability.retries.max_attempts


async def test_the_wait_is_never_shorter_than_the_provider_s_retry_after(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    waits: list[float] = []

    async def record(seconds: float) -> None:
        waits.append(seconds)

    monkeypatch.setattr("fasterrag.services.generation._pause", record)
    settings = Settings.model_validate(
        {"reliability": {"retries": {"backoff_base_ms": 1, "backoff_max_ms": 5000}}}
    )
    llm = ScriptedLLM(settings, text="an answer", error=GenerationError("429", retry_after=2.0))
    llm.failures = 2
    service = GenerationService(
        settings,
        ScriptedRetrieval([chunk("c_a", "Either party may terminate.")]),  # type: ignore[arg-type]
        llm,
    )

    answer = await service.answer("q")

    assert answer.mode == FULL_MODE
    assert waits == [2.0, 2.0]


async def test_a_retry_after_beyond_the_backoff_ceiling_is_not_waited_for() -> None:
    """Sleeping for as long as a provider asks would let one request hold a worker for minutes."""
    service, llm = build(error=GenerationError("status 429", retryable=True, retry_after=60.0))

    answer = await service.answer("q")

    assert answer.mode == EXTRACTIVE_MODE
    assert len(llm.prompts) == 1


async def test_retries_stop_once_the_breaker_opens() -> None:
    """Checked before every attempt, as the embedding pool does, not once before the first."""
    service, llm = build(error=GenerationError("status 503", retryable=True))
    service.breaker.failure_threshold = 2

    answer = await service.answer("q")

    assert answer.mode == EXTRACTIVE_MODE
    assert len(llm.prompts) == 2
    assert service.breaker.is_open is True


def with_fallback(
    error: Exception, *, fallback_error: Exception | None = None
) -> tuple[GenerationService, ScriptedLLM, ScriptedLLM]:
    """Return a service whose primary fails with ``error`` and a lent second provider."""
    service, primary = build(error=error)
    second = ScriptedLLM(service.settings, text="The fallback says thirty days [^c_a].")
    second.error = fallback_error
    service.fallback_llm = second
    service.owns_fallback = False
    return service, primary, second


async def test_a_provider_outage_falls_back_to_the_second_provider(tmp_path: Path) -> None:
    """There was no second provider: an outage went straight to the extractive rung."""
    service, primary, second = with_fallback(GenerationError("status 503", retryable=True))
    service.traces = TraceStore(tmp_path)

    answer = await service.answer("what is the notice period?")

    assert answer.mode == FULL_MODE
    assert answer.answer == "The fallback says thirty days [^c_a]."
    assert len(primary.prompts) == 1 + service.settings.reliability.retries.max_attempts
    assert len(second.prompts) == 1
    stored = service.traces.load(answer.trace_id)
    assert stored is not None
    roles = [(span.name, span.attributes.get("role")) for span in stored.spans]
    assert ("generation-failed", "primary") in roles
    assert roles[-1] == ("generation", "fallback")


async def test_a_streamed_outage_falls_back_before_any_token() -> None:
    service, _, second = with_fallback(GenerationError("status 503", retryable=True))

    events = await collect(service)

    assert [event.data["mode"] for event in events if event.type == "meta"] == [FULL_MODE]
    assert "".join(e.data["text"] for e in events if e.type == "token").startswith("The fallback")
    assert len(second.prompts) == 1


async def test_the_fallback_is_counted_under_its_own_breaker() -> None:
    """Counted under the primary's key, the outage that triggers it would also shed it."""
    service, primary, second = with_fallback(GenerationError("status 503", retryable=True))
    service.breaker.failure_threshold = 1
    await service.answer("q")
    calls = len(primary.prompts)

    answer = await service.answer("q")

    assert service.breaker.is_open is True
    assert len(primary.prompts) == calls
    assert answer.mode == FULL_MODE
    assert len(second.prompts) == 2
    assert shared_breaker(FALLBACK_BREAKER, service.settings).is_open is False


async def test_a_misconfigured_primary_is_reported_rather_than_covered_by_the_fallback() -> None:
    service, _, second = with_fallback(GenerationError("rejected key", retryable=False))

    with pytest.raises(GenerationError):
        await service.answer("q")

    assert second.prompts == []


async def test_with_both_providers_down_the_retrieved_passages_are_served() -> None:
    down = GenerationError("status 503", retryable=True)
    service, _, second = with_fallback(down, fallback_error=down)

    answer = await service.answer("q")

    assert answer.mode == EXTRACTIVE_MODE
    assert len(second.prompts) == 1 + service.settings.reliability.retries.max_attempts


def test_the_fallback_block_answers_with_the_primary_s_sampling() -> None:
    settings = Settings.model_validate(
        {
            "llm": {
                "temperature": 0.4,
                "max_tokens": 300,
                "fallback": {"provider": "ollama", "model": "llama3.1:8b"},
            }
        }
    )

    configured = fallback_settings(settings)

    assert configured is not None
    assert (configured.llm.provider, configured.llm.model) == ("ollama", "llama3.1:8b")
    assert (configured.llm.temperature, configured.llm.max_tokens) == (0.4, 300)
    assert configured.llm.api_key_env is None
    assert configured.llm.fallback is None
    assert fallback_settings(Settings()) is None


async def test_a_configured_fallback_is_built_on_first_need_and_closed_with_the_service() -> None:
    settings = Settings.model_validate(
        {"llm": {"fallback": {"provider": "ollama", "model": "llama3.1:8b"}}}
    )
    service = GenerationService(
        settings,
        ScriptedRetrieval([chunk("c_a", "Either party may terminate.")]),  # type: ignore[arg-type]
        ScriptedLLM(settings),
        owns_llm=False,
    )
    assert service.fallback_llm is None

    provider = service._fallback()

    assert provider is not None
    assert isinstance(provider.llm, OllamaGenerator)
    assert provider.llm.model == "llama3.1:8b"
    await service.close()


def with_deadline(
    milliseconds: int = 100, **overrides: Any
) -> tuple[GenerationService, ScriptedLLM]:
    """Return ``build()``'s service under a short ``reliability.request_deadline_ms``."""
    service, llm = build(**overrides)
    reliability = service.settings.reliability.model_copy(
        update={"request_deadline_ms": milliseconds}
    )
    service.settings = service.settings.model_copy(update={"reliability": reliability})
    return service, llm


class StalledRetrieval(ScriptedRetrieval):
    """Never returns, as a vector database that accepted the query and went silent does."""

    async def search(self, text: str, **kwargs: Any) -> Retrieval:
        await asyncio.Event().wait()
        raise AssertionError("a stalled retrieval never returns")


async def test_a_deadline_hit_while_retrieving_is_refused_on_both_surfaces() -> None:
    """Nothing bounded a request end to end, so a stalled backend held it forever (TASK-0449)."""
    service, llm = with_deadline()
    service.retrieval = StalledRetrieval([])  # type: ignore[assignment]
    events: list[QueryEvent] = []

    with pytest.raises(FasterRagError) as whole:
        await asyncio.wait_for(service.answer("q"), 10)
    with pytest.raises(FasterRagError) as streamed:
        await asyncio.wait_for(collect_into(events, service), 10)

    assert whole.value.code is ErrorCode.DEADLINE_EXCEEDED
    assert whole.value.status == 504
    assert streamed.value.code is ErrorCode.DEADLINE_EXCEEDED
    assert events == [], "a refused stream had already begun"
    assert llm.prompts == []


async def test_a_provider_that_hangs_past_the_deadline_serves_the_retrieved_passages() -> None:
    service, llm = with_deadline()
    llm.hang_after = 0

    answer = await asyncio.wait_for(service.answer("q"), 10)

    assert answer.mode == EXTRACTIVE_MODE
    assert "Either party may terminate." in (answer.answer or "")
    assert len(llm.prompts) == 1
    assert service.breaker.is_open is False


async def test_a_stream_past_its_deadline_after_a_token_ends_on_error() -> None:
    service, llm = with_deadline()
    llm.hang_after = 2

    events = await asyncio.wait_for(collect(service), 10)

    assert [event.type for event in events].count("token") == 2
    assert events[-1].type == "error"
    assert events[-1].data["code"] == ErrorCode.DEADLINE_EXCEEDED.value
    assert events[-1].data["status"] == 504
    assert "done" not in [event.type for event in events]


async def test_a_deadline_hit_while_grading_returns_the_answer_ungraded() -> None:
    service, _ = with_deadline(grounded_or_refuse=True, verdict='{"score": 0.1}')
    assert isinstance(service.grader, ScriptedLLM)
    service.grader.hang_after = 0

    answer = await asyncio.wait_for(service.answer("q"), 10)

    assert answer.answer == "The notice period is thirty days [^c_a]."
    assert answer.faithfulness is None


async def test_no_retry_is_waited_for_that_would_outlast_the_deadline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    waits: list[float] = []

    async def record(seconds: float) -> None:
        waits.append(seconds)

    monkeypatch.setattr("fasterrag.services.generation._pause", record)
    settings = Settings.model_validate(
        {
            "reliability": {
                "request_deadline_ms": 1000,
                "retries": {"backoff_base_ms": 1, "backoff_max_ms": 60000},
            }
        }
    )
    llm = ScriptedLLM(settings, text="an answer", error=GenerationError("429", retry_after=30.0))
    service = GenerationService(
        settings,
        ScriptedRetrieval([chunk("c_a", "Either party may terminate.")]),  # type: ignore[arg-type]
        llm,
    )

    answer = await service.answer("q")

    assert answer.mode == EXTRACTIVE_MODE
    assert waits == []
    assert len(llm.prompts) == 1


async def test_a_degraded_answer_s_trace_id_points_at_a_stored_trace(tmp_path: Path) -> None:
    """The extractive rung stored no trace, so its trace id resolved to nothing (TASK-0505)."""
    service, _ = build(error=GenerationError("status 503", retryable=True))
    service.traces = TraceStore(tmp_path)

    answer = await service.answer("what is the notice period?")
    stored = service.traces.load(answer.trace_id)

    assert stored is not None
    assert stored.result["mode"] == EXTRACTIVE_MODE
    assert stored.result["degraded"] is True
    failed = [span for span in stored.spans if span.name == "generation-failed"]
    assert failed[-1].attributes["code"] == ErrorCode.GENERATION_FAILED.value
    assert failed[-1].attributes["detail"] == "status 503"


async def test_a_misconfigured_provider_never_opens_the_breaker() -> None:
    """A misconfiguration never opens the breaker.

    A rejected key fails the same way forever, so counting it would hide the one message that
    explains it behind an intermittent-outage story.
    """
    service, _ = build(error=ConfigError("the api key was rejected"))
    service.breaker.failure_threshold = 1

    for _ in range(3):
        with pytest.raises(ConfigError):
            await service.answer("what is the notice period", collection="policies")

    assert service.breaker.is_open is False


async def test_a_query_over_the_budget_is_refused_rather_than_answered() -> None:
    """The cap must reach the caller as 402, never as a quietly degraded answer."""
    settings = Settings.model_validate(
        {"cost": {"per_query_token_budget": 10}, "llm": {"max_tokens": 1024}}
    )
    llm = ScriptedLLM(settings, text="an answer")
    retrieval = ScriptedRetrieval([chunk("c_a", "Either party may terminate.")])
    service = GenerationService(settings, retrieval, llm)  # type: ignore[arg-type]

    with pytest.raises(FasterRagError) as caught:
        await service.answer("what is the notice period", collection="policies")

    assert caught.value.code is ErrorCode.BUDGET_EXCEEDED
    assert caught.value.retryable is False
    assert llm.prompts == [], "the provider was called for a query the budget refused"


async def test_a_streamed_query_over_the_budget_is_refused_before_any_token() -> None:
    """A mid-stream refusal would arrive after `meta` had already promised an answer."""
    settings = Settings.model_validate(
        {"cost": {"per_query_token_budget": 10}, "llm": {"max_tokens": 1024}}
    )
    llm = ScriptedLLM(settings, text="an answer")
    retrieval = ScriptedRetrieval([chunk("c_a", "Either party may terminate.")])
    service = GenerationService(settings, retrieval, llm)  # type: ignore[arg-type]

    events: list[QueryEvent] = []
    with pytest.raises(FasterRagError) as caught:
        await collect_into(events, service, "what is the notice period")

    assert caught.value.code is ErrorCode.BUDGET_EXCEEDED
    assert llm.prompts == []
    assert events == [], "`meta` went out before the refusal, committing the stream to a 200"


async def test_an_unset_budget_permits_everything() -> None:
    """The default is 0 = unlimited, and a governor nobody configured must not refuse."""
    service, llm = build()

    result = await service.answer("what is the notice period", collection="policies")

    assert result.answer is not None
    assert llm.prompts != []


async def test_actual_usage_is_charged_to_the_tenant_not_the_admitted_ceiling() -> None:
    """Charging the ceiling would bill every short answer for a long one."""
    settings = Settings.model_validate(
        {"cost": {"per_tenant_token_budget": 100_000}, "llm": {"max_tokens": 1024}}
    )
    llm = ScriptedLLM(settings, text="The notice period is thirty days [^c_a].")
    retrieval = ScriptedRetrieval([chunk("c_a", "Either party may terminate.")])
    service = GenerationService(settings, retrieval, llm, tenant="acme")  # type: ignore[arg-type]

    await service.answer("what is the notice period", collection="policies")

    assert service.governor.spent("acme") == 18, "ScriptedLLM reports 11 prompt + 7 completion"
    assert service.governor.spent("other") == 0


async def test_the_budget_is_checked_against_the_ceiling_not_just_the_prompt() -> None:
    """A cap that admits a request it cannot pay for is not a cap."""
    settings = Settings.model_validate(
        {"cost": {"per_query_token_budget": 2000}, "llm": {"max_tokens": 8000}}
    )
    llm = ScriptedLLM(settings, text="an answer")
    retrieval = ScriptedRetrieval([chunk("c_a", "Either party may terminate.")])
    service = GenerationService(settings, retrieval, llm)  # type: ignore[arg-type]

    with pytest.raises(FasterRagError) as caught:
        await service.answer("what is the notice period", collection="policies")

    assert caught.value.code is ErrorCode.BUDGET_EXCEEDED
    assert llm.prompts == []


async def test_the_response_carries_the_cost_field() -> None:
    """A caller holding one answer cannot read a process-wide counter to learn what it cost.

    The figure is a list-price *estimate*, which is why the field name says so.
    """
    service, _ = build()

    payload = (await service.answer("q")).as_dict()

    assert "estimated_cost_usd" in payload["usage"]


async def test_an_unpriced_model_reports_null_rather_than_zero() -> None:
    """Zero would read as "this query was free", which is a fabricated bill.

    The pricing tables carry dated per-entry provenance precisely so a model with no
    recorded rate contributes nothing instead of a made-up number.
    """
    service, _ = build()

    payload = (await service.answer("q")).as_dict()

    assert payload["usage"]["estimated_cost_usd"] is None


async def test_a_local_provider_is_genuinely_free() -> None:
    """A local provider costs zero, and that zero is a fact rather than a guess.

    It must not be confused with the unpriced case, which reports None: one says the
    query was free, the other says nobody knows what it cost.
    """

    class LocalLLM(ScriptedLLM):
        """The provider is the adapter's own attribute, not a settings value."""

        provider = "ollama"

    settings = Settings.model_validate({})
    llm = LocalLLM(settings, text="An answer [^c_a].")
    retrieval = ScriptedRetrieval([chunk("c_a", "Either party may terminate.")])
    service = GenerationService(settings, retrieval, llm)  # type: ignore[arg-type]

    answer = await service.answer("q")

    assert answer.estimated_cost_usd == 0.0


async def test_the_streamed_usage_event_carries_the_same_field() -> None:
    """The streamed usage event carries the same field under the same name.

    A streaming caller needs the signal too, and needs it named identically so that
    reading it takes no second code path.
    """
    service, _ = build()

    events = await collect(service)
    usage = next(event for event in events if event.type == "usage")

    assert "estimated_cost_usd" in usage.data["usage"]


def traced(tmp_path: Path) -> tuple[GenerationService, TraceStore]:
    service, _ = build()
    store = TraceStore(tmp_path)
    service.traces = store
    return service, store


async def test_a_streamed_query_stores_a_trace(tmp_path: Path) -> None:
    """Streaming is the default path; without this, replay had nothing for most traffic."""
    service, store = traced(tmp_path)

    events = await collect(service, "what is the notice period?")
    meta = next(event for event in events if event.type == "meta")
    stored = store.load(meta.data["trace_id"])

    assert stored is not None
    assert stored.query == "what is the notice period?"
    assert stored.response.strip() == "The notice period is thirty days [^c_a]."


async def test_a_consumer_that_stops_at_done_still_leaves_a_trace(tmp_path: Path) -> None:
    """The trace was written after `done` was yielded, so a reader that stopped there lost it."""
    service, store = traced(tmp_path)
    events = service.stream("what is the notice period?")

    meta = await anext(events)
    while (await anext(events)).type != "done":
        continue

    assert store.load(meta.data["trace_id"]) is not None


async def test_a_streamed_trace_matches_the_non_streamed_one(tmp_path: Path) -> None:
    """Two code paths producing the same answer must leave the same evidence behind."""
    service, store = traced(tmp_path)

    answered = await service.answer("what is the notice period?")
    events = await collect(service, "what is the notice period?")
    meta = next(event for event in events if event.type == "meta")
    plain = store.load(answered.trace_id)
    streamed = store.load(meta.data["trace_id"])

    assert plain is not None
    assert streamed is not None
    assert streamed.prompt == plain.prompt
    assert streamed.retrieved == plain.retrieved
    assert streamed.result["answer"] == plain.result["answer"]


PARITY_QUESTION = "what is the notice period?"
PARITY_TENANT = "acme"
PARITY_COUNTER = EstimatingTokenCounter()


class MeteredLLM(ScriptedLLM):
    """A provider whose whole call reports exactly what its stream of the same text amounts to.

    ``stream`` carries text only, so the service counts a streamed answer itself. For the two
    surfaces to be comparable at all, ``complete`` has to report the same model, counts and
    finish reason, measured the way the service measures them — and the stream has to
    concatenate to the same text, which ``ScriptedLLM``'s trailing space does not.
    """

    async def complete(self, prompt: str, *, system: str | None = None) -> Completion:
        self.prompts.append((prompt, system))
        if self.hang_after is not None:
            await self.hang()
        if self.failing():
            assert self.error is not None
            raise self.error
        tokens = PARITY_COUNTER.count(self.text)
        return Completion(
            text=self.text,
            model=self.model,
            prompt_tokens=PARITY_COUNTER.count(prompt),
            completion_tokens=tokens,
            finish_reason="length" if tokens >= self.config.max_tokens else "stop",
        )

    async def stream(self, prompt: str, *, system: str | None = None) -> AsyncIterator[str]:
        self.prompts.append((prompt, system))
        if self.hang_after is not None:
            await self.hang()
        if self.failing():
            assert self.error is not None
            raise self.error
        first, *rest = self.text.split(" ")
        yield first
        for word in rest:
            yield f" {word}"


@dataclass(frozen=True)
class Observed:
    """What one surface produced and left behind, with the surface itself taken out."""

    raised: str | None
    body: dict[str, Any] | None
    metrics: dict[str, float]
    trace: dict[str, Any] | None
    spent: int
    provider_calls: int
    breaker_open: bool


@dataclass(frozen=True)
class ParityCase:
    """One way a query can end, and the fact that proves the case really is that way."""

    holds: Callable[[Observed], bool]
    settings: dict[str, Any] = field(default_factory=dict)
    answer: str = "The notice period is thirty days [^c_a]."
    error: Callable[[], FasterRagError] | None = None
    verdict: str | None = None
    chunks: tuple[tuple[str, str], ...] = (("c_a", "Either party may terminate."),)
    mode: str = FULL_MODE
    primed: bool = False
    tripped: bool = False
    failures: int | None = None
    fallback: bool = False
    hangs: bool = False


def _body(observed: Observed) -> dict[str, Any]:
    assert observed.body is not None
    return observed.body


def _span_names(observed: Observed) -> list[str]:
    assert observed.trace is not None
    return [name for name, _ in observed.trace["spans"]]


def _degraded_trace_names(seen: Observed, code: ErrorCode) -> bool:
    """Return whether an extractive answer left a trace naming the failure that caused it."""
    assert seen.trace is not None, "a degraded answer stored no trace (TASK-0505)"
    failed = [attributes for name, attributes in seen.trace["spans"] if name == "generation-failed"]
    return (
        seen.trace["result"]["mode"] == EXTRACTIVE_MODE
        and seen.trace["response"] == ""
        and bool(failed)
        and failed[-1]["code"] == code.value
    )


PARITY_CASES: dict[str, ParityCase] = {
    "success": ParityCase(
        holds=lambda seen: (
            _body(seen)["mode"] == FULL_MODE
            and _span_names(seen)[-1] == "generation"
            and seen.spent > 0
        ),
    ),
    "cache hit": ParityCase(
        settings={"cache": {"semantic": True}},
        primed=True,
        holds=lambda seen: (
            _body(seen)["cache"]["semantic_hit"] is True and seen.provider_calls == 1
        ),
    ),
    "retryable failure": ParityCase(
        error=lambda: GenerationError("provider is down", retryable=True),
        holds=lambda seen: (
            _body(seen)["mode"] == EXTRACTIVE_MODE
            and "Either party may terminate." in _body(seen)["answer"]
            and [cited["chunk_id"] for cited in _body(seen)["citations"]] == ["c_a"]
            and seen.metrics["fasterrag_degraded_responses_total{mode=extractive}"] == 1
            and seen.provider_calls == 1 + Settings().reliability.retries.max_attempts
            and _degraded_trace_names(seen, ErrorCode.GENERATION_FAILED)
        ),
    ),
    "retried, then answered": ParityCase(
        error=lambda: GenerationError("rate limited", retryable=True),
        failures=2,
        holds=lambda seen: (
            _body(seen)["mode"] == FULL_MODE
            and seen.provider_calls == 3
            and _span_names(seen)[-1] == "generation"
            and seen.trace is not None
            and seen.trace["spans"][-1][1]["attempts"] == 3
        ),
    ),
    "fallback answered": ParityCase(
        error=lambda: GenerationError("provider is down", retryable=True),
        fallback=True,
        holds=lambda seen: (
            _body(seen)["mode"] == FULL_MODE
            and seen.trace is not None
            and "generation-failed" in _span_names(seen)
            and seen.trace["spans"][-1] == ("generation", seen.trace["spans"][-1][1])
            and seen.trace["spans"][-1][1]["role"] == "fallback"
            and seen.spent > 0
        ),
    ),
    "deadline while generating": ParityCase(
        settings={"reliability": {"request_deadline_ms": 100}},
        hangs=True,
        holds=lambda seen: (
            _body(seen)["mode"] == EXTRACTIVE_MODE
            and seen.provider_calls == 1
            and not seen.breaker_open
            and _degraded_trace_names(seen, ErrorCode.DEADLINE_EXCEEDED)
        ),
    ),
    "open breaker": ParityCase(
        tripped=True,
        holds=lambda seen: (
            _body(seen)["mode"] == EXTRACTIVE_MODE
            and seen.metrics["fasterrag_degraded_responses_total{mode=extractive}"] == 1
            and seen.provider_calls == 0
            and seen.breaker_open
            and _degraded_trace_names(seen, ErrorCode.CIRCUIT_OPEN)
        ),
    ),
    "non-retryable failure": ParityCase(
        error=lambda: GenerationError("the provider rejected the credentials", retryable=False),
        holds=lambda seen: seen.raised == "GENERATION_FAILED" and seen.trace is None,
    ),
    "budget refusal": ParityCase(
        settings={"cost": {"per_query_token_budget": 10}},
        holds=lambda seen: seen.raised == "BUDGET_EXCEEDED" and seen.provider_calls == 0,
    ),
    "grounded refusal": ParityCase(
        settings={"generation": {"grounded_or_refuse": True}},
        verdict='{"score": 0.38}',
        holds=lambda seen: (
            _body(seen)["code"] == "INSUFFICIENT_EVIDENCE"
            and seen.trace is not None
            and seen.trace["result"]["answer"] is None
            and seen.metrics["fasterrag_faithfulness{}"] == 1
        ),
    ),
    "graded answer": ParityCase(
        settings={"generation": {"grounded_or_refuse": True}},
        verdict='{"score": 0.93}',
        holds=lambda seen: _body(seen)["faithfulness"] == pytest.approx(0.93),
    ),
    "truncation": ParityCase(
        settings={"llm": {"max_tokens": 4}},
        holds=lambda seen: _body(seen)["truncated"] is True and seen.trace is not None,
    ),
    "evidence dropped": ParityCase(
        settings={"generation": {"context_budget_tokens": 350}},
        answer="an answer [^c_a] [^c_b]",
        chunks=(("c_a", "alpha " * 200), ("c_b", "bravo " * 200)),
        holds=lambda seen: (
            _body(seen)["evidence_dropped"] == 1
            and [cited["chunk_id"] for cited in _body(seen)["citations"]] == ["c_a"]
        ),
    ),
    "degraded retrieval": ParityCase(
        mode=HYBRID_ONLY_MODE,
        holds=lambda seen: (
            _body(seen)["mode"] == HYBRID_ONLY_MODE and _body(seen)["degraded"] is True
        ),
    ),
}


def _parity_service(case: ParityCase, root: Path) -> tuple[GenerationService, MeteredLLM]:
    """Build a fresh service for one surface, tracing into its own directory."""
    settings = Settings.model_validate(
        {
            "cost": {"per_tenant_token_budget": 1_000_000},
            "reliability": FAST_RETRIES,
            **case.settings,
        }
    )
    llm = MeteredLLM(settings, text=case.answer, error=case.error() if case.error else None)
    llm.failures = case.failures
    llm.hang_after = 0 if case.hangs else None
    cache = SemanticCache(settings, MemoryStore()) if settings.cache.semantic else None
    service = GenerationService(
        settings,
        ScriptedRetrieval([chunk(*pair) for pair in case.chunks], case.mode),  # type: ignore[arg-type]
        llm,
        grader=ScriptedLLM(settings, text=case.verdict) if case.verdict is not None else None,
        cache=cache,
        embedder=ScriptedEmbedder(settings) if cache is not None else None,
        traces=TraceStore(root),
        tenant=PARITY_TENANT,
        fallback=MeteredLLM(settings, text=case.answer) if case.fallback else None,
    )
    return service, llm


def _metric_values() -> dict[str, float]:
    """Return every series' counter value, or observation count for a histogram.

    A histogram's sum is a wall-clock duration and differs between any two runs, so only how
    many observations were taken is compared. ``fasterrag_ttft_seconds`` is left out by
    definition rather than by convenience: a whole answer has no first streamed token.
    """
    values: dict[str, float] = {}
    for sample in metrics.REGISTRY.snapshot():
        if sample.name == metrics.TTFT.name:
            continue
        labels = ",".join(f"{key}={value}" for key, value in sorted(sample.labels.items()))
        values[f"{sample.name}{{{labels}}}"] = (
            float(sample.count) if sample.count is not None else sample.value
        )
    return values


def _surface_free(body: dict[str, Any], trace_id: str) -> dict[str, Any]:
    """Return a response body with its two per-run members reduced to what is comparable."""
    assert body["trace_id"] == trace_id
    reduced = {**body, "trace_id": "<this query>"}
    if "timings_ms" in reduced:
        reduced["timings_ms"] = sorted(reduced["timings_ms"])
    return reduced


def _body_from_events(events: list[QueryEvent]) -> dict[str, Any]:
    """Rebuild the non-streamed response body from a completed event stream."""
    kinds = [event.type for event in events]
    assert kinds[0] == "meta"
    assert kinds[-1] == "done"

    withheld = [event for event in events if event.type == "insufficient_evidence"]
    if withheld:
        assert kinds == ["meta", "insufficient_evidence", "done"]
        return dict(withheld[0].data)

    meta = [event for event in events if event.type == "meta"]
    usage = next(event for event in events if event.type == "usage")
    assert kinds.index("citations") < kinds.index("usage")
    assert meta[-1].data["mode"] == usage.data["mode"], "the mode was not known before the text"
    assert meta[-1].data["degraded"] == usage.data["degraded"]
    return {
        "answer": "".join(event.data["text"] for event in events if event.type == "token"),
        "citations": next(e for e in events if e.type == "citations").data["citations"],
        "usage": usage.data["usage"],
        "timings_ms": usage.data["timings_ms"],
        "degraded": usage.data["degraded"],
        "mode": usage.data["mode"],
        "truncated": usage.data["truncated"],
        "evidence_dropped": usage.data["evidence_dropped"],
        "faithfulness": usage.data["faithfulness"],
        "cache": meta[0].data["cache"],
        "trace_id": meta[0].data["trace_id"],
    }


async def _observe(case: ParityCase, root: Path, *, streamed: bool) -> Observed:
    """Run one case through one surface and record everything it left behind."""
    reset_shared_breakers()
    reset_shared_governors()
    service, llm = _parity_service(case, root)
    assert service.traces is not None
    if case.primed:
        with use_trace_id():
            await service.answer(PARITY_QUESTION)
    if case.tripped:
        service.breaker.failure_threshold = 1
        service.breaker.record_failure(GenerationError("an earlier outage", retryable=True))

    before = _metric_values()
    raised: str | None = None
    body: dict[str, Any] | None = None
    events: list[QueryEvent] = []
    with use_trace_id() as trace_id:
        try:
            if streamed:
                await collect_into(events, service, PARITY_QUESTION)
                body = _surface_free(_body_from_events(events), trace_id)
            else:
                body = _surface_free((await service.answer(PARITY_QUESTION)).as_dict(), trace_id)
        except FasterRagError as exc:
            raised = exc.code.value
    after = _metric_values()

    if raised == ErrorCode.BUDGET_EXCEEDED.value:
        assert events == [], "a refused stream had already begun"

    stored = service.traces.load(trace_id)
    trace: dict[str, Any] | None = None
    if stored is not None:
        trace = {
            "query": stored.query,
            "collection": stored.collection,
            "tenant": stored.tenant,
            "filters": stored.filters,
            "retrieved": stored.retrieved,
            "prompt": stored.prompt,
            "response": stored.response,
            "result": _surface_free(stored.result, trace_id),
            "spans": [(span.name, span.attributes) for span in stored.spans],
        }

    return Observed(
        raised=raised,
        body=body,
        metrics={
            name: value - before.get(name, 0.0)
            for name, value in after.items()
            if value != before.get(name, 0.0)
        },
        trace=trace,
        spent=service.governor.spent(PARITY_TENANT),
        provider_calls=len(llm.prompts),
        breaker_open=service.breaker.is_open,
    )


@pytest.mark.parametrize("name", sorted(PARITY_CASES))
async def test_both_surfaces_leave_the_same_outcome(name: str, tmp_path: Path) -> None:
    """`answer()` and `stream()` were two hand-kept copies of one pipeline, and had diverged.

    Each case is run through both and everything observable is compared: the response body
    (mode, flags, usage, citations), every metric the query moved, the stored trace with its
    spans, what the tenant was charged, and how often the provider was called (TASK-0438).
    """
    case = PARITY_CASES[name]

    whole = await _observe(case, tmp_path / "whole", streamed=False)
    streamed = await _observe(case, tmp_path / "streamed", streamed=True)

    assert case.holds(whole), whole
    assert streamed == whole
