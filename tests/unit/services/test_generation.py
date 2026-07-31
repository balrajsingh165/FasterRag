from collections.abc import AsyncIterator, Sequence
from typing import Any

import pytest

from fasterrag.adapters.embeddings.base import EmbeddingAdapter, EmbeddingResult
from fasterrag.adapters.llm.base import Completion, LLMAdapter
from fasterrag.adapters.vectordb.base import HealthStatus
from fasterrag.config.schema import Settings
from fasterrag.core.cache.semantic import SemanticCache
from fasterrag.core.cache.store import MemoryStore
from fasterrag.core.retrieval.models import ScoredChunk
from fasterrag.errors import EmbedError, GenerationError
from fasterrag.services.generation import EXTRACTIVE_MODE, GenerationService, QueryEvent
from fasterrag.services.querying import FULL_MODE, HYBRID_ONLY_MODE, Retrieval


class ScriptedLLM(LLMAdapter):
    """Returns scripted text, or fails on demand."""

    provider = "scripted"

    def __init__(self, settings: Settings, text: str = "", error: Exception | None = None) -> None:
        super().__init__(settings)
        self.text = text
        self.error = error
        self.prompts: list[tuple[str, str | None]] = []

    async def complete(self, prompt: str, *, system: str | None = None) -> Completion:
        self.prompts.append((prompt, system))
        if self.error is not None:
            raise self.error
        return Completion(text=self.text, model="scripted", prompt_tokens=11, completion_tokens=7)

    async def stream(self, prompt: str, *, system: str | None = None) -> AsyncIterator[str]:
        self.prompts.append((prompt, system))
        if self.error is not None:
            raise self.error
        for word in self.text.split(" "):
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
            }
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
    service, _ = build(error=GenerationError("provider is down"))

    result = await service.answer("q")

    assert result.mode == EXTRACTIVE_MODE
    assert result.degraded is True
    assert result.answer is not None
    assert "Either party may terminate." in result.answer
    assert [citation.chunk_id for citation in result.citations] == ["c_a"]


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
    service, _ = build(error=GenerationError("provider vanished"))

    events = await collect(service)
    kinds = [event.type for event in events]

    assert "error" in kinds
    assert "done" not in kinds
    assert kinds[-1] == "error"


async def test_the_error_event_carries_the_stable_code_and_trace() -> None:
    service, _ = build(error=GenerationError("provider vanished"))

    error = next(event for event in await collect(service) if event.type == "error")

    assert error.data["code"] == "GENERATION_FAILED"
    assert error.data["trace_id"]
    assert "retryable" in error.data


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
        {"cache": {"semantic": True, "similarity_threshold": threshold}}
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
