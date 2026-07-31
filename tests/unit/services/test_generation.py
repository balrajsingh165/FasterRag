from collections.abc import AsyncIterator
from typing import Any

import pytest

from fasterrag.adapters.llm.base import Completion, LLMAdapter
from fasterrag.adapters.vectordb.base import HealthStatus
from fasterrag.config.schema import Settings
from fasterrag.core.retrieval.models import ScoredChunk
from fasterrag.errors import GenerationError
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
) -> tuple[GenerationService, ScriptedLLM]:
    settings = Settings()
    llm = ScriptedLLM(settings, text=answer, error=error)
    retrieval = ScriptedRetrieval(
        chunks if chunks is not None else [chunk("c_a", "Either party may terminate.")], mode
    )
    service = GenerationService(settings, retrieval, llm)  # type: ignore[arg-type]
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
