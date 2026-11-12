import asyncio
import json
from collections.abc import AsyncIterator
from typing import Any

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from fasterrag.adapters.llm.base import Completion, LLMAdapter
from fasterrag.adapters.vectordb.base import HealthStatus
from fasterrag.api import query as query_router
from fasterrag.api.dependencies import shared_cache, shared_embedding_router
from fasterrag.api.problems import PROBLEM_MEDIA_TYPE
from fasterrag.api.query import encode_event
from fasterrag.config.schema import Settings
from fasterrag.core.context import Citation, Span
from fasterrag.core.retrieval.models import ScoredChunk
from fasterrag.errors import ErrorCode, GenerationError, RetrievalError
from fasterrag.services.generation import Answer, GenerationService, QueryEvent
from fasterrag.services.querying import FULL_MODE, HYBRID_ONLY_MODE, Retrieval

DEFAULT_EVENTS = [
    QueryEvent(type="meta", data={"trace_id": "t", "mode": FULL_MODE}),
    QueryEvent(type="token", data={"text": "thirty days"}),
    QueryEvent(type="citations", data={"citations": []}),
    QueryEvent(type="usage", data={"usage": {}}),
    QueryEvent(type="done", data={}),
]


class StubGeneration:
    """Returns a scripted answer or scripted events."""

    def __init__(
        self,
        answer: Answer | None = None,
        events: list[QueryEvent] | None = None,
        error: Exception | None = None,
    ) -> None:
        self.answer_value = answer or Answer(answer="thirty days [^c_a]", mode=FULL_MODE)
        self.events = events
        self.error = error
        self.closed = False
        self.calls: list[dict[str, Any]] = []

    async def answer(self, question: str, **kwargs: Any) -> Answer:
        self.calls.append({"question": question, **kwargs})
        if self.error is not None:
            raise self.error
        return self.answer_value

    async def stream(self, question: str, **kwargs: Any) -> AsyncIterator[QueryEvent]:
        self.calls.append({"question": question, **kwargs})
        scripted = DEFAULT_EVENTS if self.events is None else self.events
        for event in scripted:
            yield event
        if self.error is not None:
            raise self.error

    async def close(self) -> None:
        self.closed = True


class StubRetrieval:
    """Returns scripted chunks."""

    def __init__(self, chunks: list[ScoredChunk], mode: str = FULL_MODE) -> None:
        self.chunks = chunks
        self.mode = mode

    async def search(self, text: str, **kwargs: Any) -> Retrieval:
        return Retrieval(chunks=list(self.chunks), mode=self.mode)


class StubRouter:
    """Stands in for the tiered embedding router."""

    def __init__(self) -> None:
        self.closed = False
        self.default = object()

    async def close(self) -> None:
        self.closed = True


def stub_shared(app: FastAPI) -> None:
    """Override the process-scoped dependencies so no model is ever loaded in a test."""
    app.dependency_overrides[shared_embedding_router] = StubRouter
    app.dependency_overrides[shared_cache] = lambda: None


@pytest.fixture
def generation(monkeypatch: pytest.MonkeyPatch, app: FastAPI) -> StubGeneration:
    service = StubGeneration()
    stub_shared(app)
    monkeypatch.setattr(
        query_router,
        "build_generation",
        lambda settings, adapter, router, cache=None, traces=None, tenant=None, llm=None: service,
    )
    return service


def install_generation(
    monkeypatch: pytest.MonkeyPatch, app: FastAPI, service: StubGeneration
) -> StubGeneration:
    stub_shared(app)
    monkeypatch.setattr(
        query_router,
        "build_generation",
        lambda settings, adapter, router, cache=None, traces=None, tenant=None, llm=None: service,
    )
    return service


def frames(text: str) -> list[tuple[str, dict[str, Any]]]:
    """Parse an SSE body into (event, data) pairs."""
    parsed: list[tuple[str, dict[str, Any]]] = []
    for block in text.strip().split("\n\n"):
        lines = dict(line.split(": ", 1) for line in block.splitlines() if ": " in line)
        if "event" in lines:
            parsed.append((lines["event"], json.loads(lines.get("data", "{}"))))
    return parsed


def test_an_event_frame_ends_with_a_blank_line() -> None:
    frame = encode_event(QueryEvent(type="token", data={"text": "hi"}))

    assert frame.endswith("\n\n")
    assert frame.startswith("event: token\n")


def test_event_data_is_a_single_line() -> None:
    frame = encode_event(QueryEvent(type="usage", data={"usage": {"prompt_tokens": 1}}))
    data_line = next(line for line in frame.splitlines() if line.startswith("data: "))

    assert "\n" not in data_line
    assert json.loads(data_line.removeprefix("data: "))


def test_a_non_streamed_query_returns_the_answer_body(
    client: TestClient, generation: StubGeneration
) -> None:
    response = client.post("/v1/query", json={"query": "notice period?", "stream": False})

    assert response.status_code == 200
    assert response.json()["answer"] == "thirty days [^c_a]"


def test_a_non_streamed_query_carries_the_documented_members(
    client: TestClient, generation: StubGeneration
) -> None:
    body = client.post("/v1/query", json={"query": "q", "stream": False}).json()

    assert {"answer", "citations", "usage", "timings_ms", "degraded", "mode", "cache"} <= set(body)


def test_the_request_overrides_reach_the_service(
    client: TestClient, generation: StubGeneration
) -> None:
    client.post(
        "/v1/query",
        json={
            "query": "q",
            "stream": False,
            "top_k": 3,
            "collection": "legal",
            "filters": {"department": "legal"},
        },
    )

    call = generation.calls[0]
    assert call["top_k"] == 3
    assert call["collection"] == "legal"
    assert call["filters"] == {"department": "legal"}


def test_the_service_is_closed_after_a_non_streamed_query(
    client: TestClient, generation: StubGeneration
) -> None:
    client.post("/v1/query", json={"query": "q", "stream": False})

    assert generation.closed is True


def test_a_streamed_query_serves_an_event_stream(
    client: TestClient, generation: StubGeneration
) -> None:
    response = client.post("/v1/query", json={"query": "q", "stream": True})

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")


def test_a_streamed_query_follows_the_documented_event_order(
    client: TestClient, generation: StubGeneration
) -> None:
    response = client.post("/v1/query", json={"query": "q", "stream": True})

    assert [name for name, _ in frames(response.text)] == [
        "meta",
        "token",
        "citations",
        "usage",
        "done",
    ]


def test_a_streamed_query_closes_its_service(
    client: TestClient, generation: StubGeneration
) -> None:
    client.post("/v1/query", json={"query": "q", "stream": True})

    assert generation.closed is True


def test_a_failure_after_the_stream_began_becomes_an_error_event(
    client: TestClient, monkeypatch: pytest.MonkeyPatch, app: FastAPI
) -> None:
    install_generation(
        monkeypatch,
        app,
        StubGeneration(
            events=[QueryEvent(type="meta", data={"trace_id": "t"})],
            error=GenerationError("provider vanished", retryable=True),
        ),
    )

    response = client.post("/v1/query", json={"query": "q", "stream": True})
    names = [name for name, _ in frames(response.text)]

    assert response.status_code == 200
    assert names == ["meta", "error"]
    assert "done" not in names


def test_the_error_event_carries_the_stable_code(
    client: TestClient, monkeypatch: pytest.MonkeyPatch, app: FastAPI
) -> None:
    install_generation(
        monkeypatch,
        app,
        StubGeneration(
            events=[QueryEvent(type="meta", data={"trace_id": "t"})],
            error=RetrievalError("qdrant is unreachable", retryable=True),
        ),
    )

    _, data = frames(client.post("/v1/query", json={"query": "q", "stream": True}).text)[-1]

    assert data["code"] == ErrorCode.RETRIEVAL_FAILED.value
    assert data["retryable"] is True
    assert data["status"] == 503


class IdleLLM(LLMAdapter):
    """A provider that must never be reached; it counts the calls that got through."""

    provider = "idle"

    def __init__(self, settings: Settings) -> None:
        super().__init__(settings)
        self.calls = 0

    async def complete(self, prompt: str, *, system: str | None = None) -> Completion:
        self.calls += 1
        return Completion(text="an answer", model="idle")

    async def stream(self, prompt: str, *, system: str | None = None) -> AsyncIterator[str]:
        self.calls += 1
        yield "an answer"

    async def health(self) -> HealthStatus:
        return HealthStatus(healthy=True)

    async def close(self) -> None:
        return None


def test_a_streamed_query_over_the_budget_is_a_402_problem_not_an_event_stream(
    client: TestClient, monkeypatch: pytest.MonkeyPatch, app: FastAPI
) -> None:
    """A streamed budget refusal arrived as an SSE `error` over HTTP 200, after `meta`."""
    settings = Settings.model_validate(
        {"cost": {"per_query_token_budget": 10}, "llm": {"max_tokens": 1024}}
    )
    llm = IdleLLM(settings)
    chunks = [ScoredChunk(chunk_id="c_a", text="Either party may terminate.", rrf_score=0.5)]
    service = GenerationService(settings, StubRetrieval(chunks), llm)  # type: ignore[arg-type]
    stub_shared(app)
    monkeypatch.setattr(
        query_router,
        "build_generation",
        lambda settings, adapter, router, cache=None, traces=None, tenant=None, llm=None: service,
    )

    response = client.post("/v1/query", json={"query": "notice period?", "stream": True})

    assert response.status_code == 402
    assert response.headers["content-type"].startswith(PROBLEM_MEDIA_TYPE)
    assert response.json()["code"] == ErrorCode.BUDGET_EXCEEDED.value
    assert "event:" not in response.text
    assert llm.calls == 0


class DownLLM(IdleLLM):
    """A provider that fails every call the way an outage does."""

    outage = "provider is down"

    async def stream(self, prompt: str, *, system: str | None = None) -> AsyncIterator[str]:
        self.calls += 1
        if self.outage:
            raise GenerationError(self.outage, retryable=True)
        yield ""


def test_a_streamed_query_whose_provider_is_down_still_ends_with_done(
    client: TestClient, monkeypatch: pytest.MonkeyPatch, app: FastAPI
) -> None:
    """A provider outage ended a stream on `error` where the JSON body degraded (TASK-0438)."""
    settings = Settings()
    chunks = [ScoredChunk(chunk_id="c_a", text="Either party may terminate.", rrf_score=0.5)]
    service = GenerationService(settings, StubRetrieval(chunks), DownLLM(settings))  # type: ignore[arg-type]
    stub_shared(app)
    monkeypatch.setattr(
        query_router,
        "build_generation",
        lambda settings, adapter, router, cache=None, traces=None, tenant=None, llm=None: service,
    )

    response = client.post("/v1/query", json={"query": "notice period?", "stream": True})
    events = frames(response.text)
    text = "".join(data["text"] for name, data in events if name == "token")
    usage = next(data for name, data in events if name == "usage")

    assert response.status_code == 200
    assert [name for name, _ in events] == ["meta", "meta", "token", "citations", "usage", "done"]
    assert "Either party may terminate." in text
    assert events[1][1]["mode"] == "extractive"
    assert usage["mode"] == "extractive"
    assert usage["degraded"] is True


class StalledRetrieval(StubRetrieval):
    """Never answers, as a vector database that accepted the query and went silent does."""

    async def search(self, text: str, **kwargs: Any) -> Retrieval:
        await asyncio.Event().wait()
        raise AssertionError("a stalled retrieval never returns")


@pytest.mark.parametrize("stream", [False, True])
def test_a_query_past_its_deadline_before_any_event_is_a_504_problem(
    client: TestClient, monkeypatch: pytest.MonkeyPatch, app: FastAPI, stream: bool
) -> None:
    """Nothing bounded a query end to end, so a stalled backend held the request (TASK-0449)."""
    settings = Settings.model_validate({"reliability": {"request_deadline_ms": 100}})
    llm = IdleLLM(settings)
    service = GenerationService(settings, StalledRetrieval([]), llm)  # type: ignore[arg-type]
    install_generation(monkeypatch, app, service)  # type: ignore[arg-type]

    response = client.post("/v1/query", json={"query": "notice period?", "stream": stream})

    assert response.status_code == 504
    assert response.headers["content-type"].startswith(PROBLEM_MEDIA_TYPE)
    assert response.json()["code"] == ErrorCode.DEADLINE_EXCEEDED.value
    assert response.json()["retryable"] is True
    assert llm.calls == 0


class DownEverywhereLLM(DownLLM):
    """Fails the whole-answer call as well as the stream."""

    async def complete(self, prompt: str, *, system: str | None = None) -> Completion:
        self.calls += 1
        raise GenerationError(self.outage, retryable=True)


@pytest.mark.parametrize("stream", [False, True])
def test_a_degraded_answer_s_trace_resolves_through_the_traces_endpoint(
    client: TestClient, monkeypatch: pytest.MonkeyPatch, app: FastAPI, stream: bool
) -> None:
    """The extractive rung stored no trace, so the id it returned resolved to 404 (TASK-0505)."""
    settings = Settings.model_validate(
        {"reliability": {"retries": {"backoff_base_ms": 1, "backoff_max_ms": 1}}}
    )
    chunks = [ScoredChunk(chunk_id="c_a", text="Either party may terminate.", rrf_score=0.5)]
    service = GenerationService(
        settings,
        StubRetrieval(chunks),  # type: ignore[arg-type]
        DownEverywhereLLM(settings),
        traces=app.state.traces,
    )
    install_generation(monkeypatch, app, service)  # type: ignore[arg-type]

    response = client.post("/v1/query", json={"query": "notice period?", "stream": stream})
    trace_id = frames(response.text)[0][1]["trace_id"] if stream else response.json()["trace_id"]
    trace = client.get(f"/v1/traces/{trace_id}")

    assert trace.status_code == 200
    assert trace.json()["result"]["mode"] == "extractive"
    assert "generation-failed" in [span["name"] for span in trace.json()["spans"]]


def test_a_stream_that_fails_before_its_first_event_closes_its_service(
    client: TestClient, monkeypatch: pytest.MonkeyPatch, app: FastAPI
) -> None:
    """The early failure leaves through the problem handler, so nothing else would close it."""
    service = install_generation(
        monkeypatch,
        app,
        StubGeneration(events=[], error=RetrievalError("qdrant is unreachable", retryable=True)),
    )

    response = client.post("/v1/query", json={"query": "q", "stream": True})

    assert response.status_code == 503
    assert response.headers["content-type"].startswith(PROBLEM_MEDIA_TYPE)
    assert service.closed is True


def test_a_stream_with_no_events_is_an_empty_event_stream(
    client: TestClient, monkeypatch: pytest.MonkeyPatch, app: FastAPI
) -> None:
    """Looking at the first event before answering must not break a stream that has none."""
    service = install_generation(monkeypatch, app, StubGeneration(events=[]))

    response = client.post("/v1/query", json={"query": "q", "stream": True})

    assert response.status_code == 200
    assert frames(response.text) == []
    assert service.closed is True


def test_a_failure_before_the_stream_becomes_a_problem_document(
    client: TestClient, monkeypatch: pytest.MonkeyPatch, app: FastAPI
) -> None:
    install_generation(
        monkeypatch,
        app,
        StubGeneration(error=RetrievalError("qdrant is unreachable", retryable=True)),
    )

    response = client.post("/v1/query", json={"query": "q", "stream": False})

    assert response.status_code == 503
    assert response.headers["content-type"].startswith(PROBLEM_MEDIA_TYPE)


def test_an_empty_query_is_rejected(client: TestClient) -> None:
    assert client.post("/v1/query", json={"query": ""}).status_code == 422


def test_an_over_long_query_is_rejected(client: TestClient) -> None:
    assert client.post("/v1/query", json={"query": "x" * 9000}).status_code == 422


def test_a_missing_query_is_rejected(client: TestClient) -> None:
    assert client.post("/v1/query", json={}).status_code == 422


def test_an_out_of_range_top_k_is_rejected(client: TestClient) -> None:
    assert client.post("/v1/query", json={"query": "q", "top_k": 0}).status_code == 422
    assert client.post("/v1/query", json={"query": "q", "top_k": 9999}).status_code == 422


def test_a_misspelled_field_is_rejected_rather_than_ignored(client: TestClient) -> None:
    response = client.post("/v1/query", json={"query": "q", "top-k": 3})

    assert response.status_code == 422


def test_a_refusal_is_served_as_200_with_its_code(
    client: TestClient, monkeypatch: pytest.MonkeyPatch, app: FastAPI
) -> None:
    install_generation(
        monkeypatch,
        app,
        StubGeneration(
            answer=Answer(answer=None, faithfulness=0.38, threshold=0.7, best_candidates=[])
        ),
    )

    response = client.post("/v1/query", json={"query": "q", "stream": False})

    assert response.status_code == 200
    assert response.json()["code"] == ErrorCode.INSUFFICIENT_EVIDENCE.value


def test_retrieve_returns_every_leg_rank(
    client: TestClient, monkeypatch: pytest.MonkeyPatch, app: FastAPI
) -> None:
    chunk = ScoredChunk(
        chunk_id="c_a",
        text="body",
        payload={"source_uri": "a.pdf", "document_id": "d_1"},
        dense_rank=1,
        dense_score=0.9,
        bm25_rank=2,
        bm25_score=4.2,
        rrf_score=0.5,
        final_rank=1,
    )
    stub_shared(app)
    monkeypatch.setattr(
        query_router,
        "build_retrieval",
        lambda settings, adapter, router, tenant=None: StubRetrieval([chunk]),
    )

    body = client.post("/v1/retrieve", json={"query": "q"}).json()

    assert body["chunks"][0]["dense_rank"] == 1
    assert body["chunks"][0]["bm25_rank"] == 2
    assert body["chunks"][0]["final_rank"] == 1
    assert body["degraded"] is False


def test_retrieve_omits_chunk_text_unless_asked(
    client: TestClient, monkeypatch: pytest.MonkeyPatch, app: FastAPI
) -> None:
    chunk = ScoredChunk(chunk_id="c_a", text="secret body", rrf_score=0.5)
    stub_shared(app)
    monkeypatch.setattr(
        query_router,
        "build_retrieval",
        lambda settings, adapter, router, tenant=None: StubRetrieval([chunk]),
    )

    without = client.post("/v1/retrieve", json={"query": "q"}).json()
    with_text = client.post("/v1/retrieve", json={"query": "q", "include_chunks": True}).json()

    assert without["chunks"][0]["text"] is None
    assert with_text["chunks"][0]["text"] == "secret body"


def test_retrieve_reports_a_degraded_mode(
    client: TestClient, monkeypatch: pytest.MonkeyPatch, app: FastAPI
) -> None:
    stub_shared(app)
    monkeypatch.setattr(
        query_router,
        "build_retrieval",
        lambda settings, adapter, router, tenant=None: StubRetrieval([], HYBRID_ONLY_MODE),
    )

    body = client.post("/v1/retrieve", json={"query": "q"}).json()

    assert body["mode"] == HYBRID_ONLY_MODE
    assert body["degraded"] is True


def test_citations_survive_serialization(
    client: TestClient, monkeypatch: pytest.MonkeyPatch, app: FastAPI
) -> None:
    install_generation(
        monkeypatch,
        app,
        StubGeneration(
            answer=Answer(
                answer="a [^c_a]",
                citations=[
                    Citation(chunk_id="c_a", source="a.pdf", page=12, span=Span(start=0, end=4))
                ],
            )
        ),
    )

    citation = client.post("/v1/query", json={"query": "q", "stream": False}).json()["citations"][0]

    assert citation == {
        "chunk_id": "c_a",
        "source": "a.pdf",
        "page": 12,
        "span": {"start": 0, "end": 4},
    }
