"""Defined behaviour at the query edge, on every surface (TASK-0468).

A query against a collection that does not exist had no service or API test, and the code
raised depended on the adapter and its wording. A whitespace-only query passed request
validation (``min_length=1``) and was embedded and searched for. Both are now refused in
the service layer — ``NOT_FOUND`` naming the collection, ``VALIDATION_FAILED`` for blank
text — so these cases drive the service directly and then each surface that calls it: REST,
the CLI and both facades, over the in-memory doubles the REST tenancy tests use.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any, cast

import pytest
from fastapi.testclient import TestClient

from fasterrag import facade, sync
from fasterrag.adapters.vectordb.base import ScoredPoint, SearchQuery
from fasterrag.cli.commands import pipeline
from fasterrag.cli.main import main
from fasterrag.cli.output import ExitCode
from fasterrag.config.schema import Settings
from fasterrag.errors import ErrorCode, FasterRagError, RetrievalError
from fasterrag.services.composition import build_generation, build_retrieval
from fasterrag.services.journal import Journal
from fasterrag.services.traces import TraceStore
from tests.unit.api.test_query_tenancy import (
    ACME_TEXT,
    EchoLLM,
    MemoryEmbedder,
    MemoryRouter,
    MemoryVectorDB,
    _build,
)
from tests.unit.cli.conftest import write_config

BLANK_QUERIES = ["   ", "\n\t ", "　"]


class CountingEmbedder(MemoryEmbedder):
    """Counts query embeddings, so a refused query is seen to cost nothing."""

    def __init__(self) -> None:
        self.embedded: list[str] = []

    async def embed_query(self, text: str) -> list[float]:
        self.embedded.append(text)
        return await super().embed_query(text)


class CountingRouter(MemoryRouter):
    """A router over the counting embedder."""

    def __init__(self) -> None:
        self.counting = CountingEmbedder()
        self.default = self.counting


class OpaqueNotFound(MemoryVectorDB):
    """A backend whose not-found error names no collection, as a vendor's 404 may not."""

    async def search(self, query: SearchQuery) -> list[ScoredPoint]:
        if query.collection not in self.collections:
            raise FasterRagError("status 404", code=ErrorCode.NOT_FOUND, retryable=False)
        return await super().search(query)


def plain_settings() -> Settings:
    """Return settings with no reranker and no semantic cache, so nothing downloads."""
    return Settings.model_validate({"retrieval": {"rerank": False}, "cache": {"semantic": False}})


@pytest.fixture
def store() -> OpaqueNotFound:
    """Return a backend holding one chunk in the default collection."""
    backend = OpaqueNotFound()
    backend.seed("default", "c_acme", ACME_TEXT, "acme")
    return backend


@pytest.fixture
def router() -> CountingRouter:
    """Return the counting embedding router."""
    return CountingRouter()


def services(store: OpaqueNotFound, router: CountingRouter) -> tuple[Any, Any]:
    """Return a retrieval and a generation service over the doubles."""
    settings = plain_settings()
    adapter = cast(Any, store)
    embeddings = cast(Any, router)
    retrieval = build_retrieval(settings, adapter, embeddings)
    generation = build_generation(settings, adapter, embeddings, llm=cast(Any, EchoLLM(settings)))
    return retrieval, generation


async def test_retrieval_from_a_missing_collection_is_not_found_naming_it(
    store: OpaqueNotFound, router: CountingRouter
) -> None:
    """The code and its wording depended on which backend was configured."""
    retrieval, _ = services(store, router)

    with pytest.raises(FasterRagError) as caught:
        await retrieval.search("thirty days", collection="absent")

    assert caught.value.code is ErrorCode.NOT_FOUND
    assert caught.value.retryable is False
    assert "'absent'" in caught.value.detail


async def test_a_missing_collection_does_not_count_against_the_breaker(
    store: OpaqueNotFound, router: CountingRouter
) -> None:
    """A caller naming the wrong collection is not an outage of the vector database."""
    retrieval, _ = services(store, router)

    for _ in range(retrieval.breaker.failure_threshold + 1):
        with pytest.raises(FasterRagError):
            await retrieval.search("thirty days", collection="absent")

    assert [chunk.text for chunk in await retrieval.retrieve("thirty days")] == [ACME_TEXT]


async def test_another_backend_failure_keeps_its_own_code(
    store: OpaqueNotFound, router: CountingRouter, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def unreachable(query: SearchQuery) -> list[ScoredPoint]:
        raise RetrievalError("backend unreachable", retryable=True)

    monkeypatch.setattr(store, "search", unreachable)
    retrieval, _ = services(store, router)

    with pytest.raises(FasterRagError) as caught:
        await retrieval.search("thirty days")

    assert caught.value.code is ErrorCode.RETRIEVAL_FAILED


async def test_a_question_against_a_missing_collection_is_not_found(
    store: OpaqueNotFound, router: CountingRouter
) -> None:
    _, generation = services(store, router)

    with pytest.raises(FasterRagError) as answered:
        await generation.answer("thirty days", collection="absent")
    with pytest.raises(FasterRagError) as streamed:
        [event async for event in generation.stream("thirty days", collection="absent")]

    assert answered.value.code is ErrorCode.NOT_FOUND
    assert streamed.value.code is ErrorCode.NOT_FOUND


@pytest.mark.parametrize("blank", BLANK_QUERIES)
async def test_a_blank_query_is_refused_before_anything_is_embedded(
    store: OpaqueNotFound, router: CountingRouter, blank: str
) -> None:
    """Whitespace passed min_length=1, and was embedded and searched for."""
    retrieval, generation = services(store, router)

    for attempt in (
        retrieval.search(blank),
        generation.answer(blank),
    ):
        with pytest.raises(FasterRagError) as caught:
            await attempt
        assert caught.value.code is ErrorCode.VALIDATION_FAILED

    with pytest.raises(FasterRagError) as streamed:
        [event async for event in generation.stream(blank)]

    assert streamed.value.code is ErrorCode.VALIDATION_FAILED
    assert router.counting.embedded == []
    assert store.seen_filters == []


@pytest.fixture
def client(
    store: OpaqueNotFound, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> Iterator[TestClient]:
    """Return a REST client over the doubles, with auth off."""
    application = _build(plain_settings(), store, tmp_path, monkeypatch)
    with TestClient(application) as test_client:
        yield test_client


@pytest.mark.parametrize(
    ("path", "extra"),
    [("/v1/retrieve", {}), ("/v1/query", {"stream": False}), ("/v1/query", {"stream": True})],
)
def test_rest_answers_a_missing_collection_with_404_naming_it(
    client: TestClient, path: str, extra: dict[str, Any]
) -> None:
    response = client.post(path, json={"query": "thirty days", "collection": "absent", **extra})

    body = response.json()
    assert response.status_code == 404
    assert body["code"] == "NOT_FOUND"
    assert "'absent'" in body["detail"]


@pytest.mark.parametrize(
    ("path", "extra"),
    [("/v1/retrieve", {}), ("/v1/query", {"stream": False}), ("/v1/query", {"stream": True})],
)
def test_rest_refuses_a_blank_query_with_422(
    client: TestClient, path: str, extra: dict[str, Any]
) -> None:
    response = client.post(path, json={"query": "  \n ", **extra})

    assert response.status_code == 422
    assert response.json()["code"] == "VALIDATION_FAILED"


@pytest.fixture
def cli_config(store: OpaqueNotFound, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> str:
    """Return a CLI config whose query path is wired to the doubles."""
    monkeypatch.setenv("FASTERRAG_API_KEY", "test-key")
    monkeypatch.setattr(pipeline, "create_vector_db_adapter", lambda settings: store)
    monkeypatch.setattr(pipeline, "create_embedding_router", lambda settings: MemoryRouter())
    monkeypatch.setattr(pipeline, "create_llm_adapter", EchoLLM)
    monkeypatch.setattr(
        pipeline, "create_trace_store", lambda settings: TraceStore(tmp_path / "traces")
    )
    return write_config(tmp_path, "retrieval:\n  rerank: false\n")


@pytest.mark.parametrize("streamed", [True, False])
def test_the_cli_reports_a_missing_collection_as_not_found(
    cli_config: str, capsys: pytest.CaptureFixture[str], streamed: bool
) -> None:
    flags = [] if streamed else ["--no-stream"]
    code = main(["--config", cli_config, "--collection", "absent", "query", "thirty days", *flags])

    error = capsys.readouterr().err
    assert code == ExitCode.FAILURE
    assert "[NOT_FOUND]" in error
    assert "'absent'" in error


def test_the_cli_refuses_a_blank_query(cli_config: str, capsys: pytest.CaptureFixture[str]) -> None:
    code = main(["--config", cli_config, "query", "   "])

    assert code == ExitCode.FAILURE
    assert "[VALIDATION_FAILED]" in capsys.readouterr().err


def test_the_cli_still_answers_an_ordinary_query(
    cli_config: str, capsys: pytest.CaptureFixture[str]
) -> None:
    code = main(["--config", cli_config, "query", "thirty days", "--no-stream"])

    assert code == ExitCode.SUCCESS
    assert ACME_TEXT in capsys.readouterr().out


@pytest.fixture
def wired(store: OpaqueNotFound, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Wire both facades' factories to the doubles."""
    monkeypatch.setattr(facade, "create_vector_db_adapter", lambda settings: store)
    monkeypatch.setattr(facade, "create_embedding_router", lambda settings: MemoryRouter())
    monkeypatch.setattr(facade, "create_journal", lambda settings: Journal(tmp_path / "journal"))
    monkeypatch.setattr(facade, "create_llm_adapter", EchoLLM)


async def test_the_async_facade_raises_the_same_codes(wired: None) -> None:
    async with facade.FasterRag.from_settings(plain_settings()) as rag:
        with pytest.raises(FasterRagError) as missing:
            await rag.retrieve("thirty days", collection="absent")
        with pytest.raises(FasterRagError) as asked:
            await rag.query("thirty days", collection="absent")
        with pytest.raises(FasterRagError) as blank:
            await rag.query("   ")
        with pytest.raises(FasterRagError) as blank_stream:
            [event async for event in rag.query_stream("   ")]

    assert missing.value.code is ErrorCode.NOT_FOUND
    assert asked.value.code is ErrorCode.NOT_FOUND
    assert blank.value.code is ErrorCode.VALIDATION_FAILED
    assert blank_stream.value.code is ErrorCode.VALIDATION_FAILED


def test_the_blocking_facade_raises_the_same_codes(wired: None) -> None:
    with sync.FasterRag.from_settings(plain_settings()) as rag:
        with pytest.raises(FasterRagError) as missing:
            rag.query("thirty days", collection="absent")
        with pytest.raises(FasterRagError) as blank:
            rag.retrieve(" ")

    assert missing.value.code is ErrorCode.NOT_FOUND
    assert "'absent'" in missing.value.detail
    assert blank.value.code is ErrorCode.VALIDATION_FAILED
