"""One LLM adapter per application, not one per query (TASK-0466).

``build_generation`` used to call ``create_llm_adapter`` on every request, so every query
built a provider client and the router closed it again when the answer was sent: a
connection pool opened and torn down per question. These cases drive ``POST /v1/query``
through the real router and the real ``GenerationService`` with only the provider and the
backend replaced, and count the adapters the application builds and closes.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any, cast

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from fasterrag.api import dependencies as deps
from fasterrag.api.main import create_app
from fasterrag.config.schema import Settings
from fasterrag.services import composition
from fasterrag.services.journal import Journal
from fasterrag.services.traces import TraceStore
from tests.unit.api.test_query_tenancy import ACME_TEXT, EchoLLM, MemoryRouter, MemoryVectorDB


class CountingLLM(EchoLLM):
    """An echoing provider that counts how often it is closed."""

    def __init__(self, settings: Settings) -> None:
        """Start unclosed."""
        super().__init__(settings)
        self.closes = 0

    async def close(self) -> None:
        """Record the close."""
        self.closes += 1


@pytest.fixture
def built(monkeypatch: pytest.MonkeyPatch) -> list[CountingLLM]:
    """Return every LLM adapter the application builds, in order."""
    adapters: list[CountingLLM] = []

    def create(settings: Settings) -> CountingLLM:
        adapter = CountingLLM(settings)
        adapters.append(adapter)
        return adapter

    monkeypatch.setattr(composition, "create_llm_adapter", create)
    return adapters


@pytest.fixture
def application(tmp_path: Path) -> FastAPI:
    """Return the application over an in-memory backend holding one chunk."""
    store = MemoryVectorDB()
    store.seed("default", "c_acme", ACME_TEXT, "acme")
    settings = Settings.model_validate(
        {"retrieval": {"rerank": False}, "cache": {"semantic": False}}
    )
    app = create_app(settings)
    app.state.vector_db = store
    app.state.journal = Journal(tmp_path / "journal")
    app.state.traces = TraceStore(tmp_path / "traces")
    app.dependency_overrides[deps.shared_embedding_router] = MemoryRouter
    return app


@pytest.fixture
def client(application: FastAPI) -> Iterator[TestClient]:
    """Return a client whose context runs the application's lifespan."""
    with TestClient(application) as test_client:
        yield test_client


def _ask(client: TestClient, *, stream: bool) -> None:
    """Send one query and require it to succeed."""
    response = client.post("/v1/query", json={"query": "thirty days", "stream": stream})
    assert response.status_code == 200, response.text


def test_two_queries_observe_one_llm_adapter(
    client: TestClient, built: list[CountingLLM], application: FastAPI
) -> None:
    """Each query built, and then closed, a provider client of its own."""
    _ask(client, stream=False)
    _ask(client, stream=False)

    assert len(built) == 1
    assert application.state.llm is built[0]
    assert built[0].closes == 0


def test_a_streamed_query_does_not_close_the_shared_adapter(
    client: TestClient, built: list[CountingLLM]
) -> None:
    """The stream's ``finally`` releases its service; that must not release the provider."""
    _ask(client, stream=True)
    _ask(client, stream=False)
    _ask(client, stream=True)

    assert len(built) == 1
    assert built[0].closes == 0


def test_shutdown_closes_the_shared_adapter_exactly_once(
    application: FastAPI, built: list[CountingLLM]
) -> None:
    """The adapter outlives every request and is released by the lifespan alone."""
    with TestClient(application) as test_client:
        _ask(test_client, stream=False)
        _ask(test_client, stream=True)

    assert len(built) == 1
    assert built[0].closes == 1
    assert application.state.llm is None


def test_shutdown_without_a_query_builds_no_adapter(
    application: FastAPI, built: list[CountingLLM]
) -> None:
    """Nothing asked for generation, so there is nothing to build or close."""
    with TestClient(application):
        pass

    assert built == []


async def test_a_lent_adapter_outlives_its_service_and_an_owned_one_does_not(
    built: list[CountingLLM],
) -> None:
    """The facade and replay still own their provider; the API lends its own to each query."""
    settings = Settings.model_validate({"retrieval": {"rerank": False}})
    backend: Any = MemoryVectorDB()
    router: Any = MemoryRouter()
    lent = CountingLLM(settings)

    borrowing = composition.build_generation(settings, backend, router, llm=cast(Any, lent))
    owning = composition.build_generation(settings, backend, router)
    await borrowing.close()
    await owning.close()

    assert cast(Any, borrowing.llm) is lent
    assert lent.closes == 0
    assert cast(Any, owning.llm) is built[0]
    assert built[0].closes == 1
