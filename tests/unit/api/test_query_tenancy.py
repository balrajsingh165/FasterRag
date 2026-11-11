"""What a tenant actually retrieves, through the real router and a faithful backend.

``tests/unit/api/test_query.py`` stubs retrieval out entirely, so nothing there could ever
observe which tenant's chunks come back. These cases drive ``POST /v1/retrieve`` and
``POST /v1/query`` through the real ``AuthMiddleware``, the real ``RetrievalService`` and an
in-memory adapter that honours the documented filter grammar, so the answer is observed
rather than asserted about a stub (TASK-0339).
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import AsyncIterator, Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any, Protocol

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from fasterrag.adapters.embeddings.tiering import matches
from fasterrag.adapters.llm.base import Completion
from fasterrag.adapters.vectordb.base import (
    HealthStatus,
    Point,
    ScoredPoint,
    SearchQuery,
)
from fasterrag.api import dependencies as deps
from fasterrag.api.main import create_app
from fasterrag.config.schema import Settings
from fasterrag.core.retrieval.bm25 import encode_document
from fasterrag.errors import ErrorCode, FasterRagError
from fasterrag.services.journal import Journal
from fasterrag.services.traces import TraceStore

DIMENSIONS = 8
ACME_KEY = "acme-secret"
GLOBEX_KEY = "globex-secret"
KEYS = f"{ACME_KEY}:query:acme;{GLOBEX_KEY}:query:globex"

ACME_TEXT = "the acme turbo widget warranty lasts thirty days"
GLOBEX_TEXT = "the globex merger agreement closes in thirty days"


def _vector(text: str) -> list[float]:
    """Return a deterministic unit-ish vector for a piece of text."""
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    return [digest[index] / 255.0 for index in range(DIMENSIONS)]


def _dot(left: Sequence[float], right: Sequence[float]) -> float:
    """Return the dot product of two equal-length vectors."""
    return sum(a * b for a, b in zip(left, right, strict=True))


class MemoryVectorDB:
    """An in-memory adapter faithful to the documented search and filter contract.

    Both legs read the same points and apply the same pushed-down filter, which is the
    property ``services.querying``'s module docstring requires of a real backend.
    """

    def __init__(self) -> None:
        """Start with no collections."""
        self.collections: dict[str, list[Point]] = {}
        self.seen_filters: list[Any] = []

    def seed(self, collection: str, point_id: str, text: str, tenant: str) -> None:
        """Index one chunk the way ``workers.indexer`` writes it."""
        self.collections.setdefault(collection, []).append(
            Point(
                point_id=point_id,
                collection=collection,
                vector=_vector(text),
                payload={"text": text, "tenant": tenant, "document_id": point_id},
                sparse=encode_document(text),
            )
        )

    async def search(self, query: SearchQuery) -> list[ScoredPoint]:
        """Return the matching points on whichever leg the query carries."""
        self.seen_filters.append(query.filters)
        points = self.collections.get(query.collection)
        if points is None:
            raise FasterRagError(
                f"collection {query.collection!r} does not exist",
                code=ErrorCode.NOT_FOUND,
                retryable=False,
            )

        scored: list[ScoredPoint] = []
        for point in points:
            if query.filters is not None and not matches(query.filters, point.payload):
                continue
            scored.append(
                ScoredPoint(
                    point_id=point.point_id,
                    score=self._score(query, point),
                    payload=dict(point.payload),
                )
            )

        scored.sort(key=lambda hit: (-hit.score, hit.point_id))
        return scored[: query.limit]

    @staticmethod
    def _score(query: SearchQuery, point: Point) -> float:
        """Score a point on the leg the query asked for."""
        if query.sparse is not None:
            stored = point.sparse
            if stored is None:
                return 0.0
            weights: Mapping[int, float] = dict(zip(stored.indices, stored.values, strict=True))
            return sum(weights.get(index, 0.0) for index in query.sparse.indices)
        return _dot(query.vector or (), point.vector)

    async def close(self) -> None:
        """Release nothing; the store is a dictionary."""


class MemoryEmbedder:
    """Embeds queries with the same deterministic function the store was seeded with."""

    model = "memory"
    model_version = "1"

    async def embed_query(self, text: str) -> list[float]:
        """Return the query vector."""
        return _vector(text)

    async def close(self) -> None:
        """Release nothing."""


class MemoryRouter:
    """Stands in for the tiered embedding router."""

    def __init__(self) -> None:
        """Hold one embedder."""
        self.default = MemoryEmbedder()

    async def close(self) -> None:
        """Release nothing."""


class EchoLLM:
    """Returns the prompt's context verbatim, so a leak is visible in the answer."""

    provider = "echo"
    model = "echo"

    def __init__(self, settings: Settings) -> None:
        """Ignore configuration."""
        self.settings = settings

    async def complete(self, prompt: str, *, system: str | None = None) -> Completion:
        """Return the prompt back as the answer."""
        return Completion(text=prompt, model="echo", prompt_tokens=1, completion_tokens=1)

    async def stream(self, prompt: str, *, system: str | None = None) -> AsyncIterator[str]:
        """Yield the prompt as one delta."""
        yield prompt

    async def health(self) -> HealthStatus:
        """Report healthy."""
        return HealthStatus(healthy=True, detail=None, latency_ms=0.0)

    async def close(self) -> None:
        """Release nothing."""


def _settings(**security: Any) -> Settings:
    """Return settings with auth and multi-tenancy on and no model downloads."""
    return Settings.model_validate(
        {
            "security": {"auth": True, "multi_tenancy": True, **security},
            "retrieval": {"rerank": False},
            "cache": {"semantic": False},
        }
    )


@pytest.fixture
def store() -> MemoryVectorDB:
    """Return a backend holding one chunk per tenant in three collections."""
    backend = MemoryVectorDB()
    backend.seed("default", "c_acme", ACME_TEXT, "acme")
    backend.seed("default", "c_globex", GLOBEX_TEXT, "globex")
    backend.seed("acme__docs", "c_acme", ACME_TEXT, "acme")
    backend.seed("globex__docs", "c_globex", GLOBEX_TEXT, "globex")
    return backend


def _build(
    settings: Settings, store: MemoryVectorDB, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> FastAPI:
    """Assemble the application against the in-memory backend."""
    monkeypatch.setenv("FASTERRAG_API_KEY", KEYS)
    monkeypatch.setattr(deps, "create_llm_adapter", lambda _settings: EchoLLM(_settings))
    application = create_app(settings)
    application.state.vector_db = store
    application.state.journal = Journal(tmp_path / "journal")
    application.state.traces = TraceStore(tmp_path / "traces")
    application.dependency_overrides[deps.shared_embedding_router] = MemoryRouter
    return application


@pytest.fixture
def client(
    store: MemoryVectorDB, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> Iterator[TestClient]:
    """Return a client over the default, flag-off application."""
    with TestClient(_build(_settings(), store, tmp_path, monkeypatch)) as test_client:
        yield test_client


def _headers(key: str, tenant: str) -> dict[str, str]:
    """Return the credential headers for one tenant."""
    return {"Authorization": f"Bearer {key}", "X-Tenant-ID": tenant}


class _JsonResponse(Protocol):
    """The part of a test-client response these helpers touch.

    Structural rather than `httpx.Response`: starlette returns `httpx2.Response` when
    that package is installed, which it is in CI and is not here, so naming either one
    makes mypy pass on one machine and fail on the other.
    """

    def json(self) -> Any: ...


def _body(response: _JsonResponse) -> dict[str, Any]:
    """Return the decoded JSON body."""
    decoded: dict[str, Any] = response.json()
    return decoded


def _ids(response: _JsonResponse) -> list[str]:
    """Return the chunk ids a retrieve response carried."""
    return [str(chunk["chunk_id"]) for chunk in _body(response)["chunks"]]


def test_retrieve_names_another_tenants_collection(client: TestClient) -> None:
    """``acme`` reads ``globex__docs`` by naming it, which is the disclosure."""
    response = client.post(
        "/v1/retrieve",
        json={"query": "thirty days", "collection": "globex__docs", "include_chunks": True},
        headers=_headers(ACME_KEY, "acme"),
    )

    assert response.status_code == 200
    assert _ids(response) == ["c_globex"]
    assert GLOBEX_TEXT in _body(response)["chunks"][0]["text"]


def test_retrieve_from_the_shared_default_returns_every_tenant(client: TestClient) -> None:
    """With no collection named, one unscoped collection answers for everybody."""
    response = client.post(
        "/v1/retrieve",
        json={"query": "thirty days"},
        headers=_headers(ACME_KEY, "acme"),
    )

    assert response.status_code == 200
    assert sorted(_ids(response)) == ["c_acme", "c_globex"]


def test_retrieve_pushes_no_tenant_filter(client: TestClient, store: MemoryVectorDB) -> None:
    """Neither leg receives a tenant condition, so the payload field is never read."""
    client.post("/v1/retrieve", json={"query": "thirty days"}, headers=_headers(ACME_KEY, "acme"))

    assert store.seen_filters
    assert all(sent is None for sent in store.seen_filters)


def test_query_path_leaks_the_same_way(client: TestClient) -> None:
    """``/v1/query`` takes a tenant and still retrieves across tenants."""
    response = client.post(
        "/v1/query",
        json={"query": "thirty days", "stream": False},
        headers=_headers(ACME_KEY, "acme"),
    )

    assert response.status_code == 200
    assert GLOBEX_TEXT in _body(response)["answer"]


@pytest.fixture
def scoped_client(
    store: MemoryVectorDB, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> Iterator[TestClient]:
    """Return a client over an application with ``tenant_scoped_retrieval`` on."""
    settings = _settings(tenant_scoped_retrieval=True)
    with TestClient(_build(settings, store, tmp_path, monkeypatch)) as test_client:
        yield test_client


def test_scoped_retrieve_from_the_shared_collection_sees_one_tenant(
    scoped_client: TestClient,
) -> None:
    """The shared collection answers with the caller's chunks only."""
    response = scoped_client.post(
        "/v1/retrieve",
        json={"query": "thirty days", "include_chunks": True},
        headers=_headers(ACME_KEY, "acme"),
    )

    assert response.status_code == 200
    assert _ids(response) == ["c_acme"]


def test_scoped_retrieve_of_another_tenants_collection_is_empty(
    scoped_client: TestClient,
) -> None:
    """Naming ``globex__docs`` no longer discloses it: the filter removes every hit."""
    response = scoped_client.post(
        "/v1/retrieve",
        json={"query": "thirty days", "collection": "globex__docs", "include_chunks": True},
        headers=_headers(ACME_KEY, "acme"),
    )

    assert response.status_code == 200
    assert _ids(response) == []


def test_scoped_retrieve_sends_the_tenant_to_both_legs(
    scoped_client: TestClient, store: MemoryVectorDB
) -> None:
    """Both the dense and the keyword leg carry the same tenant condition."""
    scoped_client.post(
        "/v1/retrieve", json={"query": "thirty days"}, headers=_headers(ACME_KEY, "acme")
    )

    assert len(store.seen_filters) == 2
    assert all(sent == {"tenant": "acme"} for sent in store.seen_filters)


def test_scoped_retrieve_keeps_a_caller_filter_and_adds_the_tenant(
    scoped_client: TestClient, store: MemoryVectorDB
) -> None:
    """A caller's own filter survives; the tenant is added beneath it."""
    scoped_client.post(
        "/v1/retrieve",
        json={"query": "thirty days", "filters": {"document_id": "c_acme"}},
        headers=_headers(ACME_KEY, "acme"),
    )

    assert all(sent == {"document_id": "c_acme", "tenant": "acme"} for sent in store.seen_filters)


@pytest.mark.parametrize(
    "supplied",
    [
        {"tenant": "globex"},
        {"tenant": {"$in": ["acme", "globex"]}},
        {"tenant": {"$ne": "acme"}},
    ],
)
def test_scoped_retrieve_refuses_a_caller_supplied_tenant_filter(
    scoped_client: TestClient, store: MemoryVectorDB, supplied: dict[str, Any]
) -> None:
    """A body that names the tenant field is refused, never merged or widened."""
    response = scoped_client.post(
        "/v1/retrieve",
        json={"query": "thirty days", "filters": supplied},
        headers=_headers(ACME_KEY, "acme"),
    )

    assert response.status_code == 403
    assert _body(response)["code"] == ErrorCode.TENANT_FORBIDDEN.value
    assert store.seen_filters == []


def test_scoped_query_answers_from_one_tenants_chunks(scoped_client: TestClient) -> None:
    """The generated answer's context carries the caller's chunk and nothing else."""
    response = scoped_client.post(
        "/v1/query",
        json={"query": "thirty days", "stream": False},
        headers=_headers(ACME_KEY, "acme"),
    )

    assert response.status_code == 200
    body = _body(response)
    assert ACME_TEXT in body["answer"]
    assert GLOBEX_TEXT not in body["answer"]
    assert [citation["chunk_id"] for citation in body["citations"]] == ["c_acme"]


def test_each_tenant_sees_only_its_own(scoped_client: TestClient) -> None:
    """The same request from ``globex`` returns ``globex``'s chunk, not ``acme``'s."""
    response = scoped_client.post(
        "/v1/retrieve",
        json={"query": "thirty days"},
        headers=_headers(GLOBEX_KEY, "globex"),
    )

    assert response.status_code == 200
    assert _ids(response) == ["c_globex"]


def test_the_flag_requires_the_tenancy_that_feeds_it() -> None:
    """Scoping without multi-tenancy is refused at load time, not at query time."""
    with pytest.raises(ValueError, match=re.escape("security.multi_tenancy")):
        Settings.model_validate({"security": {"auth": True, "tenant_scoped_retrieval": True}})
