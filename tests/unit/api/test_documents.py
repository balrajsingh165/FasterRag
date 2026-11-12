"""``DELETE /v1/collections/{name}/documents/{document_id}`` through the real application.

The route, the real ``AuthMiddleware`` and the real ``DocumentService`` run against an
in-memory adapter that honours the filter grammar, so what a tenant can and cannot remove is
observed in what the backend still holds afterwards (TASK-0443).

**Tenancy is not decided here.** Whether a tenant's data lives under a prefixed collection
name or in a shared collection behind a payload filter is TASK-0339, and ingest does not
apply the prefix at all (TASK-0381). These cases pin what the route does under both layouts
— it addresses ``<tenant>__<name>`` like every other ``/v1/collections`` route, and it
removes only chunks tagged with the caller's tenant — so neither decision can turn it into
a way to delete somebody else's document.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from fasterrag.adapters.vectordb.base import CollectionSpec
from fasterrag.api.main import create_app
from fasterrag.config.schema import Settings
from fasterrag.errors import ErrorCode
from fasterrag.services.journal import Journal
from tests.unit.services.test_documents import DIMENSIONS, MemoryAdapter, RecordingCache, chunk

ACME_KEY = "acme-secret-long-enough"
GLOBEX_KEY = "globex-secret-long-enough"
READER_KEY = "reader-secret-long-enough"
KEYS = f"{ACME_KEY}:collections:acme;{GLOBEX_KEY}:collections:globex;{READER_KEY}:query:acme"


def _backend(*collections: str) -> MemoryAdapter:
    """Return a backend holding the named, empty collections."""
    backend = MemoryAdapter(Settings())
    for name in collections:
        backend.specs.append(CollectionSpec(name=name, dimensions=DIMENSIONS))
    return backend


def _application(settings: Settings, backend: MemoryAdapter, tmp_path: Path) -> FastAPI:
    """Assemble the application over the in-memory backend and a journal on disk."""
    application = create_app(settings)
    application.state.vector_db = backend
    application.state.journal = Journal(tmp_path / "journal")
    application.state.cache = RecordingCache()
    return application


@pytest.fixture
def backend() -> MemoryAdapter:
    """Return a single-tenant backend holding two documents in ``docs``."""
    store = _backend("docs")
    store.points = [chunk("d_1", index) for index in range(3)] + [chunk("d_2", 0)]
    return store


@pytest.fixture
def app(backend: MemoryAdapter, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> FastAPI:
    """Return a single-tenant application whose journal knows both documents."""
    monkeypatch.chdir(tmp_path)
    application = _application(Settings(), backend, tmp_path)
    application.state.journal.remember_content_blocking("docs", "hash-1", "d_1")
    application.state.journal.remember_content_blocking("docs", "hash-2", "d_2")
    return application


@pytest.fixture
def client(app: FastAPI) -> Iterator[TestClient]:
    """Return a client that runs the application's lifespan."""
    with TestClient(app) as test_client:
        yield test_client


def test_a_document_is_deleted_from_its_collection(
    client: TestClient, backend: MemoryAdapter
) -> None:
    """There was no route, command or method that removed a document."""
    response = client.delete("/v1/collections/docs/documents/d_1")

    assert response.status_code == 200
    assert backend.chunks_of("docs", "d_1") == []
    assert backend.chunks_of("docs", "d_2") == [0]


def test_the_response_reports_what_was_deleted(client: TestClient) -> None:
    body = client.delete("/v1/collections/docs/documents/d_1").json()

    assert set(body) == {"collection", "document_id", "known", "deleted_at"}
    assert body["collection"] == "docs"
    assert body["document_id"] == "d_1"
    assert body["known"] is True


def test_the_deleted_document_can_be_ingested_again(client: TestClient, app: FastAPI) -> None:
    client.delete("/v1/collections/docs/documents/d_1")

    journal: Journal = app.state.journal
    assert journal.is_duplicate_blocking("docs", "hash-1") is False
    assert journal.is_duplicate_blocking("docs", "hash-2") is True


def test_the_applications_own_cache_is_invalidated(client: TestClient, app: FastAPI) -> None:
    """Invalidating a cache built for the request would clear nothing a query reads."""
    client.delete("/v1/collections/docs/documents/d_1")

    assert len(app.state.cache.reasons) == 1


def test_deleting_a_document_that_does_not_exist_is_a_200_that_says_so(
    client: TestClient, backend: MemoryAdapter
) -> None:
    """Idempotent: the backend cannot report whether a filter matched, so this is not a 404."""
    response = client.delete("/v1/collections/docs/documents/d_never_ingested")

    assert response.status_code == 200
    assert response.json()["known"] is False
    assert len(backend.held("docs")) == 4


def test_deleting_twice_is_not_an_error(client: TestClient) -> None:
    first = client.delete("/v1/collections/docs/documents/d_1")
    second = client.delete("/v1/collections/docs/documents/d_1")

    assert (first.status_code, second.status_code) == (200, 200)
    assert (first.json()["known"], second.json()["known"]) == (True, False)


def test_a_missing_collection_is_a_404_problem(client: TestClient) -> None:
    response = client.delete("/v1/collections/absent/documents/d_1")

    assert response.status_code == 404
    assert response.headers["content-type"].startswith("application/problem+json")
    assert response.json()["code"] == ErrorCode.NOT_FOUND.value
    assert "'absent'" in response.json()["detail"]


@pytest.fixture
def tenanted_backend() -> MemoryAdapter:
    """Return a backend holding the same document id for two tenants, in three layouts."""
    store = _backend("acme__docs", "globex__docs", "docs")
    store.points = [
        chunk("d_1", 0, collection="acme__docs", tenant="acme"),
        chunk("d_1", 1, collection="acme__docs", tenant="acme"),
        chunk("d_1", 2, collection="acme__docs", tenant="globex"),
        chunk("d_1", 0, collection="globex__docs", tenant="globex"),
        chunk("d_1", 0, collection="docs", tenant="globex"),
    ]
    return store


@pytest.fixture
def tenanted(
    tenanted_backend: MemoryAdapter, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> Iterator[TestClient]:
    """Return a client for an application with auth and multi-tenancy on."""
    monkeypatch.setenv("FASTERRAG_API_KEY", KEYS)
    monkeypatch.chdir(tmp_path)
    settings = Settings.model_validate({"security": {"auth": True, "multi_tenancy": True}})
    with TestClient(_application(settings, tenanted_backend, tmp_path)) as client:
        yield client


def _headers(key: str, tenant: str) -> dict[str, str]:
    """Return the credentials one tenant's key sends."""
    return {"Authorization": f"Bearer {key}", "X-Tenant-ID": tenant}


def test_a_tenant_deletes_from_its_own_collection_only(
    tenanted: TestClient, tenanted_backend: MemoryAdapter
) -> None:
    """The same name and the same document id must reach one tenant's collection, not both."""
    response = tenanted.delete(
        "/v1/collections/docs/documents/d_1", headers=_headers(ACME_KEY, "acme")
    )

    assert response.status_code == 200
    assert tenanted_backend.chunks_of("globex__docs", "d_1") == [0]
    assert tenanted_backend.chunks_of("docs", "d_1") == [0]


def test_a_tenant_cannot_delete_another_tenants_chunks_in_its_own_collection(
    tenanted: TestClient, tenanted_backend: MemoryAdapter
) -> None:
    """Ingest does not scope names (TASK-0381), so a collection can hold a stranger's chunk."""
    tenanted.delete("/v1/collections/docs/documents/d_1", headers=_headers(ACME_KEY, "acme"))

    left = tenanted_backend.held("acme__docs")
    assert [point.payload["tenant"] for point in left] == ["globex"]


def test_naming_another_tenants_collection_reaches_nothing_of_theirs(
    tenanted: TestClient, tenanted_backend: MemoryAdapter
) -> None:
    """The prefix is applied to whatever is named, so the other tenant's name is not theirs."""
    response = tenanted.delete(
        "/v1/collections/globex__docs/documents/d_1", headers=_headers(ACME_KEY, "acme")
    )

    assert response.status_code == 404
    assert tenanted_backend.chunks_of("globex__docs", "d_1") == [0]


def test_the_backend_prefix_never_reaches_a_response(tenanted: TestClient) -> None:
    """A tenant named the collection ``docs`` and must never learn it is stored otherwise."""
    found = tenanted.delete(
        "/v1/collections/docs/documents/d_1", headers=_headers(ACME_KEY, "acme")
    )
    missing = tenanted.delete(
        "/v1/collections/absent/documents/d_1", headers=_headers(ACME_KEY, "acme")
    )

    assert found.json()["collection"] == "docs"
    assert "acme__" not in found.text
    assert "acme__" not in missing.text


def test_a_key_may_not_delete_for_another_tenant(
    tenanted: TestClient, tenanted_backend: MemoryAdapter
) -> None:
    response = tenanted.delete(
        "/v1/collections/docs/documents/d_1", headers=_headers(ACME_KEY, "globex")
    )

    assert response.status_code == 403
    assert response.json()["code"] == ErrorCode.TENANT_FORBIDDEN.value
    assert len(tenanted_backend.points) == 5


def test_a_key_without_the_collections_scope_is_refused(
    tenanted: TestClient, tenanted_backend: MemoryAdapter
) -> None:
    """A read-only key must not be able to remove what it can only query."""
    response = tenanted.delete(
        "/v1/collections/docs/documents/d_1", headers=_headers(READER_KEY, "acme")
    )

    assert response.status_code == 403
    assert response.json()["code"] == ErrorCode.AUTH_SCOPE.value
    assert len(tenanted_backend.points) == 5
