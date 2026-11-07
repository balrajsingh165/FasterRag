"""The collection endpoints, over a stubbed backend.

The create case carries TASK-0320: the response must describe the collection the backend
actually has, not the configuration the request was made under. ``ExistingCollection`` is
the fake that makes the difference visible — it mirrors Qdrant, which verifies an existing
collection and returns without adding the keyword leg a create asked for.

The tenant cases carry TASK-0324: the create must be audited under the scoped backend name
and refused under the caller's own, and only a response body shows both at once.
"""

from typing import Any

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from fasterrag.adapters.vectordb.base import CollectionInfo, CollectionSpec
from fasterrag.api import collections as collections_router
from fasterrag.api.main import create_app
from fasterrag.api.problems import PROBLEM_MEDIA_TYPE
from fasterrag.config.schema import Settings
from fasterrag.errors import ErrorCode, FasterRagError
from fasterrag.services.tenancy import SEPARATOR
from tests.unit.api.conftest import StubVectorDB

TENANT_SECRET = "acme-key"


class ExistingCollection(StubVectorDB):
    """A backend already holding a collection, whose layout a create leaves untouched."""

    def __init__(self, *, sparse: bool, name: str = "docs") -> None:
        super().__init__()
        self.collections = [
            CollectionInfo(name=name, vectors=10, dimensions=384, distance="cosine", sparse=sparse)
        ]

    async def create_collection(self, spec: CollectionSpec) -> None:
        if any(info.name == spec.name for info in self.collections):
            self.created.append(spec)
            return
        await super().create_collection(spec)


class StubEmbedder:
    """Reports a fixed vector size without loading anything."""

    def __init__(self, dimensions: int | None = 384) -> None:
        self.dimensions = dimensions
        self.probes = 0

    async def embed_query(self, text: str) -> list[float]:
        self.probes += 1
        return [0.0] * 8

    async def close(self) -> None:
        return None


class StubRouter:
    """Stands in for the tiered embedding router."""

    def __init__(self, default: StubEmbedder) -> None:
        self.default = default
        self.closed = False

    async def close(self) -> None:
        self.closed = True


@pytest.fixture
def embedder(monkeypatch: pytest.MonkeyPatch) -> StubEmbedder:
    built = StubEmbedder()
    monkeypatch.setattr(
        collections_router, "build_embedding_router", lambda settings: StubRouter(built)
    )
    return built


def test_an_empty_backend_lists_nothing(client: TestClient) -> None:
    response = client.get("/v1/collections")

    assert response.status_code == 200
    assert response.json() == {"collections": []}


def test_every_collection_is_listed(client: TestClient, vector_db: StubVectorDB) -> None:
    vector_db.collections = [
        CollectionInfo(name="docs", vectors=42, dimensions=384, distance="cosine", sparse=True)
    ]

    body = client.get("/v1/collections").json()

    assert body["collections"] == [
        {
            "name": "docs",
            "vectors": 42,
            "dimensions": 384,
            "distance": "cosine",
            "sparse": True,
        }
    ]


def test_creating_a_collection_returns_201(
    client: TestClient, vector_db: StubVectorDB, embedder: StubEmbedder
) -> None:
    response = client.post("/v1/collections", json={"name": "docs"})

    assert response.status_code == 201
    assert vector_db.created[0].name == "docs"
    assert vector_db.created[0].dimensions == 384


def test_the_collection_is_sized_from_the_configured_model_not_the_request(
    client: TestClient, vector_db: StubVectorDB, embedder: StubEmbedder
) -> None:
    client.post("/v1/collections", json={"name": "docs", "distance": "dot"})

    assert vector_db.created[0].dimensions == embedder.dimensions
    assert vector_db.created[0].distance == "dot"


def test_a_model_with_no_known_size_is_probed_once(
    client: TestClient, vector_db: StubVectorDB, monkeypatch: pytest.MonkeyPatch
) -> None:
    unknown = StubEmbedder(dimensions=None)
    monkeypatch.setattr(
        collections_router, "build_embedding_router", lambda settings: StubRouter(unknown)
    )

    client.post("/v1/collections", json={"name": "docs"})

    assert unknown.probes == 1
    assert vector_db.created[0].dimensions == 8


def test_an_unknown_field_is_rejected(client: TestClient, embedder: StubEmbedder) -> None:
    response = client.post("/v1/collections", json={"name": "docs", "dimensions": 512})

    assert response.status_code == 422


def test_an_unknown_distance_is_rejected(client: TestClient, embedder: StubEmbedder) -> None:
    response = client.post("/v1/collections", json={"name": "docs", "distance": "manhattan"})

    assert response.status_code == 422


def test_an_empty_name_is_rejected(client: TestClient, embedder: StubEmbedder) -> None:
    assert client.post("/v1/collections", json={"name": ""}).status_code == 422


def test_one_collection_is_returned_by_name(client: TestClient, vector_db: StubVectorDB) -> None:
    vector_db.collections = [CollectionInfo(name="docs", vectors=1)]

    assert client.get("/v1/collections/docs").json()["name"] == "docs"


def test_an_unknown_collection_is_a_problem_document(client: TestClient) -> None:
    response = client.get("/v1/collections/absent")

    assert response.status_code == 404
    assert response.headers["content-type"].startswith(PROBLEM_MEDIA_TYPE)
    assert response.json()["code"] == ErrorCode.NOT_FOUND.value


def test_deleting_without_force_is_refused(client: TestClient, vector_db: StubVectorDB) -> None:
    vector_db.collections = [CollectionInfo(name="docs", vectors=1)]

    response = client.delete("/v1/collections/docs")

    assert response.status_code == 422
    assert vector_db.dropped == []
    assert "force=true" in response.json()["detail"]


def test_deleting_with_force_returns_204(client: TestClient, vector_db: StubVectorDB) -> None:
    vector_db.collections = [CollectionInfo(name="docs", vectors=1)]

    response = client.delete("/v1/collections/docs?force=true")

    assert response.status_code == 204
    assert vector_db.dropped == ["docs"]


def test_deleting_an_absent_collection_is_404(client: TestClient) -> None:
    response = client.delete("/v1/collections/absent?force=true")

    assert response.status_code == 404


def test_a_backend_failure_becomes_a_problem_document(
    client: TestClient, vector_db: StubVectorDB
) -> None:
    vector_db.error = FasterRagError(
        "qdrant is unreachable", code=ErrorCode.RETRIEVAL_FAILED, retryable=True
    )

    response = client.get("/v1/collections")

    assert response.status_code == 503
    assert response.headers["content-type"].startswith(PROBLEM_MEDIA_TYPE)


def test_the_created_collection_carries_the_sparse_leg_when_hybrid(
    client: TestClient, vector_db: StubVectorDB, embedder: StubEmbedder
) -> None:
    body: dict[str, Any] = client.post("/v1/collections", json={"name": "docs"}).json()

    assert body["sparse"] is True
    assert vector_db.created[0].sparse is True


def test_a_create_that_adds_no_keyword_leg_is_refused_rather_than_reported_as_hybrid(
    client: TestClient, app: FastAPI, embedder: StubEmbedder
) -> None:
    """TASK-0320: a 201 saying ``"sparse": true`` over a dense-only collection is untrue."""
    app.state.vector_db = ExistingCollection(sparse=False)

    response = client.post("/v1/collections", json={"name": "docs"})

    assert response.status_code == 409
    assert response.headers["content-type"].startswith(PROBLEM_MEDIA_TYPE)
    assert response.json()["code"] == ErrorCode.CONFLICT.value


def test_the_refusal_names_the_collection_the_rebuild_and_the_opt_out(
    client: TestClient, app: FastAPI, embedder: StubEmbedder
) -> None:
    """The same detail ``POST /v1/ingest`` returns, because it is the same mistake."""
    app.state.vector_db = ExistingCollection(sparse=False)

    detail = client.post("/v1/collections", json={"name": "docs"}).json()["detail"]

    assert "'docs'" in detail
    assert "fasterrag index reembed" in detail
    assert "retrieval.hybrid" in detail


def test_the_response_reports_the_live_keyword_leg_not_the_configuration(
    client: TestClient, app: FastAPI, embedder: StubEmbedder
) -> None:
    """Dense-only configuration over a collection that has a keyword leg reports the leg."""
    app.state.settings = Settings.model_validate({"retrieval": {"hybrid": False}})
    app.state.vector_db = ExistingCollection(sparse=True)

    body: dict[str, Any] = client.post("/v1/collections", json={"name": "docs"}).json()

    assert body["sparse"] is True


def test_a_tenant_refusal_never_names_the_backend_prefixed_collection(
    monkeypatch: pytest.MonkeyPatch, embedder: StubEmbedder
) -> None:
    """TASK-0324: the 409 detail is a response body, and the prefix is not the tenant's.

    Asserted on the body rather than on the audit, because the audit is the only place the
    scoped name is correct — the leak exists precisely where the two namings meet.
    """
    monkeypatch.setenv("FASTERRAG_API_KEY", f"{TENANT_SECRET}:admin:acme")
    settings = Settings.model_validate({"security": {"auth": True, "multi_tenancy": True}})
    app = create_app(settings)
    app.state.vector_db = ExistingCollection(sparse=False, name=f"acme{SEPARATOR}docs")

    with TestClient(app) as client:
        response = client.post(
            "/v1/collections",
            json={"name": "docs"},
            headers={"Authorization": f"Bearer {TENANT_SECRET}", "X-Tenant-ID": "acme"},
        )

    body: dict[str, Any] = response.json()
    assert response.status_code == 409
    assert body["code"] == ErrorCode.CONFLICT.value
    assert SEPARATOR not in body["detail"]
    assert "'docs'" in body["detail"]
    assert "fasterrag index reembed docs" in body["detail"]


def test_a_tenant_create_still_audits_the_scoped_collection(
    monkeypatch: pytest.MonkeyPatch, embedder: StubEmbedder
) -> None:
    """Hiding the prefix must not move the audit onto the unscoped name nobody created."""
    monkeypatch.setenv("FASTERRAG_API_KEY", f"{TENANT_SECRET}:admin:acme")
    settings = Settings.model_validate({"security": {"auth": True, "multi_tenancy": True}})
    app = create_app(settings)
    backend = ExistingCollection(sparse=False, name="docs")
    app.state.vector_db = backend

    with TestClient(app) as client:
        response = client.post(
            "/v1/collections",
            json={"name": "docs"},
            headers={"Authorization": f"Bearer {TENANT_SECRET}", "X-Tenant-ID": "acme"},
        )

    assert response.status_code == 201
    assert backend.created[0].name == f"acme{SEPARATOR}docs"


def test_a_dense_only_create_is_not_refused_when_hybrid_is_off(
    client: TestClient, app: FastAPI, embedder: StubEmbedder
) -> None:
    """Only the mismatch is a conflict; a dense-only deployment creates dense-only."""
    app.state.settings = Settings.model_validate({"retrieval": {"hybrid": False}})
    app.state.vector_db = ExistingCollection(sparse=False)

    response = client.post("/v1/collections", json={"name": "docs"})

    assert response.status_code == 201
    assert response.json()["sparse"] is False
