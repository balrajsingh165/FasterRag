from typing import Any

import pytest
from fastapi.testclient import TestClient

from fasterrag.adapters.vectordb.base import CollectionInfo
from fasterrag.api import collections as collections_router
from fasterrag.api.problems import PROBLEM_MEDIA_TYPE
from fasterrag.errors import ErrorCode, FasterRagError
from tests.unit.api.conftest import StubVectorDB


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
