"""The keyword-leg preflight on the remaining write paths, against a live Qdrant (TASK-0320).

Everything else proving these paths uses a fake that *reports* a dense-only collection. Only
a real backend shows that a dense-only collection reads back as one — the preflight's whole
premise is that ``CollectionInfo.sparse`` reflects the server rather than what creation was
asked for.

**Moved here from ``tests/unit/services`` on 2026-11-08** (TASK-0324). The session ``qdrant``
fixture lives beside this module and is unreachable from ``tests/unit``, so every case here
skipped unconditionally wherever it was run. It provisions nothing of its own: a second place
that creates containers is a second place that can leak them, so it depends on that one
fixture and skips with it when Docker is unavailable.
"""

from collections.abc import AsyncIterator, Iterator
from pathlib import Path
from typing import Any

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from fasterrag.adapters.embeddings.tiering import TieringRouter
from fasterrag.adapters.vectordb.base import CollectionSpec
from fasterrag.adapters.vectordb.qdrant import QdrantAdapter
from fasterrag.api import collections as collections_router
from fasterrag.api.main import create_app
from fasterrag.config.schema import Settings
from fasterrag.errors import ErrorCode, FasterRagError
from fasterrag.services.ingestion import IngestionService
from fasterrag.services.journal import Journal
from tests.unit.services.test_ingestion import DIMENSIONS, StubEmbedder

pytestmark = pytest.mark.integration

COLLECTION = "keyword-leg-write-paths"


def configured(*, hybrid: bool) -> Settings:
    """Return settings pointed at the managed container in external mode."""
    return Settings.model_validate(
        {
            "vector_db": {"mode": "external", "host": "localhost"},
            "retrieval": {"hybrid": hybrid},
            "embeddings": {"batch_size": 8},
        }
    )


class StubRouter:
    """Stands in for the embedding router, so no model is downloaded to create anything."""

    def __init__(self, default: StubEmbedder) -> None:
        self.default = default

    async def close(self) -> None:
        return None


@pytest.fixture
async def dense_only(qdrant: Settings) -> AsyncIterator[QdrantAdapter]:
    """Leave one dense-only collection on the session's Qdrant, and take it away after."""
    adapter = QdrantAdapter(configured(hybrid=False))
    await adapter.create_collection(
        CollectionSpec(name=COLLECTION, dimensions=DIMENSIONS, sparse=False)
    )
    yield adapter

    await adapter.drop_collection(COLLECTION)
    await adapter.close()


@pytest.fixture
def hybrid_client(
    dense_only: QdrantAdapter, monkeypatch: pytest.MonkeyPatch
) -> Iterator[TestClient]:
    """Return an API client configured for hybrid retrieval over that live backend."""
    settings = configured(hybrid=True)
    stub = StubRouter(StubEmbedder(settings))
    monkeypatch.setattr(collections_router, "create_embedding_router", lambda _: stub)

    application: FastAPI = create_app(settings)
    application.state.vector_db = dense_only
    with TestClient(application) as client:
        yield client


async def test_accept_refuses_a_live_dense_only_collection(
    dense_only: QdrantAdapter, tmp_path: Path
) -> None:
    """The fakes assert this; only Qdrant proves the collection reads back as dense-only."""
    settings = configured(hybrid=True)
    service = IngestionService(
        settings,
        journal=Journal(tmp_path / "journal"),
        adapter=dense_only,
        router=TieringRouter(StubEmbedder(settings)),
    )

    with pytest.raises(FasterRagError) as refused:
        await service.accept(["corpus/"], collection=COLLECTION)

    assert refused.value.code is ErrorCode.CONFLICT


def test_creating_over_a_live_dense_only_collection_is_a_409(hybrid_client: TestClient) -> None:
    """Qdrant accepts the create and adds nothing; the response must not claim otherwise."""
    response = hybrid_client.post("/v1/collections", json={"name": COLLECTION})

    body: dict[str, Any] = response.json()

    assert response.status_code == 409
    assert body["code"] == ErrorCode.CONFLICT.value
