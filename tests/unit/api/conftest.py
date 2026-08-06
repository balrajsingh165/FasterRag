from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from fasterrag.adapters.vectordb.base import CollectionInfo, CollectionSpec, HealthStatus
from fasterrag.api.main import create_app
from fasterrag.config.schema import Settings
from fasterrag.services.journal import Journal
from fasterrag.services.traces import TraceStore


class StubVectorDB:
    """Stands in for a connected vector database adapter."""

    def __init__(self, healthy: bool = True, detail: str | None = None) -> None:
        self.healthy = healthy
        self.detail = detail
        self.closed = False
        self.collections: list[CollectionInfo] = []
        self.created: list[CollectionSpec] = []
        self.dropped: list[str] = []
        self.error: Exception | None = None
        self.aliases: dict[str, str] = {}
        self.snapshots: dict[str, list[str]] = {}
        self.restored: list[tuple[str, str]] = []

    async def health(self) -> HealthStatus:
        return HealthStatus(healthy=self.healthy, detail=self.detail, latency_ms=1.0)

    async def list_collections(self) -> list[CollectionInfo]:
        if self.error is not None:
            raise self.error
        return list(self.collections)

    async def create_collection(self, spec: CollectionSpec) -> None:
        if self.error is not None:
            raise self.error
        self.created.append(spec)
        self.collections.append(
            CollectionInfo(
                name=spec.name,
                vectors=0,
                dimensions=spec.dimensions,
                distance=spec.distance,
                sparse=spec.sparse,
            )
        )

    async def snapshot(self, collection: str) -> str:
        self.snapshots.setdefault(collection, []).append(f"{collection}-snap")
        return f"{collection}-snap"

    async def list_snapshots(self, collection: str) -> list[str]:
        return list(self.snapshots.get(collection, []))

    async def delete_snapshot(self, collection: str, snapshot: str) -> bool:
        return True

    async def restore_snapshot(self, collection: str, snapshot: str) -> None:
        self.restored.append((collection, snapshot))

    async def set_alias(self, alias: str, collection: str) -> None:
        self.aliases[alias] = collection

    async def alias_target(self, alias: str) -> str | None:
        return self.aliases.get(alias)

    async def delete_alias(self, alias: str) -> bool:
        return self.aliases.pop(alias, None) is not None

    async def drop_collection(self, name: str) -> bool:
        self.dropped.append(name)
        before = len(self.collections)
        self.collections = [info for info in self.collections if info.name != name]
        return len(self.collections) < before

    async def close(self) -> None:
        self.closed = True


@pytest.fixture
def vector_db() -> StubVectorDB:
    """Return the stub adapter injected into the application."""
    return StubVectorDB()


@pytest.fixture
def app(vector_db: StubVectorDB, tmp_path: Path) -> FastAPI:
    """Return an application built from schema defaults with a stubbed backend.

    The journal is rooted in a temporary directory so a test never writes job state into the
    working tree, and so each test starts with no jobs.
    """
    application = create_app(Settings())
    application.state.vector_db = vector_db
    application.state.journal = Journal(tmp_path / "journal")
    application.state.traces = TraceStore(tmp_path / "traces")
    return application


@pytest.fixture
def client(app: FastAPI) -> Iterator[TestClient]:
    """Return a client that runs the application's lifespan."""
    with TestClient(app) as test_client:
        yield test_client
