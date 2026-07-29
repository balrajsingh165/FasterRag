from collections.abc import Iterator

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from fasterrag.adapters.vectordb.base import HealthStatus
from fasterrag.api.main import create_app
from fasterrag.config.schema import Settings


class StubVectorDB:
    """Stands in for a connected vector database adapter."""

    def __init__(self, healthy: bool = True, detail: str | None = None) -> None:
        self.healthy = healthy
        self.detail = detail
        self.closed = False

    async def health(self) -> HealthStatus:
        return HealthStatus(healthy=self.healthy, detail=self.detail, latency_ms=1.0)

    async def close(self) -> None:
        self.closed = True


@pytest.fixture
def vector_db() -> StubVectorDB:
    """Return the stub adapter injected into the application."""
    return StubVectorDB()


@pytest.fixture
def app(vector_db: StubVectorDB) -> FastAPI:
    """Return an application built from schema defaults with a stubbed backend."""
    application = create_app(Settings())
    application.state.vector_db = vector_db
    return application


@pytest.fixture
def client(app: FastAPI) -> Iterator[TestClient]:
    """Return a client that runs the application's lifespan."""
    with TestClient(app) as test_client:
        yield test_client
