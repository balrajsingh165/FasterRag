from collections.abc import Iterator

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from fasterrag.api.main import create_app
from fasterrag.config.schema import Settings


@pytest.fixture
def app() -> FastAPI:
    """Return an application built from schema defaults, needing no config file."""
    return create_app(Settings())


@pytest.fixture
def client(app: FastAPI) -> Iterator[TestClient]:
    """Return a client that runs the application's lifespan."""
    with TestClient(app) as test_client:
        yield test_client
