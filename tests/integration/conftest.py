import asyncio
import os
import uuid
from collections.abc import Iterator

import pytest

from fasterrag.config.schema import Settings
from fasterrag.services.provisioning import (
    container_state,
    docker_available,
    provision_qdrant,
    run_docker,
)

TEST_API_KEY = "fasterrag-integration-key"
TEST_VOLUME = "fasterrag_test_qdrant_storage"


def docker_settings() -> Settings:
    """Return settings for the system-managed container used by integration tests."""
    return Settings.model_validate(
        {"vector_db": {"mode": "docker", "docker": {"volume": TEST_VOLUME}}}
    )


@pytest.fixture(scope="session")
def qdrant(request: pytest.FixtureRequest) -> Iterator[Settings]:
    """Provision Qdrant for the session, restoring the machine's prior state after.

    Skips rather than fails when Docker is unavailable, so the suite stays runnable on
    a developer machine without Docker while remaining a hard requirement in CI.
    """
    if not asyncio.run(docker_available()):
        pytest.skip("docker daemon is not available")

    os.environ["QDRANT_API_KEY"] = TEST_API_KEY
    settings = docker_settings()
    pre_existing = asyncio.run(container_state()).exists

    asyncio.run(provision_qdrant(settings))
    yield settings

    if not pre_existing:
        asyncio.run(run_docker(["rm", "--force", "fasterrag-qdrant"]))
        asyncio.run(run_docker(["volume", "rm", "--force", TEST_VOLUME]))


@pytest.fixture
def collection_name() -> str:
    """Return a unique collection name so cases never collide."""
    return f"contract-{uuid.uuid4().hex[:12]}"
