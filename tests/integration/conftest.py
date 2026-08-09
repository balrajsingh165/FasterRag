import asyncio
import os
import time
import uuid
from collections.abc import Iterator

import pytest

from fasterrag.config.schema import Settings
from fasterrag.services.provisioning import (
    container_state,
    docker_available,
    port_is_reachable,
    provision_qdrant,
    run_docker,
)

TEST_API_KEY = "fasterrag-integration-key"
TEST_VOLUME = "fasterrag_test_qdrant_storage"

# Deliberately not `fasterrag-redis` on 6379: a developer machine may already be running one,
# and the suite must never adopt, mutate, or remove a container it did not create.
REDIS_CONTAINER = "fasterrag-test-redis"
REDIS_IMAGE = "redis:7"
REDIS_PORT = 6399
REDIS_READY_TIMEOUT_SECONDS = 60.0


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


@pytest.fixture(scope="session")
def redis_url() -> Iterator[str]:
    """Run a Redis container for the session and yield its URL.

    Skips rather than fails without Docker or without the optional client, for the same
    reason the Qdrant fixture does: the suite stays runnable on a machine that has neither
    while remaining a hard requirement in CI, where both are installed.
    """
    pytest.importorskip("redis")
    if not asyncio.run(docker_available()):
        pytest.skip("docker daemon is not available")

    pre_existing = asyncio.run(container_state(REDIS_CONTAINER)).exists
    if not pre_existing:
        started = asyncio.run(
            run_docker(
                [
                    "run",
                    "--detach",
                    "--name",
                    REDIS_CONTAINER,
                    "--publish",
                    f"{REDIS_PORT}:6379",
                    REDIS_IMAGE,
                ]
            )
        )
        if not started.ok:
            pytest.skip(f"could not start the test redis container: {started.stderr}")

    deadline = time.monotonic() + REDIS_READY_TIMEOUT_SECONDS
    while not port_is_reachable("localhost", REDIS_PORT):
        if time.monotonic() >= deadline:
            pytest.fail(f"the redis container never answered on port {REDIS_PORT}")
        time.sleep(0.5)

    yield f"redis://localhost:{REDIS_PORT}/0"

    if not pre_existing:
        asyncio.run(run_docker(["rm", "--force", REDIS_CONTAINER]))


@pytest.fixture
def redis_namespace() -> str:
    """Return a unique key namespace so cases never see each other's entries."""
    return f"fasterrag:test:{uuid.uuid4().hex[:12]}"
