"""Provisioning for the pgvector contract run, and the one thing it needs pytest to do.

This lives in its own package rather than beside the Qdrant suite for a single reason:
the ``pytest_asyncio_loop_factories`` hook below applies to every item under the conftest
that declares it, and the Qdrant tests must keep the default loop — grpc.aio is the half
of that suite most likely to object to being moved onto a selector loop.

psycopg's async driver refuses to run on Windows' default ``ProactorEventLoop``, so the
tests here run on a ``SelectorEventLoop``. On Linux and macOS that is already the default
loop, which makes this a no-op in CI and a fix on a developer's Windows machine.

The container is deliberately named, ported, and volumed apart from anything else
fasterRag manages: nothing here may touch the ``fasterrag-qdrant`` container, and a
developer running both suites must not have one suite delete the other's data.
"""

from __future__ import annotations

import asyncio
import os
import time
from collections.abc import Callable, Iterator

import psycopg
import pytest

from fasterrag.config.schema import Settings
from fasterrag.services.provisioning import container_state, docker_available, run_docker

PGVECTOR_CONTAINER = "fasterrag-test-pgvector"
PGVECTOR_VOLUME = "fasterrag_test_pgvector_storage"
PGVECTOR_IMAGE = "pgvector/pgvector:pg17"
PGVECTOR_DATABASE = "fasterrag"
PGVECTOR_PORT = 55432
PGVECTOR_USER = "postgres"

# A throwaway credential for a container this suite creates and destroys; it is not a
# secret in the sense docs/security.md means, and nothing outside these tests can reach it.
PGVECTOR_PASSWORD = "fasterrag-integration-pg"

PGVECTOR_DSN = (
    f"postgresql://{PGVECTOR_USER}:{PGVECTOR_PASSWORD}@localhost:{PGVECTOR_PORT}/"
    f"{PGVECTOR_DATABASE}"
)
WRONG_PASSWORD_DSN = (
    f"postgresql://{PGVECTOR_USER}:not-the-password@localhost:{PGVECTOR_PORT}/{PGVECTOR_DATABASE}"
)

DSN_VAR = "PGVECTOR_DSN"
WRONG_DSN_VAR = "FASTERRAG_WRONG_PGVECTOR_DSN"

_READY_TIMEOUT_SECONDS = 90.0
_READY_POLL_SECONDS = 0.5


def pytest_asyncio_loop_factories(
    config: pytest.Config, item: pytest.Item
) -> dict[str, Callable[[], asyncio.AbstractEventLoop]]:
    """Run every test in this package on a selector loop, which psycopg requires."""
    return {"selector": asyncio.SelectorEventLoop}


def pgvector_settings() -> Settings:
    """Return settings pointing the pgvector adapter at the test container.

    ``dsn_env`` is named explicitly rather than left to a default. It has none: every
    populated ``*_env`` field in the settings tree is collected and then required to be
    present at startup, so defaulting it would demand a PostgreSQL DSN from every Qdrant
    deployment in existence. Cross-field rule 11 asks for it only when the provider is
    pgvector, which is exactly here.
    """
    return Settings.model_validate(
        {
            "vector_db": {
                "provider": "pgvector",
                "mode": "external",
                "pgvector": {"dsn_env": DSN_VAR},
            }
        }
    )


def _wait_until_ready() -> None:
    """Block until PostgreSQL accepts connections, or fail with an actionable message.

    Synchronous on purpose: this runs from a synchronous fixture, outside any event loop,
    so it cannot inherit the loop-compatibility problem the async driver has on Windows.
    """
    deadline = time.monotonic() + _READY_TIMEOUT_SECONDS
    last: Exception | None = None
    while time.monotonic() < deadline:
        try:
            with psycopg.connect(PGVECTOR_DSN, connect_timeout=3) as connection:
                connection.execute("SELECT 1")
        except psycopg.Error as exc:
            last = exc
            time.sleep(_READY_POLL_SECONDS)
            continue
        return

    raise AssertionError(
        f"{PGVECTOR_CONTAINER} did not accept connections within "
        f"{_READY_TIMEOUT_SECONDS:.0f}s: {last}"
    )


@pytest.fixture(scope="session")
def pgvector(request: pytest.FixtureRequest) -> Iterator[Settings]:
    """Provision PostgreSQL with pgvector for the session, restoring the machine after.

    Skips rather than fails when Docker is unavailable, so the suite stays runnable on a
    developer machine without Docker while remaining a hard requirement in CI.
    """
    if not asyncio.run(docker_available()):
        pytest.skip("docker daemon is not available")

    os.environ[DSN_VAR] = PGVECTOR_DSN
    os.environ[WRONG_DSN_VAR] = WRONG_PASSWORD_DSN
    pre_existing = asyncio.run(container_state(PGVECTOR_CONTAINER)).exists
    if not pre_existing:
        started = asyncio.run(
            run_docker(
                [
                    "run",
                    "--detach",
                    "--name",
                    PGVECTOR_CONTAINER,
                    "--env",
                    "POSTGRES_PASSWORD",
                    "--env",
                    f"POSTGRES_DB={PGVECTOR_DATABASE}",
                    "--publish",
                    f"{PGVECTOR_PORT}:5432",
                    "--volume",
                    f"{PGVECTOR_VOLUME}:/var/lib/postgresql/data",
                    PGVECTOR_IMAGE,
                ],
                timeout=120.0,
                env={"POSTGRES_PASSWORD": PGVECTOR_PASSWORD},
            )
        )
        if not started.ok:
            pytest.skip(f"could not start {PGVECTOR_CONTAINER}: {started.stderr}")

    _wait_until_ready()
    yield pgvector_settings()

    if not pre_existing:
        asyncio.run(run_docker(["rm", "--force", PGVECTOR_CONTAINER]))
        asyncio.run(run_docker(["volume", "rm", "--force", PGVECTOR_VOLUME]))
