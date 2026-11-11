"""Where the Milvus contract run finds a server, and why it does not start one itself.

The Qdrant, pgvector, and Chroma suites provision their own backend. This one does not,
deliberately. A self-hosted Milvus standalone is not one container: it wants etcd and an
object store beside it, and the single-container recipe that bundles them needs mounted
configuration files and a set of environment variables that differ between minor
versions. Every other provisioning fixture in this repository was written against a
daemon that was actually running; this one could not be, because there is no Docker
daemon on the machine the adapter was built on (B7.2). Shipping sixty lines of untested
`docker run` would make this suite *look* provisioned and skip for the wrong reason
forever, which is worse than not provisioning it.

So the suite points at a Milvus an operator already runs, named by
``FASTERRAG_TEST_MILVUS_URI`` or defaulting to ``localhost:19530``, and skips with the
connection error when nothing answers. Pointing it at a real server is the whole
remaining step, and it is tracked rather than hidden (TASK-0049).

``deploy/compose.test-backends.yml`` is now that server, pinned and published on the
default port above, so this suite needs no environment variable against it::

    docker compose -f deploy/compose.test-backends.yml up -d --wait
    pytest -m integration tests/integration/milvus -rs

That stack has now been run (2026-11-11): all 46 cases here pass against a real Milvus,
and getting there needed one adapter fix, because no collection could be created at all.
docs/testing-strategy.md §3.1.1 has it.

CI still starts nothing, so every case here still skips there; docs/testing-strategy.md
§3.1 carries the count and what wiring it costs (TASK-0352).
"""

from __future__ import annotations

import os
from urllib.parse import urlsplit

import pytest

from fasterrag.config.schema import Settings

MILVUS_URI_VAR = "FASTERRAG_TEST_MILVUS_URI"
DEFAULT_MILVUS_URI = "http://localhost:19530"
TOKEN_VAR = "FASTERRAG_TEST_MILVUS_TOKEN"
WRONG_TOKEN_VAR = "FASTERRAG_TEST_MILVUS_WRONG_TOKEN"
WRONG_TOKEN = "root:not-the-password"

_PROBE_TIMEOUT_SECONDS = 5.0


def milvus_settings(uri: str, *, token_env: str | None) -> Settings:
    """Return settings pointing the Milvus adapter at one server."""
    split = urlsplit(uri)
    return Settings.model_validate(
        {
            "vector_db": {
                "provider": "milvus",
                "mode": "external",
                "host": split.hostname or "localhost",
                "port": split.port or 19530,
                "https": split.scheme == "https",
                "api_key_env": token_env,
            }
        }
    )


def _reachable(uri: str) -> str | None:
    """Return the reason a Milvus at ``uri`` cannot be used, or None when it can."""
    try:
        from pymilvus import MilvusClient
        from pymilvus import exceptions as milvus_errors
    except ImportError as exc:  # pragma: no cover
        return f"pymilvus is not installed: {exc}"

    # CRITICAL: every failure below is reported as a skip reason rather than raised. This
    # is a reachability probe, so an unreachable server must skip the suite, never fail
    # collection with a connection error from a fixture.
    probe_errors = (milvus_errors.MilvusException, OSError, ValueError)
    try:
        client = MilvusClient(uri=uri, token=os.environ.get(TOKEN_VAR, ""))
    except probe_errors as exc:
        return f"{type(exc).__name__}: {exc}"

    try:
        client.get_server_version(timeout=_PROBE_TIMEOUT_SECONDS)
    except probe_errors as exc:
        return f"{type(exc).__name__}: {exc}"
    finally:
        client.close()
    return None


@pytest.fixture(scope="session")
def milvus_uri() -> str:
    """Return the Milvus the suite talks to."""
    return os.environ.get(MILVUS_URI_VAR, DEFAULT_MILVUS_URI)


@pytest.fixture(scope="session")
def milvus(milvus_uri: str) -> Settings:
    """Return settings for a reachable Milvus, skipping with the reason when there is none."""
    unreachable = _reachable(milvus_uri)
    if unreachable is not None:
        pytest.skip(f"no Milvus at {milvus_uri} ({unreachable})")

    os.environ.setdefault(WRONG_TOKEN_VAR, WRONG_TOKEN)
    return milvus_settings(milvus_uri, token_env=TOKEN_VAR if TOKEN_VAR in os.environ else None)
