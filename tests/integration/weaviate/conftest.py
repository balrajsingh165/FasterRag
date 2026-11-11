"""Where the Weaviate contract run finds a server, and why it does not start one itself.

The Qdrant, pgvector, and Chroma suites provision their own backend. This one does not,
for the reason the Milvus package gives next door: every provisioning fixture in this
repository was written against a daemon that was actually running, and there is no Docker
daemon on the machine this adapter was built on (B7.2). Weaviate's single container is
genuinely simpler than Milvus's three, but shipping an untested ``docker run`` would make
this suite *look* provisioned and skip for the wrong reason forever.

So the suite points at a Weaviate an operator already runs, named by
``FASTERRAG_TEST_WEAVIATE_HTTP`` / ``FASTERRAG_TEST_WEAVIATE_GRPC`` and defaulting to
``localhost:8080`` and ``localhost:50051``, and skips with the connection error when
nothing answers. **Both ports matter**: the v4 client refuses to work over HTTP alone, so a
server exposing only 8080 skips here rather than failing halfway through the run.

Pointing it at a real server is the whole remaining step, and it is tracked rather than
hidden (TASK-0049).

``deploy/compose.test-backends.yml`` is now that server, publishing both ports this
module needs, so this suite needs no environment variable against it::

    docker compose -f deploy/compose.test-backends.yml up -d --wait
    pytest -m integration tests/integration/weaviate -rs

CI still starts nothing, so every case here still skips there; docs/testing-strategy.md
§3.1 carries the count and what wiring it costs (TASK-0352).
"""

from __future__ import annotations

import os

import pytest

from fasterrag.config.schema import Settings

HTTP_VAR = "FASTERRAG_TEST_WEAVIATE_HTTP"
GRPC_VAR = "FASTERRAG_TEST_WEAVIATE_GRPC"
DEFAULT_HTTP = "localhost:8080"
DEFAULT_GRPC = "localhost:50051"
KEY_VAR = "FASTERRAG_TEST_WEAVIATE_KEY"
WRONG_KEY_VAR = "FASTERRAG_TEST_WEAVIATE_WRONG_KEY"
WRONG_KEY = "not-the-api-key"

_PROBE_TIMEOUT_SECONDS = 5.0


def _split(endpoint: str, fallback: int) -> tuple[str, int]:
    """Split a ``host:port`` endpoint, tolerating a bare host."""
    host, _, port = endpoint.partition(":")
    return host or "localhost", int(port) if port else fallback


def weaviate_settings(http: str, grpc: str, *, key_env: str | None) -> Settings:
    """Return settings pointing the Weaviate adapter at one server."""
    host, port = _split(http, 8080)
    _, grpc_port = _split(grpc, 50051)
    return Settings.model_validate(
        {
            "vector_db": {
                "provider": "weaviate",
                "mode": "external",
                "host": host,
                "port": port,
                "grpc_port": grpc_port,
                "https": False,
                "api_key_env": key_env,
            }
        }
    )


def _reachable(http: str, grpc: str) -> str | None:
    """Return the reason a Weaviate here cannot be used, or None when it can."""
    try:
        import weaviate
        from weaviate.classes.init import AdditionalConfig, Auth, Timeout
        from weaviate.exceptions import WeaviateBaseError
    except ImportError as exc:  # pragma: no cover
        return f"weaviate-client is not installed: {exc}"

    host, port = _split(http, 8080)
    grpc_host, grpc_port = _split(grpc, 50051)
    key = os.environ.get(KEY_VAR, "")

    # CRITICAL: every failure below is reported as a skip reason rather than raised. This
    # is a reachability probe, so an unreachable server must skip the suite, never fail
    # collection with a connection error from a fixture.
    probe_errors = (WeaviateBaseError, OSError, ValueError)
    client = None
    try:
        client = weaviate.connect_to_custom(
            http_host=host,
            http_port=port,
            http_secure=False,
            grpc_host=grpc_host,
            grpc_port=grpc_port,
            grpc_secure=False,
            auth_credentials=Auth.api_key(key) if key else None,
            additional_config=AdditionalConfig(
                timeout=Timeout(
                    init=_PROBE_TIMEOUT_SECONDS,
                    query=_PROBE_TIMEOUT_SECONDS,
                    insert=_PROBE_TIMEOUT_SECONDS,
                )
            ),
        )
        client.get_meta()
    except probe_errors as exc:
        return f"{type(exc).__name__}: {exc}"
    finally:
        if client is not None:
            client.close()
    return None


@pytest.fixture(scope="session")
def weaviate_endpoints() -> tuple[str, str]:
    """Return the HTTP and gRPC endpoints the suite talks to."""
    return (
        os.environ.get(HTTP_VAR, DEFAULT_HTTP),
        os.environ.get(GRPC_VAR, DEFAULT_GRPC),
    )


@pytest.fixture(scope="session")
def weaviate(weaviate_endpoints: tuple[str, str]) -> Settings:
    """Return settings for a reachable Weaviate, skipping with the reason when there is none."""
    http, grpc = weaviate_endpoints
    unreachable = _reachable(http, grpc)
    if unreachable is not None:
        pytest.skip(f"no Weaviate at {http} / {grpc} ({unreachable})")

    os.environ.setdefault(WRONG_KEY_VAR, WRONG_KEY)
    return weaviate_settings(http, grpc, key_env=KEY_VAR if KEY_VAR in os.environ else None)
