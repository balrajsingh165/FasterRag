"""Application factory, lifespan, and middleware.

The API tier is I/O-bound only: it validates, delegates to a service, and returns.
CPU-heavy work belongs to the worker pools, so the event loop is never blocked
(``docs/architecture.md`` §4). Endpoints land slice by slice; ``/healthz`` and
``/readyz`` are the first two.

Interactive API documentation is deliberately not served. Swagger UI and ReDoc are web
interfaces that can drive the RAG, and the control plane is programmatic only — REST
API, CLI, and library (``docs/adr/ADR-0005``). The machine-readable ``/openapi.json``
schema is still published for client generation.
"""

from __future__ import annotations

import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import cast

from fastapi import FastAPI
from starlette.types import ASGIApp, Receive, Scope, Send

from fasterrag import __version__
from fasterrag.adapters.vectordb.base import VectorDBAdapter
from fasterrag.adapters.vectordb.factory import create_vector_db_adapter
from fasterrag.api import admin, collections, health, ingest, metrics, query, traces
from fasterrag.api.auth import AuthMiddleware
from fasterrag.api.limits import BodyLimitMiddleware
from fasterrag.api.problems import install_exception_handlers
from fasterrag.config.loader import DEFAULT_CONFIG_PATH, load_settings
from fasterrag.config.schema import Settings
from fasterrag.errors import FasterRagError
from fasterrag.observability.logging import configure_logging, get_logger, use_trace_id
from fasterrag.services.traces import create_metric_pusher

__all__ = ["CONFIG_PATH_VAR", "CorrelationIdMiddleware", "create_app"]

# CRITICAL: the only way `fasterrag serve --reload --config other.yaml` can reach the
# application. Reload builds it in a child process from an import string, which carries no
# arguments, so without this the child would load ./config.yaml and silently ignore --config.
CONFIG_PATH_VAR = "FASTERRAG_CONFIG"

_logger = get_logger(__name__)


def _configured_path() -> Path:
    """Return the config file to load when no settings were passed in."""
    return Path(os.environ.get(CONFIG_PATH_VAR) or DEFAULT_CONFIG_PATH)


_default_app: FastAPI | None = None


class CorrelationIdMiddleware:
    """Bind a fresh correlation id to every HTTP request.

    Implemented as pure ASGI middleware rather than ``BaseHTTPMiddleware`` so that the
    streaming responses introduced in later slices are not buffered.
    """

    def __init__(self, app: ASGIApp) -> None:
        """Wrap the downstream ASGI application."""
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        """Bind a trace id for the duration of an HTTP request."""
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        with use_trace_id():
            await self.app(scope, receive, send)


async def _config_check() -> health.DependencyStatus:
    """Report configuration validity, which startup already proved."""
    return health.DependencyStatus(
        name="config",
        ready=True,
        detail="validated at startup",
    )


def _vector_db_check(app: FastAPI) -> health.ReadinessCheck:
    """Build the readiness check that actually asks the vector database."""

    async def check() -> health.DependencyStatus:
        adapter: VectorDBAdapter | None = getattr(app.state, "vector_db", None)
        if adapter is None:
            return health.DependencyStatus(
                name="vector_db",
                ready=False,
                detail="adapter has not started yet",
            )

        status = await adapter.health()
        return health.DependencyStatus(
            name="vector_db",
            ready=status.healthy,
            detail=status.detail or f"reachable in {status.latency_ms} ms",
        )

    return check


async def provision_enabled_observability(settings: Settings) -> None:
    """Stand up whichever observability stacks the configuration turns on.

    ``docs/observability.md`` frames the toggle itself as the trigger — "``langfuse: true``
    → doctor preflight → ``docker compose up -d``" — so requiring an operator to also run
    ``fasterrag provision`` would make the documentation wrong rather than incomplete. Both
    provisioners converge, so this is a no-op once the stack is up.

    A provisioning failure is logged and swallowed on purpose, and this is the one place
    that trade is right: observability is not on the request path, and refusing to serve
    queries because a dashboard would not start inverts the dependency. The toggle's own
    ``fasterrag provision`` command still surfaces the error in full.
    """
    if not (settings.observability.grafana or settings.observability.langfuse):
        return

    from fasterrag.services.grafana import provision_grafana
    from fasterrag.services.langfuse import provision_langfuse

    for enabled, name, provision in (
        (settings.observability.grafana, "grafana", provision_grafana),
        (settings.observability.langfuse, "langfuse", provision_langfuse),
    ):
        if not enabled:
            continue
        try:
            result = await provision(settings)
            _logger.info(
                "observability stack provisioned at startup",
                extra={"tool": name, "status": result.status, "url": result.url},
            )
        except FasterRagError as exc:
            _logger.warning(
                "observability stack could not be provisioned; the api is serving anyway",
                extra={"tool": name, "code": exc.code.value, "detail": exc.detail},
            )


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Start and stop process-wide resources.

    Later slices connect adapters, run their health checks, and start the in-process
    worker pools here, draining queues and flushing the journal on shutdown.
    """
    settings = cast(Settings, app.state.settings)
    _logger.info(
        "api starting",
        extra={
            "version": __version__,
            "vector_db_provider": settings.vector_db.provider,
            "embeddings_provider": settings.embeddings.provider,
            "llm_provider": settings.llm.provider,
        },
    )

    owns_adapter = getattr(app.state, "vector_db", None) is None
    if owns_adapter:
        app.state.vector_db = create_vector_db_adapter(settings)

    await provision_enabled_observability(settings)

    # Metrics are pushed on an interval rather than per request, so the pusher's lifetime
    # is the process's. Started here and stopped below, which is also the only place that
    # can send a final snapshot before the counters go with the process.
    app.state.metric_pusher = create_metric_pusher(settings)
    if app.state.metric_pusher is not None:
        app.state.metric_pusher.start()

    try:
        yield
    finally:
        pusher = getattr(app.state, "metric_pusher", None)
        if pusher is not None:
            await pusher.close()
            app.state.metric_pusher = None

        # CRITICAL: the embedding router and the semantic cache are process-scoped, built
        # lazily on first use and released only here. Closing either from a request handler
        # would unload a model or drop a backend connection that concurrent requests are
        # still using.
        embeddings = getattr(app.state, "embeddings", None)
        if embeddings is not None:
            await embeddings.close()
            app.state.embeddings = None

        cache = getattr(app.state, "cache", None)
        if cache is not None:
            await cache.close()
            app.state.cache = None

        if owns_adapter:
            adapter: VectorDBAdapter = app.state.vector_db
            await adapter.close()
            app.state.vector_db = None
        _logger.info("api stopped")


def create_app(settings: Settings | None = None) -> FastAPI:
    """Build the API application.

    Args:
        settings: Pre-validated settings. When omitted, the file named by
            ``FASTERRAG_CONFIG`` is loaded, falling back to ``config.yaml``; either way it
            is validated, which fails fast if any key or referenced secret is missing.

    Returns:
        The configured application.

    Raises:
        ConfigError: If configuration is absent or invalid and none was supplied.
    """
    resolved = settings if settings is not None else load_settings(_configured_path())
    configure_logging(resolved.app.log_level)

    app = FastAPI(
        title="fasterRag",
        version=__version__,
        lifespan=lifespan,
        docs_url=None,
        redoc_url=None,
        openapi_url="/openapi.json",
    )

    app.state.settings = resolved
    app.state.vector_db = None
    registry = health.ReadinessRegistry()
    registry.register(_config_check)
    registry.register(_vector_db_check(app))
    app.state.readiness = registry

    # CRITICAL: order matters and is the reverse of the call order. Starlette wraps in
    # reverse, so listing auth last runs it *first* — an unauthenticated request is refused
    # before it reaches metrics or a handler, and a rejected key cannot inflate per-endpoint
    # counters or leave a trace id in the logs of an endpoint it never reached.
    app.add_middleware(metrics.MetricsMiddleware)
    app.add_middleware(CorrelationIdMiddleware)
    app.add_middleware(AuthMiddleware, settings=resolved)
    # CRITICAL: added last, so Starlette wraps it outermost and it runs *first*. An
    # oversized body has to be refused before authentication, metrics, or routing spend
    # anything on it — checking after auth would mean an unauthenticated caller could still
    # make the server read the whole body.
    app.add_middleware(BodyLimitMiddleware, settings=resolved)
    install_exception_handlers(app)
    app.include_router(health.router)
    app.include_router(ingest.router)
    app.include_router(query.router)
    app.include_router(collections.router)
    app.include_router(admin.router)
    app.include_router(traces.router)
    app.include_router(metrics.router)

    return app


def __getattr__(name: str) -> FastAPI:
    """Build the default application on first access to ``app``.

    Deferring construction keeps importing this module free of side effects while still
    supporting the documented ``uvicorn fasterrag.api.main:app`` entry point.
    """
    if name == "app":
        global _default_app
        if _default_app is None:
            _default_app = create_app()
        return _default_app
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
