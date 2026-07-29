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

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import cast

from fastapi import FastAPI
from starlette.types import ASGIApp, Receive, Scope, Send

from fasterrag import __version__
from fasterrag.api import health
from fasterrag.api.problems import install_exception_handlers
from fasterrag.config.loader import load_settings
from fasterrag.config.schema import Settings
from fasterrag.observability.logging import configure_logging, get_logger, use_trace_id

__all__ = ["CorrelationIdMiddleware", "create_app"]

_logger = get_logger(__name__)
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
    yield
    _logger.info("api stopped")


def create_app(settings: Settings | None = None) -> FastAPI:
    """Build the API application.

    Args:
        settings: Pre-validated settings. When omitted, ``config.yaml`` is loaded and
            validated, which fails fast if any key or referenced secret is missing.

    Returns:
        The configured application.

    Raises:
        ConfigError: If configuration is absent or invalid and none was supplied.
    """
    resolved = settings if settings is not None else load_settings()
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
    registry = health.ReadinessRegistry()
    registry.register(_config_check)
    app.state.readiness = registry

    app.add_middleware(CorrelationIdMiddleware)
    install_exception_handlers(app)
    app.include_router(health.router)

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
