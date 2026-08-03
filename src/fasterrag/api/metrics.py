"""The metrics scrape endpoint and the RED middleware behind it.

``GET /metrics`` serves the Prometheus text exposition format, which is what the provisioned
Grafana datasource reads (``docs/observability.md`` §6). It is unauthenticated and carries no
request data — only aggregate counters — because a scrape endpoint that needs a credential is
one an operator will eventually disable.

The middleware records rate, errors, and duration for every request. It sits in the ASGI
chain rather than in each router so a new endpoint is instrumented by existing, and it reads
the status from the response rather than from the handler's return, so an exception turned
into a problem document is still counted as the status the client actually saw.
"""

from __future__ import annotations

import time
from typing import Final

from fastapi import APIRouter, Response
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from fasterrag.observability import metrics

__all__ = ["METRICS_MEDIA_TYPE", "MetricsMiddleware", "router"]

METRICS_MEDIA_TYPE: Final = "text/plain; version=0.0.4; charset=utf-8"

_UNKNOWN_TENANT: Final = "none"
_SERVER_ERROR: Final = 500

router = APIRouter(tags=["observability"])


def _tenant_of(scope: Scope) -> str:
    """Return the request's tenant label.

    ``none`` covers both a single-tenant deployment and an unauthenticated request, which is
    correct: neither belongs to a tenant, and inventing a label for them would split every
    series in a deployment that has no tenants at all.
    """
    tenant = scope.get("state", {}).get("tenant")
    return str(tenant) if tenant else _UNKNOWN_TENANT


def _endpoint_of(scope: Scope) -> str:
    """Return the route template for a request, never the concrete path.

    A job id or a collection name in the label would create one time series per value, which
    is the classic way to melt a Prometheus server. The template keeps cardinality bounded.
    """
    route = scope.get("route")
    path = getattr(route, "path", None)
    return str(path or scope.get("path", "unknown"))


class MetricsMiddleware:
    """Record RED metrics for every HTTP request."""

    def __init__(self, app: ASGIApp) -> None:
        """Wrap the downstream ASGI application."""
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        """Time the request and count its outcome."""
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        started = time.perf_counter()
        status_holder = {"status": _SERVER_ERROR}

        async def observing_send(message: Message) -> None:
            """Capture the status line as it goes out."""
            if message["type"] == "http.response.start":
                status_holder["status"] = int(message["status"])
            await send(message)

        try:
            await self.app(scope, receive, observing_send)
        finally:
            # CRITICAL: the route is only resolved once the application has matched it, so
            # the endpoint label has to be read here rather than before the call. Reading it
            # early would label every series with the raw path and defeat the templating.
            endpoint = _endpoint_of(scope)
            status = status_holder["status"]
            elapsed = time.perf_counter() - started

            # The tenant is read from the scope the auth middleware populated, not from the
            # header. Auth runs first and has already decided who the caller is; re-reading
            # the header here would let an unauthenticated request label its own series.
            metrics.REQUESTS.increment(
                endpoint=endpoint,
                method=str(scope.get("method", "GET")),
                status=str(status),
                tenant=_tenant_of(scope),
            )
            metrics.REQUEST_DURATION.observe(elapsed, endpoint=endpoint)


@router.get("/metrics", include_in_schema=False)
async def scrape() -> Response:
    """Serve the metrics catalogue for a Prometheus scrape."""
    return Response(content=metrics.render(), media_type=METRICS_MEDIA_TYPE)
