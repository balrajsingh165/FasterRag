"""The read-only observability dashboard (``observability.dashboard: true``).

A separate ASGI application on its own port (``observability.dashboard_port``), never mounted
into the control-plane API. Two reasons, and both are structural rather than stylistic:

* **It cannot mutate anything.** Every route is a ``GET``, and a test asserts that the
  application declares no other method. "We only added read endpoints" is a promise; an
  application with no write routes at all is a property.
* **It is separately bindable.** The dashboard shows prompts, responses, and retrieved
  corpus text, so an operator needs to expose it on a different interface from the API —
  usually an internal one (``docs/security.md`` §7). Sharing a port would remove that choice.

It reads what already exists: the metrics registry and the trace store. It holds no state of
its own, computes no aggregate the metrics catalogue does not already declare, and starting
or stopping it cannot affect a query.

HTML is rendered from strings rather than a template engine. The approved stack names no
templating dependency, the page is small, and every value that reaches it is escaped.
"""

from __future__ import annotations

from collections.abc import Iterable
from html import escape
from typing import Any, Final

from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse

from fasterrag.config.schema import Settings
from fasterrag.observability import metrics
from fasterrag.observability.logging import get_logger
from fasterrag.services.traces import TraceStore, create_trace_store

__all__ = ["DASHBOARD_TITLE", "create_dashboard", "render_page"]

DASHBOARD_TITLE: Final = "fasterRag — observability"

_RECENT_TRACES: Final = 50

_STYLE: Final = """
  body { font: 14px/1.5 system-ui, sans-serif; margin: 0; background: #0f1115; color: #e6e6e6; }
  header { padding: 16px 24px; border-bottom: 1px solid #23262d; }
  h1 { font-size: 16px; margin: 0; font-weight: 600; }
  main { padding: 24px; display: grid; gap: 24px; }
  section { border: 1px solid #23262d; border-radius: 8px; padding: 16px; }
  h2 { font-size: 13px; margin: 0 0 12px; text-transform: uppercase; color: #8b93a1; }
  table { border-collapse: collapse; width: 100%; }
  td, th { text-align: left; padding: 6px 8px; border-bottom: 1px solid #1b1e24; }
  th { color: #8b93a1; font-weight: 500; }
  code { color: #9ecbff; }
  .empty { color: #8b93a1; font-style: italic; }
  .note { color: #8b93a1; font-size: 12px; margin-top: 12px; }
"""


def _rows(pairs: Iterable[tuple[str, str]]) -> str:
    """Render label/value pairs, escaping both sides."""
    body = "".join(
        f"<tr><td>{escape(label)}</td><td><code>{escape(value)}</code></td></tr>"
        for label, value in pairs
    )
    return body or '<tr><td colspan="2" class="empty">nothing recorded yet</td></tr>'


def render_page(traces: list[dict[str, Any]]) -> str:
    """Return the dashboard HTML.

    Args:
        traces: Recent trace summaries, newest first.

    Returns:
        A complete document. Every interpolated value is escaped; the trace list carries
        user-supplied query text and model output, which is exactly the content that must
        never be able to inject markup into a page an operator trusts.
    """
    trace_rows = (
        "".join(
            "<tr>"
            f"<td><code>{escape(str(item.get('trace_id', '')))}</code></td>"
            f"<td>{escape(str(item.get('query', ''))[:120])}</td>"
            f"<td>{escape(str(item.get('collection') or '—'))}</td>"
            f"<td>{escape(str(item.get('tenant') or '—'))}</td>"
            "</tr>"
            for item in traces
        )
        or '<tr><td colspan="4" class="empty">no traces stored yet</td></tr>'
    )

    panels = [
        ("Requests", metrics.REGISTRY.series("fasterrag_requests_total")),
        ("Cache events", metrics.REGISTRY.series("fasterrag_cache_events_total")),
        ("Tokens", metrics.REGISTRY.series("fasterrag_tokens_total")),
        ("Estimated cost (USD)", metrics.REGISTRY.series("fasterrag_cost_usd_total")),
        ("Unpriced tokens", metrics.REGISTRY.series("fasterrag_unpriced_tokens_total")),
        ("Queue depth", metrics.REGISTRY.series("fasterrag_queue_depth")),
        ("Dead-letter depth", metrics.REGISTRY.series("fasterrag_dlq_depth")),
        ("Degraded responses", metrics.REGISTRY.series("fasterrag_degraded_responses_total")),
        ("Retrieval quality", metrics.REGISTRY.series("fasterrag_retrieval_quality")),
    ]

    sections = "".join(
        f"<section><h2>{escape(title)}</h2><table>{_rows(pairs)}</table></section>"
        for title, pairs in panels
    )

    return (
        f'<!doctype html><html lang="en"><head><meta charset="utf-8">'
        f"<title>{escape(DASHBOARD_TITLE)}</title><style>{_STYLE}</style></head><body>"
        f"<header><h1>{escape(DASHBOARD_TITLE)}</h1></header><main>"
        f"{sections}"
        f"<section><h2>Recent queries</h2><table>"
        f"<tr><th>trace</th><th>query</th><th>collection</th><th>tenant</th></tr>"
        f"{trace_rows}</table>"
        f'<p class="note">Read-only. This page cannot change anything — '
        f"it has no write endpoint to call.</p></section>"
        f"</main></body></html>"
    )


def create_dashboard(settings: Settings, store: TraceStore | None = None) -> FastAPI:
    """Build the dashboard application.

    Args:
        settings: Validated configuration.
        store: Trace store to read; built from configuration when omitted.

    Returns:
        An ASGI application serving the dashboard. It declares only ``GET`` routes.
    """
    traces = store if store is not None else create_trace_store(settings)
    app = FastAPI(title=DASHBOARD_TITLE, docs_url=None, redoc_url=None, openapi_url=None)
    logger = get_logger(__name__)

    def _recent() -> list[dict[str, Any]]:
        summaries: list[dict[str, Any]] = []
        for trace_id in traces.recent(_RECENT_TRACES):
            trace = traces.load(trace_id)
            if trace is None:
                continue
            summaries.append(
                {
                    "trace_id": trace.trace_id,
                    "query": trace.query,
                    "collection": trace.collection,
                    "tenant": trace.tenant,
                }
            )
        return summaries

    @app.get("/", response_class=HTMLResponse)
    async def index() -> HTMLResponse:
        """Serve the dashboard page."""
        return HTMLResponse(render_page(_recent()))

    @app.get("/api/traces")
    async def recent_traces() -> JSONResponse:
        """Return the same trace summaries as JSON, for scripted inspection."""
        return JSONResponse({"traces": _recent()})

    @app.get("/api/metrics")
    async def metrics_text() -> JSONResponse:
        """Return the metric names the registry declares."""
        return JSONResponse({"metrics": metrics.REGISTRY.names})

    logger.info(
        "observability dashboard built",
        extra={"port": settings.observability.dashboard_port},
    )
    return app
