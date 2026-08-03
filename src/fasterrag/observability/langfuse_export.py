"""Exporting query traces to a provisioned Langfuse instance.

``observability.langfuse: true`` stands the stack up (``services/langfuse.py``); this is what
puts anything in it. Without it the toggle produces a correctly-configured, permanently empty
dashboard.

Traces are still written to the local store regardless. Langfuse is an *additional* consumer,
not a replacement: replay and ``GET /v1/traces/{id}`` read the local copy, and making them
depend on a running container would mean an incident investigation needing the observability
stack to be healthy — exactly when it is least likely to be.

**Export never fails a query.** The answer has already been returned by the time this runs, so
every failure here is logged and dropped. A dashboard that is down must not become an outage.

The mapping onto Langfuse's ingestion API:

* our :class:`~fasterrag.core.tracing.Trace` becomes a ``trace-create`` event carrying the
  query as input and the answer as output;
* each :class:`~fasterrag.core.tracing.Span` becomes a ``span-create`` observation nested
  under it, with our millisecond offsets converted to the absolute timestamps Langfuse wants;
* the generation stage additionally carries model and token usage, which is what makes
  Langfuse's cost view work at all.
"""

from __future__ import annotations

import base64
from datetime import UTC, datetime, timedelta
from typing import Any, Final

import httpx

from fasterrag.core.tracing import Trace
from fasterrag.observability.logging import get_logger

__all__ = ["LangfuseExporter", "build_batch"]

INGESTION_PATH: Final = "/api/public/ingestion"
_TIMEOUT_SECONDS: Final = 10.0
_GENERATION_SPAN: Final = "generation"

_logger = get_logger(__name__)


def _isoformat(base: datetime, offset_ms: float) -> str:
    """Return an absolute timestamp for a span offset.

    Our spans record milliseconds relative to the query's start; Langfuse orders observations
    by wall-clock time. Sending the raw offsets would place every trace at the epoch and stack
    them on top of one another in the timeline.
    """
    return (base + timedelta(milliseconds=offset_ms)).isoformat()


def _parse_created_at(value: str) -> datetime:
    """Return the trace's start time, falling back to now when it is unusable.

    A malformed timestamp must not cost the whole export: an observation in the wrong place on
    a timeline is recoverable, a dropped trace is not.
    """
    try:
        return datetime.fromisoformat(value)
    except (TypeError, ValueError):
        return datetime.now(tz=UTC)


def build_batch(trace: Trace, *, release: str | None = None) -> list[dict[str, Any]]:
    """Return the ingestion events for one trace.

    Pure and side-effect free so the wire format can be asserted in a unit test without a
    running Langfuse — the mapping is where this integration is most likely to be wrong, and
    it is invisible until someone opens the dashboard.
    """
    started = _parse_created_at(trace.created_at)
    usage = trace.result.get("usage") or {}

    events: list[dict[str, Any]] = [
        {
            "id": f"{trace.trace_id}-trace",
            "type": "trace-create",
            "timestamp": started.isoformat(),
            "body": {
                "id": trace.trace_id,
                "name": "query",
                "input": trace.query,
                "output": trace.result.get("answer"),
                "release": release,
                "metadata": {
                    "collection": trace.collection,
                    "filters": trace.filters,
                    "mode": trace.result.get("mode"),
                    "degraded": trace.result.get("degraded"),
                    "faithfulness": trace.result.get("faithfulness"),
                    "retrieved": len(trace.retrieved),
                    "config": trace.config_snapshot,
                },
            },
        }
    ]

    for index, span in enumerate(trace.spans):
        body: dict[str, Any] = {
            "id": f"{trace.trace_id}-{index}",
            "traceId": trace.trace_id,
            "name": span.name,
            "startTime": _isoformat(started, span.start_ms),
            "endTime": _isoformat(started, span.end_ms),
            "metadata": span.attributes,
        }

        # The generation stage is reported as a `generation` observation rather than a plain
        # span: Langfuse derives model usage and cost only from that type, so sending it as a
        # span would leave its own cost view empty while the data sat right there.
        if span.name == _GENERATION_SPAN:
            body["input"] = trace.prompt
            body["output"] = trace.response
            configured = trace.config_snapshot.get("llm")
            body["model"] = configured.get("model") if isinstance(configured, dict) else None
            body["usage"] = {
                "input": usage.get("prompt_tokens"),
                "output": usage.get("completion_tokens"),
                "unit": "TOKENS",
            }
            events.append(
                {
                    "id": f"{trace.trace_id}-{index}-event",
                    "type": "generation-create",
                    "timestamp": started.isoformat(),
                    "body": body,
                }
            )
            continue

        events.append(
            {
                "id": f"{trace.trace_id}-{index}-event",
                "type": "span-create",
                "timestamp": started.isoformat(),
                "body": body,
            }
        )

    return events


class LangfuseExporter:
    """Ships traces to Langfuse over its ingestion API."""

    def __init__(
        self,
        host: str,
        public_key: str,
        secret_key: str,
        *,
        release: str | None = None,
        timeout: float = _TIMEOUT_SECONDS,
    ) -> None:
        """Build an exporter. No connection is opened until the first export."""
        self.host = host.rstrip("/")
        self.release = release
        self._timeout = timeout
        # CRITICAL: the credentials are held for the Authorization header and never logged.
        # Langfuse authenticates ingestion with HTTP Basic over the project's key pair.
        token = base64.b64encode(f"{public_key}:{secret_key}".encode()).decode("ascii")
        self._headers = {
            "Authorization": f"Basic {token}",
            "content-type": "application/json",
        }
        self._client: httpx.AsyncClient | None = None

    async def export(self, trace: Trace) -> bool:
        """Send one trace, reporting whether it landed.

        Returns:
            ``True`` when Langfuse accepted the batch. A ``False`` is logged with the reason
            and otherwise ignored — the query it describes was answered long before this ran,
            and a failed export must never surface to the caller.
        """
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=self._timeout, headers=self._headers)

        try:
            response = await self._client.post(
                f"{self.host}{INGESTION_PATH}",
                json={"batch": build_batch(trace, release=self.release)},
            )
        except httpx.HTTPError as exc:
            _logger.warning(
                "could not export a trace to langfuse; the query was unaffected",
                extra={"trace_id": trace.trace_id, "error": type(exc).__name__},
            )
            return False

        # 207 is Langfuse's normal success: the batch endpoint reports per-event results, so a
        # partial accept is the expected shape rather than an error.
        if response.status_code not in (200, 201, 207):
            _logger.warning(
                "langfuse rejected a trace export",
                extra={"trace_id": trace.trace_id, "status": response.status_code},
            )
            return False

        return True

    async def close(self) -> None:
        """Release the HTTP client."""
        if self._client is not None:
            await self._client.aclose()
            self._client = None
