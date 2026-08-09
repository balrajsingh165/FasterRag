"""OTLP export of query traces (``observability.otel``).

Ships the same four RAG spans the trace store already records — ``retrieval``, ``reranker``,
``context-assembly``, ``generation`` — to any OTLP endpoint, so a fasterRag query appears in
Jaeger, Tempo, Honeycomb, or a vendor backend alongside the rest of a system's traces.

**The trace id is preserved.** fasterRag mints a 32-hex trace id, which is exactly the
OpenTelemetry trace-id shape, so the id in a ``problem+json`` error body, in the logs, and in
``GET /v1/traces/{id}`` is the same id pasted into a trace viewer's search box. Minting a
separate id here would mean an operator holding a failing request's id has no way to find it.

Spans are built directly rather than through a tracer. The stages have already finished with
known offsets by the time this runs, and a tracer exists to time work as it happens — using
one would mean re-timing spans that were already timed, which is how a timeline ends up
disagreeing with the trace it came from.
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from hashlib import blake2b
from typing import Any, Final

from fasterrag.core.tracing import Trace
from fasterrag.errors import ConfigError
from fasterrag.observability.logging import get_logger

__all__ = [
    "SCOPE_NAME",
    "SERVICE_NAME",
    "OtelExporter",
    "build_spans",
    "signal_endpoint",
    "span_id_for",
]

SERVICE_NAME: Final = "fasterrag"
SCOPE_NAME: Final = "fasterrag"

_NANOSECONDS_PER_MILLISECOND: Final = 1_000_000
_SPAN_ID_BYTES: Final = 8
_ROOT_SPAN: Final = "query"
_TIMEOUT_SECONDS: Final = 10
_SIGNAL_PATHS: Final = ("/v1/traces", "/v1/metrics", "/v1/logs")

_logger = get_logger(__name__)


def signal_endpoint(endpoint: str, signal: str) -> str:
    """Return the OTLP/HTTP URL for one signal, derived from the configured endpoint.

    # CRITICAL: ``observability.otel_endpoint`` is one setting feeding two exporters, and
    # OTLP/HTTP puts each signal on its own path. Passing the configured value through
    # verbatim — which is what this did until it met a real collector — means no value works
    # for both: a bare ``http://host:4318`` 404s for traces *and* metrics, ``.../v1/traces``
    # 400s every metric push, and ``.../v1/metrics`` 400s every trace. Every one of those is
    # a logged warning and nothing else, so the toggle looks on and nothing arrives. The
    # endpoint is therefore treated as the collector's base URL, exactly as the OTel
    # specification treats ``OTEL_EXPORTER_OTLP_ENDPOINT``.

    A path already naming a signal is replaced rather than appended to, so the
    ``http://collector:4318/v1/traces`` form that earlier docs and configurations used keeps
    working for traces and starts working for metrics.

    Args:
        endpoint: The configured collector endpoint, with or without a signal path.
        signal: ``"traces"`` or ``"metrics"``.

    Returns:
        The absolute URL to POST that signal to.
    """
    base = endpoint.rstrip("/")
    for path in _SIGNAL_PATHS:
        if base.endswith(path):
            base = base[: -len(path)]
            break
    return f"{base}/v1/{signal}"


def span_id_for(trace_id: str, name: str) -> int:
    """Return the 64-bit span id for a stage of a trace.

    Derived from the trace id and the stage name rather than drawn at random, so exporting
    the same trace twice — a retry, or a replay — produces the same span ids and the backend
    treats it as one trace rather than two overlapping copies.

    Zero is not a legal span id, so a digest that lands there is nudged to one; the odds are
    negligible and a silently invalid span is not.
    """
    digest = blake2b(f"{trace_id}:{name}".encode(), digest_size=_SPAN_ID_BYTES).digest()
    return int.from_bytes(digest, "big") or 1


def _epoch_nanoseconds(moment: datetime) -> int:
    """Return a datetime as nanoseconds since the epoch, which is OTLP's time unit."""
    return int(moment.timestamp() * 1_000_000_000)


def _started_at(trace: Trace) -> datetime:
    """Return the trace's start time, falling back to now when it is unusable.

    A malformed timestamp must not cost the whole export: a trace in the wrong place on a
    timeline is recoverable, a dropped one is not.
    """
    try:
        parsed = datetime.fromisoformat(trace.created_at)
    except (TypeError, ValueError):
        return datetime.now(tz=UTC)
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=UTC)


def _attribute(value: Any) -> Any:
    """Return a value in a form OTLP accepts, stringifying anything else.

    OTLP attributes are strings, numbers, booleans, or homogeneous sequences of those. A
    dict or a None reaching the exporter is dropped by the SDK with a warning, so the
    attribute simply vanishes — stringifying keeps the information visible.
    """
    if isinstance(value, bool | int | float | str):
        return value
    return str(value)


def _attributes(trace: Trace) -> dict[str, Any]:
    """Return the root span's attributes, naming what the query did and produced."""
    result = trace.result or {}
    usage = result.get("usage") or {}

    candidate = {
        "fasterrag.trace_id": trace.trace_id,
        "fasterrag.query": trace.query,
        "fasterrag.collection": trace.collection,
        "fasterrag.tenant": trace.tenant,
        "fasterrag.mode": result.get("mode"),
        "fasterrag.degraded": result.get("degraded"),
        "fasterrag.chunks_retrieved": len(trace.retrieved),
        "fasterrag.citations": len(result.get("citations") or []),
        "fasterrag.faithfulness": result.get("faithfulness"),
        "fasterrag.prompt_tokens": usage.get("prompt_tokens"),
        "fasterrag.completion_tokens": usage.get("completion_tokens"),
    }
    return {key: _attribute(value) for key, value in candidate.items() if value is not None}


def build_spans(trace: Trace) -> list[Any]:
    """Return the OTLP spans for one trace: a root span with a child per stage.

    Pure and side-effect free so the mapping can be asserted without a collector running.
    This mapping is where the integration is most likely to be wrong and, like the Langfuse
    one, it is invisible until someone opens a dashboard.

    Raises:
        ConfigError: If the OpenTelemetry SDK is not installed.
    """
    sdk = _sdk()
    started = _started_at(trace)
    base = _epoch_nanoseconds(started)

    resource = sdk.Resource.create({"service.name": SERVICE_NAME})
    # Named for the same reason the metric payload names its scope: a backend groups and
    # filters by it, and spans arriving with an empty scope are attributed to nothing.
    scope = sdk.InstrumentationScope(SCOPE_NAME)
    trace_id = int(trace.trace_id, 16)
    root_id = span_id_for(trace.trace_id, _ROOT_SPAN)

    # The root has to span every stage, and a query's own wall clock is not recorded
    # separately — it is the envelope of what it did.
    last = max((span.end_ms for span in trace.spans), default=0.0)

    root_context = sdk.SpanContext(
        trace_id=trace_id,
        span_id=root_id,
        is_remote=False,
        trace_flags=sdk.TraceFlags(sdk.TraceFlags.SAMPLED),
    )

    spans = [
        sdk.ReadableSpan(
            name=_ROOT_SPAN,
            context=root_context,
            parent=None,
            resource=resource,
            attributes=_attributes(trace),
            kind=sdk.SpanKind.SERVER,
            instrumentation_scope=scope,
            start_time=base,
            end_time=base + int(last * _NANOSECONDS_PER_MILLISECOND),
        )
    ]

    for stage in trace.spans:
        spans.append(
            sdk.ReadableSpan(
                name=stage.name,
                context=sdk.SpanContext(
                    trace_id=trace_id,
                    span_id=span_id_for(trace.trace_id, stage.name),
                    is_remote=False,
                    trace_flags=sdk.TraceFlags(sdk.TraceFlags.SAMPLED),
                ),
                parent=root_context,
                resource=resource,
                attributes={
                    key: _attribute(value) for key, value in (stage.attributes or {}).items()
                },
                kind=sdk.SpanKind.INTERNAL,
                instrumentation_scope=scope,
                start_time=base + int(stage.start_ms * _NANOSECONDS_PER_MILLISECOND),
                end_time=base + int(stage.end_ms * _NANOSECONDS_PER_MILLISECOND),
            )
        )

    return spans


class _Sdk:
    """The SDK symbols this module uses, resolved once."""

    def __init__(self) -> None:
        """Import the OpenTelemetry SDK.

        Raises:
            ConfigError: If it is not installed.
        """
        try:
            from opentelemetry.sdk.resources import Resource
            from opentelemetry.sdk.trace import ReadableSpan
            from opentelemetry.sdk.util.instrumentation import InstrumentationScope
            from opentelemetry.trace import SpanContext, SpanKind, TraceFlags
        except ImportError as exc:
            raise ConfigError(
                "observability.otel is true, which needs the OpenTelemetry SDK; "
                "install it with 'pip install fasterrag[otel]'"
            ) from exc

        self.InstrumentationScope = InstrumentationScope
        self.Resource = Resource
        self.ReadableSpan = ReadableSpan
        self.SpanContext = SpanContext
        self.SpanKind = SpanKind
        self.TraceFlags = TraceFlags


_cached: _Sdk | None = None


def _sdk() -> _Sdk:
    """Return the resolved SDK symbols, importing them on first use.

    Imported lazily so that installing fasterRag without the ``otel`` extra still starts,
    and cached because a trace is exported per query.
    """
    global _cached
    if _cached is None:
        _cached = _Sdk()
    return _cached


class OtelExporter:
    """Ships query traces to an OTLP endpoint."""

    def __init__(self, endpoint: str, *, timeout: int = _TIMEOUT_SECONDS) -> None:
        """Build the exporter without connecting.

        Args:
            endpoint: The collector's OTLP/HTTP endpoint from ``observability.otel_endpoint``.
                Taken as a base URL; the ``/v1/traces`` path is derived from it.
            timeout: Seconds to wait for the collector.

        Raises:
            ConfigError: If the OTLP exporter is not installed.
        """
        try:
            from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
        except ImportError as exc:
            raise ConfigError(
                "observability.otel is true, which needs the OpenTelemetry OTLP exporter; "
                "install it with 'pip install fasterrag[otel]'"
            ) from exc

        self.endpoint = signal_endpoint(endpoint, "traces")
        self._exporter = OTLPSpanExporter(endpoint=self.endpoint, timeout=timeout)

    async def export(self, trace: Trace) -> bool:
        """Send one trace, reporting whether the collector accepted it.

        Never raises. A trace is a record of a query that has already been answered, so an
        unreachable collector must cost the record and nothing else.
        """
        try:
            spans = build_spans(trace)
        except (ConfigError, ValueError) as exc:
            _logger.warning(
                "could not build OTLP spans for a trace",
                extra={"trace_id": trace.trace_id, "error": str(exc)},
            )
            return False

        try:
            result = await asyncio.to_thread(self._exporter.export, spans)
        except Exception as exc:
            # Deliberately broad: this reaches a network through protobuf serialization and
            # a vendor SDK, each raising its own types, and none of them should be able to
            # affect a request that has already been answered.
            _logger.warning(
                "could not export a trace over OTLP",
                extra={
                    "trace_id": trace.trace_id,
                    "endpoint": self.endpoint,
                    "error": type(exc).__name__,
                },
            )
            return False

        # Compared by name rather than by importing the enum, which keeps the SDK import
        # lazy on a path that runs after every query.
        return str(getattr(result, "name", "")) == "SUCCESS"

    async def close(self) -> None:
        """Shut the exporter down, flushing whatever it holds."""
        await asyncio.to_thread(self._exporter.shutdown)
