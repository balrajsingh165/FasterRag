"""OTLP export of the metric catalogue (``observability.otel``).

Ships the same instruments ``/metrics`` exposes, so a deployment whose observability stack
speaks OTLP rather than Prometheus scrape sees fasterRag's counters as well as its traces.

**One source of truth.** Both views read :data:`~fasterrag.observability.metrics.REGISTRY`
through its structured snapshot rather than one parsing the other's output. Two renderers
over one registry cannot disagree; a renderer that parses another renderer's text is a lossy
copy that eventually reports a number neither of them got wrong.

Metrics are *pushed on an interval*, unlike traces which are pushed per query. A counter has
no natural moment to be sent — its value is meaningful at any instant — so something has to
choose one, and OTLP's model is a periodic reader.
"""

from __future__ import annotations

import asyncio
import time
from contextlib import suppress
from typing import Any, Final

from fasterrag.errors import ConfigError
from fasterrag.observability.logging import get_logger
from fasterrag.observability.metrics import REGISTRY, Registry, Sample
from fasterrag.observability.otel_export import SCOPE_NAME, SERVICE_NAME, signal_endpoint

__all__ = ["DEFAULT_INTERVAL_SECONDS", "MetricPusher", "build_metrics"]

DEFAULT_INTERVAL_SECONDS: Final = 60.0

_TIMEOUT_SECONDS: Final = 10

_logger = get_logger(__name__)


def _now_ns() -> int:
    """Return the current time in epoch nanoseconds, which is OTLP's time unit."""
    return int(time.time() * 1_000_000_000)


def _per_bucket(cumulative: tuple[int, ...]) -> list[int]:
    """Return per-bucket counts from the cumulative form.

    # CRITICAL: OTLP explicit-bucket histograms carry *per-bucket* counts while the
    # Prometheus exposition format carries cumulative ones. Sending the cumulative counts
    # unchanged would make every bucket include everything below it a second time, so a
    # percentile read off the OTLP copy would disagree with the same percentile read off
    # the scrape — and the OTLP one would be wrong.
    """
    counts: list[int] = []
    previous = 0
    for total in cumulative:
        counts.append(max(total - previous, 0))
        previous = total
    return counts


def _number_point(sdk: Any, sample: Sample, start_ns: int, now_ns: int) -> Any:
    """Return an OTLP number data point for a counter or gauge sample."""
    return sdk.NumberDataPoint(
        attributes=dict(sample.labels),
        start_time_unix_nano=start_ns,
        time_unix_nano=now_ns,
        value=sample.value,
    )


def _histogram_point(sdk: Any, sample: Sample, start_ns: int, now_ns: int) -> Any:
    """Return an OTLP histogram data point for a histogram sample."""
    cumulative = sample.bucket_counts or ()
    return sdk.HistogramDataPoint(
        attributes=dict(sample.labels),
        start_time_unix_nano=start_ns,
        time_unix_nano=now_ns,
        count=sample.count or 0,
        sum=sample.value,
        bucket_counts=_per_bucket(cumulative),
        explicit_bounds=list(sample.bucket_bounds or ()),
        min=None,
        max=None,
    )


def build_metrics(registry: Registry, start_ns: int, now_ns: int) -> Any:
    """Return the OTLP ``MetricsData`` for a registry snapshot.

    Pure and side-effect free so the mapping can be asserted without a collector. Series
    of one instrument are grouped into a single metric, because OTLP models labels as data
    point attributes rather than as separate metrics.

    Args:
        registry: The catalogue to read.
        start_ns: When this process started recording, in epoch nanoseconds. Cumulative
            sums are meaningless without it — a backend needs the window a total covers.
        now_ns: The moment of this snapshot, in epoch nanoseconds.

    Returns:
        The OTLP metrics payload.

    Raises:
        ConfigError: If the OpenTelemetry SDK is not installed.
    """
    sdk = _sdk()

    grouped: dict[str, list[Sample]] = {}
    for sample in registry.snapshot():
        grouped.setdefault(sample.name, []).append(sample)

    metrics: list[Any] = []
    for name, samples in grouped.items():
        first = samples[0]

        if first.kind == "histogram":
            data: Any = sdk.Histogram(
                data_points=[_histogram_point(sdk, s, start_ns, now_ns) for s in samples],
                aggregation_temporality=sdk.AggregationTemporality.CUMULATIVE,
            )
        elif first.kind == "counter":
            data = sdk.Sum(
                data_points=[_number_point(sdk, s, start_ns, now_ns) for s in samples],
                aggregation_temporality=sdk.AggregationTemporality.CUMULATIVE,
                is_monotonic=True,
            )
        else:
            data = sdk.Gauge(data_points=[_number_point(sdk, s, start_ns, now_ns) for s in samples])

        metrics.append(sdk.Metric(name=name, description=first.documentation, unit="", data=data))

    scope = sdk.ScopeMetrics(
        scope=sdk.InstrumentationScope(SCOPE_NAME),
        metrics=metrics,
        schema_url="",
    )
    return sdk.MetricsData(
        resource_metrics=[
            sdk.ResourceMetrics(
                resource=sdk.Resource.create({"service.name": SERVICE_NAME}),
                scope_metrics=[scope],
                schema_url="",
            )
        ]
    )


class _Sdk:
    """The SDK symbols this module uses, resolved once."""

    def __init__(self) -> None:
        """Import the OpenTelemetry metrics SDK.

        Raises:
            ConfigError: If it is not installed.
        """
        try:
            from opentelemetry.sdk.metrics.export import (
                AggregationTemporality,
                Gauge,
                Histogram,
                HistogramDataPoint,
                Metric,
                MetricsData,
                NumberDataPoint,
                ResourceMetrics,
                ScopeMetrics,
                Sum,
            )
            from opentelemetry.sdk.resources import Resource
            from opentelemetry.sdk.util.instrumentation import InstrumentationScope
        except ImportError as exc:
            raise ConfigError(
                "observability.otel is true, which needs the OpenTelemetry SDK; "
                "install it with 'pip install fasterrag[otel]'"
            ) from exc

        self.AggregationTemporality = AggregationTemporality
        self.Gauge = Gauge
        self.Histogram = Histogram
        self.HistogramDataPoint = HistogramDataPoint
        self.InstrumentationScope = InstrumentationScope
        self.Metric = Metric
        self.MetricsData = MetricsData
        self.NumberDataPoint = NumberDataPoint
        self.Resource = Resource
        self.ResourceMetrics = ResourceMetrics
        self.ScopeMetrics = ScopeMetrics
        self.Sum = Sum


_cached: _Sdk | None = None


def _sdk() -> _Sdk:
    """Return the resolved SDK symbols, importing them on first use."""
    global _cached
    if _cached is None:
        _cached = _Sdk()
    return _cached


class MetricPusher:
    """Pushes the metric catalogue to an OTLP endpoint on an interval."""

    def __init__(
        self,
        endpoint: str,
        *,
        registry: Registry | None = None,
        interval_seconds: float = DEFAULT_INTERVAL_SECONDS,
        timeout: int = _TIMEOUT_SECONDS,
    ) -> None:
        """Build the pusher without starting it or connecting.

        Args:
            endpoint: The collector's OTLP/HTTP endpoint from ``observability.otel_endpoint``.
                Taken as a base URL; the ``/v1/metrics`` path is derived from it.
            registry: Catalogue to read; defaults to the process-wide one.
            interval_seconds: How often to push.
            timeout: Seconds to wait for the collector.

        Raises:
            ConfigError: If the OTLP exporter is not installed.
        """
        try:
            from opentelemetry.exporter.otlp.proto.http.metric_exporter import OTLPMetricExporter
        except ImportError as exc:
            raise ConfigError(
                "observability.otel is true, which needs the OpenTelemetry OTLP exporter; "
                "install it with 'pip install fasterrag[otel]'"
            ) from exc

        self.endpoint = signal_endpoint(endpoint, "metrics")
        self.registry = registry or REGISTRY
        self.interval_seconds = interval_seconds
        self._exporter = OTLPMetricExporter(endpoint=self.endpoint, timeout=timeout)
        self._started_ns = _now_ns()
        self._task: asyncio.Task[None] | None = None

    async def push(self) -> bool:
        """Send one snapshot, reporting whether the collector accepted it.

        Never raises. Metrics describe a process that is otherwise working, so an
        unreachable collector must cost the snapshot and nothing else.
        """
        try:
            data = build_metrics(self.registry, self._started_ns, _now_ns())
        except (ConfigError, ValueError) as exc:
            _logger.warning("could not build OTLP metrics", extra={"error": str(exc)})
            return False

        try:
            result = await asyncio.to_thread(self._exporter.export, data)
        except Exception as exc:
            # Deliberately broad, for the same reason the trace exporter's is: this reaches
            # a network through protobuf serialization and a vendor SDK, and none of it
            # should be able to affect a process that is serving fine.
            _logger.warning(
                "could not export metrics over OTLP",
                extra={"endpoint": self.endpoint, "error": type(exc).__name__},
            )
            return False

        return str(getattr(result, "name", "")) == "SUCCESS"

    def start(self) -> None:
        """Begin pushing on the configured interval.

        # CRITICAL: the task is retained on the instance. asyncio holds only a weak
        # reference to a running task, so one that nothing keeps can be collected
        # mid-flight and the pushes stop with no error anywhere.
        """
        if self._task is not None:
            return
        self._task = asyncio.create_task(self._loop())

    async def _loop(self) -> None:
        """Push forever, sleeping between snapshots."""
        while True:
            await asyncio.sleep(self.interval_seconds)
            await self.push()

    async def close(self) -> None:
        """Stop pushing, sending one final snapshot so the last window is not lost."""
        if self._task is not None:
            self._task.cancel()
            with suppress(asyncio.CancelledError):
                await self._task
            self._task = None

        await self.push()
        await asyncio.to_thread(self._exporter.shutdown)
