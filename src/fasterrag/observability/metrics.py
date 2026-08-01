"""The metrics catalogue of ``docs/observability.md`` §2.

Implemented directly rather than on a client library. The catalogue is a fixed, known set of
instruments, the Prometheus text exposition format is a stable and simple contract, and the
approved stack names OpenTelemetry rather than a Prometheus client — so a dependency here
would buy dynamic registration nobody needs and a supply-chain entry nobody asked for.

Instruments are **always recording**, whether or not anything is scraping them. A counter
that only exists when an exporter is configured is a counter that is missing exactly when an
incident makes someone go looking for it.

Every instrument in the catalogue is declared here, at import, rather than created on first
use. A metric that springs into existence on its first observation is absent from a scrape
until the event happens, which reads identically to "the event happens zero times" — and the
difference between "no errors yet" and "the error counter was never wired up" is the whole
point of having the counter.
"""

from __future__ import annotations

import threading
from collections.abc import Mapping, Sequence
from typing import Final

__all__ = [
    "DURATION_BUCKETS",
    "REGISTRY",
    "Counter",
    "Gauge",
    "Histogram",
    "Registry",
    "render",
]

# CRITICAL: these bounds span a query's plausible range, from a cache hit in single-digit
# milliseconds to a slow generation in tens of seconds. p95 is read off these buckets, so a
# range that clipped either end would report a percentile that no request actually had.
DURATION_BUCKETS: Final[tuple[float, ...]] = (
    0.005,
    0.01,
    0.025,
    0.05,
    0.1,
    0.25,
    0.5,
    1.0,
    2.5,
    5.0,
    10.0,
    30.0,
    60.0,
)

_SCORE_BUCKETS: Final[tuple[float, ...]] = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0)

Labels = tuple[tuple[str, str], ...]


def _key(labels: Mapping[str, str] | None) -> Labels:
    """Return a stable, hashable key for a label set.

    Sorted so that the same labels given in a different order are the same series rather
    than two series that silently split a metric in half.
    """
    return tuple(sorted((labels or {}).items()))


def _escape(value: str) -> str:
    """Escape a label value for the text exposition format."""
    return value.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")


def _render_labels(labels: Labels, extra: tuple[tuple[str, str], ...] = ()) -> str:
    """Render a label set as the ``{k="v"}`` suffix, or nothing when there are none."""
    pairs = [*labels, *extra]
    if not pairs:
        return ""
    inner = ",".join(f'{name}="{_escape(value)}"' for name, value in pairs)
    return f"{{{inner}}}"


class Instrument:
    """Shared naming, documentation, and locking for every metric type."""

    kind = "untyped"

    def __init__(self, name: str, documentation: str, labels: Sequence[str] = ()) -> None:
        """Declare an instrument and the label names its series carry."""
        self.name = name
        self.documentation = documentation
        self.label_names = tuple(labels)
        self._lock = threading.Lock()

    def _header(self) -> list[str]:
        """Return the HELP and TYPE lines every metric is preceded by."""
        return [
            f"# HELP {self.name} {self.documentation}",
            f"# TYPE {self.name} {self.kind}",
        ]

    def render(self) -> list[str]:
        """Return this instrument's exposition lines."""
        raise NotImplementedError


class Counter(Instrument):
    """A value that only ever increases."""

    kind = "counter"

    def __init__(self, name: str, documentation: str, labels: Sequence[str] = ()) -> None:
        """Declare a counter."""
        super().__init__(name, documentation, labels)
        self._values: dict[Labels, float] = {}

    def increment(self, amount: float = 1.0, **labels: str) -> None:
        """Add to the series identified by ``labels``.

        Raises:
            ValueError: If the amount is negative. A counter that can go down is not a
                counter, and every rate built on it would be wrong rather than merely odd.
        """
        if amount < 0:
            raise ValueError(f"{self.name} is a counter and cannot decrease by {amount}")

        key = _key(labels)
        with self._lock:
            self._values[key] = self._values.get(key, 0.0) + amount

    def value(self, **labels: str) -> float:
        """Return one series' current total."""
        with self._lock:
            return self._values.get(_key(labels), 0.0)

    def render(self) -> list[str]:
        """Return one line per series."""
        with self._lock:
            series = sorted(self._values.items())
        return [
            *self._header(),
            *(f"{self.name}{_render_labels(key)} {value}" for key, value in series),
        ]


class Gauge(Instrument):
    """A value that goes up and down."""

    kind = "gauge"

    def __init__(self, name: str, documentation: str, labels: Sequence[str] = ()) -> None:
        """Declare a gauge."""
        super().__init__(name, documentation, labels)
        self._values: dict[Labels, float] = {}

    def set(self, value: float, **labels: str) -> None:
        """Record the current value of a series."""
        with self._lock:
            self._values[_key(labels)] = value

    def value(self, **labels: str) -> float:
        """Return one series' current value."""
        with self._lock:
            return self._values.get(_key(labels), 0.0)

    def render(self) -> list[str]:
        """Return one line per series."""
        with self._lock:
            series = sorted(self._values.items())
        return [
            *self._header(),
            *(f"{self.name}{_render_labels(key)} {value}" for key, value in series),
        ]


class Histogram(Instrument):
    """A distribution, exposed as cumulative buckets plus a sum and a count."""

    kind = "histogram"

    def __init__(
        self,
        name: str,
        documentation: str,
        labels: Sequence[str] = (),
        buckets: Sequence[float] = DURATION_BUCKETS,
    ) -> None:
        """Declare a histogram over ``buckets``."""
        super().__init__(name, documentation, labels)
        self.buckets = tuple(sorted(buckets))
        self._counts: dict[Labels, list[int]] = {}
        self._sums: dict[Labels, float] = {}

    def observe(self, value: float, **labels: str) -> None:
        """Record one observation into the series identified by ``labels``.

        # CRITICAL: only the *first* bucket the value fits is incremented. Counts are stored
        # per-bucket and made cumulative once, at render. Incrementing every bucket that
        # bounds the value would double-count it against the cumulative pass and inflate
        # every percentile read off the result.
        """
        key = _key(labels)
        with self._lock:
            counts = self._counts.setdefault(key, [0] * (len(self.buckets) + 1))
            self._sums[key] = self._sums.get(key, 0.0) + value
            for index, bound in enumerate(self.buckets):
                if value <= bound:
                    counts[index] += 1
                    break
            counts[-1] += 1

    def count(self, **labels: str) -> int:
        """Return how many observations a series has taken."""
        with self._lock:
            return self._counts.get(_key(labels), [0])[-1]

    def total(self, **labels: str) -> float:
        """Return the sum of a series' observations."""
        with self._lock:
            return self._sums.get(_key(labels), 0.0)

    def render(self) -> list[str]:
        """Return cumulative buckets, then the sum and count, per series."""
        with self._lock:
            series = sorted(self._counts.items())
            sums = dict(self._sums)

        lines = self._header()
        for key, counts in series:
            running = 0
            for index, bound in enumerate(self.buckets):
                running += counts[index]
                lines.append(
                    f"{self.name}_bucket{_render_labels(key, (('le', repr(bound)),))} {running}"
                )
            lines.append(f"{self.name}_bucket{_render_labels(key, (('le', '+Inf'),))} {counts[-1]}")
            lines.append(f"{self.name}_sum{_render_labels(key)} {sums.get(key, 0.0)}")
            lines.append(f"{self.name}_count{_render_labels(key)} {counts[-1]}")
        return lines


class Registry:
    """Every declared instrument, in catalogue order."""

    def __init__(self) -> None:
        """Build an empty registry."""
        self._instruments: list[Instrument] = []

    def register(self, instrument: Instrument) -> Instrument:
        """Add an instrument and return it, so declaration and registration are one step."""
        self._instruments.append(instrument)
        return instrument

    def counter(self, name: str, documentation: str, labels: Sequence[str] = ()) -> Counter:
        """Declare and register a counter."""
        instrument = Counter(name, documentation, labels)
        self._instruments.append(instrument)
        return instrument

    def gauge(self, name: str, documentation: str, labels: Sequence[str] = ()) -> Gauge:
        """Declare and register a gauge."""
        instrument = Gauge(name, documentation, labels)
        self._instruments.append(instrument)
        return instrument

    def histogram(
        self,
        name: str,
        documentation: str,
        labels: Sequence[str] = (),
        buckets: Sequence[float] = DURATION_BUCKETS,
    ) -> Histogram:
        """Declare and register a histogram."""
        instrument = Histogram(name, documentation, labels, buckets)
        self._instruments.append(instrument)
        return instrument

    def render(self) -> str:
        """Return the whole registry in the Prometheus text exposition format."""
        lines: list[str] = []
        for instrument in self._instruments:
            lines.extend(instrument.render())
        return "\n".join(lines) + "\n"

    @property
    def names(self) -> list[str]:
        """Return every registered metric name."""
        return [instrument.name for instrument in self._instruments]


REGISTRY = Registry()

REQUESTS = REGISTRY.counter(
    "fasterrag_requests_total",
    "Request volume (RED: rate).",
    ("endpoint", "method", "status", "tenant"),
)
ERRORS = REGISTRY.counter(
    "fasterrag_errors_total",
    "Error rate by problem code (RED: errors).",
    ("endpoint", "code", "tenant"),
)
REQUEST_DURATION = REGISTRY.histogram(
    "fasterrag_request_duration_seconds",
    "End-to-end latency; p50/p95 derived (RED: duration).",
    ("endpoint",),
)
STAGE_DURATION = REGISTRY.histogram(
    "fasterrag_stage_duration_seconds",
    "Per-stage latency - the retrieval-versus-generation split.",
    ("stage",),
)
TTFT = REGISTRY.histogram(
    "fasterrag_ttft_seconds",
    "Time to first streamed token.",
)
TOKENS = REGISTRY.counter(
    "fasterrag_tokens_total",
    "Token counts.",
    ("kind", "provider", "tenant"),
)
COST = REGISTRY.counter(
    "fasterrag_cost_usd_total",
    "Estimated cost per query, accumulated.",
    ("provider", "tenant"),
)
CACHE_EVENTS = REGISTRY.counter(
    "fasterrag_cache_events_total",
    "Cache hit/miss ratio source.",
    ("cache", "result"),
)
RETRIEVAL_QUALITY = REGISTRY.gauge(
    "fasterrag_retrieval_quality",
    "Latest eval-harness scores.",
    ("metric",),
)
FAITHFULNESS = REGISTRY.histogram(
    "fasterrag_faithfulness",
    "Grounding/faithfulness score distribution (D5).",
    buckets=_SCORE_BUCKETS,
)
INGEST_DOCUMENTS = REGISTRY.counter(
    "fasterrag_ingest_documents_total",
    "Ingestion outcomes.",
    ("status",),
)
INGEST_THROUGHPUT = REGISTRY.gauge(
    "fasterrag_ingest_throughput",
    "Live ingestion throughput.",
    ("unit",),
)
QUEUE_DEPTH = REGISTRY.gauge(
    "fasterrag_queue_depth",
    "Bounded-queue occupancy.",
    ("queue",),
)
DLQ_DEPTH = REGISTRY.gauge(
    "fasterrag_dlq_depth",
    "Dead-letter queue depth.",
    ("collection",),
)
CIRCUIT_STATE = REGISTRY.gauge(
    "fasterrag_circuit_state",
    "0 closed / 1 half-open / 2 open.",
    ("provider",),
)
DEGRADED_RESPONSES = REGISTRY.counter(
    "fasterrag_degraded_responses_total",
    "Degradation-ladder activations (D4).",
    ("mode",),
)


def render() -> str:
    """Return the default registry in the Prometheus text exposition format."""
    return REGISTRY.render()
