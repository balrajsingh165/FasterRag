"""OTLP export of the metric catalogue.

The cumulative-versus-per-bucket difference between the Prometheus exposition format and
OTLP is the one mapping error that would produce plausible, wrong percentiles, so it is
asserted directly rather than only through the payload.
"""

import gzip
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any, ClassVar

import pytest

pytest.importorskip("opentelemetry.sdk", reason="OTLP export ships in the optional 'otel' extra")

from fasterrag.config.schema import Settings
from fasterrag.observability.metrics import Registry
from fasterrag.observability.otel_metrics import (
    MetricPusher,
    _per_bucket,
    build_metrics,
)
from fasterrag.services.traces import create_metric_pusher

START_NS = 1_700_000_000_000_000_000
NOW_NS = START_NS + 60_000_000_000


def populated() -> Registry:
    registry = Registry()
    requests = registry.counter("fasterrag_requests_total", "requests", ("status",))
    depth = registry.gauge("fasterrag_queue_depth", "queue depth", ("queue",))
    duration = registry.histogram(
        "fasterrag_stage_duration_seconds", "stage duration", ("stage",), buckets=(0.01, 0.1, 1.0)
    )

    requests.increment(3, status="200")
    requests.increment(1, status="500")
    depth.set(42, queue="chunks")
    for value in (0.005, 0.05, 0.5, 5.0):
        duration.observe(value, stage="retrieval")
    return registry


def metrics_of(data: Any) -> list[Any]:
    """Return the metrics inside an OTLP payload."""
    metrics: list[Any] = data.resource_metrics[0].scope_metrics[0].metrics
    return metrics


def test_cumulative_counts_become_per_bucket() -> None:
    """Sending the cumulative form would make every bucket re-count everything below it."""
    assert _per_bucket((1, 2, 2, 3, 3, 4)) == [1, 1, 0, 1, 0, 1]


def test_per_bucket_counts_sum_to_the_observation_count() -> None:
    """The invariant the conversion exists to preserve; the cumulative form breaks it."""
    cumulative = (1, 2, 2, 3, 3, 4)

    assert sum(_per_bucket(cumulative)) == cumulative[-1]


def test_a_non_monotonic_input_does_not_produce_a_negative_count() -> None:
    """A negative bucket count is rejected by collectors and impossible to interpret."""
    assert all(count >= 0 for count in _per_bucket((3, 1, 5)))


def test_every_instrument_with_data_is_exported() -> None:
    names = {metric.name for metric in metrics_of(build_metrics(populated(), START_NS, NOW_NS))}

    assert names == {
        "fasterrag_requests_total",
        "fasterrag_queue_depth",
        "fasterrag_stage_duration_seconds",
    }


def test_series_of_one_instrument_share_a_metric() -> None:
    """OTLP models labels as data point attributes, not as separate metrics."""
    requests = next(
        metric
        for metric in metrics_of(build_metrics(populated(), START_NS, NOW_NS))
        if metric.name == "fasterrag_requests_total"
    )

    assert len(requests.data.data_points) == 2
    assert {point.attributes["status"] for point in requests.data.data_points} == {"200", "500"}


def test_a_counter_is_monotonic() -> None:
    """A counter sent as a non-monotonic sum makes every rate built on it wrong."""
    requests = next(
        metric
        for metric in metrics_of(build_metrics(populated(), START_NS, NOW_NS))
        if metric.name == "fasterrag_requests_total"
    )

    assert requests.data.is_monotonic is True


def test_a_gauge_is_not_a_sum() -> None:
    depth = next(
        metric
        for metric in metrics_of(build_metrics(populated(), START_NS, NOW_NS))
        if metric.name == "fasterrag_queue_depth"
    )

    assert not hasattr(depth.data, "is_monotonic")
    assert depth.data.data_points[0].value == 42


def test_a_histogram_carries_its_bounds_and_counts() -> None:
    duration = next(
        metric
        for metric in metrics_of(build_metrics(populated(), START_NS, NOW_NS))
        if metric.name == "fasterrag_stage_duration_seconds"
    )
    point = duration.data.data_points[0]

    assert point.count == 4
    assert list(point.explicit_bounds) == [0.01, 0.1, 1.0]
    assert len(point.bucket_counts) == len(point.explicit_bounds) + 1
    assert sum(point.bucket_counts) == point.count


def test_the_histogram_sum_survives() -> None:
    duration = next(
        metric
        for metric in metrics_of(build_metrics(populated(), START_NS, NOW_NS))
        if metric.name == "fasterrag_stage_duration_seconds"
    )

    assert duration.data.data_points[0].sum == pytest.approx(5.555)


def test_the_window_a_total_covers_is_carried() -> None:
    """A cumulative sum with no start time is a number a backend cannot rate."""
    point = metrics_of(build_metrics(populated(), START_NS, NOW_NS))[0].data.data_points[0]

    assert point.start_time_unix_nano == START_NS
    assert point.time_unix_nano == NOW_NS


def test_the_documentation_becomes_the_description() -> None:
    requests = next(
        metric
        for metric in metrics_of(build_metrics(populated(), START_NS, NOW_NS))
        if metric.name == "fasterrag_requests_total"
    )

    assert requests.description == "requests"


def test_an_instrument_with_no_observations_is_omitted() -> None:
    """An empty metric would report zero series, not a zero value; neither is informative."""
    registry = Registry()
    registry.counter("fasterrag_never_written_total", "never written")

    assert metrics_of(build_metrics(registry, START_NS, NOW_NS)) == []


def test_the_service_is_named() -> None:
    data = build_metrics(populated(), START_NS, NOW_NS)
    resource = data.resource_metrics[0].resource

    assert dict(resource.attributes)["service.name"] == "fasterrag"


def test_it_encodes_as_real_otlp_protobuf() -> None:
    """The mapping being right in Python says nothing about it surviving the wire."""
    from opentelemetry.exporter.otlp.proto.common._internal.metrics_encoder import encode_metrics

    encoded = encode_metrics(build_metrics(populated(), START_NS, NOW_NS))

    assert len(encoded.SerializeToString()) > 0


def test_the_snapshot_matches_what_a_scrape_sees() -> None:
    """Two readers over one registry must not disagree about a value neither got wrong."""
    registry = populated()
    exported = {
        metric.name: metric.data.data_points
        for metric in metrics_of(build_metrics(registry, START_NS, NOW_NS))
    }
    scraped = registry.render()

    for point in exported["fasterrag_requests_total"]:
        assert f'status="{point.attributes["status"]}"}} {point.value}' in scraped


class Receiver(BaseHTTPRequestHandler):
    """Accepts one OTLP export, recording the body."""

    bodies: ClassVar[list[bytes]] = []

    def do_POST(self) -> None:
        """Accept the export."""
        body = self.rfile.read(int(self.headers.get("Content-Length", 0)))
        if self.headers.get("Content-Encoding") == "gzip":
            body = gzip.decompress(body)
        Receiver.bodies.append(body)
        self.send_response(200)
        self.send_header("Content-Type", "application/x-protobuf")
        self.end_headers()

    def log_message(self, *args: object) -> None:
        """Stay quiet."""


async def test_a_snapshot_reaches_a_listening_endpoint() -> None:
    """The wire path, not just the mapping: a real POST a collector would accept."""
    from opentelemetry.proto.collector.metrics.v1.metrics_service_pb2 import (
        ExportMetricsServiceRequest,
    )

    Receiver.bodies = []
    server = HTTPServer(("127.0.0.1", 0), Receiver)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        pusher = MetricPusher(
            f"http://127.0.0.1:{server.server_address[1]}/v1/metrics", registry=populated()
        )
        accepted = await pusher.push()
        await pusher.close()
    finally:
        server.shutdown()

    assert accepted is True
    request = ExportMetricsServiceRequest()
    request.ParseFromString(Receiver.bodies[0])
    wire = request.resource_metrics[0].scope_metrics[0].metrics
    assert {metric.name for metric in wire} >= {"fasterrag_requests_total"}


def test_the_pusher_posts_to_the_metrics_path_of_the_configured_endpoint() -> None:
    """``otel_endpoint`` is shared with trace export, so it is a base URL, not a metrics URL.

    Verified against a real collector (TASK-0216): a metrics payload posted to ``/v1/traces``
    is rejected with 400, and a bare endpoint 404s, both as a warning and a lost snapshot.
    """
    assert MetricPusher("http://collector:4318/v1/traces").endpoint == (
        "http://collector:4318/v1/metrics"
    )
    assert MetricPusher("http://collector:4318").endpoint == "http://collector:4318/v1/metrics"


async def test_an_unreachable_collector_costs_the_snapshot_and_nothing_else() -> None:
    """Metrics describe a process that is otherwise working fine."""
    pusher = MetricPusher("http://127.0.0.1:1/v1/metrics", registry=populated(), timeout=1)

    assert await pusher.push() is False

    await pusher.close()


async def test_closing_sends_a_final_snapshot() -> None:
    """Otherwise the last interval's counters go with the process."""
    Receiver.bodies = []
    server = HTTPServer(("127.0.0.1", 0), Receiver)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        pusher = MetricPusher(
            f"http://127.0.0.1:{server.server_address[1]}/v1/metrics",
            registry=populated(),
            interval_seconds=3600,
        )
        pusher.start()
        await pusher.close()
    finally:
        server.shutdown()

    assert len(Receiver.bodies) == 1


async def test_starting_twice_runs_one_loop() -> None:
    pusher = MetricPusher("http://127.0.0.1:1/v1/metrics", interval_seconds=3600, timeout=1)
    pusher.start()
    first = pusher._task
    pusher.start()

    assert pusher._task is first

    await pusher.close()


def test_the_toggle_off_builds_no_pusher() -> None:
    assert create_metric_pusher(Settings.model_validate({})) is None


def test_the_toggle_on_builds_one() -> None:
    settings = Settings.model_validate(
        {"observability": {"otel": True, "otel_endpoint": "http://localhost:4318/v1/metrics"}}
    )

    assert create_metric_pusher(settings) is not None
