import pytest

from fasterrag.observability import metrics as catalogue
from fasterrag.observability.metrics import (
    Counter,
    Gauge,
    Histogram,
    Registry,
)

CATALOGUE = {
    "fasterrag_requests_total",
    "fasterrag_errors_total",
    "fasterrag_request_duration_seconds",
    "fasterrag_stage_duration_seconds",
    "fasterrag_ttft_seconds",
    "fasterrag_tokens_total",
    "fasterrag_cost_usd_total",
    "fasterrag_cache_events_total",
    "fasterrag_retrieval_quality",
    "fasterrag_faithfulness",
    "fasterrag_ingest_documents_total",
    "fasterrag_ingest_throughput",
    "fasterrag_queue_depth",
    "fasterrag_dlq_depth",
    "fasterrag_circuit_state",
    "fasterrag_degraded_responses_total",
}


def lines(text: str) -> list[str]:
    return [line for line in text.splitlines() if line and not line.startswith("#")]


def test_every_documented_metric_is_declared() -> None:
    assert set(catalogue.REGISTRY.names) == CATALOGUE


def test_a_counter_accumulates() -> None:
    counter = Counter("test_total", "docs")

    counter.increment()
    counter.increment(4)

    assert counter.value() == pytest.approx(5.0)


def test_counter_series_are_independent() -> None:
    counter = Counter("test_total", "docs", ("code",))

    counter.increment(code="a")
    counter.increment(2, code="b")

    assert counter.value(code="a") == pytest.approx(1.0)
    assert counter.value(code="b") == pytest.approx(2.0)


def test_label_order_never_splits_a_series() -> None:
    counter = Counter("test_total", "docs", ("a", "b"))

    counter.increment(a="1", b="2")
    counter.increment(b="2", a="1")

    assert counter.value(a="1", b="2") == pytest.approx(2.0)


def test_a_counter_refuses_to_decrease() -> None:
    with pytest.raises(ValueError, match="cannot decrease"):
        Counter("test_total", "docs").increment(-1)


def test_an_unobserved_series_reads_as_zero() -> None:
    assert Counter("test_total", "docs", ("code",)).value(code="never") == pytest.approx(0.0)


def test_a_gauge_replaces_rather_than_accumulates() -> None:
    gauge = Gauge("test_depth", "docs")

    gauge.set(5)
    gauge.set(2)

    assert gauge.value() == pytest.approx(2.0)


def test_a_gauge_may_go_down() -> None:
    gauge = Gauge("test_depth", "docs", ("queue",))

    gauge.set(10, queue="ingest")
    gauge.set(0, queue="ingest")

    assert gauge.value(queue="ingest") == pytest.approx(0.0)


def test_a_histogram_counts_and_sums() -> None:
    histogram = Histogram("test_seconds", "docs", buckets=(1.0, 5.0))

    histogram.observe(0.5)
    histogram.observe(2.0)

    assert histogram.count() == 2
    assert histogram.total() == pytest.approx(2.5)


def test_histogram_buckets_are_cumulative() -> None:
    histogram = Histogram("test_seconds", "docs", buckets=(1.0, 5.0))
    for value in (0.5, 2.0, 7.0):
        histogram.observe(value)

    rendered = lines("\n".join(histogram.render()))
    buckets = {
        line.split("{le=")[1].split("}")[0]: line.split()[-1]
        for line in rendered
        if "_bucket" in line
    }

    assert buckets['"1.0"'] == "1"
    assert buckets['"5.0"'] == "2"
    assert buckets['"+Inf"'] == "3"


def test_an_observation_on_the_bound_falls_inside_it() -> None:
    histogram = Histogram("test_seconds", "docs", buckets=(1.0,))
    histogram.observe(1.0)

    rendered = "\n".join(histogram.render())

    assert 'test_seconds_bucket{le="1.0"} 1' in rendered


def test_the_exposition_format_declares_help_and_type() -> None:
    registry = Registry()
    registry.counter("test_total", "how many things happened")

    rendered = registry.render()

    assert "# HELP test_total how many things happened" in rendered
    assert "# TYPE test_total counter" in rendered


def test_the_exposition_format_ends_with_a_newline() -> None:
    registry = Registry()
    registry.counter("test_total", "docs").increment()

    assert registry.render().endswith("\n")


def test_a_label_value_with_a_quote_is_escaped() -> None:
    registry = Registry()
    registry.counter("test_total", "docs", ("path",)).increment(path='a"b')

    assert '\\"' in registry.render()


def test_a_label_value_with_a_newline_is_escaped() -> None:
    registry = Registry()
    registry.counter("test_total", "docs", ("detail",)).increment(detail="a\nb")

    rendered = registry.render()

    assert "\\n" in rendered
    assert len(lines(rendered)) == 1


def test_a_histogram_renders_sum_and_count() -> None:
    registry = Registry()
    registry.histogram("test_seconds", "docs", buckets=(1.0,)).observe(0.5)

    rendered = registry.render()

    assert "test_seconds_sum 0.5" in rendered
    assert "test_seconds_count 1" in rendered


def test_a_declared_but_unobserved_metric_still_announces_itself() -> None:
    registry = Registry()
    registry.counter("test_total", "docs")

    rendered = registry.render()

    assert "# TYPE test_total counter" in rendered
    assert lines(rendered) == []


def test_the_catalogue_renders_without_error() -> None:
    assert catalogue.render().startswith("# HELP fasterrag_requests_total")


def test_faithfulness_uses_score_buckets_not_duration_buckets() -> None:
    assert catalogue.FAITHFULNESS.buckets[-1] == pytest.approx(1.0)
    assert catalogue.REQUEST_DURATION.buckets[-1] > 1.0
