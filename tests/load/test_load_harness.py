"""The load harness's own tests: everything here is about refusing, not about speed.

These run in the ordinary unit lane and carry no ``load`` marker, deliberately. The ``load``
marker means "needs reference hardware"; these need none, because what they assert is that
the harness refuses to report a number it did not measure. That property has to be checked
on every push, on a laptop, with no load generator installed — it is the property that makes
the eventual measurement trustworthy, and a test for it that only runs on the reference rig
would be checked roughly never.

Nothing here imports Locust. Every case drives the dependency-free core directly, and the
preflight cases use a throwaway ``http.server`` thread or a deliberately closed port.
"""

from __future__ import annotations

import json
import socket
import threading
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

from tests.load.harness import (
    QUERY_ENDPOINT,
    RETRIEVAL_SLI_ENTRY,
    RETRIEVE_ENDPOINT,
    STATUS_2XX_ENTRY,
    STATUS_5XX_ENTRY,
    STREAM_ENDPOINT,
    TTFT_SLI_ENTRY,
    EndpointStats,
    ExitCode,
    LoadConfig,
    LoadHarnessError,
    LoadReport,
    Problem,
    RunStats,
    audit,
    derive_slis,
    exit_code_for,
    load_queries,
    parse_exposition,
    parse_locust_stats,
)
from tests.load.preflight import preflight

EXPOSITION = """
# HELP fasterrag_requests_total Requests.
# TYPE fasterrag_requests_total counter
fasterrag_requests_total{endpoint="/v1/query",status="200"} 12
fasterrag_requests_total{endpoint="/v1/query",status="500"} 2
fasterrag_cache_events_total{cache="semantic",result="hit"} 3
fasterrag_cache_events_total{cache="semantic",result="miss"} 7
fasterrag_dlq_depth{collection="default"} 4
fasterrag_ingest_documents_total{status="succeeded"} 10
"""

EXPOSITION_AFTER = """
fasterrag_requests_total{endpoint="/v1/query",status="200"} 112
fasterrag_cache_events_total{cache="semantic",result="hit"} 8
fasterrag_cache_events_total{cache="semantic",result="miss"} 22
fasterrag_dlq_depth{collection="default"} 4
fasterrag_ingest_documents_total{status="succeeded"} 1210
"""


def make_config(dataset: Path, **overrides: object) -> LoadConfig:
    defaults: dict[str, object] = {
        "target": "http://127.0.0.1:1",
        "scenario": "query",
        "users": 4,
        "spawn_rate": 4.0,
        "duration_s": 10,
        "dataset": dataset,
        "sut_hardware": "a machine",
        "min_samples": 10,
    }
    defaults.update(overrides)
    return LoadConfig(**defaults)  # type: ignore[arg-type]


def write_dataset(root: Path, queries: list[str]) -> Path:
    dataset = root / "set"
    (dataset / "corpus").mkdir(parents=True)
    (dataset / "corpus" / "a.md").write_text("a document", encoding="utf-8")
    lines = [json.dumps({"id": f"q{index}", "query": text}) for index, text in enumerate(queries)]
    (dataset / "golden.jsonl").write_text("\n".join(lines), encoding="utf-8")
    return dataset


def stats(name: str, requests: int, failures: int = 0, bucket: float = 20.0) -> EndpointStats:
    return EndpointStats(
        name=name,
        method="POST",
        num_requests=requests,
        num_failures=failures,
        response_times={bucket: requests} if requests else {},
    )


def make_run(entries: list[EndpointStats], duration_s: float = 10.0) -> RunStats:
    return RunStats(
        label="warmed-1",
        duration_s=duration_s,
        endpoints=tuple(entries),
        metrics_before=parse_exposition(EXPOSITION),
        metrics_after=parse_exposition(EXPOSITION_AFTER),
    )


def healthy_entries(samples: int = 50) -> list[EndpointStats]:
    return [
        stats(RETRIEVE_ENDPOINT, samples),
        stats(QUERY_ENDPOINT, samples),
        stats(STREAM_ENDPOINT, samples),
        stats(RETRIEVAL_SLI_ENTRY, samples),
        stats(TTFT_SLI_ENTRY, samples),
        stats(STATUS_2XX_ENTRY, samples * 3),
    ]


@pytest.fixture
def dataset(tmp_path: Path) -> Path:
    return write_dataset(tmp_path, ["what is the leave policy?", "how do I claim expenses?"])


def test_an_unreachable_target_is_refused_before_any_traffic_is_generated(dataset: Path) -> None:
    """A dead target must raise, not produce a run of zero requests."""
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]

    config = make_config(dataset, target=f"http://127.0.0.1:{port}")

    with pytest.raises(LoadHarnessError) as caught:
        preflight(config)

    assert caught.value.exit_code is ExitCode.PREFLIGHT_FAILED
    assert "could not be reached" in caught.value.detail


class _Handler(BaseHTTPRequestHandler):
    metrics_body = b"some_other_service_requests_total 1\n"

    def log_message(self, fmt: str, *args: object) -> None:
        return

    def do_GET(self) -> None:
        body = self.metrics_body if self.path == "/metrics" else b'{"status":"ok"}'
        self.send_response(200)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


@pytest.fixture
def impostor() -> Iterator[str]:
    server = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_address[1]}"
    finally:
        server.shutdown()
        server.server_close()


def test_a_live_server_that_is_not_fasterrag_is_refused(dataset: Path, impostor: str) -> None:
    """Answering /healthz with a 200 is not evidence that the target is fasterRag."""
    with pytest.raises(LoadHarnessError) as caught:
        preflight(make_config(dataset, target=impostor))

    assert caught.value.exit_code is ExitCode.PREFLIGHT_FAILED
    assert "fasterrag_requests_total" in caught.value.detail


def test_a_dataset_with_no_queries_is_refused(tmp_path: Path) -> None:
    """An empty query set would produce a run that sent nothing and reported zeros."""
    dataset = write_dataset(tmp_path, [])

    with pytest.raises(LoadHarnessError) as caught:
        load_queries(dataset)

    assert caught.value.exit_code is ExitCode.USAGE


def test_a_missing_golden_set_names_the_path_it_looked_for(tmp_path: Path) -> None:
    """The refusal has to be actionable, so it names the file it expected."""
    with pytest.raises(LoadHarnessError, match=r"golden\.jsonl"):
        load_queries(tmp_path / "absent")


def test_an_empty_metrics_body_is_refused_rather_than_parsed_as_no_metrics() -> None:
    """Zero series means the scrape is broken, not that every counter is zero."""
    with pytest.raises(LoadHarnessError) as caught:
        parse_exposition("# HELP only comments\n\n")

    assert caught.value.exit_code is ExitCode.PREFLIGHT_FAILED


def test_exposition_parsing_keeps_labels_so_counters_can_be_filtered() -> None:
    """Cache hit rate and DLQ depth are label-selected, so labels must survive parsing."""
    series = parse_exposition(EXPOSITION)

    hits = [
        item for item in series if item.matches("fasterrag_cache_events_total", {"result": "hit"})
    ]
    assert [item.value for item in hits] == [3.0]


def test_locust_writing_no_entries_is_vacuous_not_empty() -> None:
    """A run that recorded nothing must not parse into a clean, empty report."""
    with pytest.raises(LoadHarnessError) as caught:
        parse_locust_stats([])

    assert caught.value.exit_code is ExitCode.VACUOUS


def test_a_percentile_over_an_empty_sample_is_none_and_never_zero() -> None:
    """Locust's own helper returns 0 here; 0 ms reads as infinitely fast."""
    assert stats(QUERY_ENDPOINT, 0).percentile(0.95) is None


def test_a_run_with_no_requests_is_audited_as_vacuous(dataset: Path) -> None:
    """The headline failure: traffic that never happened."""
    problems = audit(make_run([stats(QUERY_ENDPOINT, 0)]), make_config(dataset))

    assert exit_code_for(problems) is ExitCode.VACUOUS
    assert any("zero requests" in problem.detail for problem in problems)


def test_a_sample_below_the_methodology_minimum_is_refused(dataset: Path) -> None:
    """docs/performance.md §3 rule 4 fixes 1,000 samples per latency figure."""
    problems = audit(make_run(healthy_entries(50)), make_config(dataset, min_samples=1000))

    assert exit_code_for(problems) is ExitCode.VACUOUS
    assert any("below the 1000" in problem.detail for problem in problems)


def test_a_zero_p95_over_real_requests_is_refused(dataset: Path) -> None:
    """No HTTP round trip takes 0 ms, so the entry is recording something else."""
    entries = healthy_entries()
    entries[3] = stats(RETRIEVAL_SLI_ENTRY, 50, bucket=0.0)

    problems = audit(make_run(entries), make_config(dataset))

    assert any("p95 is 0" in problem.detail for problem in problems)


def test_a_scenario_that_never_hit_its_endpoints_is_refused(dataset: Path) -> None:
    """A mixed run that ingested nothing is a query run wearing the wrong label."""
    problems = audit(make_run(healthy_entries()), make_config(dataset, scenario="mixed"))

    assert any("/v1/ingest" in problem.detail for problem in problems)


def test_failures_above_the_threshold_earn_their_own_exit_code(dataset: Path) -> None:
    """A run of errors is a different finding from a run of nothing."""
    entries = healthy_entries()
    entries[0] = stats(RETRIEVE_ENDPOINT, 50, failures=25)

    problems = audit(make_run(entries), make_config(dataset))

    assert exit_code_for(problems) is ExitCode.TOO_MANY_FAILURES


def test_turning_streaming_off_does_not_fail_the_run_for_a_missing_stream(dataset: Path) -> None:
    """A guard that fires on a shape the operator chose would teach people to ignore it."""
    entries = [
        item for item in healthy_entries() if item.name not in {STREAM_ENDPOINT, TTFT_SLI_ENTRY}
    ]

    assert audit(make_run(entries), make_config(dataset, stream_weight=0)) == ()


def test_a_healthy_run_produces_no_problems_and_exits_zero(dataset: Path) -> None:
    """The guards must stay quiet on a run that did happen, or they get disabled."""
    problems = audit(make_run(healthy_entries()), make_config(dataset))

    assert problems == ()
    assert exit_code_for(problems) is ExitCode.OK


def test_an_unmeasured_sli_renders_as_not_measured_rather_than_zero() -> None:
    """The distinction the whole harness exists to protect."""
    run = make_run([stats(QUERY_ENDPOINT, 5)])

    rendered = {sli.key: sli.render() for sli in derive_slis(run)}

    assert "not measured" in rendered["ttft_p95"]
    assert "not measured" in rendered["retrieval_p95"]


def test_a_short_window_does_not_report_sustained_ingest_throughput() -> None:
    """slo.md defines the SLI over ten minutes; a ten-second burst is a different number."""
    throughput = next(
        sli for sli in derive_slis(make_run([], 10.0)) if sli.key == "ingest_throughput"
    )

    assert throughput.value is None
    assert "600s sustained" in throughput.note


def test_counter_derived_slis_read_the_labels_the_code_actually_emits() -> None:
    """Cache hit rate comes from result="hit"/"miss"; a label rename must show up here."""
    derived = {sli.key: sli for sli in derive_slis(make_run(healthy_entries()))}

    assert derived["cache_hit_rate"].value == pytest.approx(5 / 20)
    assert derived["dlq_depth"].value == 4.0


def test_a_counter_that_did_not_move_is_not_measured_rather_than_zero() -> None:
    """A cache that was never consulted is not a cache with a 0% hit rate."""
    run = RunStats(
        label="warmed-1",
        duration_s=10.0,
        endpoints=tuple(healthy_entries()),
        metrics_before=parse_exposition(EXPOSITION),
        metrics_after=parse_exposition(EXPOSITION),
    )

    hit_rate = next(sli for sli in derive_slis(run) if sli.key == "cache_hit_rate")

    assert hit_rate.value is None
    assert "did not move" in hit_rate.note


def test_a_long_enough_window_does_report_ingest_throughput() -> None:
    """The window guard must not be a blanket refusal, or the SLI is never measurable."""
    run = RunStats(
        label="warmed-1",
        duration_s=1200.0,
        endpoints=tuple(healthy_entries()),
        metrics_before=parse_exposition(EXPOSITION),
        metrics_after=parse_exposition(EXPOSITION_AFTER),
    )

    throughput = next(sli for sli in derive_slis(run) if sli.key == "ingest_throughput")

    assert throughput.value == pytest.approx(1200 / 1200 * 60)
    assert "tokens/sec cannot be read" in throughput.note


def test_availability_counts_only_the_outcomes_the_sli_definition_names() -> None:
    """slo.md excludes client 4xx from the availability ratio."""
    entries = [*healthy_entries(), stats(STATUS_5XX_ENTRY, 30)]

    availability = next(
        sli for sli in derive_slis(make_run(entries)) if sli.key == "query_availability"
    )

    assert availability.value == pytest.approx(150 / 180)


def report_for(dataset: Path, **overrides: object) -> LoadReport:
    config = make_config(dataset, **overrides)
    run = make_run(healthy_entries())
    return LoadReport(
        config=config,
        generator="a laptop",
        started_at="2026-11-07T00:00:00+00:00",
        commit="abc1234",
        cold=None,
        runs=(run,),
        slis=derive_slis(run),
        problems=audit(run, config),
    )


def test_a_run_without_an_isolation_assertion_cannot_become_a_ledger_entry(dataset: Path) -> None:
    """Co-tenant load is undetectable from here, so the operator must assert its absence."""
    report = report_for(dataset)

    assert not report.citable
    assert "--isolated" in report.ledger_draft()
    assert report.ledger_draft().startswith("NOT CITABLE")


def test_too_few_repetitions_also_blocks_the_ledger_draft(dataset: Path) -> None:
    """Three warmed repetitions, median reported, is the committed methodology."""
    draft = report_for(dataset, isolated=True).ledger_draft()

    assert "warmed repetition(s) were run" in draft


def test_the_refusal_names_every_blocker_rather_than_the_first(dataset: Path) -> None:
    """An operator who fixes one reason should not have to discover the next by rerunning."""
    draft = report_for(dataset).ledger_draft()

    assert "--isolated" in draft
    assert "repetition(s) were run" in draft


def test_the_verdict_vocabulary_never_says_pass(dataset: Path) -> None:
    """There is nothing to pass against: every slo.md target is TBD-until-measured."""
    assert report_for(dataset).verdict == "valid-sample"
    assert exit_code_for((Problem(ExitCode.VACUOUS, "x"),)) is ExitCode.VACUOUS
