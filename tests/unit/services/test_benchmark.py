import pytest

from fasterrag.services.benchmark import (
    REPETITIONS,
    BenchmarkRun,
    SuiteResult,
    commit_hash,
    fingerprint,
    ledger_entry,
    measure,
    percentile,
)

LEDGER_FIELDS = ("Claim:", "Method:", "Dataset:", "Hardware:", "Date:", "Numbers:", "Commit:")


def run(samples: list[float], seconds: float = 1.0) -> BenchmarkRun:
    return BenchmarkRun(samples=samples, operations=len(samples), seconds=seconds)


def test_an_empty_sample_set_has_no_percentile() -> None:
    assert percentile([], 0.95) == pytest.approx(0.0)


def test_a_percentile_is_a_value_that_was_actually_observed() -> None:
    values = [0.1, 0.2, 0.3, 0.4, 0.5]

    assert percentile(values, 0.95) in values
    assert percentile(values, 0.50) in values


def test_the_median_percentile_sits_in_the_middle() -> None:
    assert percentile([1.0, 2.0, 3.0], 0.50) == pytest.approx(2.0)


def test_the_top_percentile_is_the_largest_sample() -> None:
    assert percentile([1.0, 2.0, 3.0, 100.0], 0.99) == pytest.approx(100.0)


def test_a_single_sample_is_every_percentile() -> None:
    assert percentile([7.0], 0.50) == pytest.approx(7.0)
    assert percentile([7.0], 0.99) == pytest.approx(7.0)


def test_throughput_is_operations_over_elapsed() -> None:
    assert run([0.1] * 10, seconds=2.0).throughput == pytest.approx(5.0)


def test_a_zero_duration_run_reports_no_throughput() -> None:
    assert BenchmarkRun(samples=[], operations=0, seconds=0.0).throughput == pytest.approx(0.0)


def test_a_run_reports_every_documented_percentile() -> None:
    payload = run([0.01, 0.02, 0.03]).as_dict()

    assert {"p50_ms", "p95_ms", "p99_ms", "max_ms", "throughput_per_sec"} <= set(payload)


def test_the_median_is_taken_across_repetitions() -> None:
    result = SuiteResult(
        suite="query",
        dataset="fixture",
        runs=[run([0.01]), run([0.05]), run([0.03])],
    )

    assert result.median["p50_ms"] == pytest.approx(30.0)


def test_every_run_is_attached_not_just_the_median() -> None:
    result = SuiteResult(suite="query", dataset="fixture", runs=[run([0.01]), run([0.02])])

    assert len(result.as_dict()["runs"]) == 2


def test_a_suite_with_no_runs_has_no_median() -> None:
    assert SuiteResult(suite="query", dataset="fixture").median == {}


async def test_measuring_records_a_cold_run_and_the_repetitions() -> None:
    calls = {"count": 0}

    async def operation() -> None:
        calls["count"] += 1

    cold, runs = await measure(operation, iterations=4)

    assert cold is not None
    assert len(cold.samples) == 1
    assert len(runs) == REPETITIONS
    assert all(len(run.samples) == 4 for run in runs)
    assert calls["count"] == 1 + REPETITIONS * 4


async def test_the_cold_run_is_separate_from_the_warmed_ones() -> None:
    async def operation() -> None:
        return None

    cold, runs = await measure(operation, iterations=2, repetitions=1)

    assert cold is not None
    assert cold.operations == 1
    assert runs[0].operations == 2


def test_the_fingerprint_records_the_machine() -> None:
    machine = fingerprint()

    assert machine.cores >= 1
    assert machine.ram_gb > 0
    assert machine.os
    assert machine.python
    assert machine.fasterrag


def test_the_fingerprint_never_leaves_the_gpu_ambiguous() -> None:
    assert fingerprint().gpu.strip()


def test_the_fingerprint_describes_itself_in_one_line() -> None:
    described = fingerprint().describe()

    assert "\n" not in described
    assert "cores" in described


def test_the_commit_is_never_silently_blank() -> None:
    assert commit_hash().strip()


def test_a_ledger_entry_carries_every_mandatory_field() -> None:
    result = SuiteResult(suite="query", dataset="fixture@1", runs=[run([0.01])])

    entry = ledger_entry("BENCH-0001", "the query suite is this fast", result)

    for label in LEDGER_FIELDS:
        assert label in entry


def test_a_ledger_entry_names_its_identifier_and_claim() -> None:
    result = SuiteResult(suite="query", dataset="fixture@1", runs=[run([0.01])])

    entry = ledger_entry("BENCH-0007", "a specific claim", result)

    assert entry.startswith("### BENCH-0007 — a specific claim")


def test_a_ledger_entry_records_what_supersedes_it() -> None:
    result = SuiteResult(suite="query", dataset="fixture", runs=[run([0.01])])

    assert "Supersedes: none" in ledger_entry("BENCH-0001", "claim", result)
    assert "Supersedes: BENCH-0001" in ledger_entry(
        "BENCH-0002", "claim", result, supersedes="BENCH-0001"
    )


def test_a_ledger_entry_reports_the_cold_start_separately() -> None:
    result = SuiteResult(suite="query", dataset="fixture", cold=run([0.5]), runs=[run([0.01])])

    assert "cold start" in ledger_entry("BENCH-0001", "claim", result)


def test_a_ledger_entry_states_the_methodology_it_followed() -> None:
    result = SuiteResult(suite="query", dataset="fixture", runs=[run([0.01, 0.02])])

    entry = ledger_entry("BENCH-0001", "claim", result)

    assert "warmed repetitions" in entry
    assert "nearest-rank" in entry
    assert "median reported" in entry


def test_notes_reach_the_entry_when_present() -> None:
    result = SuiteResult(
        suite="ingest", dataset="fixture", runs=[run([0.01])], notes="parse and chunk only"
    )

    assert "parse and chunk only" in ledger_entry("BENCH-0001", "claim", result)
