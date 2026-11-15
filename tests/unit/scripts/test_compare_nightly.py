import json
from pathlib import Path
from typing import Any

import pytest
import yaml
from compare_nightly import EXIT_CODES, Verdict, compare, compare_files, main

MACHINE: dict[str, Any] = {
    "cpu": "x86_64",
    "cores": 4,
    "ram_gb": 16.0,
    "gpu": "none detected (no nvidia-smi)",
    "storage_gb": 100.0,
    "os": "Linux 6.5.0",
    "python": "3.12.3",
    "fasterrag": "0.1.0",
}


def document(
    *,
    p50: float | None = 80.0,
    p95: float | None = 100.0,
    p99: float | None = 150.0,
    throughput: float | None = 2.0,
    suite: str = "query",
    dataset: str = "demo",
    machine: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a document shaped exactly as `fasterrag benchmark --json` emits one."""
    median = {
        "operations": 20,
        "seconds": 10.0,
        "throughput_per_sec": throughput,
        "p50_ms": p50,
        "p95_ms": p95,
        "p99_ms": p99,
        "max_ms": 200.0,
    }
    return {
        "hardware": machine or MACHINE,
        "suites": [
            {
                "suite": suite,
                "dataset": dataset,
                "repetitions": 3,
                "cold": {"p50_ms": 9000.0},
                "median": {name: value for name, value in median.items() if value is not None},
                "runs": [],
                "notes": "",
            }
        ],
    }


def write(path: Path, payload: Any) -> Path:
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_an_unchanged_night_is_within_tolerance() -> None:
    report = compare(document(), document())

    assert report.verdict is Verdict.WITHIN_TOLERANCE
    assert len(report.comparisons) == 4


def test_a_move_inside_the_tolerance_passes() -> None:
    """A 5% p95 rise is noise on a shared runner, not a signal."""
    report = compare(document(p95=100.0), document(p95=105.0))

    assert report.verdict is Verdict.WITHIN_TOLERANCE


def test_a_p95_rise_beyond_the_tolerance_regresses() -> None:
    report = compare(document(p95=100.0), document(p95=130.0))

    assert report.verdict is Verdict.REGRESSED
    assert any("p95_ms" in entry.reason for entry in report.regressions)
    assert "30.0% worse" in report.summary()


def test_the_tolerance_boundary_is_not_a_regression() -> None:
    """Exactly at the tolerance is allowed; only a move beyond it fails."""
    report = compare(document(p95=100.0), document(p95=110.0), tolerance=0.10)

    assert report.verdict is Verdict.WITHIN_TOLERANCE


def test_an_improvement_never_fails() -> None:
    """The retrieval gate never blocks on a rise, and neither does this one."""
    report = compare(document(p95=100.0, throughput=2.0), document(p95=20.0, throughput=9.0))

    assert report.verdict is Verdict.WITHIN_TOLERANCE


def test_a_throughput_collapse_regresses() -> None:
    """Throughput is higher-is-better, so a fall is the regression, not a rise."""
    report = compare(document(throughput=2.0), document(throughput=1.0))

    assert report.verdict is Verdict.REGRESSED
    assert any("throughput_per_sec" in entry.reason for entry in report.regressions)


def test_a_throughput_rise_is_not_a_regression() -> None:
    report = compare(document(throughput=2.0), document(throughput=4.0))

    assert report.verdict is Verdict.WITHIN_TOLERANCE


def test_no_previous_run_is_not_comparable_rather_than_green(tmp_path: Path) -> None:
    """The defect this script exists to avoid: reporting a pass having compared nothing."""
    current = write(tmp_path / "current.json", document())

    report = compare_files(tmp_path / "absent.json", current)

    assert report.verdict is Verdict.NOT_COMPARABLE
    assert report.comparisons == []
    assert "NOT COMPARABLE" in report.summary()
    assert "within tolerance" not in report.summary()


def test_an_unparseable_previous_run_is_not_comparable(tmp_path: Path) -> None:
    """The benchmark step runs under `|| true`, so an empty file is an ordinary outcome."""
    previous = tmp_path / "previous.json"
    previous.write_text("", encoding="utf-8")
    current = write(tmp_path / "current.json", document())

    report = compare_files(previous, current)

    assert report.verdict is Verdict.NOT_COMPARABLE
    assert any("not a readable benchmark document" in reason for reason in report.obstacles)


def test_a_json_array_is_not_a_benchmark_document(tmp_path: Path) -> None:
    previous = write(tmp_path / "previous.json", [1, 2, 3])
    current = write(tmp_path / "current.json", document())

    report = compare_files(previous, current)

    assert report.verdict is Verdict.NOT_COMPARABLE


def test_a_different_machine_is_not_comparable() -> None:
    """Identical numbers from a different runner measure the runner, not the change."""
    other = {**MACHINE, "cpu": "aarch64", "cores": 8}

    report = compare(document(), document(machine=other))

    assert report.verdict is Verdict.NOT_COMPARABLE
    assert any("different machines" in reason for reason in report.obstacles)


def test_a_version_change_still_compares() -> None:
    """The code changing is the thing being watched for; refusing to compare hides it."""
    newer = {**MACHINE, "fasterrag": "0.2.0", "python": "3.12.7"}

    report = compare(document(p95=100.0), document(p95=130.0, machine=newer))

    assert report.verdict is Verdict.REGRESSED


def test_a_missing_fingerprint_is_not_comparable() -> None:
    """A run that names no machine is refused for that reason, not for a field mismatch."""
    without = document()
    del without["hardware"]

    report = compare(document(), without)

    assert report.verdict is Verdict.NOT_COMPARABLE
    assert any("no hardware fingerprint" in reason for reason in report.obstacles)


def test_two_runs_with_no_fingerprint_at_all_are_not_comparable() -> None:
    """Two documents that each name no machine agree about nothing; that is not evidence."""
    first = document()
    second = document()
    del first["hardware"]
    del second["hardware"]

    report = compare(first, second)

    assert report.verdict is Verdict.NOT_COMPARABLE
    assert any("no hardware fingerprint" in reason for reason in report.obstacles)


def test_a_different_dataset_is_not_comparable() -> None:
    """A different corpus produces different numbers for reasons that are not regressions."""
    report = compare(document(dataset="demo"), document(dataset="other", p95=130.0))

    assert report.verdict is Verdict.NOT_COMPARABLE
    assert report.comparisons == []


def test_a_suite_with_no_counterpart_is_not_comparable() -> None:
    report = compare(document(suite="ingest"), document(suite="query"))

    assert report.verdict is Verdict.NOT_COMPARABLE
    assert any("no counterpart" in reason for reason in report.obstacles)


def test_a_suite_that_vanished_is_reported() -> None:
    previous = document()
    previous["suites"].append(document(suite="ingest")["suites"][0])

    report = compare(previous, document())

    assert report.verdict is Verdict.NOT_COMPARABLE
    assert any("ran previously but not tonight" in reason for reason in report.obstacles)


def test_a_benchmark_that_produced_no_suites_is_not_comparable() -> None:
    report = compare(document(), {"hardware": MACHINE, "suites": []})

    assert report.verdict is Verdict.NOT_COMPARABLE
    assert any("did not complete" in reason for reason in report.obstacles)


def test_a_missing_metric_is_not_comparable_rather_than_skipped() -> None:
    """An absent p95 must not leave a green verdict covering three metrics out of four."""
    report = compare(document(), document(p95=None))

    assert report.verdict is Verdict.NOT_COMPARABLE
    assert any("p95_ms is absent" in reason for reason in report.obstacles)


def test_a_zero_baseline_is_not_comparable() -> None:
    """A percentage against zero is undefined, and a zero p95 means nothing was sampled."""
    report = compare(document(p95=0.0), document(p95=400.0))

    assert report.verdict is Verdict.NOT_COMPARABLE
    assert any("says nothing" in reason for reason in report.obstacles)


def test_a_boolean_is_not_read_as_a_number() -> None:
    """JSON true is an int in Python; treating it as 1 ms would invent a measurement."""
    report = compare(document(), document(p95=True))

    assert report.verdict is Verdict.NOT_COMPARABLE


def test_the_report_serializes_what_it_compared() -> None:
    payload = compare(document(), document(p95=130.0)).as_dict()

    assert payload["verdict"] == "regressed"
    assert payload["compared"] == 4
    assert len(payload["comparisons"]) == 4


def test_the_markdown_summary_leads_with_the_verdict(tmp_path: Path) -> None:
    current = write(tmp_path / "current.json", document())

    body = compare_files(tmp_path / "absent.json", current).markdown()

    assert "NOT COMPARABLE" in body
    assert "never citable" in body


@pytest.mark.parametrize(
    ("previous", "expected"),
    [(document(p95=100.0), 0), (document(p95=1.0), 1), (None, 2)],
)
def test_the_exit_code_distinguishes_all_three_outcomes(
    tmp_path: Path, previous: dict[str, Any] | None, expected: int
) -> None:
    """A shell step can only branch on the third outcome if it has its own code."""
    current = write(tmp_path / "current.json", document(p95=100.0))
    baseline = tmp_path / "previous.json"
    if previous is not None:
        write(baseline, previous)

    code = main(["--previous", str(baseline), "--current", str(current)])

    assert code == expected


def test_the_json_and_summary_outputs_are_written(tmp_path: Path) -> None:
    current = write(tmp_path / "current.json", document())
    previous = write(tmp_path / "previous.json", document())
    report = tmp_path / "nested" / "comparison.json"
    summary = tmp_path / "summary.md"

    code = main(
        [
            "--previous",
            str(previous),
            "--current",
            str(current),
            "--json-out",
            str(report),
            "--summary",
            str(summary),
        ]
    )

    assert code == 0
    assert json.loads(report.read_text(encoding="utf-8"))["verdict"] == "within-tolerance"
    assert "Nightly benchmark comparison" in summary.read_text(encoding="utf-8")


def test_a_tighter_tolerance_catches_a_smaller_move(tmp_path: Path) -> None:
    current = write(tmp_path / "current.json", document(p95=105.0))
    previous = write(tmp_path / "previous.json", document(p95=100.0))

    assert main(["--previous", str(previous), "--current", str(current)]) == 0
    assert (
        main(["--previous", str(previous), "--current", str(current), "--tolerance", "0.01"]) == 1
    )


def test_a_negative_tolerance_is_refused(tmp_path: Path) -> None:
    """It would fail a run that improved, which is the opposite of the gate's rule."""
    current = write(tmp_path / "current.json", document())
    previous = write(tmp_path / "previous.json", document())

    with pytest.raises(SystemExit):
        main(["--previous", str(previous), "--current", str(current), "--tolerance", "-0.5"])


def workflow(name: str) -> dict[str, Any]:
    """Load a workflow from .github/workflows as a document."""
    path = Path(__file__).resolve().parents[3] / ".github" / "workflows" / f"{name}.yml"
    loaded: Any = yaml.safe_load(path.read_text(encoding="utf-8"))
    return loaded if isinstance(loaded, dict) else {}


def step(job: dict[str, Any], name: str) -> dict[str, Any]:
    """Return the named step of a job, or an empty mapping when it is gone."""
    return next((entry for entry in job["steps"] if entry.get("name") == name), {})


def test_the_nightly_fails_on_the_not_comparable_verdict_as_the_eval_gate_does() -> None:
    """One verdict, two callers: the asymmetry between them was the whole of TASK-0314.

    The nightly mapped `not-comparable` to a warning and stayed green while ci.yml's
    eval gate exited 1 on the same word, so a night that measured nothing read as a pass.
    """
    measure = workflow("nightly")["jobs"]["measure"]
    body = step(measure, "Compare against the previous night")["run"]
    branch = body.split("2)", 1)[1].split(";;", 1)[0]

    assert "exit 1" in branch
    assert "::warning::" not in body


def test_every_exit_code_the_script_defines_has_a_branch_in_the_nightly() -> None:
    """A verdict the workflow does not name falls to the catch-all and reads as a crash."""
    measure = workflow("nightly")["jobs"]["measure"]
    body = step(measure, "Compare against the previous night")["run"]

    for code in EXIT_CODES.values():
        assert f"{code})" in body


def test_the_benchmark_step_does_not_suppress_its_own_failure() -> None:
    """`continue-on-error` plus `|| true` hid a benchmark that had not run since August."""
    measure = workflow("nightly")["jobs"]["measure"]
    benchmark = step(measure, "Benchmark suite")

    assert "|| true" not in benchmark["run"]
    assert benchmark["continue-on-error"] is True


def test_the_measure_job_declares_qdrant_only_beside_the_steps_that_reach_it() -> None:
    """A declared dependency nothing uses is how 83 days of measuring nothing hid.

    TASK-0323 removed a Qdrant service no step opened a socket to. TASK-0488 restored it with
    the steps that contact it, so this asserts the contact rather than the declaration.
    """
    measure = workflow("nightly")["jobs"]["measure"]

    assert set(measure["services"]) == {"qdrant"}
    assert "localhost:6333/readyz" in step(measure, "Wait for Qdrant")["run"]
    assert "ingest" in step(measure, "Build the handbook collection")["run"]
    benchmark = step(measure, "Benchmark suite")["run"]
    assert "--suite all" in benchmark
    assert "--collection handbook" in benchmark
    assert (
        "--faithfulness"
        in step(measure, "Generation gate (faithfulness and citations, scripted LLM)")["run"]
    )


def test_the_nightly_comparison_requires_the_query_suite_and_keeps_its_series() -> None:
    """Without the requirement a night that stopped measuring queries lost one table row."""
    compare_step = step(
        workflow("nightly")["jobs"]["measure"], "Compare against the previous night"
    )["run"]

    assert "--require-suite query" in compare_step
    assert "--series-out artifacts/series.jsonl" in compare_step


def two_suites(query_p50: float = 80.0) -> dict[str, Any]:
    """A night that measured both the ingest and the query suite."""
    ingest = document(suite="ingest", p50=2.0)
    query = document(suite="query", p50=query_p50)
    return {"hardware": MACHINE, "suites": [*ingest["suites"], *query["suites"]]}


def test_a_night_without_a_required_suite_is_not_comparable(tmp_path: Path) -> None:
    """A night that silently stopped measuring the query suite only lost a table row."""
    previous = write(tmp_path / "previous.json", two_suites())
    current = write(tmp_path / "current.json", document(suite="ingest", p50=2.0))

    report = compare_files(previous, current, required=("ingest", "query"))

    assert report.verdict is Verdict.NOT_COMPARABLE
    assert any("query suite is required" in note for note in report.notes)


def test_required_suites_that_are_present_are_compared(tmp_path: Path) -> None:
    previous = write(tmp_path / "previous.json", two_suites())
    current = write(tmp_path / "current.json", two_suites(query_p50=81.0))

    report = compare_files(previous, current, required=("ingest", "query"))

    assert report.verdict is Verdict.WITHIN_TOLERANCE
    assert {entry.suite for entry in report.comparisons} == {"ingest", "query"}


def test_the_query_latency_series_is_carried_forward_night_over_night(tmp_path: Path) -> None:
    """Nothing kept a query-latency series: each night saw only the night before."""
    previous = write(tmp_path / "previous.json", two_suites(query_p50=80.0))
    series = tmp_path / "series.jsonl"
    summary = tmp_path / "summary.md"
    nights = [(tmp_path / f"night-{n}.json", 80.0 + n) for n in range(3)]
    for night, (path, p50) in enumerate(nights):
        write(path, two_suites(query_p50=p50))
        arguments = [
            "--previous",
            str(previous),
            "--current",
            str(path),
            "--require-suite",
            "query",
            "--series-out",
            str(series),
            "--run-label",
            f"run-{night}",
            "--summary",
            str(summary),
        ]
        if night:
            arguments += ["--series", str(series)]
        assert main(arguments) == 0

    rows = [json.loads(line) for line in series.read_text(encoding="utf-8").splitlines()]
    assert [row["run"] for row in rows] == ["run-0", "run-1", "run-2"]
    assert [row["suites"]["query"]["p50_ms"] for row in rows] == [80.0, 81.0, 82.0]
    assert "`query.p50_ms` over the last 3 nights" in summary.read_text(encoding="utf-8")


def test_the_series_keeps_only_the_newest_nights() -> None:
    from compare_nightly import extend_series

    rows = [{"run": str(index)} for index in range(5)]

    assert [row["run"] for row in extend_series(rows, {"run": "5"}, limit=3)] == ["3", "4", "5"]


def test_a_missing_or_torn_series_starts_afresh(tmp_path: Path) -> None:
    from compare_nightly import load_series

    torn = tmp_path / "series.jsonl"
    torn.write_text('{"run": "a"}\n{"run": \n', encoding="utf-8")

    assert load_series(tmp_path / "absent.jsonl") == []
    assert load_series(torn) == [{"run": "a"}]
